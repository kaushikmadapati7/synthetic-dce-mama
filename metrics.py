"""Image-space evaluation metrics for generated 2D post-contrast slices.

These are reported (not back-propagated) to gauge how close a generated slice is to
ground truth. SSIM reuses the 2D implementation from the loss module.

This is a local sanity-check suite (SSIM/PSNR/MAE + optional FID); the full challenge
metric set (MSE, LPIPS, ROI-SSIM, FRD, AUROC, DICE, HD95) is computed by the official
mama-research/mama-synth evaluate.py against our exported predictions.

FID (Fréchet Inception Distance) compares the distribution of predicted vs. reference
slices using Inception-v3 features (via torch_fidelity). Lower is better.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .loss.loss import ssim2d

log = logging.getLogger("mama_synth")


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 4.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(data_range ** 2 / (mse + 1e-12))


@torch.no_grad()
def eval_metrics(pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor | None = None, data_range: float = 4.0) -> dict:
    """Per-batch metrics. If a mask is given, MAE is also computed inside the ROI."""
    out = {
        "ssim": float(ssim2d(pred, target, data_range=data_range)),
        "psnr": float(psnr(pred, target, data_range)),
        "mae": float(F.l1_loss(pred, target)),
        "mse": float(F.mse_loss(pred, target)),
    }
    if mask is not None and mask.sum() > 0:
        m = mask > 0.5
        out["mae_roi"] = float((pred[m] - target[m]).abs().mean())
        out["psnr_roi"] = float(psnr(pred[m], target[m], data_range))
        out["ssim_roi"] = float(ssim2d(pred, target, data_range=data_range,
                                       return_map=True)[m].mean())
    return out


def aggregate(metric_dicts: list[dict]) -> dict:
    """Mean over a list of per-batch metric dicts."""
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {k: float(sum(d[k] for d in metric_dicts if k in d) /
                     max(1, sum(k in d for d in metric_dicts))) for k in keys}


# ---------------------------------------------------------------------------
# FID (distribution metric over 2D slices)
# ---------------------------------------------------------------------------
def _to_uint8_rgb(slice_2d: torch.Tensor) -> torch.Tensor:
    """(H, W) -> (3, H, W) uint8 RGB for Inception.

    Data is z-scored (unbounded), so rescale per-slice by its own min/max into [0, 255]
    rather than assuming a [-1, 1] range.
    """
    s = slice_2d.float()
    lo, hi = s.min(), s.max()
    x = ((s - lo) / (hi - lo + 1e-8) * 255.0).round().to(torch.uint8)
    return x.unsqueeze(0).expand(3, -1, -1)


def slices_to_tensors(slices: list[torch.Tensor], min_size: int = 64) -> list[torch.Tensor]:
    """Each input is a (1, H, W) or (H, W) 2D slice -> (3, H, W) uint8 RGB."""
    out = []
    for sl in slices:
        s = sl[0] if sl.dim() == 3 else sl  # (H, W)
        if min(s.shape) < min_size:
            s = F.interpolate(s[None, None], size=(min_size, min_size),
                              mode="bilinear", align_corners=False)[0, 0]
        out.append(_to_uint8_rgb(s.cpu()))
    return out


class _SliceDataset(Dataset):
    """torch_fidelity-compatible dataset of RGB uint8 slices."""

    def __init__(self, slices: list[torch.Tensor]):
        self.slices = slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        return self.slices[i]


def compute_fid(preds: list[torch.Tensor], targets: list[torch.Tensor],
                device: torch.device | str = "cpu", batch_size: int = 32) -> float | None:
    """FID between predicted and reference slice distributions.

    Requires at least a few slices in each set (torch_fidelity needs enough
    samples for a stable covariance estimate).
    """
    if not preds or not targets:
        return None
    pred_slices = slices_to_tensors(preds)
    tgt_slices = slices_to_tensors(targets)
    if len(pred_slices) < 2 or len(tgt_slices) < 2:
        log.warning(f"FID skipped: need >=2 slices per set (got {len(pred_slices)}/{len(tgt_slices)})")
        return None
    try:
        from torch_fidelity import calculate_metrics
        from torch_fidelity.metric_fid import KEY_METRIC_FID
    except ImportError:
        log.warning("torch_fidelity not installed; skipping FID")
        return None

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    try:
        result = calculate_metrics(
            input1=_SliceDataset(pred_slices),
            input2=_SliceDataset(tgt_slices),
            cuda=use_cuda,
            batch_size=min(batch_size, len(pred_slices), len(tgt_slices)),
            fid=True,
            isc=False, kid=False, prc=False, ppl=False,
            verbose=False,
        )
        return float(result[KEY_METRIC_FID])
    except Exception as e:
        log.warning(f"FID computation failed: {e}")
        return None
