"""Evaluation: run a trained model's generator on the test set, report metrics,
and save qualitative samples (.mha matching the submission format + a PNG montage)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .metrics import eval_metrics, aggregate, compute_fid
from .data import save_mha

log = logging.getLogger("mama_synth")


@torch.no_grad()
def evaluate(args, gen, test_loader, device):
    """gen: callable(cond) -> predicted post-contrast slice."""
    if test_loader is None:
        log.warning("no test set available; skipping evaluation")
        return {}
    per_batch, first = [], None
    all_preds, all_targets = [], []
    compute_fid_flag = getattr(args, "compute_fid", True)
    data_range = getattr(args, "data_range", 10.0)  # match challenge eval (±5σ)

    # optional: dump every test prediction as .mha at native size for the challenge evaluate.py
    export_dir = None
    if getattr(args, "export_preds", False):
        export_dir = Path(args.output_dir) / "predictions"
        export_dir.mkdir(parents=True, exist_ok=True)
        from .data import load_mha
        from .data.preprocessing import center_crop_pad_2d
        gt_root = Path(args.data_root) / "mha" / "ground_truth"
        in_root = Path(args.data_root) / "mha" / "input"   # pre, for periphery reconstruction

    for batch in test_loader:
        cond = batch["cond"].to(device); target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        pred = gen(cond, mask)   # gens accept (cond, mask=None); only the teacher uses mask
        if pred.shape != target.shape:  # guard against any model/target size mismatch
            pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=False)
        if "norm_std" in batch:  # invert per-image normalization -> global (challenge) space
            m = batch["norm_mean"].to(device).view(-1, 1, 1, 1)
            s = batch["norm_std"].to(device).view(-1, 1, 1, 1)
            pred = pred * s + m
            target = target * s + m
        if export_dir is not None:
            for i in range(pred.size(0)):
                cid = batch["id"][i].replace("/", "_")
                p_np = pred[i, 0].cpu().numpy()
                gt_path = gt_root / f"{cid}.mha"
                if gt_path.exists():  # invert the dataset's center-crop/pad back to native size
                    native_shape = load_mha(gt_path).shape
                    in_path = in_root / f"{cid}.mha"   # fill un-predicted periphery from the pre
                    fill = load_mha(in_path).astype(p_np.dtype) if in_path.exists() else None
                    p_np = center_crop_pad_2d(p_np, native_shape, fill=fill)
                save_mha(p_np, export_dir / f"{cid}.mha")
        per_batch.append(eval_metrics(pred, target, mask, data_range=data_range))
        if compute_fid_flag:
            for i in range(pred.size(0)):
                all_preds.append(pred[i].cpu())
                all_targets.append(target[i].cpu())
        if first is None:
            first = (cond.cpu(), target.cpu(), pred.cpu(), batch["id"][0])
    metrics = aggregate(per_batch)
    if compute_fid_flag and all_preds:
        fid = compute_fid(all_preds, all_targets, device,
                          batch_size=getattr(args, "fid_batch_size", 32))
        if fid is not None:
            metrics["fid"] = fid
    log.info(f"TEST metrics: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")
    if first is not None:
        save_samples(Path(args.output_dir) / "samples", *first)
    return metrics


def save_samples(out_dir: Path, cond, target, pred, case_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    cid = case_id.replace("/", "_")
    try:
        for name, vol in [("target", target[0, 0]), ("pred", pred[0, 0])]:
            save_mha(vol.numpy(), out_dir / f"{cid}_{name}.mha")
    except Exception as e:  # noqa
        log.warning(f".mha save skipped: {e}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = [("pre-contrast", cond[0, 0]), ("post target", target[0, 0]),
                  ("post pred", pred[0, 0])]
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img.numpy(), cmap="gray"); ax.set_title(title); ax.axis("off")
        fig.suptitle(case_id); fig.tight_layout()
        fig.savefig(out_dir / "montage.png", dpi=120); plt.close(fig)
    except Exception as e:  # noqa
        log.warning(f"montage save skipped: {e}")
