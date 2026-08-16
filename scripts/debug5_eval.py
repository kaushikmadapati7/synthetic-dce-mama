#!/usr/bin/env python
"""Fast debug-5 evaluator: the three loss-trainable challenge metrics.

The Grand Challenge *debug* stage scores 5 MAMA-MIA DUKE cases
(DUKE_001/002/005/009/010), so any run can be benchmarked locally against the
leaderboard on identical cases. This computes only the metrics a differentiable
loss can directly target -- MSE, LPIPS, ssim_tumor -- with no nnU-Net/radiomics,
so it returns in seconds and is the fast feedback loop for tuning the loss
recipe. Run the official evaluate.py on a finalist for the full 8 metrics
(FRD / AUROC x2 / DICE / HD95 are downstream and not computed here).

  MSE        : exact -- unweighted global L2, identical to the challenge metric.
  LPIPS      : AlexNet, challenge preprocessing (clip +/-5 sigma -> [-1,1] -> RGB,
               normalize=False). Matches the scored metric -> comparable to the
               leader column below.
  ssim_tumor : skimage SSIM (data_range=10) over the tumor bounding box. PROXY --
               good for ranking runs against each other; VALIDATE its scale once
               against an official metrics.json (see --validate note) before
               trusting the absolute leader comparison.

Usage (from the PROJECT ROOT, after exporting predictions to runs/<name>/predictions/):
    python mama_synth/scripts/debug5_eval.py unet_681835 unet_681837 ...

Path overrides via env: MAMA_GT_DIR / MAMA_PRE_DIR / MAMA_MASK_DIR / MAMA_RUNS_DIR
"""
import os
import sys

import numpy as np
import SimpleITK as sitk

DEBUG = ["DUKE_001", "DUKE_002", "DUKE_005", "DUKE_009", "DUKE_010"]

# Current first-place on the GC debug stage (per-case), for reference.
LEADER = {
    "mse":        {"DUKE_001": 0.0065, "DUKE_002": 0.584, "DUKE_005": 0.015, "DUKE_009": 0.386, "DUKE_010": 1.333},
    "lpips":      {"DUKE_001": 0.059,  "DUKE_002": 0.127, "DUKE_005": 0.048, "DUKE_009": 0.090, "DUKE_010": 0.148},
    "ssim_tumor": {"DUKE_001": 0.457,  "DUKE_002": 0.894, "DUKE_005": 0.896, "DUKE_009": 0.877, "DUKE_010": 0.849},
}
LOWER_BETTER = {"mse": True, "lpips": True, "ssim_tumor": False}

_DATA = "/path/to/mama_mia/preprocessed"
GT_DIR   = os.environ.get("MAMA_GT_DIR",   f"{_DATA}/mha/ground_truth")
PRE_DIR  = os.environ.get("MAMA_PRE_DIR",  f"{_DATA}/mha/input")
MASK_DIR = os.environ.get("MAMA_MASK_DIR", f"{_DATA}/mha/mask")
RUNS_DIR = os.environ.get("MAMA_RUNS_DIR", "runs")

DATA_RANGE = 10.0
CLIP_SIGMA = DATA_RANGE / 2.0   # +/-5 sigma; matches the challenge LPIPS preprocessing


def load(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype("float64")


def metric_mse(pred, gt):
    return float(np.mean((pred - gt) ** 2))


_LPIPS = None
def metric_lpips(pred, gt):
    global _LPIPS
    import torch
    if _LPIPS is None:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
        _LPIPS = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).eval()

    def prep(a):
        t = torch.from_numpy(a).float().clamp(-CLIP_SIGMA, CLIP_SIGMA) / CLIP_SIGMA  # -> [-1, 1]
        return t[None, None].repeat(1, 3, 1, 1)

    with torch.no_grad():
        return float(_LPIPS(prep(pred), prep(gt)))


def metric_ssim_tumor(pred, gt, mask):
    """skimage SSIM (data_range=10) over the tumor bounding box; None if no tumor."""
    from skimage.metrics import structural_similarity as ssim
    m = mask > 0.5
    if m.sum() == 0:
        return None
    ys, xs = np.where(m.reshape(mask.shape[-2], mask.shape[-1]) if mask.ndim > 2 else m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = np.asarray(gt)[..., y0:y1, x0:x1]
    p = np.asarray(pred)[..., y0:y1, x0:x1]
    win = min(7, g.shape[-2], g.shape[-1])
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return None
    return float(ssim(g, p, data_range=DATA_RANGE, win_size=win))


def evaluate_run(name):
    pred_dir = f"{RUNS_DIR}/{name}/predictions"
    ours = {m: {} for m in LOWER_BETTER}
    ident = {m: {} for m in LOWER_BETTER}
    for c in DEBUG:
        gt, pre, pred = load(f"{GT_DIR}/{c}.mha"), load(f"{PRE_DIR}/{c}.mha"), load(f"{pred_dir}/{c}.mha")
        mask = load(f"{MASK_DIR}/{c}.mha")
        ours["mse"][c],   ident["mse"][c]   = metric_mse(pred, gt),   metric_mse(pre, gt)
        ours["lpips"][c], ident["lpips"][c] = metric_lpips(pred, gt), metric_lpips(pre, gt)
        st_o, st_i = metric_ssim_tumor(pred, gt, mask), metric_ssim_tumor(pre, gt, mask)
        if st_o is not None:
            ours["ssim_tumor"][c] = st_o
        if st_i is not None:
            ident["ssim_tumor"][c] = st_i
    return ours, ident


def _mean(d):
    return sum(d.values()) / len(d) if d else float("nan")


def main():
    runs = sys.argv[1:]
    if not runs:
        print("usage: debug5_eval.py <run_name> [<run_name> ...]")
        sys.exit(1)
    for name in runs:
        ours, ident = evaluate_run(name)
        print(f"\n### {name}")
        for metric in ("mse", "lpips", "ssim_tumor"):
            direction = "lower better" if LOWER_BETTER[metric] else "HIGHER better"
            print(f"  [{metric}]  ({direction})")
            print(f"    {'case':10}{'ours':>9}{'identity':>10}{'leader':>9}")
            for c in DEBUG:
                def fmt(v, w):
                    return f"{v:{w}.3f}" if v is not None else f"{'--':>{w}}"
                print(f"    {c:10}{fmt(ours[metric].get(c),9)}{fmt(ident[metric].get(c),10)}{fmt(LEADER[metric].get(c),9)}")
            print(f"    {'MEAN':10}{_mean(ours[metric]):9.3f}{_mean(ident[metric]):10.3f}{_mean(LEADER[metric]):9.3f}")


if __name__ == "__main__":
    main()
