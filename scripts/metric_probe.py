#!/usr/bin/env python
"""E7 metric-behavior probe (paper Fig 4): what does each of the 8 metrics reward?

We do NOT train anything. We take the ground-truth post-contrast slices and apply a
battery of CONTROLLED corruptions (blur / sharpen / noise / under- & over-enhancement /
ROI-only blur). Each corrupted set is written as a normal predictions/ dir + a config.json,
so the EXACT same scoring path as a real run applies:

  1. python -m mama_synth.scripts.metric_probe --gt-dir .../mha/ground_truth \
         --pre-dir .../mha/input --mask-dir .../mha/mask --out-dir probe
  2. for each probe/<name>: run the official evaluate.py with
         MAMA_PREDICTIONS_DIR=probe/<name>/predictions  MAMA_OUTPUT_DIR=probe/<name>/eval
     (GT/MASK/PRECONTRAST dirs as usual) -> probe/<name>/eval/metrics.json
  3. python -m mama_synth.scripts.aggregate_metrics --runs-dir probe --out paper/probe.csv
  4. python paper/figs/make_figures.py  (builds Fig 4 from paper/probe.csv)

The corruption ladder is monotone within each family, so each metric's response curve
(e.g. FRD rising with blur, or falling — the headline finding) is directly readable.
A local 4-metric readout (ssim/psnr/mae/mse) is also written inline for an instant signal.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ..data.preprocessing import load_mha, save_mha


def _unsharp(img, sigma=1.0, amount=1.0):
    return img + amount * (img - gaussian_filter(img, sigma))


def _bbox(mask):
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def build_corruptions(rng_seed: int = 0):
    """name -> fn(gt, pre, mask) -> corrupted prediction. Monotone families."""
    C = {}
    C["clean"] = lambda gt, pre, m: gt.copy()                       # perfect upper bound
    C["identity_pre"] = lambda gt, pre, m: pre.copy()               # trivial copy-input baseline
    for s in (0.5, 1.0, 2.0, 4.0):
        C[f"blur_s{s}"] = (lambda s: lambda gt, pre, m: gaussian_filter(gt, s))(s)
    for a in (0.5, 1.0, 2.0):
        C[f"sharpen_a{a}"] = (lambda a: lambda gt, pre, m: _unsharp(gt, 1.0, a))(a)
    for sd in (0.05, 0.1, 0.2):
        C[f"noise_{sd}"] = (lambda sd: lambda gt, pre, m:
                            gt + np.random.RandomState(rng_seed).randn(*gt.shape).astype(gt.dtype) * sd)(sd)
    for g in (0.5, 0.75, 1.25, 1.5):  # scale the enhancement: pre + g*(gt-pre)
        C[f"enh_x{g}"] = (lambda g: lambda gt, pre, m: pre + g * (gt - pre))(g)

    def roi_blur(gt, pre, m, s=3.0):
        out = gt.copy()
        bb = _bbox(m)
        if bb is not None:
            y0, y1, x0, x1 = bb
            out[y0:y1, x0:x1] = gaussian_filter(gt, s)[y0:y1, x0:x1]
        return out
    C["roi_blur_s3"] = roi_blur
    return C


def _local_metrics(pred, gt, mask):
    err = pred - gt
    mse = float((err ** 2).mean())
    mae = float(np.abs(err).mean())
    rng = float(gt.max() - gt.min()) + 1e-8
    psnr = float(10 * np.log10((rng ** 2) / (mse + 1e-12)))
    out = {"mse": mse, "mae": mae, "psnr": psnr}
    if mask is not None and mask.sum() > 0:
        m = mask > 0.5
        out["mae_roi"] = float(np.abs(err[m]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True, help=".../mha/ground_truth")
    ap.add_argument("--pre-dir", required=True, help=".../mha/input")
    ap.add_argument("--mask-dir", default="", help=".../mha/mask (optional)")
    ap.add_argument("--out-dir", default="probe")
    ap.add_argument("--limit", type=int, default=0, help="cap #cases (smoke test)")
    args = ap.parse_args()

    gt_dir, pre_dir = Path(args.gt_dir), Path(args.pre_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    out_root = Path(args.out_dir)
    cases = sorted(p.name for p in gt_dir.glob("*.mha"))
    if args.limit:
        cases = cases[:args.limit]
    print(f"{len(cases)} cases; corruptions:")

    corruptions = build_corruptions()
    local_rows = []
    for name, fn in corruptions.items():
        pred_dir = out_root / name / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        agg = {}
        for c in cases:
            gt = load_mha(gt_dir / c).astype(np.float32)
            pre = load_mha(pre_dir / c).astype(np.float32)
            mask = (load_mha(mask_dir / c).astype(np.float32)
                    if mask_dir and (mask_dir / c).exists() else None)
            pred = fn(gt, pre, mask).astype(np.float32)
            save_mha(pred, pred_dir / c)
            for k, v in _local_metrics(pred, gt, mask).items():
                agg[k] = agg.get(k, 0.0) + v
        agg = {k: v / len(cases) for k, v in agg.items()}
        # config.json so aggregate_metrics.py treats each corruption like a run
        (out_root / name / "config.json").write_text(json.dumps(
            {"name": name, "model": "probe", "corruption": name}, indent=2))
        # also drop the inline local metrics where the aggregator's local_* reader finds them
        (out_root / name / "metrics.json").write_text(json.dumps(agg, indent=2))
        local_rows.append({"corruption": name, **agg})
        print(f"  {name:16s} local mse={agg['mse']:.4f} mae={agg['mae']:.4f}")

    csv_path = out_root / "local_probe.csv"
    with csv_path.open("w", newline="") as f:
        keys = sorted({k for r in local_rows for k in r})
        w = csv.DictWriter(f, fieldnames=["corruption", *[k for k in keys if k != "corruption"]])
        w.writeheader()
        w.writerows(local_rows)
    print(f"\nwrote {csv_path}")
    print(f"next: run official evaluate.py per probe/<name>/predictions, then "
          f"aggregate_metrics.py --runs-dir {out_root} --out paper/probe.csv")


if __name__ == "__main__":
    main()
