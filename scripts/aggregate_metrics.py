#!/usr/bin/env python
"""Aggregate per-run results into one tidy CSV for the paper figures/tables.

Walks a runs/ directory; for each run subdir it joins:
  - config.json / config_eval.json   (vars(args) for that run)
  - metrics.json  / metrics_eval.json (THIS repo's local eval: ssim/psnr/mae/mse/*_roi/fid)
  - eval/metrics.json                 (the OFFICIAL scorer's 8 challenge metrics, if present)

The official scorer (mama-research/mama-synth evaluate.py) is run separately and its
output dropped at runs/<id>/eval/metrics.json. Its exact key spelling
varies (ssim_tumour vs ssim_tumor, auroc_contrast vs auroc_con, per-case list vs aggregate
dict), so extraction here is defensive: each of the 8 metrics has a list of candidate keys
and we search top-level, common aggregate containers, and (averaging) a per-case list.

Usage:
    python -m mama_synth.scripts.aggregate_metrics --runs-dir runs --out paper/metrics.csv
    python -m mama_synth.scripts.aggregate_metrics --runs-dir runs --names fam_unet,fam_gan
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# The 8 official challenge metrics -> candidate key spellings (first hit wins).
OFFICIAL = {
    "mse":             ["mse"],
    "lpips":           ["lpips", "lpips_alex", "perceptual"],
    "ssim_tumor":      ["ssim_tumour", "ssim_tumor", "ssim_roi", "ssim_tum"],
    "frd":             ["frd", "fréchet_radiomics_distance", "frechet_radiomics_distance"],
    "auroc_contrast":  ["auroc_contrast", "auroc_con", "auroc_pre_post", "auroc_prevspost"],
    "auroc_tumor_roi": ["auroc_tumour_roi", "auroc_tumor_roi", "auroc_tum", "auroc_roi"],
    "dice":            ["dice", "dsc", "dice_score"],
    "hd95":            ["hd95", "hausdorff95", "hd_95", "hausdorff_95"],
}
# Config fields worth pulling into the table (others are still written, see ALL_CONFIG).
CONFIG_KEYS = ["model", "spatial_size", "base_ch", "epochs", "enh_weight", "lpips", "mse",
               "l1", "ssim", "perceptual", "roi_weight", "roi_ssim", "grad_weight",
               "residual_mode", "per_image_norm", "scanner_augment", "ema",
               "stratify_split", "holdout_center", "aux_mask"]
# config loss-weight keys that collide with official metric names -> rename in the CSV
CONFIG_RENAME = {"mse": "mse_w", "lpips": "lpips_w"}
LOCAL_KEYS = ["ssim", "psnr", "mae", "mse", "mae_roi", "psnr_roi", "ssim_roi", "fid"]


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _find(obj, candidates):
    """Search a metrics blob for the first candidate key. Handles: flat dict; nested
    'aggregate'/'aggregates'/'mean'/'overall' container; per-case list (averaged)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        low = {k.lower(): v for k, v in obj.items()}
        for c in candidates:
            if c.lower() not in low:
                continue
            v = low[c.lower()]
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):  # official scorer wraps each metric as {mean, std}
                for mk in ("mean", "value", "avg"):
                    if isinstance(v.get(mk), (int, float)):
                        return float(v[mk])
        for container in ("aggregate", "aggregates", "mean", "means", "overall", "summary"):
            if container in low:
                hit = _find(low[container], candidates)
                if hit is not None:
                    return hit
        # per-case list nested under a key
        for container in ("per_case", "cases", "per_image", "results"):
            if container in low and isinstance(low[container], list):
                hit = _find(low[container], candidates)
                if hit is not None:
                    return hit
    if isinstance(obj, list):  # average the metric across per-case dicts
        vals = []
        for item in obj:
            v = _find(item, candidates)
            if v is not None:
                vals.append(v)
        if vals:
            return sum(vals) / len(vals)
    return None


def collect_run(run_dir: Path) -> dict | None:
    cfg = _load(run_dir / "config.json") or _load(run_dir / "config_eval.json")
    if cfg is None:
        return None
    row = {"name": run_dir.name}
    for k in CONFIG_KEYS:
        v = cfg.get(k)
        row[CONFIG_RENAME.get(k, k)] = " ".join(map(str, v)) if isinstance(v, list) else v

    local = _load(run_dir / "metrics.json") or _load(run_dir / "metrics_eval.json")
    for k in LOCAL_KEYS:
        row[f"local_{k}"] = _find(local, [k])

    official = (_load(run_dir / "eval" / "metrics.json")
                or _load(run_dir / "eval" / "metrics_eval.json"))
    row["has_official"] = official is not None
    for metric, cands in OFFICIAL.items():
        row[metric] = _find(official, cands)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="paper/metrics.csv")
    ap.add_argument("--names", default="", help="comma-separated subset of run names (else all)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"runs dir not found: {runs_dir}")
        return
    wanted = set(filter(None, args.names.split(",")))
    rows = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if wanted and d.name not in wanted:
            continue
        r = collect_run(d)
        if r is not None:
            rows.append(r)

    if not rows:
        print(f"no runs with a config.json found under {runs_dir}")
        return
    cols = ["name", *[CONFIG_RENAME.get(k, k) for k in CONFIG_KEYS], *OFFICIAL.keys(),
            "has_official", *[f"local_{k}" for k in LOCAL_KEYS]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    n_off = sum(r["has_official"] for r in rows)
    print(f"wrote {out}  ({len(rows)} runs, {n_off} with official 8-metric scores)")


if __name__ == "__main__":
    main()
