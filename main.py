"""End-to-end MAMA-SYNTH 2D pipeline entrypoint: data -> model -> train -> eval.

Task: synthesize the peak post-contrast breast DCE-MRI slice from the pre-contrast slice.

A 2D translation of the 3D pipeline's main.py. The harmonization stage is removed (replaced by
dataset-level z-score normalization in the data layer). Per-model training loops live in
mama_synth/training/ (one file per model); evaluation lives in mama_synth/eval.py.

Run from the project root (note the package name has no spaces):

    # smoke test on synthetic data (no downloads needed):
    python -m mama_synth.main --model ldm_flow --dummy --output-dir runs/smoke \
        --epochs 1 --vae-epochs 1 --limit 8 --device cpu

    # real data: point --data-root at the official preprocess.py output tree
    python -m mama_synth.main --model ldm_flow --data-root /path/to/preprocessed \
        --stats-json /path/to/training_pre_stats.json --output-dir runs/exp1 --epochs 100

Outputs written to --output-dir:
    config.json, train.log, checkpoints/, metrics.json, samples/{*.mha, montage.png}
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (PreprocessConfig, Stats, MamaSynthDataset, DummyMamaDataset,
                   load_stats, compute_stats)
from .loss import CustomLoss
from .training import TRAINERS
from .eval import evaluate

log = logging.getLogger("mama_synth")


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(output_dir / "train.log"); fh.setFormatter(fmt); log.addHandler(fh)


def make_criterion(args, device) -> CustomLoss:
    seg_critic = None
    if getattr(args, "seg_ckpt", ""):
        from .training.segmenter import build_segmenter
        seg_critic = build_segmenter(args, device)
        seg_critic.load_state_dict(torch.load(args.seg_ckpt, map_location=device))
        log.info(f"loaded frozen segmentation critic: {args.seg_ckpt} "
                 f"(seg_weight={args.seg_weight}, base_ch={args.seg_base_ch})")
    return CustomLoss(
        l1_weight=args.l1, ssim_weight=args.ssim, perceptual_weight=args.perceptual,
        roi_weight=args.roi_weight, lpips_weight=args.lpips, bg_weight=args.bg_weight,
        mse_weight=args.mse, data_range=args.data_range,
        roi_ssim_weight=args.roi_ssim, seg_weight=args.seg_weight, seg_critic=seg_critic,
        enh_weight=args.enh_weight, grad_weight=args.grad_weight,
    ).to(device)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _center_counts(names):
    import collections
    return dict(sorted(collections.Counter(n.split("_")[0] for n in names).items()))


def _split_indices(ids, val_frac, stratify, holdout_center=""):
    """Return (train_idx, test_idx).
      - holdout_center set -> leave-one-center-out: test = ALL of that center, train =
        the rest (the honest cross-center proxy for the challenge's external test).
      - stratify=True       -> sample val_frac of EACH center into test (in-distribution).
      - else                -> legacy first-fraction split (all one center, since sorted)."""
    if holdout_center:
        centers = [name.split("_")[0] for name in ids]
        train_idx = [i for i, c in enumerate(centers) if c != holdout_center]
        test_idx = [i for i, c in enumerate(centers) if c == holdout_center]
        return train_idx, test_idx
    if not stratify:
        n_test = max(1, int(len(ids) * val_frac))
        idx = list(range(len(ids)))
        return idx[n_test:], idx[:n_test]
    import collections
    by_center = collections.defaultdict(list)
    for i, name in enumerate(ids):
        by_center[name.split("_")[0]].append(i)
    train_idx, test_idx = [], []
    for _center, idxs in sorted(by_center.items()):
        n_test = max(1, int(len(idxs) * val_frac))
        test_idx += idxs[:n_test]
        train_idx += idxs[n_test:]
    return sorted(train_idx), sorted(test_idx)


def build_data(args):
    cfg = PreprocessConfig(spatial_size=tuple(args.spatial_size))

    if args.dummy:
        n = args.limit or 16
        train = DummyMamaDataset(n, cfg, seed=args.seed, per_image_norm=args.per_image_norm)
        test = DummyMamaDataset(max(2, n // 4), cfg, seed=args.seed + 1, per_image_norm=args.per_image_norm)
        log.info(f"[dummy] train slices: {len(train)}  test slices: {len(test)}")
    else:
        if not args.data_root:
            raise ValueError("--data-root is required unless --dummy is set")
        if args.stats_json:
            stats = load_stats(args.stats_json)
        else:
            log.warning("no --stats-json; computing z-score stats from the train inputs")
            probe = MamaSynthDataset(args.data_root, Stats(0.0, 1.0), cfg)
            from .data import load_mha
            stats = compute_stats(load_mha(probe.root / "mha" / "input" / nm) for nm in probe.ids)
        log.info(f"z-score stats: mean={stats.mean:.4f} std={stats.std:.4f}")

        pin = args.per_image_norm
        full = MamaSynthDataset(args.data_root, stats, cfg, per_image_norm=pin)        # plain (test + indexing)
        train_src = MamaSynthDataset(args.data_root, stats, cfg, augment=args.augment, per_image_norm=pin,
                                     scanner_augment=args.scanner_augment, scan_bias=args.scan_bias,
                                     scan_contrast=args.scan_contrast, scan_noise=args.scan_noise)  # train
        if args.test_root:
            train = train_src
            test = MamaSynthDataset(args.test_root, stats, cfg, per_image_norm=pin)
        else:  # hold out a fraction for a local test view (test is never augmented)
            train_idx, test_idx = _split_indices(full.ids, args.val_frac, args.stratify_split,
                                                 args.holdout_center)
            test = torch.utils.data.Subset(full, test_idx)
            train = torch.utils.data.Subset(train_src, train_idx)
            log.info(f"split: stratify={args.stratify_split} "
                     f"test centers={_center_counts([full.ids[i] for i in test_idx])}")
        if args.limit:
            train = torch.utils.data.Subset(train, range(min(args.limit, len(train))))
            test = torch.utils.data.Subset(test, range(min(max(1, args.limit // 4), len(test))))
        log.info(f"train slices: {len(train)}  test slices: {len(test)}")

    dl = lambda ds, shuf: DataLoader(ds, batch_size=args.batch_size, shuffle=shuf,
                                     num_workers=args.num_workers, drop_last=shuf,
                                     pin_memory=torch.cuda.is_available())
    return dl(train, True), (dl(test, False) if len(test) else None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    set_seed(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    # in eval-only mode don't clobber the original training config/metrics
    cfg_name = "config_eval.json" if args.eval_only else "config.json"
    metrics_name = "metrics_eval.json" if args.eval_only else "metrics.json"
    (out / cfg_name).write_text(json.dumps(vars(args), indent=2))
    log.info(f"device={device}  model={args.model}  eval_only={args.eval_only}")
    log.info(f"config: {json.dumps(vars(args))}")

    train_loader, test_loader = build_data(args)
    criterion = make_criterion(args, device)

    t0 = time.time()
    trainer = TRAINERS[args.model]
    _, gen = trainer(args, train_loader, test_loader, criterion, device)
    log.info(f"{'eval-only setup' if args.eval_only else 'training'} done in {(time.time() - t0) / 60:.1f} min")

    metrics = evaluate(args, gen, test_loader, device)
    (out / metrics_name).write_text(json.dumps(metrics, indent=2))
    log.info(f"artifacts written to {out}")


def parse_args():
    p = argparse.ArgumentParser(description="MAMA-SYNTH 2D synthetic post-contrast pipeline")
    p.add_argument("--model", choices=list(TRAINERS), default="ldm_flow")
    p.add_argument("--data-root", default="", help="official preprocess.py output tree (mha/...)")
    p.add_argument("--test-root", default="", help="separate test tree (else split off --val-frac)")
    p.add_argument("--stats-json", default="", help="training_pre_stats.json (z-score mean/std)")
    p.add_argument("--dummy", action="store_true", help="use synthetic data (smoke testing)")
    p.add_argument("--output-dir", default="runs/exp")
    # data / geometry
    p.add_argument("--spatial-size", type=int, nargs=2, default=[256, 256], help="(H, W)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--stratify-split", action="store_true",
                   help="stratify train/test across centers (legacy split is all-one-center; "
                        "requires retraining — a model's eval split must match its train split)")
    p.add_argument("--holdout-center", default="",
                   help="leave-one-center-out: test on ALL of this center (e.g. DUKE/ISPY1/"
                        "ISPY2/NACT), train on the rest — honest cross-center proxy for the "
                        "challenge's external test (takes precedence over --stratify-split)")
    p.add_argument("--augment", action="store_true",
                   help="random flip/rotation on the TRAIN split (reduces overfitting)")
    p.add_argument("--scanner-augment", action="store_true",
                   help="simulate cross-scanner appearance on the TRAIN split (bias field + "
                        "contrast gain + noise); targets cross-site generalization for the Test phase")
    p.add_argument("--scan-bias", type=float, default=0.3,
                   help="--scanner-augment: max smooth bias-field deviation from 1.0")
    p.add_argument("--scan-contrast", type=float, default=0.2,
                   help="--scanner-augment: max global contrast-gain deviation from 1.0")
    p.add_argument("--scan-noise", type=float, default=0.05,
                   help="--scanner-augment: max additive thermal-noise std (per image)")
    p.add_argument("--per-image-norm", action="store_true",
                   help="instance-normalize each image by its own cond mean/std (scanner-robust); "
                        "inverted exactly back to the global space in eval")
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--vae-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--limit", type=int, default=0, help="cap #slices (smoke testing)")
    # model size
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--residual-mode", choices=["residual", "direct", "positive"], default="residual",
                   help="unet/multitask output parameterization: residual (post=pre+f, default), "
                        "direct (post=f, no skip-add), positive (post=pre+softplus(f), non-negative "
                        "enhancement ansatz). See paper §3.6")
    p.add_argument("--ema", action="store_true",
                   help="track an exponential moving average of the weights and SAVE the EMA "
                        "copy as the checkpoint (smoother, better-generalizing model for free; "
                        "no loss/architecture change). Implemented for --model multitask")
    p.add_argument("--ema-decay", type=float, default=0.9999,
                   help="EMA decay (higher = longer averaging window)")
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--n-upsamples", type=int, default=4, help="GAN up/down sampling depth")
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--ch-mults", type=int, nargs="+", default=[1, 2, 4], help="VAE")
    p.add_argument("--unet-ch-mults", type=int, nargs="+", default=[1, 2, 4], help="diffusion UNet")
    p.add_argument("--kl-weight", type=float, default=1e-6)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--vae-ckpt", default="")
    p.add_argument("--unet-ckpt", default="",
                   help="frozen base residual-U-Net checkpoint for --model refine (the sharpener)")
    p.add_argument("--base-out-channels", type=int, default=1,
                   help="out_channels of the --unet-ckpt base for --model refine. Use 2 for a "
                        "multitask base (post + aux mask); the refiner reads channel 0 (post)")
    p.add_argument("--final-act", choices=["identity", "tanh"], default="identity",
                   help="generator/decoder output activation (identity for z-scored data)")
    # eval-only / export
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; load a checkpoint and just evaluate")
    p.add_argument("--ckpt", default="",
                   help="checkpoint for --eval-only (default: <output-dir>/checkpoints/<model>_last.pt)")
    p.add_argument("--export-preds", action="store_true",
                   help="save every test prediction as .mha (native size) for the challenge evaluate.py")
    # loss
    p.add_argument("--l1", type=float, default=1.0)
    p.add_argument("--mse", type=float, default=0.0,
                   help="weight of a global L2/MSE term (matches the challenge MSE metric; "
                        "L1 alone smooths the bright-enhancement peaks MSE punishes)")
    p.add_argument("--ssim", type=float, default=1.0)
    p.add_argument("--perceptual", type=float, default=0.1)
    p.add_argument("--lpips", type=float, default=0.0,
                   help="weight of an AlexNet-LPIPS loss term (targets the challenge LPIPS metric)")
    p.add_argument("--roi-weight", type=float, default=10.0,
                   help="tumor-ROI emphasis in the loss (1 disables; needs a mask)")
    p.add_argument("--roi-ssim", type=float, default=0.0,
                   help="weight of a tumor-ROI SSIM term (decoupled from --ssim, so it can "
                        "be on while the global SSIM term stays off; needs roi-weight>1)")
    p.add_argument("--seg-weight", type=float, default=0.0,
                   help="weight of the segmentation-perceptual term: a frozen tumor segmenter "
                        "(--seg-ckpt) scores the prediction, driving tumor DETECTABILITY")
    p.add_argument("--seg-ckpt", default="",
                   help="frozen segmenter checkpoint (from --model segmenter) for --seg-weight")
    p.add_argument("--seg-base-ch", type=int, default=32,
                   help="base channels of the segmentation critic (must match its training)")
    p.add_argument("--enh-weight", type=float, default=0.0,
                   help="enhancement-magnitude term: match the ROI's mean+peak intensity to "
                        "the real post (lifts under-enhanced tumors -> AUROC pre-vs-post). Needs a mask")
    p.add_argument("--grad-weight", type=float, default=0.0,
                   help="gradient-difference (high-frequency) term: match |grad(pred)| to "
                        "|grad(target)| globally + extra in the ROI -> sharper tumor texture, "
                        "lower FRD (supervised, non-hallucinating counterpart to the refiner GAN)")
    p.add_argument("--bg-weight", type=float, default=0.0,
                   help="background-faithfulness: penalize residual |pred-pre| outside the "
                        "tumor mask (0 disables; combats global over-enhancement)")
    p.add_argument("--aux-mask", type=float, default=1.0,
                   help="weight of the auxiliary mask-prediction loss for --model multitask")
    p.add_argument("--data-range", type=float, default=10.0,
                   help="SSIM/PSNR range; 10.0 matches the challenge eval (±5σ z-score)")
    p.add_argument("--adv-weight", type=float, default=1.0)
    p.add_argument("--roi-patch", type=int, default=128,
                   help="--model refine: ROI crop size fed to the tumor-focused discriminator")
    p.add_argument("--roi-pad-frac", type=float, default=0.25,
                   help="--model refine: context margin added around the tumor bbox before cropping")
    # evaluation
    p.add_argument("--compute-fid", action="store_true", default=True)
    p.add_argument("--no-fid", dest="compute_fid", action="store_false")
    p.add_argument("--fid-batch-size", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    main()
