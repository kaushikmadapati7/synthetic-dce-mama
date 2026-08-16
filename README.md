# When Regression Beats Generation: Virtual Contrast-Enhanced Breast MRI and What Its Metrics Reward

Code for the paper *"When Regression Beats Generation: Virtual Contrast-Enhanced Breast
MRI and What Its Metrics Reward"* (Deep-Breath workshop, MICCAI 2026).

MAMA-SYNTH challenge (MICCAI 2026 / Deep-Breath): synthesising a 2D
peak-enhancement **post-contrast** breast-MRI slice from the corresponding
**pre-contrast** slice, so the gadolinium injection can be skipped.

The core model is a **residual U-Net**: it predicts an enhancement residual that is added
to the pre-contrast input via an identity connection, `ŷ = x + f_θ(x)`, with an
enhancement-magnitude loss and an auxiliary tumour-segmentation head (used only during
training). Conditional-GAN and latent-diffusion baselines are also included for the
family comparison.

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. The perceptual loss and FID download torchvision backbones on first use;
offline clusters should pre-cache them (see `scripts/precache_weights.sh`).

## Quick start (CPU smoke test, synthetic data, no downloads)

```bash
python -m mama_synth.main --model unet --dummy --output-dir runs/smoke \
    --spatial-size 64 64 --base-ch 16 --epochs 1 --limit 8 --device cpu --no-fid
```

This trains, samples, and evaluates end-to-end on random data and writes
`config.json`, `checkpoints/`, `metrics.json`, and `samples/`.

## Training on real data

Preprocess the MAMA-MIA cohort to the official `.mha` tree, then:

```bash
python -m mama_synth.main --model unet --data-root /path/to/preprocessed \
    --stats-json /path/to/preprocessed/identity_stats.json \
    --spatial-size 512 512 --base-ch 32 --epochs 300 --batch-size 8 \
    --mse 1 --lpips 1 --ssim 0 --perceptual 0 --augment --stratify-split
```

Cluster launch scripts are in `scripts/` (SLURM). **Paths in those scripts are
placeholders** (`/path/to/...`) — edit them for your environment.

## Layout

```
mama_synth/
├── main.py                 entrypoint: data → model → train → eval
├── eval.py, metrics.py     2D evaluation (SSIM/PSNR/MAE/MSE + FID)
├── data/                   .mha I/O, dataset z-score, MamaSynthDataset
├── loss/                   SSIM + VGG perceptual + custom loss
├── models/                 residual U-Net, conditional GAN, latent DDPM / flow-matching, 2D KL-VAE
├── training/              one loop per model family (registry in __init__.py)
├── scripts/                HPC preprocessing / training / evaluation helpers
└── submission/             Grand Challenge inference container
```

Models are dispatched via `--model` from the `TRAINERS` registry
(`unet`, `multitask`, `gan`, `ldm_ddpm`, `ldm_flow`, …).

## Evaluation

`--stats-json` feeds the dataset-level z-score used by the official preprocessing; passing
per-image-normalised or re-normalised stats would double-normalise. SSIM/PSNR/FID use
`--data-range 4.0` (≈±2σ) and a per-slice min–max for the FID uint8 conversion, since
z-scored data is unbounded. Downstream challenge metrics (FRD, AUROCs, DICE, HD95) are
produced by the official MAMA-SYNTH evaluation harness.
