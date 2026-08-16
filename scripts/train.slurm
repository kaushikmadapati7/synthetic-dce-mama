#!/bin/bash
#SBATCH --job-name=mama_synth
#SBATCH --partition=pod
#SBATCH --qos=pod
#SBATCH --gres=gpu:1
# (pod = H200 143GB, less contended. Override per-job on the CLI, e.g.
#  `sbatch --partition=gpu --qos=normal mama_synth/scripts/train.slurm`)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Train the MAMA-SYNTH 2D pipeline on the preprocessed .mha tree.
#
# Submit from the PROJECT ROOT (the dir that contains mama_synth/), e.g.:
#   sbatch mama_synth/scripts/train.slurm
# Override any default via environment variables, e.g.:
#   MODEL=gan EPOCHS=200 DATA_ROOT=/path/to/mama_mia/preprocessed \
#       sbatch mama_synth/scripts/train.slurm
#
# PREREQUISITES (HPC compute nodes have NO internet — handle these first):
#   1. Data preprocessed to the .mha tree   -> mama_synth/scripts/preprocess.slurm
#   2. Torchvision weights pre-cached on the LOGIN node:
#          bash mama_synth/scripts/precache_weights.sh
#      (otherwise the VGG16 perceptual loss silently falls back to random init,
#       and FID can't run at all)
set -euo pipefail

# ---- configurable knobs (env-overridable) -------------------------------------
MODEL=${MODEL:-ldm_flow}                                   # ldm_flow | ldm_ddpm | gan
DATA_ROOT=${DATA_ROOT:-/path/to/mama_mia/preprocessed}
# The official preprocess.py ALREADY z-scores the .mha it writes, so we feed them
# as-is via identity (mean 0 / std 1) stats. Passing training_pre_stats.json here
# would DOUBLE-normalize and wreck training. Override STATS_JSON only for data that
# is NOT already z-scored.
STATS_JSON=${STATS_JSON:-${DATA_ROOT}/identity_stats.json}
TEST_ROOT=${TEST_ROOT:-}                                   # optional separate test tree
OUTPUT_DIR=${OUTPUT_DIR:-runs/${MODEL}_${SLURM_JOB_ID:-local}}
EPOCHS=${EPOCHS:-100}
VAE_EPOCHS=${VAE_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}
SPATIAL_SIZE=${SPATIAL_SIZE:-"256 256"}                    # H W (divisible by 2**n_upsamples for gan)
ROI_WEIGHT=${ROI_WEIGHT:-10}                               # tumor-ROI loss emphasis (needs masks)
ADV_WEIGHT=${ADV_WEIGHT:-1.0}                              # GAN adversarial weight; lower (~0.1) = recon-dominant (pix2pix-style)
UNET_CKPT=${UNET_CKPT:-}                                   # --model refine: frozen base U-Net checkpoint to sharpen
ROI_PATCH=${ROI_PATCH:-128}                                # --model refine: tumor ROI crop size for the focused discriminator
ROI_PAD_FRAC=${ROI_PAD_FRAC:-0.25}                         # --model refine: context margin around the tumor bbox
ROI_SSIM=${ROI_SSIM:-0.0}                                  # tumor-ROI SSIM term (decoupled from global SSIM; needs ROI_WEIGHT>1)
SEG_WEIGHT=${SEG_WEIGHT:-0.0}                              # segmentation-perceptual term weight (drives tumor detectability)
ENH_WEIGHT=${ENH_WEIGHT:-0.0}                              # enhancement-magnitude term: match ROI mean+peak intensity (lifts AUROC pre-vs-post)
GRAD_WEIGHT=${GRAD_WEIGHT:-0.0}                             # gradient-difference (high-freq) term: match |grad| globally + ROI (sharper texture -> lower FRD)
SEG_CKPT=${SEG_CKPT:-}                                     # frozen segmenter checkpoint for SEG_WEIGHT (from MODEL=segmenter)
SEG_BASE_CH=${SEG_BASE_CH:-32}                             # base channels of the segmentation critic (must match its training)
LPIPS_WEIGHT=${LPIPS_WEIGHT:-0.0}                          # AlexNet-LPIPS loss weight (targets the challenge LPIPS metric)
BG_WEIGHT=${BG_WEIGHT:-0.0}                                # background-faithfulness: penalize residual outside the tumor (combats over-enhancement)
MSE_WEIGHT=${MSE_WEIGHT:-0.0}                              # global L2 term matching the challenge MSE metric (L1 smooths peaks; MSE reproduces them)
AUGMENT=${AUGMENT:-0}                                      # 1 -> train-split flip/rotation augmentation
SCANNER_AUGMENT=${SCANNER_AUGMENT:-0}                       # 1 -> cross-scanner appearance aug (bias field + contrast + noise); targets Test cross-site
SCAN_BIAS=${SCAN_BIAS:-0.3}                                 # max bias-field deviation from 1.0
SCAN_CONTRAST=${SCAN_CONTRAST:-0.2}                         # max global contrast-gain deviation from 1.0
SCAN_NOISE=${SCAN_NOISE:-0.05}                              # max additive noise std (per image)
STRATIFY=${STRATIFY:-0}                                    # 1 -> stratify train/test split across centers
HOLDOUT_CENTER=${HOLDOUT_CENTER:-}                         # e.g. DUKE -> leave-one-center-out (cross-center test)
PER_IMAGE_NORM=${PER_IMAGE_NORM:-0}                        # 1 -> per-image (instance) normalization (scanner-robust)
RESIDUAL_MODE=${RESIDUAL_MODE:-residual}                    # residual | direct | positive (unet/multitask output ansatz)
BASE_CH=${BASE_CH:-32}                                      # U-Net base channel width (capacity ablation)
SEED=${SEED:-0}                                             # RNG seed (init + augmentation); vary for seed-variance runs
COMPUTE_FID=${COMPUTE_FID:-0}                              # 1 only if Inception weights pre-cached
EXPORT_PREDS=${EXPORT_PREDS:-0}                             # 1 -> write predictions/*.mha at end (for the official scorer)
CONDA_ENV=${CONDA_ENV:-mama_env}
EXTRA_ARGS=${EXTRA_ARGS:-}

# ---- offline weight cache (must match precache_weights.sh) ---------------------
export TORCH_HOME=${TORCH_HOME:-/path/to/torch_cache}

# ---- environment --------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# site-specific module loads (uncomment/edit as needed)
# module load cuda/12.1

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

# FID needs Inception weights + works only if precached; off by default
if [ "${COMPUTE_FID}" = "1" ]; then FID_FLAG="--compute-fid"; else FID_FLAG="--no-fid"; fi
AUGMENT_FLAG=""; [ "${AUGMENT}" = "1" ] && AUGMENT_FLAG="--augment"
SCANNER_AUGMENT_FLAG=""; [ "${SCANNER_AUGMENT}" = "1" ] && SCANNER_AUGMENT_FLAG="--scanner-augment --scan-bias ${SCAN_BIAS} --scan-contrast ${SCAN_CONTRAST} --scan-noise ${SCAN_NOISE}"
STRATIFY_FLAG=""; [ "${STRATIFY}" = "1" ] && STRATIFY_FLAG="--stratify-split"
PIN_FLAG=""; [ "${PER_IMAGE_NORM}" = "1" ] && PIN_FLAG="--per-image-norm"
EXPORT_FLAG=""; [ "${EXPORT_PREDS}" = "1" ] && EXPORT_FLAG="--export-preds"
HOLDOUT_FLAG=""; [ -n "${HOLDOUT_CENTER}" ] && HOLDOUT_FLAG="--holdout-center ${HOLDOUT_CENTER}"
# auto-create pass-through (identity) stats if missing — the .mha are already z-scored
if [ ! -f "${STATS_JSON}" ]; then
    echo '{"mean": 0.0, "std": 1.0}' > "${STATS_JSON}"
    echo "created pass-through stats ${STATS_JSON} (.mha already z-scored by preprocess.py)"
fi
STATS_FLAG="--stats-json ${STATS_JSON}"
# pass --test-root only if set
TEST_FLAG=""; [ -n "${TEST_ROOT}" ] && TEST_FLAG="--test-root ${TEST_ROOT}"
# --model refine needs a frozen base U-Net checkpoint to sharpen
UNET_CKPT_FLAG=""; [ -n "${UNET_CKPT}" ] && UNET_CKPT_FLAG="--unet-ckpt ${UNET_CKPT}"
# segmentation-perceptual term needs a frozen segmenter checkpoint
SEG_CKPT_FLAG=""; [ -n "${SEG_CKPT}" ] && SEG_CKPT_FLAG="--seg-ckpt ${SEG_CKPT}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} model=${MODEL}"
echo "data_root=${DATA_ROOT}"
echo "stats_json=${STATS_JSON} (present: $([ -f "${STATS_JSON}" ] && echo yes || echo no))"
echo "output_dir=${OUTPUT_DIR}  TORCH_HOME=${TORCH_HOME}"
nvidia-smi || true

# ---- sanity checks ------------------------------------------------------------
if [ ! -d "${DATA_ROOT}/mha/input" ]; then
    echo "ERROR: ${DATA_ROOT}/mha/input not found — has preprocess.slurm been run?" >&2
    exit 1
fi

# ---- run ----------------------------------------------------------------------
srun python -m mama_synth.main \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    ${STATS_FLAG} \
    ${TEST_FLAG} \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --vae-epochs "${VAE_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --spatial-size ${SPATIAL_SIZE} \
    --roi-weight "${ROI_WEIGHT}" \
    --roi-ssim "${ROI_SSIM}" \
    --adv-weight "${ADV_WEIGHT}" \
    --roi-patch "${ROI_PATCH}" \
    --roi-pad-frac "${ROI_PAD_FRAC}" \
    --seg-weight "${SEG_WEIGHT}" \
    --seg-base-ch "${SEG_BASE_CH}" \
    --enh-weight "${ENH_WEIGHT}" \
    --grad-weight "${GRAD_WEIGHT}" \
    --residual-mode "${RESIDUAL_MODE}" \
    --base-ch "${BASE_CH}" \
    --seed "${SEED}" \
    ${UNET_CKPT_FLAG} \
    ${SEG_CKPT_FLAG} \
    --lpips "${LPIPS_WEIGHT}" \
    --bg-weight "${BG_WEIGHT}" \
    --mse "${MSE_WEIGHT}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    ${FID_FLAG} \
    ${AUGMENT_FLAG} \
    ${SCANNER_AUGMENT_FLAG} \
    ${EXPORT_FLAG} \
    ${STRATIFY_FLAG} \
    ${PIN_FLAG} \
    ${HOLDOUT_FLAG} \
    ${EXTRA_ARGS}

echo "done -> ${OUTPUT_DIR}"
