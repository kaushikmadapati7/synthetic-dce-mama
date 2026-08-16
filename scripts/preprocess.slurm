#!/bin/bash
#SBATCH --job-name=mama_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Turn the raw MAMA-MIA 3D download into the 2D .mha tree that mama_synth trains on,
# using the OFFICIAL challenge preprocessing scripts (mama-research/mama-synth).
# CPU-only, no internet required.
#
# Submit from the project root:
#   sbatch mama_synth/scripts/preprocess.slurm
#
# PREREQUISITE — on the LOGIN node, clone the CHALLENGE repo (not the dataset repo):
#   git clone https://github.com/mama-research/mama-synth.git \
#       /path/to/mama_mia/mama-synth
#
# Paths below are confirmed against the downloaded tree:
#   images/<patient>/<patient>_0000.nii.gz (pre) .. _000N.nii.gz (post phases)
#   segmentations/expert/<patient>.nii.gz   (challenge uses EXPERT masks)
set -euo pipefail

# ---- paths --------------------------------------------------------------------
MAMA_REPO=${MAMA_REPO:-/path/to/mama_mia/mama-synth}   # challenge repo clone
RAW=${RAW:-/path/to/mama_mia/raw}
IMAGE_DIR=${IMAGE_DIR:-${RAW}/images}
SEG_DIR=${SEG_DIR:-${RAW}/segmentations/expert}
OUTPUT_DIR=${OUTPUT_DIR:-/path/to/mama_mia/preprocessed}
STATS_JSON=${STATS_JSON:-${OUTPUT_DIR}/training_pre_stats.json}
SKIP_AMBIGUOUS=${SKIP_AMBIGUOUS:-1}                  # 1 -> pass --skip_ambiguous_shapes
CONDA_ENV=${CONDA_ENV:-mama_env}

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs "${OUTPUT_DIR}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

PREP="${MAMA_REPO}/src/preprocessing"
if [ ! -f "${PREP}/preprocess.py" ]; then
    echo "ERROR: ${PREP}/preprocess.py not found. On the LOGIN node first run:" >&2
    echo "  git clone https://github.com/mama-research/mama-synth.git ${MAMA_REPO}" >&2
    exit 1
fi

echo "host=$(hostname) image_dir=${IMAGE_DIR} seg_dir=${SEG_DIR}"
echo "output_dir=${OUTPUT_DIR}  stats_json=${STATS_JSON}"

# ---- 1. dataset z-score stats from the pre-contrast images --------------------
# NOTE: this computes stats over every case in IMAGE_DIR. For a leakage-free run,
# point IMAGE_DIR at a training-only subset (see raw/train_test_splits.csv); for our
# internal training that distinction doesn't affect the held-out challenge test set.
python "${PREP}/compute_dataset_stats.py" \
    --image_dir "${IMAGE_DIR}" \
    --output_path "${STATS_JSON}"

# ---- 2. 3D -> 2D peak/largest-tumor slice extraction + z-score -> .mha tree ----
SKIP_FLAG=""; [ "${SKIP_AMBIGUOUS}" = "1" ] && SKIP_FLAG="--skip_ambiguous_shapes"
python "${PREP}/preprocess.py" \
    --image_dir "${IMAGE_DIR}" \
    --seg_dir "${SEG_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --global_stats "${STATS_JSON}" \
    ${SKIP_FLAG}

echo "done. expect ${OUTPUT_DIR}/mha/{input,ground_truth,mask}/ + ${STATS_JSON}"
echo "then train with: DATA_ROOT=${OUTPUT_DIR} sbatch mama_synth/scripts/train.slurm"
