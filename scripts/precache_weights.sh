#!/bin/bash
# Pre-cache the torchvision weights mama_synth needs, on a node WITH internet
# (i.e. the login node). HPC compute nodes (gpu/cpu partitions) have NO
# outbound internet — the http proxy is unreachable from the compute fabric —
# so VGG16 (perceptual loss) and Inception (FID) must be downloaded here first.
#
# Run ONCE on the login node, from the project root:
#   bash mama_synth/scripts/precache_weights.sh
#
# It writes into TORCH_HOME on scratch; train.slurm points TORCH_HOME at the same
# dir so the compute node reads the cached files instead of hitting the network.
set -euo pipefail

export TORCH_HOME=${TORCH_HOME:-/path/to/torch_cache}
mkdir -p "$TORCH_HOME/hub/checkpoints"
echo "TORCH_HOME=$TORCH_HOME"

# activate the same env used for training (override CONDA_ENV if named differently)
CONDA_ENV=${CONDA_ENV:-mama_env}
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

python - <<'PY'
import os, torch
print("torch", torch.__version__, "| TORCH_HOME", os.environ.get("TORCH_HOME"))

# VGG16 ImageNet weights -> used by loss.VGGPerceptual (the perceptual term)
from torchvision.models import vgg16, VGG16_Weights
vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
print("cached: VGG16 IMAGENET1K_V1")

# AlexNet LPIPS weights -> used by the --lpips training loss term
try:
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
    LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False)
    print("cached: LPIPS (alex)")
except Exception as e:
    print("LPIPS weights not cached (only needed if training with --lpips > 0):", e)

# Inception weights -> used by torch_fidelity for FID (optional; only if --compute-fid)
try:
    import torch_fidelity  # noqa
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    FeatureExtractorInceptionV3("inception", ["2048"])  # feature ids are strings, not ints
    print("cached: Inception-v3 (FID)")
except Exception as e:
    print("FID weights not cached (fine if you train with --no-fid):", e)
PY

echo "done. point train.slurm at TORCH_HOME=$TORCH_HOME"
