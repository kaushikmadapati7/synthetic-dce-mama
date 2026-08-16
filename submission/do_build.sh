#!/usr/bin/env bash
# Build the MAMA-SYNTH submission container. Build context = the mama_synth/ package dir
# (so the Dockerfile can COPY the package); the trained weights must be staged first.
set -e
SUB_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
PKG_DIR=$( cd "$SUB_DIR/.." &>/dev/null && pwd )      # the mama_synth/ package = build context

if [ ! -f "$SUB_DIR/model/model.pt" ] || [ ! -f "$SUB_DIR/model/config.json" ]; then
  echo "ERROR: stage the trained model first:"
  echo "  cp <best run>/checkpoints/<unet|multitask>_last.pt  $SUB_DIR/model/model.pt"
  echo "  cp <best run>/config.json                           $SUB_DIR/model/config.json"
  echo "(config.json supplies base_ch / n_upsamples / spatial_size / per_image_norm / out_channels)"
  exit 1
fi
docker build --no-cache --platform=linux/amd64 -t mama-synth-submit -f "$SUB_DIR/Dockerfile" "$PKG_DIR"
echo "built image: mama-synth-submit"
