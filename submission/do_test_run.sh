#!/usr/bin/env bash
# Run the submission container locally against a test pre-contrast .mha, mirroring the
# Grand Challenge runtime (no network, mounted /input read-only + /output).
set -e
SUB_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
bash "$SUB_DIR/do_build.sh"

IN_DIR="$SUB_DIR/test/input/images/pre-contrast-dce-mri-slice-breast"
mkdir -p "$IN_DIR"
if [ -z "$(ls -A "$IN_DIR"/*.mha 2>/dev/null)" ]; then
  echo "ERROR: put a pre-contrast .mha in $IN_DIR"
  echo "  e.g. cp <preprocessed>/mha/input/DUKE_001.mha $IN_DIR/"
  exit 1
fi
rm -rf "$SUB_DIR/test/output"; mkdir -p "$SUB_DIR/test/output"

docker run --rm --network=none --memory=8g \
  -v "$SUB_DIR/test/input:/input:ro" \
  -v "$SUB_DIR/test/output:/output" \
  mama-synth-submit

echo "=== output ==="
find "$SUB_DIR/test/output" -type f
