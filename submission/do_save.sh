#!/usr/bin/env bash
# Export the submission image as a .tar.gz for upload to Grand Challenge.
set -e
SUB_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
VERSION="${VERSION:-v0.1.0}"
OUT="$SUB_DIR/mama-synth-submit-${VERSION}.tar.gz"
bash "$SUB_DIR/do_build.sh"
docker save mama-synth-submit | gzip -c > "$OUT"
echo "saved: $OUT  (upload at grand-challenge.org -> your Algorithm -> Container Management)"
