#!/usr/bin/env python3
"""MAMA-SYNTH submission inference — residual U-Net (pre-contrast -> post-contrast).

Grand Challenge I/O contract:
    IN : /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha
    OUT: /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha

Inputs are 2-D globally-z-scored float32 .mha. We replicate the training pipeline
exactly: center-crop/pad to the model size, (optionally) per-image-normalize, run the
model, then INVERT both steps to land back at the input's native size and global
z-score space, preserving spacing/origin/direction metadata.

Model architecture + flags are read from a bundled `model/config.json`; the weights
are `model/model.pt`. Works for `unet` (out_channels=1) and `multitask`
(out_channels=2 -> the post is channel 0). Imports the real mama_synth code so the
preprocessing matches training byte-for-byte.
"""
from __future__ import annotations

import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from mama_synth.models import ResidualUNet2D
from mama_synth.data.preprocessing import center_crop_pad_2d

INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
INPUT_SLUG = os.environ.get("MAMA_INPUT_SLUG", "pre-contrast-dce-mri-slice-breast")
OUTPUT_SLUG = os.environ.get("MAMA_PREDICTION_SLUG", "synthetic-contrast-dce-mri-slice-breast")

HERE = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("MAMA_MODEL_DIR", str(HERE / "model")))


def _load_config() -> dict:
    cfg = {"base_ch": 32, "n_upsamples": 4, "spatial_size": [256, 256],
           "per_image_norm": True, "out_channels": 1}
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        cfg.update(json.loads(cfg_path.read_text()))
    return cfg


def _find_input_image() -> Path:
    search = INPUT_PATH / "images" / INPUT_SLUG
    for ext in ("*.mha", "*.nii.gz", "*.nii"):
        c = glob(str(search / ext))
        if c:
            if len(c) > 1:
                print(f"WARNING: {len(c)} inputs in {search}; using {c[0]}", file=sys.stderr)
            return Path(c[0])
    raise FileNotFoundError(f"No input image in {search} (check INPUT_SLUG='{INPUT_SLUG}')")


def run() -> int:
    cfg = _load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResidualUNet2D(in_channels=1, out_channels=int(cfg["out_channels"]),
                         base_ch=int(cfg["base_ch"]), depth=int(cfg["n_upsamples"])).to(device)
    net.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
    net.eval()

    f = _find_input_image()
    img = sitk.ReadImage(str(f))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # 2-D slice
    arr2d = arr[0] if arr.ndim == 3 else arr
    native = arr2d.shape

    x = center_crop_pad_2d(arr2d, tuple(cfg["spatial_size"]), 0.0)
    m, s = 0.0, 1.0
    if cfg["per_image_norm"]:
        m, s = float(x.mean()), float(x.std()) + 1e-6
        x = (x - m) / s

    with torch.no_grad():
        t = torch.from_numpy(x[None, None]).float().to(device)
        pred = net(t)[:, :1][0, 0].cpu().numpy()           # post channel
    if cfg["per_image_norm"]:
        pred = pred * s + m                                # invert -> global space
    # inverse crop/pad -> native size; reconstruct any un-predicted periphery from
    # the pre (arr2d) rather than a 0 pad, so images larger than spatial_size keep
    # their real border tissue instead of being blanked out.
    pred = center_crop_pad_2d(pred, native, fill=arr2d.astype(pred.dtype))

    out = sitk.GetImageFromArray(pred.astype(np.float32))
    out.CopyInformation(img)                               # preserve spacing/origin/direction
    out_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_dir / "output.mha"), useCompression=True)
    print(f"wrote {out_dir/'output.mha'}  size={pred.shape}  per_image_norm={cfg['per_image_norm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
