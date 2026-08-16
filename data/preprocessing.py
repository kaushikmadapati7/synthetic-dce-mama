"""Geometry + intensity preprocessing for MAMA-SYNTH 2D slices.

The official challenge preprocessing (mama-research/mama-synth preprocess.py) already
does the heavy lifting: it picks the peak-enhancement post-contrast phase, selects the
slice with the largest tumor area, and emits aligned 2D `.mha` files under
mha/{input,ground_truth,mask}/. This module only handles what the model needs at load
time: reading `.mha`, dataset-level z-score normalization, and center crop/pad to a fixed
size so a batch can be stacked.

Normalization differs from the 3D pipeline (which used per-image percentile clipping to
[-1, 1]). MAMA-SYNTH specifies dataset-level z-score using the *training pre-contrast*
mean/std, applied to both the pre- and post-contrast slices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass
class PreprocessConfig:
    spatial_size: tuple | None = (256, 256)  # (H, W); None = leave as-is
    pad_value: float = 0.0                   # background after z-score (~mean)


# ---------------------------------------------------------------------------
# .mha IO  (arrays are (H, W) float32 from GetArrayFromImage on a 2D image)
# ---------------------------------------------------------------------------
def load_mha(path) -> np.ndarray:
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return np.squeeze(arr)  # drop any singleton leading axis -> (H, W)


def save_mha(arr: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.asarray(arr, dtype=np.float32)), str(path))


# ---------------------------------------------------------------------------
# intensity + shape ops
# ---------------------------------------------------------------------------
def zscore_normalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (arr - mean) / (std + 1e-8)


@dataclass
class Stats:
    mean: float
    std: float

    def to_json(self, path):
        Path(path).write_text(json.dumps({"mean": self.mean, "std": self.std}, indent=2))

    @classmethod
    def from_json(cls, path):
        s = json.loads(Path(path).read_text())
        return cls(float(s["mean"]), float(s["std"]))


def load_stats(json_path) -> Stats:
    """Load dataset z-score stats (the challenge's training_pre_stats.json)."""
    return Stats.from_json(json_path)


def compute_stats(arrays) -> Stats:
    """Fallback: dataset mean/std over an iterable of pre-contrast arrays.

    Used for synthetic/dummy data; on real data, prefer the challenge's
    training_pre_stats.json via load_stats().
    """
    s, sq, n = 0.0, 0.0, 0
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        s += a.sum(); sq += (a ** 2).sum(); n += a.size
    mean = s / max(1, n)
    var = max(0.0, sq / max(1, n) - mean ** 2)
    return Stats(float(mean), float(var ** 0.5))


def center_crop_pad_2d(arr: np.ndarray, size, pad_value=0.0, fill=None) -> np.ndarray:
    """Center crop or pad an (H, W) array to exactly `size`.

    Border pixels not covered by `arr` default to `pad_value`. If `fill` (an
    array of shape `size`) is given, the border is taken from it instead. Used
    on the inverse (prediction -> native) step to reconstruct the un-predicted
    periphery from the pre-contrast image rather than a flat constant: when the
    native image is larger than the model's spatial_size, the model only sees
    the centre crop, and a 0.0 pad would blank out real tissue it never saw
    (worse than identity). Filling from the pre keeps that tissue intact.
    """
    if fill is not None:
        out = np.array(fill, dtype=arr.dtype, copy=True)
        if out.shape != tuple(size):
            raise ValueError(f"fill shape {out.shape} != target size {tuple(size)}")
    else:
        out = np.full(size, pad_value, dtype=arr.dtype)
    src_slices, dst_slices = [], []
    for a, s in zip(arr.shape, size):
        if a >= s:
            start = (a - s) // 2
            src_slices.append(slice(start, start + s))
            dst_slices.append(slice(0, s))
        else:
            start = (s - a) // 2
            src_slices.append(slice(0, a))
            dst_slices.append(slice(start, start + a))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out
