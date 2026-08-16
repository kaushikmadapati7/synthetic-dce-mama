"""MAMA-SYNTH datasets: pre-contrast slice -> peak post-contrast slice.

Two loaders:

  MamaSynthDataset -> reads the official preprocessing output tree
                      (mama-research/mama-synth preprocess.py):
                          <root>/mha/input/<case>.mha          (pre-contrast)
                          <root>/mha/ground_truth/<case>.mha   (peak post-contrast)
                          <root>/mha/mask/<case>.mha           (tumor ROI)

  DummyMamaDataset -> synthetic Gaussian-blob slices, so the whole pipeline can be
                      smoke-tested end-to-end with no data downloaded.

Both return:
    {
      "cond":   FloatTensor (1, H, W)  # pre-contrast slice
      "target": FloatTensor (1, H, W)  # peak post-contrast slice
      "mask":   FloatTensor (1, H, W)  # tumor ROI (zeros if unavailable)
      "id":     str
    }

Note the conditioning is single-channel here (pre-contrast only), vs the 3-channel
(T2w, DWI, ADC) stack in the 3D pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (PreprocessConfig, Stats, load_mha, zscore_normalize,
                            center_crop_pad_2d)


def _stack_sample(cond, target, mask, case_id, norm=None) -> dict:
    d = {
        "cond": torch.from_numpy(cond[None]).float(),
        "target": torch.from_numpy(target[None]).float(),
        "mask": torch.from_numpy(mask[None]).float(),
        "id": case_id,
    }
    if norm is not None:  # per-image normalization stats, for exact inversion in eval
        d["norm_mean"] = torch.tensor(float(norm[0]), dtype=torch.float32)
        d["norm_std"] = torch.tensor(float(norm[1]), dtype=torch.float32)
    return d


def _per_image_norm(cond, target):
    """Instance-normalize by the cond's own mean/std (available at test). Returns the
    normalized (cond, target) plus (mean, std) so the model output can be inverted
    EXACTLY back to the global z-score space: pred_global = pred_norm * std + mean."""
    m = float(cond.mean())
    s = float(cond.std()) + 1e-6
    return (cond - m) / s, (target - m) / s, (m, s)


class MamaSynthDataset(Dataset):
    """Loads the official `.mha` preprocessing tree and applies z-score + crop/pad."""

    def __init__(self, root, stats: Stats, cfg: PreprocessConfig | None = None,
                 require_mask=False, augment=False, aug_deg=10.0, per_image_norm=False,
                 scanner_augment=False, scan_bias=0.3, scan_contrast=0.2, scan_noise=0.05):
        self.root = Path(root)
        self.cfg = cfg or PreprocessConfig()
        self.stats = stats
        self.augment = augment       # apply random flip/rotation (TRAIN ONLY)
        self.aug_deg = aug_deg       # max abs rotation in degrees
        self.per_image_norm = per_image_norm   # instance-normalize (scanner-robust)
        self.scanner_augment = scanner_augment # simulate cross-scanner appearance (TRAIN ONLY)
        self.scan_bias = scan_bias             # max bias-field deviation from 1.0
        self.scan_contrast = scan_contrast     # max global contrast-gain deviation from 1.0
        self.scan_noise = scan_noise           # max additive-noise std (per image)
        in_dir = self.root / "mha" / "input"
        self.gt_dir = self.root / "mha" / "ground_truth"
        self.mask_dir = self.root / "mha" / "mask"
        self.ids = []
        for p in sorted(in_dir.glob("*.mha")):
            if not (self.gt_dir / p.name).exists():
                continue
            if require_mask and not (self.mask_dir / p.name).exists():
                continue
            self.ids.append(p.name)

    def __len__(self):
        return len(self.ids)

    def _prep(self, arr):
        arr = zscore_normalize(arr, self.stats.mean, self.stats.std)
        if self.cfg.spatial_size is not None:
            arr = center_crop_pad_2d(arr, self.cfg.spatial_size, self.cfg.pad_value)
        return arr

    def _augment(self, cond, target, mask):
        """Apply the SAME random spatial transform to all three (keeps them aligned).
        Random params come from torch's RNG, which DataLoader reseeds per worker
        (numpy's global RNG is not, which would duplicate augmentations across workers).
        """
        # horizontal flip (breast is bilateral -> safe; no vertical flip: chest wall side)
        if torch.rand(1).item() < 0.5:
            cond = np.ascontiguousarray(cond[:, ::-1])
            target = np.ascontiguousarray(target[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        # small rotation (bilinear for images, nearest for mask)
        if torch.rand(1).item() < 0.5 and self.aug_deg > 0:
            from scipy.ndimage import rotate
            ang = float((torch.rand(1).item() * 2 - 1) * self.aug_deg)
            cond = rotate(cond, ang, reshape=False, order=1, mode="constant", cval=0.0)
            target = rotate(target, ang, reshape=False, order=1, mode="constant", cval=0.0)
            mask = rotate(mask, ang, reshape=False, order=0, mode="constant", cval=0.0)
            mask = (mask > 0.5).astype(np.float32)
        return cond, target, mask

    def _scanner_augment(self, cond, target):
        """Simulate cross-scanner appearance differences (Test = unseen Siemens 3T / GE 1.5T).
        Applied in the already-dataset-z-scored intensity space. The bias field and contrast
        gain are SHARED by pre & post — one acquisition session renders both with the same
        coil/B1 inhomogeneity and contrast profile, so the pre->post enhancement relationship
        the model learns is preserved; only thermal noise is independent per acquisition.
        The mask is intensity-invariant and untouched. torch RNG -> reseeded per worker."""
        h, w = cond.shape
        # smooth multiplicative bias field (coil/B1 inhomogeneity), shared pre & post
        if torch.rand(1).item() < 0.5 and self.scan_bias > 0:
            ctrl = torch.randn(1, 1, 4, 4)
            field = torch.nn.functional.interpolate(
                ctrl, size=(h, w), mode="bicubic", align_corners=True)[0, 0].numpy()
            field = field / (np.abs(field).max() + 1e-6)
            field = (1.0 + self.scan_bias * field).astype(np.float32)
            cond = cond * field
            target = target * field
        # global contrast (gain) jitter, shared pre & post
        if torch.rand(1).item() < 0.5 and self.scan_contrast > 0:
            g = 1.0 + float((torch.rand(1).item() * 2 - 1) * self.scan_contrast)
            cond = cond * g
            target = target * g
        # additive thermal noise (SNR / field-strength differences), independent per image
        if torch.rand(1).item() < 0.5 and self.scan_noise > 0:
            s = float(torch.rand(1).item() * self.scan_noise)
            cond = cond + torch.randn(h, w).numpy().astype(np.float32) * s
            target = target + torch.randn(h, w).numpy().astype(np.float32) * s
        return cond.astype(np.float32), target.astype(np.float32)

    def __getitem__(self, i):
        name = self.ids[i]
        cond = self._prep(load_mha(self.root / "mha" / "input" / name))
        target = self._prep(load_mha(self.gt_dir / name))
        mask_path = self.mask_dir / name
        if mask_path.exists():
            mask = load_mha(mask_path)
            if self.cfg.spatial_size is not None:
                mask = center_crop_pad_2d(mask, self.cfg.spatial_size, 0.0)
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros(cond.shape, dtype=np.float32)
        if self.augment:
            cond, target, mask = self._augment(cond, target, mask)
        if self.scanner_augment:
            cond, target = self._scanner_augment(cond, target)
        norm = None
        if self.per_image_norm:
            cond, target, norm = _per_image_norm(cond, target)
        return _stack_sample(cond, target, mask, name[:-4], norm=norm)


class DummyMamaDataset(Dataset):
    """Synthetic slices for smoke testing. The 'post-contrast' target is the
    pre-contrast plus a bright Gaussian blob inside the (random) tumor mask, so the
    learning signal is non-trivial. Already z-scored-ish (mean 0, unit-ish std)."""

    def __init__(self, n: int = 16, cfg: PreprocessConfig | None = None, seed: int = 0,
                 per_image_norm=False):
        self.cfg = cfg or PreprocessConfig()
        self.per_image_norm = per_image_norm
        self.n = n
        self.size = self.cfg.spatial_size or (64, 64)
        self.rng = np.random.RandomState(seed)
        # precompute deterministic per-case parameters
        self.params = [(self.rng.randint(0, 1 << 30)) for _ in range(n)]

    def __len__(self):
        return self.n

    def _blob(self, h, w, cy, cx, r, rng):
        ys, xs = np.ogrid[:h, :w]
        d2 = (ys - cy) ** 2 + (xs - cx) ** 2
        return np.exp(-d2 / (2 * r ** 2)).astype(np.float32)

    def __getitem__(self, i):
        h, w = self.size
        rng = np.random.RandomState(self.params[i])
        cond = rng.randn(h, w).astype(np.float32) * 0.5
        cy, cx = rng.randint(h // 4, 3 * h // 4), rng.randint(w // 4, 3 * w // 4)
        r = rng.randint(min(h, w) // 12 + 1, min(h, w) // 6 + 2)
        blob = self._blob(h, w, cy, cx, r, rng)
        target = cond + 2.0 * blob               # contrast uptake in the lesion
        mask = (blob > 0.5).astype(np.float32)   # tumor ROI
        norm = None
        if self.per_image_norm:
            cond, target, norm = _per_image_norm(cond, target)
        return _stack_sample(cond, target, mask, f"dummy_{i:04d}", norm=norm)
