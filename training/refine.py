"""Refiner GAN: sharpen a frozen base U-Net's output with a light adversarial stage.

Cascade design (stage 2 on top of `--model unet`):
  - The trained residual U-Net (loaded via --unet-ckpt) is FROZEN and produces a
    faithful-but-soft post-contrast estimate that already wins the ROI/Segmentation
    groups.
  - A small refiner (another residual U-Net) is conditioned on that estimate:
    `refined = unet_out + g(unet_out)`, zero-initialised so it STARTS as a no-op
    (identity = the U-Net output) and only learns to add high-frequency texture.
  - The discriminator's realism pressure is FOCUSED ON THE TUMOR ROI, not the whole
    slice. The tumor is ~0.5% of a 512^2 image, so a global discriminator spends its
    capacity policing background realism (already competitive) and barely touches the
    lesion — which is the exact region the model-based ROI/Classification/Segmentation
    metrics collapse on (the challenge nnU-Net returns EMPTY masks on our soft tumors).
    Instead we crop each sample's tumor bbox (mask available at TRAIN time only),
    resize to a fixed patch, and run a conditional discriminator on the [tumor_patch,
    pre_patch] pair. This concentrates 100% of the adversarial signal where the deficit
    is, closing the synthetic-vs-real tumor domain gap that blinds the segmenter. The
    reconstruction loss (criterion) still operates on the FULL image, anchoring global
    structure so the refiner can't drift off-anatomy. No mask is needed at inference —
    the discriminator is training-only; gen(cond) = refiner(base_unet(cond)).

Targets the ROI / Segmentation / Classification groups WITHOUT giving up the global
wins. Use a SMALL --adv-weight (~0.05-0.1): this is a sharpener, not a generator.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..models import ResidualUNet2D, Discriminator2D, d_hinge_loss, g_hinge_loss
from .utils import log_epoch, save_ckpt

log = logging.getLogger("mama_synth")


def _roi_bboxes(mask, pad_frac=0.25, min_size=16):
    """Per-sample tumor bounding box from a (B,1,H,W) mask, expanded by ``pad_frac``
    on each side for context. Returns a list of (y0,y1,x0,x1) or None where the mask
    is empty. A ``min_size`` floor keeps tiny lesions large enough to survive the
    discriminator's stride-2 stack."""
    boxes = []
    for b in range(mask.shape[0]):
        m = mask[b, 0]
        ys, xs = torch.where(m > 0.5)
        if ys.numel() == 0:
            boxes.append(None)
            continue
        H, W = m.shape
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
        y0, y1 = max(0, y0 - py), min(H, y1 + py)
        x0, x1 = max(0, x0 - px), min(W, x1 + px)
        if y1 - y0 < min_size:
            c = (y0 + y1) // 2
            y0 = max(0, min(c - min_size // 2, H - min_size)); y1 = y0 + min_size
        if x1 - x0 < min_size:
            c = (x0 + x1) // 2
            x0 = max(0, min(c - min_size // 2, W - min_size)); x1 = x0 + min_size
        boxes.append((y0, y1, x0, x1))
    return boxes


def _crop_resize(img, boxes, patch):
    """Crop each sample's bbox from ``img`` (B,1,H,W) and bilinearly resize to
    (patch,patch). Returns (K,1,patch,patch) over the samples with a non-empty box
    (in order), or None if every box is empty. Differentiable in ``img``."""
    crops = [F.interpolate(img[b:b + 1, :, y0:y1, x0:x1], size=(patch, patch),
                           mode="bilinear", align_corners=False)
             for b, box in enumerate(boxes) if box is not None
             for (y0, y1, x0, x1) in [box]]
    return torch.cat(crops, 0) if crops else None


class _PostChannel(torch.nn.Module):
    """Wrap a multitask base so it exposes only the post-contrast output (channel 0);
    the aux mask head(s) are dropped — the refiner sharpens the synthesized image."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        return self.base(x)[:, :1]


def _load_base_unet(args, device):
    if not args.unet_ckpt:
        raise ValueError("--model refine requires --unet-ckpt (a trained residual U-Net checkpoint)")
    base_oc = int(getattr(args, "base_out_channels", 1))
    base = ResidualUNet2D(in_channels=1, out_channels=base_oc, base_ch=args.base_ch,
                          depth=args.n_upsamples).to(device)
    base.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    log.info(f"loaded frozen base U-Net: {args.unet_ckpt} (out_channels={base_oc})")
    # a multitask base emits (post, aux mask); the refiner only consumes the post,
    # so slice channel 0 here and the rest of the cascade stays single-channel.
    return base if base_oc == 1 else _PostChannel(base).to(device).eval()


def train_refine(args, train_loader, test_loader, criterion, device):
    n_up = args.n_upsamples
    if any(s % (2 ** n_up) for s in args.spatial_size):
        raise ValueError(f"spatial_size {args.spatial_size} must be divisible by 2**n_upsamples")
    base = _load_base_unet(args, device)
    refiner = ResidualUNet2D(in_channels=1, out_channels=1, base_ch=args.base_ch, depth=n_up).to(device)
    # discriminator sees fixed-size ROI patches (not the full slice), so size its
    # depth from the patch, not spatial_size: log2(patch)-2 stride-2 layers leave a
    # small feature grid before the global pool.
    patch = int(args.roi_patch)
    pad_frac = float(args.roi_pad_frac)
    n_down = max(2, int(math.log2(patch)) - 2)
    disc = Discriminator2D(in_channels=1, cond_channels=1, base_ch=args.base_ch,
                           n_downsamples=n_down).to(device)
    log.info(f"refiner params: {sum(p.numel() for p in refiner.parameters())/1e6:.1f}M  "
             f"D: {sum(p.numel() for p in disc.parameters())/1e6:.1f}M  "
             f"ROI patch={patch} pad_frac={pad_frac}")

    def gen(cond, mask=None):
        return refiner(base(cond))

    if getattr(args, "eval_only", False):
        ckpt = args.ckpt or str(Path(args.output_dir) / "checkpoints" / "refine_last.pt")
        refiner.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f"[eval-only] loaded {ckpt}")
        return refiner, gen

    opt_g = torch.optim.Adam(refiner.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    for epoch in range(args.epochs):
        refiner.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            cond = batch["cond"].to(device); target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            with torch.no_grad():
                base_out = base(cond)
            refined = refiner(base_out)

            # tumor-ROI crops (mask available at train time); shared bboxes for the
            # real/fake/cond patches so the discriminator compares aligned regions.
            boxes = _roi_bboxes(mask, pad_frac=pad_frac)
            real_p = _crop_resize(target, boxes, patch)
            if real_p is not None:
                cond_p = _crop_resize(cond, boxes, patch)
                # discriminator: real (tumor_patch, pre_patch) vs fake (refined_patch, pre_patch)
                d_loss = d_hinge_loss(disc(real_p, cond_vol=cond_p),
                                      disc(_crop_resize(refined.detach(), boxes, patch),
                                           cond_vol=cond_p))
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                adv = g_hinge_loss(disc(_crop_resize(refined, boxes, patch), cond_vol=cond_p))
            else:  # no tumor in this batch -> recon-only step (no adversarial signal)
                d_loss = refined.new_zeros(())
                adv = refined.new_zeros(())

            # refiner: full-image reconstruction (anchor) + light ROI-adversarial (sharpen)
            rec, parts = criterion(refined, target, mask, source=cond)
            g_loss = rec + args.adv_weight * adv
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            for k, v in {"d": float(d_loss.detach()), "g": float(g_loss.detach()),
                         "adv": float(adv.detach()), **parts}.items():
                agg[k] = agg.get(k, 0.0) + v
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, "refine", refiner, epoch, args.epochs, state_dict=True)

    return refiner, gen
