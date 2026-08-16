"""Pretrain a 2D tumor segmenter on real post-contrast slices.

Why this exists: the challenge's nnU-Net returns EMPTY masks on our soft synthetic
tumors, which zeroes the Segmentation and Classification groups (half the score). This
segmenter is trained on (real post-contrast -> tumor mask), then FROZEN and plugged into
the synthesizer's loss via --seg-ckpt/--seg-weight. Its gradients push the synthetic
image to be *segmentable as the GT mask* — i.e. it optimizes the failing downstream
metric directly, instead of a pixel proxy (L1/MSE/SSIM/LPIPS) that the segmenter ignores.

It is a plain (non-residual) ResidualUNet2D producing a 1-channel logit map; trained with
soft-Dice + BCE. Run it once on the preprocessed tree, then reuse the checkpoint across
synthesizer experiments. This is a *pretraining* run: the main pipeline's image-synthesis
eval is meaningless here (gen is identity), so launch with --no-fid and ignore TEST metrics.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from ..models import ResidualUNet2D
from ..loss.loss import dice_bce_loss
from .utils import log_epoch, save_ckpt

log = logging.getLogger("mama_synth")


def build_segmenter(args, device):
    """Plain 2D U-Net critic: post-contrast slice -> tumor logit map."""
    return ResidualUNet2D(in_channels=1, out_channels=1, base_ch=args.seg_base_ch,
                          depth=args.n_upsamples, residual=False,
                          zero_init_out=False).to(device)


def train_segmenter(args, train_loader, test_loader, criterion, device):
    net = build_segmenter(args, device)
    log.info(f"Segmenter params: {sum(p.numel() for p in net.parameters())/1e6:.1f}M "
             f"(base_ch={args.seg_base_ch})")

    identity = lambda cond, mask=None: cond   # pretraining run: no image synthesis to eval

    if getattr(args, "eval_only", False):
        ckpt = args.ckpt or str(Path(args.output_dir) / "checkpoints" / "segmenter_last.pt")
        net.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f"[eval-only] loaded {ckpt}")
        return net, identity

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        net.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            target = batch["target"].to(device)                 # train on the REAL post
            mask = (batch["mask"] > 0.5).float().to(device)
            logits = net(target)
            loss = dice_bce_loss(logits, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():                               # monitor: soft Dice
                prob = torch.sigmoid(logits)
                dice = (2 * (prob * mask).sum() / (prob.sum() + mask.sum() + 1.0)).item()
            for k, v in {"loss": float(loss.detach()), "dice": dice}.items():
                agg[k] = agg.get(k, 0.0) + v
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, "segmenter", net, epoch, args.epochs, state_dict=True)

    log.info(f"segmenter pretraining done -> {args.output_dir}/checkpoints/segmenter_last.pt "
             f"(pass it as --seg-ckpt with --seg-weight>0 on a synthesizer run)")
    return net, identity
