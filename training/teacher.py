"""Privileged 'teacher' U-Net — a DIAGNOSTIC, not a submission model.

It takes [pre-contrast, tumor-mask] as a 2-channel input (residual still on the pre
channel). The mask is PRIVILEGED: available at training but forbidden at test, so this
model can NEVER be submitted. Its sole purpose is the go/no-go test for the
privileged-information / distillation idea:

  Train it exactly like the pre-only student (same augment/lpips/roi-weight), score it
  on the same 8 challenge metrics, and compare. If the mask-teacher dramatically beats
  the student on the tumor metrics (ssim_tumor, DICE, auroc_tumor_roi, FRD), privileged
  info has real headroom -> distillation (and eventually T2/DWI/ADC) is worth the lift.
  If it barely beats the student, the whole privileged-info direction is low-value.

(The mask tests the *localization* benefit specifically; an intermediate-DCE-phase
teacher would be the closer proxy for the *functional* info T2/DWI/ADC carry.)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from ..models import ResidualUNet2D
from .utils import log_epoch, save_ckpt

log = logging.getLogger("mama_synth")


def train_teacher(args, train_loader, test_loader, criterion, device):
    depth = args.n_upsamples
    if any(s % (2 ** depth) for s in args.spatial_size):
        raise ValueError(f"spatial_size {args.spatial_size} must be divisible by 2**n_upsamples")
    net = ResidualUNet2D(in_channels=2, out_channels=1, base_ch=args.base_ch, depth=depth).to(device)
    log.info(f"PRIVILEGED teacher ([pre,mask] -> post, NOT submittable) params: "
             f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M")

    def gen(cond, mask=None):
        if mask is None:                       # at inference with no mask it degrades to pre-only-ish
            mask = torch.zeros_like(cond)
        return net(torch.cat([cond, mask], dim=1))

    if getattr(args, "eval_only", False):
        ckpt = args.ckpt or str(Path(args.output_dir) / "checkpoints" / "teacher_last.pt")
        net.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f"[eval-only] loaded {ckpt}")
        return net, gen

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        net.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            cond = batch["cond"].to(device); target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            pred = net(torch.cat([cond, mask], dim=1))
            loss, parts = criterion(pred, target, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in {"loss": float(loss.detach()), **parts}.items():
                agg[k] = agg.get(k, 0.0) + v
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, "teacher", net, epoch, args.epochs, state_dict=True)

    return net, gen
