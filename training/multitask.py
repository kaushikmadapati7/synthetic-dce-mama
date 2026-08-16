"""Multi-task student: predict [post-contrast, tumor-mask] from the pre-contrast.

The mask is a training-only AUXILIARY target — it's dropped at inference, so the model
stays a deployable pre-only synthesizer (`gen(cond)` returns only the post channel).
Forcing the student to also segment the tumor makes it *localize* the lesion, which is
the localization headroom the privileged mask-teacher revealed (DICE +68%) — captured
here without needing the mask at test.

Loss = reconstruction (CustomLoss on the post) + aux_mask * (BCE + soft-Dice) on the
mask head. The residual U-Net's residual is applied to the post channel only.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..models import ResidualUNet2D
from .utils import log_epoch, save_ckpt

log = logging.getLogger("mama_synth")


def _seg_loss(logit, mask):
    """BCE + soft-Dice on the auxiliary mask head (logit vs binary GT mask)."""
    bce = F.binary_cross_entropy_with_logits(logit, mask)
    p = torch.sigmoid(logit)
    dims = (1, 2, 3)
    inter = (p * mask).sum(dims)
    dice = 1.0 - (2 * inter + 1.0) / (p.sum(dims) + mask.sum(dims) + 1.0)
    return bce + dice.mean()


def train_multitask(args, train_loader, test_loader, criterion, device):
    depth = args.n_upsamples
    if any(s % (2 ** depth) for s in args.spatial_size):
        raise ValueError(f"spatial_size {args.spatial_size} must be divisible by 2**n_upsamples")
    rmode = getattr(args, "residual_mode", "residual")
    net = ResidualUNet2D(in_channels=1, out_channels=2, base_ch=args.base_ch, depth=depth,
                         residual=(rmode != "direct"), positive=(rmode == "positive")).to(device)
    aux_w = getattr(args, "aux_mask", 1.0)
    log.info(f"Multi-task student ([pre] -> [post, mask], aux_w={aux_w}) params: "
             f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M")

    def gen(cond, mask=None):
        return net(cond)[:, :1]   # deployable: post channel only (mask head dropped)

    if getattr(args, "eval_only", False):
        ckpt = args.ckpt or str(Path(args.output_dir) / "checkpoints" / "multitask_last.pt")
        net.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f"[eval-only] loaded {ckpt}")
        return net, gen

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Exponential moving average of the weights. Averaging out the final-trajectory
    # noise gives a smoother, better-generalizing model at no loss/architecture cost;
    # we SAVE the EMA copy so eval/inference deploys the averaged weights. Unlike every
    # texture term we tried, EMA adds no high-frequency pressure, so it can't re-break FRD.
    use_ema = getattr(args, "ema", False)
    ema_decay = float(getattr(args, "ema_decay", 0.9999))
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()} if use_ema else None
    if use_ema:
        log.info(f"EMA enabled (decay={ema_decay}); EMA weights are saved as the checkpoint")

    for epoch in range(args.epochs):
        net.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            cond = batch["cond"].to(device); target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            out = net(cond)
            post, mlogit = out[:, :1], out[:, 1:2]
            rec, parts = criterion(post, target, mask, source=cond)
            seg = _seg_loss(mlogit, mask)
            loss = rec + aux_w * seg
            opt.zero_grad(); loss.backward(); opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in net.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                        else:
                            ema[k].copy_(v)   # int buffers (e.g. counters): just track latest
            for k, v in {"loss": float(loss.detach()), "seg": float(seg.detach()), **parts}.items():
                agg[k] = agg.get(k, 0.0) + v
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        # save the EMA weights (swap in, save, restore the live weights so training continues)
        if ema is not None:
            live = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(ema)
            save_ckpt(args, "multitask", net, epoch, args.epochs, state_dict=True)
            net.load_state_dict(live)
        else:
            save_ckpt(args, "multitask", net, epoch, args.epochs, state_dict=True)

    return net, gen
