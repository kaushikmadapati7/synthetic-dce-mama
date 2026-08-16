"""Training loop for the residual U-Net (pure regression, single-stage)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from ..models import ResidualUNet2D
from .utils import log_epoch, save_ckpt

log = logging.getLogger("mama_synth")


def train_unet(args, train_loader, test_loader, criterion, device):
    depth = args.n_upsamples
    factor = 2 ** depth
    if any(s % factor for s in args.spatial_size):
        raise ValueError(f"spatial_size {args.spatial_size} must be divisible by 2**n_upsamples ({factor})")
    rmode = getattr(args, "residual_mode", "residual")
    net = ResidualUNet2D(in_channels=1, out_channels=1, base_ch=args.base_ch, depth=depth,
                         residual=(rmode != "direct"), positive=(rmode == "positive")).to(device)
    log.info(f"ResidualUNet ({rmode}) params: {sum(p.numel() for p in net.parameters())/1e6:.1f}M")

    if getattr(args, "eval_only", False):
        ckpt = args.ckpt or str(Path(args.output_dir) / "checkpoints" / "unet_last.pt")
        net.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f"[eval-only] loaded {ckpt}")
        return net, (lambda cond, mask=None: net(cond))

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        net.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            cond = batch["cond"].to(device); target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            pred = net(cond)
            loss, parts = criterion(pred, target, mask, source=cond)
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in {"loss": float(loss.detach()), **parts}.items():
                agg[k] = agg.get(k, 0.0) + v
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, "unet", net, epoch, args.epochs, state_dict=True)

    return net, (lambda cond, mask=None: net(cond))
