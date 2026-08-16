"""Shared two-stage latent-diffusion training (VAE first stage + denoiser).

Both the DDPM and flow-matching trainers wrap `train_ldm` with a different
`flow` flag; the only differences are the model class and the sampler.

2D translation of the 3D pipeline's training/_ldm_base.py: single-channel conditioning
(cond_channels=1), bilinear conditioning downsample, and the VAE decoder's final
activation is configurable (default identity for z-scored data).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from ..models import AutoencoderKL2D, LDM_DDPM, LDM_FlowMatching
from .utils import log_epoch, save_ckpt, downsample_cond

log = logging.getLogger("mama_synth")


def train_vae(args, train_loader, criterion, device):
    vae = AutoencoderKL2D(in_channels=1, out_channels=1,
                          latent_channels=args.latent_channels, base_ch=args.base_ch,
                          ch_mults=tuple(args.ch_mults), final_act=args.final_act).to(device)
    log.info(f"VAE params: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M")
    if args.vae_ckpt:
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
        log.info(f"loaded VAE checkpoint {args.vae_ckpt}")
    else:
        opt = torch.optim.Adam(vae.parameters(), lr=args.lr)
        for epoch in range(args.vae_epochs):
            vae.train(); t0 = time.time(); agg = {}
            for batch in train_loader:
                x = batch["target"].to(device)
                mask = batch["mask"].to(device)
                loss, parts = vae.loss(x, kl_weight=args.kl_weight, criterion=criterion, mask=mask)
                opt.zero_grad(); loss.backward(); opt.step()
                for k, v in {"vae": float(loss), **parts}.items():
                    agg[k] = agg.get(k, 0.0) + v
            log_epoch(epoch, args.vae_epochs, agg, len(train_loader), time.time() - t0, "VAE")
            save_ckpt(args, "vae", vae, epoch, args.vae_epochs, state_dict=True)

    # scaling factor: normalize latent std to ~1 (standard LDM practice)
    vae.eval()
    with torch.no_grad():
        zs = []
        for i, batch in enumerate(train_loader):
            zs.append(vae.encode(batch["target"].to(device)).sample())
            if i >= 4:
                break
        std = torch.cat(zs).std().item()
    vae.scaling_factor = 1.0 / (std + 1e-8)
    log.info(f"latent std={std:.4f} -> scaling_factor={vae.scaling_factor:.4f}")
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _build_ldm(args, vae, flow, device):
    unet_kwargs = dict(in_channels=args.latent_channels, out_channels=args.latent_channels,
                       cond_channels=1, base_ch=args.base_ch, ch_mults=tuple(args.unet_ch_mults))
    if flow:
        return LDM_FlowMatching(autoencoder=vae, unet_kwargs=unet_kwargs).to(device)
    return LDM_DDPM(autoencoder=vae, timesteps=args.timesteps, unet_kwargs=unet_kwargs).to(device)


def _latent_scale_and_grid(vae, train_loader, device):
    """Recompute the latent scaling factor + grid (not stored in state_dict)."""
    vae.eval()
    with torch.no_grad():
        zs = []
        for i, batch in enumerate(train_loader):
            zs.append(vae.encode(batch["target"].to(device)).sample())
            if i >= 4:
                break
        z = torch.cat(zs)
    vae.scaling_factor = 1.0 / (z.std().item() + 1e-8)
    return tuple(z.shape[2:])


def _make_gen(ldm, args, lat_spatial, device, flow):
    def gen(cond, mask=None):
        cond_ds = downsample_cond(cond, lat_spatial)
        shape = (cond.size(0), args.latent_channels, *lat_spatial)
        if flow:
            return ldm.sample(shape, device, steps=args.sample_steps, cond=cond_ds)
        return ldm.ddim_sample(shape, device, steps=args.sample_steps, cond=cond_ds)
    return gen


def _load_ldm_for_eval(args, train_loader, device, flow, name):
    vae = AutoencoderKL2D(in_channels=1, out_channels=1, latent_channels=args.latent_channels,
                          base_ch=args.base_ch, ch_mults=tuple(args.ch_mults),
                          final_act=args.final_act).to(device)
    ldm = _build_ldm(args, vae, flow, device)
    ckpt = args.ckpt or str(Path(args.output_dir) / "checkpoints" / f"{name}_last.pt")
    ldm.load_state_dict(torch.load(ckpt, map_location=device))  # restores VAE + U-Net weights
    log.info(f"[eval-only] loaded {ckpt}")
    lat_spatial = _latent_scale_and_grid(vae, train_loader, device)
    log.info(f"[eval-only] scaling_factor={vae.scaling_factor:.4f}  latent grid {args.latent_channels}x{lat_spatial}")
    for p in vae.parameters():
        p.requires_grad_(False)
    return ldm, _make_gen(ldm, args, lat_spatial, device, flow)


def _build_bridge_ldm(args, vae, device):
    # condition the velocity net on the pre latent (4ch) throughout the trajectory
    unet_kwargs = dict(in_channels=args.latent_channels, out_channels=args.latent_channels,
                       cond_channels=args.latent_channels, base_ch=args.base_ch,
                       ch_mults=tuple(args.unet_ch_mults))
    return LDM_FlowMatching(autoencoder=vae, unet_kwargs=unet_kwargs).to(device)


def _make_bridge_gen(ldm, args, device):
    def gen(cond_img, mask=None):
        z_pre = ldm.encode(cond_img)
        return ldm.sample(z_pre.shape, device, steps=args.sample_steps, cond=z_pre, x1=z_pre)
    return gen


def train_ldm_bridge(args, train_loader, test_loader, criterion, device):
    """Paired flow matching (I2SB-style): transport the pre-contrast latent -> the
    post-contrast latent, instead of noise -> post.

    The flow's t=1 endpoint is ``z_pre = encode(pre)`` and t=0 is ``z_post``, so the
    velocity learns the small, localized pre->post displacement and sampling starts
    from the pre (inheriting the strong identity baseline, like the residual U-Net).
    The U-Net is also conditioned on the pre latent. The VAE is the same first stage
    (trained on targets); encoding the pre through it is mild in-modality OOD.
    """
    name = "ldm_bridge"
    if getattr(args, "eval_only", False):
        vae = AutoencoderKL2D(in_channels=1, out_channels=1, latent_channels=args.latent_channels,
                              base_ch=args.base_ch, ch_mults=tuple(args.ch_mults),
                              final_act=args.final_act).to(device)
        ldm = _build_bridge_ldm(args, vae, device)
        ckpt = args.ckpt or str(Path(args.output_dir) / "checkpoints" / f"{name}_last.pt")
        ldm.load_state_dict(torch.load(ckpt, map_location=device))  # restores VAE + U-Net
        log.info(f"[eval-only] loaded {ckpt}")
        _latent_scale_and_grid(vae, train_loader, device)  # recompute scaling_factor
        for p in vae.parameters():
            p.requires_grad_(False)
        return ldm, _make_bridge_gen(ldm, args, device)

    vae = train_vae(args, train_loader, criterion, device)
    ldm = _build_bridge_ldm(args, vae, device)
    log.info(f"UNet params: {sum(p.numel() for p in ldm.unet.parameters())/1e6:.1f}M")

    opt = torch.optim.Adam(ldm.unet.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        ldm.unet.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            z_post = ldm.encode(batch["target"].to(device))   # encode() is @no_grad + scaled
            z_pre = ldm.encode(batch["cond"].to(device))
            loss = ldm.loss(z_post, cond=z_pre, x1=z_pre)
            opt.zero_grad(); loss.backward(); opt.step()
            agg["diff"] = agg.get("diff", 0.0) + float(loss)
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, name, ldm, epoch, args.epochs, state_dict=True)

    return ldm, _make_bridge_gen(ldm, args, device)


def train_ldm(args, train_loader, test_loader, criterion, device, flow: bool):
    name = "ldm_flow" if flow else "ldm_ddpm"
    if getattr(args, "eval_only", False):
        return _load_ldm_for_eval(args, train_loader, device, flow, name)

    vae = train_vae(args, train_loader, criterion, device)

    # infer latent spatial size from one encode
    with torch.no_grad():
        z0 = vae.encode(next(iter(train_loader))["target"].to(device)).sample()
    lat_spatial = tuple(z0.shape[2:])
    log.info(f"latent grid: {z0.shape[1]}x{lat_spatial}")

    ldm = _build_ldm(args, vae, flow, device)
    log.info(f"UNet params: {sum(p.numel() for p in ldm.unet.parameters())/1e6:.1f}M")

    opt = torch.optim.Adam(ldm.unet.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        ldm.unet.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            cond = batch["cond"].to(device)
            with torch.no_grad():
                z0 = ldm.encode(batch["target"].to(device))
            cond_ds = downsample_cond(cond, z0.shape[2:])
            loss = ldm.loss(z0, cond=cond_ds)
            opt.zero_grad(); loss.backward(); opt.step()
            agg["diff"] = agg.get("diff", 0.0) + float(loss)
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, name, ldm, epoch, args.epochs, state_dict=True)

    return ldm, _make_gen(ldm, args, lat_spatial, device, flow)
