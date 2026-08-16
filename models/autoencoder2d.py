"""2D VAE / KL-autoencoder that compresses slices into a latent grid.

Used as the first stage for both latent diffusion models. Train this first,
then run diffusion/flow matching in its latent space.

2D translation of the 3D pipeline's models/autoencoder3d.py. Key adaptation: the decoder's
final activation is configurable. tier1 used tanh (data in [-1, 1]); MAMA-SYNTH data is
dataset-level z-scored (~zero-mean/unit-var, unbounded), so the default here is identity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResBlock2D, AttentionBlock2D, Downsample2D, Upsample2D


def _final_activation(name: str):
    if name == "identity":
        return nn.Identity()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown final_act '{name}' (expected 'identity' or 'tanh')")


class DiagonalGaussian:
    """Latent posterior N(mean, var) with reparameterization + KL."""

    def __init__(self, params: torch.Tensor):
        self.mean, self.logvar = params.chunk(2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.mean(self.mean ** 2 + self.std ** 2 - 1.0 - self.logvar)


class Encoder2D(nn.Module):
    def __init__(self, in_ch, base_ch, ch_mults, latent_ch, num_res, attn_at):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        chans = [base_ch * m for m in ch_mults]
        blocks = []
        ch = base_ch
        for i, out_ch in enumerate(chans):
            for _ in range(num_res):
                blocks.append(ResBlock2D(ch, out_ch))
                ch = out_ch
                if i in attn_at:
                    blocks.append(AttentionBlock2D(ch))
            if i < len(chans) - 1:
                blocks.append(Downsample2D(ch))
        self.blocks = nn.ModuleList(blocks)
        self.mid = nn.ModuleList([ResBlock2D(ch, ch), AttentionBlock2D(ch), ResBlock2D(ch, ch)])
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, 2 * latent_ch, 3, padding=1)  # mean + logvar

    def forward(self, x):
        h = self.conv_in(x)
        for b in self.blocks:
            h = b(h)
        for b in self.mid:
            h = b(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder2D(nn.Module):
    def __init__(self, out_ch, base_ch, ch_mults, latent_ch, num_res, attn_at,
                 final_act: str = "identity"):
        super().__init__()
        chans = [base_ch * m for m in ch_mults]
        ch = chans[-1]
        self.conv_in = nn.Conv2d(latent_ch, ch, 3, padding=1)
        self.mid = nn.ModuleList([ResBlock2D(ch, ch), AttentionBlock2D(ch), ResBlock2D(ch, ch)])
        blocks = []
        n = len(chans)
        for i, out_c in enumerate(reversed(chans)):
            for _ in range(num_res):
                blocks.append(ResBlock2D(ch, out_c))
                ch = out_c
                if (n - 1 - i) in attn_at:
                    blocks.append(AttentionBlock2D(ch))
            if i < n - 1:
                blocks.append(Upsample2D(ch))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)
        self.final_act = _final_activation(final_act)

    def forward(self, z):
        h = self.conv_in(z)
        for b in self.mid:
            h = b(h)
        for b in self.blocks:
            h = b(h)
        return self.final_act(self.conv_out(F.silu(self.norm_out(h))))


class AutoencoderKL2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        base_ch: int = 32,
        ch_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple | None = None,  # ch_mults stage indices that get attention
        scaling_factor: float = 1.0,
        final_act: str = "identity",
    ):
        super().__init__()
        # Attend at the bottleneck (deepest) stage by default. A hardcoded index like
        # (2,) is a FIXED resolution (input/2**2) regardless of depth — at 512^2 that's
        # a 128x128 plane whose O(N^2) self-attention needs ~32 GiB. Tying it to
        # len(ch_mults)-1 keeps attention on the smallest grid as the VAE gets deeper.
        if attn_resolutions is None:
            attn_resolutions = (len(ch_mults) - 1,)
        self.encoder = Encoder2D(in_channels, base_ch, ch_mults, latent_channels,
                                 num_res_blocks, attn_resolutions)
        self.decoder = Decoder2D(out_channels, base_ch, ch_mults, latent_channels,
                                 num_res_blocks, attn_resolutions, final_act)
        self.scaling_factor = scaling_factor

    def encode(self, x) -> DiagonalGaussian:
        return DiagonalGaussian(self.encoder(x))

    def decode(self, z):
        return self.decoder(z / self.scaling_factor)

    def forward(self, x):
        posterior = self.encode(x)
        z = posterior.sample()
        return self.decoder(z), posterior

    def loss(self, x, kl_weight: float = 1e-6, criterion=None, mask=None):
        """Reconstruction + KL.

        criterion: optional callable(pred, target, mask=None) -> (loss, components),
        e.g. loss.CustomLoss (perceptual + SSIM + L1, with optional ROI weighting).
        Falls back to plain L1. ``mask`` drives ROI emphasis in the criterion — for
        the LDM pipeline this VAE recon term is where lesion image fidelity is set,
        since the diffusion loss itself lives in latent space.
        """
        recon, posterior = self(x)
        if criterion is not None:
            rec, parts = criterion(recon, x, mask)
        else:
            rec = F.l1_loss(recon, x)
            parts = {"recon": rec.item()}
        kl = posterior.kl()
        return rec + kl_weight * kl, {**parts, "kl": kl.item()}
