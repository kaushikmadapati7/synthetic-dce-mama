"""2D residual U-Net generator for pre->post DCE translation.

Predicts the post-contrast slice as the pre-contrast plus a learned enhancement
residual:  ``post_pred = pre + f(pre)``.

Why this architecture (vs the GAN / latent-diffusion models):
  - Breast post-contrast is the pre-contrast plus *localized* enhancement, so the
    identity map (copy pre->post) is already a very strong baseline. The GAN/LDM
    regenerate the whole image from a compressed bottleneck/latent and lose the
    anatomy that copying keeps for free, scoring *below* identity on MSE/LPIPS/ROI.
  - Full-resolution skip connections pass the anatomy straight through, and the
    output conv is zero-initialised so training *starts at the identity baseline*
    and only has to learn the enhancement residual — a much better-posed problem.

Pure regression (no adversarial / no sampling): trained with the shared CustomLoss
(L1 + SSIM + perceptual, ROI-weighted). Output is in the z-scored space (identity
activation), like the other models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, ch), ch)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), _norm(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), _norm(out_ch), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Bilinear upsample + conv (avoids transpose-conv checkerboard), then concat skip."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([skip, x], dim=1))


class ResidualUNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_ch: int = 32, depth: int = 4, residual: bool = True,
                 zero_init_out: bool = True, positive: bool = False):
        super().__init__()
        self.residual = residual
        # positive-enhancement ansatz: post = pre + softplus(f(pre)). DCE uptake is
        # physically non-negative tissue accumulation, so constraining the residual to
        # be >= 0 bakes that prior into the architecture (paper §3.6). Only meaningful
        # when residual=True.
        self.positive = positive
        chs = [base_ch * (2 ** i) for i in range(depth + 1)]
        self.in_conv = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(Down(chs[i], chs[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(Up(chs[i], chs[i - 1]) for i in range(depth, 0, -1))
        self.out_conv = nn.Conv2d(chs[0], out_channels, 1)
        # zero-init the output so the net starts as the identity map (residual ~ 0).
        # Disabled when reused as a plain (non-residual) net, e.g. the segmentation
        # critic, where an all-zero init would just be a slower start.
        if zero_init_out:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        h = self.in_conv(x)
        skips = [h]
        for down in self.downs:
            h = down(h)
            skips.append(h)
        for i, up in enumerate(self.ups):
            h = up(h, skips[-(i + 2)])
        out = self.out_conv(h)
        if self.residual:
            # residual on the FIRST (post) channel only; any extra channels (e.g. an
            # auxiliary mask head) are raw outputs.
            delta = F.softplus(out[:, :1]) if self.positive else out[:, :1]
            post = delta + x[:, :1]
            out = post if out.shape[1] == 1 else torch.cat([post, out[:, 1:]], dim=1)
        return out
