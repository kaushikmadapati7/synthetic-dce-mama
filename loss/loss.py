"""Custom image-space reconstruction loss for 2D slices.

    Loss = l1_weight * L1  +  ssim_weight * (1 - SSIM2D)  +  perceptual_weight * Perc2D

All three terms operate on a (pred, target) pair of image-space tensors shaped
(B, C, H, W).

2D translation of the 3D pipeline's loss/loss.py. Two adaptations:
  - SSIM `data_range` is configurable. tier1 used 2.0 (data in [-1, 1]); MAMA-SYNTH
    data is dataset-level z-scored (~unit std), so the default here is 4.0 (~±2 sigma).
  - The 3D MedicalNet perceptual backbone is replaced with a frozen 2D torchvision
    VGG16 feature extractor (ImageNet-pretrained), which is the standard 2D perceptual
    loss and needs no external weights beyond torchvision. (The challenge's own LPIPS
    metric is a separate, eval-side concern handled by their evaluate.py.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D SSIM
# ---------------------------------------------------------------------------
def _gaussian_window_2d(window_size: int, sigma: float, channels: int,
                        device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w.expand(channels, 1, window_size, window_size).contiguous()


def ssim2d(x: torch.Tensor, y: torch.Tensor, window_size: int = 7,
           sigma: float = 1.5, data_range: float = 4.0,
           return_map: bool = False) -> torch.Tensor:
    """2D SSIM. data_range defaults to 4.0 for z-scored inputs (~±2 sigma).

    Returns the scalar mean SSIM, or the per-pixel SSIM map (same spatial size as
    the input) when ``return_map`` — used to average SSIM inside an ROI mask.
    """
    c = x.shape[1]
    w = _gaussian_window_2d(window_size, sigma, c, x.device, x.dtype)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sig_x = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
    sig_y = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / \
               ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return ssim_map if return_map else ssim_map.mean()


# ---------------------------------------------------------------------------
# 2D VGG perceptual (feature extractor for the perceptual term)
# Replaces tier1's 3D MedicalNet backbone; uses torchvision ImageNet weights.
# ---------------------------------------------------------------------------
# VGG16 relu boundaries (feature-map indices just after relu1_2/relu2_2/relu3_3/relu4_3)
_VGG16_SLICES = (4, 9, 16, 23)

# ImageNet normalization statistics for the VGG input.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGPerceptual(nn.Module):
    """Frozen VGG16 perceptual loss over a few relu feature maps.

    Inputs are single-channel (B, 1, H, W); the channel is replicated to RGB and
    per-image min-max scaled to [0, 1] before ImageNet normalization, so the loss is
    robust to the unbounded z-scored intensity range.
    """

    def __init__(self, feature_layers=(0, 1, 2)):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        try:
            features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except Exception:  # offline / no weights cached -> random init, still differentiable
            features = vgg16(weights=None).features
        # split into the relu-bounded blocks we read features from
        blocks, prev = [], 0
        for end in _VGG16_SLICES:
            blocks.append(nn.Sequential(*[features[i] for i in range(prev, end)]))
            prev = end
        self.blocks = nn.ModuleList(blocks)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.feature_layers = feature_layers
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _prep(self, x):
        # collapse any extra channels to one, replicate to RGB, min-max to [0, 1]
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        x = x.repeat(1, 3, 1, 1)
        flat = x.flatten(2)
        lo = flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
        hi = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
        x = (x - lo) / (hi - lo + 1e-8)
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        hp, ht = self._prep(pred), self._prep(target)
        loss = pred.new_zeros(())
        for i, block in enumerate(self.blocks):
            hp = block(hp)
            with torch.no_grad():
                ht = block(ht)
            if i in self.feature_layers:
                loss = loss + F.l1_loss(hp, ht)
        return loss / max(1, len(self.feature_layers))


# ---------------------------------------------------------------------------
# LPIPS distance as a training loss (aligned with the challenge metric)
# ---------------------------------------------------------------------------
class LPIPSLoss(nn.Module):
    """AlexNet LPIPS as a differentiable loss, preprocessed to match the challenge's
    evaluate.py: clip to +/- clip_sigma then scale to [-1, 1], single channel -> RGB.
    Uses torchmetrics; the backbone is frozen (only the inputs receive gradients)."""

    def __init__(self, clip_sigma: float = 5.0, net: str = "alex"):
        super().__init__()
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=False)
        for p in self.metric.parameters():
            p.requires_grad_(False)
        self.metric.eval()
        self.clip_sigma = clip_sigma

    def _prep(self, x):
        x = x.clamp(-self.clip_sigma, self.clip_sigma) / self.clip_sigma  # -> [-1, 1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, pred, target):
        a, b = self._prep(pred), self._prep(target)
        net = getattr(self.metric, "net", None)  # call the net directly to avoid metric-state accrual
        if net is not None:
            return net(a, b).mean()
        return self.metric(a, b)


# ---------------------------------------------------------------------------
# Segmentation (Dice + BCE) — used both to pretrain the critic and as the
# segmentation-perceptual term that flows gradients back into the synthesizer.
# ---------------------------------------------------------------------------
def dice_bce_loss(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft-Dice + BCE between a segmenter's logits and a binary mask. Lower is better.
    Differentiable in ``logits`` (so when the segmenter is frozen, gradients reach the
    image that produced those logits)."""
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    num = 2.0 * (prob * mask).sum(dims) + eps
    den = prob.sum(dims) + mask.sum(dims) + eps
    dice = (num / den).mean()
    return bce + (1.0 - dice)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------
class CustomLoss(nn.Module):
    """Weighted perceptual + SSIM + L1 over 2D (pred, target) slices.

    Returns (total_loss, components_dict). Used directly as the Conditional GAN
    generator reconstruction term and as the VAE recon term.

    ROI emphasis: the breast tumor is a small fraction of the slice, so an
    unweighted loss is dominated by background. When a ``mask`` is passed to
    ``forward`` and ``roi_weight > 1``, the L1 term is reweighted so ROI pixels
    count ``roi_weight``x more, and an extra ROI-SSIM term is added. With no mask
    (or an empty one) the behaviour is identical to the unweighted loss. The
    tumor mask drives the challenge's ROI-SSIM / DICE / HD95 metrics, so this is
    where lesion fidelity is set.

    Background faithfulness: for a residual synthesizer (post = pre + f(pre)),
    over-enhancement shows up as a non-zero residual ``f(pre) = pred - source``
    *outside* the tumor — global brightening that wrecks the Image-group MSE and
    smears the segmenter (HD95). When ``bg_weight > 0`` and a ``source`` (the
    pre-contrast input) and tumor ``mask`` are supplied, ``forward`` adds
    ``bg_weight * mean(|pred - source|)`` over the non-mask pixels, hard-coding
    "stay at the pre except inside the tumor". Gated on a non-empty mask: with no
    mask we can't tell "no tumor" from "mask unavailable", so the term is skipped.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        roi_weight: float = 1.0,
        lpips_weight: float = 0.0,
        bg_weight: float = 0.0,
        mse_weight: float = 0.0,
        roi_ssim_weight: float = 0.0,
        seg_weight: float = 0.0,
        enh_weight: float = 0.0,
        grad_weight: float = 0.0,
        seg_critic: nn.Module | None = None,
        feature_layers=(0, 1, 2),
        ssim_window: int = 7,
        ssim_sigma: float = 1.5,
        data_range: float = 4.0,
    ):
        super().__init__()
        self.l1_w = l1_weight
        self.ssim_w = ssim_weight
        self.perc_w = perceptual_weight
        self.roi_w = roi_weight
        self.lpips_w = lpips_weight
        self.bg_w = bg_weight
        self.mse_w = mse_weight
        # ROI-SSIM is decoupled from the global SSIM weight so the structural ROI
        # term can be turned on WITHOUT re-enabling the global SSIM term the champion
        # recipe deliberately switched off.
        self.roi_ssim_w = roi_ssim_weight
        # Segmentation-perceptual: a FROZEN tumor segmenter (trained on real post) is
        # run on the prediction; the term pushes the synthetic tumor to be SEGMENTABLE
        # as the GT mask -> directly targets the Segmentation/Classification metrics the
        # challenge nnU-Net collapses on (empty masks). Gradients flow through the
        # frozen critic into pred; its own weights never update.
        self.seg_w = seg_weight
        # Enhancement-magnitude: match the ROI's mean AND peak intensity to the real
        # post, per-image. L1/MSE regress toward the conditional mean and leave the
        # synthetic tumor too dim, so the pre-vs-post classifier can't tell them apart
        # (low AUROC Contrast). This term directly pushes lesion brightness/peak up to
        # the real level -> a more clearly "post-like" image. Gated on a non-empty mask.
        self.enh_w = enh_weight
        # Gradient-difference (high-frequency) loss: match |grad(pred)| to |grad(target)|
        # in x and y. L1/MSE regress toward the conditional mean and SMOOTH out the high-
        # frequency texture that FRD (radiomics) scores on -> blurry tumors, large FRD.
        # This is the SUPERVISED counterpart to the refiner GAN: it pushes the prediction
        # toward the REAL tumor's edges/texture (cannot hallucinate, unlike the adversary).
        # Extra-weighted inside the ROI, where the radiomics distance is measured.
        self.grad_w = grad_weight
        self.seg_critic = seg_critic
        if seg_critic is not None:
            for p in self.seg_critic.parameters():
                p.requires_grad_(False)
            self.seg_critic.eval()
        self.ssim_window = ssim_window
        self.ssim_sigma = ssim_sigma
        self.data_range = data_range
        self.perceptual = VGGPerceptual(feature_layers) if perceptual_weight > 0 else None
        # LPIPS clip matches the challenge eval (+/- 5 sigma == data_range/2)
        self.lpips = LPIPSLoss(clip_sigma=data_range / 2.0) if lpips_weight > 0 else None

    def _has_mask(self, mask):
        return mask is not None and (mask > 0.5).any()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None,
                source: torch.Tensor | None = None):
        has_mask = self._has_mask(mask)
        use_roi_l1 = has_mask and self.roi_w > 1.0
        use_roi_ssim = has_mask and self.roi_ssim_w > 0.0

        # L1: reweighted so ROI pixels dominate the gradient (also lifts ROI metrics)
        if use_roi_l1:
            w = 1.0 + (self.roi_w - 1.0) * mask
            l1 = (w * (pred - target).abs()).sum() / w.sum()
        else:
            l1 = F.l1_loss(pred, target)

        # MSE: unweighted global L2, matching the challenge's MSE metric exactly.
        # L1 alone predicts the conditional median and smooths the bright-enhancement
        # tail (DUKE peaks reach ~26 sigma); MSE is dominated by that tail, so add it
        # to push the model to reproduce peak dynamic range. Kept global (not ROI-
        # weighted) since the metric is global and the crushed peaks live in the
        # enhancing foreground, not the small tumor mask.
        mse = F.mse_loss(pred, target) if self.mse_w > 0.0 else pred.new_zeros(())

        ssim_l = 1.0 - ssim2d(pred, target, self.ssim_window, self.ssim_sigma,
                              self.data_range)
        # extra structural term inside the ROI (averaged over mask pixels)
        if use_roi_ssim:
            smap = ssim2d(pred, target, self.ssim_window, self.ssim_sigma,
                          self.data_range, return_map=True)
            m = mask > 0.5
            ssim_roi_l = 1.0 - smap[m].mean()
        else:
            ssim_roi_l = pred.new_zeros(())

        # segmentation-perceptual: penalize a frozen segmenter failing to recover the
        # GT mask from the prediction (drives tumor detectability, not pixel fidelity)
        if self.seg_critic is not None and self.seg_w > 0.0 and has_mask:
            seg_l = dice_bce_loss(self.seg_critic(pred), (mask > 0.5).to(pred.dtype))
        else:
            seg_l = pred.new_zeros(())

        # enhancement-magnitude: match ROI mean + peak intensity to the target, per
        # image, to lift under-enhanced (too-dim) tumors toward the real brightness.
        if self.enh_w > 0.0 and has_mask:
            bm = mask > 0.5
            cnt = bm.sum(dim=(1, 2, 3))
            valid = cnt > 0
            cntc = cnt.clamp_min(1).to(pred.dtype)
            mean_p = (pred * bm).sum(dim=(1, 2, 3)) / cntc
            mean_t = (target * bm).sum(dim=(1, 2, 3)) / cntc
            neg = torch.finfo(pred.dtype).min
            max_p = pred.masked_fill(~bm, neg).amax(dim=(1, 2, 3))   # gradient -> brightest ROI pixel
            max_t = target.masked_fill(~bm, neg).amax(dim=(1, 2, 3))
            enh_per = (mean_p - mean_t).abs() + (max_p - max_t).abs()
            enh_l = enh_per[valid].mean() if bool(valid.any()) else pred.new_zeros(())
        else:
            enh_l = pred.new_zeros(())

        # gradient-difference: match horizontal/vertical finite-difference gradients so
        # the prediction reproduces the target's high-frequency structure (sharper tumor
        # texture -> closer radiomics -> lower FRD). Global term + an extra ROI term so
        # the lesion (where FRD is measured) gets the bulk of the high-frequency pressure.
        if self.grad_w > 0.0:
            pdx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            tdx = target[:, :, :, 1:] - target[:, :, :, :-1]
            pdy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            tdy = target[:, :, 1:, :] - target[:, :, :-1, :]
            gdx, gdy = (pdx - tdx).abs(), (pdy - tdy).abs()
            grad_l = gdx.mean() + gdy.mean()
            if has_mask:
                mx = (mask[:, :, :, 1:] > 0.5).to(pred.dtype)
                my = (mask[:, :, 1:, :] > 0.5).to(pred.dtype)
                grad_l = grad_l + ((gdx * mx).sum() / mx.sum().clamp_min(1.0)
                                   + (gdy * my).sum() / my.sum().clamp_min(1.0))
        else:
            grad_l = pred.new_zeros(())

        perc = (self.perceptual(pred, target)
                if self.perceptual is not None else pred.new_zeros(()))
        lp = (self.lpips(pred, target)
              if self.lpips is not None else pred.new_zeros(()))

        # background faithfulness: penalize any residual (pred - source) outside the
        # tumor so the model defaults to identity and only enhances inside the mask.
        if self.bg_w > 0.0 and source is not None and mask is not None and mask.sum() > 0:
            bg = 1.0 - (mask > 0.5).to(pred.dtype)
            denom = bg.sum().clamp_min(1.0)
            bg_l = (bg * (pred - source).abs()).sum() / denom
        else:
            bg_l = pred.new_zeros(())

        total = (self.l1_w * l1 + self.mse_w * mse + self.ssim_w * ssim_l
                 + self.roi_ssim_w * ssim_roi_l + self.perc_w * perc + self.lpips_w * lp
                 + self.bg_w * bg_l + self.seg_w * seg_l + self.enh_w * enh_l
                 + self.grad_w * grad_l)
        # NB: the segmentation-perceptual component is logged as "seg_perc" (not "seg")
        # so it can't clobber the multitask trainer's own "seg" (aux-mask) log key.
        return total, {"l1": l1.item(), "mse": float(mse.detach()),
                       "ssim": ssim_l.item(),
                       "lpips": float(lp.detach()),
                       "ssim_roi": float(ssim_roi_l.detach() if use_roi_ssim else 0.0),
                       "seg_perc": float(seg_l.detach()),
                       "enh": float(enh_l.detach()),
                       "grad": float(grad_l.detach()),
                       "bg": float(bg_l.detach()),
                       "perceptual": float(perc.detach())}
