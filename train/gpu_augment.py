"""AnyCalib's photometric augmentations (siclib's `geocalib` preset), on the GPU, batched."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["GpuAugment"]

_Q_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]
_Q_CHROMA = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
]


def _rand(n: int, lo: float, hi: float, device, generator=None) -> Tensor:
    """(n, 1, 1, 1) uniform in [lo, hi], shaped to broadcast over (B, C, H, W)."""
    r = torch.rand(n, 1, 1, 1, device=device, generator=generator)
    return lo + (hi - lo) * r


def _apply_to_some(x: Tensor, p: float, fn, generator=None) -> Tensor:
    """Apply `fn` to a random subset of the batch, as a per-image probability `p` does."""
    if p >= 1.0:
        return fn(x)
    keep = torch.rand(len(x), device=x.device, generator=generator) < p
    if not bool(keep.any()):
        return x
    out = x.clone()
    out[keep] = fn(x[keep])
    return out


def _dct_matrix(n: int, device, dtype) -> Tensor:
    """Orthonormal DCT-II matrix, so that `D @ block @ D.T` is the 2-D DCT."""
    k = torch.arange(n, device=device, dtype=dtype)
    d = torch.cos(math.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    d[0] *= math.sqrt(1 / n)
    d[1:] *= math.sqrt(2 / n)
    return d


class GpuAugment(nn.Module):
    """AnyCalib's `geocalib` augmentation stack for a batch of images in [0, 1]."""

    def __init__(self, p: float = 1.0, seed: int | None = None, device=None):
        super().__init__()
        self.p = p
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(device=device or "cuda")
            self.generator.manual_seed(seed)
        self.register_buffer("q_luma", torch.tensor(_Q_LUMA, dtype=torch.float32).view(8, 8))
        self.register_buffer("q_chroma", torch.tensor(_Q_CHROMA, dtype=torch.float32).view(8, 8))
        self.register_buffer("sepia", torch.tensor([
            [0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]
        ]))
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))


    def random_gamma(self, x: Tensor) -> Tensor:
        """`A.RandomGamma(gamma_limit=(80, 180), p=0.8)`."""
        def fn(z):
            g = _rand(len(z), 0.8, 1.8, z.device, self.generator)
            return z.clamp_min(0).pow(g)
        return _apply_to_some(x, 0.8, fn, self.generator)

    def random_tone_curve(self, x: Tensor) -> Tensor:
        """`A.RandomToneCurve(scale=0.1, p=0.5)`: albumentations' own low/high control points."""
        def fn(z):
            n = len(z)
            low = (_rand(n, 0.0, 1.0, z.device, self.generator) * 0.1 + 0.25).clamp(0.01, 0.99)
            high = (_rand(n, 0.0, 1.0, z.device, self.generator) * 0.1 + 0.75).clamp(0.01, 0.99)
            t = z.clamp(0, 1)
            return (
                (1 - t) ** 3 * 0
                + 3 * (1 - t) ** 2 * t * low
                + 3 * (1 - t) * t**2 * high
                + t**3
            ).clamp(0, 1)
        return _apply_to_some(x, 0.5, fn, self.generator)

    def random_brightness_contrast(self, x: Tensor) -> Tensor:
        """`A.RandomBrightnessContrast(p=0.5)`: albumentations' defaults are +-0.2 on both."""
        def fn(z):
            n = len(z)
            alpha = 1.0 + _rand(n, -0.2, 0.2, z.device, self.generator)
            beta = _rand(n, -0.2, 0.2, z.device, self.generator)
            mean = z.mean(dim=(1, 2, 3), keepdim=True)
            return ((z - mean) * alpha + mean + beta).clamp(0, 1)
        return _apply_to_some(x, 0.5, fn, self.generator)

    def color_jitter(self, x: Tensor) -> Tensor:
        """`A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.4)`, the brightness/contrast/saturation part."""
        def fn(z):
            n = len(z)
            z = (z * (1 + _rand(n, -0.2, 0.2, z.device, self.generator))).clamp(0, 1)
            mean = (z * self.luma).sum(1, keepdim=True).mean(dim=(2, 3), keepdim=True)
            z = ((z - mean) * (1 + _rand(n, -0.2, 0.2, z.device, self.generator)) + mean).clamp(0, 1)
            gray = (z * self.luma).sum(1, keepdim=True)
            z = (gray + (z - gray) * (1 + _rand(n, -0.2, 0.2, z.device, self.generator))).clamp(0, 1)
            try:
                from kornia.color import hsv_to_rgb, rgb_to_hsv

                hsv = rgb_to_hsv(z)
                shift = _rand(n, -0.2, 0.2, z.device, self.generator) * 2 * math.pi
                hsv = torch.cat(((hsv[:, :1] + shift) % (2 * math.pi), hsv[:, 1:]), dim=1)
                z = hsv_to_rgb(hsv).clamp(0, 1)
            except ImportError:
                pass
            return z
        return _apply_to_some(x, 0.4, fn, self.generator)

    def gray_or_sepia(self, x: Tensor) -> Tensor:
        """`A.OneOf([ToGray(p=.1), ToSepia(p=.1), Identity(p=.8)], p=1)`."""
        draw = torch.rand(len(x), device=x.device, generator=self.generator)
        out = x.clone()
        gray_m, sepia_m = draw < 0.1, (draw >= 0.1) & (draw < 0.2)
        if bool(gray_m.any()):
            g = (x[gray_m] * self.luma).sum(1, keepdim=True)
            out[gray_m] = g.expand(-1, 3, -1, -1)
        if bool(sepia_m.any()):
            z = x[sepia_m]
            out[sepia_m] = torch.einsum("ij,bjhw->bihw", self.sepia, z).clamp(0, 1)
        return out


    def gauss_noise(self, x: Tensor) -> Tensor:
        """`_gauss_noise_transform()`: std in [sqrt(5), sqrt(112)]/255, per channel, p=0.75."""
        def fn(z):
            std = _rand(len(z), math.sqrt(5) / 255, math.sqrt(112) / 255, z.device, self.generator)
            noise = torch.randn(z.shape, device=z.device, generator=self.generator)
            return (z + noise * std).clamp(0, 1)
        return _apply_to_some(x, 0.75, fn, self.generator)

    def iso_noise(self, x: Tensor) -> Tensor:
        """`A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)`."""
        def fn(z):
            try:
                from kornia.color import hsv_to_rgb, rgb_to_hsv
            except ImportError:
                return z
            n = len(z)
            hsv = rgb_to_hsv(z)
            intensity = _rand(n, 0.1, 0.5, z.device, self.generator)
            color_shift = _rand(n, 0.01, 0.05, z.device, self.generator)
            value = hsv[:, 2:3]
            stddev = value.std(dim=(1, 2, 3), keepdim=True)
            noise = torch.randn(value.shape, device=z.device, generator=self.generator)
            value = (value + noise * stddev * intensity).clamp(0, 1)
            hue = (hsv[:, :1] + color_shift * 2 * math.pi) % (2 * math.pi)
            sat = (hsv[:, 1:2] * (1 + intensity * 0.5)).clamp(0, 1)
            return hsv_to_rgb(torch.cat((hue, sat, value), dim=1)).clamp(0, 1)
        return _apply_to_some(x, 0.5, fn, self.generator)

    def jpeg(self, x: Tensor) -> Tensor:
        """`A.ImageCompression(quality_lower=20, quality_upper=100, p=1)`, approximated."""
        b, c, h, w = x.shape
        ph, pw = (-h) % 8, (-w) % 8
        z = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        H, W = z.shape[-2:]

        quality = _rand(b, 20, 100, x.device, self.generator).view(b, 1, 1, 1, 1, 1)
        scale = torch.where(quality < 50, 5000 / quality, 200 - 2 * quality) / 100
        q = torch.stack((self.q_luma, self.q_chroma, self.q_chroma))
        q = (q.view(1, 3, 1, 1, 8, 8) * scale).clamp(1, 255)

        blocks = z.view(b, c, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 3, 5)
        d = _dct_matrix(8, x.device, x.dtype)
        coeff = d @ (blocks - 0.5) @ d.T
        coeff = torch.round(coeff * 255 / q) * q / 255
        out = (d.T @ coeff @ d + 0.5).clamp(0, 1)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(b, c, H, W)
        return out[..., :h, :w]


    def blur_and_sharpen(self, x: Tensor) -> Tensor:
        """`A.OneOrOther(AdvancedBlur+Sharpen, Sharpen+AdvancedBlur)`."""
        sigma = float(_rand(1, 0.2, 1.0, x.device, self.generator))
        k = 7
        grid = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        g = torch.exp(-grid**2 / (2 * sigma**2))
        g = (g / g.sum()).view(1, 1, 1, k)

        def blur(z):
            z = F.conv2d(F.pad(z, (k // 2,) * 2 + (0, 0), mode="reflect"),
                         g.expand(z.shape[1], 1, 1, k), groups=z.shape[1])
            return F.conv2d(F.pad(z, (0, 0) + (k // 2,) * 2, mode="reflect"),
                            g.transpose(-1, -2).expand(z.shape[1], 1, k, 1), groups=z.shape[1])

        def sharpen(z):
            alpha = _rand(len(z), 0.2, 0.5, z.device, self.generator)
            return (z + alpha * (z - blur(z))).clamp(0, 1)

        first_blur = bool(torch.rand((), device=x.device, generator=self.generator) < 0.5)
        x = blur(x) if first_blur else _apply_to_some(x, 0.5, sharpen, self.generator)
        x = _apply_to_some(x, 0.5, sharpen, self.generator) if first_blur else blur(x)
        return x.clamp(0, 1)

    def downscale(self, x: Tensor) -> Tensor:
        """`A.OneOf([Downscale(0.5-0.99, 4 interpolation pairs)], p=1)`."""
        pairs = (("area", "bilinear"), ("bilinear", "bicubic"),
                 ("bicubic", "bilinear"), ("bilinear", "area"))
        down, up = pairs[int(torch.randint(0, 4, (1,), device=x.device, generator=self.generator))]
        scale = float(_rand(1, 0.5, 0.99, x.device, self.generator))
        h, w = x.shape[-2:]
        small = F.interpolate(x, (max(1, int(h * scale)), max(1, int(w * scale))), mode=down)
        kw = {} if up == "area" else {"align_corners": False}
        return F.interpolate(small, (h, w), mode=up, **kw).clamp(0, 1)


    def forward(self, x: Tensor) -> Tensor:
        """(B, 3, H, W) in [0, 1] -> the same, augmented. AnyCalib's order."""
        if self.p < 1.0 and float(torch.rand((), device=x.device, generator=self.generator)) > self.p:
            return x
        x = self.random_gamma(x)
        x = self.random_tone_curve(x)
        x = self.random_brightness_contrast(x)
        x = self.color_jitter(x)
        x = self.gray_or_sepia(x)
        x = self.gauss_noise(x)
        x = self.jpeg(x)
        x = self.iso_noise(x)
        x = self.blur_and_sharpen(x)
        x = self.downscale(x)
        return x
