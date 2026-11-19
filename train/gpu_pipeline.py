"""Decode, resize and augment a training batch on the GPU."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.io import decode_jpeg

from gpu_augment import GpuAugment

__all__ = ["GpuPipeline"]


class GpuPipeline(nn.Module):
    """Batched JPEG decode -> the worker's resize plan -> flips -> photometric augmentation."""

    def __init__(self, augment: bool = True, seed: int | None = None, device=None):
        super().__init__()
        self.device = torch.device(device or "cuda")
        self.augment = GpuAugment(seed=seed, device=self.device).to(self.device) if augment else None

    @torch.no_grad()
    def forward(self, batch: dict) -> Tensor:
        images = decode_jpeg(batch["jpeg"], device=self.device)
        ht, wt = (int(v) for v in batch["target_size"][0])
        groups: dict[tuple, list[int]] = {}
        for i, (img, plan) in enumerate(zip(images, batch["plan"].tolist())):
            groups.setdefault((tuple(img.shape[-2:]), tuple(plan)), []).append(i)
        x = torch.empty(len(images), 3, ht, wt, device=self.device)
        for ((h, w), (top, left, hc, wc, h1, w1, crop_h, crop_w)), idx in groups.items():
            g = torch.stack([images[i] for i in idx]).to(torch.float32).div_(255)
            if (top, left, hc, wc) != (0, 0, h, w):
                g = g[..., top : top + hc, left : left + wc]
                h, w = hc, wc
            if (h1, w1) != (h, w):
                g = F.interpolate(
                    g, (h1, w1), mode="bicubic", align_corners=False, antialias=h1 < h or w1 < w
                ).clamp_(0, 1)
            if crop_h or crop_w:
                g = g[..., crop_h : h1 - crop_h, crop_w : w1 - crop_w]
            assert g.shape[-2:] == (ht, wt), f"{tuple(g.shape[-2:])} != {(ht, wt)}"
            x[idx] = g

        flip = batch["flip"].to(self.device)
        for code, dim in ((1, -1), (2, -2)):
            mask = flip == code
            if bool(mask.any()):
                x[mask] = torch.flip(x[mask], dims=(dim,))

        return self.augment(x) if self.augment is not None else x
