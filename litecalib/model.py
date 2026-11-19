"""The LiteCalib network: a DINOv2 backbone, position-aware tokens and a small query decoder that reads
out the camera's FoV field at a K x K grid of image points (or, for the ablations, a radial angle profile
or the one-parameter division model directly)."""

import inspect
import logging
import math
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from litecalib import geometry
from litecalib.anycalib.model.dpt_light_decoder import LightDPTDecoder
from litecalib.anycalib.model.ray_decoder import ConvexTangentDecoder
from litecalib.anycalib.model.vision_transformer import vit_base, vit_large, vit_small

logger = logging.getLogger(__name__)

BACKBONES = {
    "vits14": (vit_small, 384),
    "vitb14": (vit_base, 768),
    "vitl14": (vit_large, 1024),
}
DEPTHS = {"vits14": 12, "vitb14": 12, "vitl14": 24}
PATCH_SIZE = 14
PROFILES = ("field", "radial", "division")
REGRESSION_CAMS = {"division": "division:1"}
DENSE_TAPS = {"vits14": (2, 5, 8), "vitb14": (2, 5, 8), "vitl14": (4, 11, 17)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sinusoidal(v: Tensor, num_freqs: int = 16) -> Tensor:
    """Sinusoidal encoding of a scalar: sin/cos of ``2^i pi v``. (..., 2F)"""
    freqs = 2 ** torch.arange(num_freqs, device=v.device, dtype=v.dtype) * math.pi
    x = v.unsqueeze(-1) * freqs
    return torch.cat((x.sin(), x.cos()), dim=-1)


def _sdpa_attention_forward(self, x: Tensor, attn_bias=None) -> Tensor:
    """DINOv2's `Attention.forward` through `scaled_dot_product_attention` (same result, less memory)."""
    assert attn_bias is None, "attention bias (nested tensors) is not used here"
    b, n, c = x.shape
    qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    x = x.transpose(1, 2).reshape(b, n, c)
    return self.proj_drop(self.proj(x))


def use_sdpa_attention(vit: nn.Module) -> None:
    """Swap every attention module of a DINOv2 ViT to the SDPA path."""
    for block in vit.blocks:
        block.attn.forward = types.MethodType(_sdpa_attention_forward, block.attn)


class Backbone(nn.Module):
    """DINOv2 ViT/14 returning the last layer's patch tokens and class token (inputs in [0, 1])."""

    image_mean: Tensor
    image_std: Tensor

    def __init__(self, name: str = "vitl14", sdpa: bool = True):
        super().__init__()
        assert name in BACKBONES, f"unknown backbone {name!r}, known: {sorted(BACKBONES)}"
        constructor, dim = BACKBONES[name]
        self.model = constructor(
            img_size=518,
            patch_size=PATCH_SIZE,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        self.name = name
        self.embed_dim = dim
        self.patch_size = PATCH_SIZE
        self.input_divisor = PATCH_SIZE
        self.model.mask_token.requires_grad_(False)
        if sdpa:
            use_sdpa_attention(self.model)
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(B, 3, H, W) in [0, 1] -> (B, N, D) patch tokens and (B, D) class token."""
        tokens, cls, _ = self.forward_taps(x, [])
        return tokens, cls

    def forward_taps(self, x: Tensor, taps: list[int]) -> tuple[Tensor, Tensor, list[Tensor]]:
        """`forward`, plus the normed (B, 1 + N, D) sequences after the blocks in `taps` (0-indexed)."""
        x = (x - self.image_mean) / self.image_std
        x = self.model.prepare_tokens_with_masks(x)
        outs = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in taps:
                outs.append(x)
        x = self.model.norm(x)
        outs = [self.model.norm(o) for o in outs]
        return x[:, 1:], x[:, 0], outs


def build_anycalib(backbone: str = "vitl14", dense_dim: int = 256, cls=None, sdpa: bool = False):
    """AnyCalib's network (DINOv2, light DPT decoder, convex tangent head) at any DINOv2 size."""
    from litecalib.anycalib.model import dinov2 as ac_dinov2
    from litecalib.anycalib.model.anycalib_pretrained import AnyCalib, Calibrator

    cls = cls or AnyCalib
    if backbone == "vitl14" and dense_dim == 256:
        net = cls()
    else:
        assert backbone in BACKBONES, f"unknown backbone {backbone!r}, known: {sorted(BACKBONES)}"
        name = f"dinov2_{backbone}"
        constructor, dim = BACKBONES[backbone]
        ac_dinov2.DINOV2_CFG.setdefault(name, {
            "embed_dim": dim, "constructor": constructor,
            "out_index": [*DENSE_TAPS[backbone], DEPTHS[backbone] - 1],
        })
        net = cls.__new__(cls)
        nn.Module.__init__(net)
        net.backbone = ac_dinov2.DINOv2(model_name=name)
        net.decoder = LightDPTDecoder(embed_dim=net.backbone.embed_dim, dim_dhat=dense_dim)
        net.head = ConvexTangentDecoder(in_channels=net.decoder.out_channels)
        defaults = {k: p.default for k, p in inspect.signature(AnyCalib.__init__).parameters.items()
                    if k not in ("self", "model_id")}
        net.calibrator = Calibrator(**defaults)
    if sdpa:
        use_sdpa_attention(net.backbone.model)
    return net


class DecoderLayer(nn.Module):
    """Pre-norm self-attention over the queries, cross-attention to the tokens, MLP."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim)
        )

    def forward(self, q: Tensor, memory: Tensor) -> Tensor:
        h = self.norm_q(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        kv = self.norm_kv(memory)
        q = q + self.cross_attn(self.norm_q2(q), kv, kv, need_weights=False)[0]
        return q + self.mlp(self.norm_mlp(q))


class RadialProfileNet(nn.Module):
    """Backbone + position-aware tokens + query decoder.

    Args:
        backbone: ``vits14`` | ``vitb14`` | ``vitl14``.
        k: grid points per side (``field``) or knots (``radial``).
        profile: ``"field"`` (the method: the FoV field at a ``k x k`` grid, ``2 k^2`` outputs),
            ``"radial"`` (a monotone angle profile at ``k`` radii along the diagonal) or
            ``"division"`` (``division:1``'s ``(f, k1)`` regressed directly).
        field_repr: what the field's outputs are, ``"fov"`` (``(atan2(X, Z), atan2(Y, Z))``) or
            ``"logmap"`` (AnyCalib's log map of the ray).
        dense_head: add AnyCalib's DPT decoder and tangent head as a training-only dense branch; it
            runs only in training mode and only sends its gradient into the backbone.
        dense_dim: the dense branch's width.
        dense_repr: what the dense branch predicts per pixel, ``"logmap"`` or ``"fov"``.
    """

    def __init__(
        self,
        backbone: str = "vitl14",
        k: int = geometry.DEFAULT_K,
        placement: str = geometry.DEFAULT_PLACEMENT,
        layers: int = 2,
        q_dim: int | None = None,
        num_heads: int = 8,
        num_freqs: int = 16,
        sdpa: bool = True,
        init_vfov_deg: float = 60.0,
        init_logb: float = math.log(math.pi / 180),
        profile: str = "field",
        dense_head: bool = False,
        dense_dim: int = 256,
        dense_repr: str = "logmap",
        field_repr: str = "fov",
    ):
        super().__init__()
        assert profile in PROFILES, f"unknown profile {profile!r}, known: {PROFILES}"
        assert field_repr in geometry.FIELD_REPRS, f"unknown field representation {field_repr!r}"
        assert dense_repr in geometry.FIELD_REPRS, f"unknown dense representation {dense_repr!r}"
        self.backbone = Backbone(backbone, sdpa=sdpa)
        dim = self.backbone.embed_dim
        q_dim = dim if q_dim is None else q_dim
        self.k = k
        self.placement = placement
        self.profile = profile
        self.field_repr = field_repr
        self.dense_repr = dense_repr
        self.num_freqs = num_freqs
        self._u_cache: dict[tuple, Tensor] = {}
        self.dense = None
        self.dense_taps: list[int] = []
        self.memory_proj = nn.Identity() if q_dim == dim else nn.Linear(dim, q_dim)
        self.layers = nn.ModuleList(DecoderLayer(q_dim, num_heads) for _ in range(layers))
        self.norm_out = nn.LayerNorm(q_dim)
        if profile == "field":
            self._init_field(dim, q_dim, init_vfov_deg, init_logb)
        elif profile == "division":
            self._init_division(dim, q_dim, init_vfov_deg)
        else:
            self._init_radial(dim, q_dim, init_vfov_deg, init_logb)
        if dense_head:
            decoder = LightDPTDecoder(embed_dim=dim, dim_dhat=dense_dim)
            self.dense_taps = list(DENSE_TAPS[backbone])
            self.dense = nn.ModuleDict({
                "decoder": decoder,
                "head": ConvexTangentDecoder(in_channels=decoder.out_channels,
                                             up_factor=self.backbone.patch_size // 2),
            })

    def _init_field(self, dim: int, q_dim: int, vfov_deg: float, logb: float) -> None:
        """One query per grid point, told its ``(x, y)``; a linear head emits the two field values,
        biased to a centered pinhole with `vfov_deg` on a square image."""
        k = self.k
        v = geometry.grid_knot_v(k, dtype=torch.float32)
        n = geometry.grid_num_knots(k)
        enc = 4 * self.num_freqs
        self.token_radius = nn.Linear(enc, dim)
        self.query_radius = nn.Linear(enc, q_dim)
        self.knot_embed = nn.Parameter(torch.zeros(n, q_dim))
        nn.init.normal_(self.knot_embed, std=0.02)
        self.knot_head = nn.Linear(q_dim, 4)
        nn.init.zeros_(self.knot_head.bias)
        nn.init.normal_(self.knot_head.weight, std=1e-4)
        tan_half = math.tan(0.5 * math.radians(vfov_deg))
        yy, xx = torch.meshgrid(v, v, indexing="ij")
        pos = torch.stack((xx, yy), dim=-1).reshape(n, 2)
        bias = geometry.field_values(torch.atan(pos * tan_half))
        if self.field_repr == "logmap":
            bias = geometry.field_to_logmap(bias)
        self.z_bias = nn.Parameter(bias)
        self.theta_index = (k // 2) * k + k - 1
        self.logb_bias = nn.Parameter(torch.full((2 * n,), logb))
        self.register_buffer("knot_pos", pos.contiguous(), persistent=False)

    def _init_radial(self, dim: int, q_dim: int, vfov_deg: float, logb: float) -> None:
        """K knot queries and one FoV query at their radii; the outputs parameterize a monotone
        profile (`geometry.profile_from_outputs`) initialized to a centered pinhole."""
        k = self.k
        self.token_radius = nn.Linear(2 * self.num_freqs, dim)
        self.query_radius = nn.Linear(2 * self.num_freqs, q_dim)
        self.knot_embed = nn.Parameter(torch.zeros(k, q_dim))
        self.fov_embed = nn.Parameter(torch.zeros(1, q_dim))
        nn.init.normal_(self.knot_embed, std=0.02)
        nn.init.normal_(self.fov_embed, std=0.02)
        self.knot_head = nn.Linear(q_dim, 2)
        self.fov_head = nn.Linear(q_dim, 1)
        self.z_bias = nn.Parameter(torch.zeros(k))
        self.logb_bias = nn.Parameter(torch.zeros(k))
        u = geometry.knot_u(k, self.placement, dtype=torch.float32)
        theta = torch.atan(u * math.sqrt(2) * math.tan(0.5 * math.radians(vfov_deg)))
        s_bias, z_bias = geometry.bias_init_for_profile(theta)
        for head in (self.knot_head, self.fov_head):
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=1e-4)
        with torch.no_grad():
            self.fov_head.bias.fill_(float(s_bias))
            self.z_bias.copy_(z_bias)
            self.logb_bias.fill_(logb)

    def _init_division(self, dim: int, q_dim: int, vfov_deg: float) -> None:
        """Two queries regress ``log(f / rho_max)`` and ``kappa = 1 - softplus(x)`` of the division model
        in half-diagonal units; the K knots are rendered from them for the loss."""
        self.token_radius = nn.Linear(2 * self.num_freqs, dim)
        self.knot_embed = nn.Parameter(torch.zeros(2, q_dim))
        nn.init.normal_(self.knot_embed, std=0.02)
        self.f_head = nn.Linear(q_dim, 1)
        self.kappa_head = nn.Linear(q_dim, 1)
        for head in (self.f_head, self.kappa_head):
            nn.init.normal_(head.weight, std=1e-4)
        with torch.no_grad():
            self.f_head.bias.fill_(-math.log(math.sqrt(2) * math.tan(0.5 * math.radians(vfov_deg))))
            self.kappa_head.bias.fill_(math.log(math.e - 1))
        self.register_buffer(
            "knot_u", geometry.knot_u(self.k, self.placement, dtype=torch.float32), persistent=False
        )

    def load_dinov2_backbone(self, ckpt: str | Path) -> dict:
        """Load DINOv2's own released weights into the backbone."""
        return load_dinov2_backbone(self, ckpt)

    def patch_u_sq(self, h: int, w: int, device, dtype) -> Tensor:
        """``u^2`` (squared radius over the half-diagonal) of every patch center, row-major. (N,)"""
        key = ("u", h, w, str(device), dtype)
        if key not in self._u_cache:
            p = self.backbone.patch_size
            assert h % p == 0 == w % p, f"input {h}x{w} must be divisible by {p}"
            ys = (torch.arange(h // p, dtype=dtype) + 0.5) * p - 0.5 * h
            xs = (torch.arange(w // p, dtype=dtype) + 0.5) * p - 0.5 * w
            u_sq = (ys[:, None] ** 2 + xs[None, :] ** 2) / geometry.rho_max_of(h, w) ** 2
            self._u_cache[key] = u_sq.reshape(-1).to(device)
        return self._u_cache[key]

    def patch_xy(self, h: int, w: int, device, dtype) -> Tensor:
        """Normalized ``(x, y)`` in ``[-1, 1]`` of every patch center, row-major. (N, 2)"""
        key = ("xy", h, w, str(device), dtype)
        if key not in self._u_cache:
            p = self.backbone.patch_size
            assert h % p == 0 == w % p, f"input {h}x{w} must be divisible by {p}"
            ys = ((torch.arange(h // p, dtype=dtype) + 0.5) * p - 0.5 * h) / (0.5 * h)
            xs = ((torch.arange(w // p, dtype=dtype) + 0.5) * p - 0.5 * w) / (0.5 * w)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            self._u_cache[key] = torch.stack((xx, yy), dim=-1).reshape(-1, 2).to(device)
        return self._u_cache[key]

    def _decode(self, q: Tensor, memory: Tensor) -> Tensor:
        for layer in self.layers:
            q = layer(q, memory)
        return self.norm_out(q)

    def _dense_field(self, tokens: Tensor, taps: list[Tensor], h: int, w: int) -> Tensor:
        """The dense branch: (B, H*W, 2) per-pixel log map (or FoV field)."""
        b, p = len(tokens), self.backbone.patch_size
        maps = [t[:, 1:] for t in taps] + [tokens]
        maps = [m.transpose(1, 2).reshape(b, -1, h // p, w // p) for m in maps]
        out = self.dense["head"](self.dense["decoder"]({"outputs": maps}))
        tangent = out["tangent_coords"]
        assert tangent.shape[-2:] == (h, w), tangent.shape
        return tangent.flatten(2).transpose(1, 2)

    def _forward_field(self, tokens: Tensor, cls: Tensor, h: int, w: int) -> dict:
        xy = self.patch_xy(h, w, tokens.device, tokens.dtype)
        tokens = tokens + self.token_radius(sinusoidal(xy, self.num_freqs).flatten(-2))
        memory = self.memory_proj(torch.cat((tokens, cls.unsqueeze(1)), dim=1))
        q = self.knot_embed + self.query_radius(
            sinusoidal(self.knot_pos.to(self.knot_embed.dtype), self.num_freqs).flatten(-2))
        q = self._decode(q.unsqueeze(0).expand(len(tokens), -1, -1), memory)
        knots = self.knot_head(q)
        theta = geometry.field_values(knots[..., :2]) + self.z_bias
        logb = geometry.field_values(knots[..., 2:]) + self.logb_bias
        return {"theta": theta, "logb": logb, "Theta": theta[:, self.theta_index]}

    def _forward_division(self, tokens: Tensor, cls: Tensor, h: int, w: int) -> dict:
        u_sq = self.patch_u_sq(h, w, tokens.device, tokens.dtype)
        tokens = tokens + self.token_radius(sinusoidal(u_sq, self.num_freqs))
        memory = self.memory_proj(torch.cat((tokens, cls.unsqueeze(1)), dim=1))
        q = self._decode(self.knot_embed.unsqueeze(0).repeat(len(tokens), 1, 1), memory)
        log_f = self.f_head(q[:, 0]).squeeze(-1).float()
        kappa = 1 - F.softplus(self.kappa_head(q[:, 1]).squeeze(-1).float())
        f_rel = log_f.exp()
        theta = geometry.division_profile(self.knot_u.float(), f_rel, kappa)
        theta_max = geometry.division_profile(self.knot_u.new_ones(1), f_rel, kappa)[:, 0]
        return {"theta": theta, "Theta": theta_max, "logb": torch.zeros_like(theta),
                "log_f": log_f, "kappa": kappa}

    def _forward_radial(self, tokens: Tensor, cls: Tensor, h: int, w: int) -> dict:
        u_sq = self.patch_u_sq(h, w, tokens.device, tokens.dtype)
        tokens = tokens + self.token_radius(sinusoidal(u_sq, self.num_freqs))
        memory = self.memory_proj(torch.cat((tokens, cls.unsqueeze(1)), dim=1))
        q = torch.cat((self.knot_embed, self.fov_embed), dim=0)
        u_sq_k = geometry.knot_u_sq(self.k, self.placement, dtype=q.dtype, device=q.device)
        u_sq_k = torch.cat((u_sq_k, u_sq_k.new_ones(1)))
        q = q + self.query_radius(sinusoidal(u_sq_k, self.num_freqs))
        q = self._decode(q.unsqueeze(0).expand(len(tokens), -1, -1), memory)
        knots = self.knot_head(q[:, : self.k])
        z = knots[..., 0] + self.z_bias
        logb = knots[..., 1] + self.logb_bias
        s = self.fov_head(q[:, self.k]).squeeze(-1)
        theta, theta_max = geometry.profile_from_outputs(s, z)
        return {"s": s, "z": z, "logb": logb, "theta": theta, "Theta": theta_max}

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        """(B, 3, H, W) images in [0, 1] -> dict with ``theta`` (the field values or knot angles), ``logb``
        and ``Theta``; in training mode with the dense branch also ``tangent_coords`` (B, H*W, 2)."""
        dense = self.dense is not None and self.training
        if dense:
            tokens, cls, taps = self.backbone.forward_taps(image, self.dense_taps)
        else:
            tokens, cls = self.backbone(image)
        h, w = image.shape[-2:]
        decode = {"field": self._forward_field, "division": self._forward_division,
                  "radial": self._forward_radial}[self.profile]
        out = decode(tokens, cls, h, w)
        if dense:
            out["tangent_coords"] = self._dense_field(tokens, taps, h, w)
        return out

    def param_groups(self, lr: float, backbone_lr: float, weight_decay: float) -> list[dict]:
        """AdamW groups: the backbone at its own lr, no weight decay on 1-D parameters."""
        groups: list[dict] = []
        for name, is_backbone in (("backbone", True), ("head", False)):
            params = [(n, p) for n, p in self.named_parameters()
                      if p.requires_grad and n.startswith("backbone.") == is_backbone]
            group_lr = backbone_lr if is_backbone else lr
            groups.append({"params": [p for _, p in params if p.ndim >= 2], "lr": group_lr,
                           "weight_decay": weight_decay, "name": f"{name}_decay"})
            groups.append({"params": [p for _, p in params if p.ndim < 2], "lr": group_lr,
                           "weight_decay": 0.0, "name": f"{name}_no_decay"})
        return [g for g in groups if g["params"]]


def load_dinov2_backbone(net: RadialProfileNet, ckpt: str | Path) -> dict:
    """Load DINOv2's released weights (``dinov2_vit{s,b,l}14_pretrain.pth``) into `net`'s backbone."""
    ckpt = Path(ckpt).expanduser()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    own_state = net.backbone.model.state_dict()
    bad_shape = [k for k, v in state.items() if k in own_state and own_state[k].shape != v.shape]
    if bad_shape:
        raise ValueError(f"{ckpt.name} does not fit backbone {net.backbone.name!r}: {bad_shape[:3]}")
    result = net.backbone.model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if k in dict(net.backbone.model.named_parameters())]
    if missing:
        raise ValueError(f"{ckpt.name} does not fit backbone {net.backbone.name!r}: missing {missing[:3]}")
    logger.info("loaded %d DINOv2 backbone tensors from %s", len(state), ckpt.name)
    return {"ckpt": str(ckpt), "loaded": len(state), "missing": list(result.missing_keys),
            "unexpected": list(result.unexpected_keys)}
