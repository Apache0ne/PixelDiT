from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixdit_core.modules import (
    ModulatedDiTBlock,
    TimestepConditioner,
    precompute_freqs_cis_2d,
)


@dataclass(frozen=True)
class EncoderSpec:
    timm_name: str
    feature_dim: int
    patch_size: int


GROUNDING_ENCODERS: Dict[str, EncoderSpec] = {
    "dinov3_vits16": EncoderSpec("vit_small_patch16_dinov3_qkvb.lvd1689m", 384, 16),
    "dinov3_vitb16": EncoderSpec("vit_base_patch16_dinov3_qkvb.lvd1689m", 768, 16),
    "dinov3_vitl16": EncoderSpec("vit_large_patch16_dinov3_qkvb.lvd1689m", 1024, 16),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_encoder(name: str, pretrained: bool = True) -> nn.Module:
    import timm

    if name not in GROUNDING_ENCODERS:
        raise ValueError(
            f"Unknown grounding encoder {name!r}. Available: {sorted(GROUNDING_ENCODERS)}"
        )
    return timm.create_model(
        GROUNDING_ENCODERS[name].timm_name,
        pretrained=pretrained,
        dynamic_img_size=True,
    )


class TimestepProjection(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_size: int,
        num_blocks: int = 1,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"TimestepProjection hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.head_dim = hidden_size // num_heads
        self.input_proj = nn.Linear(in_dim, hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.blocks = nn.ModuleList(
            [
                ModulatedDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(num_blocks)
            ]
        )
        self.output_proj = nn.Linear(hidden_size, out_dim, bias=True)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        self._pos_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def fetch_pos(self, height: int, width: int, device) -> torch.Tensor:
        key = (height, width)
        if key not in self._pos_cache:
            self._pos_cache[key] = precompute_freqs_cis_2d(self.head_dim, height, width)
        return self._pos_cache[key].to(device)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.input_proj(tokens)
        c = self.t_embedder(t).unsqueeze(1).to(dtype=x.dtype)
        pos = self.fetch_pos(grid_hw[0], grid_hw[1], x.device)
        for block in self.blocks:
            x = block(x, c, pos)
        return self.output_proj(x)


class GroundingConditioner(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        hidden_size: int,
        target_patch_size: int,
        proj_num_blocks: int = 1,
        proj_num_heads: int = 16,
        pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.spec = GROUNDING_ENCODERS[encoder_name]
        self.encoder_name = encoder_name
        self.target_patch_size = int(target_patch_size)
        self.encoder = build_encoder(encoder_name, pretrained=pretrained_encoder)
        self.encoder.requires_grad_(False)
        self.dino_proj = TimestepProjection(
            in_dim=self.spec.feature_dim,
            out_dim=hidden_size,
            hidden_size=hidden_size,
            num_blocks=proj_num_blocks,
            num_heads=proj_num_heads,
        )
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = x.shape[-2:]
        grid_hw = (height // self.target_patch_size, width // self.target_patch_size)
        encoder_hw = (grid_hw[0] * self.spec.patch_size, grid_hw[1] * self.spec.patch_size)
        x = (x + 1.0) * 0.5
        x = (x - self.image_mean.to(x.dtype)) / self.image_std.to(x.dtype)
        if encoder_hw != (height, width):
            x = F.interpolate(x, size=encoder_hw, mode="bicubic", align_corners=False)
        return x, grid_hw

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        encoder_input, grid_hw = self.preprocess(x)
        with torch.no_grad():
            features = self.encoder.forward_features(encoder_input)
        prefix = int(getattr(self.encoder, "num_prefix_tokens", 0))
        tokens = features[:, prefix:, :]
        encoder_grid = (
            encoder_input.shape[-2] // self.spec.patch_size,
            encoder_input.shape[-1] // self.spec.patch_size,
        )
        if encoder_grid != grid_hw:
            batch, _, channels = tokens.shape
            tokens = tokens.transpose(1, 2).reshape(batch, channels, *encoder_grid)
            tokens = F.interpolate(tokens, size=grid_hw, mode="bilinear", align_corners=False)
            tokens = tokens.flatten(2).transpose(1, 2)
        return tokens, grid_hw

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tokens, grid_hw = self.encode(x)
        return self.dino_proj(tokens.to(dtype=self.dino_proj.input_proj.weight.dtype), t, grid_hw)
