# Modified from https://github.com/MCG-NJU/PixNerd and https://github.com/LTH14/JiT

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixdit_core.grounding import GroundingConditioner
from pixdit_core.modules import (
    ClassEmbedder,
    ModulatedDiTBlock,
    PatchFinalLayer,
    PatchTokenEmbedder,
    TimestepConditioner,
    precompute_freqs_cis_2d,
)


class PixelDiT2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1280,
        num_heads: int = 16,
        depth: int = 32,
        patch_size: int = 16,
        num_classes: int = 1000,
        in_context_len: int = 32,
        in_context_start: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_hidden_multiple: int = 0,
        grounding_encoder: str = "dinov3_vits16",
        grounding_proj_blocks: int = 1,
        grounding_proj_heads: Optional[int] = None,
        single_silu_cond: bool = False,
        pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.single_silu_cond = single_silu_cond

        if not 0 <= in_context_start < depth:
            raise ValueError(
                f"in_context_start must be in [0, {depth - 1}], got {in_context_start}"
            )

        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = ClassEmbedder(num_classes + 1, hidden_size)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.in_context_posemb = (
            nn.Parameter(torch.zeros(1, in_context_len, hidden_size))
            if in_context_len > 0
            else None
        )
        self.patch_blocks = nn.ModuleList(
            [
                ModulatedDiTBlock(
                    hidden_size,
                    num_heads,
                    attn_drop=attn_dropout if depth // 4 <= i < depth // 4 * 3 else 0.0,
                    proj_drop=proj_dropout if depth // 4 <= i < depth // 4 * 3 else 0.0,
                    mlp_hidden_multiple=mlp_hidden_multiple,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = PatchFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        self.dino_conditioner = GroundingConditioner(
            encoder_name=grounding_encoder,
            hidden_size=hidden_size,
            target_patch_size=patch_size,
            proj_num_blocks=grounding_proj_blocks,
            proj_num_heads=num_heads if grounding_proj_heads is None else grounding_proj_heads,
            pretrained_encoder=pretrained_encoder,
        )
        self._pos_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.in_context_posemb is not None:
            nn.init.normal_(self.in_context_posemb, std=0.02)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)

    def fetch_pos(self, height: int, width: int, device, num_context_tokens: int = 0):
        key = (height, width, num_context_tokens)
        if key not in self._pos_cache:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width)
            if num_context_tokens > 0:
                identity = torch.ones(num_context_tokens, pos.shape[-1], dtype=pos.dtype)
                pos = torch.cat([identity, pos], dim=0)
            self._pos_cache[key] = pos
        return self._pos_cache[key].to(device)

    def _grounded_condition(self, x, t, t_emb, y_emb):
        grounding = self.dino_conditioner(x, t)
        c = grounding + t_emb.unsqueeze(1) + y_emb
        return c if self.single_silu_cond else F.silu(c)

    def _null_condition(self, t_emb, y_emb, num_patches):
        c = t_emb.unsqueeze(1) + y_emb
        if not self.single_silu_cond:
            c = F.silu(c)
        return c.expand(-1, num_patches, -1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        skip_grounding: bool = False,
        skip_grounding_mask: Optional[torch.Tensor] = None,
        return_grounding_tokens: bool = False,
    ):
        batch, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input resolution ({height}, {width}) must be divisible by "
                f"patch_size={self.patch_size}"
            )
        t = t.view(batch)
        hs, ws = height // self.patch_size, width // self.patch_size
        num_patches = hs * ws

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y).view(batch, 1, self.hidden_size)

        if skip_grounding:
            c = self._null_condition(t_emb, y_emb, num_patches)
        else:
            c = self._grounded_condition(x, t, t_emb, y_emb)
            if skip_grounding_mask is not None:
                null_c = self._null_condition(t_emb, y_emb, num_patches)
                c = torch.where(skip_grounding_mask.view(batch, 1, 1), null_c, c)

        s = self.s_embedder(
            F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        )

        use_context = self.in_context_len > 0
        pos = self.fetch_pos(hs, ws, x.device)
        pos_with_context = self.fetch_pos(hs, ws, x.device, self.in_context_len) if use_context else None
        if use_context:
            c_context = c.mean(dim=1, keepdim=True).expand(-1, self.in_context_len, -1)
            c_extended = torch.cat([c_context, c], dim=1)

        context_inserted = False
        for i, block in enumerate(self.patch_blocks):
            if use_context and not context_inserted and i == self.in_context_start:
                context_tokens = y_emb.expand(-1, self.in_context_len, -1)
                s = torch.cat([context_tokens + self.in_context_posemb.to(s.dtype), s], dim=1)
                context_inserted = True
            s = block(
                s,
                c_extended if context_inserted else c,
                pos_with_context if context_inserted else pos,
            )
        if context_inserted:
            s = s[:, self.in_context_len:, :]

        out = self.final_layer(s, c)
        out = out.view(batch, num_patches, self.patch_size ** 2, self.out_channels)
        out = out.permute(0, 3, 2, 1).reshape(batch, self.out_channels * self.patch_size ** 2, num_patches)
        out = F.fold(out, (height, width), kernel_size=self.patch_size, stride=self.patch_size)

        if return_grounding_tokens:
            return out, self.dino_conditioner.encode(x)[0]
        return out


def _factory(hidden_size, num_heads, depth, patch_size, in_context_start):
    def build(**kwargs):
        kwargs.setdefault("hidden_size", hidden_size)
        kwargs.setdefault("num_heads", num_heads)
        kwargs.setdefault("depth", depth)
        kwargs.setdefault("patch_size", patch_size)
        kwargs.setdefault("in_context_start", in_context_start)
        return PixelDiT2(**kwargs)

    return build


PixelDiT2_B_16 = _factory(768, 12, 12, 16, 4)
PixelDiT2_L_16 = _factory(1024, 16, 24, 16, 8)
PixelDiT2_H_16 = _factory(1280, 16, 32, 16, 8)
PixelDiT2_H_32 = _factory(1280, 16, 32, 32, 8)
PixelDiT2_G_32 = _factory(1664, 16, 40, 32, 8)

PixelDiT2_models = {
    "PixelDiT2-B/16": PixelDiT2_B_16,
    "PixelDiT2-L/16": PixelDiT2_L_16,
    "PixelDiT2-H/16": PixelDiT2_H_16,
    "PixelDiT2-H/32": PixelDiT2_H_32,
    "PixelDiT2-G/32": PixelDiT2_G_32,
}
