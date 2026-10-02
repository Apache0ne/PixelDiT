"""PixelDiT2 image-to-image editing via the model's frozen DINOv3 representation prior."""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixdit_core.edit_adapter import load_edit_adapter
from pixdit_core.lora import freeze_module, lora_parameter_counts
from pixdit_core.pixeldit2_c2i import PixelDiT2


class PixelDiT2Edit(PixelDiT2):
    """PixelDiT2 with clean-source DINO conditioning for image editing.

    The existing noisy-image DINO grounding remains unchanged. The source image is
    encoded by the same frozen DINO encoder and injected as a gated residual into
    each block's spatial AdaLN conditioning. All source gates start at exactly zero,
    so this class reproduces the pretrained C2I model at initialization.
    """

    def __init__(
        self,
        *args,
        source_projection: str = "clone",
        source_projection_t: str = "clean",
        train_source_projection: bool = True,
        source_gate_per_channel: bool = True,
        freeze_base: bool = True,
        edit_adapter_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if source_projection not in ("clone", "shared"):
            raise ValueError("source_projection must be 'clone' or 'shared'")
        if source_projection_t not in ("clean", "current"):
            raise ValueError("source_projection_t must be 'clean' or 'current'")
        if source_projection == "shared" and train_source_projection:
            raise ValueError("The shared source projection is the frozen pretrained P_g; it cannot be trained")

        self.source_projection = source_projection
        self.source_projection_t = source_projection_t
        self.train_source_projection = bool(train_source_projection)
        self.source_gate_per_channel = bool(source_gate_per_channel)
        self.edit_adapter_path = edit_adapter_path

        # LoRA injection already freezes the base. Do the same for edit-only runs.
        if freeze_base and self.lora_rank <= 0:
            freeze_module(self)

        if source_projection == "clone":
            self.source_dino_proj = copy.deepcopy(self.dino_conditioner.dino_proj)
            self.source_dino_proj.requires_grad_(self.train_source_projection)
        else:
            self.source_dino_proj = None

        gate_width = self.hidden_size if self.source_gate_per_channel else 1
        self.source_gates = nn.Parameter(torch.zeros(self.depth, gate_width))
        self.source_final_gate = nn.Parameter(torch.zeros(gate_width))

        self.source_gates.requires_grad_(True)
        self.source_final_gate.requires_grad_(True)

        # The shared vision encoder remains frozen for both noisy and clean paths.
        self.dino_conditioner.encoder.requires_grad_(False)

        if edit_adapter_path:
            load_edit_adapter(self, edit_adapter_path, strict=True)

        self.edit_trainable_params, self.edit_total_params = lora_parameter_counts(self)

    def encode_source(self, source_image: torch.Tensor):
        """Encode a clean source once; samplers can cache these raw DINO tokens."""
        return self.dino_conditioner.encode(source_image)

    def _source_project(self, tokens, grid_hw: Tuple[int, int], t):
        source_t = torch.ones_like(t) if self.source_projection_t == "clean" else t
        projector = (
            self.source_dino_proj
            if self.source_dino_proj is not None
            else self.dino_conditioner.dino_proj
        )
        dtype = projector.input_proj.weight.dtype
        return projector(tokens.to(dtype=dtype), source_t, grid_hw)

    def _source_condition(
        self,
        t,
        source_image=None,
        source_tokens=None,
        source_grid: Optional[Tuple[int, int]] = None,
        source_drop_mask=None,
        source_strength=1.0,
    ):
        if source_tokens is None:
            if source_image is None:
                return None
            source_tokens, source_grid = self.encode_source(source_image)
        if source_grid is None:
            raise ValueError("source_grid is required when source_tokens are supplied")
        source = self._source_project(source_tokens, source_grid, t)
        if source_drop_mask is not None:
            source = source.masked_fill(source_drop_mask.view(-1, 1, 1).bool(), 0)
        if torch.is_tensor(source_strength):
            strength = source_strength.to(device=source.device, dtype=source.dtype)
            while strength.ndim < source.ndim:
                strength = strength.unsqueeze(-1)
            source = source * strength
        else:
            source = source * float(source_strength)
        return source

    def _gate(self, index: int, source: torch.Tensor):
        gate = self.source_gates[index]
        return gate.view(1, 1, -1).to(dtype=source.dtype, device=source.device)

    def _final_gate(self, source: torch.Tensor):
        return self.source_final_gate.view(1, 1, -1).to(
            dtype=source.dtype, device=source.device
        )

    def _activate_condition(self, c):
        return c if self.single_silu_cond else F.silu(c)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        skip_grounding: bool = False,
        skip_grounding_mask: Optional[torch.Tensor] = None,
        return_grounding_tokens: bool = False,
        source_image: Optional[torch.Tensor] = None,
        source_tokens: Optional[torch.Tensor] = None,
        source_grid: Optional[Tuple[int, int]] = None,
        source_drop_mask: Optional[torch.Tensor] = None,
        source_strength=1.0,
    ):
        batch, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input resolution ({height}, {width}) must be divisible by patch_size={self.patch_size}"
            )
        t = t.view(batch)
        hs, ws = height // self.patch_size, width // self.patch_size
        num_patches = hs * ws

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y).view(batch, 1, self.hidden_size)
        null_pre = (t_emb.unsqueeze(1) + y_emb).expand(-1, num_patches, -1)

        current_tokens = None
        if skip_grounding:
            base_pre = null_pre
        else:
            grounding = self.dino_conditioner(x, t)
            base_pre = grounding + t_emb.unsqueeze(1) + y_emb
            if skip_grounding_mask is not None:
                base_pre = torch.where(
                    skip_grounding_mask.view(batch, 1, 1).bool(), null_pre, base_pre
                )

        source = self._source_condition(
            t,
            source_image=source_image,
            source_tokens=source_tokens,
            source_grid=source_grid,
            source_drop_mask=source_drop_mask,
            source_strength=source_strength,
        )
        if source is not None and source.shape[1] != num_patches:
            raise ValueError(
                f"Source token grid has {source.shape[1]} patches but current image has {num_patches}. "
                "Source and target must use the same spatial patch grid."
            )

        s = self.s_embedder(
            F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        )
        use_context = self.in_context_len > 0
        pos = self.fetch_pos(hs, ws, x.device)
        pos_with_context = (
            self.fetch_pos(hs, ws, x.device, self.in_context_len) if use_context else None
        )

        context_inserted = False
        for i, block in enumerate(self.patch_blocks):
            if use_context and not context_inserted and i == self.in_context_start:
                context_tokens = y_emb.expand(-1, self.in_context_len, -1)
                s = torch.cat(
                    [context_tokens + self.in_context_posemb.to(s.dtype), s], dim=1
                )
                context_inserted = True

            block_pre = base_pre
            if source is not None:
                block_pre = block_pre + source * self._gate(i, source)
            block_c = self._activate_condition(block_pre)
            if context_inserted:
                context_c = block_c.mean(dim=1, keepdim=True).expand(
                    -1, self.in_context_len, -1
                )
                block_c = torch.cat([context_c, block_c], dim=1)
            s = block(
                s,
                block_c,
                pos_with_context if context_inserted else pos,
            )

        if context_inserted:
            s = s[:, self.in_context_len :, :]

        final_pre = base_pre
        if source is not None:
            final_pre = final_pre + source * self._final_gate(source)
        final_c = self._activate_condition(final_pre)
        out = self.final_layer(s, final_c)
        out = out.view(batch, num_patches, self.patch_size**2, self.out_channels)
        out = out.permute(0, 3, 2, 1).reshape(
            batch, self.out_channels * self.patch_size**2, num_patches
        )
        out = F.fold(
            out,
            (height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        if return_grounding_tokens:
            if current_tokens is None:
                current_tokens = self.dino_conditioner.encode(x)[0]
            return out, current_tokens
        return out


def PixelDiT2Edit_H_16(**kwargs):
    kwargs.setdefault("hidden_size", 1280)
    kwargs.setdefault("num_heads", 16)
    kwargs.setdefault("depth", 32)
    kwargs.setdefault("patch_size", 16)
    kwargs.setdefault("in_context_start", 8)
    return PixelDiT2Edit(**kwargs)
