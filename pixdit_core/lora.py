"""Lightweight LoRA support for PixelDiT2.

The implementation is intentionally self-contained so PixelDiT2 LoRA training does
not depend on PEFT. LoRA adapters are injected into selected ``nn.Linear``
modules while the pretrained base weights remain frozen. Adapter-only weights
can be saved as safetensors and loaded back on top of the same base checkpoint.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from safetensors.torch import load_file, save_file


DEFAULT_PIXELDIT2_LORA_TARGETS: Tuple[str, ...] = (
    "patch_blocks.*.attn.qkv",
    "patch_blocks.*.attn.proj",
    "patch_blocks.*.mlp.w1",
    "patch_blocks.*.mlp.w2",
    "patch_blocks.*.mlp.w3",
)


class LoRALinear(nn.Module):
    """LoRA residual around a frozen ``nn.Linear`` layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout_p = float(dropout)
        self.lora_dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        self.base_layer.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        adapter_input = self.lora_dropout(x).to(self.lora_A.dtype)
        adapter = F.linear(F.linear(adapter_input, self.lora_A), self.lora_B)
        return base + adapter.to(base.dtype) * self.scaling


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _parent_and_child(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def freeze_module(module: Optional[nn.Module]) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = False


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Optional[Sequence[str]] = None,
    exclude_modules: Optional[Sequence[str]] = None,
    modules_to_save: Optional[Sequence[str]] = None,
) -> List[str]:
    """Freeze ``model`` and replace matching linear layers with LoRA wrappers.

    Patterns use shell-style glob matching against full module names, for example
    ``patch_blocks.*.attn.qkv``.
    """

    targets = tuple(target_modules or DEFAULT_PIXELDIT2_LORA_TARGETS)
    excludes = tuple(exclude_modules or ())
    save_patterns = tuple(modules_to_save or ())

    freeze_module(model)

    selected: List[str] = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if isinstance(module, LoRALinear):
            continue
        if not _matches(name, targets) or (excludes and _matches(name, excludes)):
            continue
        parent, child = _parent_and_child(model, name)
        setattr(parent, child, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        selected.append(name)

    if not selected:
        raise ValueError(
            "LoRA did not match any nn.Linear modules. "
            f"target_modules={list(targets)!r}, exclude_modules={list(excludes)!r}"
        )

    if save_patterns:
        for name, module in model.named_modules():
            if name and _matches(name, save_patterns):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    return selected


def lora_parameter_counts(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _adapter_metadata(model: nn.Module, extra: Optional[Mapping[str, object]] = None) -> Dict[str, str]:
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            modules.append(
                {
                    "name": name,
                    "rank": module.rank,
                    "alpha": module.alpha,
                    "dropout": module.dropout_p,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                }
            )
    metadata: Dict[str, str] = {
        "format": "pixeldit2-lora-v1",
        "modules": json.dumps(modules, separators=(",", ":")),
    }
    if extra:
        for key, value in extra.items():
            if value is not None:
                metadata[str(key)] = str(value)
    return metadata


def lora_state_dict(
    model: nn.Module,
    *,
    dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    """Return adapter-only tensors using the model's module names."""

    state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        for leaf in ("lora_A", "lora_B"):
            tensor = getattr(module, leaf).detach().cpu().contiguous()
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            state[f"{name}.{leaf}"] = tensor
    if not state:
        raise RuntimeError("No LoRA parameters found while creating adapter state_dict")
    return state


def save_lora_adapter(
    model: nn.Module,
    path: str,
    *,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    metadata: Optional[Mapping[str, object]] = None,
) -> str:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(
        lora_state_dict(model, dtype=dtype),
        path,
        metadata=_adapter_metadata(model, metadata),
    )
    return path


def load_lora_adapter(model: nn.Module, path: str, strict: bool = True) -> List[str]:
    """Load an adapter into an already LoRA-injected model."""

    state = load_file(os.fspath(path), device="cpu")
    named_parameters = dict(model.named_parameters())
    loaded: List[str] = []
    unexpected: List[str] = []

    for key, value in state.items():
        parameter = named_parameters.get(key)
        if parameter is None:
            unexpected.append(key)
            continue
        if tuple(parameter.shape) != tuple(value.shape):
            raise ValueError(
                f"LoRA tensor shape mismatch for {key}: adapter={tuple(value.shape)} "
                f"model={tuple(parameter.shape)}"
            )
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        loaded.append(key)

    expected = {
        name
        for name in named_parameters
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    }
    missing = sorted(expected.difference(loaded))
    if strict and (missing or unexpected):
        raise RuntimeError(
            "LoRA adapter does not exactly match the injected model. "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    return loaded


class LoRAAdapterCheckpoint(Callback):
    """Save small EMA LoRA adapter files alongside regular Lightning checkpoints."""

    def __init__(
        self,
        every_n_train_steps: int = 1000,
        save_dir: str = "lora_adapters",
        save_last: bool = True,
        save_dtype: str = "bf16",
    ) -> None:
        super().__init__()
        self.every_n_train_steps = int(every_n_train_steps)
        self.save_dir = save_dir
        self.save_last = bool(save_last)
        if save_dtype not in ("bf16", "fp32"):
            raise ValueError("save_dtype must be 'bf16' or 'fp32'")
        self.save_dtype = save_dtype
        self._last_saved_step = -1

    def _dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.save_dtype == "bf16" else torch.float32

    def _save(self, trainer: Trainer, pl_module: LightningModule, filename: str) -> None:
        model = getattr(pl_module, "ema_denoiser", None)
        if model is None:
            raise RuntimeError("LoRAAdapterCheckpoint requires pl_module.ema_denoiser")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out_dir = Path(trainer.default_root_dir) / self.save_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        save_lora_adapter(
            model,
            str(out_dir / filename),
            dtype=self._dtype(),
            metadata={
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "base_checkpoint": getattr(pl_module, "pretrained_checkpoint", None),
            },
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if self.every_n_train_steps <= 0 or step <= 0:
            return
        if step % self.every_n_train_steps == 0 and step != self._last_saved_step:
            if trainer.is_global_zero:
                self._save(trainer, pl_module, f"step-{step:08d}.safetensors")
            self._last_saved_step = step

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.save_last and trainer.is_global_zero:
            self._save(trainer, pl_module, "last.safetensors")
