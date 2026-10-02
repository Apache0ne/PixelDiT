"""Adapter serialization for PixelDiT2 image-edit models.

An edit adapter contains the transformer LoRA tensors plus the trainable clean-source
projection and source gates. The frozen PixelDiT2 base and DINO encoder are not
duplicated in the adapter file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from safetensors.torch import load_file, save_file


def _is_edit_parameter(name: str) -> bool:
    return (
        name.endswith(".lora_A")
        or name.endswith(".lora_B")
        or name.startswith("source_dino_proj.")
        or name in ("source_gates", "source_final_gate")
    )


def edit_state_dict(
    model,
    *,
    dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not _is_edit_parameter(name):
            continue
        tensor = parameter.detach().cpu().contiguous()
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        state[name] = tensor
    if not state:
        raise RuntimeError("No PixelDiT2 edit-adapter parameters were found")
    return state


def _metadata(model, extra: Optional[Mapping[str, object]] = None) -> Dict[str, str]:
    metadata = {
        "format": "pixeldit2-edit-adapter-v1",
        "model_class": type(model).__name__,
        "source_projection": str(getattr(model, "source_projection", "unknown")),
        "source_projection_t": str(getattr(model, "source_projection_t", "unknown")),
        "lora_rank": str(getattr(model, "lora_rank", 0)),
        "trainable_parameters": str(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }
    if extra:
        for key, value in extra.items():
            if value is not None:
                metadata[str(key)] = str(value)
    return metadata


def save_edit_adapter(
    model,
    path: str,
    *,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    metadata: Optional[Mapping[str, object]] = None,
) -> str:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(
        edit_state_dict(model, dtype=dtype),
        path,
        metadata=_metadata(model, metadata),
    )
    return path


def load_edit_adapter(model, path: str, strict: bool = True):
    state = load_file(os.fspath(path), device="cpu")
    named = dict(model.named_parameters())
    expected = {name for name in named if _is_edit_parameter(name)}
    loaded = []
    unexpected = []
    for name, value in state.items():
        parameter = named.get(name)
        if parameter is None or not _is_edit_parameter(name):
            unexpected.append(name)
            continue
        if tuple(parameter.shape) != tuple(value.shape):
            raise ValueError(
                f"Edit-adapter tensor shape mismatch for {name}: "
                f"adapter={tuple(value.shape)} model={tuple(parameter.shape)}"
            )
        parameter.data.copy_(
            value.to(device=parameter.device, dtype=parameter.dtype)
        )
        loaded.append(name)
    missing = sorted(expected.difference(loaded))
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Edit adapter does not exactly match this model. "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    return loaded


class EditAdapterCheckpoint(Callback):
    """Save EMA edit adapters without duplicating the frozen base."""

    def __init__(
        self,
        every_n_train_steps: int = 1000,
        save_dir: str = "edit_adapters",
        save_last: bool = True,
        save_dtype: str = "bf16",
    ):
        super().__init__()
        self.every_n_train_steps = int(every_n_train_steps)
        self.save_dir = save_dir
        self.save_last = bool(save_last)
        if save_dtype not in ("bf16", "fp32"):
            raise ValueError("save_dtype must be 'bf16' or 'fp32'")
        self.save_dtype = save_dtype
        self._last_saved_step = -1

    def _dtype(self):
        return torch.bfloat16 if self.save_dtype == "bf16" else torch.float32

    def _save(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        filename: str,
    ):
        model = getattr(pl_module, "ema_denoiser", None)
        if model is None:
            raise RuntimeError(
                "EditAdapterCheckpoint requires pl_module.ema_denoiser"
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out_dir = Path(trainer.default_root_dir) / self.save_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        save_edit_adapter(
            model,
            str(out_dir / filename),
            dtype=self._dtype(),
            metadata={
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "base_checkpoint": getattr(model, "pretrained_checkpoint", None),
            },
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        step = int(trainer.global_step)
        if self.every_n_train_steps <= 0 or step <= 0:
            return
        if (
            step % self.every_n_train_steps == 0
            and step != self._last_saved_step
        ):
            if trainer.is_global_zero:
                self._save(
                    trainer,
                    pl_module,
                    f"step-{step:08d}.safetensors",
                )
            self._last_saved_step = step

    def on_train_end(self, trainer, pl_module):
        if self.save_last and trainer.is_global_zero:
            self._save(trainer, pl_module, "last.safetensors")
