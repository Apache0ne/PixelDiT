#!/usr/bin/env python3
import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pixdit_core.pixeldit_c2i import PixDiT, PiTBlock
from pixdit_core.pixeldit2_c2i import PixelDiT2
from pixdit_core.pixeldit_t2i import PixDiT_T2I, MMDiTJointAttention
from pixdit_core.modules import RotaryAttention
import pixdit_core.modules as _pixdit_modules


def _install_sdpa_stub():
    stub = lambda q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False: v
    torch.nn.functional.scaled_dot_product_attention = stub
    _pixdit_modules.scaled_dot_product_attention = stub


class FlopsCounter:
    def __init__(self, model):
        self.total = 0
        self._handles = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear_hook))
            elif isinstance(m, nn.Conv2d):
                self._handles.append(m.register_forward_hook(self._conv_hook))
            elif isinstance(m, RotaryAttention):
                self._handles.append(m.register_forward_pre_hook(self._rotary_hook))
            elif isinstance(m, MMDiTJointAttention):
                self._handles.append(m.register_forward_pre_hook(self._mmdit_hook))
            elif "Attention" in type(m).__name__:
                # Attention inside the frozen timm grounding encoder.
                self._handles.append(m.register_forward_pre_hook(self._rotary_hook))

    def _linear_hook(self, mod, inp, out):
        x = inp[0]
        if not torch.is_tensor(x) or x.dim() == 0:
            return
        n = x.numel() // x.shape[-1]
        self.total += 2 * n * mod.in_features * mod.out_features

    def _conv_hook(self, mod, inp, out):
        if not torch.is_tensor(out):
            return
        per_output = mod.in_channels // mod.groups
        for k in mod.kernel_size:
            per_output *= k
        self.total += 2 * out.numel() * per_output

    def _rotary_hook(self, mod, inp):
        if not inp:
            return
        x = inp[0]
        if not torch.is_tensor(x) or x.dim() != 3:
            return
        B, N, C = x.shape
        self.total += 4 * B * N * N * C

    def _mmdit_hook(self, mod, inp):
        if not inp or len(inp) < 2:
            return
        x, y = inp[0], inp[1]
        if not (torch.is_tensor(x) and torch.is_tensor(y) and x.dim() == 3 and y.dim() == 3):
            return
        B, Nx, C = x.shape
        S = Nx + y.shape[1]
        self.total += 4 * B * S * S * C

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def count_with_dispatcher(model, x, t, y):
    """Count FLOPs for a conditional PixelDiT2 forward pass."""
    from torch.utils.flop_counter import FlopCounterMode

    original = torch.nn.functional.scaled_dot_product_attention

    def explicit(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        scale = (query.shape[-1] ** -0.5) if scale is None else scale
        logits = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_mask is not None:
            logits = logits + attn_mask
        return torch.matmul(torch.softmax(logits, dim=-1), value)

    torch.nn.functional.scaled_dot_product_attention = explicit
    _pixdit_modules.scaled_dot_product_attention = explicit
    try:
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(x, t, y)
        return int(counter.get_total_flops())
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
        _pixdit_modules.scaled_dot_product_attention = original


def build_model(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})

    if "denoiser" in model_cfg:
        init_args = dict(model_cfg["denoiser"].get("init_args", {}))
        class_path = model_cfg["denoiser"].get("class_path", "")
        if class_path.endswith("PixelDiT2"):
            init_args.setdefault("pretrained_encoder", False)
            return PixelDiT2(**init_args), "c2i-v2", cfg
        return PixDiT(**init_args), "c2i", cfg

    if "extra" in model_cfg:
        e = model_cfg["extra"]
        model = PixDiT_T2I(
            in_channels=3,
            num_groups=int(e.get("num_groups", 24)),
            hidden_size=int(e.get("hidden_size", 1536)),
            pixel_hidden_size=int(e.get("pixel_hidden_size", 16)),
            pixel_attn_hidden_size=e.get("pixel_attn_hidden_size"),
            pixel_num_groups=e.get("pixel_num_groups"),
            patch_depth=int(e.get("patch_depth", 14)),
            pixel_depth=int(e.get("pixel_depth", 2)),
            num_text_blocks=int(e.get("num_text_blocks", 4)),
            patch_size=int(e.get("patch_size", 16)),
            txt_embed_dim=int(e.get("txt_embed_dim", 2304)),
            txt_max_length=int(e.get("txt_max_length", 300)),
            use_text_rope=bool(e.get("use_text_rope", True)),
            text_rope_theta=float(e.get("text_rope_theta", 10000.0)),
        )
        return model, "t2i", cfg

    raise ValueError(f"Cannot detect model type from {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute GFLOPs for PixelDiT models (c2i / t2i).")
    parser.add_argument("--config", required=True, help="Path to a c2i or t2i YAML config.")
    parser.add_argument("--height", type=int, default=256, help="Input image height (default: 256).")
    parser.add_argument("--width", type=int, default=256, help="Input image width (default: 256).")
    args = parser.parse_args()

    model, mode, cfg = build_model(args.config)
    H, W = args.height, args.width
    params = sum(p.numel() for p in model.parameters())

    model.eval().to("cpu")
    x = torch.randn(1, 3, H, W)
    t = torch.tensor([0.5])

    if mode.startswith("c2i"):
        y = torch.zeros(1, dtype=torch.long)
    else:
        e = cfg["model"]["extra"]
        y = torch.randn(1, int(e.get("txt_max_length", 300)), int(e.get("txt_embed_dim", 2304)))

    if mode == "c2i-v2":
        gflops = count_with_dispatcher(model, x, t, y) / 1e9
    else:
        _install_sdpa_stub()
        with FlopsCounter(model) as fc, torch.no_grad():
            model(x, t, y)
            gflops = fc.total / 1e9

    print(f"Model:      {'PixelDiT2' if mode == 'c2i-v2' else 'PixelDiT'} ({mode})")
    print(f"Config:     {os.path.basename(args.config)}")
    print(f"Parameters: {params / 1e6:.1f}M")
    print(f"Resolution: {H} x {W}")
    print(f"GFLOPs:     {gflops:.9f}")


if __name__ == "__main__":
    main()
