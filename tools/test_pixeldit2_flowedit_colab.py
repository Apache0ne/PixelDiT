"""Focused validation for PixelDiT2's training-free FlowEdit transport."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
REPORT = Path("/content/pixeldit2_flowedit_test_report")
ZIP = Path("/content/pixeldit2_flowedit_test_report.zip")
REPORT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "c2i"))

from pixdit_core.pixeldit2_edit import PixelDiT2Edit
from c2i.src.flowedit_pixeldit2 import PixelDiT2FlowEditSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = []
DETAILS = {}


def test(name, fn):
    try:
        DETAILS[name] = fn() or {}
        RESULTS.append({"name": name, "status": "PASS"})
        print("[PASS]", name)
    except Exception as exc:
        tb = traceback.format_exc()
        RESULTS.append(
            {"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        )
        DETAILS[name] = {"traceback": tb}
        print("[FAIL]", name, "\n", tb)


def tiny():
    return dict(
        in_channels=3,
        hidden_size=192,
        num_heads=3,
        depth=2,
        patch_size=16,
        num_classes=10,
        in_context_len=0,
        in_context_start=0,
        grounding_encoder="dinov3_vits16",
        grounding_proj_blocks=1,
        grounding_proj_heads=3,
        pretrained_encoder=False,
        lora_rank=0,
        source_projection="clone",
        source_projection_t="clean",
        train_source_projection=False,
        freeze_base=True,
    )


def static_test():
    files = [
        "c2i/src/flowedit_pixeldit2.py",
        "c2i/src/edit_data.py",
        "c2i/src/edit_lightning.py",
    ]
    p = subprocess.run([sys.executable, "-m", "py_compile", *files], cwd=ROOT)
    assert p.returncode == 0
    cfg = yaml.safe_load(
        (ROOT / "c2i/configs/pixeldit2_h16_in256_edit_lora.yaml").read_text()
    )
    sampler = cfg["model"]["diffusion_sampler"]
    assert sampler["class_path"] == "src.flowedit_pixeldit2.PixelDiT2FlowEditSampler"
    assert sampler["init_args"]["localization"] in ("none", "velocity_prior")
    return {"sampler": sampler}


test("static FlowEdit source/config", static_test)

holder = {}


def model_test():
    torch.manual_seed(17)
    model = PixelDiT2Edit(**tiny()).to(DEVICE).eval()
    # Make the tiny randomly initialized model non-degenerate for conditioning tests.
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(0.0, 0.01)
        for block in model.patch_blocks:
            block.adaLN_modulation[-1].weight.normal_(0.0, 0.01)
            block.adaLN_modulation[-1].bias.normal_(0.0, 0.01)
    source = torch.randn(1, 3, 32, 32, device=DEVICE).clamp(-1, 1)
    noise = torch.randn_like(source)
    holder.update(model=model, source=source, noise=noise)
    assert all(not p.requires_grad for p in model.dino_conditioner.encoder.parameters())
    return {"dino_frozen": True}


test("tiny edit model and frozen DINO", model_test)


def identity_transport_test():
    model = holder["model"]
    source = holder["source"]
    noise = holder["noise"]
    cond = torch.tensor([3], device=DEVICE)
    uncond = torch.tensor([10], device=DEVICE)
    sampler = PixelDiT2FlowEditSampler(
        num_steps=3,
        n_avg=1,
        source_guidance=1.0,
        target_guidance=1.0,
        edit_t_min=0.25,
        edit_t_max=0.80,
        localization="none",
        terminal_steps=0,
    ).to(DEVICE)
    out = sampler(
        model,
        noise,
        cond,
        uncond,
        source_image=source,
        source_condition=cond,
    )
    error = float((out.float() - source.float()).abs().max().cpu())
    # Source and target vector fields are exactly the same, so the transport
    # increment must vanish apart from mixed-precision roundoff.
    assert error < 2e-3, error
    return {"max_identity_error": error, "diagnostics": sampler.last_diagnostics}


test("same-class FlowEdit identity transport", identity_transport_test)


def target_delta_test():
    model = holder["model"]
    source = holder["source"]
    noise = holder["noise"]
    src = torch.tensor([3], device=DEVICE)
    tar = torch.tensor([4], device=DEVICE)
    uncond = torch.tensor([10], device=DEVICE)
    sampler = PixelDiT2FlowEditSampler(
        num_steps=3,
        n_avg=1,
        source_guidance=1.0,
        target_guidance=2.0,
        edit_t_min=0.25,
        edit_t_max=0.80,
        localization="velocity_prior",
        localization_strength=0.75,
        terminal_steps=0,
    ).to(DEVICE)
    out = sampler(
        model,
        noise,
        tar,
        uncond,
        source_image=source,
        source_condition=src,
    )
    delta = float((out.float() - source.float()).square().mean().sqrt().cpu())
    assert torch.isfinite(out).all()
    assert delta > 0.0
    diag = sampler.last_diagnostics
    assert 0.0 < diag["mean_localization_mask"] <= 1.0
    return {"output_delta_rms": delta, "diagnostics": diag}


test("different-class transport plus localization", target_delta_test)


def image_only_source_test():
    model = holder["model"]
    source = holder["source"]
    noise = holder["noise"]
    tar = torch.tensor([4], device=DEVICE)
    uncond = torch.tensor([10], device=DEVICE)
    sampler = PixelDiT2FlowEditSampler(
        num_steps=2,
        edit_t_min=0.30,
        edit_t_max=0.75,
        localization="none",
    ).to(DEVICE)
    out = sampler(
        model,
        noise,
        tar,
        uncond,
        source_image=source,
        source_condition=torch.tensor([-1], device=DEVICE),
    )
    assert out.shape == source.shape and torch.isfinite(out).all()
    return {"shape": list(out.shape)}


test("image-only source-class fallback", image_only_source_test)

passed = sum(x["status"] == "PASS" for x in RESULTS)
summary = {
    "totals": {"tests": len(RESULTS), "passed": passed, "failed": len(RESULTS) - passed},
    "results": RESULTS,
    "details": DETAILS,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
(REPORT / "summary.json").write_text(json.dumps(summary, indent=2))
if ZIP.exists():
    ZIP.unlink()
shutil.make_archive(str(ZIP.with_suffix("")), "zip", REPORT)
print(f"FINAL: {passed} PASS / {len(RESULTS)-passed} FAIL / {len(RESULTS)} TOTAL")
print("ZIP:", ZIP)
try:
    from google.colab import files
    files.download(str(ZIP))
except Exception:
    pass
