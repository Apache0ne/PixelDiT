"""Colab validation for the PixelDiT2Edit branch."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
REPORT = Path("/content/pixeldit2_edit_test_report")
ZIP = Path("/content/pixeldit2_edit_test_report.zip")
REPORT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "c2i"))

from pixdit_core.edit_adapter import edit_state_dict, save_edit_adapter
from pixdit_core.pixeldit2_edit import PixelDiT2Edit
from c2i.src.edit_diffusion import EditGroundedEulerSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS, DETAILS = [], {}


def test(name, fn):
    t0 = time.time()
    try:
        DETAILS[name] = fn() or {}
        RESULTS.append(
            {"name": name, "status": "PASS", "seconds": time.time() - t0}
        )
        print("[PASS]", name)
    except Exception as exc:
        tb = traceback.format_exc()
        RESULTS.append(
            {
                "name": name,
                "status": "FAIL",
                "seconds": time.time() - t0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        DETAILS[name] = {"traceback": tb}
        (REPORT / ("FAIL_" + name.replace(" ", "_") + ".txt")).write_text(tb)
        print("[FAIL]", name, "\n", tb)


def tiny(**extra):
    cfg = dict(
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
        lora_rank=4,
        lora_alpha=4.0,
        source_projection="clone",
        source_projection_t="clean",
        train_source_projection=True,
        freeze_base=True,
    )
    cfg.update(extra)
    return cfg


def static_test():
    files = [
        "pixdit_core/pixeldit2_edit.py",
        "pixdit_core/edit_adapter.py",
        "c2i/src/edit_diffusion.py",
        "c2i/src/edit_data.py",
        "c2i/src/edit_lightning.py",
        "c2i/main_edit.py",
    ]
    p = subprocess.run(
        [sys.executable, "-m", "py_compile", *files], cwd=ROOT
    )
    assert p.returncode == 0
    p = subprocess.run(["bash", "-n", "c2i/train_edit.sh"], cwd=ROOT)
    assert p.returncode == 0
    cfg = yaml.safe_load(
        (ROOT / "c2i/configs/pixeldit2_h16_in256_edit_lora.yaml").read_text()
    )
    args = cfg["model"]["denoiser"]["init_args"]
    assert args["lora_target_modules"] == [
        "patch_blocks.*.attn.qkv",
        "patch_blocks.*.attn.proj",
        "patch_blocks.*.mlp.w1",
        "patch_blocks.*.mlp.w2",
        "patch_blocks.*.mlp.w3",
    ]
    assert args["source_projection"] == "clone"
    assert args["train_source_projection"] is True
    return {"files": files}


test("static source and config", static_test)

holder = {}


def equivalence_test():
    torch.manual_seed(123)
    m = PixelDiT2Edit(**tiny()).to(DEVICE).eval()
    with torch.no_grad():
        m.final_layer.linear.weight.normal_(0, 0.01)
        for block in m.patch_blocks:
            block.adaLN_modulation[-1].weight.normal_(0, 0.01)
            block.adaLN_modulation[-1].bias.normal_(0, 0.01)
    x = torch.randn(1, 3, 32, 32, device=DEVICE)
    src = torch.randn_like(x)
    t = torch.tensor([0.5], device=DEVICE)
    y = torch.tensor([3], device=DEVICE)
    with torch.no_grad():
        parent = super(PixelDiT2Edit, m).forward(
            x, t, y, skip_grounding=True
        )
        no_source = m(x, t, y, skip_grounding=True)
        zero_source = m(
            x, t, y, skip_grounding=True, source_image=src
        )
    assert torch.equal(parent, no_source)
    assert torch.equal(parent, zero_source)
    assert torch.count_nonzero(m.source_gates).item() == 0
    assert torch.count_nonzero(m.source_final_gate).item() == 0
    holder.update(m=m, x=x, src=src, t=t, y=y)
    return {"max_error": float((parent - zero_source).abs().max().cpu())}


test("zero-gate exact base equivalence", equivalence_test)


def isolation_test():
    m = holder["m"]
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    allowed = all(
        n.endswith(".lora_A")
        or n.endswith(".lora_B")
        or n.startswith("source_dino_proj.")
        or n in ("source_gates", "source_final_gate")
        for n in trainable
    )
    assert allowed
    assert all(
        not p.requires_grad for p in m.dino_conditioner.encoder.parameters()
    )
    return {
        "trainable_parameters": sum(
            p.numel() for p in m.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in m.parameters()),
        "dino_frozen": True,
    }


test("trainable isolation and frozen DINO", isolation_test)


def two_step_gradient_test():
    m = holder["m"].train()
    x, src, t, y = holder["x"], holder["src"], holder["t"], holder["y"]
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=1e-2
    )

    opt.zero_grad(True)
    loss1 = m(
        x, t, y, skip_grounding=True, source_image=src
    ).float().square().mean()
    loss1.backward()
    gate_grad1 = float(m.source_final_gate.grad.float().norm().cpu())
    proj_grad1 = sum(
        float(p.grad.float().norm().cpu())
        for p in m.source_dino_proj.parameters()
        if p.grad is not None
    )
    assert gate_grad1 > 0
    assert proj_grad1 == 0.0
    opt.step()

    opt.zero_grad(True)
    loss2 = m(
        x, t, y, skip_grounding=True, source_image=src
    ).float().square().mean()
    loss2.backward()
    proj_grad2 = sum(
        float(p.grad.float().norm().cpu())
        for p in m.source_dino_proj.parameters()
        if p.grad is not None
    )
    assert proj_grad2 > 0
    opt.step()
    return {
        "step1_gate_grad": gate_grad1,
        "step1_source_projector_grad": proj_grad1,
        "step2_source_projector_grad": proj_grad2,
    }


test(
    "zero-gate bootstrap then source-projector gradients",
    two_step_gradient_test,
)


def adapter_test():
    m = holder["m"]
    path = REPORT / "tiny_edit_adapter_fp32.safetensors"
    save_edit_adapter(m, str(path), dtype=torch.float32)
    fresh = PixelDiT2Edit(
        **tiny(edit_adapter_path=str(path))
    ).to(DEVICE)
    a = edit_state_dict(m, dtype=torch.float32)
    b = edit_state_dict(fresh, dtype=torch.float32)
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    return {"tensors": len(a), "size_mib": path.stat().st_size / 2**20}


test("edit adapter strict round-trip", adapter_test)


def sampler_test():
    m = holder["m"].eval()
    counter = {"n": 0}
    original = m.encode_source

    def counted(source):
        counter["n"] += 1
        return original(source)

    m.encode_source = counted
    sampler = EditGroundedEulerSampler(
        num_steps=2,
        source_guidance=1.0,
        edit_guidance=1.0,
        t_start=0.6,
        guidance_interval_min=0.0,
        guidance_interval_max=1.0,
        noise_scale=1.0,
        t_eps=0.05,
    ).to(DEVICE)
    noise = torch.randn_like(holder["x"])
    cond = holder["y"]
    uncond = torch.tensor([10], device=DEVICE)
    with torch.no_grad():
        out = sampler(
            m,
            noise,
            cond,
            uncond,
            source_image=holder["src"],
        )
    m.encode_source = original
    assert counter["n"] == 1
    assert out.shape == noise.shape and torch.isfinite(out).all()
    return {"source_encode_calls": counter["n"], "shape": list(out.shape)}


test("edit sampler caches source DINO once", sampler_test)

passed = sum(r["status"] == "PASS" for r in RESULTS)
summary = {
    "totals": {
        "tests": len(RESULTS),
        "passed": passed,
        "failed": len(RESULTS) - passed,
    },
    "results": RESULTS,
    "details": DETAILS,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
(REPORT / "summary.json").write_text(json.dumps(summary, indent=2))
(REPORT / "REPORT.md").write_text(
    f"# PixelDiT2Edit Colab Test\n\n"
    f"**{passed} PASS / {len(RESULTS)-passed} FAIL / {len(RESULTS)} total**\n"
)
if ZIP.exists():
    ZIP.unlink()
shutil.make_archive(str(ZIP.with_suffix("")), "zip", REPORT)
print(
    f"FINAL: {passed} PASS / {len(RESULTS)-passed} FAIL / {len(RESULTS)} TOTAL"
)
print("ZIP:", ZIP)
try:
    from google.colab import files

    files.download(str(ZIP))
except Exception:
    pass
