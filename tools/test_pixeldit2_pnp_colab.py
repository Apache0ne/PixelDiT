"""Colab regression tests for PixelDiT2 PnP structural-attention editing."""
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
REPORT = Path("/content/pixeldit2_pnp_test_report")
ZIP = Path("/content/pixeldit2_pnp_test_report.zip")
REPORT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "c2i"))

from pixdit_core.pixeldit2_c2i import PixelDiT2
from c2i.src.pnp_pixeldit2 import PixelDiT2PnPEditSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS, DETAILS = [], {}


def test(name, fn):
    t0 = time.time()
    try:
        DETAILS[name] = fn() or {}
        RESULTS.append({"name": name, "status": "PASS", "seconds": time.time() - t0})
        print("[PASS]", name)
    except Exception as exc:
        tb = traceback.format_exc()
        RESULTS.append({
            "name": name,
            "status": "FAIL",
            "seconds": time.time() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        })
        DETAILS[name] = {"traceback": tb}
        (REPORT / ("FAIL_" + name.replace(" ", "_") + ".txt")).write_text(tb)
        print("[FAIL]", name, "\n", tb)


def tiny_model():
    torch.manual_seed(7)
    model = PixelDiT2(
        in_channels=3,
        hidden_size=192,
        num_heads=3,
        depth=3,
        patch_size=16,
        num_classes=10,
        in_context_len=2,
        in_context_start=1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        grounding_encoder="dinov3_vits16",
        grounding_proj_blocks=1,
        grounding_proj_heads=3,
        pretrained_encoder=False,
        lora_rank=0,
    ).to(DEVICE).eval()
    # PixelDiT initializes residual modulation/final projection at zero for training.
    # Give the tiny test model nonzero deterministic weights so control paths can be
    # observed without changing the architecture being tested.
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(0, 0.01)
        model.final_layer.adaLN_modulation[-1].weight.normal_(0, 0.01)
        model.final_layer.adaLN_modulation[-1].bias.normal_(0, 0.01)
        for block in model.patch_blocks:
            block.adaLN_modulation[-1].weight.normal_(0, 0.01)
            block.adaLN_modulation[-1].bias.normal_(0, 0.01)
    return model


def static_test():
    files = [
        "c2i/src/pnp_pixeldit2.py",
        "c2i/src/edit_data.py",
        "c2i/src/edit_lightning.py",
    ]
    p = subprocess.run([sys.executable, "-m", "py_compile", *files], cwd=ROOT)
    assert p.returncode == 0
    cfg = yaml.safe_load(
        (ROOT / "c2i/configs/pixeldit2_h16_in256_pnp_edit.yaml").read_text()
    )
    assert cfg["model"]["denoiser"]["class_path"] == "pixdit_core.pixeldit2_c2i.PixelDiT2"
    assert cfg["model"]["denoiser"]["init_args"]["lora_rank"] == 0
    assert cfg["model"]["diffusion_sampler"]["class_path"] == "src.pnp_pixeldit2.PixelDiT2PnPEditSampler"
    return {"files": files}


test("static source and clean PnP config", static_test)

holder = {}


def setup_test():
    model = tiny_model()
    sampler = PixelDiT2PnPEditSampler(
        num_steps=3,
        guidance=2.4,
        qk_injection_strength=1.0,
        qk_injection_until=0.75,
        qk_fade_start=0.4,
        qk_block_start=0,
        qk_block_end=3,
        solver="euler",
    ).to(DEVICE)
    torch.manual_seed(11)
    source = torch.randn(1, 3, 32, 32, device=DEVICE)
    target = torch.randn_like(source)
    noise = torch.randn_like(source)
    t = torch.tensor([0.5], device=DEVICE)
    source_y = torch.tensor([3], device=DEVICE)
    target_y = torch.tensor([6], device=DEVICE)
    uncond = torch.tensor([10], device=DEVICE)
    holder.update(
        model=model,
        sampler=sampler,
        source=source,
        target=target,
        noise=noise,
        t=t,
        source_y=source_y,
        target_y=target_y,
        uncond=uncond,
    )
    return {"device": str(DEVICE)}


test("tiny PnP setup", setup_test)


def zero_injection_equivalence():
    m = holder["model"]
    s = holder["sampler"]
    x_src, x_tar, t = holder["source"], holder["target"], holder["t"]
    sy, ty, uy = holder["source_y"], holder["target_y"], holder["uncond"]
    with torch.no_grad():
        ref_c = m(x_tar, t, ty)
        ref_u = m(x_tar, t, uy, skip_grounding=True)
        got_c, got_u = s._pnp_forward(
            m, x_src, x_tar, t, sy, ty, uy, torch.zeros(1, device=DEVICE)
        )
    err_c = float((ref_c - got_c).abs().max().cpu())
    err_u = float((ref_u - got_u).abs().max().cpu())
    assert torch.allclose(ref_c, got_c, atol=1e-5, rtol=1e-5), err_c
    assert torch.allclose(ref_u, got_u, atol=1e-5, rtol=1e-5), err_u
    return {"conditional_max_error": err_c, "unconditional_max_error": err_u}


test("zero QK injection equals base PixelDiT2", zero_injection_equivalence)


def source_qk_changes_target():
    m = holder["model"]
    s = holder["sampler"]
    x_src, x_tar, t = holder["source"], holder["target"], holder["t"]
    sy, ty, uy = holder["source_y"], holder["target_y"], holder["uncond"]
    with torch.no_grad():
        base, _ = s._pnp_forward(
            m, x_src, x_tar, t, sy, ty, uy, torch.zeros(1, device=DEVICE)
        )
        controlled, _ = s._pnp_forward(
            m, x_src, x_tar, t, sy, ty, uy, torch.ones(1, device=DEVICE)
        )
    delta = float((controlled - base).abs().mean().cpu())
    assert delta > 1e-8
    assert torch.isfinite(controlled).all()
    return {"mean_abs_control_delta": delta}


test("source QK changes target while staying finite", source_qk_changes_target)


def source_image_matters_only_when_controlled():
    m = holder["model"]
    s = holder["sampler"]
    x_src, x_tar, t = holder["source"], holder["target"], holder["t"]
    sy, ty, uy = holder["source_y"], holder["target_y"], holder["uncond"]
    other = torch.flip(x_src, dims=[-1])
    with torch.no_grad():
        a0, _ = s._pnp_forward(m, x_src, x_tar, t, sy, ty, uy, torch.zeros(1, device=DEVICE))
        b0, _ = s._pnp_forward(m, other, x_tar, t, sy, ty, uy, torch.zeros(1, device=DEVICE))
        a1, _ = s._pnp_forward(m, x_src, x_tar, t, sy, ty, uy, torch.ones(1, device=DEVICE))
        b1, _ = s._pnp_forward(m, other, x_tar, t, sy, ty, uy, torch.ones(1, device=DEVICE))
    zero_err = float((a0 - b0).abs().max().cpu())
    controlled_delta = float((a1 - b1).abs().mean().cpu())
    assert zero_err <= 1e-5
    assert controlled_delta > 1e-8
    return {"zero_control_source_error": zero_err, "controlled_source_delta": controlled_delta}


test("source affects output only through PnP control", source_image_matters_only_when_controlled)


def end_to_end_sampler():
    m = holder["model"]
    s = holder["sampler"]
    before = {n: p.detach().cpu().clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        out = s(
            m,
            holder["noise"],
            holder["target_y"],
            holder["uncond"],
            source_image=holder["source"],
            source_condition=torch.tensor([-1], device=DEVICE),
            source_strength=torch.tensor([1.0], device=DEVICE),
            edit_strength=torch.tensor([1.0], device=DEVICE),
            t_start=0.72,
        )
    assert out.shape == holder["noise"].shape
    assert torch.isfinite(out).all()
    changed = [
        n for n, p in m.named_parameters()
        if not torch.equal(before[n], p.detach().cpu())
    ]
    assert not changed, changed[:10]
    return {"shape": list(out.shape), "diagnostics": s.last_diagnostics, "changed_parameters": changed}


test("end-to-end PnP edit leaves model frozen", end_to_end_sampler)

passed = sum(r["status"] == "PASS" for r in RESULTS)
summary = {
    "totals": {"tests": len(RESULTS), "passed": passed, "failed": len(RESULTS) - passed},
    "results": RESULTS,
    "details": DETAILS,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
(REPORT / "summary.json").write_text(json.dumps(summary, indent=2))
(REPORT / "REPORT.md").write_text(
    f"# PixelDiT2 PnP Edit Test\n\n**{passed} PASS / {len(RESULTS)-passed} FAIL / {len(RESULTS)} total**\n"
)
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
