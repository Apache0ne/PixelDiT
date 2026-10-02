"""Real PixelDiT2 H/16 256 LoRA training-step validation for Google Colab L4.

This is intentionally heavier than the unit validator. It downloads the released
NVIDIA H/16 256 checkpoint, loads the actual pretrained DINOv3 grounding model,
runs one real flow-matching training step through the LoRA-only model, exercises
SimpleEMA, verifies every frozen parameter remained bit-identical, saves adapters,
reloads a fresh model, compares outputs, then writes a downloadable report ZIP.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
REPORT = Path("/content/pixeldit2_lora_real_h16_report")
ZIP = Path("/content/pixeldit2_lora_real_h16_report.zip")
REPORT.mkdir(parents=True, exist_ok=True)
LOG = REPORT / "full.log"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "c2i"))

from huggingface_hub import HfApi, hf_hub_download
from pixdit_core.lora import lora_parameter_counts, save_lora_adapter
from pixdit_core.pixeldit2_c2i import PixelDiT2
from src.lora_diffusion import LoRAGroundedFlowTrainer
from src.utils import SimpleEMA, no_grad

DEVICE = torch.device("cuda")
CHECKPOINT_REPO = "nvidia/PixelDiT2-ImageNet"
CHECKPOINT_NAME = "pixeldit2_h16_in256_ep600.ckpt"
EXPECTED_LORA_PARAMS = 11_140_608
NULL_CLASS = 1000
TEST_CLASS = 207
RESULTS = []
DETAILS = {}


def log(msg=""):
    msg = str(msg)
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def stage(name, fn):
    log("\n" + "=" * 100)
    log(name)
    log("=" * 100)
    t0 = time.perf_counter()
    try:
        detail = fn()
        dt = time.perf_counter() - t0
        RESULTS.append({"name": name, "status": "PASS", "seconds": round(dt, 3)})
        DETAILS[name] = detail if detail is not None else {}
        log(f"[PASS] {name} ({dt:.3f}s)")
        return detail
    except Exception as exc:
        dt = time.perf_counter() - t0
        tb = traceback.format_exc()
        RESULTS.append({
            "name": name,
            "status": "FAIL",
            "seconds": round(dt, 3),
            "error": f"{type(exc).__name__}: {exc}",
        })
        DETAILS[name] = {"traceback": tb}
        (REPORT / ("FAIL_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".txt")).write_text(tb)
        log(tb)
        raise


def run_cmd(args, cwd=ROOT, timeout=300):
    p = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout


def model_kwargs(**extra):
    kw = dict(
        in_channels=3,
        hidden_size=1280,
        num_heads=16,
        depth=32,
        patch_size=16,
        num_classes=1000,
        in_context_len=32,
        in_context_start=8,
        attn_dropout=0.0,
        proj_dropout=0.2,
        mlp_hidden_multiple=0,
        grounding_encoder="dinov3_vits16",
        grounding_proj_blocks=1,
        grounding_proj_heads=16,
        single_silu_cond=False,
        pretrained_encoder=True,
        pretrained_weights="ema",
        lora_rank=16,
        lora_alpha=16.0,
        lora_dropout=0.0,
    )
    kw.update(extra)
    return kw


def frozen_digest(model):
    """SHA256 over every frozen parameter name, shape, dtype, and raw tensor bytes."""
    h = hashlib.sha256()
    tensors = 0
    elements = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            continue
        q = p.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(q.shape)).encode("ascii"))
        h.update(str(q.dtype).encode("ascii"))
        h.update(q.numpy().tobytes())
        tensors += 1
        elements += q.numel()
        del q
    return h.hexdigest(), tensors, elements


def adapter_snapshot(model):
    return {
        name: p.detach().float().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def adapter_delta(before, model):
    changed = 0
    total = 0
    sq = 0.0
    max_abs = 0.0
    by_kind = defaultdict(lambda: {"tensors": 0, "changed": 0, "sq": 0.0, "max_abs": 0.0})
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        d = p.detach().float().cpu() - before[name]
        n = d.numel()
        a = float(d.abs().max()) if n else 0.0
        s = float(d.square().sum())
        is_changed = bool(torch.count_nonzero(d).item())
        changed += int(is_changed)
        total += 1
        sq += s
        max_abs = max(max_abs, a)
        kind = "A" if name.endswith("lora_A") else "B" if name.endswith("lora_B") else "other"
        by_kind[kind]["tensors"] += 1
        by_kind[kind]["changed"] += int(is_changed)
        by_kind[kind]["sq"] += s
        by_kind[kind]["max_abs"] = max(by_kind[kind]["max_abs"], a)
    out = {
        "tensors": total,
        "changed_tensors": changed,
        "l2": sq ** 0.5,
        "max_abs": max_abs,
        "by_kind": {},
    }
    for k, v in by_kind.items():
        out["by_kind"][k] = {
            "tensors": v["tensors"], "changed": v["changed"],
            "l2": v["sq"] ** 0.5, "max_abs": v["max_abs"],
        }
    return out


def grad_stats(model):
    global_sq = 0.0
    finite = True
    nonzero = 0
    tensors = 0
    block_sq = defaultdict(float)
    block_nonzero = defaultdict(int)
    kind_sq = defaultdict(float)
    kind_nonzero = defaultdict(int)
    rows = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        tensors += 1
        if p.grad is None:
            rows.append({"name": name, "grad": None})
            continue
        g = p.grad.detach().float()
        n = float(g.norm().cpu())
        mx = float(g.abs().max().cpu())
        ok = bool(torch.isfinite(g).all().item())
        finite &= ok
        nz = int(torch.count_nonzero(g).item() > 0)
        nonzero += nz
        global_sq += n * n
        m = re.match(r"patch_blocks\.(\d+)\.", name)
        if m:
            b = int(m.group(1))
            block_sq[b] += n * n
            block_nonzero[b] += nz
        kind = "A" if name.endswith("lora_A") else "B" if name.endswith("lora_B") else "other"
        kind_sq[kind] += n * n
        kind_nonzero[kind] += nz
        rows.append({"name": name, "norm": n, "max_abs": mx, "finite": ok})
    return {
        "trainable_tensors": tensors,
        "nonzero_grad_tensors": nonzero,
        "all_finite": finite,
        "global_l2": global_sq ** 0.5,
        "by_block_l2": {str(k): block_sq[k] ** 0.5 for k in sorted(block_sq)},
        "by_block_nonzero_tensors": {str(k): block_nonzero[k] for k in sorted(block_nonzero)},
        "by_kind_l2": {k: v ** 0.5 for k, v in kind_sq.items()},
        "by_kind_nonzero_tensors": dict(kind_nonzero),
        "per_tensor": rows,
    }


def save_preview(x, path):
    img = ((x[0].detach().float().cpu().clamp(-1, 1) + 1) * 127.5).byte()
    img = img.permute(1, 2, 0).numpy()
    Image.fromarray(img).save(path)


def finalize():
    passed = sum(x["status"] == "PASS" for x in RESULTS)
    failed = len(RESULTS) - passed
    summary = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "results": RESULTS,
        "details": DETAILS,
        "totals": {"tests": len(RESULTS), "passed": passed, "failed": failed},
    }
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md = [
        "# PixelDiT2 H/16 256 Real LoRA Training-Step Report", "",
        f"Result: **{passed} PASS / {failed} FAIL / {len(RESULTS)} total**", "",
        "| # | Stage | Status | Seconds |", "|---:|---|:---:|---:|",
    ]
    for i, r in enumerate(RESULTS, 1):
        md.append(f"| {i} | {r['name']} | **{r['status']}** | {r['seconds']:.3f} |")
    if failed:
        md += ["", "## Failures", ""]
        for r in RESULTS:
            if r["status"] == "FAIL":
                md += [f"### {r['name']}", "", f"`{r.get('error', '')}`", ""]
    md += ["", "Full numeric details are in `summary.json` and execution output in `full.log`."]
    (REPORT / "REPORT.md").write_text("\n".join(md))
    if ZIP.exists():
        ZIP.unlink()
    shutil.make_archive(str(ZIP.with_suffix("")), "zip", REPORT)
    log(f"\nFINAL: {passed} PASS / {failed} FAIL / {len(RESULTS)} TOTAL")
    log(f"REPORT ZIP: {ZIP} ({ZIP.stat().st_size / 2**20:.2f} MiB)")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    torch.set_float32_matmul_precision("high")

    commit = run_cmd(["git", "rev-parse", "HEAD"])[1].strip()
    env = {
        "commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3),
    }
    DETAILS["environment"] = env
    log(json.dumps(env, indent=2))
    (REPORT / "environment.json").write_text(json.dumps(env, indent=2))
    rc, smi = run_cmd(["nvidia-smi"], timeout=30)
    (REPORT / "nvidia-smi.txt").write_text(smi)

    state = {}

    def download_checkpoint():
        api = HfApi()
        info = api.model_info(CHECKPOINT_REPO, files_metadata=True)
        meta = next((s for s in info.siblings if s.rfilename == CHECKPOINT_NAME), None)
        if meta is None:
            raise RuntimeError(f"{CHECKPOINT_NAME} not found in {CHECKPOINT_REPO}")
        t0 = time.perf_counter()
        path = hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_NAME)
        dt = time.perf_counter() - t0
        size = os.path.getsize(path)
        state["checkpoint_path"] = path
        return {
            "repo": CHECKPOINT_REPO, "revision": info.sha, "filename": CHECKPOINT_NAME,
            "local_path": path, "size_bytes": size, "size_gib": size / 2**30,
            "download_or_cache_seconds": dt,
        }
    stage("1. official checkpoint download/cache", download_checkpoint)

    def sample_image():
        url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        path = REPORT / "dog.jpg"
        urllib.request.urlretrieve(url, path)
        image = Image.open(path).convert("RGB")
        tfm = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda z: z * 2.0 - 1.0),
        ])
        x = tfm(image).unsqueeze(0)
        state["x_cpu"] = x
        save_preview(x, REPORT / "training_input_256.png")
        return {"url": url, "shape": list(x.shape), "range": [float(x.min()), float(x.max())], "class_id": TEST_CLASS}
    stage("2. real 256x256 image preprocessing", sample_image)

    def load_model_and_ema():
        t0 = time.perf_counter()
        model = PixelDiT2(**model_kwargs(pretrained_checkpoint=state["checkpoint_path"]))
        load_s = time.perf_counter() - t0
        tr, total = lora_parameter_counts(model)
        if tr != EXPECTED_LORA_PARAMS:
            raise AssertionError(f"LoRA params {tr:,} != expected {EXPECTED_LORA_PARAMS:,}")
        if model.pretrained_checkpoint_prefix != "ema_denoiser.":
            raise AssertionError(f"unexpected checkpoint prefix: {model.pretrained_checkpoint_prefix}")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not trainable_names or not all(n.endswith("lora_A") or n.endswith("lora_B") for n in trainable_names):
            raise AssertionError("non-LoRA parameters are trainable")
        frozen_hash, frozen_tensors, frozen_elems = frozen_digest(model)
        state["frozen_hash_before"] = frozen_hash
        state["adapter_before"] = adapter_snapshot(model)
        ema = copy.deepcopy(model)
        no_grad(ema)
        ema.eval()
        model.train()
        model.to(DEVICE)
        ema.to(DEVICE)
        tracker = SimpleEMA(decay=0.9999)
        tracker.setup_models(model, ema)
        if len(tracker.net_params) != len(trainable_names):
            raise AssertionError("EMA is not tracking exactly the trainable LoRA tensors")
        state.update(model=model, ema=ema, tracker=tracker)
        torch.cuda.synchronize()
        return {
            "checkpoint_prefix": model.pretrained_checkpoint_prefix,
            "model_load_seconds": load_s,
            "lora_params": tr,
            "total_params": total,
            "trainable_percent": 100.0 * tr / total,
            "trainable_tensors": len(trainable_names),
            "frozen_tensors": frozen_tensors,
            "frozen_elements": frozen_elems,
            "frozen_sha256_before": frozen_hash,
            "ema_tracked_tensors": len(tracker.net_params),
            "gpu_allocated_gib_after_model_and_ema": torch.cuda.memory_allocated() / 2**30,
        }
    stage("3. real H16 base load + LoRA injection + EMA construction", load_model_and_ema)

    def real_train_step():
        model = state["model"]
        ema = state["ema"]
        tracker = state["tracker"]
        x = state["x_cpu"].to(DEVICE, non_blocking=True)
        y = torch.tensor([TEST_CLASS], dtype=torch.long, device=DEVICE)
        null_y = torch.full_like(y, NULL_CLASS)

        flow = LoRAGroundedFlowTrainer(
            p_mean=-0.8, p_std=0.8, noise_scale=1.0, t_eps=0.05,
            null_condition_p=0.0,
            grounding_drop_mode="independent", grounding_drop_p=0.0,
            grounding_joint_drop_p=0.0,
            repa_weight=0.0, repa_align_layer=8, repa_encoder=None,
            proj_denoiser_dim=1280, proj_hidden_dim=1280, proj_encoder_dim=768,
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0,
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        losses = flow(model, ema, None, x, y, null_y, metadata={})
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        forward_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_s = time.perf_counter() - t1
        gs = grad_stats(model)
        if not gs["all_finite"] or gs["nonzero_grad_tensors"] == 0:
            raise RuntimeError("invalid LoRA gradients")

        unclipped = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        ).detach().cpu())

        t2 = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        optim_s = time.perf_counter() - t2

        delta = adapter_delta(state["adapter_before"], model)
        if delta["changed_tensors"] == 0:
            raise RuntimeError("optimizer did not change any LoRA tensor")

        # Actual repository SimpleEMA path. It updates only the trainable tensors.
        ema_before = [p.detach().clone() for p in tracker.ema_params]
        t3 = time.perf_counter()
        tracker.ema_step()
        torch.cuda.synchronize()
        ema_s = time.perf_counter() - t3
        ema_sq = 0.0
        ema_changed = 0
        for a, b in zip(ema_before, tracker.ema_params):
            d = (b.detach() - a).float()
            ema_sq += float(d.square().sum().cpu())
            ema_changed += int(torch.count_nonzero(d).item() > 0)
        del ema_before

        peak = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        state["flow"] = flow
        state["optimizer"] = optimizer
        state["train_x_gpu"] = x
        state["train_y_gpu"] = y
        return {
            "loss": float(loss.detach().float().cpu()),
            "fm_loss": float(losses["fm_loss"].detach().float().cpu()),
            "forward_seconds": forward_s,
            "backward_seconds": backward_s,
            "optimizer_seconds": optim_s,
            "ema_seconds": ema_s,
            "step_compute_seconds": forward_s + backward_s + optim_s + ema_s,
            "allocated_before_step_gib": allocated_before / 2**30,
            "peak_allocated_gib": peak / 2**30,
            "peak_reserved_gib": peak_reserved / 2**30,
            "unclipped_grad_norm": unclipped,
            "gradient_stats": gs,
            "adapter_delta": delta,
            "ema_changed_tensors": ema_changed,
            "ema_delta_l2": ema_sq ** 0.5,
        }
    stage("4. genuine 256px grounded flow LoRA forward/backward/AdamW/EMA step", real_train_step)

    def verify_frozen():
        after_hash, tensors, elems = frozen_digest(state["model"])
        before = state["frozen_hash_before"]
        if after_hash != before:
            raise AssertionError(f"frozen parameter digest changed: before={before} after={after_hash}")
        return {
            "every_frozen_parameter_bit_identical": True,
            "frozen_tensors_checked": tensors,
            "frozen_elements_checked": elems,
            "sha256_before": before,
            "sha256_after": after_hash,
        }
    stage("5. verify every frozen parameter stayed bit-identical", verify_frozen)

    def save_adapters():
        model = state["model"]
        ema = state["ema"]
        fp32 = REPORT / "real_step_adapter_fp32.safetensors"
        bf16 = REPORT / "real_step_adapter_bf16.safetensors"
        ema_bf16 = REPORT / "real_step_ema_adapter_bf16.safetensors"
        meta = {"base": CHECKPOINT_NAME, "class_id": TEST_CLASS, "steps": 1}
        save_lora_adapter(model, str(fp32), dtype=torch.float32, metadata=meta)
        save_lora_adapter(model, str(bf16), dtype=torch.bfloat16, metadata=meta)
        save_lora_adapter(ema, str(ema_bf16), dtype=torch.bfloat16, metadata={**meta, "ema": True})
        state["adapter_fp32"] = str(fp32)
        return {
            "trained_fp32_bytes": fp32.stat().st_size,
            "trained_fp32_mib": fp32.stat().st_size / 2**20,
            "trained_bf16_bytes": bf16.stat().st_size,
            "trained_bf16_mib": bf16.stat().st_size / 2**20,
            "ema_bf16_bytes": ema_bf16.stat().st_size,
            "ema_bf16_mib": ema_bf16.stat().st_size / 2**20,
        }
    stage("6. save real trained and EMA adapters", save_adapters)

    def reference_output_and_release():
        model = state["model"].eval()
        x = F.interpolate(state["x_cpu"], size=(64, 64), mode="bicubic", align_corners=False).to(DEVICE)
        t = torch.tensor([0.5], device=DEVICE)
        y = torch.tensor([TEST_CLASS], dtype=torch.long, device=DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, t, y)
        ref = out.detach().float().cpu()
        torch.save({"x": x.float().cpu(), "t": t.cpu(), "y": y.cpu(), "out": ref}, REPORT / "reload_probe.pt")
        state["probe_ref"] = ref
        state["probe_x"] = x.float().cpu()
        state["probe_t"] = t.cpu()
        state["probe_y"] = y.cpu()

        # Release the full model/EMA before constructing the fresh reload model.
        for key in ["model", "ema", "tracker", "flow", "optimizer", "train_x_gpu", "train_y_gpu"]:
            state.pop(key, None)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return {"probe_shape": list(ref.shape), "gpu_allocated_after_release_gib": torch.cuda.memory_allocated() / 2**30}
    stage("7. capture deterministic post-step probe and release first model", reference_output_and_release)

    def reload_compare():
        fresh = PixelDiT2(**model_kwargs(
            pretrained_checkpoint=state["checkpoint_path"],
            lora_adapter_path=state["adapter_fp32"],
        )).to(DEVICE).eval()
        x = state["probe_x"].to(DEVICE)
        t = state["probe_t"].to(DEVICE)
        y = state["probe_y"].to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = fresh(x, t, y).detach().float().cpu()
        ref = state["probe_ref"]
        diff = (out - ref).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        same = bool(torch.equal(out, ref))
        close = bool(torch.allclose(out, ref, atol=1e-4, rtol=1e-4))
        if not close:
            raise AssertionError(f"fresh adapter reload output mismatch: max_abs={max_abs}")
        del fresh
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "exact_tensor_equal": same,
            "allclose_1e-4": close,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
        }
    stage("8. fresh-base FP32 adapter reload output equivalence", reload_compare)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\nTOP-LEVEL FAILURE\n" + traceback.format_exc())
    finally:
        finalize()
