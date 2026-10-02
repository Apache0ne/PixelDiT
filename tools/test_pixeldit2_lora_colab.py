"""Comprehensive PixelDiT2 LoRA validation for Google Colab.

Run from the repository root. Every test is isolated, results are written to
/content/pixeldit2_lora_test_report, zipped, and automatically downloaded when
Google Colab is available.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
REPORT = Path("/content/pixeldit2_lora_test_report")
ZIP = Path("/content/pixeldit2_lora_test_report.zip")
REPORT.mkdir(parents=True, exist_ok=True)
LOG = REPORT / "full.log"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = []
DETAILS = {}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "c2i"))

from pixdit_core.grounding import GroundingConditioner
from pixdit_core.lora import (
    DEFAULT_PIXELDIT2_LORA_TARGETS,
    LoRAAdapterCheckpoint,
    LoRALinear,
    lora_parameter_counts,
    lora_state_dict,
    save_lora_adapter,
)
from pixdit_core.pixeldit2_c2i import PixelDiT2
from c2i.src.lora_diffusion import LoRAGroundedFlowTrainer
from c2i.src.utils import SimpleEMA, no_grad


def log(msg=""):
    print(msg)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def cmd(args, cwd=ROOT, timeout=300):
    p = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    log("$ " + " ".join(map(str, args)))
    log(p.stdout)
    return p


def test(name, fn):
    t0 = time.time()
    try:
        value = fn()
        RESULTS.append({"name": name, "status": "PASS", "seconds": round(time.time()-t0, 3)})
        if value is not None:
            DETAILS[name] = value
        log(f"[PASS] {name}")
    except Exception as e:
        tb = traceback.format_exc()
        RESULTS.append({"name": name, "status": "FAIL", "seconds": round(time.time()-t0, 3),
                        "error": f"{type(e).__name__}: {e}"})
        DETAILS[name] = {"traceback": tb}
        (REPORT / ("FAIL_" + name.replace(" ", "_").replace("/", "_") + ".txt")).write_text(tb)
        log(f"[FAIL] {name}\n{tb}")


def tiny(**kw):
    d = dict(in_channels=3, hidden_size=192, num_heads=3, depth=2, patch_size=16,
             num_classes=10, in_context_len=0, in_context_start=0,
             attn_dropout=0.0, proj_dropout=0.0, mlp_hidden_multiple=0,
             grounding_encoder="dinov3_vits16", grounding_proj_blocks=1,
             grounding_proj_heads=3, single_silu_cond=False, pretrained_encoder=False)
    d.update(kw)
    return d


def sha(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


commit = cmd(["git", "rev-parse", "HEAD"]).stdout.strip()
env = {
    "utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
    "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_count": torch.cuda.device_count(), "ram_gib": round(psutil.virtual_memory().total/2**30, 2),
}
if torch.cuda.is_available():
    env["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory/2**30, 2)
(REPORT / "environment.json").write_text(json.dumps(env, indent=2))
log(json.dumps(env, indent=2))
if torch.cuda.is_available():
    (REPORT / "nvidia-smi.txt").write_text(cmd(["nvidia-smi"], timeout=30).stdout)
(REPORT / "pip_freeze.txt").write_text(cmd([sys.executable, "-m", "pip", "freeze"]).stdout)


def static_checks():
    assert cmd(["git", "diff", "--check", "origin/master...HEAD"]).returncode == 0
    files = ["pixdit_core/lora.py", "pixdit_core/pixeldit2_c2i.py", "c2i/src/lora_diffusion.py"]
    assert cmd([sys.executable, "-m", "py_compile"] + files).returncode == 0
    assert cmd(["bash", "-n", "c2i/train_c2i.sh"]).returncode == 0
    audit = {}
    for f in ["pixeldit2_h16_in256_lora.yaml", "pixeldit2_h16_in512_lora.yaml"]:
        cfg = yaml.safe_load((ROOT / "c2i/configs" / f).read_text())
        a = cfg["model"]["denoiser"]["init_args"]
        tr = cfg["model"]["diffusion_trainer"]
        callbacks = [x["class_path"] for x in cfg["trainer"]["callbacks"]]
        assert a["lora_rank"] > 0 and len(a["lora_target_modules"]) == 5
        assert a["pretrained_checkpoint"] and a["pretrained_weights"] == "ema"
        assert tr["class_path"] == "src.lora_diffusion.LoRAGroundedFlowTrainer"
        assert tr["init_args"]["repa_weight"] == 0.0 and tr["init_args"]["repa_encoder"] is None
        assert "pixdit_core.lora.LoRAAdapterCheckpoint" in callbacks
        dim, depth, rank = int(a["hidden_size"]), int(a["depth"]), int(a["lora_rank"])
        ff = int(2 * int(dim*4) / 3)
        m = int(a.get("mlp_hidden_multiple", 0))
        if m:
            ff = ((ff+m-1)//m)*m
        n = depth * (rank*(dim+3*dim) + rank*(dim+dim) + 3*rank*(dim+ff))
        audit[f] = {"target_modules": depth*5, "lora_params": n,
                    "bf16_adapter_mib": round(n*2/2**20, 2)}
    return audit

test("static source/config checks", static_checks)


def cli_check():
    p = cmd([sys.executable, "c2i/main.py", "fit", "--help"], timeout=180)
    assert p.returncode == 0
    (REPORT / "lightning_cli_help.txt").write_text(p.stdout)

test("Lightning CLI import/help", cli_check)


def linear_test():
    torch.manual_seed(1)
    base = torch.nn.Linear(32, 48).to(DEVICE)
    x = torch.randn(4, 32, device=DEVICE)
    ref = base(x).detach()
    layer = LoRALinear(base, 4, 8).to(DEVICE)
    got = layer(x).detach()
    assert torch.equal(ref, got)
    assert not layer.base_layer.weight.requires_grad and layer.lora_A.requires_grad and layer.lora_B.requires_grad
    compiled = torch.compile(layer.eval(), backend="eager", fullgraph=False)
    assert torch.allclose(layer(x), compiled(x), atol=1e-6, rtol=1e-5)
    with torch.no_grad(): layer.lora_B.normal_(0, .01)
    assert not torch.allclose(ref, layer(x))
    return {"zero_init_max_error": float((ref-got).abs().max().cpu()), "dynamo": "PASS"}

test("LoRALinear zero-init/freeze/torch.compile", linear_test)

holder = {}

def injection_test():
    torch.manual_seed(7)
    m = PixelDiT2(**tiny(lora_rank=4, lora_alpha=4.0)).to(DEVICE)
    holder["m"] = m
    assert len(m.lora_injected_modules) == 10
    train = [n for n,p in m.named_parameters() if p.requires_grad]
    assert train and all(n.endswith(".lora_A") or n.endswith(".lora_B") for n in train)
    assert all(not p.requires_grad for p in m.dino_conditioner.encoder.parameters())
    tr, total = lora_parameter_counts(m)
    return {"injected": m.lora_injected_modules, "trainable_params": tr, "total_params": total,
            "trainable_percent": 100*tr/total}

test("tiny PixelDiT2 adapter-only injection", injection_test)


def train_step_test():
    m = holder["m"].train()
    with torch.no_grad():
        for b in m.patch_blocks:
            b.adaLN_modulation[-1].weight.normal_(0,.01); b.adaLN_modulation[-1].bias.normal_(0,.01)
        m.final_layer.linear.weight.normal_(0,.01)
    q = m.patch_blocks[0].attn.qkv.base_layer.weight
    d = next(m.dino_conditioner.encoder.parameters())
    q0, d0 = sha(q), sha(d)
    b0 = m.patch_blocks[0].attn.qkv.lora_B.detach().clone()
    x = torch.randn(1,3,32,32,device=DEVICE); t = torch.tensor([.45],device=DEVICE)
    y = torch.tensor([3],dtype=torch.long,device=DEVICE)
    out = m(x,t,y,skip_grounding=True); loss = out.float().square().mean(); loss.backward()
    grads = {n:float(p.grad.float().norm().cpu()) for n,p in m.named_parameters() if p.requires_grad and p.grad is not None}
    assert any(v>0 for n,v in grads.items() if n.endswith(".lora_B"))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2); opt.step(); opt.zero_grad(True)
    assert not torch.equal(b0, m.patch_blocks[0].attn.qkv.lora_B)
    assert q0 == sha(q) and d0 == sha(d)
    return {"loss":float(loss.cpu()), "nonzero_grad_tensors":sum(v>0 for v in grads.values()),
            "base_unchanged":True, "dino_unchanged":True}

test("forward/backward/optimizer frozen-base test", train_step_test)


def adapter_io_test():
    from safetensors import safe_open
    m = holder["m"]
    path = REPORT / "tiny_adapter.safetensors"
    save_lora_adapter(m, str(path), dtype=torch.bfloat16, metadata={"test":"colab"})
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys, meta = list(f.keys()), f.metadata()
    assert len(keys)==20 and meta["format"]=="pixeldit2-lora-v1"
    loaded = PixelDiT2(**tiny(lora_rank=4,lora_alpha=4,lora_adapter_path=str(path))).to(DEVICE)
    state = lora_state_dict(m, dtype=torch.bfloat16); named = dict(loaded.named_parameters())
    assert all(torch.equal(named[k].detach().cpu().to(torch.bfloat16),v) for k,v in state.items())
    rejected = False
    try: PixelDiT2(**tiny(lora_rank=2,lora_alpha=2,lora_adapter_path=str(path)))
    except (ValueError, RuntimeError): rejected = True
    assert rejected
    return {"file_bytes":path.stat().st_size,"tensors":len(keys),"strict_reload":True,"rank_mismatch_rejected":True}

test("adapter safetensors save/reload/mismatch", adapter_io_test)


def ema_callback_test():
    m = holder["m"]
    cb = LoRAAdapterCheckpoint(1,"callback_adapters",True,"bf16")
    fake_trainer = SimpleNamespace(default_root_dir=str(REPORT),global_step=123,current_epoch=4,is_global_zero=True)
    cb._save(fake_trainer, SimpleNamespace(ema_denoiser=copy.deepcopy(m),pretrained_checkpoint="test.ckpt"), "test.safetensors")
    p = REPORT/"callback_adapters/test.safetensors"; assert p.exists()
    if not torch.cuda.is_available(): return {"callback":True,"ema_stream":"SKIP-no-CUDA"}
    e = copy.deepcopy(m); no_grad(e); tracker = SimpleEMA(.5); tracker.setup_models(m,e)
    assert len(tracker.net_params) == sum(1 for x in m.parameters() if x.requires_grad)
    before = tracker.ema_params[0].detach().clone()
    with torch.no_grad(): tracker.net_params[0].add_(1)
    tracker.ema_step(); torch.cuda.synchronize()
    delta = float((tracker.ema_params[0]-before).float().mean().cpu()); assert abs(delta-.5)<1e-4
    return {"callback":True,"ema_tracked_tensors":len(tracker.net_params),"ema_delta":delta}

test("EMA + adapter checkpoint callback", ema_callback_test)


def trainer_test():
    tr = LoRAGroundedFlowTrainer(p_mean=-.8,p_std=.8,noise_scale=1,t_eps=.05,null_condition_p=.1,
        grounding_drop_mode="independent",grounding_drop_p=.1,grounding_joint_drop_p=0,
        repa_weight=0,repa_align_layer=1,repa_encoder=None,proj_denoiser_dim=192,proj_hidden_dim=192,proj_encoder_dim=64)
    bad = [(n,p.numel()) for n,p in tr.named_parameters() if p.requires_grad]
    assert not bad, bad
    return {"accidental_trainable_params":0}

test("REPA-disabled trainer optimizer isolation", trainer_test)


def checkpoint_test():
    path = REPORT/"synthetic_base.ckpt"
    torch.manual_seed(222); base = PixelDiT2(**tiny(lora_rank=0)).cpu()
    torch.save({"state_dict":{"ema_denoiser."+k:v.cpu() for k,v in base.state_dict().items()}},path)
    loaded = PixelDiT2(**tiny(pretrained_checkpoint=str(path),pretrained_weights="ema",lora_rank=4,lora_alpha=4)).cpu()
    assert loaded.pretrained_checkpoint_prefix == "ema_denoiser."
    assert torch.equal(loaded.patch_blocks[0].attn.qkv.base_layer.weight,base.patch_blocks[0].attn.qkv.weight)
    path.unlink()
    return {"prefix":"ema_denoiser.","targeted_base_equal":True}

test("pretrained checkpoint loader synthetic EMA path", checkpoint_test)


def custom_target_test():
    targets = list(DEFAULT_PIXELDIT2_LORA_TARGETS)+["dino_conditioner.dino_proj.input_proj"]
    m = PixelDiT2(**tiny(lora_rank=2,lora_alpha=2,lora_target_modules=targets))
    assert "dino_conditioner.dino_proj.input_proj" in m.lora_injected_modules
    assert all(not p.requires_grad for p in m.dino_conditioner.encoder.parameters())
    return {"grounding_projection_lora":True,"dino_frozen":True}

test("optional grounding-projector LoRA target", custom_target_test)


def dino_test():
    holder["m"].cpu(); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    c = GroundingConditioner("dinov3_vits16",192,16,1,3,True).to(DEVICE).eval()
    x = torch.randn(1,3,32,32,device=DEVICE).clamp(-1,1); t = torch.tensor([.5],device=DEVICE)
    with torch.no_grad(): out = c(x,t); raw,grid = c.encode(x)
    assert out.shape==(1,4,192) and grid==(2,2) and raw.shape[1]==4 and torch.isfinite(out).all()
    assert all(not p.requires_grad for p in c.encoder.parameters())
    return {"condition_shape":list(out.shape),"raw_tokens":list(raw.shape),"grid":grid,"frozen":True}

test("actual pretrained DINOv3 grounding forward", dino_test)


def hf_test():
    from huggingface_hub import HfApi
    info = HfApi().model_info("nvidia/PixelDiT2-ImageNet",files_metadata=True)
    wanted={"pixeldit2_h16_in256_ep600.ckpt","pixeldit2_h16_in512_ep680.ckpt"}; found={}
    for s in info.siblings:
        if s.rfilename in wanted: found[s.rfilename]={"size_bytes":getattr(s,"size",None),"blob_id":getattr(s,"blob_id",None)}
    assert set(found)==wanted,found
    return {"repo":info.id,"revision":info.sha,"files":found}

test("released NVIDIA checkpoint remote metadata", hf_test)

# Save exact diff and tested source snapshots.
cmd(["git","fetch","origin","master:refs/remotes/origin/master"],timeout=120)
(REPORT/"git_diff.patch").write_text(cmd(["git","diff","origin/master...HEAD"],timeout=120).stdout)
snap=REPORT/"tested_sources"; snap.mkdir(exist_ok=True)
for rel in ["pixdit_core/lora.py","pixdit_core/pixeldit2_c2i.py","c2i/src/lora_diffusion.py",
            "c2i/configs/pixeldit2_h16_in256_lora.yaml","c2i/configs/pixeldit2_h16_in512_lora.yaml","c2i/LORA.md"]:
    src=ROOT/rel; dst=snap/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)

passed=sum(x["status"]=="PASS" for x in RESULTS); failed=len(RESULTS)-passed
summary={"environment":env,"totals":{"tests":len(RESULTS),"passed":passed,"failed":failed},"tests":RESULTS,"details":DETAILS,
 "notes":["Colab's existing torch/torchvision are retained to avoid a runtime restart; exact versions are recorded.",
          "The multi-GB released H/16 files are verified by remote metadata; the new checkpoint loader is tested end-to-end with a synthetic EMA-prefixed checkpoint.",
          "The DINO test uses the actual pretrained DINOv3-S/16 weights.",
          "A one-GPU Colab cannot prove multi-node/multi-GPU DDP behavior."]}
(REPORT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
md=["# PixelDiT2 LoRA Colab Test Report","",f"Commit: `{commit}`",f"GPU: `{env['gpu']}`",f"Result: **{passed} PASS / {failed} FAIL / {len(RESULTS)} total**","",
    "| # | Test | Status | Seconds |","|---:|---|:---:|---:|"]
for i,r in enumerate(RESULTS,1): md.append(f"| {i} | {r['name']} | **{r['status']}** | {r['seconds']:.3f} |")
if failed:
    md += ["","## Failures",""]
    for r in RESULTS:
        if r["status"]=="FAIL": md += [f"### {r['name']}","",f"`{r.get('error','')}`",""]
md += ["","Full details are in `summary.json`, `full.log`, `git_diff.patch`, and `tested_sources/`."]
(REPORT/"REPORT.md").write_text("\n".join(md))
if torch.cuda.is_available():
    (REPORT/"cuda_memory.json").write_text(json.dumps({"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved(),"max_allocated":torch.cuda.max_memory_allocated()},indent=2))
if ZIP.exists(): ZIP.unlink()
shutil.make_archive(str(ZIP.with_suffix("")),"zip",REPORT)
log(f"FINAL: {passed} PASS / {failed} FAIL / {len(RESULTS)} TOTAL")
log(f"ZIP: {ZIP} ({ZIP.stat().st_size/2**20:.2f} MiB)")
try:
    from google.colab import files
    files.download(str(ZIP))
except Exception as e:
    log(f"Auto-download unavailable: {e}; manually download {ZIP}")
