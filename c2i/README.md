# Class-to-image generation

PixelDiT and PixelDiT2 share the training and inference entrypoint
`main.py`. Select the model through its configuration and checkpoint;
the workflow below covers both generations.

[PixelDiT2 · NeurIPS 2026](https://pixeldit.github.io/pixeldit2/) ·
[PixelDiT · CVPR 2026](https://pixeldit.github.io/)

[Overview](../README.md) · [Models](#pretrained-models) ·
[Inference](#inference-and-evaluation) · [Training](#training)

## Setup

Run installation commands from the repository root. To reproduce the results, use separate python environments for the two model families.

### PixelDiT2 environment

Use Python 3.10. The requirements select PyTorch 2.5.1 / torchvision 0.20.1
with CUDA 12.4 and retain the pinned dependencies of the PixelDiT2 recipe:

```bash
python -m pip install -r requirements-pixeldit2.txt
```

The frozen DINOv3 grounding encoder is loaded through `timm`; its weights are
downloaded on first use. PixelDiT2 evaluation uses the pinned
[LTH14 torch-fidelity fork](https://github.com/LTH14/torch-fidelity) and
[JiT reference statistics](https://github.com/LTH14/JiT/tree/main/fid_stats).
Inception weights are downloaded when scoring images for the first time.

### PixelDiT1 environment

PixelDiT is the original model family (called PixelDiT1 in the requirements
filename). Its dependency file preserves the original dependency specification,
including PyTorch 2.5.0 and the dependencies used by PixelDiT-T2I:

```bash
python -m pip install -r requirements-pixeldit1.txt
```

`requirements.txt` remains an alias for this PixelDiT1 dependency file.
Some original dependencies are unpinned; this file is not a complete environment
lock. PixelDiT1 C2I does not use the DINOv3 grounding encoder.

PixelDiT1's reported C2I results use the
[ADM evaluation toolkit](https://github.com/openai/guided-diffusion/tree/main/evaluations).
Set up that toolkit's TensorFlow dependencies in a separate evaluation
environment; they are not installed by either model requirements file.

### Checkpoints and numerical reproducibility

Checkpoints are downloaded automatically when a released filename is passed to
`--ckpt_path`; a local checkpoint path also works. Use each model family's
sampling settings, evaluator and reference statistics below. Numerical results
can vary with hardware and software versions; changing the evaluation backend
or reference statistics does not preserve the reported FID by assumption.

## Pretrained models

| Model | Resolution | Epochs | FID ↓ | IS ↑ | Checkpoint |
|---|:---:|---:|---:|---:|---|
| **PixelDiT2-H/16** | 256×256 | 600 | **1.46** | **301.6** | `pixeldit2_h16_in256_ep600.ckpt` |
| **PixelDiT2-H/16** | 512×512 | 680 | **1.48** | **295.7** | `pixeldit2_h16_in512_ep680.ckpt` |
| PixelDiT-XL | 256×256 | 80 | 2.36 | — | `imagenet256_pixeldit_xl_epoch80.ckpt` |
| PixelDiT-XL | 256×256 | 160 | 1.97 | — | `imagenet256_pixeldit_xl_epoch160.ckpt` |
| PixelDiT-XL | 256×256 | 320 | 1.61 | — | `imagenet256_pixeldit_xl_epoch320.ckpt` |
| PixelDiT-XL | 512×512 | 850 | 1.81 | — | `imagenet512_pixeldit_xl.ckpt` |

Pass a checkpoint filename to `--ckpt_path` for automatic download, or use a
local file path. Weights are hosted in
[PixelDiT2-ImageNet](https://huggingface.co/nvidia/PixelDiT2-ImageNet) and
[PixelDiT-ImageNet](https://huggingface.co/nvidia/PixelDiT-ImageNet).
PixelDiT2 checkpoints contain EMA weights; frozen encoder weights are loaded
separately through `timm`.

<details>
<summary>PixelDiT2 checkpoint checksums</summary>

| Checkpoint | SHA256 |
|---|---|
| `pixeldit2_h16_in256_ep600.ckpt` | `c647db081729812a6008b58593c0b8cb59eb73761290cbb8586935e590fbcf06` |
| `pixeldit2_h16_in512_ep680.ckpt` | `f9b88695e157b38aff492bd9d310537a0bf1037a4aeb58d4064d452cf62d46dd` |

</details>

<a id="evaluation"></a>

## Inference and evaluation

Both models generate samples through `main.py predict`. Their sampling and
evaluation settings are:

| Model | Sampler | Steps | Evaluation |
|---|---|---:|---|
| PixelDiT2 | Heun | 50 | LTH14 torch-fidelity + JiT statistics |
| PixelDiT | FlowDPMSolver | 100 | ADM evaluation toolkit |

Run each command block from the repository root unless stated otherwise.

<a id="pixeldit2"></a>

### PixelDiT2 inference

Activate the PixelDiT2 environment. Generate 50,000 PNG images and compute
FID/IS with `tools/evaluate_fid.py`, the pinned LTH14 torch-fidelity fork and
the [JiT reference statistics](https://github.com/LTH14/JiT/tree/main/fid_stats).
These scoring instructions apply to PixelDiT2.

```bash
mkdir -p fid_stats outputs
```

| Setting | 256×256 | 512×512 |
|---|:---:|:---:|
| Sampler / steps | Heun / 50 | Heun / 50 |
| CFG scale | 2.4 | 2.1 |
| Guidance interval | [0.125, 0.9] | [0.1, 0.925] |
| Per-process batch size | 64 | 16 |
| Prediction workers | 1 | 1 |

<a id="reproduce-the-imagenet-256-result"></a>

#### ImageNet 256×256

Run from the repository root. A new output directory is created for every run.

```bash
curl -fL https://raw.githubusercontent.com/LTH14/JiT/main/fid_stats/jit_in256_stats.npz \
  -o fid_stats/jit_in256_stats.npz
OUTPUT_DIR=$(mktemp -d "$PWD/outputs/pixeldit2_256.XXXXXX")

(
  cd c2i
  TORCHDYNAMO_DISABLE=1 torchrun --nproc_per_node=8 main.py predict \
    -c configs/pixeldit2_h16_in256.yaml \
    --ckpt_path=pixeldit2_h16_in256_ep600.ckpt \
    --trainer.num_nodes=1 \
    --trainer.default_root_dir="$OUTPUT_DIR" \
    --trainer.logger=false \
    --data.pred_batch_size=64 \
    --data.pred_num_workers=1 \
    --model.diffusion_sampler.init_args.num_steps=50 \
    --model.diffusion_sampler.init_args.guidance=2.4 \
    --model.diffusion_sampler.init_args.guidance_interval_min=0.125 \
    --model.diffusion_sampler.init_args.guidance_interval_max=0.9 \
    --per_run_seed=false --seed_everything=5
) && python tools/evaluate_fid.py \
  --images "$OUTPUT_DIR/exp_pixeldit2_h16_in256/val_256_h16/predict" \
  --fid-stats fid_stats/jit_in256_stats.npz \
  --fid-stats-sha256 412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c \
  --output "$OUTPUT_DIR/result.json"
```

<a id="imagenet-512-evaluation"></a>

#### ImageNet 512×512

Run from the repository root. A new output directory is created for every run.

```bash
curl -fL https://raw.githubusercontent.com/LTH14/JiT/main/fid_stats/jit_in512_stats.npz \
  -o fid_stats/jit_in512_stats.npz
OUTPUT_DIR=$(mktemp -d "$PWD/outputs/pixeldit2_512.XXXXXX")

(
  cd c2i
  TORCHDYNAMO_DISABLE=0 torchrun --nproc_per_node=8 main.py predict \
    -c configs/pixeldit2_h16_in512.yaml \
    --ckpt_path=pixeldit2_h16_in512_ep680.ckpt \
    --trainer.num_nodes=1 \
    --trainer.default_root_dir="$OUTPUT_DIR" \
    --trainer.logger=false \
    --data.pred_batch_size=16 \
    --data.pred_num_workers=1 \
    --model.diffusion_sampler.init_args.num_steps=50 \
    --model.diffusion_sampler.init_args.guidance=2.1 \
    --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
    --model.diffusion_sampler.init_args.guidance_interval_max=0.925 \
    --per_run_seed=false --seed_everything=100
) && python tools/evaluate_fid.py \
  --images "$OUTPUT_DIR/exp_pixeldit2_h16_in512/val_512_h16/predict" \
  --fid-stats fid_stats/jit_in512_stats.npz \
  --fid-stats-sha256 cdd15b54f9d1f26a881fcdc920a3410fa7a934047d6c4f5b8395b87182a9ab42 \
  --output "$OUTPUT_DIR/result.json"
```

The scorer checks the statistics checksum and requires exactly 50,000 PNG
images. Metrics are saved to `result.json` in the output directory.

<a id="pixeldit"></a>

### PixelDiT1 inference

Activate the PixelDiT1 environment. The commands below generate samples via
`main.py predict`; PixelDiT1 uses the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations)
to score the resulting `output.npz`.

#### Step 1: Generate Samples

All commands below generate images under `c2i/train_logs/`. Override sampler params on the CLI as needed.

**Epoch 80 (ImageNet 256×256):**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/pix256_xl.yaml \
  --ckpt_path=imagenet256_pixeldit_xl_epoch80.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=3.25 \
  --model.diffusion_sampler.init_args.timeshift=1.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=1.0 \
  --per_run_seed=false --seed_everything=5000
```

**Epoch 160 (ImageNet 256×256):**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/pix256_xl.yaml \
  --ckpt_path=imagenet256_pixeldit_xl_epoch160.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=3.25 \
  --model.diffusion_sampler.init_args.timeshift=1.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=1.0 \
  --per_run_seed=false --seed_everything=5000
```

**Epoch 320 (ImageNet 256×256):**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/pix256_xl.yaml \
  --ckpt_path=imagenet256_pixeldit_xl_epoch320.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=2.75 \
  --model.diffusion_sampler.init_args.timeshift=1.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=0.9 \
  --per_run_seed=false --seed_everything=1600
```

**ImageNet 512×512:**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/pix512_xl.yaml \
  --ckpt_path=imagenet512_pixeldit_xl.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=3.5 \
  --model.diffusion_sampler.init_args.timeshift=2.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=1.0 \
  --per_run_seed=false --seed_everything=10000
```

#### Sampler Settings Summary

| Setting | 80 ep (256) | 160 ep (256) | 320 ep (256) | 512×512 |
|---------|:-----------:|:------------:|:------------:|:-------:|
| CFG Scale | 3.25 | 3.25 | 2.75 | 3.5 |
| Steps | 100 | 100 | 100 | 100 |
| Time Shift | 1.0 | 1.0 | 1.0 | 2.0 |
| CFG Interval | [0.1, 1.0] | [0.1, 1.0] | [0.1, 0.9] | [0.1, 1.0] |
| Sampler | FlowDPMSolver | FlowDPMSolver | FlowDPMSolver | FlowDPMSolver |

#### Step 2: Compute FID

Switch to the separate ADM evaluation environment and follow the
[ADM evaluation instructions](https://github.com/openai/guided-diffusion/tree/main/evaluations)
with the ImageNet reference batch matching the image resolution and the
generated `output.npz` in the predict output directory.

The PixelDiT presets use `save_compressed: true`: the full generated sample set
is stored in `output.npz`, while only the first batch on each rank is also saved
as PNGs. Do not pass that partial PNG directory to `tools/evaluate_fid.py`, which
expects a complete PNG sample set. PixelDiT2's JiT statistics and scorer are not
used to reproduce the PixelDiT results listed above.

## Training

### Data preparation

ImageNet-256 uses the preprocessing workflow from
[REPA-E](https://github.com/End2End-Diffusion/REPA-E). Set `data_dir` in the
chosen configuration to the processed dataset.

At 512×512, the model families used different inputs:

- **PixelDiT2:** raw ImageNet JPEGs under a `train/` directory of class folders,
  center-cropped at load time.
- **PixelDiT:** prepare ImageNet with
  [EDM2's dataset tool](https://github.com/NVlabs/edm2/blob/main/dataset_tool.py)
  and set `data_dir` in `configs/pix512_xl.yaml`.

PixelDiT2 training uses random horizontal flips at both resolutions.

### PixelDiT2 training

```bash
cd c2i/
# ImageNet 256x256: 4 nodes x 8 GPUs x batch 32 = global batch 1024
bash train_c2i.sh --num-nodes 4 --num-gpus 8 --config configs/pixeldit2_h16_in256.yaml

# ImageNet 512x512: 8 nodes x 8 GPUs x batch 16 = global batch 1024
bash train_c2i.sh --num-nodes 8 --num-gpus 8 --config configs/pixeldit2_h16_in512.yaml
```

The 512×512 preset is not a resolution switch alone. It also uses a DINOv3-B/16
grounding encoder, SwiGLU hidden dimensions rounded to a multiple of 32.

#### Delayed grounding dropout (ImageNet 256×256)

The standard 256px preset enables independent class and grounding dropout from
the start. To implement the delayed schedule in
[Section 4.2 of the paper](https://arxiv.org/html/2609.24919v2#S4.SS2), make two
copies of `configs/pixeldit2_h16_in256.yaml` and change the following fields:

| Configuration field | Stage 1: first 160 epochs | Stage 2: continue to epoch 480 |
|---|---|---|
| `auto_resume` | `false` | `false` (resume explicitly) |
| `trainer.max_steps` | `-1` | `-1` |
| `trainer.max_epochs` | `160` | `480` |
| `model.diffusion_trainer.init_args.null_condition_p` | `0.1` | `0.1` |
| `model.diffusion_trainer.init_args.grounding_drop_mode` | `independent` | `independent` |
| `model.diffusion_trainer.init_args.grounding_drop_p` | `0.0` | `0.1` |
| `model.diffusion_trainer.init_args.grounding_joint_drop_p` | `0.0` | `0.0` |

Keep the remaining model, data, optimizer and scheduler settings unchanged.
Use a dedicated experiment name/output directory for this run. Stage 1 retains
class dropout while always keeping grounding; stage 2 drops the two conditions
independently. `coupled` uses the same mask for both and does not implement this
schedule. There is no automatic dropout switch in the trainer.

Save a **full training checkpoint after 160 completed epochs**. To save at
epoch boundaries, replace the preset's `trainer.callbacks` list in both copies
with the following checkpoint callback (online image-saving callbacks are
omitted in this example):

```yaml
trainer:
  callbacks:
    - class_path: src.utils.CheckpointHook
      init_args:
        every_n_train_steps: 0
        every_n_epochs: 1
        save_on_train_epoch_end: true
        save_top_k: 1
        save_last: true
```

Launch stage 1 with its configuration, then launch stage 2 with its configuration
and `--ckpt-path /path/to/stage1/last.ckpt` when using `train_c2i.sh`
(`--ckpt_path=/path/to/stage1/last.ckpt` for `main.py fit`). Resume the optimizer,
scheduler, EMA weights and training progress from this checkpoint. The released
EMA inference checkpoints are not substitutes for a full training checkpoint.
`max_epochs: 480` is the total target, including stage 1; `max_steps: -1` removes
the preset's step limit. This describes the dropout schedule and is not a
guarantee of reproducing a particular FID.

### PixelDiT1 training

#### ImageNet 256×256

```bash
cd c2i/
bash train_c2i.sh --num-gpus 8 --config configs/pix256_xl.yaml
```

#### ImageNet 512×512

```bash
cd c2i/
bash train_c2i.sh --num-gpus 8 --config configs/pix512_xl.yaml
```

#### Resume from Checkpoint

```bash
cd c2i/
bash train_c2i.sh --num-gpus 8 --config configs/pix256_xl.yaml \
  --ckpt-path /path/to/checkpoint.ckpt
```

#### `train_c2i.sh` Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | configs/pix256_xl.yaml | Config YAML path |
| `--ckpt-path` | (empty) | Checkpoint to resume from |

Auto-resume is enabled by default in the config (`auto_resume: true`). If a previous checkpoint exists in the output directory, training resumes automatically.

Checkpoints are auto-downloaded from HuggingFace if the file does not exist locally. Just pass the filename to `--ckpt_path`.

#### Training Stability: Post-Modulation for PiT Blocks

If you observe sudden loss / gradient-norm spikes during training (see [#6](https://github.com/NVlabs/PixelDiT/issues/6)), enable **post-modulation adaLN** for the pixel-level (PiT) blocks. Instead of the default 6-way pre-modulation (shift/scale/gate applied to the attention & MLP inputs), each PiT block applies a 4-way scale/shift to the attention & MLP **outputs** (no gate), which mitigates the spikes. Only the PiT blocks are affected; the patch-level blocks are unchanged.

This is controlled by `pit_adaln_post_modulation: true` in the denoiser config and is fully backward compatible (default `false`). Ready-to-use configs are provided:

```bash
cd c2i/
## ImageNet 256×256
bash train_c2i.sh --num-gpus 8 --config configs/pix256_xl_pit_post_modulation.yaml
## ImageNet 512×512
bash train_c2i.sh --num-gpus 8 --config configs/pix512_xl_pit_post_modulation.yaml
```

## Model details

PixelDiT1 combines a patch-level transformer with a pixel-level transformer.
PixelDiT2 uses a single patch-level transformer with representation grounding
from a frozen DINOv3 encoder. They share the surrounding training and sampling framework; their reported
results use the separate evaluation backends described above.

<details>
<summary>PixelDiT2 architecture, training settings and GFLOPs</summary>

#### Architecture and training settings

| | **B/16** (256px) | **L/16** (256px) | **H/16** (256px) | **H/16** (512px) |
|---|:---:|:---:|:---:|:---:|
| depth | 12 | 24 | 32 | 32 |
| hidden dim | 768 | 1024 | 1280 | 1280 |
| heads | 12 | 16 | 16 | 16 |
| patch size | 16 | 16 | 16 | 16 |
| in-context class tokens | 32 | 32 | 32 | 32 |
| in-context start block | 4 | 8 | 8 | 8 |
| SwiGLU hidden multiple | – | – | – | 32 |
| block dropout | 0 | 0 | 0.2 | 0.2 |
| total params (M) | 165.3 | 502.4 | 1007.8 | 1073.7 |
| GFLOPs | 91 | 281 | 568 | 2438 |
| grounding encoder | DINOv3-S/16 | DINOv3-S/16 | DINOv3-S/16 | DINOv3-B/16 |
| projection blocks | 1 | 1 | 1 | 1 |
| REPA encoder (training only) | DINOv2-B/14 | DINOv2-B/14 | DINOv2-B/14 | DINOv2-B/14 |
| REPA block / weight | 4 / 0.5 | 8 / 0.5 | 8 / 0.5 | 8 / 0.5 |
| optimizer | AdamW, β=(0.9, 0.95), weight decay 0 | ← | ← | ← |
| learning rate | 2e-4, constant after a 5-epoch linear warmup | ← | ← | ← |
| global batch | 1024 | 1024 | 1024 | 1024 |
| EMA decay | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| time sampler | logit-normal(μ=−0.8, σ=0.8) | ← | ← | ← |
| clip of (1−t) | 0.05 | 0.05 | 0.05 | 0.05 |
| class / grounding dropout | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 (+0.05 joint) |
| noise scale | 1.0 | 1.0 | 1.0 | 2.0 |
| precision | bf16 | bf16 | bf16 | bf16 |

There is no gradient clipping. B/16 and L/16 are provided as architecture
definitions only; no checkpoints are released for them.

The REPA teacher uses DINOv2-B/14 with bf16 weights and DINOv2 positional-embedding resampling.

GFLOPs count one conditional forward, including the frozen encoder, under
1 MAC = 2 FLOPs:

```bash
python tools/compute_flops.py --config c2i/configs/pixeldit2_h16_in256.yaml
python tools/compute_flops.py --config c2i/configs/pixeldit2_h16_in512.yaml --height 512 --width 512
```


</details>
