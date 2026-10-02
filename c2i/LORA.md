# PixelDiT2 LoRA training

This branch adds native LoRA fine-tuning for the PixelDiT2 class-to-image models without adding PEFT as a dependency.

## What is trained

The released PixelDiT2 EMA checkpoint is loaded first. The pretrained PixelDiT2 base, including the frozen DINOv3 vision encoder and its existing grounding path, is then frozen. LoRA matrices are injected into the selected PixelDiT2 transformer linear layers.

The default targets are:

```text
patch_blocks.*.attn.qkv
patch_blocks.*.attn.proj
patch_blocks.*.mlp.w1
patch_blocks.*.mlp.w2
patch_blocks.*.mlp.w3
```

The target list is configurable through `model.denoiser.init_args.lora_target_modules`. This makes it possible to experiment with LoRA on the grounding projection later without unfreezing DINOv3 itself.

The LoRA example configs disable REPA (`repa_weight: 0.0`) because a pretrained PixelDiT2 representation is already being adapted and the extra DINOv2 teacher is not required for ordinary adapter tuning.

## Configs

- `configs/pixeldit2_h16_in256_lora.yaml`
- `configs/pixeldit2_h16_in512_lora.yaml`

Both default to rank 16, alpha 16, bf16 training, and the released EMA checkpoint as the base.

## Train

From `c2i/`:

```bash
# 256x256
bash train_c2i.sh --num-nodes 1 --num-gpus 8 \
  --config configs/pixeldit2_h16_in256_lora.yaml

# 512x512
bash train_c2i.sh --num-nodes 1 --num-gpus 8 \
  --config configs/pixeldit2_h16_in512_lora.yaml
```

For one GPU:

```bash
bash train_c2i.sh --num-nodes 1 --num-gpus 1 \
  --config configs/pixeldit2_h16_in256_lora.yaml
```

The released base checkpoint is resolved through `tools/download.py` and downloaded from the NVIDIA Hugging Face repository when needed. `pretrained_checkpoint` is deliberately separate from Lightning's `--ckpt_path`: loading a released base starts a fresh LoRA run at step 0 instead of pretending the base model is a training resume.

## Outputs

Regular Lightning checkpoints are still written for exact training resume, including optimizer, scheduler, EMA and trainer progress.

The `pixdit_core.lora.LoRAAdapterCheckpoint` callback also writes adapter-only safetensors under:

```text
<run_dir>/lora_adapters/
```

For example:

```text
step-00001000.safetensors
step-00002000.safetensors
...
last.safetensors
```

The adapter files contain only LoRA A/B tensors and metadata, so they can be distributed independently of the PixelDiT2 base checkpoint.

## Resume an interrupted LoRA training run

Resume from the full Lightning checkpoint, not the adapter-only safetensors:

```bash
bash train_c2i.sh --num-nodes 1 --num-gpus 8 \
  --config configs/pixeldit2_h16_in256_lora.yaml \
  --ckpt-path /path/to/run/last.ckpt
```

The base is constructed and LoRA modules are injected before Lightning restores the full training checkpoint.

## Load an adapter for inference or a fresh continuation

Set `lora_adapter_path` while keeping the same base model, LoRA rank, and target module list:

```bash
cd c2i
python main.py predict \
  -c configs/pixeldit2_h16_in256_lora.yaml \
  --model.denoiser.init_args.lora_adapter_path=/path/to/last.safetensors \
  --trainer.devices=1 \
  --trainer.num_nodes=1 \
  --trainer.logger=false \
  --per_run_seed=false \
  --seed_everything=5
```

Do not pass the released base checkpoint through `--ckpt_path` for this case. The LoRA config already loads it through `pretrained_checkpoint` before applying the adapter.

## Change rank or targets

These fields control the adapter layout:

```yaml
model:
  denoiser:
    init_args:
      lora_rank: 16
      lora_alpha: 16.0
      lora_dropout: 0.0
      lora_target_modules:
        - patch_blocks.*.attn.qkv
        - patch_blocks.*.attn.proj
        - patch_blocks.*.mlp.w1
        - patch_blocks.*.mlp.w2
        - patch_blocks.*.mlp.w3
```

An existing adapter must be loaded with the same rank and target layout that created it.

## Grounding projection experiments

The DINOv3 encoder itself should stay frozen. If an experiment needs to adapt the trainable projection that maps DINO tokens into PixelDiT2 hidden space, add specific `dino_conditioner.dino_proj...` linear-module patterns to `lora_target_modules`. This changes the adapter layout and should be treated as a separate experiment.
