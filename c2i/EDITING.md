# PixelDiT2 image-to-image editing

This branch adds an opt-in `PixelDiT2Edit` path on top of the released PixelDiT2 class-to-image model. The original C2I model is unchanged.

## Architecture

`PixelDiT2Edit` keeps PixelDiT2's existing noisy-image representation grounding and reuses the exact same frozen DINOv3 encoder for the clean source image:

```text
noisy state x_t -> frozen DINOv3 -> pretrained P_g -----------+
                                                               +-> spatial AdaLN -> PixelDiT2
clean source ----> same frozen DINOv3 -> source P_src -> gates +
```

- DINOv3 remains frozen.
- `P_src` defaults to a clone of the released checkpoint's pretrained `P_g`.
- Source gates are zero initialized, per block and per channel.
- With zero source gates the edit model is exactly the pretrained C2I model.
- Transformer LoRA remains supported.
- The source encoder is evaluated once during sampling and its raw DINO tokens are cached for the full ODE trajectory.

The default H/16 recipe trains the LoRA tensors + `P_src` + roughly 41K source gates while leaving the ~1B-parameter base and DINO encoder frozen.

## Training data

`PairedEditDataset` reads JSONL rows:

```json
{"source":"pairs/0001_source.png","target":"pairs/0001_target.png","source_class":207,"target_class":281}
```

`source` and `target` should preserve geometry/composition while the target contains the desired edit. Teacher-generated FlowEdit/DecFlowEdit pairs are a practical way to bootstrap this dataset from ImageNet.

The target image is used by the normal PixelDiT2 flow-matching objective. The source is conditioning only. The trainer independently drops class, normal noisy-image grounding, and clean-source conditioning so the sampler can form three-way guidance.

## Train

Single GPU / Colab:

```bash
cd c2i
bash train_edit.sh --num-gpus 1 --config configs/pixeldit2_h16_in256_edit_lora.yaml
```

Regular Lightning checkpoints still support exact resume. Small deployable edit adapters are written to `<run>/edit_adapters/` and contain LoRA + source projection + source gates, not the frozen PixelDiT2/DINO base.

## Inference manifest

```json
{"source":"inputs/dog.png","target_class":281,"seed":1234,"t_start":0.70,"source_strength":1.0}
```

Run prediction through `main_edit.py predict`. All samples in one prediction batch must share `t_start`. Use batch size 1 or separate manifests for different start strengths.

The sampler starts from:

```text
x_start = t_start * source + (1 - t_start) * noise
```

Higher `t_start` preserves more source pixels. Lower `t_start` permits larger changes.

## Three-way edit guidance

The edit sampler predicts:

```text
v_base   = null target, no source
v_source = null target, source enabled
v_target = target class, source enabled
```

and combines them as:

```text
v = v_base
  + source_guidance * (v_source - v_base)
  + edit_guidance   * (v_target - v_source)
```

`source_guidance` controls source preservation and `edit_guidance` controls target-class pressure independently.

## Important files

- `pixdit_core/pixeldit2_edit.py` — clean-source DINO edit model
- `pixdit_core/edit_adapter.py` — edit adapter save/load/checkpoint callback
- `c2i/src/edit_diffusion.py` — edit flow trainer + Euler/Heun samplers
- `c2i/src/edit_data.py` — paired training and prediction datasets
- `c2i/src/edit_lightning.py` — metadata-aware prediction wrapper
- `c2i/main_edit.py` — Lightning CLI entry point
- `c2i/train_edit.sh` — training launcher
- `c2i/configs/pixeldit2_h16_in256_edit_lora.yaml` — H/16 256 recipe
