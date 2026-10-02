# PixelDiT2 image-to-image editing

This branch adds an opt-in image-editing path on top of the released PixelDiT2 class-to-image model. The original C2I model is unchanged.

## Why the first zero-shot method was replaced

The first implementation used SDEdit-style partial noising:

```text
x_start = t_start * source + (1 - t_start) * noise
```

followed by target-class sampling. Real-image tests showed the wrong behavior: high `t_start` mostly reconstructed the input, while lower `t_start` damaged the source without reliably moving it toward the target class. PixelDiT2's frozen-DINO grounding strongly reinforces the current image semantics, so simply changing the class label is not a strong enough edit direction.

The default zero-shot sampler is now a FlowEdit-style source-to-target transport.

## Zero-shot FlowEdit transport

PixelDiT2 is trained with data-time

```text
x_t = t * x_data + (1 - t) * noise
```

and its predicted data-direction velocity is

```text
v(x_t,t,c) = (x_pred - x_t) / (1 - t)
```

For an input image `x_src`, source class `c_src`, and target class `c_tar`, the new sampler keeps a clean edit state `z_edit`, initialized to the source image. At each time it creates synchronized source/target states:

```text
x_src_t = t * x_src + (1 - t) * noise
x_tar_t = z_edit + x_src_t - x_src
```

then evaluates both PixelDiT2 vector fields and integrates their difference:

```text
delta_v = v_target(x_tar_t, t, c_tar) - v_source(x_src_t, t, c_src)
z_edit  = z_edit + dt * delta_v
```

This is inversion-free and does not need a trained edit adapter.

The source class is optional. If it is missing or negative, the source branch uses the null class but **keeps PixelDiT2's frozen-DINO grounding active**, giving an image-only source condition.

## Guidance

Each source/target field is evaluated with PixelDiT2's native CFG definition. The default recipe uses:

```text
source_guidance = 1.0
target_guidance = 3.0
```

Source and target conditional/unconditional predictions are packed into one model forward per FlowEdit sample.

## Velocity-prior localization

`PixelDiT2FlowEditSampler` also provides `localization: velocity_prior`.

It computes the source/target velocity difference **without asymmetric CFG**, temporally aggregates its per-pixel magnitude, and uses that prior to attenuate the normal guided update outside edit-relevant areas. This follows the guidance-decoupling idea behind newer FlowEdit localization work, but is intentionally described as a PixelDiT2-specific velocity prior rather than an exact reproduction of another implementation.

Set:

```yaml
localization: none
```

for plain FlowEdit.

## Trainable source-DINO adapter

The branch still includes `PixelDiT2Edit` for later distillation/fine-tuning:

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
- The deployable edit adapter contains LoRA + `P_src` + source gates, not the frozen base.

The intended training workflow is now to use the zero-shot FlowEdit sampler as a teacher to generate source/target pairs, then distill those edits into this smaller direct source-conditioned adapter.

## Training data

`PairedEditDataset` reads JSONL rows:

```json
{"source":"pairs/0001_source.png","target":"pairs/0001_target.png","source_class":207,"target_class":281}
```

The target image is used by the normal PixelDiT2 flow-matching objective. The clean source is conditioning only.

## Train

Single GPU / Colab:

```bash
cd c2i
bash train_edit.sh --num-gpus 1 --config configs/pixeldit2_h16_in256_edit_lora.yaml
```

Regular Lightning checkpoints support exact resume. Small deployable edit adapters are written to `<run>/edit_adapters/`.

## Inference manifest

Recommended FlowEdit manifest:

```json
{"source":"inputs/dog.png","source_class":207,"target_class":282,"seed":1234,"t_start":0.35,"edit_strength":1.0}
```

`source_class` is optional. `t_start` is retained as a compatibility key, but under FlowEdit it now means the **first PixelDiT data-time at which transport is integrated**, not the amount of source pixels mixed with noise.

Lower `t_start` exposes the transport to a larger/noisier part of the vector field and generally allows more structural change. Higher `t_start` is more conservative.

## Important files

- `c2i/src/flowedit_pixeldit2.py` — default training-free source-to-target FlowEdit sampler
- `pixdit_core/pixeldit2_edit.py` — trainable clean-source DINO edit model
- `pixdit_core/edit_adapter.py` — edit adapter save/load/checkpoint callback
- `c2i/src/edit_diffusion.py` — paired edit trainer + legacy partial-noise samplers for ablation
- `c2i/src/edit_data.py` — paired training and prediction datasets
- `c2i/src/edit_lightning.py` — metadata-aware prediction wrapper
- `tools/test_pixeldit2_flowedit_colab.py` — focused FlowEdit transport validation
- `c2i/main_edit.py` — Lightning CLI entry point
- `c2i/train_edit.sh` — training launcher
- `c2i/configs/pixeldit2_h16_in256_edit_lora.yaml` — H/16 256 recipe
