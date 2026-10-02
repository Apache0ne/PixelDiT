# PixelDiT2 image-to-image editing

This branch provides two separate editing paths. They are intentionally not mixed:

1. **Training-free PnP editing** — the primary zero-shot editor for the released H/16 checkpoint.
2. **Trainable source-DINO adapter** — LoRA + source projection + gates for later distillation/fine-tuning.

The original C2I implementation on `exp` is unchanged.

## Why the first partial-noise editor was removed

The original experiment initialized the target trajectory from a partially noised source image while keeping PixelDiT2's normal DINO grounding active. Real-image tests showed the expected failure: high source strength reconstructed the original object almost exactly, while stronger noising damaged the source without reliably changing its class.

The source pixels and frozen-DINO semantics were both pushing the target trajectory back toward the original class. This is not the zero-shot editing path anymore.

## Primary zero-shot method: PnP structural attention

`src.pnp_pixeldit2.PixelDiT2PnPEditSampler` starts the target trajectory from **pure noise**, exactly like normal PixelDiT2 generation.

A synchronized source trajectory is available analytically from PixelDiT2's training path:

```text
x_source(t) = t * source + (1 - t) * noise
```

At early/middle sampling times the source branch and target branch are run through the same frozen PixelDiT2. Selected self-attention blocks use:

```text
Q_target <- blend(Q_target, Q_source)
K_target <- blend(K_target, K_source)
V_target <- V_target
```

The target branch keeps its own:

- target class embedding
- target AdaLN modulation
- target attention values V
- target MLP activations
- target final layer

After class-context tokens are inserted, their Q/K remain target-owned; only spatial patch-token Q/K are injected from the source.

This is the transformer adaptation of Plug-and-Play diffusion feature/self-attention injection: source attention geometry preserves layout, while target values/conditioning are free to change appearance and class.

The same frozen DINOv3 used by PixelDiT2 remains active on the synchronized source trajectory and on the conditional target trajectory. Nothing is trained for this mode.

### Default zero-shot config

```text
c2i/configs/pixeldit2_h16_in256_pnp_edit.yaml
```

It uses the released `PixelDiT2` class directly:

```text
PixelDiT2 H/16 256
official EMA checkpoint
LoRA rank = 0
no source projector
no source gates
no trainable edit weights
```

Default controls:

```yaml
num_steps: 40
guidance: 2.4
qk_injection_strength: 0.90
qk_injection_until: 0.72
qk_fade_start: 0.45
qk_block_start: 0
qk_block_end: 20
solver: heun
```

## Inference manifest

```json
{
  "source": "images/horse.jpg",
  "source_class": 339,
  "target_class": 340,
  "seed": 1234,
  "t_start": 0.72,
  "structure_strength": 1.0,
  "edit_strength": 1.0,
  "filename": "horse_to_zebra"
}
```

`source_class` is optional. A missing or negative value uses the null ImageNet class while retaining the source branch's frozen-DINO grounding. Supplying the correct source class is preferred for class-to-class ImageNet edits.

### Controls

`target_class`
: Desired ImageNet class.

`t_start`
: Legacy field retained for compatibility. In the PnP editor it means **structure injection cutoff**, not partial-noise strength. Larger values preserve source layout for more of the target trajectory.

`structure_strength`
: Scales source Q/K injection. `1.0` is the default. Lower values release source geometry; higher values preserve it more strongly.

`edit_strength`
: Scales classifier-free target guidance around guidance=1. `1.0` gives the configured CFG value, `0.0` reduces it to ordinary conditional scale 1.

## Trainable direct editor

The separate `PixelDiT2Edit` class remains for a future fast direct editor:

```text
noisy target state -> frozen DINOv3 -> pretrained P_g --------+
                                                               +-> spatial AdaLN -> PixelDiT2
clean source ------> same frozen DINOv3 -> P_src -> gates -----+
```

The source projector and gates are **not** expected to edit before training. Their zero initialization is deliberately exact-base behavior.

The training recipe is:

```text
c2i/configs/pixeldit2_h16_in256_edit_lora.yaml
```

Trainable parameters are transformer LoRA + source projector + source gates while PixelDiT2 base and DINOv3 remain frozen. A practical later path is to distill successful PnP/paired edits into this adapter.

## Validation

PnP regression harness:

```bash
python tools/test_pixeldit2_pnp_colab.py
```

It verifies:

- zero Q/K injection reproduces normal PixelDiT2
- source changes have no effect with injection disabled
- source changes do affect target generation with Q/K control enabled
- outputs remain finite
- the editor changes no model parameter
- the zero-shot config contains no LoRA or untrained source adapter

The older FlowEdit experiment remains in `src/flowedit_pixeldit2.py` for comparison, but it is no longer the default editor.

## Important files

- `c2i/src/pnp_pixeldit2.py` — primary zero-shot PnP attention editor
- `c2i/configs/pixeldit2_h16_in256_pnp_edit.yaml` — clean released-model PnP config
- `c2i/src/edit_data.py` — source/target manifest loader and independent controls
- `c2i/src/edit_lightning.py` — edit prediction wrapper
- `tools/test_pixeldit2_pnp_colab.py` — PnP regression tests
- `pixdit_core/pixeldit2_edit.py` — trainable source-DINO adapter model
- `pixdit_core/edit_adapter.py` — adapter serialization
- `c2i/configs/pixeldit2_h16_in256_edit_lora.yaml` — trainable direct-editor recipe
