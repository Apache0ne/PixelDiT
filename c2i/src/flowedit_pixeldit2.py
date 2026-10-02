"""Training-free source->target editing for PixelDiT2 using FlowEdit-style velocity transport.

PixelDiT2 uses data-time t with

    x_t = t * x_data + (1 - t) * noise

and its denoiser output is converted to data-direction velocity by

    v = (x_pred - x_t) / (1 - t).

FlowEdit transports an edit image directly by integrating the difference between
target and source velocities while evaluating both on synchronized noisy source
and target states. This module adapts that construction to PixelDiT2's time
convention and class conditioning.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PixelDiT2FlowEditSampler(nn.Module):
    """Inversion-free class-to-class editing for pretrained PixelDiT2.

    The sampler does not require a trained source adapter. It uses the released
    PixelDiT2 model itself as both the source and target vector field and updates
    a clean edit state with ``v_target - v_source``.

    ``source_condition`` is optional. If omitted (or negative), the source branch
    uses PixelDiT2's null class while retaining the normal frozen-DINO grounding,
    which gives an image-only source condition.

    ``localization='velocity_prior'`` is an experimental DecFlowEdit-inspired
    option. It temporally aggregates the no-CFG source/target velocity difference
    and uses that map to attenuate updates outside edit-relevant regions. It is
    deliberately named generically because it is not a line-for-line reproduction
    of DecFlowEdit.
    """

    requires_source_condition = True

    def __init__(
        self,
        num_steps: int = 28,
        n_avg: int = 1,
        source_guidance: float = 1.0,
        target_guidance: float = 3.0,
        edit_t_min: float = 0.35,
        edit_t_max: float = 0.95,
        edit_strength: float = 1.0,
        noise_scale: float = 1.0,
        t_eps: float = 0.05,
        terminal_steps: int = 0,
        localization: str = "velocity_prior",
        localization_strength: float = 0.75,
        localization_ema: float = 0.85,
        localization_quantile: float = 0.80,
        localization_floor: float = 0.20,
        localization_power: float = 1.0,
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.n_avg = int(n_avg)
        self.source_guidance = float(source_guidance)
        self.target_guidance = float(target_guidance)
        self.edit_t_min = float(edit_t_min)
        self.edit_t_max = float(edit_t_max)
        self.edit_strength = float(edit_strength)
        self.noise_scale = float(noise_scale)
        self.t_eps = float(t_eps)
        self.terminal_steps = int(terminal_steps)
        self.localization = str(localization)
        self.localization_strength = float(localization_strength)
        self.localization_ema = float(localization_ema)
        self.localization_quantile = float(localization_quantile)
        self.localization_floor = float(localization_floor)
        self.localization_power = float(localization_power)
        self.last_diagnostics = {}

        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        if self.n_avg <= 0:
            raise ValueError("n_avg must be > 0")
        if not 0.0 <= self.edit_t_min < self.edit_t_max <= 1.0:
            raise ValueError("need 0 <= edit_t_min < edit_t_max <= 1")
        if self.terminal_steps < 0:
            raise ValueError("terminal_steps must be >= 0")
        if self.localization not in ("none", "velocity_prior"):
            raise ValueError("localization must be 'none' or 'velocity_prior'")
        if not 0.0 <= self.localization_strength <= 1.0:
            raise ValueError("localization_strength must be in [0,1]")
        if not 0.0 <= self.localization_ema < 1.0:
            raise ValueError("localization_ema must be in [0,1)")
        if not 0.0 < self.localization_quantile <= 1.0:
            raise ValueError("localization_quantile must be in (0,1]")
        if not 0.0 <= self.localization_floor <= 1.0:
            raise ValueError("localization_floor must be in [0,1]")
        if self.localization_power <= 0:
            raise ValueError("localization_power must be > 0")

    @staticmethod
    def _as_batch_condition(value, reference, null_condition):
        batch = reference.shape[0]
        if value is None:
            return null_condition
        if torch.is_tensor(value):
            value = value.to(device=reference.device, dtype=torch.long).view(-1)
            if value.numel() == 1 and batch > 1:
                value = value.expand(batch)
            if value.numel() != batch:
                raise ValueError(
                    f"source_condition has {value.numel()} items for batch {batch}"
                )
            # Negative source classes mean image-only source conditioning:
            # keep DINO grounding, substitute the null class label.
            return torch.where(value >= 0, value, null_condition)
        ivalue = int(value)
        if ivalue < 0:
            return null_condition
        return torch.full(
            (batch,), ivalue, device=reference.device, dtype=torch.long
        )

    @staticmethod
    def _strength_tensor(value, reference):
        batch = reference.shape[0]
        if torch.is_tensor(value):
            out = value.to(device=reference.device, dtype=torch.float32).view(-1)
            if out.numel() == 1 and batch > 1:
                out = out.expand(batch)
            if out.numel() != batch:
                raise ValueError(
                    f"edit strength has {out.numel()} items for batch {batch}"
                )
        else:
            out = torch.full(
                (batch,), float(value), device=reference.device, dtype=torch.float32
            )
        return out.view(batch, 1, 1, 1)

    @staticmethod
    def _deterministic_step_noise(base_noise, step_index: int, avg_index: int):
        """Create deterministic Gaussian-marginal noise from the provided seed noise.

        A spatial/channel permutation of an iid Gaussian tensor is still Gaussian.
        This lets prediction manifests remain deterministic without needing a
        separate RNG object in Lightning's sampler interface.
        """
        _, channels, height, width = base_noise.shape
        shift_y = (17 * step_index + 7 * avg_index) % max(height, 1)
        shift_x = (29 * step_index + 11 * avg_index) % max(width, 1)
        channel_shift = (step_index + 2 * avg_index) % max(channels, 1)
        noise = torch.roll(
            base_noise,
            shifts=(channel_shift, shift_y, shift_x),
            dims=(1, 2, 3),
        )
        if (step_index + avg_index) & 1:
            noise = -noise
        return noise

    def _paired_velocities(
        self,
        net,
        x_source,
        x_target,
        t,
        source_condition,
        target_condition,
        uncondition,
    ):
        """Evaluate source/target conditional and unconditional fields in one pass."""
        batch = x_source.shape[0]
        x4 = torch.cat([x_source, x_source, x_target, x_target], dim=0)
        t4 = torch.cat([t, t, t, t], dim=0)
        y4 = torch.cat(
            [source_condition, uncondition, target_condition, uncondition], dim=0
        )
        skip_grounding = torch.cat(
            [
                torch.zeros(batch, device=x_source.device, dtype=torch.bool),
                torch.ones(batch, device=x_source.device, dtype=torch.bool),
                torch.zeros(batch, device=x_source.device, dtype=torch.bool),
                torch.ones(batch, device=x_source.device, dtype=torch.bool),
            ],
            dim=0,
        )

        pred = net(
            x4,
            t4,
            y4,
            skip_grounding_mask=skip_grounding,
        )
        p_src_cond, p_src_uncond, p_tar_cond, p_tar_uncond = pred.chunk(4, dim=0)

        scale = (1.0 - t).clamp_min(self.t_eps).view(batch, 1, 1, 1)
        v_src_cond = (p_src_cond - x_source) / scale
        v_src_uncond = (p_src_uncond - x_source) / scale
        v_tar_cond = (p_tar_cond - x_target) / scale
        v_tar_uncond = (p_tar_uncond - x_target) / scale

        v_source = v_src_uncond + self.source_guidance * (
            v_src_cond - v_src_uncond
        )
        v_target = v_tar_uncond + self.target_guidance * (
            v_tar_cond - v_tar_uncond
        )
        return v_source, v_target, v_src_cond, v_tar_cond

    def _target_velocity(self, net, x, t, target_condition, uncondition):
        batch = x.shape[0]
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        y2 = torch.cat([target_condition, uncondition], dim=0)
        skip = torch.cat(
            [
                torch.zeros(batch, device=x.device, dtype=torch.bool),
                torch.ones(batch, device=x.device, dtype=torch.bool),
            ]
        )
        pred_cond, pred_uncond = net(
            x2,
            t2,
            y2,
            skip_grounding_mask=skip,
        ).chunk(2, dim=0)
        scale = (1.0 - t).clamp_min(self.t_eps).view(batch, 1, 1, 1)
        v_cond = (pred_cond - x) / scale
        v_uncond = (pred_uncond - x) / scale
        return v_uncond + self.target_guidance * (v_cond - v_uncond)

    def _localization_mask(self, prior):
        flat = prior.flatten(1)
        # quantile is evaluated in fp32 to avoid bf16 edge cases.
        q = torch.quantile(
            flat.float(), self.localization_quantile, dim=1, keepdim=True
        ).clamp_min(1e-6)
        q = q.view(-1, 1, 1, 1)
        normalized = (prior.float() / q).clamp(0.0, 1.0)
        normalized = normalized.pow(self.localization_power)
        mask = self.localization_floor + (1.0 - self.localization_floor) * normalized
        # Blend the localized update with the original update. 0 = pure FlowEdit.
        return 1.0 + self.localization_strength * (mask - 1.0)

    @torch.inference_mode()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(
        self,
        net,
        noise,
        condition,
        uncondition,
        source_image,
        source_condition: Optional[torch.Tensor] = None,
        source_strength=1.0,
        edit_strength=None,
        t_start: Optional[float] = None,
        return_x_trajs: bool = False,
        return_v_trajs: bool = False,
    ):
        if return_v_trajs:
            raise NotImplementedError(
                "PixelDiT2FlowEditSampler does not expose velocity trajectories"
            )
        if source_image.shape != noise.shape:
            raise ValueError(
                f"source/noise shape mismatch: {source_image.shape} vs {noise.shape}"
            )

        batch = source_image.shape[0]
        target_condition = condition.to(source_image.device, dtype=torch.long).view(batch)
        uncondition = uncondition.to(source_image.device, dtype=torch.long).view(batch)
        source_condition = self._as_batch_condition(
            source_condition, source_image, uncondition
        )

        # ``t_start`` is accepted as a backward-compatible per-record override.
        # Under FlowEdit it means the first data-time at which transport is active,
        # not the partial-noise SDEdit start used by the old sampler.
        edit_t_min = self.edit_t_min if t_start is None else float(t_start)
        if not 0.0 <= edit_t_min < self.edit_t_max:
            raise ValueError(
                f"FlowEdit t_start/edit_t_min must be < edit_t_max={self.edit_t_max}"
            )

        strength_value = source_strength if edit_strength is None else edit_strength
        strength = self._strength_tensor(strength_value, source_image)
        strength = strength * self.edit_strength

        z_edit = source_image.clone()
        trajectory = [z_edit.clone()] if return_x_trajs else None
        prior = None
        delta_norm_acc = 0.0
        mask_mean_acc = 0.0

        steps = torch.linspace(
            edit_t_min,
            self.edit_t_max,
            self.num_steps + 1,
            device=source_image.device,
            dtype=torch.float32,
        )

        for i in range(self.num_steps):
            t_value = steps[i]
            t_next_value = steps[i + 1]
            dt = (t_next_value - t_value).float()
            t = torch.full(
                (batch,),
                float(t_value),
                device=source_image.device,
                dtype=torch.float32,
            )
            delta_avg = torch.zeros_like(source_image)

            for k in range(self.n_avg):
                eps = self._deterministic_step_noise(noise, i, k) * self.noise_scale
                t_image = t.view(batch, 1, 1, 1).to(source_image.dtype)
                x_source = t_image * source_image + (1.0 - t_image) * eps
                x_target = z_edit + x_source - source_image

                v_source, v_target, v_src_raw, v_tar_raw = self._paired_velocities(
                    net,
                    x_source,
                    x_target,
                    t,
                    source_condition,
                    target_condition,
                    uncondition,
                )
                delta = v_target - v_source

                if self.localization == "velocity_prior":
                    local_signal = (
                        v_tar_raw.float() - v_src_raw.float()
                    ).square().mean(dim=1, keepdim=True).sqrt()
                    if prior is None:
                        prior = local_signal
                    else:
                        prior = (
                            self.localization_ema * prior
                            + (1.0 - self.localization_ema) * local_signal
                        )
                    mask = self._localization_mask(prior).to(delta.dtype)
                    delta = delta * mask
                    mask_mean_acc += float(mask.float().mean().cpu())

                delta_avg.add_(delta, alpha=1.0 / self.n_avg)

            delta_norm_acc += float(delta_avg.float().square().mean().sqrt().cpu())
            z_edit = z_edit.float() + dt * strength * delta_avg.float()
            z_edit = z_edit.to(source_image.dtype)
            if return_x_trajs:
                trajectory.append(z_edit.clone())

        # Optional final target-only generation, matching FlowEdit's n_min idea.
        if self.terminal_steps > 0:
            t0 = self.edit_t_max
            eps = self._deterministic_step_noise(noise, self.num_steps, 0) * self.noise_scale
            t_image = torch.full(
                (batch, 1, 1, 1),
                t0,
                device=source_image.device,
                dtype=source_image.dtype,
            )
            x_source = t_image * source_image + (1.0 - t_image) * eps
            x = z_edit + x_source - source_image
            terminal_times = torch.linspace(
                t0,
                1.0,
                self.terminal_steps + 1,
                device=source_image.device,
                dtype=torch.float32,
            )
            for i in range(self.terminal_steps):
                t_value = terminal_times[i]
                t_next_value = terminal_times[i + 1]
                t = torch.full(
                    (batch,),
                    float(t_value),
                    device=source_image.device,
                    dtype=torch.float32,
                )
                dt = (t_next_value - t_value).float()
                v = self._target_velocity(
                    net, x, t, target_condition, uncondition
                )
                x = (x.float() + dt * v.float()).to(source_image.dtype)
                if return_x_trajs:
                    trajectory.append(x.clone())
            z_edit = x

        denom = max(self.num_steps * self.n_avg, 1)
        self.last_diagnostics = {
            "edit_t_min": float(edit_t_min),
            "edit_t_max": float(self.edit_t_max),
            "steps": int(self.num_steps),
            "n_avg": int(self.n_avg),
            "source_guidance": float(self.source_guidance),
            "target_guidance": float(self.target_guidance),
            "mean_delta_rms": float(delta_norm_acc / max(self.num_steps, 1)),
            "mean_localization_mask": (
                float(mask_mean_acc / denom)
                if self.localization == "velocity_prior"
                else 1.0
            ),
            "terminal_steps": int(self.terminal_steps),
        }
        return (z_edit, trajectory) if return_x_trajs else z_edit
