"""Flow-matching training and sampling for PixelDiT2 image editing."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .lora_diffusion import LoRAGroundedFlowTrainer
from .grounded_diffusion import GroundedSampler


class EditGroundedFlowTrainer(LoRAGroundedFlowTrainer):
    """Grounded flow matching with an independently droppable clean source image."""

    def __init__(self, source_drop_p: float = 0.15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= source_drop_p <= 1.0:
            raise ValueError("source_drop_p must be in [0, 1]")
        self.source_drop_p = float(source_drop_p)

    def preproprocess(self, x, condition, uncondition, metadata):
        x, condition, metadata = super().preproprocess(
            x, condition, uncondition, metadata
        )
        metadata = dict(metadata or {})
        if "source_image" not in metadata:
            raise KeyError("Edit training requires metadata['source_image']")
        batch = x.shape[0]
        metadata["source_drop_mask"] = (
            torch.rand(batch, device=condition.device) < self.source_drop_p
        )
        return x, condition, metadata

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        metadata = metadata or {}
        source_image = metadata.get("source_image")
        if source_image is None:
            raise KeyError("Edit training requires metadata['source_image']")

        batch = x.shape[0]
        t = torch.sigmoid(
            torch.randn(batch, device=x.device, dtype=torch.float32)
            * self.p_std
            + self.p_mean
        ).view(-1, *([1] * (x.ndim - 1)))
        noise = torch.randn_like(x) * self.noise_scale
        x_t = t * x + (1 - t) * noise
        scale = (1 - t).clamp_min(self.t_eps)
        velocity = (x - x_t) / scale

        use_repa = self.repa_weight > 0 and self.repa_encoder is not None
        features = []
        handle = None
        if use_repa:
            handle = net.patch_blocks[
                self.repa_align_layer - 1
            ].register_forward_hook(
                lambda module, inputs, output: features.append(
                    output[0] if isinstance(output, tuple) else output
                )
            )

        x_pred = net(
            x_t,
            t.flatten(),
            y,
            skip_grounding_mask=metadata.get("grounding_drop_mask"),
            source_image=source_image,
            source_drop_mask=metadata.get("source_drop_mask"),
        )
        if handle is not None:
            handle.remove()

        velocity_pred = (x_pred - x_t) / scale
        fm_loss = ((velocity - velocity_pred) ** 2).mean(
            dim=tuple(range(1, x.ndim))
        ).mean()
        losses = {"fm_loss": fm_loss, "loss": fm_loss}

        if use_repa:
            src = self._strip_context_tokens(net, features[0])
            src = self.proj(src)
            with torch.no_grad():
                dst = self.repa_encoder((x + 1.0) / 2.0, net.patch_size)
            src = self._match_token_grid(src, dst.shape[1])
            cos_loss = (
                1.0 - F.cosine_similarity(src, dst, dim=-1)
            ).mean()
            losses["cos_loss"] = cos_loss
            losses["loss"] = fm_loss + self.repa_weight * cos_loss
        return losses


class EditGroundedSampler(GroundedSampler):
    """Three-way guidance: base -> source preservation -> target edit."""

    def __init__(
        self,
        source_guidance: float = 1.0,
        edit_guidance: float = 2.4,
        t_start: float = 0.70,
        *args,
        **kwargs,
    ):
        super().__init__(guidance=edit_guidance, *args, **kwargs)
        if not 0.0 <= t_start < 1.0:
            raise ValueError("t_start must be in [0, 1)")
        self.source_guidance = float(source_guidance)
        self.edit_guidance = float(edit_guidance)
        self.t_start = float(t_start)
        self._source_tokens = None
        self._source_grid = None
        self._source_strength = 1.0

    def _guide_value(self, value, t):
        if torch.is_tensor(value):
            out = value.to(device=t.device, dtype=t.dtype)
            while out.ndim < t.ndim:
                out = out.unsqueeze(-1)
            return out
        return torch.full_like(t, float(value))

    def velocity(self, net, x, t, condition, uncondition):
        batch = x.shape[0]
        scale = (1 - t).clamp_min(self.t_eps)
        x3 = torch.cat([x, x, x], dim=0)
        t3 = torch.cat([t, t, t], dim=0)
        y3 = torch.cat([uncondition, uncondition, condition], dim=0)
        current_drop = torch.cat(
            [
                torch.ones(batch, device=x.device, dtype=torch.bool),
                torch.zeros(batch * 2, device=x.device, dtype=torch.bool),
            ]
        )
        source_drop = torch.cat(
            [
                torch.ones(batch, device=x.device, dtype=torch.bool),
                torch.zeros(batch * 2, device=x.device, dtype=torch.bool),
            ]
        )
        source_tokens = self._source_tokens.repeat(3, 1, 1)
        if torch.is_tensor(self._source_strength):
            strength = self._source_strength.to(x.device).repeat(3)
        else:
            strength = self._source_strength

        pred = net(
            x3,
            t3.flatten(),
            y3,
            skip_grounding_mask=current_drop,
            source_tokens=source_tokens,
            source_grid=self._source_grid,
            source_drop_mask=source_drop,
            source_strength=strength,
        )
        p_base, p_source, p_target = pred.chunk(3, dim=0)
        v_base = (p_base - x) / scale
        v_source = (p_source - x) / scale
        v_target = (p_target - x) / scale

        above_min = (
            t > self.guidance_interval_min
            if self.guidance_interval_min > 0
            else torch.ones_like(t, dtype=torch.bool)
        )
        inside = (t < self.guidance_interval_max) & above_min
        source_cfg = self._guide_value(self.source_guidance, t)
        edit_cfg = self._guide_value(self.edit_guidance, t)
        source_cfg = torch.where(
            inside, source_cfg, torch.ones_like(source_cfg)
        )
        edit_cfg = torch.where(
            inside, edit_cfg, torch.ones_like(edit_cfg)
        )
        return (
            v_base
            + source_cfg * (v_source - v_base)
            + edit_cfg * (v_target - v_source)
        )

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(
        self,
        net,
        noise,
        condition,
        uncondition,
        source_image,
        source_strength=1.0,
        t_start: Optional[float] = None,
        return_x_trajs: bool = False,
        return_v_trajs: bool = False,
    ):
        if return_v_trajs:
            raise NotImplementedError(
                "EditGroundedSampler does not expose velocity trajectories"
            )
        start = self.t_start if t_start is None else float(t_start)
        if not 0.0 <= start < 1.0:
            raise ValueError("t_start must be in [0, 1)")

        # Cache the clean source vision features once for the complete ODE trajectory.
        self._source_tokens, self._source_grid = net.encode_source(source_image)
        self._source_strength = source_strength
        try:
            x = (
                start * source_image
                + (1.0 - start) * noise * self.noise_scale
            )
            steps = torch.linspace(
                start,
                1.0,
                self.num_steps + 1,
                device=x.device,
                dtype=x.dtype,
            )
            steps = steps.view(-1, *([1] * x.ndim)).expand(
                -1, x.shape[0], *([-1] * (x.ndim - 1))
            )
            trajectory = [x]
            for i in range(self.num_steps - 1):
                x = self.step(
                    net,
                    x,
                    steps[i],
                    steps[i + 1],
                    condition,
                    uncondition,
                )
                trajectory.append(x)
            x = self.euler_step(
                net,
                x,
                steps[-2],
                steps[-1],
                condition,
                uncondition,
            )
            trajectory.append(x)
            return (x, trajectory) if return_x_trajs else x
        finally:
            self._source_tokens = None
            self._source_grid = None
            self._source_strength = 1.0


class EditGroundedEulerSampler(EditGroundedSampler):
    def step(self, net, x, t, t_next, condition, uncondition):
        return self.euler_step(net, x, t, t_next, condition, uncondition)


class EditGroundedHeunSampler(EditGroundedSampler):
    def step(self, net, x, t, t_next, condition, uncondition):
        v = self.velocity(net, x, t, condition, uncondition)
        v_next = self.velocity(
            net,
            x + (t_next - t) * v,
            t_next,
            condition,
            uncondition,
        )
        return x + (t_next - t) * 0.5 * (v + v_next)
