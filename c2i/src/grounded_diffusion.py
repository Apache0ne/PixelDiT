"""Flow matching, REPA alignment and sampling for PixelDiT2."""

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from .diffusion import BaseSampler, BaseTrainer
from .utils import no_grad

REPA_ENCODERS = {
    "dinov2_vitb14": ("vit_base_patch14_dinov2.lvd142m", 14, 768),
    "dinov2_vitl14": ("vit_large_patch14_dinov2.lvd142m", 14, 1024),
}

# Offset used by DINOv2 positional-embedding interpolation.
_DINOV2_INTERPOLATE_OFFSET = 0.1


def _dinov2_pos_embed(self, x):
    """Apply positional embeddings with DINOv2 bicubic resampling."""
    batch, height, width, channels = x.shape
    grid = self.patch_embed.grid_size[0]
    n_prefix = self.num_prefix_tokens
    pos = self.pos_embed.float()
    prefix_pos, patch_pos = pos[:, :n_prefix], pos[:, n_prefix:]
    patch_pos = patch_pos.reshape(1, grid, grid, channels).permute(0, 3, 1, 2)
    if (height, width) != (grid, grid):
        scale = (
            float(width + _DINOV2_INTERPOLATE_OFFSET) / grid,
            float(height + _DINOV2_INTERPOLATE_OFFSET) / grid,
        )
        patch_pos = F.interpolate(patch_pos, scale_factor=scale, mode="bicubic", antialias=False)
        if tuple(patch_pos.shape[-2:]) != (height, width):
            raise RuntimeError(
                f"positional embedding resampled to {tuple(patch_pos.shape[-2:])}, "
                f"expected {(height, width)}"
            )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, height * width, channels)
    pos = torch.cat([prefix_pos, patch_pos], dim=1).to(x.dtype)
    x = x.view(batch, -1, channels)
    prefix = [
        token.expand(batch, -1, -1)
        for token in (self.cls_token, getattr(self, "reg_token", None))
        if token is not None
    ]
    x = torch.cat(prefix + [x], dim=1)
    return self.pos_drop(x + pos)


class REPATeacher(nn.Module):
    """Frozen DINOv2 encoder for REPA alignment targets."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        bf16_weights: bool = True,
        dinov2_pos_embed: bool = True,
    ):
        import timm

        super().__init__()
        timm_name, self.patch_size, self.feature_dim = REPA_ENCODERS[model_name]
        self.encoder = timm.create_model(timm_name, pretrained=True, dynamic_img_size=True)
        no_grad(self.encoder)
        if bf16_weights:
            self.encoder.to(torch.bfloat16)
        if dinov2_pos_embed:
            if getattr(self.encoder, "no_embed_class", False) or not getattr(
                self.encoder, "dynamic_img_size", False
            ):
                raise ValueError(
                    "dinov2_pos_embed needs a timm ViT built with dynamic_img_size=True "
                    "whose positional embedding covers the class token"
                )
            self.encoder._pos_embed = types.MethodType(_dinov2_pos_embed, self.encoder)
        self.bf16_weights = bf16_weights
        self.dinov2_pos_embed = dinov2_pos_embed

    def train(self, mode: bool = True):
        # Keep the frozen encoder in evaluation mode.
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images, denoiser_patch_size: int):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(images)
        height, width = x.shape[-2:]
        x = F.interpolate(
            x,
            (
                int(self.patch_size * height / denoiser_patch_size),
                int(self.patch_size * width / denoiser_patch_size),
            ),
            mode="bicubic",
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = self.encoder.forward_features(x)
        prefix = int(getattr(self.encoder, "num_prefix_tokens", 0))
        return features[:, prefix:, :].to(torch.bfloat16)


class GroundedFlowTrainer(BaseTrainer):
    """Velocity-space flow matching with REPA alignment and grounding dropout."""

    def __init__(
        self,
        p_mean: float = -0.8,
        p_std: float = 0.8,
        noise_scale: float = 1.0,
        t_eps: float = 0.05,
        grounding_drop_mode: str = "independent",
        grounding_drop_p: Optional[float] = None,
        grounding_joint_drop_p: float = 0.0,
        repa_weight: float = 0.5,
        repa_align_layer: int = 8,
        repa_encoder: Optional[nn.Module] = None,
        proj_denoiser_dim: int = 1280,
        proj_hidden_dim: int = 1280,
        proj_encoder_dim: int = 768,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if grounding_drop_mode not in ("independent", "coupled"):
            raise ValueError(f"unknown grounding_drop_mode: {grounding_drop_mode}")
        self.p_mean = p_mean
        self.p_std = p_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.grounding_drop_mode = grounding_drop_mode
        self.grounding_drop_p = (
            self.null_condition_p if grounding_drop_p is None else grounding_drop_p
        )
        self.grounding_joint_drop_p = grounding_joint_drop_p
        self.repa_weight = repa_weight
        self.repa_align_layer = repa_align_layer
        self.repa_encoder = repa_encoder
        if repa_encoder is not None:
            no_grad(repa_encoder)
        self.proj = nn.Sequential(
            nn.Linear(proj_denoiser_dim, proj_hidden_dim),
            nn.SiLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.SiLU(),
            nn.Linear(proj_hidden_dim, proj_encoder_dim),
        )

    def preproprocess(self, x, condition, uncondition, metadata):
        batch = x.shape[0]
        device = condition.device
        class_drop = torch.rand(batch, device=device) < self.null_condition_p

        if self.grounding_drop_mode == "coupled":
            grounding_drop = class_drop
        else:
            grounding_drop = torch.rand(batch, device=device) < self.grounding_drop_p
            if self.grounding_joint_drop_p > 0:
                joint = torch.rand(batch, device=device) < self.grounding_joint_drop_p
                class_drop = class_drop | joint
                grounding_drop = grounding_drop | joint

        condition = torch.where(class_drop, uncondition, condition)
        metadata = dict(metadata or {})
        metadata["grounding_drop_mask"] = grounding_drop
        return x, condition, metadata

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        batch = x.shape[0]
        t = torch.sigmoid(
            torch.randn(batch, device=x.device, dtype=torch.float32) * self.p_std + self.p_mean
        ).view(-1, *([1] * (x.ndim - 1)))
        noise = torch.randn_like(x) * self.noise_scale
        x_t = t * x + (1 - t) * noise
        scale = (1 - t).clamp_min(self.t_eps)
        velocity = (x - x_t) / scale

        use_repa = self.repa_weight > 0 and self.repa_encoder is not None
        features = []
        handle = None
        if use_repa:
            handle = net.patch_blocks[self.repa_align_layer - 1].register_forward_hook(
                lambda module, inputs, output: features.append(
                    output[0] if isinstance(output, tuple) else output
                )
            )

        x_pred = net(
            x_t,
            t.flatten(),
            y,
            skip_grounding_mask=metadata.get("grounding_drop_mask"),
        )
        if handle is not None:
            handle.remove()

        velocity_pred = (x_pred - x_t) / scale
        fm_loss = ((velocity - velocity_pred) ** 2).mean(dim=tuple(range(1, x.ndim))).mean()
        losses = {"fm_loss": fm_loss, "loss": fm_loss}

        if use_repa:
            src = self._strip_context_tokens(net, features[0])
            src = self.proj(src)
            with torch.no_grad():
                dst = self.repa_encoder((x + 1.0) / 2.0, net.patch_size)
            src = self._match_token_grid(src, dst.shape[1])
            cos_loss = (1.0 - F.cosine_similarity(src, dst, dim=-1)).mean()
            losses["cos_loss"] = cos_loss
            losses["loss"] = fm_loss + self.repa_weight * cos_loss
        return losses

    def _strip_context_tokens(self, net, feature):
        context_len = int(getattr(net, "in_context_len", 0))
        context_start = int(getattr(net, "in_context_start", 0))
        if context_len > 0 and (self.repa_align_layer - 1) >= context_start:
            return feature[:, context_len:, :]
        return feature

    @staticmethod
    def _match_token_grid(src, target_len):
        if src.shape[1] == target_len:
            return src
        batch, length, channels = src.shape
        side, target_side = int(length ** 0.5), int(target_len ** 0.5)
        spatial = src.view(batch, side, side, channels).permute(0, 3, 1, 2)
        if target_len < length:
            spatial = F.adaptive_avg_pool2d(spatial, (target_side, target_side))
        else:
            spatial = F.interpolate(
                spatial, size=(target_side, target_side), mode="bilinear", align_corners=False
            )
        return spatial.permute(0, 2, 3, 1).reshape(batch, target_len, channels)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.proj.state_dict(
            destination=destination, prefix=prefix + "proj.", keep_vars=keep_vars
        )


class GroundedSampler(BaseSampler):
    """ODE sampler with interval-restricted classifier-free guidance."""

    def __init__(
        self,
        num_steps: int = 50,
        guidance: float = 2.4,
        guidance_interval_min: float = 0.1,
        guidance_interval_max: float = 0.9,
        noise_scale: float = 1.0,
        t_eps: float = 0.05,
        *args,
        **kwargs,
    ):
        super().__init__(num_steps=num_steps, guidance=guidance, *args, **kwargs)
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max
        self.noise_scale = noise_scale
        self.t_eps = t_eps

    def velocity(self, net, x, t, condition, uncondition):
        scale = (1 - t).clamp_min(self.t_eps)
        v_cond = (net(x, t.flatten(), condition) - x) / scale
        v_uncond = (net(x, t.flatten(), uncondition, skip_grounding=True) - x) / scale
        above_min = t > self.guidance_interval_min if self.guidance_interval_min > 0 else torch.ones_like(t, dtype=torch.bool)
        inside = (t < self.guidance_interval_max) & above_min
        guidance = torch.where(inside, torch.full_like(t, self.guidance), torch.ones_like(t))
        return v_uncond + guidance * (v_cond - v_uncond)

    def euler_step(self, net, x, t, t_next, condition, uncondition):
        return x + (t_next - t) * self.velocity(net, x, t, condition, uncondition)

    def step(self, net, x, t, t_next, condition, uncondition):
        raise NotImplementedError

    def forward(self, net, noise, condition, uncondition, return_x_trajs=False, return_v_trajs=False):
        if return_v_trajs:
            raise NotImplementedError("GroundedSampler does not expose velocity trajectories")
        return super().forward(net, noise, condition, uncondition, return_x_trajs=return_x_trajs)

    def _impl_sampling(self, net, noise, condition, uncondition):
        x = noise * self.noise_scale
        steps = torch.linspace(0.0, 1.0, self.num_steps + 1, device=x.device, dtype=x.dtype)
        steps = steps.view(-1, *([1] * x.ndim)).expand(-1, x.shape[0], *([-1] * (x.ndim - 1)))
        trajectory = [x]
        for i in range(self.num_steps - 1):
            x = self.step(net, x, steps[i], steps[i + 1], condition, uncondition)
            trajectory.append(x)
        x = self.euler_step(net, x, steps[-2], steps[-1], condition, uncondition)
        trajectory.append(x)
        return trajectory, None


class GroundedEulerSampler(GroundedSampler):
    def step(self, net, x, t, t_next, condition, uncondition):
        return self.euler_step(net, x, t, t_next, condition, uncondition)


class GroundedHeunSampler(GroundedSampler):
    def step(self, net, x, t, t_next, condition, uncondition):
        v = self.velocity(net, x, t, condition, uncondition)
        v_next = self.velocity(net, x + (t_next - t) * v, t_next, condition, uncondition)
        return x + (t_next - t) * 0.5 * (v + v_next)
