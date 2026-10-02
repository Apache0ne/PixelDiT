"""Training-free PixelDiT2 image editing via structural self-attention injection.

The target trajectory starts from pure noise exactly like normal PixelDiT2 sampling.
A synchronized analytically-noised source trajectory is run through the same frozen
model. During early/mid denoising, source self-attention Q/K are blended into the
target branches while target V, target AdaLN modulation and target class conditioning
remain untouched. This preserves source layout without forcing source appearance or
class identity into the generated target.

This is the transformer analogue of Plug-and-Play diffusion feature/self-attention
injection, adapted to PixelDiT2's pixel-space rectified-flow parameterization.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from pixdit_core.modules import apply_adaln, apply_rotary_emb


class PixelDiT2PnPEditSampler(nn.Module):
    """Source-structure / target-semantics editor for pretrained PixelDiT2.

    Source information is used only to control self-attention geometry. The target
    image is generated from noise, so target-class semantics are not trapped by
    source pixels or source DINO semantics.

    ``t_start`` is retained as the per-example structure-injection cutoff for
    compatibility with the existing edit manifests: larger values preserve source
    structure for more of the trajectory; smaller values release the target sooner.
    """

    requires_source_condition = True

    def __init__(
        self,
        num_steps: int = 40,
        guidance: float = 2.4,
        guidance_interval_min: float = 0.10,
        guidance_interval_max: float = 0.90,
        qk_injection_strength: float = 0.90,
        qk_injection_until: float = 0.72,
        qk_fade_start: float = 0.45,
        qk_block_start: int = 0,
        qk_block_end: int = 20,
        solver: str = "heun",
        noise_scale: float = 1.0,
        t_eps: float = 0.05,
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.guidance = float(guidance)
        self.guidance_interval_min = float(guidance_interval_min)
        self.guidance_interval_max = float(guidance_interval_max)
        self.qk_injection_strength = float(qk_injection_strength)
        self.qk_injection_until = float(qk_injection_until)
        self.qk_fade_start = float(qk_fade_start)
        self.qk_block_start = int(qk_block_start)
        self.qk_block_end = int(qk_block_end)
        self.solver = str(solver).lower()
        self.noise_scale = float(noise_scale)
        self.t_eps = float(t_eps)
        # Existing Lightning wrapper looks for this attribute.
        self.t_start = self.qk_injection_until
        self.last_diagnostics = {}

        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        if self.solver not in ("euler", "heun"):
            raise ValueError("solver must be 'euler' or 'heun'")
        if not 0.0 <= self.guidance_interval_min < self.guidance_interval_max <= 1.0:
            raise ValueError("invalid guidance interval")
        if not 0.0 <= self.qk_injection_strength <= 1.5:
            raise ValueError("qk_injection_strength must be in [0, 1.5]")
        if not 0.0 <= self.qk_fade_start <= self.qk_injection_until <= 1.0:
            raise ValueError("need 0 <= qk_fade_start <= qk_injection_until <= 1")
        if self.qk_block_start < 0 or self.qk_block_end <= self.qk_block_start:
            raise ValueError("invalid Q/K block range")

    @staticmethod
    def _as_source_condition(value, reference, uncondition):
        batch = reference.shape[0]
        if value is None:
            return uncondition
        if torch.is_tensor(value):
            value = value.to(device=reference.device, dtype=torch.long).view(-1)
            if value.numel() == 1 and batch > 1:
                value = value.expand(batch)
            if value.numel() != batch:
                raise ValueError(
                    f"source_condition has {value.numel()} items for batch {batch}"
                )
            return torch.where(value >= 0, value, uncondition)
        value = int(value)
        if value < 0:
            return uncondition
        return torch.full((batch,), value, device=reference.device, dtype=torch.long)

    @staticmethod
    def _batch_scalar(value, reference, default: float = 1.0):
        batch = reference.shape[0]
        if value is None:
            return torch.full(
                (batch,), float(default), device=reference.device, dtype=torch.float32
            )
        if torch.is_tensor(value):
            out = value.to(device=reference.device, dtype=torch.float32).view(-1)
            if out.numel() == 1 and batch > 1:
                out = out.expand(batch)
            if out.numel() != batch:
                raise ValueError(f"expected {batch} values, got {out.numel()}")
            return out
        return torch.full(
            (batch,), float(value), device=reference.device, dtype=torch.float32
        )

    @staticmethod
    def _patch_embed(net, x):
        patches = F.unfold(
            x,
            kernel_size=net.patch_size,
            stride=net.patch_size,
        ).transpose(1, 2)
        return net.s_embedder(patches)

    @staticmethod
    def _activate(net, c):
        return c if net.single_silu_cond else F.silu(c)

    def _condition_triplet(
        self,
        net,
        x_source,
        x_target,
        t,
        source_condition,
        target_condition,
        uncondition,
    ):
        batch = x_target.shape[0]
        t_emb = net.t_embedder(t)
        y_source = net.y_embedder(source_condition).view(batch, 1, net.hidden_size)
        y_target = net.y_embedder(target_condition).view(batch, 1, net.hidden_size)
        y_uncond = net.y_embedder(uncondition).view(batch, 1, net.hidden_size)

        # Source and conditional target retain PixelDiT2's frozen DINO grounding.
        source_grounding = net.dino_conditioner(x_source, t)
        target_grounding = net.dino_conditioner(x_target, t)

        source_c = self._activate(
            net, source_grounding + t_emb.unsqueeze(1) + y_source
        )
        target_cond_c = self._activate(
            net, target_grounding + t_emb.unsqueeze(1) + y_target
        )
        # Match the released PixelDiT2 CFG path: null class + no grounding.
        target_uncond_c = self._activate(
            net, t_emb.unsqueeze(1) + y_uncond
        )
        return (
            source_c,
            torch.cat([target_cond_c, target_uncond_c], dim=0),
            y_source,
            torch.cat([y_target, y_uncond], dim=0),
        )

    @staticmethod
    def _extended_condition(c, context_len):
        if context_len <= 0:
            return c
        context_c = c.mean(dim=1, keepdim=True).expand(-1, context_len, -1)
        return torch.cat([context_c, c], dim=1)

    @staticmethod
    def _attention_qk_injected(attn, target_x, source_x, pos, alpha, context_len=0):
        """Use source Q/K geometry and target V content.

        ``target_x`` has batch 2B (target conditional + target unconditional), while
        ``source_x`` has batch B. Source Q/K are duplicated to both target branches.
        Context-token Q/K are always kept from the target so source class context is
        never copied into the target class context.
        """
        bt, n, c = target_x.shape
        bs = source_x.shape[0]
        if bt != 2 * bs:
            raise ValueError(f"expected target batch 2*source batch, got {bt} vs {bs}")

        def qkv_for(x):
            b = x.shape[0]
            qkv = (
                attn.qkv(x)
                .reshape(b, n, 3, attn.num_heads, c // attn.num_heads)
                .permute(2, 0, 1, 3, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            q, k = apply_rotary_emb(q, k, freqs_cis=pos)
            return q, k, v

        q_t, k_t, v_t = qkv_for(target_x)
        q_s, k_s, _ = qkv_for(source_x)
        q_s = torch.cat([q_s, q_s], dim=0)
        k_s = torch.cat([k_s, k_s], dim=0)

        a = alpha.to(device=target_x.device, dtype=target_x.dtype).view(-1, 1, 1, 1)
        a = torch.cat([a, a], dim=0)
        token_alpha = a.expand(-1, n, 1, 1).clone()
        if context_len > 0:
            token_alpha[:, :context_len] = 0

        q = q_t + token_alpha * (q_s - q_t)
        k = k_t + token_alpha * (k_s - k_t)

        q = q.view(bt, n, attn.num_heads, c // attn.num_heads).transpose(1, 2)
        k = (
            k.view(bt, n, attn.num_heads, c // attn.num_heads)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            v_t.view(bt, n, attn.num_heads, c // attn.num_heads)
            .transpose(1, 2)
            .contiguous()
        )
        out = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(bt, n, c)
        out = attn.proj(out)
        return attn.proj_drop(out)

    def _controlled_block(
        self,
        block,
        target_x,
        target_c,
        source_x,
        source_c,
        pos,
        alpha,
        context_len,
    ):
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = (
            block.adaLN_modulation(target_c).chunk(6, dim=-1)
        )
        s_shift_msa, s_scale_msa, _, _, _, _ = (
            block.adaLN_modulation(source_c).chunk(6, dim=-1)
        )

        target_attn_in = apply_adaln(
            block.norm1(target_x), t_shift_msa, t_scale_msa
        )
        source_attn_in = apply_adaln(
            block.norm1(source_x), s_shift_msa, s_scale_msa
        )
        attn_out = self._attention_qk_injected(
            block.attn,
            target_attn_in,
            source_attn_in,
            pos,
            alpha,
            context_len=context_len,
        )
        target_x = target_x + t_gate_msa * attn_out
        target_x = target_x + t_gate_mlp * block.mlp(
            apply_adaln(block.norm2(target_x), t_shift_mlp, t_scale_mlp)
        )
        return target_x

    @staticmethod
    def _fold_output(net, s, c, height, width):
        batch = s.shape[0]
        hs, ws = height // net.patch_size, width // net.patch_size
        num_patches = hs * ws
        out = net.final_layer(s, c)
        out = out.view(
            batch,
            num_patches,
            net.patch_size ** 2,
            net.out_channels,
        )
        out = out.permute(0, 3, 2, 1).reshape(
            batch,
            net.out_channels * net.patch_size ** 2,
            num_patches,
        )
        return F.fold(
            out,
            (height, width),
            kernel_size=net.patch_size,
            stride=net.patch_size,
        )

    def _pnp_forward(
        self,
        net,
        x_source,
        x_target,
        t,
        source_condition,
        target_condition,
        uncondition,
        alpha,
    ):
        """Return target conditional/unconditional x0 predictions."""
        batch, _, height, width = x_target.shape
        hs, ws = height // net.patch_size, width // net.patch_size

        source_c, target_c, source_y, target_y = self._condition_triplet(
            net,
            x_source,
            x_target,
            t,
            source_condition,
            target_condition,
            uncondition,
        )

        source_s = self._patch_embed(net, x_source)
        target_base = self._patch_embed(net, x_target)
        target_s = torch.cat([target_base, target_base], dim=0)

        use_context = net.in_context_len > 0
        pos = net.fetch_pos(hs, ws, x_target.device)
        pos_ctx = (
            net.fetch_pos(hs, ws, x_target.device, net.in_context_len)
            if use_context
            else None
        )
        context_inserted = False

        for i, block in enumerate(net.patch_blocks):
            if use_context and not context_inserted and i == net.in_context_start:
                src_ctx = source_y.expand(-1, net.in_context_len, -1)
                tgt_ctx = target_y.expand(-1, net.in_context_len, -1)
                posemb = net.in_context_posemb.to(source_s.dtype)
                source_s = torch.cat([src_ctx + posemb, source_s], dim=1)
                target_s = torch.cat([tgt_ctx + posemb, target_s], dim=1)
                context_inserted = True

            block_pos = pos_ctx if context_inserted else pos
            context_len = net.in_context_len if context_inserted else 0
            source_block_c = self._extended_condition(source_c, context_len)
            target_block_c = self._extended_condition(target_c, context_len)

            source_pre = source_s
            # Source follows its own normal PixelDiT2 branch.
            source_s = block(source_s, source_block_c, block_pos)

            if self.qk_block_start <= i < min(self.qk_block_end, net.depth):
                target_s = self._controlled_block(
                    block,
                    target_s,
                    target_block_c,
                    source_pre,
                    source_block_c,
                    block_pos,
                    alpha,
                    context_len,
                )
            else:
                target_s = block(target_s, target_block_c, block_pos)

        if context_inserted:
            target_s = target_s[:, net.in_context_len :, :]

        # Final layer remains entirely target-conditioned.
        pred = self._fold_output(net, target_s, target_c, height, width)
        return pred.chunk(2, dim=0)

    def _alpha(self, t, structure_until, structure_strength):
        # Full injection through qk_fade_start, then linearly release it so late
        # target details are free to form.
        until = float(structure_until)
        fade = min(self.qk_fade_start, until)
        if until <= 0:
            return torch.zeros_like(structure_strength)
        t_scalar = float(t.flatten()[0])
        if t_scalar >= until:
            sched = 0.0
        elif t_scalar <= fade or until <= fade + 1e-8:
            sched = 1.0
        else:
            sched = (until - t_scalar) / (until - fade)
        return (
            structure_strength
            * self.qk_injection_strength
            * float(max(0.0, min(1.0, sched)))
        ).clamp(0.0, 1.5)

    def _velocity(
        self,
        net,
        x,
        t,
        base_noise,
        source_image,
        source_condition,
        target_condition,
        uncondition,
        structure_until,
        structure_strength,
        edit_strength,
    ):
        batch = x.shape[0]
        t_img = t.view(batch, 1, 1, 1).to(source_image.dtype)
        x_source = t_img * source_image + (1.0 - t_img) * base_noise
        alpha = self._alpha(t, structure_until, structure_strength)

        pred_cond, pred_uncond = self._pnp_forward(
            net,
            x_source,
            x,
            t,
            source_condition,
            target_condition,
            uncondition,
            alpha,
        )
        scale = (1.0 - t).clamp_min(self.t_eps).view(batch, 1, 1, 1)
        v_cond = (pred_cond - x) / scale
        v_uncond = (pred_uncond - x) / scale

        inside = (
            (t > self.guidance_interval_min)
            & (t < self.guidance_interval_max)
        ).view(batch, 1, 1, 1)
        # edit_strength=1 -> configured CFG; 0 -> ordinary conditional scale 1.
        cfg = 1.0 + edit_strength.view(batch, 1, 1, 1) * (self.guidance - 1.0)
        cfg = torch.where(inside, cfg, torch.ones_like(cfg))
        return v_uncond + cfg * (v_cond - v_uncond), alpha

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
        edit_strength=1.0,
        t_start: Optional[float] = None,
        return_x_trajs: bool = False,
        return_v_trajs: bool = False,
    ):
        if return_v_trajs:
            raise NotImplementedError("PnP editor does not expose velocity trajectories")
        if source_image.shape != noise.shape:
            raise ValueError(
                f"source/noise shape mismatch: {source_image.shape} vs {noise.shape}"
            )
        batch = source_image.shape[0]
        target_condition = condition.to(source_image.device, dtype=torch.long).view(batch)
        uncondition = uncondition.to(source_image.device, dtype=torch.long).view(batch)
        source_condition = self._as_source_condition(
            source_condition, source_image, uncondition
        )
        structure_strength = self._batch_scalar(source_strength, source_image, 1.0)
        edit_strength = self._batch_scalar(edit_strength, source_image, 1.0)
        structure_until = self.qk_injection_until if t_start is None else float(t_start)
        if not 0.0 <= structure_until <= 1.0:
            raise ValueError("t_start/structure cutoff must be in [0,1]")

        base_noise = noise * self.noise_scale
        x = base_noise.clone()
        trajectory = [x.clone()] if return_x_trajs else None
        steps = torch.linspace(
            0.0,
            1.0,
            self.num_steps + 1,
            device=x.device,
            dtype=torch.float32,
        )
        alpha_sum = 0.0
        velocity_rms_sum = 0.0

        for i in range(self.num_steps - 1):
            t0 = torch.full(
                (batch,), float(steps[i]), device=x.device, dtype=torch.float32
            )
            t1 = torch.full(
                (batch,), float(steps[i + 1]), device=x.device, dtype=torch.float32
            )
            dt = float(steps[i + 1] - steps[i])
            v0, alpha0 = self._velocity(
                net,
                x,
                t0,
                base_noise,
                source_image,
                source_condition,
                target_condition,
                uncondition,
                structure_until,
                structure_strength,
                edit_strength,
            )
            if self.solver == "heun":
                x_euler = (x.float() + dt * v0.float()).to(x.dtype)
                v1, _ = self._velocity(
                    net,
                    x_euler,
                    t1,
                    base_noise,
                    source_image,
                    source_condition,
                    target_condition,
                    uncondition,
                    structure_until,
                    structure_strength,
                    edit_strength,
                )
                x = (x.float() + dt * 0.5 * (v0.float() + v1.float())).to(x.dtype)
            else:
                x = (x.float() + dt * v0.float()).to(x.dtype)
            alpha_sum += float(alpha0.float().mean().cpu())
            velocity_rms_sum += float(v0.float().square().mean().sqrt().cpu())
            if return_x_trajs:
                trajectory.append(x.clone())

        # Match the released sampler: final interval uses Euler to avoid t=1
        # predictor evaluation and the (1-t) velocity singularity.
        t0 = torch.full(
            (batch,), float(steps[-2]), device=x.device, dtype=torch.float32
        )
        dt = float(steps[-1] - steps[-2])
        v0, alpha0 = self._velocity(
            net,
            x,
            t0,
            base_noise,
            source_image,
            source_condition,
            target_condition,
            uncondition,
            structure_until,
            structure_strength,
            edit_strength,
        )
        x = (x.float() + dt * v0.float()).to(x.dtype)
        if return_x_trajs:
            trajectory.append(x.clone())

        self.last_diagnostics = {
            "num_steps": self.num_steps,
            "solver": self.solver,
            "structure_until": float(structure_until),
            "mean_qk_alpha": float(
                (alpha_sum + float(alpha0.float().mean().cpu())) / max(self.num_steps, 1)
            ),
            "mean_velocity_rms": float(
                velocity_rms_sum / max(self.num_steps - 1, 1)
            ),
            "qk_blocks": [self.qk_block_start, min(self.qk_block_end, net.depth)],
            "guidance": self.guidance,
        }
        return (x, trajectory) if return_x_trajs else x
