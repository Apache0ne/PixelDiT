"""Lightning wrapper for PixelDiT2 image-edit prediction."""
from __future__ import annotations

import torch

from .lightning import LightningModel, fp2uint8


class EditLightningModel(LightningModel):
    def predict_step(self, batch, batch_idx):
        noise, y, metadatas = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y)

        source_image = torch.stack(
            [m["source_image"] for m in metadatas], dim=0
        ).to(noise.device)

        default_start = getattr(
            self.diffusion_sampler,
            "qk_injection_until",
            getattr(
                self.diffusion_sampler,
                "edit_t_min",
                getattr(self.diffusion_sampler, "t_start", 0.72),
            ),
        )
        t_start = [
            float(m.get("t_start", default_start)) for m in metadatas
        ]
        if max(t_start) - min(t_start) > 1e-8:
            raise ValueError(
                "All samples in an edit prediction batch must use the same t_start"
            )

        structure_strength = torch.tensor(
            [
                float(m.get("structure_strength", m.get("source_strength", 1.0)))
                for m in metadatas
            ],
            device=noise.device,
            dtype=noise.dtype,
        )
        edit_strength = torch.tensor(
            [float(m.get("edit_strength", 1.0)) for m in metadatas],
            device=noise.device,
            dtype=noise.dtype,
        )
        source_condition = torch.tensor(
            [int(m.get("source_class", -1)) for m in metadatas],
            device=noise.device,
            dtype=torch.long,
        )

        model = self.denoiser if self.eval_original_model else self.ema_denoiser
        kwargs = dict(
            source_image=source_image,
            source_strength=structure_strength,
            t_start=t_start[0],
        )
        if getattr(self.diffusion_sampler, "requires_source_condition", False):
            kwargs["source_condition"] = source_condition
            kwargs["edit_strength"] = edit_strength

        samples = self.diffusion_sampler(
            model,
            noise,
            condition,
            uncondition,
            **kwargs,
        )
        return fp2uint8(samples)
