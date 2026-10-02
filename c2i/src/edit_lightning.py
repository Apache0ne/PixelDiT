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
        t_start = [
            float(m.get("t_start", self.diffusion_sampler.t_start))
            for m in metadatas
        ]
        source_strength = torch.tensor(
            [float(m.get("source_strength", 1.0)) for m in metadatas],
            device=noise.device,
            dtype=noise.dtype,
        )
        if max(t_start) - min(t_start) > 1e-8:
            raise ValueError(
                "All samples in an edit prediction batch must use the same t_start"
            )

        model = self.denoiser if self.eval_original_model else self.ema_denoiser
        samples = self.diffusion_sampler(
            model,
            noise,
            condition,
            uncondition,
            source_image=source_image,
            source_strength=source_strength,
            t_start=t_start[0],
        )
        return fp2uint8(samples)
