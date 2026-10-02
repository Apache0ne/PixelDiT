"""LoRA-specific diffusion trainer helpers for PixelDiT2.

This keeps adapter tuning truly parameter-efficient. The base GroundedFlowTrainer
always constructs the REPA projection MLP; when REPA is disabled that projector
is unused, so it must not remain trainable or Lightning will optimize millions
of unrelated parameters alongside LoRA.
"""

from .grounded_diffusion import GroundedFlowTrainer
from .utils import no_grad


class LoRAGroundedFlowTrainer(GroundedFlowTrainer):
    """GroundedFlowTrainer that freezes the unused REPA projector.

    The LoRA configs set ``repa_weight=0`` and ``repa_encoder=None``. In that
    configuration ``self.proj`` is not part of the loss, so freezing it prevents
    accidental non-LoRA optimizer parameters while preserving the original
    GroundedFlowTrainer behavior whenever REPA is enabled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.repa_weight <= 0.0 or self.repa_encoder is None:
            no_grad(self.proj)
