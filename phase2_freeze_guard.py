"""Runtime assertions that Phase 2 never changes the Phase 1 denoiser."""

import json
import os
from datetime import datetime, timezone

import torch


class FrozenDiffusionGuard:
    def __init__(self, model):
        self.model = model
        self.initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        self.parameter_ids = {id(parameter) for parameter in model.parameters()}
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        model.eval()

    def verify(self, optimizers=()):
        if self.model.training:
            raise RuntimeError("Frozen diffusion must remain in eval mode.")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("A frozen diffusion parameter has requires_grad=True.")
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("A frozen diffusion parameter accumulated a gradient.")
        for optimizer in optimizers:
            if any(id(p) in self.parameter_ids for group in optimizer.param_groups for p in group["params"]):
                raise RuntimeError("A Phase 2 optimizer contains a frozen diffusion parameter.")
        current = self.model.state_dict()
        if current.keys() != self.initial_state.keys():
            raise RuntimeError("Frozen diffusion state structure changed during Phase 2.")
        for key, value in current.items():
            if not torch.equal(value.detach().cpu(), self.initial_state[key]):
                raise RuntimeError(f"Frozen diffusion state changed during Phase 2: {key}")
        return {"frozen": True, "state_entries": len(current)}

    def verify_or_record_failure(self, optimizers=(), failure_path=None, context=None):
        """Verify the invariant and persist failure context before propagating it."""
        try:
            return self.verify(optimizers)
        except RuntimeError as error:
            if failure_path is not None:
                record = {
                    "status": "failed",
                    "invariant": "phase2_diffusion_frozen",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "detected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "context": context or {},
                }
                parent = os.path.dirname(os.path.realpath(failure_path))
                os.makedirs(parent, exist_ok=True)
                temporary_path = failure_path + ".tmp"
                try:
                    with open(temporary_path, "w", encoding="utf-8") as handle:
                        json.dump(record, handle, indent=2, sort_keys=True)
                        handle.write("\n")
                    os.replace(temporary_path, failure_path)
                except OSError as save_error:
                    raise RuntimeError(
                        f"{error}; additionally failed to save freeze-failure metadata: {save_error}"
                    ) from error
            raise
