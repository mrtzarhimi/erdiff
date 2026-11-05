"""Runtime assertions that Phase 2 never changes the Phase 1 denoiser."""

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
