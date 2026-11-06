"""Stop numerical failures before corrupting Phase 2 read-ins or checkpoints."""

import json
import os
from pathlib import Path

import torch


class NumericalFailure(FloatingPointError):
    def __init__(self, stage, tensors=None, reason=None):
        summaries = {}
        for name, tensor in (tensors or {}).items():
            value = tensor.detach()
            mask = torch.isfinite(value)
            finite = value[mask]
            summaries[name] = {
                "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                "nonfinite_count": int((~mask).sum().cpu()),
                "finite_min": None if finite.numel() == 0 else float(finite.min().cpu()),
                "finite_max": None if finite.numel() == 0 else float(finite.max().cpu()),
            }
        self.record = {"status": "failed", "stage": stage, "reason": reason,
                       "tensors": summaries}
        super().__init__(f"Phase 2 numerical failure at {stage}: {reason or summaries}")


def require_finite(stage, **tensors):
    if any(not bool(torch.isfinite(value.detach()).all()) for value in tensors.values()):
        raise NumericalFailure(stage, tensors)


class NumericalGuard:
    def __init__(self, failure_path=None, context=None):
        self.failure_path = None if failure_path is None else Path(failure_path)
        self.context = dict(context or {})

    def call(self, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except NumericalFailure as error:
            error.record["context"] = dict(self.context)
            if self.failure_path is not None:
                self.failure_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.failure_path.with_name(self.failure_path.name + ".tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(error.record, handle, indent=2, sort_keys=True, allow_nan=False)
                    handle.write("\n")
                os.replace(temporary, self.failure_path)
            raise

    def check(self, stage, **tensors):
        self.call(require_finite, stage, **tensors)

    def fail(self, stage, tensors=None, reason=None):
        def failure():
            raise NumericalFailure(stage, tensors, reason)
        self.call(failure)


def checked_backward_step(loss, optimizer, named_parameters, guard, max_norm=1.0):
    """Never step on a nonfinite gradient; retain the original clipping policy."""
    named_parameters = tuple(named_parameters)
    guard.check("total_loss_before_backward", loss=loss)
    loss.backward()
    gradients = {name: parameter.grad for name, parameter in named_parameters
                 if parameter.grad is not None}
    guard.check("read_in_gradients", **gradients)
    if not gradients:
        guard.fail("read_in_gradients", reason="No read-in gradients reached the optimizer.")
    parameters = tuple(parameter for _, parameter in named_parameters)
    try:
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm,
                                              error_if_nonfinite=True)
    except RuntimeError as error:
        guard.fail("read_in_gradient_norm", gradients, reason=str(error))
    guard.check("clipped_read_in_gradients", **gradients)
    optimizer.step()
    guard.check("read_in_parameters_after_step", **dict(named_parameters))
    return norm
