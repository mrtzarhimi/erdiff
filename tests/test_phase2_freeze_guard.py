import json

import pytest
import torch
from torch import nn

from phase2_freeze_guard import FrozenDiffusionGuard


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.register_buffer("running_marker", torch.tensor(0.0))

    def forward(self, value):
        return self.projection(value)


def test_unchanged_denoiser_remains_frozen_and_in_eval_mode():
    model = TinyDenoiser()
    guard = FrozenDiffusionGuard(model)

    result = guard.verify()

    assert result["frozen"] is True
    assert result["state_entries"] == len(model.state_dict())
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_optimizer_cannot_include_frozen_denoiser_parameters():
    model = TinyDenoiser()
    guard = FrozenDiffusionGuard(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="optimizer contains"):
        guard.verify([optimizer])


def test_buffer_mutation_is_recorded_then_rejected(tmp_path):
    model = TinyDenoiser()
    guard = FrozenDiffusionGuard(model)
    with torch.no_grad():
        model.running_marker.add_(1)
    failure_path = tmp_path / "phase2_freeze_failure.json"

    with pytest.raises(RuntimeError, match="state changed"):
        guard.verify_or_record_failure(
            failure_path=failure_path,
            context={"epoch": 4, "target_session": "synthetic"},
        )

    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "failed"
    assert failure["invariant"] == "phase2_diffusion_frozen"
    assert failure["context"]["epoch"] == 4


def test_frozen_denoiser_keeps_input_gradient_without_parameter_gradients():
    model = TinyDenoiser()
    guard = FrozenDiffusionGuard(model)
    value = torch.randn(2, 3, requires_grad=True)

    model(value).square().sum().backward()

    assert value.grad is not None
    assert torch.count_nonzero(value.grad) > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    guard.verify()
