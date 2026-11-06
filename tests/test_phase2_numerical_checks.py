import json

import pytest
import torch

from numerical_checks import (NumericalFailure, NumericalGuard,
                              checked_backward_step, require_finite)
from diffusion import sinkhorn_loss


def test_nonfinite_tensor_identifies_stage_and_finite_range():
    with pytest.raises(NumericalFailure) as caught:
        require_finite("posterior_scale", scale=torch.tensor([1.0, 2.0, float("inf")]))
    record = caught.value.record
    assert record["stage"] == "posterior_scale"
    assert record["tensors"]["scale"]["nonfinite_count"] == 1
    assert record["tensors"]["scale"]["finite_min"] == 1.0
    assert record["tensors"]["scale"]["finite_max"] == 2.0


def test_finite_loss_with_infinite_backward_never_steps_optimizer(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    before = parameter.detach().clone()
    failure_path = tmp_path / "phase2_numerical_failure.json"
    guard = NumericalGuard(failure_path, {"epoch": 1, "batch": 1, "trial_ids": [7, 9]})
    # Finite forward value zero, infinite derivative at sqrt(0).
    loss = torch.sqrt(parameter - parameter.detach())

    with pytest.raises(NumericalFailure):
        checked_backward_step(loss, optimizer, (("weight", parameter),), guard)

    torch.testing.assert_close(parameter.detach(), before)
    assert optimizer.state == {}
    record = json.loads(failure_path.read_text())
    assert record["stage"] == "read_in_gradients"
    assert record["context"] == {"epoch": 1, "batch": 1, "trial_ids": [7, 9]}
    assert record["status"] == "failed"


def test_finite_step_preserves_configured_gradient_clipping(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    path = tmp_path / "failure.json"
    norm = checked_backward_step(parameter.square(), optimizer,
                                 (("weight", parameter),), NumericalGuard(path))
    torch.testing.assert_close(norm, torch.tensor(2.0))
    torch.testing.assert_close(parameter.detach(), torch.tensor(0.9))
    assert not path.exists()


def test_transport_rejects_nonfinite_input_before_solver(tmp_path):
    guard = NumericalGuard(tmp_path / "failure.json", {"epoch": 1, "batch": 2})
    with pytest.raises(NumericalFailure) as caught:
        guard.call(sinkhorn_loss, torch.zeros(2, 1, 1), torch.full((2, 1, 1), float("nan")))
    assert caught.value.record["stage"] == "transport_inputs"
    assert json.loads((tmp_path / "failure.json").read_text())["context"]["batch"] == 2


def test_training_rejects_bad_batch_without_updating_readin(tmp_path):
    from train_phase2 import SourcePosteriorBank, train_epoch_phase2
    from vae import SequentialVAE

    model = SequentialVAE(2, hidden_dim=4, latent_dim=2)
    model.freeze_for_alignment()
    optimizer = torch.optim.Adam(model.encoder.read_in.parameters(), lr=0.01)
    before = {key: value.clone() for key, value in model.encoder.read_in.state_dict().items()}
    bank = SourcePosteriorBank(torch.zeros(2, 3, 2), torch.zeros(2, 3, 2), torch.tensor([1, 2]))
    batch = {"spikes": torch.full((2, 3, 2), float("nan")), "trial_id": torch.tensor([8, 9])}
    path = tmp_path / "failure.json"
    with pytest.raises(NumericalFailure):
        train_epoch_phase2(model, torch.nn.Identity(), None, [batch], bank, optimizer,
                           torch.device("cpu"), numerical_failure_path=path,
                           numerical_context={"epoch": 1})
    for key, value in model.encoder.read_in.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert optimizer.state == {}
    record = json.loads(path.read_text())
    assert record["stage"] == "target_input"
    assert record["context"]["trial_ids"] == [8, 9]


@pytest.mark.parametrize("module_name", ["evaluate", "evaluate_cross_day"])
def test_evaluators_reject_corrupted_readin_before_reporting_metrics(module_name):
    from importlib import import_module
    from vae import SequentialVAE

    model = SequentialVAE(2, hidden_dim=4, latent_dim=2).eval()
    with torch.no_grad():
        model.encoder.read_in.weight.fill_(float("nan"))
    batch = {"spikes": torch.ones(2, 3, 2), "velocity": torch.zeros(2, 3, 2),
             "target": torch.zeros(2, dtype=torch.long)}
    with pytest.raises(NumericalFailure) as caught:
        import_module(module_name).run_inference(model, [batch], torch.device("cpu"))
    assert caught.value.record["stage"] == "evaluation_predictions"


def test_unclamped_posterior_overflow_is_localized_before_optimizer_step(tmp_path):
    from train_phase2 import SourcePosteriorBank, train_epoch_phase2
    from vae import SequentialVAE

    model = SequentialVAE(2, hidden_dim=4, latent_dim=2)
    with torch.no_grad():
        model.encoder.fc_logvar.weight.zero_()
        model.encoder.fc_logvar.bias.fill_(200.0)
    model.freeze_for_alignment()
    optimizer = torch.optim.Adam(model.encoder.read_in.parameters(), lr=0.01)
    bank = SourcePosteriorBank(torch.zeros(2, 3, 2), torch.zeros(2, 3, 2), torch.tensor([1, 2]))
    batch = {"spikes": torch.ones(2, 3, 2), "trial_id": torch.tensor([8, 9])}
    path = tmp_path / "failure.json"
    with pytest.raises(NumericalFailure):
        train_epoch_phase2(model, torch.nn.Identity(), None, [batch], bank, optimizer,
                           torch.device("cpu"), numerical_failure_path=path)
    assert optimizer.state == {}
    record = json.loads(path.read_text())
    assert record["stage"] == "target_posterior_scale"
    assert record["tensors"]["logvar"]["finite_max"] == 200.0
