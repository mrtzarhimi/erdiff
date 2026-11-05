import pytest
import torch

from mla_stabilization import source_trial_variance, source_variance_floor_loss


def test_matching_source_means_have_no_variance_deficit():
    source = torch.tensor([[[-1.0]], [[1.0]]], requires_grad=True)
    reference = source_trial_variance(source)
    assert not reference.requires_grad
    assert source_variance_floor_loss(source, reference).item() == 0.0


def test_deficit_gradient_pushes_distinct_means_apart():
    target = torch.tensor([[[-0.1]], [[0.1]]], requires_grad=True)
    reference = torch.tensor([[2.0]])
    loss = source_variance_floor_loss(target, reference, floor_ratio=0.9)
    gradient = torch.autograd.grad(loss, target)[0]
    assert loss.item() > 0
    assert gradient[0].item() > 0  # Descent moves the negative mean farther down.
    assert gradient[1].item() < 0


def test_time_varying_identical_trials_do_not_satisfy_trial_variance_floor():
    target = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1).expand(4, 3, 2)
    assert source_variance_floor_loss(target, torch.ones(3, 2)).item() > 0


def test_single_trial_batch_fails_clearly():
    with pytest.raises(ValueError, match="at least two trials"):
        source_variance_floor_loss(torch.zeros(1, 3, 2), torch.ones(3, 2))
