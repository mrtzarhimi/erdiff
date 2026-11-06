import ot
import pytest
import torch

from diffusion import sinkhorn_loss


def test_transport_matches_author_sinkhorn2_value_and_autograd():
    source = torch.tensor([[[0.0, 0.3]], [[1.1, -0.2]]], dtype=torch.float64)
    target = torch.tensor([[[0.2, 0.1]], [[0.9, 0.7]], [[1.7, -0.3]]],
                          dtype=torch.float64, requires_grad=True)
    actual = sinkhorn_loss(source, target, epsilon=0.07, num_iters=1000)
    actual_gradient = torch.autograd.grad(actual, target)[0]

    reference_target = target.detach().clone().requires_grad_(True)
    cost = ot.dist(source.flatten(1), reference_target.flatten(1)) + 1e-4
    expected = ot.sinkhorn2(source.new_full((2,), 0.5), source.new_full((3,), 1 / 3),
                            cost / cost.max(), reg=0.07, numItermax=1000)
    expected_gradient = torch.autograd.grad(expected, reference_target)[0]

    # The production log-domain solve targets the same value/gradient as
    # conventional Sinkhorn on this well-conditioned float64 fixture.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-8)
    assert actual_gradient.norm() > 0


def test_transport_gradient_matches_finite_difference_through_solver_and_maximum():
    source = torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float64)
    target = torch.tensor([[[0.2]], [[1.6]]], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([[[0.3]], [[-0.4]]], dtype=torch.float64)
    loss = sinkhorn_loss(source, target, epsilon=0.1, num_iters=1000)
    gradient = torch.autograd.grad(loss, target)[0]
    step = 1e-5
    plus = sinkhorn_loss(source, target.detach() + step * direction, epsilon=0.1)
    minus = sinkhorn_loss(source, target.detach() - step * direction, epsilon=0.1)
    torch.testing.assert_close((gradient * direction).sum(), (plus - minus) / (2 * step),
                               rtol=2e-3, atol=1e-6)


def test_symmetric_transport_returns_cost_without_explicit_entropy():
    source = torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float64)
    epsilon = 0.2
    diagonal = 1e-4 / 1.0001
    off_diagonal_mass_ratio = torch.exp(torch.tensor(-(1 - diagonal) / epsilon, dtype=torch.float64))
    expected = (diagonal + off_diagonal_mass_ratio) / (1 + off_diagonal_mass_ratio)
    torch.testing.assert_close(sinkhorn_loss(source, source, epsilon=epsilon),
                               expected.to(torch.float64), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("epsilon,num_iters", [(0, 100), (-1, 100), (0.01, 0)])
def test_invalid_transport_settings_are_rejected(epsilon, num_iters):
    with pytest.raises(ValueError):
        sinkhorn_loss(torch.zeros(2, 1, 1), torch.ones(2, 1, 1), epsilon, num_iters)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_small_regularization_with_separated_float32_domains_has_finite_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # All normalized costs are near one, so ordinary exp(-C/0.01) produces
    # subnormal kernels and unsafe reciprocal updates in float32.
    source = torch.tensor([[[0.0]], [[1.0]]], device=device)
    target = torch.tensor([[[100.0]], [[101.0]]], device=device, requires_grad=True)
    value = sinkhorn_loss(source, target, epsilon=0.01)
    gradient = torch.autograd.grad(value, target)[0]

    assert value.dtype == target.dtype
    assert value.device == target.device
    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    # Independent float64 reference from POT's log-domain implementation.
    reference_target = target.detach().double().requires_grad_(True)
    cost = ot.dist(source.double().flatten(1), reference_target.flatten(1)) + 1e-4
    mass = cost.new_full((2,), 0.5)
    expected = ot.sinkhorn2(mass, mass, cost / cost.max(), reg=0.01,
                            method="sinkhorn_log", numItermax=1000, stopThr=1e-9)
    expected_gradient = torch.autograd.grad(expected, reference_target)[0]
    torch.testing.assert_close(value, expected.float())
    torch.testing.assert_close(gradient, expected_gradient.float())
