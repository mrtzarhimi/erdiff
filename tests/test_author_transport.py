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

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)
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
