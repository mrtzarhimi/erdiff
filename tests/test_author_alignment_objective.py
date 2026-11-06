import torch

from diffusion import (VPSDEScheduler, fixed_noise_mla_loss, skilling_divergence,
                       total_alignment_loss, validate_alignment_source_loss)


class ZeroDenoiser(torch.nn.Module):
    def forward(self, latent, time):
        return latent * 0


def test_author_residual_is_scaled_element_mean_smooth_l1():
    latent = torch.zeros(1, 2, 2, dtype=torch.float64, requires_grad=True)
    noise = torch.tensor([[[2.0, -0.5], [3.0, 0.0]]], dtype=torch.float64)
    rng = torch.random.get_rng_state().clone()
    total, residual, skilling = fixed_noise_mla_loss(
        ZeroDenoiser(), VPSDEScheduler(), latent, torch.tensor([0.2], dtype=torch.float64),
        noise, return_components=True,
    )
    # Huber(beta=1): (1.5 + .125 + 2.5 + 0)/4, scaled by 1.2.
    torch.testing.assert_close(residual, torch.tensor(1.2375, dtype=torch.float64))
    torch.testing.assert_close(total, residual + skilling)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_skilling_value_and_gradient_match_literal_author_vjp():
    scheduler = VPSDEScheduler()
    latent = torch.tensor([[[0.3, -0.2]], [[0.7, 0.4]]], dtype=torch.float64, requires_grad=True)
    time = torch.tensor([0.1, 0.6], dtype=torch.float64)
    noise = torch.tensor([[[1.0, -0.4]], [[-0.3, 0.2]]], dtype=torch.float64)
    noisy, _ = scheduler.add_noise(latent, time, noise)
    vjp = torch.autograd.grad(noisy, latent, torch.ones_like(noisy), retain_graph=True)[0]
    expected = (noisy * vjp * (1 / 50)).mean()
    expected_gradient = torch.autograd.grad(expected, latent, retain_graph=True)[0]
    actual = skilling_divergence(noisy, scheduler, time)
    actual_gradient = torch.autograd.grad(actual, latent)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_author_sum_and_posterior_mean_mla_are_independent_of_legacy_alpha():
    scheduler = VPSDEScheduler()
    source = torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float64)
    posterior_sample = torch.tensor([[[0.2]], [[1.5]]], dtype=torch.float64, requires_grad=True)
    mean = torch.tensor([[[0.1]], [[0.8]]], dtype=torch.float64, requires_grad=True)
    time = torch.tensor([0.2, 0.6], dtype=torch.float64)
    noise = torch.ones_like(mean)
    result = total_alignment_loss(ZeroDenoiser(), scheduler, posterior_sample, source,
                                   alpha=0.1, mla_target=mean, mla_noise=noise,
                                   mla_timesteps=time, return_mla_components=True)
    total, mla, ot, residual, skilling = result
    expected_mla = fixed_noise_mla_loss(ZeroDenoiser(), scheduler, mean, time, noise)
    torch.testing.assert_close(mla, expected_mla)
    torch.testing.assert_close(total, residual + skilling + ot)


def test_paper_squared_mode_retains_trajectory_norm_without_skilling():
    latent = torch.zeros(1, 2, 2, dtype=torch.float64)
    noise = torch.tensor([[[2.0, -0.5], [3.0, 0.0]]], dtype=torch.float64)
    total, residual, skilling = fixed_noise_mla_loss(
        ZeroDenoiser(), VPSDEScheduler(), latent, torch.tensor([0.2], dtype=torch.float64),
        noise, alignment_objective="paper_squared", return_components=True,
    )
    torch.testing.assert_close(total, torch.tensor(13.25, dtype=torch.float64))
    torch.testing.assert_close(residual, total)
    assert skilling == 0


def test_author_mode_rejects_mse_or_unknown_density_training_loss():
    import pytest
    for config in ({}, {"denoising_loss": "mse", "denoising_reduction": "element_mean"}):
        with pytest.raises(ValueError, match="retrain Phase 1"):
            validate_alignment_source_loss(config, "author_2023")
    validate_alignment_source_loss({"denoising_loss": "l1", "denoising_reduction": "element_mean"}, "author_2023")
    validate_alignment_source_loss({}, "paper_squared")


def test_author_training_sum_reaches_readin_through_frozen_denoiser():
    from train_phase2 import SourcePosteriorBank, train_epoch_phase2
    from vae import SequentialVAE
    from phase2_freeze_guard import FrozenDiffusionGuard

    class LinearDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, latent, time):
            return self.scale * latent

    model = SequentialVAE(2, hidden_dim=4, latent_dim=2, dropout=0)
    model.freeze_for_alignment()
    denoiser = LinearDenoiser()
    guard = FrozenDiffusionGuard(denoiser)
    optimizer = torch.optim.SGD(model.encoder.read_in.parameters(), lr=0.001)
    before = model.encoder.read_in.weight.detach().clone()
    bank = SourcePosteriorBank(torch.zeros(2, 3, 2), torch.zeros(2, 3, 2), torch.tensor([1, 2]))
    batch = {"spikes": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 12,
             "trial_id": torch.tensor([8, 9])}
    metrics = train_epoch_phase2(model, denoiser, VPSDEScheduler(), [batch], bank,
                                 optimizer, torch.device("cpu"), alignment_objective="author_2023")
    assert abs(metrics["mla"] - metrics["denoising"] - metrics["skilling"]) < 1e-6
    assert abs(metrics["total"] - metrics["mla"] - metrics["ot"]) < 1e-6
    assert not torch.equal(before, model.encoder.read_in.weight.detach())
    assert guard.verify((optimizer,))["frozen"]
