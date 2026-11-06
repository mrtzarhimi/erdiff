import pytest
import torch

from diffusion import VPSDEScheduler, mla_loss


def test_fixed_and_uniform_times_preserve_following_noise_draws():
    scheduler = VPSDEScheduler()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        scheduler.sample_timesteps(8, "cpu")
        uniform_next_noise = torch.randn(8)
        torch.manual_seed(42)
        times = scheduler.sample_timesteps(8, "cpu", fixed_time=0.2)
        fixed_next_noise = torch.randn(8)
    assert torch.equal(times, torch.full((8,), 0.2))
    assert torch.equal(uniform_next_noise, fixed_next_noise)


def test_noise_ranges_and_invalid_controls():
    scheduler = VPSDEScheduler()
    with torch.random.fork_rng(devices=[]):
        times = scheduler.sample_timesteps(100, "cpu", time_min=0.1, time_max=0.4)
    assert bool(((times >= 0.1) & (times < 0.4)).all())
    with pytest.raises(ValueError):
        scheduler.sample_timesteps(1, "cpu", fixed_time=0.2, time_min=0.1)
    with pytest.raises(ValueError):
        scheduler.sample_timesteps(1, "cpu", fixed_time=-1)


def test_mla_uses_explicit_fixed_bank_noise_and_times_without_global_draws():
    class ScaledDenoiser(torch.nn.Module):
        def forward(self, z_t, times):
            return 0.25 * z_t

    scheduler = VPSDEScheduler()
    z0 = torch.zeros(2, 3, 1, requires_grad=True)
    explicit_noise = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    explicit_times = torch.tensor([0.2, 0.8])
    rng_state = torch.random.get_rng_state().clone()
    noisy, _ = scheduler.add_noise(z0.detach(), explicit_times, explicit_noise)
    expected = (explicit_noise - 0.25 * noisy).square().flatten(start_dim=1).sum(dim=1).mean()

    value = mla_loss(
        ScaledDenoiser(), scheduler, z0, noise=explicit_noise, timesteps=explicit_times,
        alignment_objective="paper_squared",
    )
    value.backward()

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.allclose(value, expected)
    assert z0.grad is not None
    assert torch.count_nonzero(z0.grad) > 0
