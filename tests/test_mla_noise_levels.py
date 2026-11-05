import pytest
import torch

from diffusion import VPSDEScheduler


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
