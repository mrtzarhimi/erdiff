import pytest
import torch

from alignment_noise_bank import AlignmentNoiseBank


def _registered_bank(seed=42, replicates=8, trial_ids=range(6)):
    bank = AlignmentNoiseBank(seed, replicates)
    bank.register_domain("target:session-A", trial_ids, seq_len=5, latent_dim=3)
    return bank


def test_fixed_draws_replay_under_batch_permutation():
    bank = _registered_bank()
    first = bank.get("target:session-A", torch.tensor([5, 2, 4]), 3, 5, 3, "cpu")
    second = bank.get("target:session-A", torch.tensor([4, 5, 2]), 3, 5, 3, "cpu")

    for first_index, trial_id in enumerate((5, 2, 4)):
        second_index = (4, 5, 2).index(trial_id)
        assert torch.equal(first.posterior_eta[first_index], second.posterior_eta[second_index])
        assert torch.equal(first.diffusion_eps[first_index], second.diffusion_eps[second_index])
        assert torch.equal(first.uniform_time[first_index], second.uniform_time[second_index])


def test_posterior_and_diffusion_noise_are_independent_streams():
    draws = _registered_bank().get("target:session-A", [0, 1, 2], 0, 5, 3, "cpu")

    assert not torch.equal(draws.posterior_eta, draws.diffusion_eps)
    assert not torch.equal(draws.posterior_eta[0], draws.posterior_eta[1])
    assert not torch.equal(draws.diffusion_eps[0], draws.diffusion_eps[1])


def test_probe_namespace_has_distinct_draws_from_fit_namespace():
    bank = AlignmentNoiseBank(seed=42, replicates=8)
    fit = bank.get("source:fit", [1, 4], 0, 5, 3, "cpu")
    probe = bank.get("source:probe", [1, 4], 0, 5, 3, "cpu")

    assert not torch.equal(fit.posterior_eta, probe.posterior_eta)
    assert not torch.equal(fit.diffusion_eps, probe.diffusion_eps)


def test_noise_levels_do_not_change_fixed_gaussian_arrays():
    bank = _registered_bank()
    low_noise = bank.get("target:session-A", [0, 3], 2, 5, 3, "cpu")
    high_noise = bank.get("target:session-A", [0, 3], 2, 5, 3, "cpu")
    # A noise-level ablation changes the time transform, not the Gaussian fixture.
    low_times = low_noise.uniform_time * (0.2 - 0.001) + 0.001
    high_times = low_noise.uniform_time * (1.0 - 0.001) + 0.001

    assert torch.equal(low_noise.posterior_eta, high_noise.posterior_eta)
    assert torch.equal(low_noise.diffusion_eps, high_noise.diffusion_eps)
    assert torch.equal(low_noise.uniform_time, high_noise.uniform_time)
    assert not torch.equal(low_times, high_times)


def test_noise_bank_does_not_advance_global_rng():
    bank = _registered_bank()
    state = torch.random.get_rng_state().clone()

    bank.get("target:session-A", [0, 1], 1, 5, 3, "cpu")

    assert torch.equal(torch.random.get_rng_state(), state)


def test_saved_bank_hash_reuse_and_identity_shape_rejection(tmp_path):
    path = tmp_path / "noise_bank.pt"
    bank = _registered_bank()
    bank.get("target:session-A", [0, 2, 5], 1, 5, 3, "cpu")
    saved_hash = bank.save(path)
    loaded = AlignmentNoiseBank.load(path, seed=42, replicates=8)

    assert loaded.content_hash() == saved_hash
    assert torch.equal(
        bank.get("target:session-A", [5, 0], 1, 5, 3, "cpu").diffusion_eps,
        loaded.get("target:session-A", [5, 0], 1, 5, 3, "cpu").diffusion_eps,
    )
    with pytest.raises(ValueError, match="identity or shape"):
        loaded.register_domain("target:session-A", range(5), seq_len=5, latent_dim=3)
    with pytest.raises(ValueError, match="incompatible with domain"):
        loaded.get("target:session-A", [0], 1, 4, 3, "cpu")
    with pytest.raises(ValueError, match="seed is incompatible"):
        AlignmentNoiseBank.load(path, seed=99, replicates=8)
