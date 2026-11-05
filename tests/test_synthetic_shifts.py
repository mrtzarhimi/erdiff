import torch

from synthetic_shifts import apply_shift, make_shift, oracle_readin_weight, validation_development_split


def test_permutation_and_gain_have_exact_readin_inverses():
    spikes = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5) / 10
    weight = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 7
    bias = torch.tensor([1.0, -2.0, 0.5])
    for kind in ("identity", "permutation", "gain"):
        shift = make_shift(kind, 5, shift_seed=42)
        shifted = apply_shift(spikes, shift)
        inverse = oracle_readin_weight(weight, shift)
        assert torch.allclose(shifted @ inverse.t() + bias, spikes @ weight.t() + bias, atol=1e-5)


def test_shift_fixtures_replay_without_changing_global_rng():
    rng = torch.random.get_rng_state().clone()
    left = make_shift("gain", 96, shift_seed=42)
    right = make_shift("gain", 96, shift_seed=42)
    removal = make_shift("masked_removal", 96, shift_seed=42)
    assert torch.equal(left["gain"], right["gain"])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert len(removal["removed_neurons"]) == 19
    assert oracle_readin_weight(torch.zeros(3, 96), removal) is None


def test_channel_removal_reduces_width_and_reuses_the_same_retained_channels():
    spikes = torch.randn(4, 7, 96)
    first = make_shift("channel_removal", 96, shift_seed=42, mask_fraction=0.2)
    second = make_shift("channel_removal", 96, shift_seed=42, mask_fraction=0.2)

    adapted = apply_shift(spikes, first)
    evaluation = apply_shift(spikes.clone(), second)

    assert adapted.shape[-1] == 77
    assert torch.equal(first["removed_neurons"], second["removed_neurons"])
    assert torch.equal(first["kept_indices"], second["kept_indices"])
    assert torch.equal(adapted, evaluation)
    assert oracle_readin_weight(torch.zeros(3, 96), first) is None


def test_development_split_preserves_overlap_groups_and_covers_validation():
    intervals = torch.tensor([[0, 10], [5, 15], [30, 40], [45, 55], [60, 70], [65, 75]])
    adaptation, evaluation = validation_development_split(intervals, 42)
    assert sorted(adaptation.tolist() + evaluation.tolist()) == list(range(6))
    assert not set(adaptation.tolist()) & set(evaluation.tolist())
    for left in adaptation.tolist():
        for right in evaluation.tolist():
            assert intervals[left, 1] <= intervals[right, 0] or intervals[right, 1] <= intervals[left, 0]
