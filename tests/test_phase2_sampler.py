import torch

from phase2_sampling import BalancedBatchSampler


def test_135_trials_are_balanced_without_dropping_or_repeating():
    batches = list(BalancedBatchSampler(135, 64, seed=42))

    assert [len(batch) for batch in batches] == [45, 45, 45]
    assert sorted(index for batch in batches for index in batch) == list(range(135))


def test_epochs_shuffle_reproducibly_without_changing_global_rng():
    global_state = torch.random.get_rng_state().clone()
    first = BalancedBatchSampler(135, 64, seed=42)
    replay = BalancedBatchSampler(135, 64, seed=42)
    epoch_one = list(first)
    epoch_two = list(first)

    assert epoch_one == list(replay)
    assert epoch_two == list(replay)
    assert epoch_one != epoch_two
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_small_or_uneven_datasets_have_no_tiny_tail():
    for count in (1, 7, 63, 64, 65, 128, 129, 191):
        batches = list(BalancedBatchSampler(count, 64, seed=42))
        sizes = [len(batch) for batch in batches]
        assert max(sizes) <= 64
        assert max(sizes) - min(sizes) <= 1
        assert sorted(index for batch in batches for index in batch) == list(range(count))
