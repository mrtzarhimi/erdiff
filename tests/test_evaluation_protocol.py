import torch

from dataloader import denormalize_velocity, fixed_window_bounds, split_trial_indices


def test_trial_split_is_deterministic_and_disjoint():
    first = split_trial_indices(num_trials=100, val_ratio=0.15, test_ratio=0.15, seed=23)
    second = split_trial_indices(num_trials=100, val_ratio=0.15, test_ratio=0.15, seed=23)

    assert all(torch.equal(left, right) for left, right in zip(first, second))
    train, validation, test = first
    assert len(set(train.tolist()) & set(validation.tolist())) == 0
    assert len(set(train.tolist()) & set(test.tolist())) == 0
    assert len(set(validation.tolist()) & set(test.tolist())) == 0
    assert len(train) + len(validation) + len(test) == 100


def test_overlapping_trial_windows_stay_in_the_same_partition():
    intervals = torch.tensor([
        [0, 39], [35, 74], [100, 139], [145, 184], [220, 259], [250, 289]
    ])
    splits = split_trial_indices(
        num_trials=6,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=17,
        window_intervals=intervals,
    )
    partition = {index: split_id for split_id, split in enumerate(splits) for index in split.tolist()}

    assert partition[0] == partition[1]
    assert partition[4] == partition[5]


def test_windows_are_fixed_from_go_cue_and_never_depend_on_velocity():
    bounds = fixed_window_bounds(go_index=100, seq_len=39, neural_lag_bins=4)

    assert bounds == (100, 139, 96, 135)


def test_velocity_denormalization_restores_physical_values():
    normalized = torch.tensor([[[1.0, -2.0]]])
    stats = {"velocity_mean": torch.tensor([3.0, 5.0]), "velocity_std": torch.tensor([2.0, 4.0])}

    assert torch.equal(denormalize_velocity(normalized, stats), torch.tensor([[[5.0, -3.0]]]))
