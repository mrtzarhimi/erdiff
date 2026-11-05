import os
import mat73
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader


class AddGaussianNoise:
    """Proportional Gaussian noise transform from original ERDiff repo."""
    def __init__(self, factor: float = 0.08):
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std()
        noise = torch.randn_like(x) * self.factor * std
        return x + noise


class NHPDataset(Dataset):
    def __init__(
        self,
        spikes: torch.Tensor,
        velocities: torch.Tensor,
        targets: torch.Tensor,
        is_train: bool = False,
        augment_multiplier: int = 5,
        noise_factor: float = 0.08,
        include_behavior: bool = True,
    ):
        self.spikes = spikes
        self.velocities = velocities
        self.targets = targets
        self.is_train = is_train
        self.include_behavior = include_behavior
        self.multiplier = augment_multiplier if is_train else 1
        self.noise_transform = AddGaussianNoise(factor=noise_factor)

    def __len__(self):
        return self.spikes.shape[0] * self.multiplier

    def __getitem__(self, idx):
        base_idx = idx % self.spikes.shape[0]
        spk = self.spikes[base_idx].clone()

        if self.is_train:
            spk = self.noise_transform(spk)
            spk = torch.clamp(spk, min=0.0)

        sample = {"spikes": spk}
        if self.include_behavior:
            sample["velocity"] = self.velocities[base_idx].clone()
            sample["target"] = self.targets[base_idx]
        return sample


def load_nhp_session(
    file_path: str,
    seq_len: int = 39,
    smooth_sigma: float = 2.5,
    neural_lag_bins: int = 4,  # Motor cortex spikes lead hand velocity by ~80ms (4 bins)
):
    """Load fixed cue-aligned trials without using behavior values to place windows.

    Velocity peaks are deliberately not used for window selection: doing so would
    let target behavior determine which neural samples enter evaluation.
    """
    data_dict = mat73.loadmat(file_path)
    xds = data_dict["xds"]

    raw_spikes = np.array(xds["spike_counts"], dtype=np.float32)
    if raw_spikes.shape[0] < raw_spikes.shape[1]:
        raw_spikes = raw_spikes.T

    raw_velocity = np.array(xds["curs_v"], dtype=np.float32)
    if raw_velocity.shape[0] < raw_velocity.shape[1]:
        raw_velocity = raw_velocity.T
    time_frame = np.array(xds["time_frame"]).flatten()
    go_cues = np.array(xds.get("trial_gocue_time", xds.get("trial_start_time"))).flatten()
    if np.all(np.isnan(go_cues)):
        go_cues = np.array(xds["trial_start_time"]).flatten()

    trial_end = np.array(xds.get("trial_end_time", go_cues + 1.5)).flatten()
    trial_results = xds.get("trial_result", None)
    target_angles = xds.get("target_dir", None)
    if target_angles is not None:
        target_angles = np.array(target_angles).flatten()

    trials_spikes = []
    trials_vel = []
    trials_targets = []
    trial_intervals = []

    for i in range(len(go_cues)):
        t_go = go_cues[i]
        t_stop = trial_end[i]
        if np.isnan(t_go) or np.isnan(t_stop):
            continue

        if trial_results is not None:
            res = trial_results[i]
            if isinstance(res, str) and res != "R":
                continue

        idx_go = np.searchsorted(time_frame, t_go)
        idx_stop = np.searchsorted(time_frame, t_stop)
        vel_start, vel_end, spk_start, spk_end = fixed_window_bounds(
            go_index=idx_go,
            seq_len=seq_len,
            neural_lag_bins=neural_lag_bins,
        )

        if (
            idx_stop > idx_go
            and vel_end <= idx_stop
            and spk_start >= 0
            and spk_end <= raw_spikes.shape[0]
            and vel_end <= raw_velocity.shape[0]
        ):
            spike_window = raw_spikes[spk_start:spk_end, :]
            if smooth_sigma > 0:
                # Smooth within a trial only, so a neighboring trial assigned
                # to another split cannot contribute signal across its boundary.
                spike_window = scipy.ndimage.gaussian_filter1d(
                    spike_window, sigma=smooth_sigma, axis=0
                )
            trials_spikes.append(spike_window)
            trials_vel.append(raw_velocity[vel_start:vel_end, :])
            trial_intervals.append((spk_start, max(spk_end, vel_end)))

            if target_angles is not None:
                deg = np.rad2deg(target_angles[i]) % 360
                direction_class = int(np.round(deg / 45.0)) % 8
            else:
                direction_class = 0
            trials_targets.append(direction_class)

    if len(trials_spikes) == 0:
        raise ValueError(f"No valid trials extracted from {file_path}")

    x_data = torch.tensor(np.array(trials_spikes), dtype=torch.float32)
    y_data = torch.tensor(np.array(trials_vel), dtype=torch.float32)
    targets = torch.tensor(np.array(trials_targets), dtype=torch.long)

    return x_data, y_data, targets, torch.tensor(trial_intervals, dtype=torch.long)


def fixed_window_bounds(go_index: int, seq_len: int, neural_lag_bins: int = 0):
    """Return velocity and spike window bounds from the go cue alone."""
    vel_start = go_index
    vel_end = vel_start + seq_len
    spk_start = vel_start - neural_lag_bins
    spk_end = spk_start + seq_len
    return vel_start, vel_end, spk_start, spk_end


def denormalize_velocity(velocity: torch.Tensor, velocity_stats):
    """Convert source-standardized decoder values back to physical units."""
    mean = torch.as_tensor(velocity_stats["velocity_mean"], dtype=velocity.dtype, device=velocity.device)
    std = torch.as_tensor(velocity_stats["velocity_std"], dtype=velocity.dtype, device=velocity.device)
    return velocity * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def split_trial_indices(
    num_trials: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    window_intervals=None,
):
    """Create a reproducible, disjoint train/validation/test trial split."""
    if num_trials < 3:
        raise ValueError("At least three trials are required for train/validation/test splits.")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")

    val_size = max(1, int(num_trials * val_ratio))
    test_size = max(1, int(num_trials * test_ratio))
    if val_size + test_size >= num_trials:
        raise ValueError("Split ratios leave no training trials; use more trials or smaller ratios.")

    generator = torch.Generator().manual_seed(seed)
    if window_intervals is None:
        groups = [torch.tensor([i], dtype=torch.long) for i in range(num_trials)]
    else:
        intervals = torch.as_tensor(window_intervals, dtype=torch.long)
        if intervals.shape != (num_trials, 2):
            raise ValueError("window_intervals must have shape (num_trials, 2).")
        ordered = sorted(range(num_trials), key=lambda i: (intervals[i, 0].item(), intervals[i, 1].item()))
        groups = []
        current = []
        current_end = -1
        for index in ordered:
            start, end = intervals[index].tolist()
            if current and start >= current_end:
                groups.append(torch.tensor(current, dtype=torch.long))
                current = []
            current.append(index)
            current_end = max(current_end, end)
        if current:
            groups.append(torch.tensor(current, dtype=torch.long))
        if len(groups) < 3:
            raise ValueError("At least three non-overlapping trial-window groups are required for evaluation splits.")

    group_order = torch.randperm(len(groups), generator=generator).tolist()
    train_groups = [groups[group_order[0]]]
    val_groups = [groups[group_order[1]]]
    test_groups = [groups[group_order[2]]]
    counts = {"train": len(train_groups[0]), "val": len(val_groups[0]), "test": len(test_groups[0])}
    targets = {"train": num_trials - val_size - test_size, "val": val_size, "test": test_size}
    for group_index in group_order[3:]:
        split_name = max(targets, key=lambda name: targets[name] - counts[name])
        group = groups[group_index]
        if split_name == "train":
            train_groups.append(group)
        elif split_name == "val":
            val_groups.append(group)
        else:
            test_groups.append(group)
        counts[split_name] += len(group)

    train_idx = torch.cat(train_groups)
    val_idx = torch.cat(val_groups)
    test_idx = torch.cat(test_groups)
    return train_idx, val_idx, test_idx


def get_nhp_dataloaders(
    file_path: str,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    seq_len: int = 39,
    augment_multiplier: int = 5,
    velocity_stats=None,
    include_validation: bool = True,
    include_test: bool = True,
    include_behavior: bool = True,
):
    x_data, y_data, targets, window_intervals = load_nhp_session(file_path=file_path, seq_len=seq_len)
    num_trials = x_data.shape[0]

    train_idx, val_idx, test_idx = split_trial_indices(
        num_trials=num_trials,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        window_intervals=window_intervals,
    )

    x_train = x_data[train_idx]
    if include_behavior:
        y_train = y_data[train_idx]
        t_train = targets[train_idx]
    else:
        if velocity_stats is None:
            raise ValueError("velocity_stats are required when behavior labels are excluded.")
        y_train = torch.zeros((len(train_idx), seq_len, y_data.shape[-1]), dtype=y_data.dtype)
        t_train = torch.zeros(len(train_idx), dtype=targets.dtype)

    # Fit source-domain units on training trials only. Target sessions pass the
    # source statistics here so target labels never fit their own scale.
    if velocity_stats is None:
        y_mean = y_train.mean(dim=(0, 1), keepdim=True)
        y_std = y_train.std(dim=(0, 1), keepdim=True) + 1e-6
    else:
        y_mean = torch.as_tensor(
            velocity_stats["velocity_mean"], dtype=y_train.dtype, device="cpu"
        ).reshape(1, 1, -1)
        y_std = torch.as_tensor(
            velocity_stats["velocity_std"], dtype=y_train.dtype, device="cpu"
        ).reshape(1, 1, -1)

    y_train = (y_train - y_mean) / y_std

    train_dataset = NHPDataset(
        x_train, y_train, t_train, is_train=True, augment_multiplier=augment_multiplier,
        include_behavior=include_behavior,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = None
    if include_validation:
        x_val, y_val = x_data[val_idx], y_data[val_idx]
        t_val = targets[val_idx]
        y_val = (y_val - y_mean) / y_std
        val_dataset = NHPDataset(
            x_val, y_val, t_val, is_train=False, include_behavior=include_behavior
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    test_loader = None
    if include_test:
        x_test, y_test = x_data[test_idx], y_data[test_idx]
        t_test = targets[test_idx]
        y_test = (y_test - y_mean) / y_std
        test_dataset = NHPDataset(
            x_test, y_test, t_test, is_train=False, include_behavior=include_behavior
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    stats = {
        "velocity_mean": y_mean.reshape(-1).detach().cpu(),
        "velocity_std": y_std.reshape(-1).detach().cpu(),
    }
    return train_loader, val_loader, test_loader, x_data.shape[-1], stats
