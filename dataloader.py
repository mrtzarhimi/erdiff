import os
import mat73
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader


class AddGaussianNoise:
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
        augment_multiplier: int = 3,
        noise_factor: float = 0.08,
    ):
        self.spikes = spikes
        self.velocities = velocities
        self.targets = targets
        self.is_train = is_train
        self.multiplier = augment_multiplier if is_train else 1
        self.noise_transform = AddGaussianNoise(factor=noise_factor)

    def __len__(self):
        return self.spikes.shape[0] * self.multiplier

    def __getitem__(self, idx):
        base_idx = idx % self.spikes.shape[0]
        spk = self.spikes[base_idx].clone()
        vel = self.velocities[base_idx].clone()
        tgt = self.targets[base_idx]

        if self.is_train and (idx >= self.spikes.shape[0]):
            spk = self.noise_transform(spk)
            spk = torch.clamp(spk, min=0.0)

        return {
            "spikes": spk,
            "velocity": vel,
            "target": tgt,
        }


def find_movement_onset(time_frame, raw_speed, t_start, t_end, threshold_ratio=0.15, min_threshold=3.0):
    """
    Finds the exact continuous movement onset timestamp between t_start and t_end
    where hand speed crosses threshold_ratio * peak_speed.
    """
    idx_start = np.searchsorted(time_frame, t_start)
    idx_end = np.searchsorted(time_frame, t_end)
    if idx_end <= idx_start:
        return np.nan

    segment_speed = raw_speed[idx_start:idx_end]
    if len(segment_speed) == 0:
        return np.nan

    peak_speed = np.max(segment_speed)
    thresh = max(min_threshold, peak_speed * threshold_ratio)

    cross_indices = np.where(segment_speed >= thresh)[0]
    if len(cross_indices) == 0:
        return time_frame[idx_start]

    onset_idx = idx_start + cross_indices[0]
    return time_frame[onset_idx]


def load_nhp_session(
    file_path: str,
    seq_len: int = 39,
    smooth_sigma: float = 2.5,
):
    """
    Loads session aligned to CONTINUOUS MOVEMENT ONSET rather than variable go-cues.
    Captures: 6 bins (~120ms) before onset, and 33 bins (~660ms) through reach.
    """
    data_dict = mat73.loadmat(file_path)
    xds = data_dict["xds"]

    # 1. Spikes
    raw_spikes = np.array(xds["spike_counts"], dtype=np.float32)
    if raw_spikes.shape[0] < raw_spikes.shape[1]:
        raw_spikes = raw_spikes.T
    if smooth_sigma > 0:
        raw_spikes = scipy.ndimage.gaussian_filter1d(raw_spikes, sigma=smooth_sigma, axis=0)

    # 2. Kinematics
    raw_velocity = np.array(xds["curs_v"], dtype=np.float32)
    if raw_velocity.shape[0] < raw_velocity.shape[1]:
        raw_velocity = raw_velocity.T
    raw_speed = np.linalg.norm(raw_velocity, axis=1)

    time_frame = np.array(xds["time_frame"]).flatten()

    # Priority alignment markers
    trial_gocue = np.array(xds.get("trial_gocue_time", xds.get("trial_start_time"))).flatten()
    trial_end = np.array(xds.get("trial_end_time", trial_gocue + 1.5)).flatten()
    trial_results = xds.get("trial_result", None)
    target_angles = xds.get("target_dir", None)
    if target_angles is not None:
        target_angles = np.array(target_angles).flatten()

    trials_spikes = []
    trials_vel = []
    trials_targets = []

    pre_bins = 6  # ~120ms continuous preparatory phase

    for i in range(len(trial_gocue)):
        t_go = trial_gocue[i]
        t_stop = trial_end[i]
        if np.isnan(t_go) or np.isnan(t_stop):
            continue

        if trial_results is not None:
            res = trial_results[i]
            if isinstance(res, str) and res != "R":
                continue

        # Find continuous movement onset within [t_go, t_stop]
        t_onset = find_movement_onset(time_frame, raw_speed, t_go, t_stop)
        if np.isnan(t_onset):
            continue

        center_idx = np.searchsorted(time_frame, t_onset)
        start_idx = center_idx - pre_bins
        end_idx = start_idx + seq_len

        if start_idx >= 0 and end_idx <= raw_spikes.shape[0]:
            trials_spikes.append(raw_spikes[start_idx:end_idx, :])
            trials_vel.append(raw_velocity[start_idx:end_idx, :])

            if target_angles is not None:
                deg = np.rad2deg(target_angles[i]) % 360
                direction_class = int(np.round(deg / 45.0)) % 8
            else:
                direction_class = 0
            trials_targets.append(direction_class)

    if len(trials_spikes) == 0:
        raise ValueError(f"No valid movement-aligned trials extracted from {file_path}")

    x_data = torch.tensor(np.array(trials_spikes), dtype=torch.float32)
    y_data = torch.tensor(np.array(trials_vel), dtype=torch.float32)
    targets = torch.tensor(np.array(trials_targets), dtype=torch.long)

    return x_data, y_data, targets


def get_nhp_dataloaders(
    file_path: str,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
    seq_len: int = 39,
    augment_multiplier: int = 3,
):
    x_data, y_data, targets = load_nhp_session(file_path=file_path, seq_len=seq_len)
    num_trials = x_data.shape[0]

    torch.manual_seed(seed)
    perm = torch.randperm(num_trials)
    val_size = int(num_trials * val_ratio)

    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    x_train, y_train = x_data[train_idx], y_data[train_idx]
    x_val, y_val = x_data[val_idx], y_data[val_idx]
    t_train, t_val = targets[train_idx], targets[val_idx]

    # Compute normalization STRICTLY from training set
    y_mean = y_train.mean(dim=(0, 1), keepdim=True)
    y_std = y_train.std(dim=(0, 1), keepdim=True) + 1e-8

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    train_dataset = NHPDataset(x_train, y_train, t_train, is_train=True, augment_multiplier=augment_multiplier)
    val_dataset = NHPDataset(x_val, y_val, t_val, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, x_data.shape[-1]
