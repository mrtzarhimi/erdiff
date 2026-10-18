import os
import mat73
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


class AddGaussianNoise:
    """
    Proportional Gaussian jitter matching the paper's augmentation strategy.
    Adds noise scaled to 8% of the signal's standard deviation.
    """
    def __init__(self, factor: float = 0.08):
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std()
        noise = torch.randn_like(x) * self.factor * std
        return x + noise


class NHPDataset(Dataset):
    """
    PyTorch Dataset with optional on-the-fly trial augmentation.
    """
    def __init__(
        self,
        spikes: torch.Tensor,
        velocities: torch.Tensor,
        targets: torch.Tensor,
        is_train: bool = False,
        augment_multiplier: int = 5,
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

        # Apply augmentation only to training copies
        if self.is_train and (idx >= self.spikes.shape[0]):
            spk = self.noise_transform(spk)
            spk = torch.clamp(spk, min=0.0)

        return {
            "spikes": spk,
            "velocity": vel,
            "target": tgt,
        }


def load_nhp_session(
    file_path: str,
    seq_len: int = 39,
    smooth_sigma: float = 2.5,
    normalize_vel: bool = True,
):
    """
    Loads Chewie_CO .mat sessions aligned to movement onset.
    """
    data_dict = mat73.loadmat(file_path)
    xds = data_dict["xds"]

    # 1. Spikes
    raw_spikes = np.array(xds["spike_counts"], dtype=np.float32)
    if raw_spikes.shape[0] < raw_spikes.shape[1]:
        raw_spikes = raw_spikes.T

    if smooth_sigma > 0:
        raw_spikes = scipy.ndimage.gaussian_filter1d(raw_spikes, sigma=smooth_sigma, axis=0)

    # 2. Kinematics (Cursor Velocity)
    raw_velocity = np.array(xds["curs_v"], dtype=np.float32)
    if raw_velocity.shape[0] < raw_velocity.shape[1]:
        raw_velocity = raw_velocity.T

    # 3. Alignment Events
    time_frame = np.array(xds["time_frame"]).flatten()
    
    # Priority: movement onset time > go cue time > trial start
    align_times = xds.get("trial_movement_time", None)
    if align_times is None or np.all(np.isnan(align_times)):
        align_times = xds.get("trial_gocue_time", xds.get("trial_start_time")).flatten()
    else:
        align_times = np.array(align_times).flatten()

    trial_results = xds.get("trial_result", None)
    target_angles = xds.get("target_dir", None)
    if target_angles is not None:
        target_angles = np.array(target_angles).flatten()

    trials_spikes = []
    trials_vel = []
    trials_targets = []

    for i, event_time in enumerate(align_times):
        if np.isnan(event_time):
            continue

        if trial_results is not None:
            res = trial_results[i]
            if isinstance(res, str) and res != "R":
                continue

        # Start 5 bins (~100 ms) before movement onset to capture preparatory neural dynamics
        center_idx = np.searchsorted(time_frame, event_time)
        start_idx = max(0, center_idx - 5)
        end_idx = start_idx + seq_len

        if end_idx <= raw_spikes.shape[0]:
            trials_spikes.append(raw_spikes[start_idx:end_idx, :])
            trials_vel.append(raw_velocity[start_idx:end_idx, :])

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

    # Normalize velocity across trials and bins
    if normalize_vel:
        y_mean = y_data.mean(dim=(0, 1), keepdim=True)
        y_std = y_data.std(dim=(0, 1), keepdim=True) + 1e-8
        y_data = (y_data - y_mean) / y_std

    return x_data, y_data, targets


def get_nhp_dataloaders(
    file_path: str,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
    seq_len: int = 39,
    augment_multiplier: int = 5,
):
    x_data, y_data, targets = load_nhp_session(file_path=file_path, seq_len=seq_len)
    num_trials = x_data.shape[0]

    torch.manual_seed(seed)
    perm = torch.randperm(num_trials)
    val_size = int(num_trials * val_ratio)

    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_dataset = NHPDataset(
        x_data[train_idx],
        y_data[train_idx],
        targets[train_idx],
        is_train=True,
        augment_multiplier=augment_multiplier,
    )
    val_dataset = NHPDataset(
        x_data[val_idx],
        y_data[val_idx],
        targets[val_idx],
        is_train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, x_data.shape[-1]
