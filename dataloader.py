import os
import mat73
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader


class AddGaussianNoise:
    """Exact noise transform from yulewang97/ERDiff."""
    def __init__(self, factor: float = 0.08):
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std()
        noise = torch.randn_like(x) * self.factor * std
        return x + noise


class NHPDataset(Dataset):
    """
    Applies dynamic Gaussian noise on EVERY forward pass during training,
    and replicates the small trial set to match the '_aug_' dataset protocol.
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

        # Dynamically inject fresh Gaussian noise on every training access
        if self.is_train:
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
    smooth_sigma: float = 2.0,
    gocue_offset_bins: int = 5,  # +100ms after go_cue where movement reliably starts
):
    data_dict = mat73.loadmat(file_path)
    xds = data_dict["xds"]

    raw_spikes = np.array(xds["spike_counts"], dtype=np.float32)
    if raw_spikes.shape[0] < raw_spikes.shape[1]:
        raw_spikes = raw_spikes.T

    # Smooth spikes along temporal axis
    if smooth_sigma > 0:
        raw_spikes = scipy.ndimage.gaussian_filter1d(raw_spikes, sigma=smooth_sigma, axis=0)

    raw_velocity = np.array(xds["curs_v"], dtype=np.float32)
    if raw_velocity.shape[0] < raw_velocity.shape[1]:
        raw_velocity = raw_velocity.T

    time_frame = np.array(xds["time_frame"]).flatten()
    go_cues = np.array(xds.get("trial_gocue_time", xds.get("trial_start_time"))).flatten()
    if np.all(np.isnan(go_cues)):
        go_cues = np.array(xds["trial_start_time"]).flatten()

    trial_results = xds.get("trial_result", None)
    target_angles = xds.get("target_dir", None)
    if target_angles is not None:
        target_angles = np.array(target_angles).flatten()

    trials_spikes = []
    trials_vel = []
    trials_targets = []

    for i, go_time in enumerate(go_cues):
        if np.isnan(go_time):
            continue

        if trial_results is not None:
            res = trial_results[i]
            if isinstance(res, str) and res != "R":
                continue

        # Consistent alignment window: go_cue + fixed reaction offset (100ms)
        start_idx = np.searchsorted(time_frame, go_time) + gocue_offset_bins
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
    augment_multiplier: int = 5,
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

    # Z-score standardize both spikes and velocity using training statistics
    x_mean = x_train.mean(dim=(0, 1), keepdim=True)
    x_std = x_train.std(dim=(0, 1), keepdim=True) + 1e-6

    y_mean = y_train.mean(dim=(0, 1), keepdim=True)
    y_std = y_train.std(dim=(0, 1), keepdim=True) + 1e-6

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    train_dataset = NHPDataset(
        x_train, y_train, t_train, is_train=True, augment_multiplier=augment_multiplier
    )
    val_dataset = NHPDataset(x_val, y_val, t_val, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, x_data.shape[-1]
