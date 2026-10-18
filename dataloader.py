import os
import mat73
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


class NHPDataset(Dataset):
    """
    PyTorch Dataset for Non-Human Primate (NHP) motor cortex recordings.
    Returns:
        spikes: (seq_len, n_neurons)
        velocity: (seq_len, 2)
        target_class: int (0 to 7)
    """
    def __init__(self, spikes: torch.Tensor, velocities: torch.Tensor, targets: torch.Tensor):
        self.spikes = spikes
        self.velocities = velocities
        self.targets = targets

    def __len__(self):
        return self.spikes.shape[0]

    def __getitem__(self, idx):
        return {
            "spikes": self.spikes[idx],          # (seq_len, n_neurons)
            "velocity": self.velocities[idx],      # (seq_len, 2)
            "target": self.targets[idx],          # scalar long (0..7)
        }


def load_nhp_session(
    file_path: str,
    seq_len: int = 39,
    bin_size_ms: int = 20,
    smooth_sigma: float = 2.5,
    normalize_vel: bool = True,
):
    """
    Loads and processes Chewie_CO .mat files according to the paper's data protocol:
      - 39 time bins of 20 ms (total 780 ms window per trial).
      - Filtered for successful trials only.
      - Aligned to go cue time.
      - Velocity binned and normalized.
      - 8-direction reaching angle discretized into labels 0..7.
    """
    data_dict = mat73.loadmat(file_path)
    xds = data_dict["xds"]

    # 1. Spikes
    raw_spikes = np.array(xds["spike_counts"], dtype=np.float32)
    if raw_spikes.shape[0] < raw_spikes.shape[1]:
        raw_spikes = raw_spikes.T
    
    # Gaussian smoothing along the temporal axis
    if smooth_sigma > 0:
        raw_spikes = scipy.ndimage.gaussian_filter1d(raw_spikes, sigma=smooth_sigma, axis=0)

    # 2. Behavior Kinematics (Cursor Velocity)
    raw_velocity = np.array(xds["curs_v"], dtype=np.float32)
    if raw_velocity.shape[0] < raw_velocity.shape[1]:
        raw_velocity = raw_velocity.T

    # 3. Time axes and trial markers
    time_frame = np.array(xds["time_frame"]).flatten()
    go_cues = np.array(xds.get("trial_gocue_time", xds.get("trial_start_time"))).flatten()
    
    # Filter for successful trials ('R' = Reward) if available
    trial_results = xds.get("trial_result", None)
    
    # Target direction angles (radians or degrees)
    target_angles = xds.get("target_dir", None)
    if target_angles is not None:
        target_angles = np.array(target_angles).flatten()

    trials_spikes = []
    trials_vel = []
    trials_targets = []

    for i, go_time in enumerate(go_cues):
        if np.isnan(go_time):
            continue

        # Keep only rewarded/successful trials
        if trial_results is not None:
            res = trial_results[i]
            if isinstance(res, str) and res != "R":
                continue

        start_idx = np.searchsorted(time_frame, go_time)
        end_idx = start_idx + seq_len

        if end_idx <= raw_spikes.shape[0]:
            trials_spikes.append(raw_spikes[start_idx:end_idx, :])
            trials_vel.append(raw_velocity[start_idx:end_idx, :])

            # Discretize 8 center-out reaching targets (0 to 7)
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

    # Velocity standardization
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
):
    """
    Creates train and validation DataLoaders for a given source or target session.
    Matches the 80%:20% cross-validation setup with batch size = 64.
    """
    x_data, y_data, targets = load_nhp_session(file_path=file_path, seq_len=seq_len)
    num_trials = x_data.shape[0]

    # Deterministic split
    torch.manual_seed(seed)
    perm = torch.randperm(num_trials)
    val_size = int(num_trials * val_ratio)
    
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_dataset = NHPDataset(x_data[train_idx], y_data[train_idx], targets[train_idx])
    val_dataset = NHPDataset(x_data[val_idx], y_data[val_idx], targets[val_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) > batch_size),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    n_neurons = x_data.shape[-1]
    return train_loader, val_loader, n_neurons


def get_cross_validation_loaders(
    file_path: str,
    n_splits: int = 5,
    batch_size: int = 64,
    seq_len: int = 39,
    seed: int = 42,
):
    """
    Generates 5-fold cross-validation loaders as conducted in Section 4.2 of the paper.
    """
    x_data, y_data, targets = load_nhp_session(file_path=file_path, seq_len=seq_len)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_loaders = []
    for train_idx, val_idx in kf.split(x_data):
        train_ds = NHPDataset(x_data[train_idx], y_data[train_idx], targets[train_idx])
        val_ds = NHPDataset(x_data[val_idx], y_data[val_idx], targets[val_idx])

        t_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        v_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        fold_loaders.append((t_loader, v_loader))

    n_neurons = x_data.shape[-1]
    return fold_loaders, n_neurons