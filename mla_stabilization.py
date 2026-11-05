"""Optional source-referenced variance floor on trial posterior means."""

import math

import torch
import torch.nn.functional as F


def source_trial_variance(source_mu):
    if source_mu.ndim != 3 or source_mu.size(0) < 2:
        raise ValueError("Source variance requires at least two [trial, time, latent] trajectories.")
    return source_mu.detach().var(dim=0, unbiased=True)


def source_variance_floor_loss(target_mu, source_variance, floor_ratio=0.9, epsilon=1e-8):
    """Sum squared standard-deviation deficits across time and latent coordinates.

    Reference and target variance are across trials at each time bin. This
    constraint uses encoder means, not sampled latents or behavior labels.
    """
    if not math.isfinite(floor_ratio) or not 0 < floor_ratio <= 1:
        raise ValueError("Variance-floor ratio must be finite and in (0, 1].")
    if target_mu.ndim != 3 or target_mu.size(0) < 2:
        raise ValueError("Variance stabilization requires at least two trials per batch.")
    if source_variance.shape != target_mu.shape[1:]:
        raise ValueError("Source variance must match target [time, latent] dimensions.")
    reference = source_variance.detach().to(device=target_mu.device, dtype=target_mu.dtype)
    source_std = torch.sqrt(reference.clamp_min(0) + epsilon)
    target_std = torch.sqrt(target_mu.var(dim=0, unbiased=True) + epsilon)
    return F.relu(floor_ratio * source_std - target_std).square().sum()
