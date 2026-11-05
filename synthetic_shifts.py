"""Fixed source-only neural shifts and overlap-safe development partitions."""

import torch


def validation_development_split(intervals, split_seed):
    intervals = torch.as_tensor(intervals, dtype=torch.long).cpu()
    if intervals.ndim != 2 or intervals.size(1) != 2:
        raise ValueError("Validation intervals must have shape [trial, 2].")
    order = sorted(range(len(intervals)), key=lambda i: intervals[i, 0].item())
    groups, current, end = [], [], -1
    for index in order:
        start, stop = intervals[index].tolist()
        if current and start >= end:
            groups.append(current)
            current = []
        current.append(index)
        end = max(end, stop)
    if current:
        groups.append(current)
    if len(groups) < 2:
        raise ValueError("Source validation needs two non-overlapping groups for development checks.")
    generator = torch.Generator().manual_seed(split_seed)
    sides, counts = [[], []], [0, 0]
    for group_index in torch.randperm(len(groups), generator=generator).tolist():
        side = 0 if counts[0] <= counts[1] else 1
        sides[side].extend(groups[group_index])
        counts[side] += len(groups[group_index])
    if min(counts) < 2:
        raise ValueError("Development checks need at least two trials in each disjoint group.")
    return tuple(torch.tensor(side, dtype=torch.long) for side in sides)


def make_shift(kind, n_neurons, shift_seed=42, mask_fraction=0.2):
    generator = torch.Generator().manual_seed(shift_seed)
    shift = {"kind": kind, "n_neurons": n_neurons, "seed": shift_seed}
    if kind == "permutation":
        shift["permutation"] = torch.randperm(n_neurons, generator=generator)
    elif kind == "gain":
        shift["gain"] = 0.5 + torch.rand(n_neurons, generator=generator)
    elif kind == "masked_removal":
        if n_neurons < 2 or not 0 < mask_fraction < 1:
            raise ValueError("Masked removal needs multiple neurons and a fraction in (0, 1).")
        count = min(n_neurons - 1, max(1, round(mask_fraction * n_neurons)))
        removed = torch.randperm(n_neurons, generator=generator)[:count]
        shift["mask"] = torch.ones(n_neurons)
        shift["mask"][removed] = 0
        shift["removed_neurons"] = removed
    elif kind == "channel_removal":
        if n_neurons < 2 or not 0 < mask_fraction < 1:
            raise ValueError("Channel removal needs multiple neurons and a fraction in (0, 1).")
        count = min(n_neurons - 1, max(1, round(mask_fraction * n_neurons)))
        removed = torch.randperm(n_neurons, generator=generator)[:count].sort().values
        keep = torch.ones(n_neurons, dtype=torch.bool)
        keep[removed] = False
        shift["removed_neurons"] = removed
        shift["kept_indices"] = torch.arange(n_neurons)[keep]
    elif kind != "identity":
        raise ValueError(f"Unknown synthetic shift: {kind}")
    return shift


def apply_shift(spikes, shift):
    if spikes.size(-1) != shift["n_neurons"]:
        raise ValueError("Synthetic shift and spike channels differ.")
    kind = shift["kind"]
    if kind == "identity":
        return spikes.clone()
    if kind == "permutation":
        return spikes[..., shift["permutation"].to(spikes.device)]
    if kind == "gain":
        return spikes * shift["gain"].to(spikes)
    if kind == "masked_removal":
        return spikes * shift["mask"].to(spikes)
    if kind == "channel_removal":
        return spikes.index_select(-1, shift["kept_indices"].to(spikes.device))
    raise ValueError(f"Unknown synthetic shift: {kind}")


def oracle_readin_weight(source_weight, shift):
    """Known inverse for calibration only; never used to initialize adaptation."""
    if shift["kind"] == "identity":
        return source_weight.clone()
    if shift["kind"] == "permutation":
        return source_weight[:, shift["permutation"].to(source_weight.device)].clone()
    if shift["kind"] == "gain":
        return source_weight / shift["gain"].to(source_weight).unsqueeze(0)
    return None  # Lost channels have no exact inverse.
