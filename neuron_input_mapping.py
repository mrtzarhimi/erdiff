"""Serializable channel mapping for optional target-to-source input sizing."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class NeuronInputMap:
    raw_width: int
    model_width: int
    kept_indices: torch.Tensor
    padding: int

    def __post_init__(self):
        indices = torch.as_tensor(self.kept_indices, dtype=torch.long).cpu()
        expected_count = min(int(self.raw_width), int(self.model_width))
        expected_padding = max(int(self.model_width) - int(self.raw_width), 0)
        if self.raw_width < 1 or self.model_width < 1:
            raise ValueError("Input widths must be positive.")
        if indices.ndim != 1 or indices.numel() != expected_count:
            raise ValueError("kept_indices length must equal min(raw_width, model_width).")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.raw_width):
            raise ValueError("kept_indices contains an out-of-range channel.")
        if indices.unique().numel() != indices.numel():
            raise ValueError("kept_indices must contain unique channels.")
        if int(self.padding) != expected_padding:
            raise ValueError("padding is inconsistent with the raw and model widths.")
        object.__setattr__(self, "kept_indices", indices)

    @classmethod
    def create(cls, raw_width: int, model_width: int, seed: int = 42):
        if raw_width < 1 or model_width < 1:
            raise ValueError("Input widths must be positive.")
        if raw_width >= model_width:
            if raw_width == model_width:
                indices = torch.arange(raw_width)
            else:
                generator = torch.Generator().manual_seed(int(seed))
                indices = torch.randperm(raw_width, generator=generator)[:model_width].sort().values
            return cls(raw_width, model_width, indices, 0)
        return cls(raw_width, model_width, torch.arange(raw_width), model_width - raw_width)

    def apply(self, spikes):
        if spikes.shape[-1] != self.raw_width:
            raise ValueError(f"Expected {self.raw_width} channels; received {spikes.shape[-1]}.")
        selected = spikes.index_select(-1, self.kept_indices.to(spikes.device))
        if self.padding:
            selected = torch.nn.functional.pad(selected, (0, self.padding))
        return selected

    def state_dict(self):
        return {"raw_width": self.raw_width, "model_width": self.model_width,
                "kept_indices": self.kept_indices.clone(), "padding": self.padding}

    @classmethod
    def from_state_dict(cls, state):
        return cls(int(state["raw_width"]), int(state["model_width"]),
                   torch.as_tensor(state["kept_indices"], dtype=torch.long), int(state["padding"]))

    def map_readin(self, readin: nn.Linear):
        if readin.in_features != self.raw_width:
            raise ValueError("Read-in width does not match the mapping's raw input width.")
        # Construct on CPU in a forked RNG scope; this projection must not perturb
        # training randomness or depend on a CUDA initialization stream.
        with torch.random.fork_rng(devices=[]):
            mapped = nn.Linear(
                self.model_width, readin.out_features, bias=readin.bias is not None,
                dtype=readin.weight.dtype,
            )
        mapped = mapped.to(device=readin.weight.device)
        with torch.no_grad():
            mapped.weight.zero_()
            mapped.weight[:, :self.kept_indices.numel()].copy_(readin.weight[:, self.kept_indices])
            if readin.bias is not None:
                mapped.bias.copy_(readin.bias)
        return mapped


def mapping_from_checkpoint(checkpoint, raw_width):
    """Restore and validate the persisted mapping contract for evaluation."""
    matched = bool(checkpoint.get("match_input_size", False))
    state = checkpoint.get("input_mapping")
    if matched and state is None:
        raise ValueError("Matched-width checkpoint is missing its saved input mapping.")
    if state is None:
        model_width = int(checkpoint.get("model_input_width", raw_width))
        if model_width != raw_width:
            raise ValueError("Checkpoint model width differs from raw target width without a mapping.")
        return None, model_width
    mapping = NeuronInputMap.from_state_dict(state)
    if mapping.raw_width != int(raw_width):
        raise ValueError("Saved input mapping raw width differs from the target data.")
    model_width = int(checkpoint.get("model_input_width", mapping.model_width))
    if mapping.model_width != model_width:
        raise ValueError("Saved input mapping model width differs from the checkpoint model width.")
    return mapping, model_width


def apply_mapping_to_loader(loader, mapping):
    """Apply the saved mapping to preprocessed spike inputs before model inference."""
    if mapping is not None:
        if loader is None:
            raise ValueError("Cannot map a missing evaluation loader.")
        loader.dataset.spikes = mapping.apply(loader.dataset.spikes)
    return loader
