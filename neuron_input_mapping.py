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
        mapped = nn.Linear(self.model_width, readin.out_features, bias=readin.bias is not None)
        with torch.no_grad():
            mapped.weight.zero_()
            mapped.weight[:, :self.kept_indices.numel()].copy_(readin.weight[:, self.kept_indices])
            if readin.bias is not None:
                mapped.bias.copy_(readin.bias)
        return mapped
