"""Balanced optimization batches, separate from deterministic statistics loaders."""

import torch
from torch.utils.data import DataLoader, Sampler


class BalancedBatchSampler(Sampler):
    """Shuffle each epoch and cover every item once in nearly equal batches."""

    def __init__(self, num_items: int, max_batch_size: int, seed: int):
        if num_items < 1 or max_batch_size < 1:
            raise ValueError("The dataset and maximum batch size must be positive.")
        self.num_items = num_items
        self.num_batches = (num_items + max_batch_size - 1) // max_batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        indices = torch.randperm(self.num_items, generator=self.generator)
        for batch in torch.tensor_split(indices, self.num_batches):
            yield batch.tolist()


def make_alignment_loader(dataset, max_batch_size: int, seed: int):
    return DataLoader(
        dataset,
        batch_sampler=BalancedBatchSampler(len(dataset), max_batch_size, seed),
        # Keep iterator seed draws separate from posterior and diffusion noise.
        generator=torch.Generator().manual_seed(seed),
    )
