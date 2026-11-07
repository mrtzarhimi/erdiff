"""Phase 2 optimization batch strategies, separate from statistics loaders."""

import hashlib

import torch
from torch.utils.data import DataLoader, Sampler


def derive_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}|{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def source_sampling_generator(seed: int):
    return torch.Generator(device="cpu").manual_seed(derive_seed(seed, "source-bank-trial-selection"))


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
    return make_alignment_loader_for_strategy(dataset, max_batch_size, seed, "balanced_minibatch")


class FullDomainBatchSampler(Sampler):
    """Yield every adaptation trial together, matching the author's full-tensor update."""

    def __init__(self, num_items: int, seed: int):
        if num_items < 1:
            raise ValueError("The dataset must contain at least one item.")
        self.num_items = num_items
        # Keep a generator attribute so run checkpoints have a uniform RNG-state schema.
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return 1

    def __iter__(self):
        yield list(range(self.num_items))


def make_alignment_loader_for_strategy(dataset, max_batch_size: int, seed: int, strategy: str):
    """Create the current balanced loader or the paper's single full-domain batch."""
    batch_seed = derive_seed(seed, "alignment-batch-order")
    if strategy == "balanced_minibatch":
        sampler = BalancedBatchSampler(len(dataset), max_batch_size, batch_seed)
    elif strategy == "paper_full_domain":
        sampler = FullDomainBatchSampler(len(dataset), batch_seed)
    else:
        raise ValueError(f"Unknown Phase 2 batch strategy: {strategy!r}")
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        # DataLoader's iterator stream is distinct from batch order and latent noise.
        generator=torch.Generator().manual_seed(derive_seed(seed, "alignment-loader")),
    )
