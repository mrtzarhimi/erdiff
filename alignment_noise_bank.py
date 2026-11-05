"""Stable, independently keyed Gaussian draws for paired Phase 2 runs."""

from dataclasses import dataclass
import hashlib

import torch


@dataclass(frozen=True)
class NoiseBatch:
    posterior_eta: torch.Tensor
    diffusion_eps: torch.Tensor
    uniform_time: torch.Tensor


class AlignmentNoiseBank:
    """Generate fixed draws from (seed, namespace, trial ID, replicate)."""

    def __init__(self, seed: int = 42, replicates: int = 8):
        if replicates < 1:
            raise ValueError("replicates must be positive")
        self.seed = int(seed)
        self.replicates = int(replicates)

    def _generator(self, domain, trial_id, replicate, stream, device):
        key = f"{self.seed}|{domain}|{int(trial_id)}|{int(replicate)}|{stream}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(value)
        return generator

    def get(self, domain, trial_ids, replicate, seq_len, latent_dim, device):
        if not 0 <= int(replicate) < self.replicates:
            raise ValueError("replicate index is outside the configured bank")
        ids = torch.as_tensor(trial_ids).detach().cpu().tolist()
        eta, eps, times = [], [], []
        for trial_id in ids:
            for stream, output in (("posterior", eta), ("diffusion", eps)):
                generator = self._generator(domain, trial_id, replicate, stream, device)
                output.append(torch.randn((seq_len, latent_dim), generator=generator, device=device))
            generator = self._generator(domain, trial_id, replicate, "time", device)
            times.append(torch.rand((), generator=generator, device=device))
        return NoiseBatch(torch.stack(eta), torch.stack(eps), torch.stack(times))
