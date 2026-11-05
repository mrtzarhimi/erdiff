"""Stable, independently keyed Gaussian draws for paired Phase 2 runs."""

from dataclasses import dataclass
import hashlib
import os

import torch


BANK_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NoiseBatch:
    posterior_eta: torch.Tensor
    diffusion_eps: torch.Tensor
    uniform_time: torch.Tensor


class AlignmentNoiseBank:
    """Cache fixed draws keyed by (seed, domain, trial ID, replicate, stream).

    Noise is generated on CPU so a saved bank remains identical when the same
    run is resumed on a different accelerator. ``register_domain`` binds the
    bank to the complete trial identity and latent shape before optimization.
    """

    def __init__(self, seed: int = 42, replicates: int = 8):
        if replicates < 1:
            raise ValueError("replicates must be positive")
        self.seed = int(seed)
        self.replicates = int(replicates)
        self.domains = {}
        self.entries = {}

    @staticmethod
    def _seed_for(seed, domain, trial_id, replicate, stream):
        key = f"{seed}|{domain}|{int(trial_id)}|{int(replicate)}|{stream}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)

    def _generator(self, domain, trial_id, replicate, stream):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed_for(self.seed, domain, trial_id, replicate, stream))
        return generator

    def register_domain(self, domain, trial_ids, seq_len, latent_dim):
        """Bind a namespace to stable IDs and expected per-trial draw shapes."""
        ids = tuple(sorted({int(value) for value in torch.as_tensor(trial_ids).cpu().tolist()}))
        if not domain or not ids or seq_len < 1 or latent_dim < 1:
            raise ValueError("A noise domain needs a name, trial IDs and positive latent dimensions.")
        identity = {"trial_ids": ids, "seq_len": int(seq_len), "latent_dim": int(latent_dim)}
        current = self.domains.get(str(domain))
        if current is not None and current != identity:
            raise ValueError(f"Noise bank domain identity or shape is incompatible: {domain}")
        self.domains[str(domain)] = identity
        for (entry_domain, trial_id, _), value in self.entries.items():
            if entry_domain == str(domain):
                if trial_id not in ids or tuple(value[0].shape) != (seq_len, latent_dim):
                    raise ValueError(f"Saved noise entry is incompatible with domain {domain}.")

    def get(self, domain, trial_ids, replicate, seq_len, latent_dim, device):
        if not 0 <= int(replicate) < self.replicates:
            raise ValueError("replicate index is outside the configured bank")
        domain = str(domain)
        ids = [int(value) for value in torch.as_tensor(trial_ids).detach().cpu().tolist()]
        registered = self.domains.get(domain)
        if registered is not None:
            if registered["seq_len"] != int(seq_len) or registered["latent_dim"] != int(latent_dim):
                raise ValueError(f"Requested noise shape is incompatible with domain {domain}.")
            unknown = set(ids) - set(registered["trial_ids"])
            if unknown:
                raise ValueError(f"Unknown trial IDs in noise domain {domain}: {sorted(unknown)}")

        eta, eps, times = [], [], []
        for trial_id in ids:
            key = (domain, trial_id, int(replicate))
            if key not in self.entries:
                eta_gen = self._generator(domain, trial_id, replicate, "posterior_eta")
                eps_gen = self._generator(domain, trial_id, replicate, "diffusion_epsilon")
                time_gen = self._generator(domain, trial_id, replicate, "uniform_time")
                self.entries[key] = (
                    torch.randn((seq_len, latent_dim), generator=eta_gen, dtype=torch.float32),
                    torch.randn((seq_len, latent_dim), generator=eps_gen, dtype=torch.float32),
                    torch.rand((), generator=time_gen, dtype=torch.float32),
                )
            eta_i, eps_i, time_i = self.entries[key]
            if tuple(eta_i.shape) != (seq_len, latent_dim) or tuple(eps_i.shape) != (seq_len, latent_dim):
                raise ValueError(f"Saved noise entry has an incompatible shape for {domain}/{trial_id}.")
            eta.append(eta_i)
            eps.append(eps_i)
            times.append(time_i)

        if not ids:
            empty = torch.empty((0, seq_len, latent_dim), dtype=torch.float32, device=device)
            return NoiseBatch(empty, empty.clone(), torch.empty((0,), dtype=torch.float32, device=device))
        return NoiseBatch(
            torch.stack(eta).to(device=device),
            torch.stack(eps).to(device=device),
            torch.stack(times).to(device=device),
        )

    def content_hash(self):
        """Hash canonical domain metadata and cached tensor bytes."""
        digest = hashlib.sha256()
        for domain in sorted(self.domains):
            identity = self.domains[domain]
            digest.update(f"D|{domain}|{identity['seq_len']}|{identity['latent_dim']}|".encode())
            digest.update(",".join(map(str, identity["trial_ids"])).encode())
        for key in sorted(self.entries):
            domain, trial_id, replicate = key
            digest.update(f"E|{domain}|{trial_id}|{replicate}|".encode())
            for tensor in self.entries[key]:
                value = tensor.detach().cpu().contiguous()
                digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def save(self, path):
        """Atomically persist the noise bank and its canonical content hash."""
        path = os.fspath(path)
        payload = {
            "schema_version": BANK_SCHEMA_VERSION,
            "seed": self.seed,
            "replicates": self.replicates,
            "domains": {
                domain: {"trial_ids": list(value["trial_ids"]),
                         "seq_len": value["seq_len"], "latent_dim": value["latent_dim"]}
                for domain, value in self.domains.items()
            },
            "entries": [
                {"domain": key[0], "trial_id": key[1], "replicate": key[2],
                 "posterior_eta": values[0], "diffusion_eps": values[1], "uniform_time": values[2]}
                for key, values in sorted(self.entries.items())
            ],
            "content_sha256": self.content_hash(),
        }
        parent = os.path.dirname(os.path.realpath(path))
        os.makedirs(parent, exist_ok=True)
        temporary_path = path + ".tmp"
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        return payload["content_sha256"]

    @classmethod
    def load(cls, path, seed=None, replicates=None):
        path = os.fspath(path)
        payload = torch.load(path, map_location="cpu")
        if payload.get("schema_version") != BANK_SCHEMA_VERSION:
            raise ValueError("Unsupported alignment noise bank schema.")
        if seed is not None and int(seed) != int(payload["seed"]):
            raise ValueError("Saved alignment noise bank seed is incompatible.")
        if replicates is not None and int(replicates) != int(payload["replicates"]):
            raise ValueError("Saved alignment noise bank replicate count is incompatible.")
        bank = cls(payload["seed"], payload["replicates"])
        for domain, identity in payload.get("domains", {}).items():
            bank.register_domain(domain, identity["trial_ids"], identity["seq_len"], identity["latent_dim"])
        for entry in payload.get("entries", []):
            key = (str(entry["domain"]), int(entry["trial_id"]), int(entry["replicate"]))
            if key in bank.entries:
                raise ValueError(f"Duplicate saved alignment noise entry: {key}")
            eta = entry["posterior_eta"].detach().cpu().contiguous()
            eps = entry["diffusion_eps"].detach().cpu().contiguous()
            time = entry["uniform_time"].detach().cpu().reshape(()).contiguous()
            if eta.shape != eps.shape or eta.ndim != 2 or not 0 <= key[2] < bank.replicates:
                raise ValueError(f"Malformed saved alignment noise entry: {key}")
            identity = bank.domains.get(key[0])
            if identity is not None and (
                key[1] not in identity["trial_ids"]
                or tuple(eta.shape) != (identity["seq_len"], identity["latent_dim"])
            ):
                raise ValueError(f"Saved alignment noise identity/shape mismatch: {key}")
            bank.entries[key] = (eta, eps, time)
        if bank.content_hash() != payload.get("content_sha256"):
            raise ValueError("Saved alignment noise bank hash does not match its contents.")
        return bank
