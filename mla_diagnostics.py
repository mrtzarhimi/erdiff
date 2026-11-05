"""Source-only MLA probes; no optimizer steps or global random draws."""

import json
import math
import os
from contextlib import contextmanager

import torch

from diffusion import fixed_noise_mla_loss
from vae import sample_latent_posterior


@contextmanager
def probe_modes(vae, denoiser):
    modes = [(module, module.training) for model in (vae, denoiser) for module in model.modules()]
    try:
        vae.eval()
        vae.encoder.rnn.train()  # cuDNN input gradients; this GRU has no internal dropout.
        denoiser.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


@torch.no_grad()
def variance_summary(vae, spikes, device, batch_size):
    means, logvars, predictions = [], [], []
    for chunk in spikes.split(batch_size):
        mu, logvar = vae.encoder(chunk.to(device))
        means.append(mu.cpu())
        logvars.append(logvar.cpu())
        predictions.append(vae.behavior_decoder(mu).cpu())
    mu = torch.cat(means)
    logvar = torch.cat(logvars)
    prediction = torch.cat(predictions)
    return {
        "mu_variance": mu.var(dim=(0, 1), unbiased=False).tolist(),
        "mu_trial_variance": mu.var(dim=0, unbiased=False).mean(dim=0).tolist(),
        "mu_temporal_variance": mu.var(dim=1, unbiased=False).mean(dim=0).tolist(),
        "posterior_variance": logvar.exp().mean(dim=(0, 1)).tolist(),
        "prediction_variance": prediction.var(dim=(0, 1), unbiased=False).tolist(),
        "mu_mean": mu.mean(dim=(0, 1)).tolist(),
        "prediction_mean": prediction.mean(dim=(0, 1)).tolist(),
    }


class MLADiagnostics:
    def __init__(self, dataset, train_indices, latent_dim, times, seed, trial_count, save_dir, context=None):
        generator = torch.Generator().manual_seed(seed)
        count = min(trial_count, len(dataset.spikes))
        if count < 1:
            raise ValueError("MLA diagnostics require at least one training trial.")
        indices = torch.randperm(len(dataset.spikes), generator=generator)[:count]
        self.spikes = dataset.spikes[indices].clone()
        self.all_spikes = dataset.spikes
        shape = (count, self.spikes.size(1), latent_dim)
        self.posterior_noise = torch.randn(shape, generator=generator)
        self.diffusion_noise = torch.randn(shape, generator=generator)
        self.times = times
        self.context = {} if context is None else context
        self.baseline = None
        self.path = os.path.join(save_dir, "mla_diagnostics.jsonl")
        with open(self.path, "w"):
            pass
        torch.save({
            "seed": seed,
            "training_offsets": indices,
            "extracted_trial_indices": torch.as_tensor(train_indices).cpu()[indices],
            "posterior_noise": self.posterior_noise,
            "diffusion_noise": self.diffusion_noise,
            "times": times,
            "configuration": self.context,
        }, os.path.join(save_dir, "mla_probe_fixture.pt"))

    def record(self, epoch, vae, denoiser, scheduler, device, batch_size):
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices), probe_modes(vae, denoiser):
            summary = variance_summary(vae, self.all_spikes, device, batch_size)
            if self.baseline is None:
                self.baseline = summary
            ratios = {
                key + "_ratio": sum(summary[key]) / max(sum(self.baseline[key]), 1e-12)
                for key in ("mu_variance", "mu_trial_variance", "posterior_variance", "prediction_variance")
            }
            probes = [self.gradient_probe(t, vae, denoiser, scheduler, device) for t in self.times]
        row = {"epoch": epoch, "configuration": self.context, "variance": summary, "ratios": ratios, "probes": probes}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"MLA variance epoch {epoch}: " + ", ".join(f"{name}={value:.4g}" for name, value in ratios.items()))
        for probe in probes:
            print(
                f"MLA fixed probe t={probe['time']:g}, std={probe['diffusion_std']:.4g}: "
                f"loss={probe['loss']:.4g}, read-in grad={probe['readin_grad_norm']:.4g}, "
                f"mu radial dot={probe['mu_radial_dot']:.4g}, FD error={probe['fd_relative_error']:.4g}"
            )
        return row

    def gradient_probe(self, time, vae, denoiser, scheduler, device):
        with torch.enable_grad():
            mu, logvar = vae.encoder(self.spikes.to(device))
            z0 = sample_latent_posterior(mu, logvar, self.posterior_noise.to(device))
            eps = self.diffusion_noise.to(device)
            t = z0.new_full((z0.size(0),), time)
            loss = fixed_noise_mla_loss(denoiser, scheduler, z0, t, eps)
            parameters = tuple(vae.encoder.read_in.parameters())
            gradients = torch.autograd.grad(loss, (z0, mu, logvar) + parameters)
        gz, gmu, glogvar = (g.detach() for g in gradients[:3])
        direction = eps / eps.norm().clamp_min(1e-12)
        step = 1e-3 * max(1.0, z0.detach().square().mean().sqrt().item())
        with torch.no_grad():
            plus = fixed_noise_mla_loss(denoiser, scheduler, z0.detach() + step * direction, t, eps)
            minus = fixed_noise_mla_loss(denoiser, scheduler, z0.detach() - step * direction, t, eps)
            numeric = ((plus - minus) / (2 * step)).item()
            analytic = (gz * direction).sum().item()
            a, std = scheduler.marginal_params(t[:1])
            centered_mu = mu.detach() - mu.detach().mean(dim=(0, 1), keepdim=True)
        return {
            "time": time, "signal_coefficient": a.item(), "diffusion_std": std.item(),
            "signal_noise_power_ratio": (a.square() / std.square().clamp_min(1e-12)).item(),
            "loss": loss.item(), "latent_grad_norm": gz.norm().item(),
            "mu_grad_norm": gmu.norm().item(), "logvar_grad_norm": glogvar.norm().item(),
            "logvar_grad_sum": glogvar.sum().item(),
            "mu_radial_dot": (gmu * centered_mu).sum().item(),
            "readin_grad_norm": math.sqrt(sum(g.detach().square().sum().item() for g in gradients[3:])),
            "fd_step": step, "fd_analytic": analytic, "fd_numeric": numeric,
            "fd_absolute_error": abs(analytic - numeric),
            "fd_relative_error": abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-6),
        }
