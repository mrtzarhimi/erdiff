import argparse
import copy
import hashlib
import json
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import NHPDataset, get_nhp_dataloaders
from vae import SequentialVAE, trajectory_smoothness_loss, sample_latent_posterior
from train_phase1 import evaluate_phase1
from checkpoint_provenance import code_revision, file_fingerprint, file_sha256
from phase2_freeze_guard import FrozenDiffusionGuard
from alignment_noise_bank import AlignmentNoiseBank
from neuron_input_mapping import NeuronInputMap
from alignment_gradient_diagnostics import (
    component_gradient_report,
    calibrate_source_gradient_scale,
    load_gradient_calibration,
    save_gradient_calibration,
)
from diffusion import (
    LatentDiffusionModel,
    VPSDEScheduler,
    total_alignment_loss,
)
from phase2_run_manifest import (
    build_run_manifest,
    exact_reproduction_claimable,
    load_reference,
    resolve_run,
    save_run_manifest,
    validate_reference_compatibility,
    validate_resolved_config,
)
from phase2_sampling import derive_seed, source_sampling_generator


@dataclass(frozen=True)
class SourcePosteriorBank:
    mu: torch.Tensor
    logvar: torch.Tensor
    trial_ids: torch.Tensor


def set_seed(seed: int = 42):
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_training_view(loader, batch_size: int):
    """Visit every training trial once, without spike noise or random omission."""
    dataset = loader.dataset
    return DataLoader(
        NHPDataset(dataset.spikes, dataset.velocities, dataset.targets,
                   is_train=False, include_behavior=False, trial_ids=dataset.trial_ids),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )


@torch.no_grad()
def extract_source_latent_bank(vae: SequentialVAE, source_loader, device: torch.device):
    """
    Extract posterior parameters for every clean source training trial.
    """
    vae.eval()
    source_means = []
    source_logvars = []
    source_trial_ids = []
    for batch in source_loader:
        spikes = batch["spikes"].to(device)
        mu, logvar = vae.encoder(spikes)
        source_means.append(mu)
        source_logvars.append(logvar)
        source_trial_ids.append(batch["trial_id"])
    return SourcePosteriorBank(
        mu=torch.cat(source_means, dim=0),
        logvar=torch.cat(source_logvars, dim=0),
        trial_ids=torch.cat(source_trial_ids),
    )


def build_source_gradient_calibration(
    vae, diffusion_model, scheduler, source_loader, source_latents_bank, device,
    *, alpha, ot_epsilon, sinkhorn_iters, probe_batches, seed, provenance,
    mla_variance_weight=0.0, mla_variance_floor=0.9, mla_variance_scope="batch",
):
    """Measure fixed source-only loss gradients through an independent read-in copy."""
    if probe_batches < 1:
        raise ValueError("Gradient calibration needs at least one source probe batch.")
    probe_model = copy.deepcopy(vae)
    probe_model.freeze_for_alignment()
    probe_model.eval()
    probe_model.encoder.rnn.train()
    parameters = tuple(probe_model.encoder.read_in.parameters())

    bank_index = {int(trial_id): index for index, trial_id in
                  enumerate(source_latents_bank.trial_ids.tolist())}
    generators = {
        name: torch.Generator(device="cpu").manual_seed(int(seed) + offset)
        for offset, name in enumerate(("target_eta", "source_eta", "time", "diffusion_eps"), start=101)
    }
    probes, sample_ids = [], []
    source_variance = None
    source_variance_pool = None
    if mla_variance_weight > 0:
        from mla_stabilization import source_trial_variance
        source_variance = source_trial_variance(source_latents_bank.mu)
        if mla_variance_scope == "full_train":
            source_variance_pool = source_loader.dataset.spikes.to(device)
    diffusion_model.eval()
    for batch_index, batch in enumerate(source_loader):
        if batch_index >= probe_batches:
            break
        spikes = batch["spikes"].to(device)
        trial_ids = batch["trial_id"].cpu().long()
        indices_cpu = torch.tensor([bank_index[int(i)] for i in trial_ids.tolist()], dtype=torch.long)
        indices = indices_cpu.to(source_latents_bank.mu.device)

        mu, logvar = probe_model.encoder(spikes)
        posterior_eta = torch.randn(
            mu.shape, generator=generators["target_eta"], dtype=mu.dtype, device="cpu"
        ).to(device)
        z_target = sample_latent_posterior(mu, logvar, posterior_eta)

        source_mu = source_latents_bank.mu[indices]
        source_logvar = source_latents_bank.logvar[indices]
        source_eta = torch.randn(
            source_mu.shape, generator=generators["source_eta"], dtype=source_mu.dtype, device="cpu"
        ).to(source_mu.device)
        z_source = sample_latent_posterior(source_mu, source_logvar, source_eta).detach()

        times_cpu = torch.rand(
            (len(trial_ids),), generator=generators["time"], dtype=mu.dtype
        )
        times = (scheduler.t_min + times_cpu * (scheduler.T - scheduler.t_min)).to(device)
        diffusion_eps = torch.randn(
            mu.shape, generator=generators["diffusion_eps"], dtype=mu.dtype, device="cpu"
        ).to(device)
        _, loss_mla, loss_ot = total_alignment_loss(
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            z_target=z_target,
            z_source=z_source,
            alpha=alpha,
            ot_epsilon=ot_epsilon,
            sinkhorn_iters=sinkhorn_iters,
            return_components=True,
            mla_noise=diffusion_eps,
            mla_timesteps=times,
        )
        probe = {"mla": loss_mla, "ot": loss_ot}
        if mla_variance_weight > 0:
            from mla_stabilization import encoder_pool_variance_loss, source_variance_floor_loss
            if mla_variance_scope == "full_train":
                variance_loss = encoder_pool_variance_loss(
                    probe_model.encoder, source_variance_pool, source_variance, mla_variance_floor
                )
            else:
                variance_loss = source_variance_floor_loss(mu, source_variance, mla_variance_floor)
            probe["variance"] = variance_loss
        probes.append(probe)
        sample_ids.extend(trial_ids.tolist())

    if not probes:
        raise ValueError("No source trials were available for gradient calibration.")
    result = calibrate_source_gradient_scale(
        probes, parameters, alpha=alpha, mla_variance_weight=mla_variance_weight,
        metadata={**provenance, "sample_ids": sample_ids,
                  "probe_batches": len(probes), "calibration_dtype": str(parameters[0].dtype)},
    )
    result["provenance"] = {**provenance, "sample_ids": sample_ids}
    del probe_model
    return result


@torch.no_grad()
def initialize_target_read_in(
    source_read_in: nn.Linear,
    src_loader,
    tgt_loader,
    n_src_neurons: int,
    n_tgt_neurons: int,
    hidden_dim: int,
    device: torch.device,
    init_mode: str = "source_or_random",
) -> nn.Linear:
    """
    Initializes target read_in layer.
    - If neuron counts match, copies source read_in weights directly.
    - If neuron counts differ, a fresh target read-in matches the 2023 author code.
      The previous ridge warm start remains an optional experimental variant.
    """
    new_read_in = nn.Linear(n_tgt_neurons, hidden_dim).to(device)
    if init_mode not in ("source_or_random", "ridge", "random"):
        raise ValueError(f"Unknown target read-in initialization: {init_mode}")

    if init_mode == "random":
        return new_read_in

    if n_tgt_neurons == n_src_neurons:
        new_read_in.bias.copy_(source_read_in.bias)
        new_read_in.weight.copy_(source_read_in.weight)
        return new_read_in

    if init_mode == "source_or_random":
        return new_read_in

    new_read_in.bias.copy_(source_read_in.bias)

    # Gather temporal mean profile across training trials: shape (seq_len, n_neurons)
    src_batches = [b["spikes"].to(device) for b in src_loader]
    tgt_batches = [b["spikes"].to(device) for b in tgt_loader]

    # Compute mean firing rate trajectory across trials: (seq_len, N)
    X_src_mean = torch.cat(src_batches, dim=0).mean(dim=0)  # (L, N_src)
    X_tgt_mean = torch.cat(tgt_batches, dim=0).mean(dim=0)  # (L, N_tgt)

    # Source hidden activations before bias: H_src = X_src_mean @ W_src^T  -> (L, hidden_dim)
    H_src = torch.matmul(X_src_mean, source_read_in.weight.t())

    # Solve Ridge least squares: X_tgt_mean @ W_tgt^T = H_src
    # W_tgt^T = (X_tgt^T X_tgt + lambda I)^(-1) X_tgt^T H_src
    reg = 10.0 * torch.eye(n_tgt_neurons, device=device)
    A = torch.matmul(X_tgt_mean.t(), X_tgt_mean) + reg
    B = torch.matmul(X_tgt_mean.t(), H_src)
    W_tgt_T = torch.linalg.solve(A, B)  # (N_tgt, hidden_dim)

    new_read_in.weight.copy_(W_tgt_T.t())
    return new_read_in


def train_epoch_phase2(
    vae: SequentialVAE,
    diffusion_model: LatentDiffusionModel,
    scheduler: VPSDEScheduler,
    target_loader,
    source_latents_bank,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha: float = 0.5,
    ot_epsilon: float = 0.1,
    sinkhorn_iters: int = 100,
    smooth_weight: float = 0.0,
    mla_t_min=None,
    mla_t_max=None,
    mla_fixed_t=None,
    source_variance=None,
    mla_variance_weight: float = 0.0,
    mla_variance_floor: float = 0.9,
    mla_variance_scope: str = "batch",
    variance_pool_spikes=None,
    noise_bank=None,
    noise_replicate=0,
    gradient_balance="none",
    ot_scale=1.0,
    log_component_gradients=False,
    source_sampling_rng=None,
    target_noise_domain="target",
    source_noise_domain="source",
):
    """
    Phase 2: Maximum Likelihood Alignment (Algorithm 2).
    """
    if gradient_balance not in ("none", "source_calibrated"):
        raise ValueError(f"Unknown gradient balance mode: {gradient_balance}")
    if not torch.isfinite(torch.tensor(float(ot_scale))) or ot_scale <= 0:
        raise ValueError("OT gradient scale must be finite and positive.")
    vae.eval()                    # Disable dropout in encoder/decoder
    vae.encoder.rnn.train()       # Required by cuDNN to backprop through frozen GRU into read_in
    diffusion_model.eval()        # Frozen score network

    total_loss = 0.0
    total_mla = 0.0
    total_ot = 0.0
    total_smooth = 0.0
    total_variance_floor = 0.0
    total_variance_contribution = 0.0
    total_grad_norm = 0.0
    clipped_steps = 0
    sample_exposures = 0
    full_variance_forwards = 0
    full_variance_sample_exposures = 0
    component_gradient_totals = {}
    component_cosine_totals = {}
    component_cosine_counts = {}
    component_gradient_batches = 0
    source_mu = source_latents_bank.mu
    source_logvar = source_latents_bank.logvar
    source_trial_ids = source_latents_bank.trial_ids
    num_source_samples = source_mu.size(0)

    for batch in target_loader:
        target_spikes = batch["spikes"].to(device)
        batch_size = target_spikes.size(0)
        sample_exposures += batch_size

        optimizer.zero_grad()

        # Algorithm 2 samples target latents from the encoder posterior.
        mu_t, logvar_t = vae.encoder(target_spikes)
        target_noise = source_noise = mla_noise = mla_times = None
        if noise_bank is None:
            z_target = sample_latent_posterior(mu_t, logvar_t)
        else:
            target_noise = noise_bank.get(target_noise_domain, batch["trial_id"], noise_replicate,
                                          mu_t.shape[1], mu_t.shape[2], device)
            z_target = sample_latent_posterior(mu_t, logvar_t, target_noise.posterior_eta)

        # 2. Sample matching batch of source latents for Sinkhorn regularizer
        if source_sampling_rng is None:
            perm = torch.randperm(num_source_samples, device=device)[:batch_size]
            selected_source_ids = source_trial_ids[perm.detach().cpu()]
        else:
            source_indices_cpu = torch.randperm(
                num_source_samples, generator=source_sampling_rng
            )[:batch_size]
            perm = source_indices_cpu.to(device)
            selected_source_ids = source_trial_ids[source_indices_cpu]
        if noise_bank is None:
            z_source_batch = sample_latent_posterior(source_mu[perm], source_logvar[perm])
        else:
            source_noise = noise_bank.get(source_noise_domain, selected_source_ids, noise_replicate,
                                          source_mu.shape[1], source_mu.shape[2], device)
            z_source_batch = sample_latent_posterior(source_mu[perm], source_logvar[perm], source_noise.posterior_eta)
            low = scheduler.t_min if mla_t_min is None else mla_t_min
            high = scheduler.T if mla_t_max is None else mla_t_max
            mla_times = (torch.full_like(target_noise.uniform_time, mla_fixed_t) if mla_fixed_t is not None
                         else target_noise.uniform_time * (high - low) + low)
            mla_noise = target_noise.diffusion_eps

        # 3. Compute paper MLA + regularized OT; smoothness is an optional variant.
        loss_align, loss_mla, loss_ot = total_alignment_loss(
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            z_target=z_target,
            z_source=z_source_batch,
            alpha=alpha,
            ot_epsilon=ot_epsilon,
            sinkhorn_iters=sinkhorn_iters,
            return_components=True,
            mla_t_min=mla_t_min,
            mla_t_max=mla_t_max,
            mla_fixed_t=mla_fixed_t,
            mla_noise=mla_noise,
            mla_timesteps=mla_times,
        )
        loss = (1.0 - alpha) * loss_mla + alpha * ot_scale * loss_ot
        gradient_losses = {"mla": loss_mla, "ot": loss_ot}
        gradient_coefficients = {"mla": 1.0 - alpha, "ot": alpha * ot_scale}
        variance_value = 0.0
        variance_contribution = 0.0
        if mla_variance_weight > 0 and alpha < 1.0:
            from mla_stabilization import source_variance_floor_loss
            if source_variance is None:
                raise ValueError("Variance stabilization requires a frozen source reference.")
            if mla_variance_scope == "full_train":
                from mla_stabilization import encoder_pool_variance_loss
                if variance_pool_spikes is None:
                    raise ValueError("Full-training variance requires the clean adaptation training pool.")
                loss_variance = encoder_pool_variance_loss(
                    vae.encoder, variance_pool_spikes, source_variance, mla_variance_floor
                )
                full_variance_forwards += 1
                full_variance_sample_exposures += variance_pool_spikes.size(0)
            elif mla_variance_scope == "batch":
                loss_variance = source_variance_floor_loss(mu_t, source_variance, mla_variance_floor)
            else:
                raise ValueError(f"Unknown variance scope: {mla_variance_scope}")
            weighted_variance = (1.0 - alpha) * mla_variance_weight * loss_variance
            loss = loss + weighted_variance
            gradient_losses["variance"] = loss_variance
            gradient_coefficients["variance"] = (1.0 - alpha) * mla_variance_weight
            variance_value = loss_variance.item()
            variance_contribution = weighted_variance.item()
        smooth_value = 0.0
        if smooth_weight:
            loss_smooth = trajectory_smoothness_loss(mu_t)
            smooth_value = loss_smooth.item()
            loss = loss + smooth_weight * loss_smooth
            gradient_losses["smooth"] = loss_smooth
            gradient_coefficients["smooth"] = smooth_weight

        if log_component_gradients:
            report = component_gradient_report(
                gradient_losses, tuple(vae.encoder.read_in.parameters()), gradient_coefficients
            )
            for name, values in report["components"].items():
                totals = component_gradient_totals.setdefault(name, {"norm": 0.0, "weighted_norm": 0.0})
                if values["norm"] is not None:
                    totals["norm"] += values["norm"]
                    totals["weighted_norm"] += values["weighted_norm"]
            for name, value in report["cosines"].items():
                if value is not None:
                    component_cosine_totals[name] = component_cosine_totals.get(name, 0.0) + value
                    component_cosine_counts[name] = component_cosine_counts.get(name, 0) + 1
            component_gradient_batches += 1

        # 4. Update ONLY encoder.read_in
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(vae.encoder.read_in.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mla += loss_mla.item()
        total_ot += loss_ot.item()
        total_smooth += smooth_value
        total_variance_floor += variance_value
        total_variance_contribution += variance_contribution
        total_grad_norm += grad_norm.item()
        clipped_steps += int(grad_norm.item() > 1.0)

    batches = len(target_loader)
    component_gradient_summary = None
    if component_gradient_batches:
        component_gradient_summary = {
            "components": {
                name: {key: value / component_gradient_batches for key, value in totals.items()}
                for name, totals in component_gradient_totals.items()
            },
            "cosines": {
                name: value / component_cosine_counts[name]
                for name, value in component_cosine_totals.items()
            },
        }
    return {
        "total": total_loss / batches,
        "mla": total_mla / batches,
        "ot": total_ot / batches,
        "smooth": total_smooth / batches,
        "variance_floor": total_variance_floor / batches,
        "variance_contribution": total_variance_contribution / batches,
        "grad_norm": total_grad_norm / batches,
        "clipped_steps": clipped_steps,
        "updates": batches,
        "sample_exposures": sample_exposures,
        "full_variance_forwards": full_variance_forwards,
        "full_variance_sample_exposures": full_variance_sample_exposures,
        "component_gradient_report": component_gradient_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Target Domain Maximum Likelihood Alignment (ERDiff)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed used only to define trial partitions")
    parser.add_argument("--source_checkpoint", type=str, required=True, help="Path to saved Phase 1 checkpoint (.pt)")
    parser.add_argument("--source_self_check", action="store_true",
                        help="Diagnostic: align source training data to itself and report source validation R2 without checkpoint selection")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to source session .mat file")
    parser.add_argument("--target_data_path", type=str, required=True, help="Path to target session .mat file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=39, help="Number of time bins per trial")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for target read-in probing")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Optional read-in weight decay variant")
    parser.add_argument("--epochs", type=int, default=150, help="Alignment epochs")
    parser.add_argument("--reference_run", type=str, default=None,
                        help="Optional immutable JSON run manifest used to lock settings and splits")
    parser.add_argument("--factor_override", action="append", default=[], metavar="KEY=JSON",
                        help="Declared reference factor override, e.g. mla_variance_weight=10")
    parser.add_argument("--frozen_check", choices=("final", "every_epoch"), default="every_epoch")
    parser.add_argument("--alignment_noise_mode", choices=("resampled", "fixed_bank"), default="resampled")
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--noise_bank_replicates", type=int, default=8)
    parser.add_argument("--noise_bank_path", type=str, default=None,
                        help="Optional saved bank path; fixed_bank loads/reuses it or creates it there")
    parser.add_argument("--log_component_gradients", action="store_true")
    parser.add_argument("--gradient_balance", choices=("none", "source_calibrated"), default="none",
                        help="Optionally scale the OT gradient using fixed source-training probes")
    parser.add_argument("--gradient_calibration_path", default=None,
                        help="Reusable source-only calibration JSON; defaults inside save_dir")
    parser.add_argument("--gradient_calibration_batches", type=int, default=3,
                        help="Number of fixed clean source-training batches used for calibration")
    parser.add_argument("--gradient_calibration_seed", type=int, default=42,
                        help="Seed for independent fixed source-only calibration fixtures")
    parser.add_argument("--augment_mult", type=int, default=3, help="Target data repetition when --augment_target is enabled")
    parser.add_argument("--augment_target", action="store_true", help="Experimental variant: augment target adaptation spikes")
    parser.add_argument("--readin_init", choices=("source_or_random", "ridge", "random"), default="source_or_random",
                        help="Initialize from source when raw widths match, fit ridge, or use random weights")
    parser.add_argument("--match_input_size", action="store_true",
                        help="Map target channels to source checkpoint input width before the read-in")
    parser.add_argument("--input_mapping_seed", type=int, default=42,
                        help="Seed for deterministic channel selection when target width exceeds source width")
    parser.add_argument("--alpha", type=float, default=0.5, help="Trade-off weighting Sinkhorn loss")
    parser.add_argument("--ot_epsilon", type=float, default=None,
                        help="Entropy weight for full-trajectory transport; default preserves the former per-element temperature")
    parser.add_argument("--sinkhorn_iters", type=int, default=100, help="Log-domain transport scaling iterations")
    parser.add_argument("--smooth_weight", type=float, default=0.0, help="Optional trajectory-smoothness variant; paper default is zero")
    parser.add_argument("--mla_diagnostics", action="store_true", help="Source-only variance and fixed-noise gradient probes")
    parser.add_argument("--mla_diagnostics_only", action="store_true", help="Record initial probes and exit without optimizer updates")
    parser.add_argument("--mla_probe_times", type=float, nargs="+", default=[0.01, 0.05, 0.2, 0.5, 0.9])
    parser.add_argument("--mla_probe_trials", type=int, default=16)
    parser.add_argument("--mla_probe_seed", type=int, default=42)
    parser.add_argument("--mla_diagnostic_every", type=int, default=10)
    parser.add_argument("--mla_t_min", type=float, default=None, help="Diagnostic ablation: restrict MLA diffusion-time range")
    parser.add_argument("--mla_t_max", type=float, default=None, help="Diagnostic ablation: restrict MLA diffusion-time range")
    parser.add_argument("--mla_fixed_t", type=float, default=None, help="Diagnostic ablation: train MLA at one diffusion time")
    parser.add_argument("--mla_variance_weight", type=float, default=0.0,
                        help="Optional source-variance stabilization weight within the MLA branch; zero disables it")
    parser.add_argument("--mla_variance_floor", type=float, default=0.9,
                        help="Floor as a fraction of source trial standard deviation, in (0, 1]")
    parser.add_argument("--mla_variance_scope", choices=("batch", "full_train"), default="batch",
                        help="Estimate penalty variance on the minibatch or all clean adaptation training trials")
    args = parser.parse_args()
    factor_overrides = {}
    for item in args.factor_override:
        if "=" not in item:
            raise ValueError("--factor_override must use KEY=JSON_VALUE.")
        key, raw_value = item.split("=", 1)
        try:
            factor_overrides[key] = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON value for --factor_override {key!r}.") from exc
    reference = load_reference(args.reference_run) if args.reference_run else None
    if reference is not None and factor_overrides:
        # Resolve first so undeclared changes fail before loading data or constructing models.
        resolve_run(reference, factor_overrides)
        for key, value in factor_overrides.items():
            if not hasattr(args, key):
                raise ValueError(f"Reference factor {key!r} has no Phase 2 CLI setting.")
            setattr(args, key, value)
    elif factor_overrides:
        raise ValueError("--factor_override requires --reference_run.")
    import math
    if not math.isfinite(args.mla_variance_weight) or args.mla_variance_weight < 0:
        raise ValueError("MLA variance weight must be finite and nonnegative.")
    if not math.isfinite(args.mla_variance_floor) or not 0 < args.mla_variance_floor <= 1:
        raise ValueError("MLA variance floor must lie in (0, 1].")
    if args.gradient_calibration_batches < 1:
        raise ValueError("Gradient calibration batch count must be positive.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    if args.gradient_balance == "source_calibrated" and args.alpha in (0.0, 1.0):
        raise ValueError("Source gradient balancing requires both MLA and OT to be active (0 < alpha < 1).")
    if args.noise_bank_replicates < 1:
        raise ValueError("Noise bank replicate count must be positive.")

    if args.mla_diagnostics_only:
        args.mla_diagnostics = True
    if args.mla_diagnostics and not args.source_self_check:
        raise ValueError("MLA diagnostics currently require --source_self_check; only source training trials are probed.")
    if args.mla_probe_trials < 1 or args.mla_diagnostic_every < 1:
        raise ValueError("Diagnostic trial count and interval must be positive.")

    if args.source_self_check and os.path.realpath(args.source_data_path) != os.path.realpath(args.target_data_path):
        raise ValueError("--source_self_check requires identical source and target data paths.")

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Source and Target Data
    source_fingerprint = file_sha256(args.source_checkpoint)
    checkpoint = torch.load(args.source_checkpoint, map_location=device)
    print(f"Source checkpoint: {os.path.realpath(args.source_checkpoint)} | SHA256: {source_fingerprint}")
    print(
        f"Source selected epoch: {checkpoint.get('epoch', 'unknown')} | "
        f"validation R2: {checkpoint.get('best_r2', 'unknown')} | "
        f"training seed: {checkpoint.get('training_seed', 'unknown')}"
    )
    noise_config = checkpoint.get("diffusion_training", {})
    if noise_config.get("latent_distribution") != "posterior_sample":
        raise ValueError(
            "Source diffusion checkpoint was not trained on posterior samples. "
            "Retrain Phase 1 before using paper-form posterior sampling in Phase 2."
        )
    if "split_indices" not in checkpoint or "split_seed" not in checkpoint:
        raise ValueError("Source checkpoint is missing saved trial splits; retrain Phase 1.")
    if "velocity_mean" not in checkpoint or "velocity_std" not in checkpoint:
        raise ValueError(
            "Source checkpoint is missing training velocity statistics; retrain Phase 1 "
            "before alignment so target labels cannot determine the decoder scale."
        )
    src_train_loader, src_validation_loader, _, n_src_neurons, source_data_metadata = get_nhp_dataloaders(
        file_path=args.source_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        augment_multiplier=1,
        split_seed=checkpoint["split_seed"],
        split_indices=checkpoint["split_indices"],
        include_validation=args.source_self_check,
        include_test=False,
    )
    checkpoint_velocity_stats = {
        "velocity_mean": checkpoint["velocity_mean"],
        "velocity_std": checkpoint["velocity_std"],
    }
    reference_target_splits = None
    if reference is not None:
        reference_target_splits = reference.get("target_split_indices", reference.get("split_indices"))
        if reference_target_splits is None:
            raise ValueError("Reference manifest must contain exact target split indices.")
    tgt_train_loader, _, _, n_tgt_neurons, target_data_metadata = get_nhp_dataloaders(
        file_path=args.target_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        augment_multiplier=args.augment_mult,
        split_seed=args.split_seed,
        velocity_stats=checkpoint_velocity_stats,
        split_indices=(checkpoint["split_indices"] if args.source_self_check else reference_target_splits),
        include_validation=False,
        include_test=False,
        include_behavior=False,
    )
    input_mapping = (
        NeuronInputMap.create(n_tgt_neurons, n_src_neurons, args.input_mapping_seed)
        if args.match_input_size else None
    )
    model_input_width = input_mapping.model_width if input_mapping is not None else n_tgt_neurons
    print(
        f"Source neurons: {n_src_neurons} | Target raw channels: {n_tgt_neurons} | "
        f"Model input channels: {model_input_width}"
    )
    clean_source_loader = clean_training_view(src_train_loader, args.batch_size)
    clean_target_loader = clean_training_view(tgt_train_loader, args.batch_size)
    from phase2_sampling import make_alignment_loader
    optimization_dataset = tgt_train_loader.dataset if args.augment_target else clean_target_loader.dataset
    alignment_loader = make_alignment_loader(optimization_dataset, args.batch_size, args.seed)
    batch_count = len(alignment_loader)
    small_batch, extra = divmod(len(optimization_dataset), batch_count)
    print(
        f"Adaptation batches: {batch_count}, sizes {small_batch}–{small_batch + bool(extra)} | "
        f"all {len(optimization_dataset)} samples visited once per shuffled epoch"
    )

    # 2. Restore Source Models from Phase 1
    latent_dim = checkpoint["latent_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    if args.ot_epsilon is None:
        args.ot_epsilon = 0.1 * args.seq_len * latent_dim
    print(f"Transport entropy weight: {args.ot_epsilon:g}")

    runtime_config = {
        "seed": args.seed,
        "split_seed": target_data_metadata["split_seed"],
        "epochs": args.epochs,
        "seq_len": args.seq_len,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "source_neurons": n_src_neurons,
        "target_neurons": n_tgt_neurons,
        "model_input_width": model_input_width,
        "match_input_size": args.match_input_size,
        "input_mapping_seed": args.input_mapping_seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha,
        "ot_epsilon": args.ot_epsilon,
        "sinkhorn_iters": args.sinkhorn_iters,
        "smooth_weight": args.smooth_weight,
        "mla_variance_weight": args.mla_variance_weight,
        "mla_variance_floor": args.mla_variance_floor,
        "mla_variance_scope": args.mla_variance_scope,
        "mla_t_min": args.mla_t_min,
        "mla_t_max": args.mla_t_max,
        "mla_fixed_t": args.mla_fixed_t,
        "readin_init": args.readin_init,
        "augment_target": args.augment_target,
        "augment_mult": args.augment_mult,
        "alignment_noise_mode": args.alignment_noise_mode,
        "noise_seed": args.noise_seed,
        "noise_bank_replicates": args.noise_bank_replicates,
        "log_component_gradients": args.log_component_gradients,
        "gradient_balance": args.gradient_balance,
        "frozen_check": args.frozen_check,
        "source_self_check": args.source_self_check,
        "optimizer": "Adam",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8,
        "gradient_clip_norm": 1.0,
        "batch_sampler": "seeded_balanced_shuffle",
        "source_sampling_seed": derive_seed(args.seed, "source-bank-trial-selection"),
        "alignment_batch_seed": derive_seed(args.seed, "alignment-batch-order"),
        "diffusion_beta_min": noise_config["beta_min"],
        "diffusion_beta_max": noise_config["beta_max"],
        "diffusion_T": noise_config["T"],
        "diffusion_t_min": noise_config["t_min"],
    }
    if args.gradient_balance == "source_calibrated":
        runtime_config["gradient_calibration_batches"] = args.gradient_calibration_batches
        runtime_config["gradient_calibration_seed"] = args.gradient_calibration_seed
    source_data_fingerprint = checkpoint.get("source_data_fingerprint")
    if source_data_fingerprint is not None:
        from checkpoint_provenance import verify_data_fingerprint
        verify_data_fingerprint(args.source_data_path, source_data_fingerprint, "source data")
    data_fingerprints = {
        "source": file_fingerprint(args.source_data_path),
        "target": file_fingerprint(args.target_data_path),
    }
    if reference is not None:
        resolved_reference = validate_resolved_config(
            reference,
            {**runtime_config, "factor_overrides": factor_overrides},
        )
        validate_reference_compatibility(
            reference,
            source_checkpoint_sha256=source_fingerprint,
            data_fingerprints=data_fingerprints,
            split_indices=target_data_metadata["split_indices"],
            source_split_indices=source_data_metadata["split_indices"],
            preprocessing=target_data_metadata["preprocessing"],
        )
        print(f"Reference compatibility validated: {reference.get('reference_id', 'unnamed')}")
    else:
        resolved_reference = None

    vae = SequentialVAE(
        n_neurons=n_src_neurons,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        behavior_dim=2,
    ).to(device)
    vae.load_state_dict(checkpoint["vae_state_dict"])
    source_vae_for_calibration = copy.deepcopy(vae) if args.gradient_balance == "source_calibrated" else None

    diffusion_model = LatentDiffusionModel(
        seq_len=args.seq_len,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_blocks=4,
        n_heads=4,
    ).to(device)
    diffusion_model.load_state_dict(checkpoint["dm_state_dict"])
    diffusion_guard = FrozenDiffusionGuard(diffusion_model)
    freeze_failure_path = os.path.join(args.save_dir, "phase2_freeze_failure.json")
    diffusion_guard.verify_or_record_failure(
        failure_path=freeze_failure_path,
        context={"stage": "after_phase1_load", "source_checkpoint": os.path.realpath(args.source_checkpoint)},
    )

    # 3. Precompute Source Latent Bank
    print("Pre-computing source domain latent bank...")
    source_latents_bank = extract_source_latent_bank(vae, clean_source_loader, device)
    scheduler = VPSDEScheduler(
        beta_min=noise_config["beta_min"], beta_max=noise_config["beta_max"],
        T=noise_config["T"], t_min=noise_config["t_min"],
    )
    noise_bank_path = args.noise_bank_path or os.path.join(args.save_dir, "alignment_noise_bank.pt")
    noise_bank = None
    source_noise_domain = f"source:{source_fingerprint}"
    target_noise_domain = f"target:{data_fingerprints['target']['sha256']}"
    if args.alignment_noise_mode == "fixed_bank":
        if os.path.isfile(noise_bank_path):
            noise_bank = AlignmentNoiseBank.load(
                noise_bank_path, seed=args.noise_seed, replicates=args.noise_bank_replicates
            )
        else:
            noise_bank = AlignmentNoiseBank(args.noise_seed, args.noise_bank_replicates)
        noise_bank.register_domain(
            source_noise_domain, source_latents_bank.trial_ids, args.seq_len, latent_dim
        )
        noise_bank.register_domain(
            target_noise_domain, optimization_dataset.trial_ids, args.seq_len, latent_dim
        )
    source_rng = source_sampling_generator(args.seed)
    gradient_calibration = None
    ot_scale = 1.0
    gradient_calibration_path = args.gradient_calibration_path or os.path.join(
        args.save_dir, "source_gradient_calibration.json"
    )
    if args.gradient_balance == "source_calibrated":
        sample_count = min(len(clean_source_loader.dataset),
                           args.gradient_calibration_batches * args.batch_size)
        sample_ids = clean_source_loader.dataset.trial_ids[:sample_count].tolist()
        fixture_recipe = {
            "source_checkpoint_sha256": source_fingerprint,
            "source_data_sha256": data_fingerprints["source"]["sha256"],
            "sample_ids": sample_ids,
            "seed": args.gradient_calibration_seed,
            "schedule": {key: noise_config[key] for key in ("beta_min", "beta_max", "T", "t_min")},
            "alpha": args.alpha,
            "ot_epsilon": args.ot_epsilon,
            "sinkhorn_iters": args.sinkhorn_iters,
            "mla_variance_weight": args.mla_variance_weight,
            "mla_variance_floor": args.mla_variance_floor,
            "mla_variance_scope": args.mla_variance_scope,
            "batch_size": args.batch_size,
            "streams": ["target_eta", "source_eta", "time", "diffusion_eps"],
        }
        fixture_id = hashlib.sha256(
            json.dumps(fixture_recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        calibration_provenance = {
            "source_checkpoint_sha256": source_fingerprint,
            "source_data_sha256": data_fingerprints["source"]["sha256"],
            "alpha": args.alpha,
            "ot_epsilon": args.ot_epsilon,
            "sinkhorn_iters": args.sinkhorn_iters,
            "diffusion_schedule": fixture_recipe["schedule"],
            "sample_ids": sample_ids,
            "noise_bank_identity": f"independent_fixed_source_fixture:{fixture_id}",
            "calibration_seed": args.gradient_calibration_seed,
            "probe_batches": args.gradient_calibration_batches,
            "batch_size": args.batch_size,
            "fixture_streams": fixture_recipe["streams"],
            "mla_variance_weight": args.mla_variance_weight,
            "mla_variance_floor": args.mla_variance_floor,
            "mla_variance_scope": args.mla_variance_scope,
        }
        if os.path.isfile(gradient_calibration_path):
            gradient_calibration = load_gradient_calibration(
                gradient_calibration_path, calibration_provenance
            )
        else:
            gradient_calibration = build_source_gradient_calibration(
                source_vae_for_calibration, diffusion_model, scheduler, clean_source_loader,
                source_latents_bank, device, alpha=args.alpha, ot_epsilon=args.ot_epsilon,
                sinkhorn_iters=args.sinkhorn_iters, probe_batches=args.gradient_calibration_batches,
                seed=args.gradient_calibration_seed, provenance=calibration_provenance,
                mla_variance_weight=args.mla_variance_weight,
                mla_variance_floor=args.mla_variance_floor,
                mla_variance_scope=args.mla_variance_scope,
            )
            save_gradient_calibration(gradient_calibration_path, gradient_calibration)
            if not gradient_calibration["valid"]:
                raise ValueError(
                    "Source-only gradient calibration is invalid: "
                    f"{gradient_calibration.get('reason')}"
                )
        ot_scale = float(gradient_calibration["c_ot"])
        print(
            f"Source-calibrated gradient factors: c_MLA={gradient_calibration['c_mla']:.6g} | "
            f"c_OT={ot_scale:.6g} | source-only probes={len(gradient_calibration['provenance']['sample_ids'])}"
        )
    source_variance = None
    variance_pool_spikes = None

    # 4. Initialize Target Read-In (handles both equal and mismatched neuron counts)
    raw_width_read_in = initialize_target_read_in(
        source_read_in=vae.encoder.read_in,
        src_loader=clean_source_loader,
        tgt_loader=clean_target_loader,
        n_src_neurons=n_src_neurons,
        n_tgt_neurons=n_tgt_neurons,
        hidden_dim=hidden_dim,
        device=device,
        init_mode=args.readin_init,
    )
    warm_read_in = (
        input_mapping.map_readin(raw_width_read_in).to(device)
        if input_mapping is not None else raw_width_read_in
    )
    vae.encoder.read_in = warm_read_in
    if input_mapping is not None:
        mapped_datasets = {id(tgt_train_loader.dataset): tgt_train_loader.dataset,
                           id(clean_target_loader.dataset): clean_target_loader.dataset}
        for dataset in mapped_datasets.values():
            dataset.spikes = input_mapping.apply(dataset.spikes)
        print(
            f"Applied target input mapping after spike preprocessing: "
            f"{n_tgt_neurons} raw -> {model_input_width} model channels; "
            f"padding={input_mapping.padding}, selected={input_mapping.kept_indices.numel()}"
        )

    if args.mla_variance_weight > 0 and args.alpha < 1.0:
        from mla_stabilization import source_trial_variance
        source_variance = source_trial_variance(source_latents_bank.mu)
        variance_trials = len(clean_target_loader.dataset)
        if args.mla_variance_scope == "batch":
            variance_trials = len(optimization_dataset) // len(alignment_loader)
        if variance_trials < 2:
            raise ValueError("Variance stabilization needs at least two trials in each variance estimate.")
        if args.mla_variance_scope == "full_train":
            variance_pool_spikes = clean_target_loader.dataset.spikes.to(device)
        print(f"Variance penalty scope: {args.mla_variance_scope} | trials per estimate >= {variance_trials}")
    vae.freeze_for_alignment()

    alignment_optimizer = optim.Adam(
        vae.encoder.read_in.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    diffusion_guard.verify_or_record_failure(
        (alignment_optimizer,), failure_path=freeze_failure_path,
        context={"stage": "before_phase2_optimization", "target_session": os.path.realpath(args.target_data_path)},
    )
    # Validate sampling controls without drawing noise or changing RNG state.
    if args.mla_fixed_t is not None:
        import math
        if args.mla_t_min is not None or args.mla_t_max is not None:
            raise ValueError("Choose --mla_fixed_t or a time range, not both.")
        if not math.isfinite(args.mla_fixed_t) or not scheduler.t_min <= args.mla_fixed_t <= scheduler.T:
            raise ValueError("Fixed MLA time must lie within the source training interval.")
        print(f"MLA training noise: fixed t={args.mla_fixed_t:g}")
    else:
        import math
        low = scheduler.t_min if args.mla_t_min is None else args.mla_t_min
        high = scheduler.T if args.mla_t_max is None else args.mla_t_max
        if not math.isfinite(low) or not math.isfinite(high) or not scheduler.t_min <= low < high <= scheduler.T:
            raise ValueError("MLA time range must lie within the source training interval.")
        print(f"MLA training noise: uniform t in [{low:g}, {high:g}]")

    # Save the initialization for a paired, pre-alignment target baseline.
    unaligned_read_in_state = {k: v.clone() for k, v in vae.encoder.read_in.state_dict().items()}
    source_self_history = []
    completed_updates = 0
    sample_exposures = 0
    full_variance_forwards = 0
    full_variance_sample_exposures = 0
    if args.source_self_check:
        source_r2, source_rmse = evaluate_phase1(vae, src_validation_loader, device)
        source_self_history.append({"epoch": 0, "r2": source_r2, "normalized_rmse": source_rmse})
        print(f"Source self-check validation baseline: R2={source_r2:.2f}% | normalized RMSE={source_rmse:.4f}")

    save_path = os.path.join(args.save_dir, "phase2_aligned_target_model.pt")
    diagnostics = None
    component_gradient_history = []
    if args.mla_diagnostics:
        import math
        from mla_diagnostics import MLADiagnostics
        if not args.mla_probe_times or any(
            not math.isfinite(t) or not scheduler.t_min <= t <= scheduler.T for t in args.mla_probe_times
        ):
            raise ValueError("Probe times must lie within the source diffusion training interval.")
        diagnostics = MLADiagnostics(
            clean_target_loader.dataset, checkpoint["split_indices"]["train"], latent_dim,
            args.mla_probe_times, args.mla_probe_seed, args.mla_probe_trials, args.save_dir,
            context={
                "source_checkpoint": os.path.realpath(args.source_checkpoint),
                "source_epoch": checkpoint.get("epoch"), "seed": args.seed, "split_seed": args.split_seed,
                "alpha": args.alpha, "mla_t_min": args.mla_t_min,
                "mla_t_max": args.mla_t_max, "mla_fixed_t": args.mla_fixed_t,
                "mla_variance_weight": args.mla_variance_weight,
                "mla_variance_floor": args.mla_variance_floor,
                "mla_variance_scope": args.mla_variance_scope,
            },
        )
        diagnostics.record(0, vae, diffusion_model, scheduler, device, args.batch_size)
        if args.mla_diagnostics_only:
            print("MLA diagnostics complete; no optimizer updates or aligned checkpoint were produced.")
            return

    print("\nStarting Maximum Likelihood Alignment (Phase 2)...")
    for epoch in range(1, args.epochs + 1):
        align_metrics = train_epoch_phase2(
            vae=vae,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            target_loader=alignment_loader,
            source_latents_bank=source_latents_bank,
            optimizer=alignment_optimizer,
            device=device,
            alpha=args.alpha,
            ot_epsilon=args.ot_epsilon,
            sinkhorn_iters=args.sinkhorn_iters,
            smooth_weight=args.smooth_weight,
            mla_t_min=args.mla_t_min,
            mla_t_max=args.mla_t_max,
            mla_fixed_t=args.mla_fixed_t,
            source_variance=source_variance,
            mla_variance_weight=args.mla_variance_weight,
            mla_variance_floor=args.mla_variance_floor,
            mla_variance_scope=args.mla_variance_scope,
            variance_pool_spikes=variance_pool_spikes,
            noise_bank=noise_bank,
            noise_replicate=(epoch - 1) % args.noise_bank_replicates,
            gradient_balance=args.gradient_balance,
            ot_scale=ot_scale,
            log_component_gradients=args.log_component_gradients,
            source_sampling_rng=source_rng,
            target_noise_domain=target_noise_domain,
            source_noise_domain=source_noise_domain,
        )
        if args.frozen_check == "every_epoch":
            diffusion_guard.verify_or_record_failure(
                (alignment_optimizer,), failure_path=freeze_failure_path,
                context={"stage": "epoch_end", "epoch": epoch,
                         "source_checkpoint": os.path.realpath(args.source_checkpoint),
                         "target_session": os.path.realpath(args.target_data_path)},
            )
        completed_updates += align_metrics["updates"]
        sample_exposures += align_metrics["sample_exposures"]
        full_variance_forwards += align_metrics["full_variance_forwards"]
        full_variance_sample_exposures += align_metrics["full_variance_sample_exposures"]
        if args.log_component_gradients:
            component_gradient_history.append({
                "epoch": epoch, "components": align_metrics["component_gradient_report"]["components"],
                "cosines": align_metrics["component_gradient_report"]["cosines"],
            })

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"Alignment Loss: {align_metrics['total']:.4f} | "
                f"MLA: {align_metrics['mla']:.4f} | OT: {align_metrics['ot']:.4f} | "
                f"Smooth: {align_metrics['smooth']:.4f} | "
                f"Var floor: {align_metrics['variance_floor']:.4f} "
                f"(weighted {align_metrics['variance_contribution']:.4f}) | "
                f"effective coefficients: MLA={1.0 - args.alpha:.4g}, "
                f"OT={args.alpha * ot_scale:.4g} | "
                f"Read-in grad: {align_metrics['grad_norm']:.4f} | "
                f"Clipped: {align_metrics['clipped_steps']}/{align_metrics['updates']}"
            )
            if args.log_component_gradients and align_metrics["component_gradient_report"] is not None:
                components = align_metrics["component_gradient_report"]["components"]
                pairs = align_metrics["component_gradient_report"]["cosines"]
                component_text = ", ".join(
                    f"{name}={values['norm']:.4g}/{values['weighted_norm']:.4g}"
                    for name, values in components.items()
                )
                cosine_text = ", ".join(f"{name}={value:+.3f}" for name, value in pairs.items())
                print(f"  Component gradient norms raw/weighted: {component_text}; cosines: {cosine_text}")
            if args.source_self_check:
                source_r2, source_rmse = evaluate_phase1(vae, src_validation_loader, device)
                source_self_history.append({"epoch": epoch, "r2": source_r2, "normalized_rmse": source_rmse})
                print(f"Source self-check validation: R2={source_r2:.2f}% | normalized RMSE={source_rmse:.4f}")

        if diagnostics is not None and (
            epoch in (1, 5) or epoch % args.mla_diagnostic_every == 0 or epoch == args.epochs
        ):
            diagnostics.record(epoch, vae, diffusion_model, scheduler, device, args.batch_size)

    diffusion_guard.verify_or_record_failure(
        (alignment_optimizer,), failure_path=freeze_failure_path,
        context={"stage": "final", "epoch": args.epochs,
                 "source_checkpoint": os.path.realpath(args.source_checkpoint),
                 "target_session": os.path.realpath(args.target_data_path)},
    )
    unresolved = []
    if resolved_reference is None:
        unresolved.append("reference_run_not_supplied")
    if checkpoint.get("code_revision") is None:
        unresolved.append("phase1_code_revision_missing")
    if checkpoint.get("training_config") is None:
        unresolved.append("phase1_training_config_missing")
    if checkpoint.get("training_seed") is None:
        unresolved.append("phase1_training_seed_missing")
    if checkpoint.get("source_data_fingerprint") is None:
        unresolved.append("phase1_source_data_fingerprint_missing")
    if code_revision(os.path.dirname(os.path.abspath(__file__))) is None:
        unresolved.append("phase2_code_revision_unavailable")
    if reference is not None:
        unresolved.extend(reference.get("unresolved", []))
        reference_metadata_fields = (
            "source_checkpoint_sha256", "data_fingerprints", "source_split_indices",
            "preprocessing", "code_revision",
        )
        for field in reference_metadata_fields:
            if field not in reference or reference[field] is None:
                unresolved.append(f"reference_missing_{field}")
        if not reference.get("target_split_indices", reference.get("split_indices")):
            unresolved.append("reference_missing_target_split_indices")
    recorded_config = {}
    if resolved_reference is not None:
        recorded_config.update(resolved_reference["resolved_config"])
    recorded_config.update(runtime_config)
    noise_bank_hash = None
    if noise_bank is not None:
        noise_bank_hash = noise_bank.save(noise_bank_path)
    manifest = build_run_manifest(
        config=recorded_config,
        source_checkpoint=args.source_checkpoint,
        split_indices=target_data_metadata["split_indices"],
        source_split_indices=source_data_metadata["split_indices"],
        data_fingerprints=data_fingerprints,
        code_revision=code_revision(os.path.dirname(os.path.abspath(__file__))),
        updates=completed_updates,
        sample_exposures=sample_exposures,
        full_variance_forwards=full_variance_forwards,
        full_variance_sample_exposures=full_variance_sample_exposures,
        requested_epochs=args.epochs,
        completed_epochs=args.epochs,
        unresolved=unresolved,
        preprocessing=target_data_metadata["preprocessing"],
        reference_id=None if reference is None else reference.get("reference_id"),
    )
    manifest["requested_updates"] = args.epochs * len(alignment_loader)
    manifest["completed_updates"] = completed_updates
    manifest["factor_overrides"] = factor_overrides
    manifest["allowed_factor_overrides"] = ([] if reference is None else
                                            reference.get("allowed_factor_overrides", []))
    manifest["source_training_config"] = checkpoint.get("training_config")
    manifest["source_training_seed"] = checkpoint.get("training_seed")
    manifest["source_selected_epoch"] = checkpoint.get("epoch")
    mapping_state = None if input_mapping is None else input_mapping.state_dict()
    manifest["input_mapping"] = (
        None if mapping_state is None else {
            "raw_width": mapping_state["raw_width"],
            "model_width": mapping_state["model_width"],
            "kept_indices": mapping_state["kept_indices"].tolist(),
            "padding": mapping_state["padding"],
        }
    )
    manifest["input_mapping_lossy"] = bool(
        input_mapping is not None and input_mapping.raw_width > input_mapping.model_width
    )
    manifest["input_mapping_note"] = (
        "wider-target channel selection discards information"
        if manifest["input_mapping_lossy"] else None
    )
    manifest["alignment_noise_bank_path"] = (
        os.path.realpath(noise_bank_path) if noise_bank is not None else None
    )
    manifest["alignment_noise_bank_sha256"] = noise_bank_hash
    manifest["gradient_balance"] = args.gradient_balance
    manifest["gradient_calibration_path"] = (
        os.path.realpath(gradient_calibration_path)
        if gradient_calibration is not None else None
    )
    manifest["gradient_calibration"] = gradient_calibration
    manifest["gradient_calibration_factor"] = ot_scale
    manifest["gradient_calibration_variant_note"] = (
        "source-only gradient scaling changes the optimization variant and cannot repair opposed gradients"
        if args.gradient_balance == "source_calibrated" else None
    )
    manifest["finite_bank_is_training_variant"] = noise_bank is not None
    manifest["probe_noise_namespace"] = "independent_mla_diagnostics_fixture"
    manifest["exact_reproduction_claimable"] = exact_reproduction_claimable(manifest)
    manifest_path = os.path.join(args.save_dir, "phase2_run_manifest.json")
    save_run_manifest(manifest_path, manifest)
    torch.save(
        {
            "unaligned_read_in_state": unaligned_read_in_state,
            "target_read_in_state": {k: v.clone() for k, v in vae.encoder.read_in.state_dict().items()},
            "target_n_neurons": n_tgt_neurons,
            "model_input_width": model_input_width,
            "match_input_size": args.match_input_size,
            "input_mapping": mapping_state,
            "input_mapping_seed": args.input_mapping_seed,
            "input_mapping_lossy": manifest["input_mapping_lossy"],
            "velocity_mean": checkpoint_velocity_stats["velocity_mean"],
            "velocity_std": checkpoint_velocity_stats["velocity_std"],
            "alignment_epochs": args.epochs,
            "diffusion_frozen_verified": True,
            "source_trial_variance": None if source_variance is None else source_variance.cpu(),
            "gradient_balance": args.gradient_balance,
            "gradient_calibration": gradient_calibration,
            "gradient_calibration_path": manifest["gradient_calibration_path"],
            "gradient_calibration_factor": ot_scale,
            "component_gradient_history": component_gradient_history,
            "variance_pool_trials": len(clean_target_loader.dataset),
            "source_checkpoint_sha256": source_fingerprint,
            "source_checkpoint_path": os.path.realpath(args.source_checkpoint),
            "source_epoch": checkpoint.get("epoch"),
            "source_validation_r2": checkpoint.get("best_r2"),
            "source_self_check": args.source_self_check,
            "source_self_history": source_self_history,
            "diffusion_training": noise_config,
            "alignment_updates": completed_updates,
            "sample_exposures": sample_exposures,
            "full_variance_forwards": full_variance_forwards,
            "full_variance_sample_exposures": full_variance_sample_exposures,
            "alignment_noise_bank_path": manifest["alignment_noise_bank_path"],
            "alignment_noise_bank_sha256": noise_bank_hash,
            "source_sampling_rng_state": source_rng.get_state(),
            "alignment_batch_order_rng_state": alignment_loader.batch_sampler.generator.get_state(),
            "alignment_loader_rng_state": alignment_loader.generator.get_state(),
            "source_data_fingerprint": data_fingerprints["source"],
            "target_data_fingerprint": data_fingerprints["target"],
            "preprocessing": target_data_metadata["preprocessing"],
            "code_revision": manifest["code_revision"],
            "training_seed": args.seed,
            "run_manifest": manifest,
            "run_manifest_path": os.path.realpath(manifest_path),
            "split_indices": target_data_metadata["split_indices"],
            "split_seed": target_data_metadata["split_seed"],
            "alignment_config": {
                "alpha": args.alpha,
                "ot_epsilon": args.ot_epsilon,
                "sinkhorn_iters": args.sinkhorn_iters,
                "smooth_weight": args.smooth_weight,
                "mla_t_min": args.mla_t_min,
                "mla_t_max": args.mla_t_max,
                "mla_fixed_t": args.mla_fixed_t,
                "mla_variance_weight": args.mla_variance_weight,
                "mla_variance_floor": args.mla_variance_floor,
                "mla_variance_scope": args.mla_variance_scope,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "augment_mult": args.augment_mult,
                "augment_target": args.augment_target,
                "readin_init": args.readin_init,
                "match_input_size": args.match_input_size,
                "input_mapping_seed": args.input_mapping_seed,
                "model_input_width": model_input_width,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "batch_sampler": "seeded_balanced_shuffle",
                "optimizer": "Adam",
                "alignment_noise_mode": args.alignment_noise_mode,
                "noise_seed": args.noise_seed,
                "noise_bank_replicates": args.noise_bank_replicates,
                "log_component_gradients": args.log_component_gradients,
                "gradient_balance": args.gradient_balance,
                "gradient_calibration_batches": args.gradient_calibration_batches,
                "gradient_calibration_seed": args.gradient_calibration_seed,
                "gradient_calibration_path": manifest["gradient_calibration_path"],
                "gradient_calibration_factor": ot_scale,
            },
        },
        save_path,
    )

    print(f"\nAlignment Complete after {args.epochs} fixed epochs.")
    print(f"Saved aligned target checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
