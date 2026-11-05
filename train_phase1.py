import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from checkpoint_provenance import code_revision, file_fingerprint

from dataloader import NHPDataset, denormalize_velocity, get_nhp_dataloaders
from vae import SequentialVAE, poisson_nll_loss, kl_divergence_loss, trajectory_smoothness_loss, sample_latent_posterior
from diffusion import LatentDiffusionModel, VPSDEScheduler, dsm_loss
from decoder_models import compute_r2_score, compute_rmse

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

def train_epoch_phase1(
    vae: SequentialVAE,
    diffusion_model: LatentDiffusionModel,
    scheduler: VPSDEScheduler,
    loader,
    vae_optimizer: optim.Optimizer,
    dm_optimizer: optim.Optimizer,
    device: torch.device,
    kl_weight: float = 0.05,
    behavior_weight: float = 2.0,
):
    vae.train()
    diffusion_model.train()

    total_vae_loss = 0.0
    total_dsm_loss = 0.0

    for batch in loader:
        spikes = batch["spikes"].to(device)
        velocity = batch["velocity"].to(device)

        # -------------------------------------------------------------
        # 1. Update VAE & Ridge Kinematics Decoder
        # -------------------------------------------------------------
        # 1. Forward pass through VAE (uses stochastic z during train, deterministic mu during eval)
        vae_optimizer.zero_grad()
        recon_rate, pred_vel, mu, logvar, z = vae(spikes)

        loss_rec = poisson_nll_loss(recon_rate, spikes)
        loss_kl = kl_divergence_loss(mu, logvar)
        loss_smooth = trajectory_smoothness_loss(mu)
        loss_mse = F.mse_loss(pred_vel, velocity) + vae.behavior_decoder.ridge_penalty()

        # Combined loss with continuous trajectory smoothness
        loss_vae = (
            loss_rec
            + kl_weight * loss_kl
            + 0.5 * loss_smooth
            + behavior_weight * loss_mse
        )
        loss_vae.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        vae_optimizer.step()

        # -------------------------------------------------------------
        # 2. Cooperative Diffusion Training (Algorithm 1)
        # -------------------------------------------------------------
        dm_optimizer.zero_grad()

        # Algorithm 1 infers a fresh posterior sample after the VAE update.
        # Match Phase 2's frozen encoder/dropout mode while retaining q(z|x)
        # variance; eval mode alone would otherwise return only the mean.
        with torch.no_grad():
            vae.encoder.eval()
            mu_density, logvar_density = vae.encoder(spikes)
            z_0 = sample_latent_posterior(mu_density, logvar_density)
            vae.encoder.train()
        loss_dm = dsm_loss(diffusion_model, scheduler, z_0)
        loss_dm.backward()
        torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
        dm_optimizer.step()

        total_vae_loss += loss_vae.item()
        total_dsm_loss += loss_dm.item()

    avg_vae = total_vae_loss / len(loader)
    avg_dm = total_dsm_loss / len(loader)
    return avg_vae, avg_dm


@torch.no_grad()
def evaluate_phase1(vae: SequentialVAE, loader, device: torch.device):
    vae.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        spikes = batch["spikes"].to(device)
        velocity = batch["velocity"].to(device)

        mu, _ = vae.encoder(spikes)
        pred_vel = vae.behavior_decoder(mu)

        all_preds.append(pred_vel.cpu())
        all_targets.append(velocity.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    r2 = compute_r2_score(all_targets, all_preds)
    rmse = compute_rmse(all_targets, all_preds)
    return r2, rmse


def inspect_source_timebase(file_path: str, seq_len: int):
    """Report stored sample spacing without assuming that one sample is 20 ms."""
    import mat73
    import numpy as np

    xds = mat73.loadmat(file_path)["xds"]
    times = np.asarray(xds["time_frame"], dtype=np.float64).ravel()
    deltas = np.diff(times)
    valid_deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if valid_deltas.size == 0:
        print("Time-frame spacing could not be determined from positive finite differences.")
        return

    median_dt = float(np.median(valid_deltas))
    print(
        f"Stored time_frame spacing: median={median_dt:.8g}, "
        f"5th/95th percentiles={np.percentile(valid_deltas, [5, 95]).tolist()} "
        f"(native time units); {seq_len} raw samples span approximately "
        f"{(seq_len - 1) * median_dt:.8g} native time units."
    )
    print(
        f"Raw signal shapes: spike_counts={np.shape(xds['spike_counts'])}, "
        f"curs_v={np.shape(xds['curs_v'])}; target_dir present={xds.get('target_dir') is not None}."
    )
    go_cues = np.asarray(xds.get("trial_gocue_time", xds.get("trial_start_time")), dtype=np.float64).ravel()
    if np.all(np.isnan(go_cues)):
        go_cues = np.asarray(xds["trial_start_time"], dtype=np.float64).ravel()
    if "trial_end_time" in xds:
        trial_ends = np.asarray(xds["trial_end_time"], dtype=np.float64).ravel()
        n_trials = min(len(go_cues), len(trial_ends))
        durations = trial_ends[:n_trials] - go_cues[:n_trials]
        durations = durations[np.isfinite(durations) & (durations > 0)]
        if durations.size:
            print(f"Go-cue to trial-end duration: median={np.median(durations):.8g} native time units.")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Source Domain Training (ERDiff)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed used only to define trial partitions")
    parser.add_argument("--data_path", type=str, required=True, help="Path to source session .mat file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=39, help="Number of time bins per trial")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dynamics dimensionality")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden state dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="L2 weight decay regularization")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--augment_mult", type=int, default=5, help="Data augmentation replication factor")
    parser.add_argument("--spike_noise_factor", type=float, default=0.0,
                        help="Optional extra spike-noise variant; zero keeps only posterior/diffusion noise")
    parser.add_argument("--kl_weight", type=float, default=0.05, help="ELBO KL loss weight")
    parser.add_argument("--behavior_weight", type=float, default=2.0, help="Velocity MSE loss weight")
    parser.add_argument("--inspect_data_only", action="store_true", help="Report source trial and velocity-window diagnostics without training")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data with Augmentation
    train_loader, val_loader, _, n_neurons, velocity_stats = get_nhp_dataloaders(
        file_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        augment_multiplier=args.augment_mult,
        split_seed=args.split_seed,
        train_noise_factor=args.spike_noise_factor,
    )
    print(
        f"Loaded source domain with {n_neurons} neurons across trials of "
        f"{args.seq_len} bins at {velocity_stats['preprocessing']['bin_width_s'] * 1000:g} ms/bin."
    )

    # Use the same clean, deterministic evaluation path for both partitions.
    # The optimization loader repeats trials and perturbs spikes, so its batch
    # R² cannot be compared directly with held-out validation R².
    train_data = train_loader.dataset
    clean_train_loader = DataLoader(
        NHPDataset(
            train_data.spikes, train_data.velocities, train_data.targets,
            is_train=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    train_idx = velocity_stats["split_indices"]["train"]
    val_idx = velocity_stats["split_indices"]["validation"]
    test_idx = velocity_stats["split_indices"]["test"]
    val_velocity = val_loader.dataset.velocities
    source_mean_baseline = train_data.velocities.mean(dim=(0, 1), keepdim=True)
    baseline_r2 = compute_r2_score(val_velocity, source_mean_baseline.expand_as(val_velocity))
    print(
        f"Source trials: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} | "
        f"validation source-mean R²={baseline_r2:.2f}%"
    )
    print(
        "Source direction counts (train/val, classes 0–7): "
        f"{torch.bincount(train_data.targets, minlength=8).tolist()} / "
        f"{torch.bincount(val_loader.dataset.targets, minlength=8).tolist()}"
    )
    if torch.count_nonzero(train_data.targets) == 0 and torch.count_nonzero(val_loader.dataset.targets) == 0:
        print("All encoded directions are zero; this also occurs when target_dir is absent in the source file.")

    for split_name, split_data in (("train", train_data), ("val", val_loader.dataset)):
        physical_velocity = denormalize_velocity(split_data.velocities, velocity_stats)
        speeds = torch.linalg.vector_norm(physical_velocity, dim=-1)
        peak_bins = speeds.argmax(dim=1)
        edge_bins = min(5, args.seq_len // 2)
        edge_fraction = ((peak_bins < edge_bins) | (peak_bins >= args.seq_len - edge_bins)).float().mean()
        speed_profile = speeds.mean(dim=0)
        print(
            f"{split_name} velocity peaks: median bin={peak_bins.median().item()}, "
            f"edge fraction={edge_fraction.item():.2f}; mean speed by bin="
            f"{[round(value, 3) for value in speed_profile.tolist()]}"
        )

    if args.inspect_data_only:
        inspect_source_timebase(args.data_path, args.seq_len)
        print("Source data inspection complete; no model was trained.")
        return

    # 2. Instantiate Models
    vae = SequentialVAE(
        n_neurons=n_neurons,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        behavior_dim=2,
        dropout=0.4,
    ).to(device)

    diffusion_model = LatentDiffusionModel(
        seq_len=args.seq_len,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=4,
        n_heads=4,
    ).to(device)

    scheduler = VPSDEScheduler()

    # 3. Optimizers with L2 Weight Decay
    vae_optimizer = optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dm_optimizer = optim.AdamW(diffusion_model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Cooperative Training Loop
    best_val_r2 = -float("inf")
    best_epoch = None
    save_path = os.path.join(args.save_dir, "phase1_source_models.pt")
    source_code_revision = code_revision(os.path.dirname(os.path.abspath(__file__)))
    source_data_provenance = file_fingerprint(args.data_path)

    print("\nStarting Cooperative Source Domain Training (Phase 1)...")
    for epoch in range(1, args.epochs + 1):
        vae_loss, dm_loss = train_epoch_phase1(
            vae=vae,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            loader=train_loader,
            vae_optimizer=vae_optimizer,
            dm_optimizer=dm_optimizer,
            device=device,
            kl_weight=args.kl_weight,
            behavior_weight=args.behavior_weight,
        )

        val_r2, val_rmse = evaluate_phase1(vae, val_loader, device)

        if epoch % 10 == 0 or epoch == args.epochs:
            clean_train_r2, _ = evaluate_phase1(vae, clean_train_loader, device)
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"Train (clean) R^2: {clean_train_r2:6.2f}% | "
                f"Val R^2: {val_r2:6.2f}% | Val RMSE: {val_rmse:.4f} | "
                f"DSM Loss: {dm_loss:.4f}"
            )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "training_seed": args.seed,
                    "training_config": vars(args).copy(),
                    "code_revision": source_code_revision,
                    "source_data_fingerprint": source_data_provenance,
                    "vae_state_dict": vae.state_dict(),
                    "dm_state_dict": diffusion_model.state_dict(),
                    "n_neurons": n_neurons,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "best_r2": best_val_r2,
                    "velocity_mean": velocity_stats["velocity_mean"],
                    "velocity_std": velocity_stats["velocity_std"],
                    "split_indices": velocity_stats["split_indices"],
                    "split_seed": velocity_stats["split_seed"],
                    "preprocessing": velocity_stats["preprocessing"],
                    "diffusion_training": {
                        "latent_distribution": "posterior_sample",
                        "beta_min": scheduler.beta_min,
                        "beta_max": scheduler.beta_max,
                        "T": scheduler.T,
                        "t_min": scheduler.t_min,
                        "spike_noise_factor": args.spike_noise_factor,
                    },
                },
                save_path,
            )

    print(
        f"\nPhase 1 Complete. Best Source Val R^2: {best_val_r2:.2f}% "
        f"at epoch {best_epoch}. Saved to {save_path}."
    )


if __name__ == "__main__":
    main()
