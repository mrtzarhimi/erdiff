import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from dataloader import get_nhp_dataloaders
from vae import SequentialVAE, poisson_nll_loss, kl_divergence_loss
from diffusion import LatentDiffusionModel, VPSDEScheduler, dsm_loss
from decoder_models import compute_r2_score, compute_rmse


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
    total_r2 = 0.0

    for batch in loader:
        spikes = batch["spikes"].to(device)
        velocity = batch["velocity"].to(device)

        # -------------------------------------------------------------
        # 1. Update VAE & Ridge Kinematics Decoder
        # -------------------------------------------------------------
        vae_optimizer.zero_grad()

        mu, logvar = vae.encoder(spikes)
        z = vae.encoder.reparameterize(mu, logvar)

        # Poisson spike reconstruction
        recon_rate = vae.decoder(z)
        loss_rec = poisson_nll_loss(recon_rate, spikes)
        loss_kl = kl_divergence_loss(mu, logvar)

        # Behavioral velocity prediction + Ridge L2 regularization
        pred_vel = vae.behavior_decoder(mu)
        loss_mse = F.mse_loss(pred_vel, velocity) + vae.behavior_decoder.ridge_penalty()

        # Joint ELBO + Behavior loss
        loss_vae = loss_rec + kl_weight * loss_kl + behavior_weight * loss_mse
        loss_vae.backward()

        # Clip gradients to stabilize recurrent updates
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        vae_optimizer.step()

        # -------------------------------------------------------------
        # 2. Cooperative Diffusion Training (Algorithm 1)
        # -------------------------------------------------------------
        dm_optimizer.zero_grad()

        z_0 = mu.detach()
        loss_dm = dsm_loss(diffusion_model, scheduler, z_0)
        loss_dm.backward()
        torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
        dm_optimizer.step()

        total_vae_loss += loss_vae.item()
        total_dsm_loss += loss_dm.item()
        total_r2 += compute_r2_score(velocity, pred_vel)

    avg_vae = total_vae_loss / len(loader)
    avg_dm = total_dsm_loss / len(loader)
    avg_train_r2 = total_r2 / len(loader)
    return avg_vae, avg_dm, avg_train_r2


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


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Source Domain Training (ERDiff)")
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
    parser.add_argument("--kl_weight", type=float, default=0.05, help="ELBO KL loss weight")
    parser.add_argument("--behavior_weight", type=float, default=2.0, help="Velocity MSE loss weight")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data with Augmentation
    train_loader, val_loader, n_neurons = get_nhp_dataloaders(
        file_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        augment_multiplier=args.augment_mult,
    )
    print(f"Loaded source domain with {n_neurons} neurons across trials of length {args.seq_len}.")

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
    save_path = os.path.join(args.save_dir, "phase1_source_models.pt")

    print("\nStarting Cooperative Source Domain Training (Phase 1)...")
    for epoch in range(1, args.epochs + 1):
        vae_loss, dm_loss, train_r2 = train_epoch_phase1(
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
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"Train R^2: {train_r2:6.2f}% | "
                f"Val R^2: {val_r2:6.2f}% | Val RMSE: {val_rmse:.4f} | "
                f"DSM Loss: {dm_loss:.4f}"
            )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(
                {
                    "epoch": epoch,
                    "vae_state_dict": vae.state_dict(),
                    "dm_state_dict": diffusion_model.state_dict(),
                    "n_neurons": n_neurons,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "best_r2": best_val_r2,
                },
                save_path,
            )

    print(f"\nPhase 1 Complete. Best Source Val R^2: {best_val_r2:.2f}%. Saved to {save_path}.")


if __name__ == "__main__":
    main()
