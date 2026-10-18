"""

Training script for phase 1

train all the four components of the model (encoder, decoder, regressor, and regressor) in a single training loop.
"""

import argparse
import os
import torch
import torch.optim as optim

from dataloader import get_nhp_dataloaders
from vae import SequentialVAE, source_domain_loss
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
    kl_weight: float = 1.0,
    behavior_weight: float = 1.0,
):
    vae.train()
    diffusion_model.train()

    total_vae_loss = 0.0
    total_dsm_loss = 0.0

    for batch in loader:
        spikes = batch["spikes"].to(device)       # (batch, seq_len, n_neurons)
        velocity = batch["velocity"].to(device)   # (batch, seq_len, 2)

        # -------------------------------------------------------------
        # Step 1: Forward pass VAE & Update VAE + Behavior Regressor
        # -------------------------------------------------------------
        vae_optimizer.zero_grad()

        recon_rate, pred_vel, mu, logvar, z = vae(spikes)

        loss_vae = source_domain_loss(
            recon_rate=recon_rate,
            spikes=spikes,
            pred_vel=pred_vel,
            true_vel=velocity,
            mu=mu,
            logvar=logvar,
            kl_weight=kl_weight,
            behavior_weight=behavior_weight,
        )

        loss_vae.backward()
        vae_optimizer.step()

        # -------------------------------------------------------------
        # Step 2: Cooperative Training of Diffusion Model (Algorithm 1)
        # Using current latents Z_0 as dynamic input to DSM
        # -------------------------------------------------------------
        dm_optimizer.zero_grad()

        # Detach latent trajectory Z_0 so DM gradients don't retroactively alter VAE
        z_0 = z.detach()

        loss_dm = dsm_loss(diffusion_model, scheduler, z_0)
        loss_dm.backward()
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

        _, pred_vel, _, _, _ = vae(spikes)
        all_preds.append(pred_vel.cpu())
        all_targets.append(velocity.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    r2 = compute_r2_score(all_targets, all_preds)
    rmse = compute_rmse(all_targets, all_preds)
    return r2, rmse


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Source Domain Training (ERDiff)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to source domain .mat file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=39, help="Number of time bins per trial")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dynamics dimensionality")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Internal hidden state dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (0.005 or 0.013 in paper)")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="ELBO KL loss weight")
    parser.add_argument("--behavior_weight", type=float, default=1.0, help="Velocity decoding loss weight")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Source Data
    train_loader, val_loader, n_neurons = get_nhp_dataloaders(
        file_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    print(f"Loaded source domain with {n_neurons} neurons across trials of length {args.seq_len}.")

    # 2. Instantiate Models
    vae = SequentialVAE(
        n_neurons=n_neurons,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        behavior_dim=2,
    ).to(device)

    diffusion_model = LatentDiffusionModel(
        seq_len=args.seq_len,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=4,
        n_heads=4,
    ).to(device)

    scheduler = VPSDEScheduler()

    # 3. Optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    dm_optimizer = optim.Adam(diffusion_model.parameters(), lr=args.lr)

    # 4. Cooperative Training Loop (Algorithm 1)
    best_val_r2 = -float("inf")
    save_path = os.path.join(args.save_dir, "phase1_source_models.pt")

    print("Starting Cooperative Source Domain Training (Phase 1)...")
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

        if epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"VAE Loss: {vae_loss:.4f} | DSM Loss: {dm_loss:.4f} | "
                f"Val R^2: {val_r2:.2f}% | Val RMSE: {val_rmse:.4f}"
            )

        # Save checkpoint based on validation behavior decoding quality
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

    print(f"Phase 1 complete. Best source validation R^2: {best_val_r2:.2f}%. Model saved to {save_path}.")


if __name__ == "__main__":
    main()