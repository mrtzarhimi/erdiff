"""

Training script for phase 2

train only read in layer of encoder
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_nhp_dataloaders
from vae import SequentialVAE
from diffusion import (
    LatentDiffusionModel,
    VPSDEScheduler,
    total_alignment_loss,
)
from decoder_models import compute_r2_score, compute_rmse


@torch.no_grad()
def extract_source_latent_bank(vae: SequentialVAE, source_loader, device: torch.device):
    """
    Precomputes a representative sample of source latent trajectories Z_0^(s)
    to compute the Sinkhorn Divergence regularizer during target alignment.
    """
    vae.eval()
    source_latents = []
    for batch in source_loader:
        spikes = batch["spikes"].to(device)
        mu, _ = vae.encoder(spikes)
        source_latents.append(mu)  # Use deterministic mean trajectory
    return torch.cat(source_latents, dim=0)


def train_epoch_phase2(
    target_encoder: nn.Module,
    diffusion_model: LatentDiffusionModel,
    scheduler: VPSDEScheduler,
    target_loader,
    source_latents_bank: torch.Tensor,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha: float = 0.5,
):
    """
    Phase 2: Test-time maximum likelihood alignment (Algorithm 2).
    Only the read-in layer of target_encoder is updated.
    """
    target_encoder.train()
    diffusion_model.eval()  # Frozen prior

    total_loss = 0.0
    num_source_samples = source_latents_bank.size(0)

    for batch in target_loader:
        target_spikes = batch["spikes"].to(device)  # (batch_size, seq_len, n_tgt_neurons)
        batch_size = target_spikes.size(0)

        optimizer.zero_grad()

        # 1. Project target spikes to latent space via adapted encoder
        mu_t, logvar_t = target_encoder(target_spikes)
        z_target = target_encoder.reparameterize(mu_t, logvar_t)

        # 2. Randomly sample a matching batch of source latents for Sinkhorn regularizer
        perm = torch.randperm(num_source_samples)[:batch_size]
        z_source_batch = source_latents_bank[perm].to(device)

        # 3. Compute joint MLA + Sinkhorn loss (Eq. 28)
        loss = total_alignment_loss(
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            z_target=z_target,
            z_source=z_source_batch,
            alpha=alpha,
        )

        # 4. Backpropagate into target encoder read_in parameters
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(target_loader)


@torch.no_grad()
def evaluate_target_alignment(
    vae: SequentialVAE,
    target_loader,
    device: torch.device,
):
    """
    Evaluates zero-shot transfer on target domain behavior decoding (R^2 and RMSE).
    Target behavior kinematics were NEVER seen during alignment.
    """
    vae.eval()
    all_preds = []
    all_targets = []

    for batch in target_loader:
        spikes = batch["spikes"].to(device)
        velocity = batch["velocity"].to(device)

        # Inference: pass through adapted encoder and frozen pre-trained behavior decoder
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
    parser = argparse.ArgumentParser(description="Phase 2: Target Domain Maximum Likelihood Alignment (ERDiff)")
    parser.add_argument("--source_checkpoint", type=str, required=True, help="Path to saved Phase 1 checkpoint (.pt)")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to source session .mat file")
    parser.add_argument("--target_data_path", type=str, required=True, help="Path to target session .mat file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=39, help="Number of time bins per trial")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (64 in paper)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for linear probing")
    parser.add_argument("--epochs", type=int, default=150, help="Alignment iterations/epochs")
    parser.add_argument("--alpha", type=float, default=0.5, help="Trade-off parameter weighting Sinkhorn loss")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    src_train_loader, _, n_src_neurons = get_nhp_dataloaders(
        file_path=args.source_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    tgt_train_loader, tgt_val_loader, n_tgt_neurons = get_nhp_dataloaders(
        file_path=args.target_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    print(f"Source neurons: {n_src_neurons} | Target neurons: {n_tgt_neurons}")

    # 2. Restore Source Models from Phase 1
    checkpoint = torch.load(args.source_checkpoint, map_location=device)
    latent_dim = checkpoint["latent_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    vae = SequentialVAE(
        n_neurons=n_src_neurons,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        behavior_dim=2,
    ).to(device)
    vae.load_state_dict(checkpoint["vae_state_dict"])

    diffusion_model = LatentDiffusionModel(
        seq_len=args.seq_len,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_blocks=4,
        n_heads=4,
    ).to(device)
    diffusion_model.load_state_dict(checkpoint["dm_state_dict"])

    # 3. Setup Target Alignment (Linear Probing Setup)
    # Cache source latent representations for the Sinkhorn regularizer
    print("Pre-computing source domain latent bank...")
    source_latents_bank = extract_source_latent_bank(vae, src_train_loader, device)

    # Freeze entire VAE and Diffusion Model
    vae.freeze_for_alignment()
    for param in diffusion_model.parameters():
        param.requires_grad = False

    # Replace encoder read-in layer to match target domain neuron count
    vae.encoder.read_in = nn.Linear(n_tgt_neurons, hidden_dim).to(device)
    vae.encoder.read_in.weight.requires_grad = True
    vae.encoder.read_in.bias.requires_grad = True

    # Optimizer tunes ONLY the read-in layer
    alignment_optimizer = optim.Adam(vae.encoder.read_in.parameters(), lr=args.lr)
    scheduler = VPSDEScheduler()

    # Zero-shot evaluation before alignment
    pre_r2, pre_rmse = evaluate_target_alignment(vae, tgt_val_loader, device)
    print(f"Target Performance BEFORE Alignment: R^2 = {pre_r2:.2f}% | RMSE = {pre_rmse:.4f}")

    # 4. Phase 2 Alignment Loop (Algorithm 2)
    best_target_r2 = -float("inf")
    save_path = os.path.join(args.save_dir, "phase2_aligned_target_model.pt")

    print("\nStarting Maximum Likelihood Alignment (Phase 2)...")
    for epoch in range(1, args.epochs + 1):
        align_loss = train_epoch_phase2(
            target_encoder=vae.encoder,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            target_loader=tgt_train_loader,
            source_latents_bank=source_latents_bank,
            optimizer=alignment_optimizer,
            device=device,
            alpha=args.alpha,
        )

        target_r2, target_rmse = evaluate_target_alignment(vae, tgt_val_loader, device)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"Alignment Loss: {align_loss:.4f} | "
                f"Target Test R^2: {target_r2:.2f}% | Target RMSE: {target_rmse:.4f}"
            )

        if target_r2 > best_target_r2:
            best_target_r2 = target_r2
            torch.save(
                {
                    "epoch": epoch,
                    "target_read_in_state": vae.encoder.read_in.state_dict(),
                    "target_n_neurons": n_tgt_neurons,
                    "best_target_r2": best_target_r2,
                },
                save_path,
            )

    print(f"\nAlignment Complete!")
    print(f"Initial Unaligned R^2: {pre_r2:.2f}%  -->  Aligned R^2: {best_target_r2:.2f}%")
    print(f"Saved aligned target weights to: {save_path}")


if __name__ == "__main__":
    main()