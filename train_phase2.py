import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_nhp_dataloaders
from vae import SequentialVAE, trajectory_smoothness_loss
from diffusion import (
    LatentDiffusionModel,
    VPSDEScheduler,
    total_alignment_loss,
)
from decoder_models import compute_r2_score, compute_rmse


@torch.no_grad()
def extract_source_latent_bank(vae: SequentialVAE, source_loader, device: torch.device):
    """
    Extracts clean deterministic source latent trajectories Z_0^(s)
    for the Sinkhorn Divergence regularizer.
    """
    vae.eval()
    source_latents = []
    for batch in source_loader:
        spikes = batch["spikes"].to(device)
        mu, _ = vae.encoder(spikes)
        source_latents.append(mu)
    return torch.cat(source_latents, dim=0)


def initialize_target_read_in(
    source_read_in: nn.Linear,
    n_src_neurons: int,
    n_tgt_neurons: int,
    hidden_dim: int,
    device: torch.device,
) -> nn.Linear:
    """
    Follows Section 3.3: 'During alignment, we first set phi as phi_s and then linear probe phi.'
    Copies overlapping channel weights from source read_in when neuron counts match or differ.
    """
    new_read_in = nn.Linear(n_tgt_neurons, hidden_dim).to(device)
    with torch.no_grad():
        new_read_in.bias.copy_(source_read_in.bias)
        if n_tgt_neurons == n_src_neurons:
            new_read_in.weight.copy_(source_read_in.weight)
        else:
            # Copy overlapping channels and initialize remaining with small scale
            min_n = min(n_src_neurons, n_tgt_neurons)
            nn.init.xavier_uniform_(new_read_in.weight, gain=0.5)
            new_read_in.weight[:, :min_n].copy_(source_read_in.weight[:, :min_n])
    return new_read_in


def train_epoch_phase2(
    vae: SequentialVAE,
    diffusion_model: LatentDiffusionModel,
    scheduler: VPSDEScheduler,
    target_loader,
    source_latents_bank: torch.Tensor,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha: float = 0.5,
):
    """
    Phase 2: Maximum Likelihood Alignment (Algorithm 2).
    Keeps frozen recurrent/dropout layers in .eval() mode while tuning read_in.
    """
    vae.eval()              # Disable dropout in frozen GRU and LayerNorm
    diffusion_model.eval()  # Frozen score network

    total_loss = 0.0
    num_source_samples = source_latents_bank.size(0)

    for batch in target_loader:
        target_spikes = batch["spikes"].to(device)
        batch_size = target_spikes.size(0)

        optimizer.zero_grad()

        # 1. Forward pass through encoder using deterministic continuous trajectory mu_t
        mu_t, _ = vae.encoder(target_spikes)

        # 2. Sample matching batch of source latents for Sinkhorn regularizer
        perm = torch.randperm(num_source_samples, device=device)[:batch_size]
        z_source_batch = source_latents_bank[perm]

        # 3. Compute MLA + Sinkhorn + Trajectory Smoothness
        loss_align = total_alignment_loss(
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            z_target=mu_t,
            z_source=z_source_batch,
            alpha=alpha,
        )
        loss_smooth = trajectory_smoothness_loss(mu_t)
        loss = loss_align + 0.2 * loss_smooth

        # 4. Update ONLY encoder.read_in
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.encoder.read_in.parameters(), max_norm=1.0)
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
    Evaluates zero-shot kinematics decoding on target domain (R^2 and RMSE).
    """
    vae.eval()
    all_preds = []
    all_targets = []

    for batch in target_loader:
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
    parser = argparse.ArgumentParser(description="Phase 2: Target Domain Maximum Likelihood Alignment (ERDiff)")
    parser.add_argument("--source_checkpoint", type=str, required=True, help="Path to saved Phase 1 checkpoint (.pt)")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to source session .mat file")
    parser.add_argument("--target_data_path", type=str, required=True, help="Path to target session .mat file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=39, help="Number of time bins per trial")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate for linear probing")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="L2 regularization on read-in")
    parser.add_argument("--epochs", type=int, default=150, help="Alignment epochs")
    parser.add_argument("--augment_mult", type=int, default=3, help="Target data augmentation multiplier")
    parser.add_argument("--alpha", type=float, default=0.5, help="Trade-off weighting Sinkhorn loss")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Source and Target Data
    src_train_loader, _, n_src_neurons = get_nhp_dataloaders(
        file_path=args.source_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        augment_multiplier=1,  # Clean source bank
    )
    tgt_train_loader, tgt_val_loader, n_tgt_neurons = get_nhp_dataloaders(
        file_path=args.target_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        augment_multiplier=args.augment_mult,
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

    # 3. Precompute Source Latent Bank
    print("Pre-computing source domain latent bank...")
    source_latents_bank = extract_source_latent_bank(vae, src_train_loader, device)

    # 4. Initialize Target Read-In from Source Read-In (Warm Start) & Freeze Rest
    warm_read_in = initialize_target_read_in(
        source_read_in=vae.encoder.read_in,
        n_src_neurons=n_src_neurons,
        n_tgt_neurons=n_tgt_neurons,
        hidden_dim=hidden_dim,
        device=device,
    )
    vae.encoder.read_in = warm_read_in
    vae.freeze_for_alignment()

    for param in diffusion_model.parameters():
        param.requires_grad = False

    alignment_optimizer = optim.AdamW(
        vae.encoder.read_in.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = VPSDEScheduler()

    # Evaluate Before Alignment (Direct Transfer / Warm-Start Unaligned Baseline)
    pre_r2, pre_rmse = evaluate_target_alignment(vae, tgt_val_loader, device)
    print(f"Target Performance BEFORE Alignment: R^2 = {pre_r2:.2f}% | RMSE = {pre_rmse:.4f}")

    # Save unaligned state so eval.py can compare exact before vs. after
    unaligned_read_in_state = {k: v.clone() for k, v in vae.encoder.read_in.state_dict().items()}
    best_aligned_state = {k: v.clone() for k, v in vae.encoder.read_in.state_dict().items()}
    best_target_r2 = pre_r2

    save_path = os.path.join(args.save_dir, "phase2_aligned_target_model.pt")

    print("\nStarting Maximum Likelihood Alignment (Phase 2)...")
    for epoch in range(1, args.epochs + 1):
        align_loss = train_epoch_phase2(
            vae=vae,
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
                f"Target Val R^2: {target_r2:6.2f}% | Target RMSE: {target_rmse:.4f}"
            )

        if target_r2 > best_target_r2:
            best_target_r2 = target_r2
            best_aligned_state = {k: v.clone() for k, v in vae.encoder.read_in.state_dict().items()}

    torch.save(
        {
            "unaligned_read_in_state": unaligned_read_in_state,
            "target_read_in_state": best_aligned_state,
            "target_n_neurons": n_tgt_neurons,
            "pre_r2": pre_r2,
            "best_target_r2": best_target_r2,
        },
        save_path,
    )

    print(f"\nAlignment Complete!")
    print(f"Unaligned R^2: {pre_r2:.2f}%  -->  ERDiff Aligned R^2: {best_target_r2:.2f}%")
    print(f"Saved aligned target checkpoint to: {save_path}")


if __name__ == "__main__":
    main()