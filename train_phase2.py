import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import NHPDataset, get_nhp_dataloaders
from vae import SequentialVAE, trajectory_smoothness_loss, sample_latent_posterior
from diffusion import (
    LatentDiffusionModel,
    VPSDEScheduler,
    total_alignment_loss,
)


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
                   is_train=False, include_behavior=False),
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
    for batch in source_loader:
        spikes = batch["spikes"].to(device)
        mu, logvar = vae.encoder(spikes)
        source_means.append(mu)
        source_logvars.append(logvar)
    return torch.cat(source_means, dim=0), torch.cat(source_logvars, dim=0)


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
    if init_mode not in ("source_or_random", "ridge"):
        raise ValueError(f"Unknown target read-in initialization: {init_mode}")

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
):
    """
    Phase 2: Maximum Likelihood Alignment (Algorithm 2).
    """
    vae.eval()                    # Disable dropout in encoder/decoder
    vae.encoder.rnn.train()       # Required by cuDNN to backprop through frozen GRU into read_in
    diffusion_model.eval()        # Frozen score network

    total_loss = 0.0
    total_mla = 0.0
    total_ot = 0.0
    total_smooth = 0.0
    total_grad_norm = 0.0
    clipped_steps = 0
    source_mu, source_logvar = source_latents_bank
    num_source_samples = source_mu.size(0)

    for batch in target_loader:
        target_spikes = batch["spikes"].to(device)
        batch_size = target_spikes.size(0)

        optimizer.zero_grad()

        # Algorithm 2 samples target latents from the encoder posterior.
        mu_t, logvar_t = vae.encoder(target_spikes)
        z_target = sample_latent_posterior(mu_t, logvar_t)

        # 2. Sample matching batch of source latents for Sinkhorn regularizer
        perm = torch.randperm(num_source_samples, device=device)[:batch_size]
        z_source_batch = sample_latent_posterior(source_mu[perm], source_logvar[perm])

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
        )
        loss = loss_align
        smooth_value = 0.0
        if smooth_weight:
            loss_smooth = trajectory_smoothness_loss(mu_t)
            smooth_value = loss_smooth.item()
            loss = loss + smooth_weight * loss_smooth

        # 4. Update ONLY encoder.read_in
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(vae.encoder.read_in.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mla += loss_mla.item()
        total_ot += loss_ot.item()
        total_smooth += smooth_value
        total_grad_norm += grad_norm.item()
        clipped_steps += int(grad_norm.item() > 1.0)

    batches = len(target_loader)
    return {
        "total": total_loss / batches,
        "mla": total_mla / batches,
        "ot": total_ot / batches,
        "smooth": total_smooth / batches,
        "grad_norm": total_grad_norm / batches,
        "clipped_steps": clipped_steps,
        "updates": batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Target Domain Maximum Likelihood Alignment (ERDiff)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed used only to define trial partitions")
    parser.add_argument("--source_checkpoint", type=str, required=True, help="Path to saved Phase 1 checkpoint (.pt)")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to source session .mat file")
    parser.add_argument("--target_data_path", type=str, required=True, help="Path to target session .mat file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=39, help="Number of time bins per trial")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for target read-in probing")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Optional read-in weight decay variant")
    parser.add_argument("--epochs", type=int, default=150, help="Alignment epochs")
    parser.add_argument("--augment_mult", type=int, default=3, help="Target data repetition when --augment_target is enabled")
    parser.add_argument("--augment_target", action="store_true", help="Experimental variant: augment target adaptation spikes")
    parser.add_argument("--readin_init", choices=("source_or_random", "ridge"), default="source_or_random",
                        help="Initialize equal-width read-ins from source; use random or ridge for changed neuron counts")
    parser.add_argument("--alpha", type=float, default=0.5, help="Trade-off weighting Sinkhorn loss")
    parser.add_argument("--ot_epsilon", type=float, default=None,
                        help="Entropy weight for full-trajectory transport; default preserves the former per-element temperature")
    parser.add_argument("--sinkhorn_iters", type=int, default=100, help="Log-domain transport scaling iterations")
    parser.add_argument("--smooth_weight", type=float, default=0.0, help="Optional trajectory-smoothness variant; paper default is zero")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Source and Target Data
    checkpoint = torch.load(args.source_checkpoint, map_location=device)
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
    src_train_loader, _, _, n_src_neurons, _ = get_nhp_dataloaders(
        file_path=args.source_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        augment_multiplier=1,
        split_seed=checkpoint["split_seed"],
        split_indices=checkpoint["split_indices"],
    )
    checkpoint_velocity_stats = {
        "velocity_mean": checkpoint["velocity_mean"],
        "velocity_std": checkpoint["velocity_std"],
    }
    tgt_train_loader, _, _, n_tgt_neurons, target_data_metadata = get_nhp_dataloaders(
        file_path=args.target_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        augment_multiplier=args.augment_mult,
        split_seed=args.split_seed,
        velocity_stats=checkpoint_velocity_stats,
        include_validation=False,
        include_test=False,
        include_behavior=False,
    )
    print(f"Source neurons: {n_src_neurons} | Target neurons: {n_tgt_neurons}")
    clean_source_loader = clean_training_view(src_train_loader, args.batch_size)
    clean_target_loader = clean_training_view(tgt_train_loader, args.batch_size)

    # 2. Restore Source Models from Phase 1
    latent_dim = checkpoint["latent_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    if args.ot_epsilon is None:
        args.ot_epsilon = 0.1 * args.seq_len * latent_dim
    print(f"Transport entropy weight: {args.ot_epsilon:g}")

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
    source_latents_bank = extract_source_latent_bank(vae, clean_source_loader, device)

    # 4. Initialize Target Read-In (handles both equal and mismatched neuron counts)
    warm_read_in = initialize_target_read_in(
        source_read_in=vae.encoder.read_in,
        src_loader=clean_source_loader,
        tgt_loader=clean_target_loader,
        n_src_neurons=n_src_neurons,
        n_tgt_neurons=n_tgt_neurons,
        hidden_dim=hidden_dim,
        device=device,
        init_mode=args.readin_init,
    )
    vae.encoder.read_in = warm_read_in
    vae.freeze_for_alignment()

    for param in diffusion_model.parameters():
        param.requires_grad = False

    alignment_optimizer = optim.Adam(
        vae.encoder.read_in.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = VPSDEScheduler(
        beta_min=noise_config["beta_min"], beta_max=noise_config["beta_max"],
        T=noise_config["T"], t_min=noise_config["t_min"],
    )

    # Save the initialization for a paired, pre-alignment target baseline.
    unaligned_read_in_state = {k: v.clone() for k, v in vae.encoder.read_in.state_dict().items()}

    save_path = os.path.join(args.save_dir, "phase2_aligned_target_model.pt")

    print("\nStarting Maximum Likelihood Alignment (Phase 2)...")
    for epoch in range(1, args.epochs + 1):
        align_metrics = train_epoch_phase2(
            vae=vae,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            target_loader=tgt_train_loader if args.augment_target else clean_target_loader,
            source_latents_bank=source_latents_bank,
            optimizer=alignment_optimizer,
            device=device,
            alpha=args.alpha,
            ot_epsilon=args.ot_epsilon,
            sinkhorn_iters=args.sinkhorn_iters,
            smooth_weight=args.smooth_weight,
        )

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"Alignment Loss: {align_metrics['total']:.4f} | "
                f"MLA: {align_metrics['mla']:.4f} | OT: {align_metrics['ot']:.4f} | "
                f"Smooth: {align_metrics['smooth']:.4f} | "
                f"Read-in grad: {align_metrics['grad_norm']:.4f} | "
                f"Clipped: {align_metrics['clipped_steps']}/{align_metrics['updates']}"
            )

    torch.save(
        {
            "unaligned_read_in_state": unaligned_read_in_state,
            "target_read_in_state": {k: v.clone() for k, v in vae.encoder.read_in.state_dict().items()},
            "target_n_neurons": n_tgt_neurons,
            "velocity_mean": checkpoint_velocity_stats["velocity_mean"],
            "velocity_std": checkpoint_velocity_stats["velocity_std"],
            "alignment_epochs": args.epochs,
            "diffusion_training": noise_config,
            "alignment_updates": args.epochs * len(tgt_train_loader if args.augment_target else clean_target_loader),
            "split_indices": target_data_metadata["split_indices"],
            "split_seed": target_data_metadata["split_seed"],
            "alignment_config": {
                "alpha": args.alpha,
                "ot_epsilon": args.ot_epsilon,
                "sinkhorn_iters": args.sinkhorn_iters,
                "smooth_weight": args.smooth_weight,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "augment_mult": args.augment_mult,
                "augment_target": args.augment_target,
                "readin_init": args.readin_init,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "optimizer": "Adam",
            },
        },
        save_path,
    )

    print(f"\nAlignment Complete after {args.epochs} fixed epochs.")
    print(f"Saved aligned target checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
