import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn

from dataloader import denormalize_velocity, get_nhp_dataloaders
from vae import SequentialVAE
from decoder_models import compute_r2_score, compute_rmse, compute_accuracy


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

@torch.no_grad()
def run_inference(vae: SequentialVAE, loader, device: torch.device):
    """
    Runs full inference on a dataset split, extracting latents, predictions, and true labels.
    """
    vae.eval()
    latents_list = []
    pred_vel_list = []
    true_vel_list = []
    targets_list = []

    for batch in loader:
        spikes = batch["spikes"].to(device)
        velocity = batch["velocity"].to(device)
        targets = batch["target"].to(device)

        # 1. Map to latent dynamics
        mu, _ = vae.encoder(spikes)
        # 2. Map latents to predicted velocity
        pred_vel = vae.behavior_decoder(mu)

        latents_list.append(mu.cpu())
        pred_vel_list.append(pred_vel.cpu())
        true_vel_list.append(velocity.cpu())
        targets_list.append(targets.cpu())

    latents = torch.cat(latents_list, dim=0)        # (N, seq_len, latent_dim)
    pred_vel = torch.cat(pred_vel_list, dim=0)      # (N, seq_len, 2)
    true_vel = torch.cat(true_vel_list, dim=0)      # (N, seq_len, 2)
    targets = torch.cat(targets_list, dim=0)        # (N,)

    return latents, pred_vel, true_vel, targets


def plot_latent_dynamics_3d(latents: torch.Tensor, targets: torch.Tensor, title: str, save_path: str):
    """
    Recreates the 3D neural manifold visualization across the 8 reaching directions (Figure 4 in paper).
    Averages latent trajectories per target direction and plots the top 3 latent dimensions.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    latents_np = latents.numpy()
    targets_np = targets.numpy()

    for direction in range(8):
        mask = (targets_np == direction)
        if np.sum(mask) == 0:
            continue
        
        # Mean trajectory for this direction: (seq_len, latent_dim)
        mean_traj = np.mean(latents_np[mask], axis=0)

        # Plot first 3 latent dimensions
        z1, z2, z3 = mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2]
        ax.plot(z1, z2, z3, color=colors[direction], label=f"Dir {direction}", linewidth=2.5)
        # Mark trajectory start
        ax.scatter(z1[0], z2[0], z3[0], color=colors[direction], marker="o", s=30)
        # Mark trajectory end
        ax.scatter(z1[-1], z2[-1], z3[-1], color=colors[direction], marker="^", s=40)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_zlabel("Latent Dim 3")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved 3D latent trajectory plot to: {save_path}")


def plot_kinematics_comparison(true_vel: torch.Tensor, pred_vel: torch.Tensor, save_path: str, n_samples: int = 5):
    """
    Plots sample decoded velocity trajectories against ground truth (Figure 5A in paper).
    """
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples), sharex=True)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_samples):
        # Vx
        axes[i, 0].plot(true_vel[i, :, 0].numpy(), label="True Vx", color="black", linestyle="--")
        axes[i, 0].plot(pred_vel[i, :, 0].numpy(), label="Decoded Vx", color="crimson")
        axes[i, 0].set_ylabel(f"Trial {i+1}\nVelocity X")
        if i == 0:
            axes[i, 0].set_title("X-Velocity Kinematics")
            axes[i, 0].legend(loc="upper right")

        # Vy
        axes[i, 1].plot(true_vel[i, :, 1].numpy(), label="True Vy", color="black", linestyle="--")
        axes[i, 1].plot(pred_vel[i, :, 1].numpy(), label="Decoded Vy", color="royalblue")
        axes[i, 1].set_ylabel("Velocity Y")
        if i == 0:
            axes[i, 1].set_title("Y-Velocity Kinematics")
            axes[i, 1].legend(loc="upper right")

    axes[-1, 0].set_xlabel("Time Bins")
    axes[-1, 1].set_xlabel("Time Bins")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved kinematics decoding plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Aligned Latent Dynamics and Behavior Decoding")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--source_checkpoint", type=str, required=True, help="Path to Phase 1 source checkpoint (.pt)")
    parser.add_argument("--aligned_checkpoint", type=str, required=True, help="Path to Phase 2 target checkpoint (.pt)")
    parser.add_argument("--target_data_path", type=str, required=True, help="Path to target domain .mat file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save plots and metrics")
    parser.add_argument("--seq_len", type=int, default=39, help="Number of time bins per trial")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load checkpoints before constructing the test loader so it uses
    # source-training velocity units, never target-fitted statistics.
    src_ckpt = torch.load(args.source_checkpoint, map_location=device)
    aligned_ckpt = torch.load(args.aligned_checkpoint, map_location=device)
    velocity_stats = aligned_ckpt if "velocity_mean" in aligned_ckpt else src_ckpt
    if "velocity_mean" not in velocity_stats or "velocity_std" not in velocity_stats:
        raise ValueError(
            "Checkpoint is missing source-training velocity statistics; retrain Phase 1/2 "
            "before evaluating to avoid target-derived normalization."
        )

    _, _, target_test_loader, n_tgt_neurons, _ = get_nhp_dataloaders(
        file_path=args.target_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        augment_multiplier=1,
        velocity_stats=velocity_stats,
        include_validation=False,
    )

    # 2. Load Base Model Structure
    latent_dim = src_ckpt["latent_dim"]
    hidden_dim = src_ckpt["hidden_dim"]

    vae = SequentialVAE(
        n_neurons=n_tgt_neurons,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        behavior_dim=2,
    ).to(device)

    # Load core recurrent and decoding parameters from source training
    base_state = src_ckpt["vae_state_dict"]
    # Drop neuron-count-specific layers while retaining the source behavior head.
    filtered_base_state = {
        k: v for k, v in base_state.items()
        if not k.startswith("encoder.read_in") and not k.startswith("decoder.read_out")
    }
    vae.load_state_dict(filtered_base_state, strict=False)

    # -------------------------------------------------------------
    # 3. Unaligned Target Evaluation (Random/Zero initialization of read-in)
    # -------------------------------------------------------------
    # 3. Unaligned Target Evaluation (Direct Transfer Baseline)
    if "unaligned_read_in_state" in aligned_ckpt:
        vae.encoder.read_in.load_state_dict(aligned_ckpt["unaligned_read_in_state"])
    unaligned_latents, unaligned_pred_v, true_v, targets = run_inference(vae, target_test_loader, device)
    true_v = denormalize_velocity(true_v.to(device), velocity_stats).cpu()
    unaligned_pred_v = denormalize_velocity(unaligned_pred_v.to(device), velocity_stats).cpu()
    r2_unaligned = compute_r2_score(true_v, unaligned_pred_v)
    rmse_unaligned = compute_rmse(true_v, unaligned_pred_v)

    # -------------------------------------------------------------
    # 4. Aligned Target Evaluation (Load ERDiff adapted read-in layer)
    # -------------------------------------------------------------
    vae.encoder.read_in.load_state_dict(aligned_ckpt["target_read_in_state"])

    aligned_latents, aligned_pred_v, _, _ = run_inference(vae, target_test_loader, device)
    aligned_pred_v = denormalize_velocity(aligned_pred_v.to(device), velocity_stats).cpu()
    r2_aligned = compute_r2_score(true_v, aligned_pred_v)
    rmse_aligned = compute_rmse(true_v, aligned_pred_v)

    plot_latent_dynamics_3d(
        aligned_latents,
        targets,
        title=f"ERDiff Aligned Target Latents (R^2: {r2_aligned:.2f}%)",
        save_path=os.path.join(args.output_dir, "aligned_latents_3d.png"),
    )

    # Plot sample trial velocity curves
    plot_kinematics_comparison(
        true_vel=true_v,
        pred_vel=aligned_pred_v,
        save_path=os.path.join(args.output_dir, "aligned_kinematics_decoding.png"),
        n_samples=5,
    )

    # -------------------------------------------------------------
    # 5. Summary Report (Matching Paper Results)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("           EXPERIMENTAL RESULTS SUMMARY           ")
    print("=" * 50)
    print(f"Target Session      : {os.path.basename(args.target_data_path)}")
    print(f"Target Neurons      : {n_tgt_neurons}")
    print(f"Latent Dimension    : {latent_dim}")
    print("-" * 50)
    print(f"Unaligned Performance: R^2 = {r2_unaligned:6.2f}% | RMSE = {rmse_unaligned:.4f}")
    print(f"ERDiff Aligned       : R^2 = {r2_aligned:6.2f}% | RMSE = {rmse_aligned:.4f}")
    print(f"Absolute Gain        : ΔR^2 = {r2_aligned - r2_unaligned:+6.2f}% | ΔRMSE = {rmse_aligned - rmse_unaligned:+6.4f}")
    print("=" * 50)

    # Save metrics to text file
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"Unaligned R2: {r2_unaligned:.4f}\n")
        f.write(f"Unaligned RMSE: {rmse_unaligned:.4f}\n")
        f.write(f"Aligned R2: {r2_aligned:.4f}\n")
        f.write(f"Aligned RMSE: {rmse_aligned:.4f}\n")


if __name__ == "__main__":
    main()
