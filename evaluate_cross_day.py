import argparse
from checkpoint_provenance import verify_data_fingerprint, verify_preprocessing, verify_source_pair
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader import denormalize_velocity, get_nhp_dataloaders
from vae import SequentialVAE, posterior_logvar_mode_from_checkpoint
from numerical_checks import require_finite
from decoder_models import compute_r2_score, compute_rmse
from neuron_input_mapping import apply_mapping_to_loader, mapping_from_checkpoint

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
    Runs deterministic inference on a dataset split, extracting latents,
    velocity predictions, and ground-truth labels.
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

        # 1. Map spikes to continuous deterministic latent trajectory mu
        mu, _ = vae.encoder(spikes)
        # 2. Map latents to predicted velocity
        pred_vel = vae.behavior_decoder(mu)
        require_finite("evaluation_predictions", latent=mu, prediction=pred_vel, target=velocity)

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
    Recreates the 3D neural manifold visualization across the 8 reaching directions.
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

        mean_traj = np.mean(latents_np[mask], axis=0)
        z1, z2, z3 = mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2]
        ax.plot(z1, z2, z3, color=colors[direction], label=f"Dir {direction}", linewidth=2.5)
        ax.scatter(z1[0], z2[0], z3[0], color=colors[direction], marker="o", s=30)
        ax.scatter(z1[-1], z2[-1], z3[-1], color=colors[direction], marker="^", s=40)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_zlabel("Latent Dim 3")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_kinematics_comparison(true_vel: torch.Tensor, pred_vel: torch.Tensor, save_path: str, n_samples: int = 5):
    """
    Plots sample decoded velocity trajectories against ground truth.
    """
    n_samples = min(n_samples, true_vel.size(0))
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.2 * n_samples), sharex=True)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_samples):
        axes[i, 0].plot(true_vel[i, :, 0].numpy(), label="True Vx", color="black", linestyle="--")
        axes[i, 0].plot(pred_vel[i, :, 0].numpy(), label="Decoded Vx", color="crimson")
        axes[i, 0].set_ylabel(f"Trial {i+1}\nVx")
        if i == 0:
            axes[i, 0].set_title("X-Velocity Kinematics")
            axes[i, 0].legend(loc="upper right")

        axes[i, 1].plot(true_vel[i, :, 1].numpy(), label="True Vy", color="black", linestyle="--")
        axes[i, 1].plot(pred_vel[i, :, 1].numpy(), label="Decoded Vy", color="royalblue")
        axes[i, 1].set_ylabel("Vy")
        if i == 0:
            axes[i, 1].set_title("Y-Velocity Kinematics")
            axes[i, 1].legend(loc="upper right")

    axes[-1, 0].set_xlabel("Time Bins")
    axes[-1, 1].set_xlabel("Time Bins")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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

    # 1. Load checkpoints and source-fitted velocity units before making the
    # untouched target test loader.
    src_ckpt = torch.load(args.source_checkpoint, map_location=device)
    aligned_ckpt = torch.load(args.aligned_checkpoint, map_location=device)
    verify_source_pair(args.source_checkpoint, aligned_ckpt)
    verify_data_fingerprint(args.target_data_path, aligned_ckpt.get("target_data_fingerprint"), "target data")
    velocity_stats = aligned_ckpt if "velocity_mean" in aligned_ckpt else src_ckpt
    if "velocity_mean" not in velocity_stats or "velocity_std" not in velocity_stats:
        raise ValueError(
            "Checkpoint is missing source-training velocity statistics; retrain Phase 1/2 "
            "before evaluating to avoid target-derived normalization."
        )
    if "split_indices" not in aligned_ckpt:
        raise ValueError("Aligned checkpoint is missing saved target trial splits; rerun Phase 2.")

    _, _, target_test_loader, n_tgt_neurons, target_metadata = get_nhp_dataloaders(
        file_path=args.target_data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        augment_multiplier=1,
        velocity_stats=velocity_stats,
        split_seed=aligned_ckpt.get("split_seed", 42),
        split_indices=aligned_ckpt["split_indices"],
        include_validation=False,
    )
    verify_preprocessing(target_metadata["preprocessing"], aligned_ckpt.get("preprocessing"))
    input_mapping, model_input_width = mapping_from_checkpoint(aligned_ckpt, n_tgt_neurons)
    apply_mapping_to_loader(target_test_loader, input_mapping)
    readin_state = aligned_ckpt.get("target_read_in_state", {})
    if readin_state and readin_state["weight"].shape[1] != model_input_width:
        raise ValueError("Aligned read-in width differs from the saved input mapping contract.")

    latent_dim = src_ckpt["latent_dim"]
    hidden_dim = src_ckpt["hidden_dim"]

    vae = SequentialVAE(
        n_neurons=model_input_width,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        behavior_dim=2,
        posterior_logvar_mode=posterior_logvar_mode_from_checkpoint(src_ckpt),
    ).to(device)

    # Load frozen recurrent & behavior decoder weights from source
    # Exclude both encoder.read_in and decoder.read_out since neuron counts vary across days
    base_state = src_ckpt["vae_state_dict"]
    filtered_base_state = {
        k: v for k, v in base_state.items()
        if not k.startswith("encoder.read_in") and not k.startswith("decoder.read_out")
    }
    vae.load_state_dict(filtered_base_state, strict=False)

    # 3. Pre-alignment initialization baseline. A ridge variant may use target
    # training spikes when neuron counts differ, so this is not raw direct transfer.
    if "unaligned_read_in_state" in aligned_ckpt:
        vae.encoder.read_in.load_state_dict(aligned_ckpt["unaligned_read_in_state"])

    unaligned_latents, unaligned_pred_v, true_v, targets = run_inference(vae, target_test_loader, device)
    true_v = denormalize_velocity(true_v.to(device), velocity_stats).cpu()
    unaligned_pred_v = denormalize_velocity(unaligned_pred_v.to(device), velocity_stats).cpu()
    r2_unaligned = compute_r2_score(true_v, unaligned_pred_v)
    rmse_unaligned = compute_rmse(true_v, unaligned_pred_v)

    plot_latent_dynamics_3d(
        unaligned_latents,
        targets,
        title=f"Unaligned Target Latents (R^2: {r2_unaligned:.2f}%)",
        save_path=os.path.join(args.output_dir, "unaligned_latents_3d.png"),
    )

    # 4. Aligned Target Evaluation (ERDiff Probed Read-In)
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

    plot_kinematics_comparison(
        true_vel=true_v,
        pred_vel=aligned_pred_v,
        save_path=os.path.join(args.output_dir, "aligned_kinematics_decoding.png"),
        n_samples=5,
    )

    # 5. Summary Report
    print("\n" + "=" * 55)
    print("             EXPERIMENTAL RESULTS SUMMARY              ")
    print("=" * 55)
    print(f"Target Session       : {os.path.basename(args.target_data_path)}")
    print(f"Target Raw Channels  : {n_tgt_neurons}")
    print(f"Model Input Channels : {model_input_width}")
    print("-" * 55)
    print(f"Pre-alignment baseline: R^2 = {r2_unaligned:6.2f}% | RMSE = {rmse_unaligned:.4f}")
    print(f"ERDiff Aligned       : R^2 = {r2_aligned:6.2f}% | RMSE = {rmse_aligned:.4f}")
    print(f"Absolute Gain        : dR^2 = {r2_aligned - r2_unaligned:+6.2f}% | dRMSE = {rmse_aligned - rmse_unaligned:+6.4f}")
    print("=" * 55)

    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Test trials: {len(target_test_loader.dataset)}\n")
        f.write(f"Alignment epochs: {aligned_ckpt.get('alignment_epochs', -1)}\n")
        f.write(f"Split seed: {aligned_ckpt.get('split_seed', -1)}\n")
        f.write(f"Target raw channels: {n_tgt_neurons}\n")
        f.write(f"Model input channels: {model_input_width}\n")
        f.write(f"Unaligned R2: {r2_unaligned:.4f}\n")
        f.write(f"Unaligned RMSE: {rmse_unaligned:.4f}\n")
        f.write(f"Aligned R2: {r2_aligned:.4f}\n")
        f.write(f"Aligned RMSE: {rmse_aligned:.4f}\n")


if __name__ == "__main__":
    main()
