import os
import re
import subprocess
import argparse
import hashlib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


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

# All 11 Chewie_CO sessions in chronological order
ALL_SESSIONS = [
    "Chewie_20160928_001.mat",
    "Chewie_20160929_001.mat",
    "Chewie_20160930_001.mat",
    "Chewie_20161006_001.mat",
    "Chewie_20161007_001.mat",
    "Chewie_20161011_001.mat",
    "Chewie_20161012_001.mat",
    "Chewie_20161013_001.mat",
    "Chewie_20161102_001.mat",
    "Chewie_20161103_001.mat",
    "Chewie_20161104_001.mat",
]


def extract_date(filename: str) -> datetime:
    """Extracts YYYYMMDD date object from 'Chewie_YYYYMMDD_001.mat'."""
    match = re.search(r"(\d{8})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    raise ValueError(f"Could not parse date from filename: {filename}")


def parse_metrics_file(metrics_path: str) -> dict:
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":")
                    metrics[key.strip()] = float(val.strip())
    return metrics


def plot_cross_day_drift(df: pd.DataFrame, save_path: str):
    """
    Plots Unaligned vs. ERDiff Aligned R^2 across elapsed days.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(
        df["days_elapsed"],
        df["unaligned_r2"],
        marker="o",
        linestyle="--",
        color="gray",
        linewidth=2,
        label="Pre-alignment baseline",
    )
    plt.plot(
        df["days_elapsed"],
        df["aligned_r2"],
        marker="s",
        linestyle="-",
        color="crimson",
        linewidth=2.5,
        label="ERDiff Aligned",
    )

    plt.title("Cross-Day Neural Decoding Performance (Chewie M1)", fontsize=13)
    plt.xlabel("Days Elapsed from Source Session", fontsize=11)
    plt.ylabel("Velocity Decoding $R^2$ (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved cross-day drift plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run 11-Day Cross-Session Alignment Benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--split_seed", type=int, default=42, help="Fixed seed for trial partitions, independent of model seed")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where .mat files are stored")
    parser.add_argument("--source_file", type=str, default="Chewie_20160928_001.mat", help="Source session file")
    parser.add_argument("--source_data_path", type=str, default=None,
                        help="Explicit source data path; avoids assuming source data_dir layout")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
    parser.add_argument("--phase1_checkpoint", type=str, default=None,
                        help="Use this Phase 1 checkpoint; a missing specified checkpoint is an error")
    parser.add_argument("--results_dir", type=str, default="./results", help="Evaluation output directory")
    parser.add_argument("--seq_len", type=int, default=39, help="Time bins per trial")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=32, help="GRU hidden dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--phase2_paper_batch_strategy",
                        choices=("balanced_minibatch", "paper_full_domain"),
                        default="balanced_minibatch",
                        help="Phase 2 batch updates: balanced minibatches or one full domain batch per epoch")
    parser.add_argument("--phase1_epochs", type=int, default=200, help="Epochs for Phase 1 source training")
    parser.add_argument("--phase2_epochs", type=int, default=150, help="Epochs for Phase 2 alignment per day")
    parser.add_argument("--phase2_lr", type=float, default=0.001, help="Target read-in learning rate")
    parser.add_argument("--phase2_weight_decay", type=float, default=0.0)
    parser.add_argument("--phase2_augment_mult", type=int, default=3)
    parser.add_argument("--phase2_sinkhorn_iters", type=int, default=1000)
    parser.add_argument("--phase2_gradient_calibration_batches", type=int, default=3)
    parser.add_argument("--phase2_alignment_objective", choices=("author_2023", "paper_squared"), default="author_2023")
    parser.add_argument("--phase2_alpha", type=float, default=0.5, help="MLA/OT mixture only for paper_squared; author mode uses a sum")
    parser.add_argument("--phase2_ot_epsilon", type=float, default=None, help="POT normalized-cost regularization; default 0.01")
    parser.add_argument("--phase2_smooth_weight", type=float, default=0.0, help="Optional smoothness variant")
    parser.add_argument("--phase2_mla_variance_weight", type=float, default=0.0, help="Optional MLA source-variance constraint")
    parser.add_argument("--phase2_mla_variance_floor", type=float, default=0.9, help="Source standard-deviation floor ratio")
    parser.add_argument("--phase2_mla_variance_scope", choices=("batch", "full_train"), default="batch")
    parser.add_argument("--phase2_readin_init", choices=("source_or_random", "ridge", "random"), default="source_or_random")
    parser.add_argument("--phase2_match_input_size", action="store_true",
                        help="Map each target session to the source checkpoint input width")
    parser.add_argument("--input_mapping_seed", type=int, default=42,
                        help="Seed for deterministic wider-target channel selection")
    parser.add_argument("--phase2_augment_target", action="store_true", help="Experimental target-spike augmentation")
    parser.add_argument("--phase2_noise_mode", choices=("resampled", "fixed_bank"), default="resampled")
    parser.add_argument("--phase2_gradient_balance", choices=("none", "source_calibrated"), default="none")
    parser.add_argument("--frozen_check", choices=("final", "every_epoch"), default="every_epoch")
    parser.add_argument("--log_component_gradients", action="store_true")
    parser.add_argument("--controlled_reference", default=None,
                        help="Reference run manifest with saved split/fingerprint per target session")
    parser.add_argument("--controlled_case", choices=("reference", "control", "fixed_noise", "matched_size", "gradient_balance", "all"))
    parser.add_argument("--controlled_output_root", default=None)
    parser.add_argument("--target_sessions", nargs="*", default=None)
    parser.add_argument("--force_phase1", action="store_true", help="Force re-training Phase 1 even if ckpt exists")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    source_path = args.source_data_path or os.path.join(args.data_dir, args.source_file)
    phase1_ckpt = args.phase1_checkpoint or os.path.join(args.checkpoints_dir, "phase1_source_models.pt")
    source_date = extract_date(args.source_file)

    # -------------------------------------------------------------------------
    # Step 1: Phase 1 Source Domain Training
    # -------------------------------------------------------------------------
    if args.phase1_checkpoint and not os.path.isfile(phase1_ckpt):
        raise FileNotFoundError(f"Specified Phase 1 checkpoint does not exist: {phase1_ckpt}")
    if args.controlled_reference and not args.controlled_case:
        raise ValueError("--controlled_reference requires --controlled_case.")
    if not args.phase1_checkpoint and (args.force_phase1 or not os.path.exists(phase1_ckpt)):
        print(f"=== Starting Phase 1 Source Training on {args.source_file} ===")
        cmd_p1 = [
            "python", "train_phase1.py",
            "--seed", str(args.seed),
            "--split_seed", str(args.split_seed),
            "--data_path", source_path,
            "--save_dir", args.checkpoints_dir,
            "--seq_len", str(args.seq_len),
            "--latent_dim", str(args.latent_dim),
            "--hidden_dim", str(args.hidden_dim),
            "--batch_size", str(args.batch_size),
            "--lr", "0.005",
            "--weight_decay", "0.001",
            "--epochs", str(args.phase1_epochs),
            "--augment_mult", "5",
            "--kl_weight", "0.05",
            "--behavior_weight", "2.0",
        ]
        subprocess.run(cmd_p1, check=True)
    else:
        print(f"Found existing Phase 1 checkpoint at {phase1_ckpt}. Skipping Phase 1 training.")

    # -------------------------------------------------------------------------
    # Step 2: Phase 2 Alignment & Evaluation Loop Across Target Days
    # -------------------------------------------------------------------------
    target_sessions = args.target_sessions or [s for s in ALL_SESSIONS if s != args.source_file]
    summary_records = []

    for idx, target_file in enumerate(target_sessions, 1):
        target_path = os.path.join(args.data_dir, target_file)
        if not os.path.exists(target_path):
            if args.controlled_reference:
                raise FileNotFoundError(f"Controlled cross-day target session is missing: {target_path}")
            print(f"[Warning] File not found: {target_path}. Skipping.")
            continue

        session_tag = target_file.replace(".mat", "")
        case_root = args.controlled_output_root or args.checkpoints_dir
        target_run_root = os.path.join(case_root, session_tag) if args.controlled_reference else args.checkpoints_dir
        target_ckpt_name = f"phase2_aligned_{session_tag}.pt"
        target_ckpt_path = os.path.join(target_run_root, target_ckpt_name)
        target_out_dir = (os.path.join(target_run_root, "results")
                          if args.controlled_reference else os.path.join(args.results_dir, session_tag))
        os.makedirs(target_run_root, exist_ok=not bool(args.controlled_reference))
        os.makedirs(target_out_dir, exist_ok=not bool(args.controlled_reference))

        days_elapsed = (extract_date(target_file) - source_date).days

        print(f"\n=======================================================")
        print(f"[{idx}/{len(target_sessions)}] Aligning Target Day: {target_file} (+{days_elapsed} days)")
        print(f"=======================================================")

        # 1. Run Phase 2 Alignment
        phase2_reference = None
        if args.controlled_reference:
            from controlled_phase2_runs import (derive_session_reference, resolve_case_config)
            from phase2_run_manifest import save_run_manifest
            with open(args.controlled_reference, "r", encoding="utf-8") as handle:
                parent_reference = json.load(handle)
            controlled_reference = resolve_case_config(parent_reference, "cross_day", args.controlled_case)
            session_artifact = parent_reference.get("target_sessions", {}).get(target_file)
            if session_artifact is None:
                raise ValueError(f"Reference lacks saved per-session split for {target_file}; refusing a new split.")
            digest = hashlib.sha256()
            with open(target_path, "rb") as data_handle:
                for block in iter(lambda: data_handle.read(1024 * 1024), b""):
                    digest.update(block)
            session_reference = derive_session_reference(
                parent_reference, controlled_reference, target_file, session_artifact,
                digest.hexdigest(),
            )
            phase2_reference = os.path.join(target_run_root, "phase2_reference.json")
            save_run_manifest(phase2_reference, session_reference)

        cmd_p2 = [
            "python", "train_phase2.py",
            "--seed", str(args.seed),
            "--split_seed", str(args.split_seed),
            "--source_checkpoint", phase1_ckpt,
            "--source_data_path", source_path,
            "--target_data_path", target_path,
            "--save_dir", target_run_root,
            "--seq_len", str(args.seq_len),
            "--batch_size", str(args.batch_size),
            "--paper_batch_strategy", args.phase2_paper_batch_strategy,
            "--lr", str(args.phase2_lr),
            "--weight_decay", str(args.phase2_weight_decay),
            "--epochs", str(args.phase2_epochs),
            "--augment_mult", str(args.phase2_augment_mult),
            "--alpha", str(args.phase2_alpha),
            "--alignment_objective", args.phase2_alignment_objective,
            "--smooth_weight", str(args.phase2_smooth_weight),
            "--mla_variance_weight", str(args.phase2_mla_variance_weight),
            "--mla_variance_floor", str(args.phase2_mla_variance_floor),
            "--mla_variance_scope", args.phase2_mla_variance_scope,
            "--readin_init", args.phase2_readin_init,
            "--alignment_noise_mode", args.phase2_noise_mode,
            "--gradient_balance", args.phase2_gradient_balance,
            "--frozen_check", args.frozen_check,
            "--sinkhorn_iters", str(args.phase2_sinkhorn_iters),
        ]
        if phase2_reference:
            cmd_p2.extend(("--reference_run", phase2_reference))
        if args.log_component_gradients:
            cmd_p2.append("--log_component_gradients")
        if args.phase2_noise_mode == "fixed_bank":
            shared_dir = os.path.join(os.path.dirname(case_root), "shared_artifacts") if args.controlled_reference else args.checkpoints_dir
            os.makedirs(shared_dir, exist_ok=True)
            cmd_p2.extend(("--noise_seed", "42", "--noise_bank_replicates", "8",
                           "--noise_bank_path", os.path.join(shared_dir, f"noise_{session_tag}.pt")))
        if args.phase2_gradient_balance == "source_calibrated":
            shared_dir = os.path.join(os.path.dirname(case_root), "shared_artifacts") if args.controlled_reference else args.checkpoints_dir
            os.makedirs(shared_dir, exist_ok=True)
            cmd_p2.extend(("--gradient_calibration_seed", "42", "--gradient_calibration_batches",
                           str(args.phase2_gradient_calibration_batches),
                           "--gradient_calibration_path", os.path.join(shared_dir, "source_gradient_calibration.json")))
        if args.phase2_ot_epsilon is not None:
            cmd_p2.extend(("--ot_epsilon", str(args.phase2_ot_epsilon)))
        if args.phase2_augment_target:
            cmd_p2.append("--augment_target")
        if args.phase2_match_input_size:
            cmd_p2.extend(("--match_input_size", "--input_mapping_seed", str(args.input_mapping_seed)))
        subprocess.run(cmd_p2, check=True)

        # Rename saved Phase 2 checkpoint to session-specific filename
        default_p2_saved = os.path.join(target_run_root, "phase2_aligned_target_model.pt")
        if os.path.exists(default_p2_saved):
            os.replace(default_p2_saved, target_ckpt_path)
        if not os.path.isfile(target_ckpt_path):
            raise FileNotFoundError(f"Phase 2 did not produce its expected checkpoint: {target_ckpt_path}")

        # 2. Run Evaluation & Plot Generation
        cmd_eval = [
            "python", "evaluate_cross_day.py",
            "--seed", str(args.seed),
            "--source_checkpoint", phase1_ckpt,
            "--aligned_checkpoint", target_ckpt_path,
            "--target_data_path", target_path,
            "--output_dir", target_out_dir,
            "--seq_len", str(args.seq_len),
            "--batch_size", str(args.batch_size),
        ]
        subprocess.run(cmd_eval, check=True)

        # 3. Collect Metrics
        metrics_file = os.path.join(target_out_dir, "metrics.txt")
        res = parse_metrics_file(metrics_file)
        unaligned_r2 = res.get("Unaligned R2", 0.0)
        aligned_r2 = res.get("Aligned R2", 0.0)

        import torch
        target_checkpoint = torch.load(target_ckpt_path, map_location="cpu")
        summary_records.append({
            "target_session": target_file,
            "seed": args.seed,
            "split_seed": int(res.get("Split seed", args.split_seed)),
            "test_trials": int(res.get("Test trials", 0)),
            "alignment_epochs": int(res.get("Alignment epochs", 0)),
            "days_elapsed": days_elapsed,
            "unaligned_r2": round(unaligned_r2, 2),
            "aligned_r2": round(aligned_r2, 2),
            "r2_gain": round(aligned_r2 - unaligned_r2, 2),
            "unaligned_rmse": round(res.get("Unaligned RMSE", 0.0), 4),
            "aligned_rmse": round(res.get("Aligned RMSE", 0.0), 4),
            "alignment_updates": int(target_checkpoint.get("alignment_updates", 0)),
            "sample_exposures": int(target_checkpoint.get("sample_exposures", 0)),
            "raw_target_channels": int(target_checkpoint.get("target_n_neurons", 0)),
            "model_input_channels": int(target_checkpoint.get("model_input_width", 0)),
        })

    # -------------------------------------------------------------------------
    # Step 3: Aggregate Table & Drift Visualization
    # -------------------------------------------------------------------------
    if len(summary_records) > 0:
        df = pd.DataFrame(summary_records).sort_values(by="days_elapsed")
        summary_dir = args.controlled_output_root or args.results_dir
        os.makedirs(summary_dir, exist_ok=True)
        csv_path = os.path.join(summary_dir, "cross_day_summary.csv")
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 75)
        print("                     11-DAY CROSS-SESSION BENCHMARK SUMMARY               ")
        print("=" * 75)
        print(df.to_string(index=False))
        print("-" * 75)
        print(f"Mean Unaligned R^2 : {df['unaligned_r2'].mean():6.2f}%")
        print(f"Mean Aligned R^2   : {df['aligned_r2'].mean():6.2f}%")
        print(f"Mean R^2 Gain      : {df['r2_gain'].mean():+6.2f}%")
        print("=" * 75)
        print(f"Summary CSV saved to: {csv_path}")

        plot_path = os.path.join(summary_dir, "cross_day_r2_curve.png")
        plot_cross_day_drift(df, plot_path)


if __name__ == "__main__":
    main()
