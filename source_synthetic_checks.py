"""Source-only development checks; source test and real target data are unused."""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from dataloader import NHPDataset, denormalize_velocity, get_nhp_dataloaders
from decoder_models import compute_r2_score, compute_rmse
from diffusion import LatentDiffusionModel, VPSDEScheduler
from mla_diagnostics import MLADiagnostics, variance_summary
from mla_stabilization import source_trial_variance
from alignment_noise_bank import AlignmentNoiseBank
from phase2_freeze_guard import FrozenDiffusionGuard
from phase2_sampling import make_alignment_loader, source_sampling_generator
from synthetic_shifts import apply_shift, make_shift, oracle_readin_weight, validation_development_split
from train_phase2 import clean_training_view, extract_source_latent_bank, set_seed, train_epoch_phase2
from vae import SequentialVAE


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def predict(model, spikes, device, batch_size):
    model.eval()
    chunks = []
    for batch in spikes.split(batch_size):
        mu, _ = model.encoder(batch.to(device))
        chunks.append(model.behavior_decoder(mu).cpu())
    return torch.cat(chunks)


def metrics(prediction, velocity, stats):
    prediction = denormalize_velocity(prediction, stats)
    velocity = denormalize_velocity(velocity, stats)
    return {"r2": compute_r2_score(velocity, prediction), "rmse": compute_rmse(velocity, prediction)}


def write_summary(path, records):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--source_data_path", required=True)
    parser.add_argument("--output_dir", default="./source_synthetic42")
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--shifts", nargs="+", choices=("identity", "permutation", "gain", "masked_removal"),
                        default=["identity", "permutation", "gain", "masked_removal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--shift_seed", type=int, default=42)
    parser.add_argument("--mask_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=39)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mla_variance_floor", type=float, default=0.9)
    parser.add_argument("--mla_variance_scope", choices=("batch", "full_train"), default="full_train")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--alignment_noise_mode", choices=("resampled", "fixed_bank"), default="resampled")
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--noise_bank_replicates", type=int, default=8)
    parser.add_argument("--noise_bank_path", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Use positive epochs/lr and batch size >= 2.")
    if not 0 < args.mla_variance_floor <= 1 or any(not math.isfinite(w) or w < 0 for w in args.weights):
        raise ValueError("Variance floor must be in (0, 1] and weights finite/nonnegative.")
    if args.noise_bank_replicates < 1:
        raise ValueError("Noise bank replicate count must be positive.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
    if checkpoint.get("diffusion_training", {}).get("latent_distribution") != "posterior_sample":
        raise ValueError("These checks require the posterior-sample source diffusion checkpoint.")
    if checkpoint.get("split_seed") != args.split_seed:
        raise ValueError("Requested split seed differs from the saved source checkpoint split seed.")
    source_hash = fingerprint(args.source_checkpoint)
    data_hash = fingerprint(args.source_data_path)
    train, validation, _, n_neurons, _ = get_nhp_dataloaders(
        args.source_data_path, batch_size=args.batch_size, seq_len=args.seq_len,
        augment_multiplier=1, split_seed=args.split_seed, split_indices=checkpoint["split_indices"],
        velocity_stats=checkpoint, train_noise_factor=0.0, include_test=False,
    )
    if n_neurons != checkpoint["n_neurons"]:
        raise ValueError("Source neuron count differs from the checkpoint.")
    validation_ids = checkpoint["split_indices"]["validation"].cpu()
    intervals = checkpoint["split_indices"]["window_intervals"].cpu()[validation_ids]
    adaptation_offsets, evaluation_offsets = validation_development_split(intervals, args.split_seed)
    validation_spikes = validation.dataset.spikes
    evaluation_velocity = validation.dataset.velocities[evaluation_offsets]

    source = SequentialVAE(n_neurons, checkpoint["latent_dim"], checkpoint["hidden_dim"], 2).to(device)
    source.load_state_dict(checkpoint["vae_state_dict"])
    source.freeze_for_alignment()
    source.eval()
    denoiser = LatentDiffusionModel(
        seq_len=args.seq_len, latent_dim=checkpoint["latent_dim"], hidden_dim=checkpoint["hidden_dim"],
        num_blocks=4, n_heads=4,
    ).to(device)
    denoiser.load_state_dict(checkpoint["dm_state_dict"])
    denoiser_guard = FrozenDiffusionGuard(denoiser)
    denoiser_guard.verify_or_record_failure(
        failure_path=str(output / "phase2_freeze_failure.json"),
        context={"stage": "after_phase1_load", "source_checkpoint": str(Path(args.source_checkpoint).resolve())},
    )
    noise = checkpoint["diffusion_training"]
    scheduler = VPSDEScheduler(noise["beta_min"], noise["beta_max"], noise["T"], noise["t_min"])
    probe_times = [t for t in (0.01, 0.05, 0.2, 0.5, 0.9) if scheduler.t_min <= t <= scheduler.T]
    if not probe_times:
        probe_times = [(scheduler.t_min + scheduler.T) / 2]
    bank = extract_source_latent_bank(source, clean_training_view(train, args.batch_size), device)
    reference_variance = source_trial_variance(bank.mu)
    clean_predictions = predict(source, validation_spikes[evaluation_offsets], device, args.batch_size)
    clean_metrics = metrics(clean_predictions, evaluation_velocity, checkpoint)
    shifts = {kind: make_shift(kind, n_neurons, args.shift_seed, args.mask_fraction) for kind in args.shifts}
    noise_bank_path = args.noise_bank_path or str(output / "alignment_noise_bank.pt")
    noise_bank = None
    source_noise_domain = f"source:{source_hash}"
    target_noise_domains = {kind: f"synthetic-target:{kind}:{source_hash}" for kind in shifts}
    if args.alignment_noise_mode == "fixed_bank":
        if Path(noise_bank_path).is_file():
            noise_bank = AlignmentNoiseBank.load(
                noise_bank_path, seed=args.noise_seed, replicates=args.noise_bank_replicates
            )
        else:
            noise_bank = AlignmentNoiseBank(args.noise_seed, args.noise_bank_replicates)
        noise_bank.register_domain(source_noise_domain, bank.trial_ids, args.seq_len, checkpoint["latent_dim"])
        for kind, domain in target_noise_domains.items():
            noise_bank.register_domain(
                domain, validation_ids[adaptation_offsets], args.seq_len, checkpoint["latent_dim"]
            )
    torch.save({
        "protocol": "source_validation_development_v1", "arguments": vars(args),
        "source_checkpoint_sha256": source_hash, "source_data_sha256": data_hash,
        "source_training_indices": checkpoint["split_indices"]["train"].cpu(),
        "adaptation_indices": validation_ids[adaptation_offsets],
        "evaluation_indices": validation_ids[evaluation_offsets],
        "validation_window_intervals": intervals, "shifts": shifts,
        "alignment_noise_mode": args.alignment_noise_mode,
        "noise_seed": args.noise_seed,
        "noise_bank_replicates": args.noise_bank_replicates,
        "noise_bank_path": noise_bank_path if noise_bank is not None else None,
        "probe_noise_namespace": "independent_mla_diagnostics_fixture",
    }, output / "synthetic_manifest.pt")
    print(f"Source-only development: adaptation={len(adaptation_offsets)}, evaluation={len(evaluation_offsets)}")
    print(f"Clean source evaluation R2={clean_metrics['r2']:.2f}% | source SHA256={source_hash}")
    print("These are source validation development subsets; no source test or real target metric is used.")

    records = []
    for kind, shift in shifts.items():
        adaptation_spikes = apply_shift(validation_spikes[adaptation_offsets], shift)
        evaluation_spikes = apply_shift(validation_spikes[evaluation_offsets], shift)
        dataset = NHPDataset(adaptation_spikes, torch.zeros(len(adaptation_spikes), args.seq_len, 2),
                             torch.zeros(len(adaptation_spikes), dtype=torch.long), include_behavior=False,
                             trial_ids=validation_ids[adaptation_offsets])
        oracle_weight = oracle_readin_weight(source.encoder.read_in.weight, shift)
        oracle_r2, oracle_error = None, None
        if oracle_weight is not None:
            oracle = copy.deepcopy(source)
            with torch.no_grad():
                oracle.encoder.read_in.weight.copy_(oracle_weight)
            oracle_predictions = predict(oracle, evaluation_spikes, device, args.batch_size)
            oracle_error = (oracle_predictions - clean_predictions).abs().max().item()
            if not torch.allclose(oracle_predictions, clean_predictions, rtol=1e-4, atol=1e-5):
                raise ValueError(f"Known inverse read-in failed the {kind} calibration check.")
            oracle_r2 = metrics(oracle_predictions, evaluation_velocity, checkpoint)["r2"]
            del oracle

        for weight in args.weights:
            # Restart the same initialization, sampler and noise stream for each candidate.
            set_seed(args.seed)
            model = copy.deepcopy(source)
            loader = make_alignment_loader(dataset, args.batch_size, args.seed)
            optimizer = torch.optim.Adam(model.encoder.read_in.parameters(), lr=args.lr)
            source_rng = source_sampling_generator(args.seed)
            pool = adaptation_spikes.to(device) if args.mla_variance_scope == "full_train" else None
            run_dir = output / f"{kind}_w{weight:g}"
            run_dir.mkdir(parents=True, exist_ok=True)
            failure_path = run_dir / "phase2_freeze_failure.json"
            denoiser_guard.verify_or_record_failure(
                (optimizer,), failure_path=str(failure_path),
                context={"stage": "before_synthetic_alignment", "shift": kind,
                         "variance_weight": weight, "seed": args.seed},
            )
            before = metrics(predict(model, evaluation_spikes, device, args.batch_size), evaluation_velocity, checkpoint)
            variance_before = variance_summary(model, adaptation_spikes, device, args.batch_size)
            diagnostics = None
            if args.diagnostics:
                diagnostics = MLADiagnostics(
                    dataset, validation_ids[adaptation_offsets], checkpoint["latent_dim"],
                    probe_times, args.seed, 16, str(run_dir),
                    context={"shift": kind, "variance_weight": weight, "scope": args.mla_variance_scope,
                             "seed": args.seed, "split_seed": args.split_seed, "shift_seed": args.shift_seed,
                             "alignment_noise_mode": args.alignment_noise_mode,
                             "noise_seed": args.noise_seed,
                             "noise_bank_replicates": args.noise_bank_replicates,
                             "probe_noise_is_independent_of_fit_bank": True},
                )
                diagnostics.record(0, model, denoiser, scheduler, device, args.batch_size)
            history = [{"epoch": 0, "evaluation": before}]
            print(f"\n{kind}, weight={weight:g}: baseline R2={before['r2']:.2f}% | updates/epoch={len(loader)}")
            for epoch in range(1, args.epochs + 1):
                losses = train_epoch_phase2(
                    model, denoiser, scheduler, loader, bank, optimizer, device, alpha=0.0,
                    ot_epsilon=0.1 * args.seq_len * checkpoint["latent_dim"],
                    source_variance=reference_variance, mla_variance_weight=weight,
                    mla_variance_floor=args.mla_variance_floor, mla_variance_scope=args.mla_variance_scope,
                    variance_pool_spikes=pool,
                    noise_bank=noise_bank,
                    noise_replicate=(epoch - 1) % args.noise_bank_replicates,
                    source_sampling_rng=source_rng,
                    target_noise_domain=target_noise_domains[kind],
                    source_noise_domain=source_noise_domain,
                )
                denoiser_guard.verify_or_record_failure(
                    (optimizer,), failure_path=str(failure_path),
                    context={"stage": "synthetic_epoch_end", "epoch": epoch,
                             "shift": kind, "variance_weight": weight, "seed": args.seed},
                )
                if epoch in (1, 5) or epoch % 10 == 0 or epoch == args.epochs:
                    current = metrics(predict(model, evaluation_spikes, device, args.batch_size), evaluation_velocity, checkpoint)
                    history.append({"epoch": epoch, "losses": losses, "evaluation": current})
                    print(f"{kind}, weight={weight:g}, epoch={epoch}: R2={current['r2']:.2f}% | loss={losses['total']:.4f}")
                    if diagnostics is not None:
                        diagnostics.record(epoch, model, denoiser, scheduler, device, args.batch_size)
            denoiser_guard.verify_or_record_failure(
                (optimizer,), failure_path=str(failure_path),
                context={"stage": "synthetic_final", "epoch": args.epochs,
                         "shift": kind, "variance_weight": weight, "seed": args.seed},
            )
            after = metrics(predict(model, evaluation_spikes, device, args.batch_size), evaluation_velocity, checkpoint)
            variance_after = variance_summary(model, adaptation_spikes, device, args.batch_size)
            row = {
                "shift": kind, "weight": weight, "scope": args.mla_variance_scope,
                "variance_floor": args.mla_variance_floor, "lr": args.lr,
                "seed": args.seed, "split_seed": args.split_seed, "shift_seed": args.shift_seed,
                "adaptation_trials": len(adaptation_offsets), "evaluation_trials": len(evaluation_offsets),
                "epochs": args.epochs, "updates": args.epochs * len(loader),
                "clean_source_r2": clean_metrics["r2"], "oracle_r2": oracle_r2,
                "oracle_prediction_max_error": oracle_error,
                "before_r2": before["r2"], "after_r2": after["r2"], "r2_gain": after["r2"] - before["r2"],
                "before_rmse": before["rmse"], "after_rmse": after["rmse"],
                "source_checkpoint_sha256": source_hash,
                "source_data_sha256": data_hash, "source_epoch": checkpoint.get("epoch"),
            }
            for key in ("mu_variance", "mu_trial_variance", "posterior_variance", "prediction_variance"):
                row[key + "_ratio"] = sum(variance_after[key]) / max(sum(variance_before[key]), 1e-12)
            records.append(row)
            write_summary(output / "summary.csv", records)
            with open(run_dir / "history.json", "w") as handle:
                json.dump(history, handle, indent=2, allow_nan=False)
            torch.save({
                "read_in": {name: value.detach().cpu() for name, value in model.encoder.read_in.state_dict().items()},
                "shift": shift, "metrics": row, "arguments": vars(args),
                "adaptation_indices": validation_ids[adaptation_offsets],
                "evaluation_indices": validation_ids[evaluation_offsets],
            }, run_dir / "synthetic_readin.pt")
            print(f"{kind}, weight={weight:g}: final R2={after['r2']:.2f}% | gain={row['r2_gain']:+.2f}")
            del model, optimizer

    print(f"Saved source-only development summary: {output / 'summary.csv'}")
    if noise_bank is not None:
        noise_bank_hash = noise_bank.save(noise_bank_path)
        with open(output / "alignment_noise_bank_manifest.json", "w", encoding="utf-8") as handle:
            json.dump({"path": str(Path(noise_bank_path).resolve()), "sha256": noise_bank_hash,
                       "seed": args.noise_seed, "replicates": args.noise_bank_replicates,
                       "domains": sorted(noise_bank.domains)}, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
