"""Resolve immutable, budget-matched Phase 2 ablations and launch their protocols.

This orchestration module intentionally does not inspect target labels. Cross-day
cases require a reference artifact containing a saved split and fingerprint for
every target session; missing provenance is an error, never a new random split.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from phase2_run_manifest import load_reference, save_run_manifest


CASES = ("reference", "control", "fixed_noise", "matched_size", "gradient_balance", "all")
PROTOCOLS = ("source_self", "synthetic", "cross_day")
BASE_CONTROL = {
    "mla_variance_weight": 10.0,
    "mla_variance_floor": 0.9,
    "mla_variance_scope": "full_train",
    "alignment_noise_mode": "resampled",
    "match_input_size": False,
    "gradient_balance": "none",
    "log_component_gradients": False,
    "frozen_check": "every_epoch",
}
FACTOR_CHANGES = {
    "reference": {},
    "control": dict(BASE_CONTROL),
    "fixed_noise": {**BASE_CONTROL, "alignment_noise_mode": "fixed_bank"},
    "matched_size": {**BASE_CONTROL, "match_input_size": True},
    "gradient_balance": {**BASE_CONTROL, "gradient_balance": "source_calibrated",
                         "log_component_gradients": True},
    "all": {**BASE_CONTROL, "alignment_noise_mode": "fixed_bank",
            "match_input_size": True, "gradient_balance": "source_calibrated",
            "log_component_gradients": True},
}


def case_directory(root, protocol, case):
    return Path(root) / protocol / case


def ensure_new_case_directory(path):
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing controlled run: {path}") from exc
    return path


def validate_saved_split(split_artifact, label="target"):
    required = ("train", "validation", "test", "window_intervals")
    if not isinstance(split_artifact, dict) or any(key not in split_artifact for key in required):
        raise ValueError(f"Saved {label} split must include train/validation/test indices and window_intervals.")
    partitions = [set(int(index) for index in split_artifact[name])
                  for name in ("train", "validation", "test")]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError(f"Saved {label} trial partitions overlap.")
    intervals = split_artifact["window_intervals"]
    if not isinstance(intervals, (tuple, list)):
        raise ValueError(f"Saved {label} window intervals must be an indexed list.")
    for partition in partitions:
        if any(index < 0 or index >= len(intervals) for index in partition):
            raise ValueError(f"Saved {label} split index is outside the window interval table.")
    occupied = []
    for name, partition in zip(("train", "validation", "test"), partitions):
        for index in partition:
            start, stop = intervals[index]
            occupied.append((int(start), int(stop), name, index))
    occupied.sort()
    max_stop_by_partition = {}
    for start, stop, name, _ in occupied:
        if any(other != name and prior_stop > start
               for other, prior_stop in max_stop_by_partition.items()):
            raise ValueError(f"Saved {label} raw time windows overlap across partitions.")
        max_stop_by_partition[name] = max(max_stop_by_partition.get(name, stop), stop)


def resolve_case_config(reference, protocol, case):
    if protocol not in PROTOCOLS or case not in CASES:
        raise ValueError("Unknown controlled Phase 2 protocol or case.")
    base = reference.get("resolved_config")
    if not isinstance(base, dict):
        raise ValueError("Reference must contain a resolved_config object.")
    if base.get("seed") != 42 or base.get("split_seed") != 42:
        raise ValueError("Controlled comparisons require reference model and split seed 42.")
    if base.get("epochs") != 150:
        raise ValueError("Controlled comparisons require a matched 150-epoch reference budget.")
    for key in ("noise_seed", "input_mapping_seed", "gradient_calibration_seed", "shift_seed"):
        if key in base and base[key] != 42:
            raise ValueError(f"Controlled comparisons require {key}=42 in the reference.")
    if base.get("noise_bank_replicates", 8) != 8:
        raise ValueError("Controlled fixed-noise comparisons require the reference bank size of 8 replicates.")
    factors = dict(FACTOR_CHANGES[case])
    config = dict(base)
    config.update(factors)
    # Every case keeps the reference optimization budget and source objective.
    config["seed"] = 42
    config["split_seed"] = 42
    config["epochs"] = 150
    config.setdefault("input_mapping_seed", 42)
    config.setdefault("noise_seed", 42)
    config.setdefault("noise_bank_replicates", 8)
    config.setdefault("gradient_calibration_seed", 42)
    config.setdefault("gradient_calibration_batches", 3)
    return {
        **copy.deepcopy(reference),
        "resolved_config": config,
        "case": case,
        "protocol": protocol,
        "factor_changes": factors,
        "reference_id": reference.get("reference_id"),
        "comparison_status": "known_config_replay" if reference.get("unresolved") else "verified_reference",
        "unresolved": list(reference.get("unresolved", [])),
        "exact_reproduction_claimable": not bool(reference.get("unresolved")),
        "budget": {"requested_epochs": 150, "updates": "recorded at runtime",
                   "sample_exposures": "cohort-specific; recorded at runtime"},
    }


def derive_source_self_reference(reference, controlled):
    source_splits = reference.get("source_split_indices")
    source_fingerprint = reference.get("data_fingerprints", {}).get("source")
    if not source_splits or not source_fingerprint:
        raise ValueError("Source-self requires saved source split indices and source data fingerprint.")
    validate_saved_split(source_splits, "source")
    result = copy.deepcopy(controlled)
    result["target_split_indices"] = copy.deepcopy(source_splits)
    result["split_indices"] = copy.deepcopy(source_splits)
    result.setdefault("data_fingerprints", {})["target"] = copy.deepcopy(source_fingerprint)
    result["resolved_config"]["source_self_check"] = True
    result["resolved_config"]["target_neurons"] = result["resolved_config"].get("source_neurons")
    result["resolved_config"]["model_input_width"] = result["resolved_config"].get("source_neurons")
    result["preprocessing"] = copy.deepcopy(reference.get("source_preprocessing", reference.get("preprocessing", {})))
    return result


def derive_session_reference(reference, controlled, session, artifact, actual_fingerprint):
    if not isinstance(artifact, dict) or not artifact.get("target_split_indices"):
        raise ValueError(f"Missing saved per-session split for {session}; refusing to create one.")
    validate_saved_split(artifact["target_split_indices"], f"target session {session}")
    saved_fingerprint = artifact.get("data_fingerprint")
    if not saved_fingerprint or saved_fingerprint.get("sha256") != actual_fingerprint:
        raise ValueError(f"Target data fingerprint mismatch for {session}.")
    preprocessing = artifact.get("preprocessing", reference.get("preprocessing"))
    if not preprocessing:
        raise ValueError(f"Reference lacks preprocessing provenance for {session}.")
    result = copy.deepcopy(controlled)
    result["target_split_indices"] = copy.deepcopy(artifact["target_split_indices"])
    result["split_indices"] = copy.deepcopy(artifact["target_split_indices"])
    result.setdefault("data_fingerprints", {})["target"] = copy.deepcopy(saved_fingerprint)
    session_config = artifact.get("resolved_config", {})
    result["resolved_config"].update(controlled["resolved_config"])
    if "target_neurons" in session_config:
        result["resolved_config"]["target_neurons"] = session_config["target_neurons"]
    elif "raw_target_channels" in session_config:
        result["resolved_config"]["target_neurons"] = session_config["raw_target_channels"]
    result["resolved_config"]["raw_target_channels"] = result["resolved_config"].get("target_neurons")
    if result["resolved_config"].get("match_input_size"):
        result["resolved_config"]["model_input_width"] = result["resolved_config"].get("source_neurons")
    else:
        result["resolved_config"]["model_input_width"] = result["resolved_config"].get("target_neurons")
    result["resolved_config"]["target_session"] = session
    result["preprocessing"] = copy.deepcopy(preprocessing)
    result["session_reference"] = session
    return result


def _json_dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_phase2_flags(config, prefix=""):
    flags = []
    pairs = {
        "seed": "--seed", "split_seed": "--split_seed", "epochs": "--epochs",
        "seq_len": "--seq_len", "batch_size": "--batch_size", "lr": "--lr",
        "alpha": "--alpha", "ot_epsilon": "--ot_epsilon",
        "mla_variance_weight": "--mla_variance_weight",
        "mla_variance_floor": "--mla_variance_floor",
        "mla_variance_scope": "--mla_variance_scope",
        "alignment_noise_mode": "--alignment_noise_mode", "noise_seed": "--noise_seed",
        "noise_bank_replicates": "--noise_bank_replicates",
        "match_input_size": "--match_input_size", "input_mapping_seed": "--input_mapping_seed",
        "gradient_balance": "--gradient_balance", "frozen_check": "--frozen_check",
        "readin_init": "--readin_init", "smooth_weight": "--smooth_weight",
        "weight_decay": "--weight_decay", "augment_mult": "--augment_mult",
        "gradient_calibration_batches": "--gradient_calibration_batches",
        "gradient_calibration_seed": "--gradient_calibration_seed",
        "sinkhorn_iters": "--sinkhorn_iters",
        "mla_t_min": "--mla_t_min", "mla_t_max": "--mla_t_max",
        "mla_fixed_t": "--mla_fixed_t",
    }
    for key, flag in pairs.items():
        if key not in config:
            continue
        value = config[key]
        if value is None:
            continue
        full = f"{prefix}{flag}"
        if key == "match_input_size":
            if value:
                flags.append(full)
        else:
            flags.extend((full, str(value)))
    if config.get("log_component_gradients"):
        flags.append(f"{prefix}--log_component_gradients")
    if config.get("augment_target"):
        flags.append(f"{prefix}--augment_target")
    return flags


def _append_config(command, config, prefix=""):
    command.extend(build_phase2_flags(config, prefix))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_run", required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_checkpoint", default=None)
    parser.add_argument("--source_data_path", default=None)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--source_file", default=None)
    parser.add_argument("--target_sessions", nargs="*", default=None)
    args = parser.parse_args()

    reference = load_reference(args.reference_run)
    # Every protocol, including the synthetic launcher, must reject a reference
    # from the former objective rather than silently relabeling it as a replay.
    from diffusion import transport_configuration
    if reference["resolved_config"].get("transport") != transport_configuration():
        raise ValueError("Reference transport objective or POT version differs; generate a new reference run.")
    resolved = resolve_case_config(reference, args.protocol, args.case)
    if args.protocol == "synthetic" and resolved["resolved_config"].get("augment_target"):
        raise ValueError("Source-synthetic cases require unique clean adaptation trials; target augmentation is unsupported.")
    out = ensure_new_case_directory(case_directory(args.output_root, args.protocol, args.case))
    manifest_path = out / "resolved_run_manifest.json"
    save_run_manifest(str(manifest_path), resolved)

    config = resolved["resolved_config"]
    source_checkpoint = args.source_checkpoint or reference.get("source_checkpoint")
    source_data = args.source_data_path or reference.get("source_data_path")
    if not source_checkpoint or not Path(source_checkpoint).is_file():
        raise FileNotFoundError("A specified Phase 1 source checkpoint is required; orchestration will not train Phase 1.")
    if not source_data:
        raise ValueError("Supply --source_data_path or include it in the reference manifest.")
    source_file = args.source_file or reference.get("source_file") or Path(source_data).name

    if args.protocol in ("source_self", "synthetic"):
        shared_protocol_root = Path(args.output_root) / args.protocol
        child_reference = (derive_source_self_reference(reference, resolved)
                           if args.protocol == "source_self" else resolved)
        child_ref_path = out / "protocol_reference.json"
        _json_dump(child_ref_path, child_reference)
        if args.protocol == "source_self":
            command = [sys.executable, "train_phase2.py", "--source_self_check",
                       "--source_checkpoint", str(source_checkpoint), "--source_data_path", source_data,
                       "--target_data_path", source_data, "--save_dir", str(out),
                       "--reference_run", str(child_ref_path)]
            _append_config(command, config)
            if config.get("alignment_noise_mode") == "fixed_bank":
                command.extend(("--noise_bank_path", str(shared_protocol_root / "shared_noise_bank.pt")))
            if config.get("gradient_balance") == "source_calibrated":
                command.extend(("--gradient_calibration_path", str(shared_protocol_root / "shared_source_gradient_calibration.json")))
        else:
            command = [sys.executable, "source_synthetic_checks.py",
                       "--source_checkpoint", str(source_checkpoint), "--source_data_path", source_data,
                       "--output_dir", str(out), "--weights", str(config.get("mla_variance_weight", 10.0)),
                       "--epochs", str(config["epochs"]), "--seq_len", str(config["seq_len"]),
                       "--batch_size", str(config["batch_size"]),
                       "--lr", str(config["lr"]), "--mla_variance_floor", str(config.get("mla_variance_floor", 0.9)),
                       "--weight_decay", str(config.get("weight_decay", 0.0)),
                       "--mla_variance_scope", str(config.get("mla_variance_scope", "full_train")),
                       "--seed", "42", "--split_seed", "42", "--shift_seed", "42",
                       "--alpha", str(config.get("alpha", 0.0)), "--ot_epsilon", str(config.get("ot_epsilon", 0.01)),
                       "--sinkhorn_iters", str(config.get("sinkhorn_iters", 1000)),
                       "--smooth_weight", str(config.get("smooth_weight", 0.0)),
                       "--gradient_balance", str(config.get("gradient_balance", "none")),
                       "--gradient_calibration_batches", str(config.get("gradient_calibration_batches", 3)),
                       "--gradient_calibration_seed", str(config.get("gradient_calibration_seed", 42)),
                       "--frozen_check", str(config.get("frozen_check", "every_epoch"))]
            for key in ("mla_t_min", "mla_t_max", "mla_fixed_t"):
                if config.get(key) is not None:
                    command.extend((f"--{key}", str(config[key])))
            if config.get("log_component_gradients"):
                command.append("--log_component_gradients")
            if config.get("alignment_noise_mode") == "fixed_bank":
                command.extend(("--noise_bank_path", str(shared_protocol_root / "shared_synthetic_noise_bank.pt")))
            if config.get("gradient_balance") == "source_calibrated":
                command.extend(("--gradient_calibration_path", str(shared_protocol_root / "shared_synthetic_gradient_calibration.json")))
            if config.get("alignment_noise_mode") == "fixed_bank":
                command.extend(("--alignment_noise_mode", "fixed_bank", "--noise_seed", "42",
                                "--noise_bank_replicates", str(config.get("noise_bank_replicates", 8))))
            if config.get("match_input_size"):
                command.extend(("--match_input_size", "--input_mapping_seed", "42"))
        _json_dump(out / "launch_command.json", {"command": command, "manifest": str(manifest_path)})
        subprocess.run(command, check=True)
        return

    sessions = args.target_sessions or reference.get("target_sessions", {}).keys()
    if not sessions:
        raise ValueError("Cross-day requires --target_sessions and a saved per-session target split map in the reference.")
    artifacts = reference.get("target_sessions", {})
    missing = [session for session in sessions if session not in artifacts]
    if missing:
        raise ValueError(f"Reference lacks saved per-session split artifacts for: {missing}")
    # Cross-day replay must know each source/target fingerprint; train_phase2 will
    # verify those fingerprints before optimization when it reads the reference.
    for session in sessions:
        if not artifacts[session].get("data_fingerprint", {}).get("sha256"):
            raise ValueError(f"Reference lacks target fingerprint for {session}.")
    command = [sys.executable, "cross_day.py", "--data_dir", args.data_dir,
               "--source_file", source_file, "--source_data_path", source_data,
               "--phase1_checkpoint", str(source_checkpoint),
               "--phase2_epochs", "150", "--controlled_reference", str(args.reference_run),
               "--controlled_case", args.case, "--controlled_output_root", str(out),
               "--target_sessions", *sessions,
               "--seed", "42", "--split_seed", "42", "--phase2_lr", str(config["lr"]),
               "--seq_len", str(config["seq_len"]), "--batch_size", str(config["batch_size"]),
               "--phase2_weight_decay", str(config.get("weight_decay", 0.0)),
               "--phase2_augment_mult", str(config.get("augment_mult", 3)),
               "--phase2_sinkhorn_iters", str(config.get("sinkhorn_iters", 1000)),
               "--phase2_gradient_calibration_batches", str(config.get("gradient_calibration_batches", 3)),
               "--phase2_alpha", str(config.get("alpha", 0.5)),
               "--phase2_mla_variance_weight", str(config.get("mla_variance_weight", 10.0)),
               "--phase2_mla_variance_floor", str(config.get("mla_variance_floor", 0.9)),
               "--phase2_mla_variance_scope", str(config.get("mla_variance_scope", "full_train")),
               "--phase2_noise_mode", str(config.get("alignment_noise_mode", "resampled")),
               "--phase2_gradient_balance", str(config.get("gradient_balance", "none")),
               "--phase2_readin_init", str(config.get("readin_init", "source_or_random")),
               "--phase2_smooth_weight", str(config.get("smooth_weight", 0.0)),
               "--frozen_check", str(config.get("frozen_check", "every_epoch"))]
    if config.get("log_component_gradients"):
        command.append("--log_component_gradients")
    if config.get("match_input_size"):
        command.extend(("--phase2_match_input_size", "--input_mapping_seed", "42"))
    if config.get("augment_target"):
        command.append("--phase2_augment_target")
    if config.get("ot_epsilon") is not None:
        command.extend(("--phase2_ot_epsilon", str(config["ot_epsilon"])))
    _json_dump(out / "launch_command.json", {"command": command, "manifest": str(manifest_path)})
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
