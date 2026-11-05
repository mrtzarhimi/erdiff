"""Aggregate an R150 cross-day run's saved checkpoints into a controlled-run reference."""

import argparse
import json
from pathlib import Path

import torch

from checkpoint_provenance import file_sha256, file_fingerprint
from cross_day import ALL_SESSIONS
from phase2_run_manifest import save_run_manifest


SESSION_FIELDS = {"target_neurons", "raw_target_channels", "model_input_width", "target_session"}


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def assemble_reference(source_checkpoint_path, source_checkpoint, source_data_path,
                       session_checkpoints, session_data_paths):
    """Validate and combine saved Phase 2 run manifests without inferring settings."""
    source_sha = file_sha256(source_checkpoint_path)
    source_data_fp = file_fingerprint(source_data_path)
    saved_source_fp = source_checkpoint.get("source_data_fingerprint")
    if not saved_source_fp or saved_source_fp.get("sha256") != source_data_fp["sha256"]:
        raise ValueError("Source data does not match the Phase 1 checkpoint fingerprint.")
    source_splits = jsonable(source_checkpoint.get("split_indices"))
    if not source_splits:
        raise ValueError("Phase 1 checkpoint lacks saved source split indices.")

    first_manifest = None
    first_config = None
    target_sessions = {}
    unresolved = set()
    for session, checkpoint_path in session_checkpoints.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        manifest = checkpoint.get("run_manifest")
        if not isinstance(manifest, dict):
            raise ValueError(f"{session} checkpoint has no embedded Phase 2 run manifest.")
        if manifest.get("source_checkpoint_sha256") != source_sha:
            raise ValueError(f"{session} was trained from a different Phase 1 checkpoint.")
        fingerprints = manifest.get("data_fingerprints", {})
        target_fp = fingerprints.get("target")
        actual_target_fp = file_fingerprint(session_data_paths[session])
        if not target_fp or target_fp.get("sha256") != actual_target_fp["sha256"]:
            raise ValueError(f"Current target data fingerprint differs for {session}.")
        target_split = jsonable(manifest.get("target_split_indices", manifest.get("split_indices")))
        if not target_split:
            raise ValueError(f"{session} checkpoint has no saved target split indices.")
        preprocessing = jsonable(manifest.get("preprocessing"))
        if not preprocessing:
            raise ValueError(f"{session} checkpoint has no preprocessing metadata.")
        config = jsonable(manifest.get("resolved_config"))
        if not isinstance(config, dict):
            raise ValueError(f"{session} checkpoint has no resolved Phase 2 configuration.")
        if config.get("epochs") != 150 or config.get("seed") != 42 or config.get("split_seed") != 42:
            raise ValueError(f"{session} is not a 150-epoch seed-42/split-seed-42 reference run.")
        if first_config is None:
            first_config = config
            first_manifest = manifest
        else:
            first_core = {key: value for key, value in first_config.items() if key not in SESSION_FIELDS}
            current_core = {key: value for key, value in config.items() if key not in SESSION_FIELDS}
            if current_core != first_core:
                raise ValueError(f"Resolved Phase 2 settings differ for {session}; cannot form one R150 reference.")
            if manifest.get("code_revision") != first_manifest.get("code_revision"):
                raise ValueError("Reference session checkpoints come from different code revisions.")
            if manifest.get("source_split_indices") != first_manifest.get("source_split_indices"):
                raise ValueError("Reference sessions do not share the same source split.")
        unresolved.update(manifest.get("unresolved", []))
        target_sessions[session] = {
            "target_split_indices": target_split,
            "data_fingerprint": actual_target_fp,
            "preprocessing": preprocessing,
            "resolved_config": {
                key: config[key] for key in SESSION_FIELDS if key in config
            },
            "checkpoint": str(Path(checkpoint_path).resolve()),
        }

    if not target_sessions or first_manifest is None or first_config is None:
        raise ValueError("No session checkpoints were supplied.")
    if jsonable(first_manifest.get("source_split_indices")) != source_splits:
        raise ValueError("Phase 2 source split differs from the Phase 1 checkpoint split.")
    source_in_manifest = first_manifest.get("data_fingerprints", {}).get("source")
    if not source_in_manifest or source_in_manifest.get("sha256") != source_data_fp["sha256"]:
        raise ValueError("Phase 2 source data fingerprint differs from the supplied source data.")

    first_target = next(iter(target_sessions.values()))
    reference = {
        "schema_version": 1,
        "reference_id": "R150-seed42-split42",
        "resolved_config": first_config,
        "allowed_factor_overrides": [],
        "unresolved": sorted(unresolved),
        "exact_reproduction_claimable": not unresolved,
        "source_checkpoint": str(Path(source_checkpoint_path).resolve()),
        "source_checkpoint_sha256": source_sha,
        "source_data_path": str(Path(source_data_path).resolve()),
        "source_file": Path(source_data_path).name,
        "data_fingerprints": {"source": source_data_fp,
                              "target": first_target["data_fingerprint"]},
        "source_split_indices": source_splits,
        "target_split_indices": first_target["target_split_indices"],
        "split_indices": first_target["target_split_indices"],
        "preprocessing": first_target["preprocessing"],
        "source_preprocessing": jsonable(source_checkpoint.get("preprocessing")),
        "code_revision": first_manifest.get("code_revision"),
        "target_sessions": target_sessions,
    }
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--source_data_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoints_dir", required=True,
                        help="Directory containing phase2_aligned_<session>.pt from the R150 run")
    parser.add_argument("--source_file", required=True)
    parser.add_argument("--target_sessions", nargs="*", default=None)
    parser.add_argument("--output", default="./reference_r150.json")
    args = parser.parse_args()

    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite existing reference manifest: {args.output}")
    source_checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    sessions = args.target_sessions or [name for name in ALL_SESSIONS if name != args.source_file]
    checkpoint_paths = {
        session: Path(args.checkpoints_dir) / f"phase2_aligned_{session.removesuffix('.mat')}.pt"
        for session in sessions
    }
    data_paths = {session: Path(args.data_dir) / session for session in sessions}
    missing = [str(path) for path in [*checkpoint_paths.values(), *data_paths.values()]
               if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing reference inputs: " + ", ".join(missing))
    reference = assemble_reference(
        args.source_checkpoint, source_checkpoint, args.source_data_path,
        checkpoint_paths, data_paths,
    )
    save_run_manifest(args.output, reference)
    print(f"Wrote R150 cross-day reference manifest: {args.output}")
    print(f"Sessions: {len(reference['target_sessions'])} | unresolved fields: {len(reference['unresolved'])}")


if __name__ == "__main__":
    main()
