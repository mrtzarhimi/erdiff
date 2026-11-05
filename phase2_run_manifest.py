"""Load and write explicit Phase 2 run manifests without inferring legacy settings."""

import json
import os

from checkpoint_provenance import file_sha256


SCHEMA_VERSION = 1


def load_reference(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported or missing Phase 2 manifest schema_version.")
    if not isinstance(manifest.get("resolved_config"), dict):
        raise ValueError("Reference manifest must contain resolved_config.")
    if not isinstance(manifest.get("unresolved"), list):
        raise ValueError("Reference manifest must list unresolved historical fields.")
    return manifest


def resolve_run(reference: dict, factor_overrides: dict) -> dict:
    """Apply only explicitly declared factors; reject accidental hyperparameter drift."""
    resolved = dict(reference["resolved_config"])
    factors = set(reference.get("allowed_factor_overrides", []))
    unexpected = set(factor_overrides) - factors
    if unexpected:
        raise ValueError(f"Overrides are not declared experimental factors: {sorted(unexpected)}")
    resolved.update(factor_overrides)
    return {
        "schema_version": SCHEMA_VERSION,
        "resolved_config": resolved,
        "unresolved": list(reference.get("unresolved", [])),
        "reference_id": reference.get("reference_id"),
        "allowed_factor_overrides": sorted(factors),
    }


def build_run_manifest(config: dict, source_checkpoint: str, split_indices: dict,
                       data_fingerprints: dict, code_revision: str,
                       updates: int, sample_exposures: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "resolved_config": dict(config),
        "unresolved": [],
        "source_checkpoint": os.path.realpath(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "data_fingerprints": dict(data_fingerprints),
        "split_indices": split_indices,
        "code_revision": code_revision,
        "completed_updates": int(updates),
        "sample_exposures": int(sample_exposures),
    }


def save_run_manifest(path: str, resolved: dict) -> None:
    parent = os.path.dirname(os.path.realpath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)
        handle.write("\n")
