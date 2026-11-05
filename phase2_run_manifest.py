"""Load and write explicit Phase 2 run manifests without inferring legacy settings."""

import json
import os
from collections.abc import Mapping

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
    if not isinstance(manifest.get("allowed_factor_overrides", []), list):
        raise ValueError("allowed_factor_overrides must be a list of configuration keys.")
    return manifest


def resolve_run(reference: dict, factor_overrides: dict) -> dict:
    """Apply only explicitly declared factors; reject accidental hyperparameter drift."""
    if not isinstance(reference.get("resolved_config"), dict):
        raise ValueError("Reference manifest must contain resolved_config.")
    if not isinstance(reference.get("unresolved"), list):
        raise ValueError("Reference manifest must list unresolved historical fields.")
    if not isinstance(factor_overrides, Mapping):
        raise TypeError("factor_overrides must be a mapping.")
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


def _to_jsonable(value):
    """Convert checkpoint-style tensors and nested containers into JSON values."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_reference_compatibility(reference: dict, *, source_checkpoint_sha256: str,
                                      data_fingerprints: dict, split_indices: dict,
                                      preprocessing: dict,
                                      source_split_indices: dict | None = None) -> None:
    """Reject known source/data/split/window mismatches before reference replay."""
    expected_source = reference.get("source_checkpoint_sha256")
    if expected_source is not None and expected_source != source_checkpoint_sha256:
        raise ValueError("Reference source checkpoint fingerprint does not match.")

    expected_data = reference.get("data_fingerprints")
    if expected_data is not None and _to_jsonable(expected_data) != _to_jsonable(data_fingerprints):
        raise ValueError("Reference data fingerprints do not match the supplied data.")

    expected_splits = reference.get("target_split_indices", reference.get("split_indices"))
    if expected_splits is not None and _to_jsonable(expected_splits) != _to_jsonable(split_indices):
        raise ValueError("Saved split indices or raw-window intervals differ from the reference.")

    expected_source_splits = reference.get("source_split_indices")
    if (expected_source_splits is not None and source_split_indices is not None
            and _to_jsonable(expected_source_splits) != _to_jsonable(source_split_indices)):
        raise ValueError("Source trial split indices differ from the reference.")

    expected_preprocessing = reference.get("preprocessing")
    if expected_preprocessing is not None and _to_jsonable(expected_preprocessing) != _to_jsonable(preprocessing):
        raise ValueError("Raw-window or preprocessing configuration differs from the reference.")


def validate_resolved_config(reference: dict, runtime_config: dict) -> dict:
    """Resolve declared factors and reject runtime settings that drift from the result."""
    resolved = resolve_run(reference, runtime_config.get("factor_overrides", {}))
    expected = resolved["resolved_config"]
    unresolved_fields = set(reference.get("unresolved", []))
    missing = {
        key for key in runtime_config
        if key != "factor_overrides" and key not in expected and key not in unresolved_fields
    }
    if missing:
        raise ValueError(
            "Reference omits runtime settings without marking them unresolved: "
            f"{sorted(missing)}"
        )
    mismatches = {
        key: (expected[key], value)
        for key, value in runtime_config.items()
        if key != "factor_overrides" and key in expected and expected[key] != value
    }
    if mismatches:
        details = ", ".join(f"{key}: expected {want!r}, got {got!r}"
                             for key, (want, got) in sorted(mismatches.items()))
        raise ValueError(f"Runtime configuration differs from reference: {details}")
    return resolved


def exact_reproduction_claimable(manifest: dict) -> bool:
    """Unknown or absent historical provenance always blocks an exact claim."""
    required = (
        "source_checkpoint_sha256", "data_fingerprints", "split_indices",
        "target_split_indices", "source_split_indices", "preprocessing", "code_revision",
    )
    if manifest.get("unresolved") or not all(manifest.get(key) for key in required):
        return False
    fingerprints = manifest["data_fingerprints"]
    return all(fingerprints.get(domain, {}).get("sha256") for domain in ("source", "target"))


def build_run_manifest(config: dict, source_checkpoint: str, split_indices: dict,
                       data_fingerprints: dict, code_revision: str,
                       updates: int, sample_exposures: int,
                       full_variance_forwards: int = 0,
                       full_variance_sample_exposures: int = 0,
                       requested_epochs: int | None = None,
                       completed_epochs: int | None = None,
                       unresolved: list | None = None,
                       preprocessing: dict | None = None,
                       reference_id: str | None = None,
                       source_split_indices: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "resolved_config": _to_jsonable(dict(config)),
        "unresolved": list(unresolved or []),
        "reference_id": reference_id,
        "source_checkpoint": os.path.realpath(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "data_fingerprints": _to_jsonable(dict(data_fingerprints)),
        "split_indices": _to_jsonable(split_indices),
        "target_split_indices": _to_jsonable(split_indices),
        "source_split_indices": _to_jsonable(source_split_indices),
        "preprocessing": _to_jsonable(preprocessing or {}),
        "code_revision": code_revision,
        "requested_epochs": requested_epochs,
        "completed_epochs": completed_epochs,
        "completed_updates": int(updates),
        "sample_exposures": int(sample_exposures),
        "full_variance_forwards": int(full_variance_forwards),
        "full_variance_sample_exposures": int(full_variance_sample_exposures),
    }


def save_run_manifest(path: str, resolved: dict) -> None:
    parent = os.path.dirname(os.path.realpath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)
        handle.write("\n")
