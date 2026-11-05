import pytest

from phase2_run_manifest import (
    exact_reproduction_claimable,
    resolve_run,
    validate_reference_compatibility,
)


def _reference():
    return {
        "schema_version": 1,
        "reference_id": "controlled-seed-42",
        "resolved_config": {"epochs": 150, "lr": 0.001},
        "allowed_factor_overrides": ["alignment_noise_mode"],
        "unresolved": [],
        "source_checkpoint_sha256": "source-hash",
        "data_fingerprints": {
            "source": {"sha256": "source-data"},
            "target": {"sha256": "target-data"},
        },
        "target_split_indices": {
            "train": [0, 1], "validation": [2], "test": [3],
            "window_intervals": [[0, 39], [100, 139], [200, 239], [300, 339]],
        },
        "preprocessing": {"window_anchor": "go_cue", "bin_width_s": 0.02},
    }


def test_rejects_undeclared_hyperparameter_override():
    with pytest.raises(ValueError, match="not declared"):
        resolve_run(_reference(), {"lr": 0.01})


def test_mismatched_source_or_split_fails():
    reference = _reference()
    common = {
        "data_fingerprints": reference["data_fingerprints"],
        "split_indices": reference["target_split_indices"],
        "preprocessing": reference["preprocessing"],
    }
    with pytest.raises(ValueError, match="source checkpoint"):
        validate_reference_compatibility(
            reference, source_checkpoint_sha256="wrong-source", **common
        )
    with pytest.raises(ValueError, match="split indices"):
        validate_reference_compatibility(
            reference,
            source_checkpoint_sha256="source-hash",
            **{**common, "split_indices": {"train": [1, 0]}},
        )


def test_missing_historical_fields_prevent_exact_reproduction_claim():
    manifest = {
        "source_checkpoint_sha256": "source-hash",
        "data_fingerprints": {"source": {"sha256": "source-data"}},
        "split_indices": {"train": [0], "validation": [1], "test": [2]},
        "target_split_indices": {"train": [0], "validation": [1], "test": [2]},
        "source_split_indices": {"train": [0], "validation": [1], "test": [2]},
        "preprocessing": {"window_anchor": "go_cue"},
        "code_revision": "abc123",
        "unresolved": ["historical_sampler_unknown"],
    }
    assert exact_reproduction_claimable(manifest) is False
    manifest["unresolved"] = []
    assert exact_reproduction_claimable(manifest) is True
