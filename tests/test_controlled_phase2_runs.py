import copy

import pytest

from controlled_phase2_runs import (
    case_directory,
    derive_session_reference,
    derive_source_self_reference,
    ensure_new_case_directory,
    build_phase2_flags,
    validate_saved_split,
    resolve_case_config,
)


def _reference():
    source_splits = {
        "train": [0, 1], "validation": [2], "test": [3],
        "window_intervals": [[0, 39], [100, 139], [200, 239], [300, 339]],
    }
    target_splits = {
        "train": [4, 5], "validation": [6], "test": [7],
        "window_intervals": [[0, 39], [100, 139], [200, 239], [300, 339],
                             [400, 439], [500, 539], [600, 639], [700, 739]],
    }
    return {
        "schema_version": 1,
        "reference_id": "reference-r150",
        "resolved_config": {
            "seed": 42, "split_seed": 42, "epochs": 150, "seq_len": 39,
            "batch_size": 64, "lr": 0.001, "weight_decay": 0.0, "alpha": 0.5,
            "ot_epsilon": 31.2, "sinkhorn_iters": 100, "smooth_weight": 0.0,
            "mla_variance_weight": 0.0, "mla_variance_floor": 0.9,
            "mla_variance_scope": "batch", "alignment_noise_mode": "resampled",
            "noise_seed": 42, "noise_bank_replicates": 8,
            "match_input_size": False, "input_mapping_seed": 42,
            "readin_init": "source_or_random", "gradient_balance": "none",
            "log_component_gradients": False, "frozen_check": "every_epoch",
            "augment_target": False, "augment_mult": 3,
            "source_neurons": 96, "target_neurons": 95,
        },
        "allowed_factor_overrides": [], "unresolved": [],
        "source_checkpoint": "/models/phase1.pt",
        "source_checkpoint_sha256": "model-hash",
        "data_fingerprints": {
            "source": {"sha256": "source-hash"},
            "target": {"sha256": "target-hash"},
        },
        "source_split_indices": source_splits,
        "target_split_indices": target_splits,
        "split_indices": target_splits,
        "preprocessing": {"window_anchor": "go_cue", "bin_width_s": 0.02},
    }


def test_case_matrix_preserves_budget_and_changes_only_declared_factors():
    reference = _reference()
    expected = {
        "reference": {},
        "control": {"mla_variance_weight": 10.0, "mla_variance_floor": 0.9,
                    "mla_variance_scope": "full_train"},
        "fixed_noise": {"mla_variance_weight": 10.0, "mla_variance_floor": 0.9,
                        "mla_variance_scope": "full_train", "alignment_noise_mode": "fixed_bank"},
        "matched_size": {"mla_variance_weight": 10.0, "mla_variance_floor": 0.9,
                         "mla_variance_scope": "full_train", "match_input_size": True},
        "gradient_balance": {"mla_variance_weight": 10.0, "mla_variance_floor": 0.9,
                             "mla_variance_scope": "full_train", "gradient_balance": "source_calibrated"},
        "all": {"mla_variance_weight": 10.0, "mla_variance_floor": 0.9,
                "mla_variance_scope": "full_train", "alignment_noise_mode": "fixed_bank",
                "match_input_size": True, "gradient_balance": "source_calibrated"},
    }

    for case, factors in expected.items():
        resolved = resolve_case_config(reference, "source_self", case)
        config = resolved["resolved_config"]
        assert config["epochs"] == 150
        assert config["seed"] == config["split_seed"] == 42
        assert config["lr"] == reference["resolved_config"]["lr"]
        assert config["alpha"] == reference["resolved_config"]["alpha"]
        assert resolved["factor_changes"] == factors
        for key, value in factors.items():
            assert config[key] == value
        for key, value in reference["resolved_config"].items():
            if key not in factors and key not in ("target_neurons",):
                assert config[key] == value


def test_source_self_reference_restores_saved_source_splits_and_identity_data():
    reference = _reference()
    controlled = resolve_case_config(reference, "source_self", "control")

    self_reference = derive_source_self_reference(reference, controlled)

    assert self_reference["target_split_indices"] == reference["source_split_indices"]
    assert self_reference["split_indices"] == reference["source_split_indices"]
    assert self_reference["data_fingerprints"]["target"] == reference["data_fingerprints"]["source"]
    assert self_reference["resolved_config"]["source_self_check"] is True
    assert self_reference["resolved_config"]["target_neurons"] == 96


def test_session_reference_requires_matching_saved_test_split_and_fingerprint():
    reference = _reference()
    controlled = resolve_case_config(reference, "cross_day", "all")
    artifact = {
        "data_fingerprint": {"sha256": "day-2-hash"},
        "target_split_indices": {"train": [0], "validation": [1], "test": [2],
                                 "window_intervals": [[0, 39], [100, 139], [200, 239]]},
        "resolved_config": {"target_neurons": 94},
    }

    derived = derive_session_reference(reference, controlled, "day2.mat", artifact, "day-2-hash")

    assert derived["target_split_indices"] == artifact["target_split_indices"]
    assert derived["data_fingerprints"]["target"] == artifact["data_fingerprint"]
    assert derived["resolved_config"]["target_neurons"] == 94
    assert derived["resolved_config"]["match_input_size"] is True
    assert derived["resolved_config"]["model_input_width"] == 96
    assert derived["resolved_config"]["epochs"] == 150
    with pytest.raises(ValueError, match="fingerprint"):
        derive_session_reference(reference, controlled, "day2.mat", artifact, "different-hash")


def test_cross_day_reference_cannot_fall_back_to_another_sessions_split():
    reference = _reference()
    controlled = resolve_case_config(reference, "cross_day", "all")
    with pytest.raises(ValueError, match="saved per-session split"):
        derive_session_reference(reference, controlled, "day2.mat", {}, "day-2-hash")


def test_saved_partitions_reject_raw_time_overlap():
    split = {"train": [0], "validation": [1], "test": [2],
             "window_intervals": [[0, 40], [39, 79], [100, 140]]}
    with pytest.raises(ValueError, match="raw time windows overlap"):
        validate_saved_split(split)
    nested = {"train": [0, 1], "validation": [2], "test": [3],
              "window_intervals": [[0, 100], [10, 20], [90, 110], [120, 140]]}
    with pytest.raises(ValueError, match="raw time windows overlap"):
        validate_saved_split(nested)


def test_case_directories_are_unique_and_existing_runs_are_not_overwritten(tmp_path):
    reference = _reference()
    first = case_directory(tmp_path, "synthetic", "control")
    second = case_directory(tmp_path, "synthetic", "all")

    assert first != second
    ensure_new_case_directory(first)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ensure_new_case_directory(first)


def test_unmatched_seed_or_non_150_reference_is_rejected():
    reference = _reference()
    reference["resolved_config"]["split_seed"] = 7
    with pytest.raises(ValueError, match="seed 42"):
        resolve_case_config(reference, "source_self", "control")

    reference = copy.deepcopy(_reference())
    reference["resolved_config"]["epochs"] = 50
    with pytest.raises(ValueError, match="150-epoch"):
        resolve_case_config(reference, "source_self", "control")


def test_all_case_forwards_noise_mapping_gradient_and_freeze_options():
    reference = _reference()
    config = resolve_case_config(reference, "cross_day", "all")["resolved_config"]
    flags = build_phase2_flags(config)
    assert "--match_input_size" in flags
    assert "--alignment_noise_mode" in flags
    assert flags[flags.index("--alignment_noise_mode") + 1] == "fixed_bank"
    assert "--gradient_balance" in flags
    assert flags[flags.index("--gradient_balance") + 1] == "source_calibrated"
    assert "--frozen_check" in flags
    assert "--log_component_gradients" in flags
