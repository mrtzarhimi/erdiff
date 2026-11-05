from pathlib import Path

import torch

from build_cross_day_reference import assemble_reference
from checkpoint_provenance import file_fingerprint, file_sha256


def test_aggregates_session_manifests_and_keeps_per_session_splits(tmp_path, monkeypatch):
    source_path = tmp_path / "source.mat"
    source_path.write_bytes(b"source data")
    source_checkpoint_path = tmp_path / "phase1.pt"
    source_checkpoint_path.write_bytes(b"phase1 checkpoint")
    source_split = {
        "train": [0], "validation": [1], "test": [2],
        "window_intervals": [[0, 39], [100, 139], [200, 239]],
    }
    source_checkpoint = {
        "source_data_fingerprint": file_fingerprint(source_path),
        "split_indices": source_split,
        "preprocessing": {"bin_width_s": 0.02},
    }
    config = {"seed": 42, "split_seed": 42, "epochs": 150,
              "source_neurons": 96, "target_neurons": 95,
              "model_input_width": 95, "lr": 0.001}
    loaded = {}
    session_paths = {}
    data_paths = {}
    for day, width in (("day1.mat", 95), ("day2.mat", 94)):
        checkpoint_path = tmp_path / f"{day}.pt"
        checkpoint_path.write_bytes(day.encode())
        data_path = tmp_path / day
        data_path.write_bytes((day + " data").encode())
        session_config = {**config, "target_neurons": width, "model_input_width": width}
        target_split = {
            "train": [0], "validation": [1], "test": [2],
            "window_intervals": [[0, 39], [100, 139], [200, 239]],
        }
        loaded[str(checkpoint_path)] = {
            "run_manifest": {
                "source_checkpoint_sha256": file_sha256(source_checkpoint_path),
                "source_split_indices": source_split,
                "data_fingerprints": {
                    "source": file_fingerprint(source_path),
                    "target": file_fingerprint(data_path),
                },
                "target_split_indices": target_split,
                "preprocessing": {"bin_width_s": 0.02},
                "resolved_config": session_config,
                "code_revision": "revision",
                "unresolved": [],
            }
        }
        session_paths[day] = checkpoint_path
        data_paths[day] = data_path

    monkeypatch.setattr(torch, "load", lambda path, **kwargs: loaded[str(path)])
    manifest = assemble_reference(
        source_checkpoint_path, source_checkpoint, source_path, session_paths, data_paths
    )

    assert manifest["resolved_config"]["epochs"] == 150
    assert manifest["target_sessions"]["day1.mat"]["target_split_indices"]["test"] == [2]
    assert manifest["target_sessions"]["day2.mat"]["resolved_config"]["target_neurons"] == 94
