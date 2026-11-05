import pytest

from checkpoint_provenance import file_sha256, verify_source_pair


def test_evaluation_rejects_a_different_source_checkpoint(tmp_path):
    original = tmp_path / "original.pt"
    other = tmp_path / "other.pt"
    original.write_bytes(b"source model A")
    other.write_bytes(b"source model B")

    with pytest.raises(ValueError, match="differs from the model"):
        verify_source_pair(other, {"source_checkpoint_sha256": file_sha256(original)})


def test_source_pair_survives_moving_identical_checkpoint_bytes(tmp_path):
    original = tmp_path / "original.pt"
    moved = tmp_path / "moved.pt"
    original.write_bytes(b"same source model")
    moved.write_bytes(original.read_bytes())

    verify_source_pair(moved, {"source_checkpoint_sha256": file_sha256(original)})
