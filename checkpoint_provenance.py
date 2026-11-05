"""Identify the exact frozen source model paired with an aligned read-in."""

import hashlib
import os
import subprocess


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_revision(cwd=None):
    """Return the checked-out Git revision, or None when provenance is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_fingerprint(path):
    """Record stable file identity without reading dataset contents into memory."""
    real_path = os.path.realpath(path)
    return {"path": real_path, "sha256": file_sha256(real_path)}


def verify_source_pair(source_path, aligned_checkpoint):
    expected = aligned_checkpoint.get("source_checkpoint_sha256")
    if expected is None:
        raise ValueError(
            "Aligned checkpoint lacks its source model fingerprint. Rerun Phase 2 "
            "to establish which source encoder/decoder it was trained with."
        )
    if file_sha256(source_path) != expected:
        raise ValueError(
            "Source checkpoint differs from the model used for alignment. "
            "Evaluate with the original source checkpoint or rerun Phase 2."
        )


def verify_data_fingerprint(data_path, expected, label="data"):
    if expected is None:
        raise ValueError(
            f"Checkpoint lacks a {label} fingerprint. Rerun training to establish "
            "which data file produced this result."
        )
    actual = file_fingerprint(data_path)
    if actual["sha256"] != expected.get("sha256"):
        raise ValueError(f"The {label} file differs from the file used to create this checkpoint.")


def verify_preprocessing(actual, expected):
    if expected is None:
        raise ValueError("Aligned checkpoint lacks preprocessing metadata; rerun Phase 2.")
    if actual != expected:
        raise ValueError("Evaluation preprocessing differs from the configuration used for alignment.")
