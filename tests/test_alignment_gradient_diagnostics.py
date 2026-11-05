import torch
import pytest

from alignment_gradient_diagnostics import (
    calibrate_source_gradient_scale,
    component_gradient_report,
    load_gradient_calibration,
    save_gradient_calibration,
)


def test_component_report_finds_aligned_opposed_and_zero_gradients_without_mutation():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    rng_before = torch.random.get_rng_state().clone()
    aligned = parameter.square().sum()
    opposed = -(parameter.square().sum())
    zero = parameter.detach().sum()

    report = component_gradient_report(
        {"aligned": aligned, "opposed": opposed, "zero": zero},
        (parameter,), {"aligned": 0.5, "opposed": 0.25},
    )

    assert report["cosines"]["aligned:opposed"] == pytest.approx(-1.0)
    assert report["components"]["aligned"]["norm"] == pytest.approx(2 * torch.linalg.vector_norm(parameter).item())
    assert report["components"]["aligned"]["weighted_norm"] == pytest.approx(
        0.5 * report["components"]["aligned"]["norm"]
    )
    assert report["components"]["zero"]["norm"] == 0.0
    assert report["cosines"]["aligned:zero"] is None
    assert parameter.grad is None
    assert optimizer.state == {}
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_source_calibration_uses_weighted_rms_norms_and_clips_factor():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    rng_before = torch.random.get_rng_state().clone()
    probes = [
        {"mla": 2.0 * parameter, "ot": 10.0 * parameter},
        {"mla": 4.0 * parameter, "ot": 20.0 * parameter},
    ]

    result = calibrate_source_gradient_scale(
        probes, (parameter,), alpha=0.25, minimum=0.01, maximum=100.0,
        metadata={"variance_gradient": "inactive"},
    )

    # RMS weighted magnitudes are 0.75*sqrt((2^2+4^2)/2) and
    # 0.25*sqrt((10^2+20^2)/2), so the bounded OT multiplier is 0.6.
    assert result["valid"] is True
    assert result["c_mla"] == 1.0
    assert result["c_ot"] == pytest.approx(0.6)
    assert result["metadata"]["variance_gradient"] == "inactive"
    assert result["variance_scaled"] is False
    assert parameter.grad is None
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_calibration_rejects_inactive_or_nonfinite_components():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    inactive = calibrate_source_gradient_scale(
        [{"mla": parameter * 0.0, "ot": parameter}], (parameter,), alpha=0.5
    )
    nonfinite = calibrate_source_gradient_scale(
        [{"mla": parameter * torch.tensor(float("nan")), "ot": parameter}],
        (parameter,), alpha=0.5,
    )

    assert inactive["valid"] is False
    assert inactive["c_ot"] is None
    assert nonfinite["valid"] is False
    assert nonfinite["c_ot"] is None


def test_inactive_variance_is_not_amplified_and_active_variance_uses_mla_branch_coefficient():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    probe = {"mla": 2.0 * parameter, "ot": 4.0 * parameter,
             "variance": 100.0 * parameter}

    inactive = calibrate_source_gradient_scale(
        [probe], (parameter,), alpha=0.5, mla_variance_weight=0.0
    )
    active = calibrate_source_gradient_scale(
        [probe], (parameter,), alpha=0.5, mla_variance_weight=0.1
    )

    assert inactive["c_ot"] == pytest.approx(0.5)
    assert active["c_ot"] == pytest.approx(3.0)


def test_saved_source_calibration_reuses_exact_factors_and_checks_provenance(tmp_path):
    expected = {
        "source_checkpoint_sha256": "source-hash",
        "alpha": 0.5,
        "ot_epsilon": 12.0,
        "sinkhorn_iters": 100,
        "diffusion_schedule": {"beta_min": 0.1, "beta_max": 20.0, "T": 1.0},
        "sample_ids": [2, 5, 9],
        "noise_bank_identity": "fixed-source-fixture-sha256",
    }
    result = {"valid": True, "c_mla": 1.0, "c_ot": 0.037, "provenance": expected}
    path = tmp_path / "source_gradient_calibration.json"
    save_gradient_calibration(path, result)

    loaded = load_gradient_calibration(path, expected)

    assert loaded["c_mla"] == result["c_mla"]
    assert loaded["c_ot"] == result["c_ot"]
    assert loaded["provenance"] == expected
    with pytest.raises(ValueError, match="provenance"):
        load_gradient_calibration(path, {**expected, "alpha": 0.4})
