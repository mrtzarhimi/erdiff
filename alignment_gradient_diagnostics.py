"""Gradient norms/cosines and source-only loss scale calibration."""

import json
import math
from pathlib import Path

import torch


def _flat_gradient(loss, parameters):
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters:
        return None
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in parameters), device=parameters[0].device,
                           dtype=parameters[0].dtype)
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    pieces = [
        (torch.zeros_like(parameter) if grad is None else grad).detach().reshape(-1)
        for parameter, grad in zip(parameters, grads)
    ]
    return torch.cat(pieces)


def component_gradient_report(losses, parameters, coefficients=None):
    """Report raw and weighted gradient norms/cosines without touching ``.grad``."""
    coefficients = coefficients or {}
    vectors = {name: _flat_gradient(loss, parameters) for name, loss in losses.items()}
    result = {}
    for name, vector in vectors.items():
        norm = None if vector is None else float(torch.linalg.vector_norm(vector).cpu())
        coefficient = float(coefficients.get(name, 1.0))
        result[name] = {"norm": norm, "coefficient": coefficient,
                        "weighted_norm": None if norm is None else abs(coefficient) * norm}
    cosines, weighted_cosines = {}, {}
    names = list(vectors)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = vectors[left], vectors[right]
            value = weighted_value = None
            if a is not None and b is not None:
                norms = (torch.linalg.vector_norm(a), torch.linalg.vector_norm(b))
                denom = norms[0] * norms[1]
                if bool(torch.isfinite(denom)) and float(denom) > 1e-12:
                    value = float(torch.dot(a, b).div(denom).cpu())
                    weighted_denom = abs(float(coefficients.get(left, 1.0))) * norms[0]
                    weighted_denom *= abs(float(coefficients.get(right, 1.0))) * norms[1]
                    if float(weighted_denom) > 1e-12:
                        weighted_value = float(
                            torch.dot(a * float(coefficients.get(left, 1.0)),
                                      b * float(coefficients.get(right, 1.0)))
                            .div(weighted_denom).cpu()
                        )
            cosines[f"{left}:{right}"] = value
            weighted_cosines[f"{left}:{right}"] = weighted_value
    return {"components": result, "cosines": cosines, "weighted_cosines": weighted_cosines}


def calibrate_source_gradient_scale(
    probe_losses,
    parameters,
    *,
    alpha,
    minimum=0.01,
    maximum=100.0,
    inactive_threshold=1e-12,
    mla_variance_weight=0.0,
    metadata=None,
    alignment_objective="paper_squared",
):
    """Calibrate the OT multiplier from fixed source-only MLA/OT loss probes.

    The ratio uses RMS *weighted* gradient norms of the complete MLA branch,
    using mode-specific coefficients: (1-alpha, alpha) in paper_squared
    and (1, 1) in author_2023. The MLA branch includes Skilling in author
    mode and the configured variance penalty in either mode. If a
    variance loss is supplied, it participates only through its configured
    coefficient; ``c_ot`` never scales the variance term separately.
    """
    from diffusion import alignment_coefficients
    mla_coefficient, ot_coefficient = alignment_coefficients(alignment_objective, alpha)
    if not mla_coefficient > 0 or not ot_coefficient > 0:
        return {"valid": False, "reason": "alpha_must_mix_both_components",
                "c_mla": 1.0, "c_ot": None, "metadata": metadata or {}}
    if minimum <= 0 or maximum < minimum or inactive_threshold <= 0:
        raise ValueError("Gradient calibration bounds and inactive threshold must be positive and ordered.")
    probe_losses = list(probe_losses)
    if not probe_losses:
        return {"valid": False, "reason": "no_source_probes", "c_mla": 1.0,
                "c_ot": None, "metadata": metadata or {}}

    probe_norms = []
    for losses in probe_losses:
        mla_branch = losses["mla"] + mla_variance_weight * losses.get("variance", losses["mla"] * 0.0)
        report = component_gradient_report(
            {"mla": mla_branch, "ot": losses["ot"]}, parameters,
            {"mla": mla_coefficient, "ot": ot_coefficient},
        )
        mla_norm = report["components"]["mla"]["norm"]
        ot_norm = report["components"]["ot"]["norm"]
        probe_norms.append({"mla_raw": mla_norm, "ot_raw": ot_norm,
                            "mla_weighted": report["components"]["mla"]["weighted_norm"],
                            "ot_weighted": report["components"]["ot"]["weighted_norm"]})

    dtype = next((str(p.dtype) for p in parameters if p.requires_grad), "unknown")
    values = [value for row in probe_norms for value in
              (row["mla_raw"], row["ot_raw"], row["mla_weighted"], row["ot_weighted"])]
    if any(value is None or not math.isfinite(value) for value in values):
        return {"valid": False, "reason": "nonfinite_or_missing_gradient", "c_mla": 1.0,
                "c_ot": None, "probe_norms": probe_norms, "calibration_dtype": dtype,
                "inactive_threshold": inactive_threshold, "metadata": metadata or {}}
    weighted_mla = torch.tensor([row["mla_weighted"] for row in probe_norms], dtype=torch.float64)
    weighted_ot = torch.tensor([row["ot_weighted"] for row in probe_norms], dtype=torch.float64)
    mla_rms = float(weighted_mla.square().mean().sqrt())
    ot_rms = float(weighted_ot.square().mean().sqrt())
    if mla_rms <= inactive_threshold or ot_rms <= inactive_threshold:
        return {"valid": False, "reason": "nonfinite_or_inactive_gradient", "c_mla": 1.0,
                "c_ot": None, "probe_norms": probe_norms, "calibration_dtype": dtype,
                "inactive_threshold": inactive_threshold, "metadata": metadata or {}}
    factor = min(maximum, max(minimum, mla_rms / ot_rms))
    return {
        "valid": True, "reason": None, "c_mla": 1.0, "c_ot": factor,
        "mla_weighted_rms_norm": mla_rms, "ot_weighted_rms_norm": ot_rms,
        "probe_norms": probe_norms, "probe_count": len(probe_norms),
        "calibration_dtype": dtype, "inactive_threshold": inactive_threshold,
        "bounds": [minimum, maximum], "variance_scaled": False,
        "metadata": metadata or {},
    }


def save_gradient_calibration(path, calibration):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def load_gradient_calibration(path, expected_provenance):
    with Path(path).open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if not calibration.get("valid") or calibration.get("c_mla") != 1.0:
        raise ValueError("Saved source gradient calibration is invalid.")
    if calibration.get("provenance") != expected_provenance:
        raise ValueError("Saved source gradient calibration provenance does not match this run.")
    factor = calibration.get("c_ot")
    if not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("Saved source gradient calibration has an invalid OT factor.")
    return calibration
