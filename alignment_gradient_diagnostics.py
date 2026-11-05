"""Gradient norms/cosines and source-only loss scale calibration."""

import math

import torch


def _flat_gradient(loss, parameters):
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters or not loss.requires_grad:
        return None
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    pieces = [g.detach().reshape(-1) for g in grads if g is not None]
    return torch.cat(pieces) if pieces else None


def component_gradient_report(losses, parameters, coefficients=None):
    """Report per-loss norms and cosine similarity without accumulating .grad."""
    coefficients = coefficients or {}
    vectors = {name: _flat_gradient(loss, parameters) for name, loss in losses.items()}
    result = {}
    for name, vector in vectors.items():
        norm = None if vector is None else float(torch.linalg.vector_norm(vector).cpu())
        coefficient = float(coefficients.get(name, 1.0))
        result[name] = {"norm": norm, "coefficient": coefficient,
                        "weighted_norm": None if norm is None else abs(coefficient) * norm}
    cosines = {}
    names = list(vectors)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = vectors[left], vectors[right]
            value = None
            if a is not None and b is not None:
                denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
                if torch.isfinite(denom) and denom > 1e-12:
                    value = float(torch.dot(a, b).div(denom).cpu())
            cosines[f"{left}:{right}"] = value
    return {"components": result, "cosines": cosines}


def calibrate_source_gradient_scale(mla_norm, ot_norm, minimum=0.01, maximum=100.0,
                                   inactive_threshold=1e-12):
    values = (float(mla_norm), float(ot_norm))
    if not all(math.isfinite(value) and value > inactive_threshold for value in values):
        return {"valid": False, "reason": "nonfinite_or_inactive_gradient", "c_mla": 1.0, "c_ot": None}
    return {"valid": True, "c_mla": 1.0,
            "c_ot": min(maximum, max(minimum, values[0] / values[1]))}
