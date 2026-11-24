"""Certified fallback for the unchanged leaf Lasso objective.

Column normalization is only a numerical change of coordinates. The L1
weights transform inversely, so the statistical penalty is unchanged.
ADMM follows Boyd et al. (2011); a feasible dual point certifies convergence
at the same gap threshold as the primary coordinate-descent solver.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


def lasso_gap(design, target, coefficients, penalty):
    """Gap for .5 ||A theta-y||^2 + penalty ||theta||_1, divided by n."""
    residual = target - design @ coefficients
    correlations = np.asarray(design.T @ residual).ravel()
    largest = float(np.abs(correlations).max(initial=0))
    dual = residual * min(1.0, penalty / largest) if largest else residual
    primal = 0.5 * residual @ residual + penalty * np.abs(coefficients).sum()
    dual_value = target @ dual - 0.5 * dual @ dual
    return max(0.0, float(primal - dual_value)) / len(target)


def solve_lasso_admm(design, target, penalty, tolerance, initial, max_iterations=100000):
    """Solve the same L1 problem without adding any ridge penalty."""
    n, dimensions = design.shape
    norms = np.sqrt(np.asarray(design.power(2).sum(axis=0)).ravel())
    active = norms > 0
    coefficients = np.zeros(dimensions)
    if not active.any():
        return coefficients, {"solver": "admm", "iterations": 0, "dual_gap": 0.0}
    scale = norms[active]
    normalized = design[:, active].multiply(1 / scale).toarray()
    gram = normalized.T @ normalized
    linear = normalized.T @ target
    weights = penalty / scale
    z = initial[active] * scale
    rho = 1.0
    u = (linear - gram @ z) / rho
    factor = cho_factor(gram + rho * np.eye(len(z)), check_finite=False)
    threshold = tolerance * float(target @ target) / n
    gap = np.inf
    for iteration in range(1, max_iterations + 1):
        x = cho_solve(factor, linear + rho * (z - u), check_finite=False)
        previous = z.copy()
        relaxed = 1.6 * x - 0.6 * previous
        proposal = relaxed + u
        z = np.sign(proposal) * np.maximum(np.abs(proposal) - weights / rho, 0)
        u += relaxed - z
        if iteration == 1 or iteration % 25 == 0:
            coefficients[active] = z / scale
            gap = lasso_gap(design, target, coefficients, penalty)
            if np.isfinite(gap) and gap <= threshold:
                return coefficients, {
                    "solver": "admm",
                    "iterations": iteration,
                    "dual_gap": gap,
                    "gap_tolerance": threshold,
                    "rho": rho,
                }
        if iteration % 50 == 0:
            primal_residual = np.linalg.norm(x - z)
            dual_residual = rho * np.linalg.norm(z - previous)
            updated = rho
            if primal_residual > 10 * dual_residual:
                updated = rho * 2
            elif dual_residual > 10 * primal_residual:
                updated = rho / 2
            updated = float(np.clip(updated, 1e-8, 1e8))
            if updated != rho:
                u *= rho / updated
                rho = updated
                factor = cho_factor(gram + rho * np.eye(len(z)), check_finite=False)
    raise RuntimeError(f"Lasso ADMM did not converge: gap={gap:.8g}, threshold={threshold:.8g}")


def solve_lasso_qp(design, target, penalty, tolerance):
    """Interior-point solve of the exact Lasso using residual/slack variables.

    Avoid forming A.T @ A, which squares the condition number. This changes
    neither the coefficient penalty nor the final dual-gap acceptance test.
    """
    import clarabel

    n, dimension = design.shape
    norms = np.sqrt(np.asarray(design.power(2).sum(axis=0)).ravel())
    active = norms > 0
    scale = norms[active]
    normalized = design[:, active].multiply(1 / scale).tocsc()
    p = len(scale)
    identity = sparse.eye(p, format="csc")
    constraints = sparse.bmat(
        [
            [sparse.eye(n), -normalized, sparse.csc_matrix((n, p))],
            [sparse.csc_matrix((p, n)), identity, -identity],
            [sparse.csc_matrix((p, n)), -identity, -identity],
        ],
        format="csc",
    )
    quadratic = sparse.diags(np.r_[np.ones(n), np.zeros(2 * p)], format="csc")
    linear = np.r_[np.zeros(n + p), penalty / scale]
    rhs = np.r_[-target, np.zeros(2 * p)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 500
    settings.tol_gap_abs = 1e-12
    settings.tol_gap_rel = 1e-12
    settings.tol_feas = 1e-12
    settings.time_limit = 120.0
    solver = clarabel.DefaultSolver(
        quadratic,
        linear,
        constraints,
        rhs,
        [clarabel.ZeroConeT(n), clarabel.NonnegativeConeT(2 * p)],
        settings,
    )
    result = solver.solve()
    coefficients = np.zeros(dimension)
    coefficients[active] = np.asarray(result.x)[n : n + p] / scale
    gap = lasso_gap(design, target, coefficients, penalty)
    threshold = tolerance * float(target @ target) / n
    if not np.isfinite(gap) or gap > threshold:
        raise RuntimeError(
            f"Lasso QP not certified: status={result.status}, "
            f"gap={gap:.8g}, threshold={threshold:.8g}"
        )
    return coefficients, {
        "solver": "clarabel_qp",
        "iterations": result.iterations,
        "dual_gap": gap,
        "gap_tolerance": threshold,
        "solver_status": str(result.status),
        "solve_seconds": result.solve_time,
    }


def solve_lasso_certified(design, target, penalty, tolerance, initial, max_iterations=100000):
    """Prefer an interior-point QP solver; retain ADMM as an optional fallback."""
    try:
        return solve_lasso_qp(design, target, penalty, tolerance)
    except (ImportError, RuntimeError) as error:
        coefficients, details = solve_lasso_admm(
            design, target, penalty, tolerance, initial, max_iterations
        )
        details["qp_fallback_reason"] = str(error)
        return coefficients, details
