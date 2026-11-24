from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from scistest import (
    GeneratorConfig,
    RandomForestDirectionConfig,
    RandomForestHazardConfig,
    SciTestConfig,
    scis_test,
)
from scistest._lasso import lasso_gap, solve_lasso_admm, solve_lasso_qp


def test_weighted_normalization_preserves_original_l1_objective():
    scales = np.array([0.01, 0.3, 2, 40.0])
    design = csr_matrix(np.diag(scales))
    target = np.array([0.2, -0.6, 1.2, -0.7])
    penalty = 0.03
    expected = (
        np.sign(scales * target) * np.maximum(np.abs(scales * target) - penalty, 0) / scales**2
    )
    fit, details = solve_lasso_admm(design, target, penalty, 1e-10, np.zeros(4))
    np.testing.assert_allclose(fit, expected, rtol=1e-4, atol=1e-6)
    assert details["dual_gap"] <= details["gap_tolerance"]


def test_rank_deficient_leaf_basis_has_certified_solution():
    rng = np.random.default_rng(82)
    base = rng.normal(size=(80, 6))
    design = csr_matrix(np.c_[base, base, base[:, :2] * 30, np.zeros(80)])
    target = rng.normal(size=80)
    fit, details = solve_lasso_admm(design, target, 0.1, 1e-8, np.zeros(design.shape[1]))
    assert details["dual_gap"] <= details["gap_tolerance"]
    assert lasso_gap(design, target, fit, 0.1) <= details["gap_tolerance"]
    assert fit[-1] == 0


def test_qp_backend_preserves_l1_with_redundant_scaled_columns():
    pytest.importorskip("clarabel")
    rng = np.random.default_rng(82)
    base = rng.normal(size=(80, 6))
    design = csr_matrix(np.c_[base, base, base[:, :2] * 30, np.zeros(80)])
    target = rng.normal(size=80)
    coefficients, details = solve_lasso_qp(design, target, 0.1, 1e-8)
    assert details["solver"] == "clarabel_qp"
    assert lasso_gap(design, target, coefficients, 0.1) <= details["gap_tolerance"]


def test_nonconverged_primary_solver_uses_fallback_in_full_test():
    torch.set_num_threads(1)
    rng = np.random.default_rng(14)
    x, z = rng.normal(size=(2, 60, 1))
    result = scis_test(
        x,
        z,
        rng.exponential(size=60),
        np.tile([0, 1], 30),
        random_state=42,
        hazard_config=RandomForestHazardConfig(
            n_estimators=4, max_depth=2, min_samples_leaf=5, l1_penalty=0.01
        ),
        direction_config=RandomForestDirectionConfig(
            n_estimators=4,
            max_depth=2,
            min_samples_leaf=5,
            n_jobs=1,
            l1_penalty=0.1,
            solver_tolerance=1e-5,
            solver_max_iterations=1,
        ),
        generator_config=GeneratorConfig(epochs=2, hidden_dim=4, noise_dim=4),
        test_config=SciTestConfig(integration_points=10, generator_draws=10),
    )
    details = result.nuisance_diagnostics["direction"]
    assert details["solver"] in {"admm", "clarabel_qp"}
    assert details["dual_gap"] <= details["gap_tolerance"]
    assert np.isfinite(result.test_statistic)
