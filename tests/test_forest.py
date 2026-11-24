from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import check_grad
from scipy.sparse import csr_matrix

from scistest import (
    GeneratorConfig,
    RandomForestDirectionConfig,
    RandomForestHazardConfig,
    SciTestConfig,
    scis_test,
)
from scistest._forest import fit_hazard, likelihood_loss_gradient


def test_survival_likelihood_gradient():
    design = csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    exposure, events = np.array([0.1, 0.3, 0.2]), np.array([0.2, 0.3])
    error = check_grad(
        lambda b: likelihood_loss_gradient(b, design, exposure, events, 0)[0],
        lambda b: likelihood_loss_gradient(b, design, exposure, events, 0)[1],
        np.array([0.4, -0.1]),
    )
    assert error < 1e-7


def test_forest_hazard_l1_shrinks_effective_leaf_coefficients():
    torch.manual_seed(91)
    z = torch.randn(60, 2)
    time = torch.linspace(0.01, 0.8, 60)
    event = torch.tensor([0.0, 1.0] * 30)
    models = [
        fit_hazard(
            z,
            time,
            event,
            RandomForestHazardConfig(
                n_estimators=4, max_depth=2, min_samples_leaf=5, l1_penalty=penalty
            ),
            10,
            4,
        )
        for penalty in (0.001, 0.2)
    ]
    assert models[1].diagnostics["l1_norm"] < models[0].diagnostics["l1_norm"]
    for model in models:
        assert model.diagnostics["objective"] <= model.diagnostics["initial_objective"]
        assert model.diagnostics["kkt_residual"] < 1e-4
        assert torch.all(model.hazard(time, z) > 0)


def test_three_forest_estimators_use_penalized_manuscript_losses():
    torch.set_num_threads(1)
    rng = np.random.default_rng(14)
    x, z = rng.normal(size=(2, 60, 1))
    time = rng.exponential(size=60)
    event = np.tile([0, 1], 30)
    result = scis_test(
        x,
        z,
        time,
        event,
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
            solver_max_iterations=50000,
        ),
        generator_config=GeneratorConfig(epochs=2, hidden_dim=4, noise_dim=4),
        test_config=SciTestConfig(integration_points=10, generator_draws=10),
    )
    assert set(result.nuisance_diagnostics) == {"hazard", "hazard_evaluation", "direction"}
    assert result.scoring_hazard == "evaluation"
    assert np.isfinite(result.test_statistic)
    for diagnostics in result.nuisance_diagnostics.values():
        assert diagnostics["l1_norm"] > 0
        assert diagnostics["active_leaves"] > 0
    assert result.direction_loss == result.nuisance_diagnostics["direction"]["objective"]
