from __future__ import annotations

import numpy as np
import pytest
import torch

from scistest import RandomForestDirectionConfig, ScisTestResult, scis_test, scisTest
from scistest._models import HazardNetwork, RandomForestDirection
from scistest.core import (
    _fit_direction,
    _quadratic_direction_loss,
    _raw_direction_scores,
    validate_survival_data,
)
from scistest.simulation import simulate_survival_data


def test_quadratic_direction_loss_matches_definition() -> None:
    scores = torch.tensor([1.0, -2.0, 3.0])
    expected = torch.sum(scores**2) - torch.sum(scores)
    assert torch.equal(_quadratic_direction_loss(scores), expected)


def test_camel_case_main_function_alias() -> None:
    assert scisTest is scis_test


def test_result_type_uses_scis_name() -> None:
    assert ScisTestResult.__name__ == "ScisTestResult"


def test_random_forest_direction_minimizes_quadratic_criterion() -> None:
    torch.manual_seed(1)
    n = 16
    z = torch.randn(n, 2)
    x = torch.randn(n, 1)
    time = torch.linspace(0.2, 1.0, n)
    event = torch.tensor([0.0, 1.0] * (n // 2))
    hazard = HazardNetwork(z_dim=2, hidden_dims=(4,))
    hazard.eval()
    for parameter in hazard.parameters():
        parameter.requires_grad_(False)

    config = RandomForestDirectionConfig(
        n_estimators=5,
        max_depth=3,
        min_samples_leaf=2,
        n_jobs=1,
    )
    direction, reported_loss = _fit_direction(
        hazard,
        z,
        x,
        time,
        event,
        config,
        integration_points=5,
        seed=17,
    )
    scores = _raw_direction_scores(
        direction,
        hazard,
        z,
        x,
        time,
        event,
        integration_points=5,
    )

    assert isinstance(direction, RandomForestDirection)
    assert np.isclose(reported_loss, float(_quadratic_direction_loss(scores)))
    assert reported_loss <= 1e-6  # The zero function has objective value zero.
    assert torch.isfinite(direction(time, z, x)).all()


def test_validate_survival_data_accepts_vector_x_and_z() -> None:
    x = np.arange(12.0)
    z = np.arange(12.0) / 2
    time = np.arange(1.0, 13.0)
    event = np.tile([0.0, 1.0], 6)
    x_out, z_out, _, _ = validate_survival_data(x, z, time, event)
    assert x_out.shape == (12, 1)
    assert z_out.shape == (12, 1)


def test_validate_survival_data_rejects_nonbinary_event() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_survival_data(
            np.ones(12), np.ones((12, 2)), np.ones(12), np.full(12, 0.5)
        )


def test_simulation_null_and_censoring_are_well_formed() -> None:
    data = simulate_survival_data(100, effect=0.0, random_state=22)
    assert data.x.shape == (100, 1)
    assert data.z.shape == (100, 3)
    assert np.array_equal(data.time, np.minimum(data.event_time, data.censoring_time))
    assert set(np.unique(data.event)).issubset({0.0, 1.0})
