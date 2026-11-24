"""Explicit main6 settings for simulations and the Framingham application."""

from __future__ import annotations

import math

from .config import (
    DirectionConfig,
    GeneratorConfig,
    HazardConfig,
    RandomForestDirectionConfig,
    RandomForestHazardConfig,
    SciTestConfig,
)


def simulation_configs(n: int, *, integration_points: int = 100) -> dict:
    """Use total dataset size n, including for every calibration subsample.

    The fixed 100-point grid follows main6.tex. The historical figure run used
    floor(10 sqrt(n)); callers can request that grid explicitly.
    """
    if isinstance(n, bool) or int(n) != n or n < 12:
        raise ValueError("n must be an integer of at least 12")
    depth = math.floor(math.log10(n))
    width = math.floor(2 * math.sqrt(n))
    epochs = math.floor(50 * math.log(n, 20))
    return {
        "hazard_config": HazardConfig((width,) * depth, 2 / n, epochs, 5 / n, 128),
        "direction_config": DirectionConfig((width,) * depth, 2 / n, epochs, 5 / n),
        "generator_config": GeneratorConfig(2 / n, epochs, 128, 1 + depth, width),
        "test_config": SciTestConfig(
            integration_points=integration_points, generator_draws=100, score_chunk_size=8
        ),
    }


def framingham_configs() -> dict:
    """Fixed application settings, including standardized hunting objective."""
    return {
        "hazard_config": HazardConfig((32,), 0.01, 100, 0.001, 128),
        "direction_config": DirectionConfig((32, 32), 0.01, 100, 0.001, "standardized"),
        "generator_config": GeneratorConfig(0.005, 100, 128, 2, 100),
        "test_config": SciTestConfig(
            integration_points=100, generator_draws=100, score_chunk_size=8
        ),
    }


def forest_configs(n: int, parameters: dict | None = None) -> dict:
    """Two-split RF/L1 settings at total input size n; generator unchanged.

    Penalties apply to effective leaf contributions, whose sum is the forest
    prediction. Equivalently, these are conventional averaged leaf values
    divided by the number of trees. Defaults follow main6's RF table: 32
    hazard trees, 16 direction trees, depth 3, and penalties 5/n and 50/n.
    This constructs estimator settings only; it does not run experiments.
    """
    parameters = parameters or {}
    configs = simulation_configs(n)
    configs["hazard_config"] = RandomForestHazardConfig(
        n_estimators=parameters.get("hazard_trees", 32),
        max_depth=parameters.get("hazard_depth", 3),
        min_samples_leaf=parameters.get("min_samples_leaf", 20),
        max_features=parameters.get("max_features", 0.8),
        n_jobs=1,
        l1_penalty=parameters.get("hazard_l1_factor", 5.0) / n,
        solver_tolerance=1e-5,
        solver_max_iterations=5000,
    )
    configs["direction_config"] = RandomForestDirectionConfig(
        n_estimators=parameters.get("direction_trees", 16),
        max_depth=parameters.get("direction_depth", 3),
        min_samples_leaf=parameters.get("min_samples_leaf", 20),
        max_features=parameters.get("max_features", 0.8),
        n_jobs=1,
        l1_penalty=parameters.get("direction_l1_factor", 50.0) / n,
        solver_tolerance=1e-5,
        solver_max_iterations=100000,
    )
    return configs


def forest_simulation_configs(n: int, parameters: dict | None = None) -> dict:
    """Compatibility name for :func:`forest_configs`; no experiment is run."""
    return forest_configs(n, parameters)
