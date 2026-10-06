from __future__ import annotations

import inspect

import numpy as np

from scistest import (
    aggregate_p_value,
    aggregated_scis_test,
    calibrate_aggregated_p_value,
    make_subsamples,
    normal_rank_transform,
)


def test_aggregation_defaults_are_k3_j5_l6() -> None:
    for function in (aggregate_p_value, aggregated_scis_test):
        parameters = inspect.signature(function).parameters
        assert parameters["K"].default == 3
        assert parameters["J"].default == 5
        assert parameters["L"].default == 6


def test_subsamples_are_disjoint_within_each_permutation() -> None:
    blocks = make_subsamples(20, 3, 3, random_state=9)
    assert blocks.shape == (9, 6)
    grouped = blocks.reshape(3, 3, 6)
    for permutation_blocks in grouped:
        assert len(np.unique(permutation_blocks)) == 18


def test_rank_transform_is_finite_and_reproducible_with_ties() -> None:
    values = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 4.0]])
    first = normal_rank_transform(values, random_state=12)
    second = normal_rank_transform(values, random_state=12)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert len(np.unique(first)) == values.size


def test_calibration_returns_smoothed_and_empirical_tails() -> None:
    observed = np.array([0.5, 0.8, 0.6])
    subsampled = np.array(
        [
            [-1.2, -0.8, -0.6],
            [-0.5, 0.1, -0.2],
            [0.0, 0.2, 0.1],
            [0.4, 0.7, 0.5],
            [0.8, 1.0, 0.9],
            [1.1, 1.2, 1.4],
        ]
    )
    result = calibrate_aggregated_p_value(
        observed, subsampled, 3, 2, 3, n=35, random_state=3
    )
    assert result.B == 6
    assert result.K == 3
    assert result.subsample_size == 11
    assert 0.0 <= result.aggregated_p_value <= 1.0
    assert 0.0 <= result.empirical_p_value <= 1.0
    assert result.p_value_decimal.endswith("E-01") or "E" in result.p_value_decimal
    assert np.isclose(result.aggregate_statistic, observed.mean())
    default_output = result.as_dict()
    assert "aggregated_p_value" in default_output
    assert "p_value" not in default_output
    assert "p_value_decimal" not in default_output
    assert "log_p_value" not in default_output
    assert "log10_p_value" not in default_output
    diagnostic_output = result.as_dict(include_p_value_diagnostics=True)
    assert "p_value_decimal" in diagnostic_output
    assert "log_p_value" in diagnostic_output
    assert "log10_p_value" in diagnostic_output


def test_generic_aggregation_passes_indices_and_independent_seeds() -> None:
    calls: list[tuple[np.ndarray, int]] = []

    def statistic(indices: np.ndarray, split_seed: int) -> float:
        calls.append((indices.copy(), split_seed))
        rng = np.random.default_rng(split_seed)
        return float(indices.mean() / 40 + rng.normal(scale=0.01))

    result = aggregate_p_value(40, statistic, 3, 2, 3, random_state=14)
    assert result.B == 6
    assert len(calls) == (result.B + 1) * result.L
    assert len({seed for _, seed in calls}) == len(calls)
    assert all(len(indices) == 40 for indices, _ in calls[:3])
    assert all(len(indices) == 13 for indices, _ in calls[3:])
