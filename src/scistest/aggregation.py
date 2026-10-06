"""Guo--Shah rank-transformed subsampling for aggregating split statistics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr, logsumexp, ndtri

from .config import (
    DirectionConfig,
    GeneratorConfig,
    HazardConfig,
    RandomForestDirectionConfig,
    SciTestConfig,
)
from .core import scis_test, validate_survival_data

StatisticFunction = Callable[[NDArray[np.int64], int], float]


@dataclass(frozen=True)
class AggregatedPValueResult:
    """Output of rank-transformed subsampling calibration."""

    aggregated_p_value: float
    p_value_decimal: str = field(repr=False)
    log_p_value: float = field(repr=False)
    log10_p_value: float = field(repr=False)
    empirical_p_value: float
    aggregate_statistic: float
    exceedance_count: int
    empirical_resolution: float
    bandwidth: float
    n: int | None
    K: int
    subsample_size: int | None
    J: int
    L: int
    B: int
    observed_statistics: NDArray[np.float64] = field(repr=False)
    subsample_statistics: NDArray[np.float64] = field(repr=False)
    rank_transformed_statistics: NDArray[np.float64] = field(repr=False)
    calibration_statistics: NDArray[np.float64] = field(repr=False)
    subsample_indices: NDArray[np.int64] | None = field(default=None, repr=False)

    def as_dict(
        self,
        *,
        include_arrays: bool = False,
        include_p_value_diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Return a JSON-friendly representation of the result."""

        output = asdict(self)
        if not include_p_value_diagnostics:
            for key in ("p_value_decimal", "log_p_value", "log10_p_value"):
                output.pop(key)
        array_keys = (
            "observed_statistics",
            "subsample_statistics",
            "rank_transformed_statistics",
            "calibration_statistics",
            "subsample_indices",
        )
        if include_arrays:
            for key in array_keys:
                if output[key] is not None:
                    output[key] = output[key].tolist()
        else:
            for key in array_keys:
                output.pop(key)
        return output


def make_subsamples(
    n: int,
    K: int = 3,
    J: int = 5,
    *,
    random_state: int = 1,
) -> NDArray[np.int64]:
    """Generate Algorithm 1 subsamples.

    Each of ``J`` independent permutations contributes exactly ``K`` disjoint
    blocks of size ``floor(n / K)``. Any remainder is unused.
    """

    n, K, J = int(n), int(K), int(J)
    if n < 4 or not 2 <= K <= n // 2:
        raise ValueError("Require n >= 4 and 2 <= K <= floor(n / 2)")
    if J < 1:
        raise ValueError("J must be at least one")
    if int(random_state) < 0:
        raise ValueError("random_state must be nonnegative")
    rng = np.random.default_rng(int(random_state))
    subsample_size = n // K
    return np.vstack(
        [
            rng.permutation(n)[: K * subsample_size].reshape(K, subsample_size)
            for _ in range(J)
        ]
    )


def normal_rank_transform(
    statistics: NDArray[np.float64],
    *,
    random_state: int = 1,
) -> NDArray[np.float64]:
    """Pool all ranks and transform them to standard-normal scores.

    A seeded random ordering resolves ties without changing non-tied ranks.
    """

    values = np.asarray(statistics, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("statistics must be a B by L matrix with B,L >= 2")
    if not np.isfinite(values).all():
        raise ValueError("statistics must be finite")
    flat = values.ravel()
    permutation = np.random.default_rng(int(random_state)).permutation(flat.size)
    order = permutation[np.argsort(flat[permutation], kind="stable")]
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(1, flat.size + 1, dtype=np.float64)
    probabilities = (ranks - 0.5) / flat.size
    return ndtri(probabilities).reshape(values.shape)


def _bandwidth_nrd0(values: NDArray[np.float64]) -> float:
    """R's ``stats::bw.nrd0`` rule, used by the MultiSplit implementation."""

    values = np.asarray(values, dtype=np.float64)
    standard_deviation = float(np.std(values, ddof=1))
    iqr = float(np.diff(np.quantile(values, [0.25, 0.75]))[0])
    scale = min(standard_deviation, iqr / 1.34)
    if scale == 0:
        scale = standard_deviation or abs(float(values[0])) or 1.0
    return 0.9 * scale * len(values) ** (-0.2)


def _decimal_from_log(log_value: float, significant_digits: int = 17) -> str:
    """Represent a tiny tail without binary64 underflow."""

    if not math.isfinite(log_value):
        raise ValueError("log probability must be finite")
    with localcontext() as context:
        context.prec = 60
        context.Emin = -999999999
        context.Emax = 999999999
        return format(Decimal(str(log_value)).exp(), f".{significant_digits - 1}E")


def calibrate_aggregated_p_value(
    observed_statistics: NDArray[np.float64],
    subsample_statistics: NDArray[np.float64],
    K: int = 3,
    J: int = 5,
    L: int = 6,
    *,
    n: int | None = None,
    random_state: int = 1,
    subsample_indices: NDArray[np.int64] | None = None,
) -> AggregatedPValueResult:
    """Calibrate already-computed split statistics.

    Parameters ``K``, ``J``, and ``L`` are explicit so that saved statistic
    matrices cannot silently be analyzed under a different aggregation design.
    The primary p-value is the Gaussian-kernel-smoothed right tail used by the
    authors' MultiSplit implementation; the unsmoothed empirical tail is also
    returned.
    """

    K, J, L = int(K), int(J), int(L)
    if K < 2 or J < 1 or L < 2:
        raise ValueError("Require K >= 2, J >= 1, and L >= 2")
    observed = np.asarray(observed_statistics, dtype=np.float64)
    subsampled = np.asarray(subsample_statistics, dtype=np.float64)
    if observed.shape != (L,):
        raise ValueError(f"observed_statistics must have shape ({L},)")
    if subsampled.ndim != 2 or subsampled.shape[1] != L:
        raise ValueError(f"subsample_statistics must have shape (B, {L})")
    expected_B = J * K
    if subsampled.shape[0] != expected_B:
        raise ValueError(
            f"Expected B=J*K={expected_B} subsample rows; got {subsampled.shape[0]}"
        )
    if not np.isfinite(observed).all() or not np.isfinite(subsampled).all():
        raise ValueError("All statistics must be finite")

    B = len(subsampled)
    subsample_size: int | None = None
    if n is not None:
        n = int(n)
        if n < 4 or not 2 <= K <= n // 2:
            raise ValueError("Require n >= 4 and 2 <= K <= floor(n / 2)")
        subsample_size = n // K
    if subsample_indices is not None:
        subsample_indices = np.asarray(subsample_indices, dtype=np.int64)
        if subsample_indices.ndim != 2 or subsample_indices.shape[0] != B:
            raise ValueError(f"subsample_indices must have {B} rows")
        if subsample_size is None:
            subsample_size = subsample_indices.shape[1]
        if subsample_indices.shape != (B, subsample_size):
            raise ValueError(
                f"subsample_indices must have shape ({B}, {subsample_size})"
            )

    transformed = normal_rank_transform(subsampled, random_state=random_state)
    calibration = transformed.mean(axis=1, dtype=np.float64)
    aggregate_statistic = float(observed.mean(dtype=np.float64))
    exceedance_count = int(np.count_nonzero(calibration > aggregate_statistic))
    bandwidth = _bandwidth_nrd0(calibration)
    log_p_value = float(
        logsumexp(log_ndtr((calibration - aggregate_statistic) / bandwidth))
        - math.log(B)
    )
    log_p_value = min(0.0, log_p_value)
    return AggregatedPValueResult(
        aggregated_p_value=float(math.exp(log_p_value)),
        p_value_decimal=_decimal_from_log(log_p_value),
        log_p_value=log_p_value,
        log10_p_value=log_p_value / math.log(10),
        empirical_p_value=exceedance_count / B,
        aggregate_statistic=aggregate_statistic,
        exceedance_count=exceedance_count,
        empirical_resolution=1 / B,
        bandwidth=bandwidth,
        n=n,
        K=K,
        subsample_size=subsample_size,
        J=J,
        L=L,
        B=B,
        observed_statistics=observed,
        subsample_statistics=subsampled,
        rank_transformed_statistics=transformed,
        calibration_statistics=calibration,
        subsample_indices=subsample_indices,
    )


def aggregate_p_value(
    n: int,
    statistic: StatisticFunction,
    K: int = 3,
    J: int = 5,
    L: int = 6,
    *,
    random_state: int = 1,
) -> AggregatedPValueResult:
    """Compute a Guo--Shah aggregated p-value from a split-statistic callback.

    ``statistic(indices, split_seed)`` must refit the complete base procedure on
    the selected observations and return its one-sided, asymptotically
    standard-normal test statistic. The callback is invoked ``L`` times on the
    full sample and ``L`` times on every rank-calibration subsample.
    """

    n, K, J, L = int(n), int(K), int(J), int(L)
    if L < 2:
        raise ValueError("L must be at least two")
    if int(random_state) < 0:
        raise ValueError("random_state must be nonnegative")
    seed_sequence = np.random.SeedSequence(int(random_state))
    subsample_seed_sequence, fit_seed_sequence, rank_seed_sequence = seed_sequence.spawn(3)
    subsample_seed = int(subsample_seed_sequence.generate_state(1)[0])
    subsamples = make_subsamples(n, K, J, random_state=subsample_seed)
    B = len(subsamples)
    fit_seeds = [
        int(child.generate_state(1)[0])
        for child in fit_seed_sequence.spawn((B + 1) * L)
    ]

    observed = np.empty(L, dtype=np.float64)
    full_indices = np.arange(n, dtype=np.int64)
    position = 0
    for split in range(L):
        observed[split] = float(statistic(full_indices, fit_seeds[position]))
        position += 1

    subsampled = np.empty((B, L), dtype=np.float64)
    for row, indices in enumerate(subsamples):
        for split in range(L):
            subsampled[row, split] = float(statistic(indices, fit_seeds[position]))
            position += 1

    rank_seed = int(rank_seed_sequence.generate_state(1)[0])
    return calibrate_aggregated_p_value(
        observed,
        subsampled,
        K,
        J,
        L,
        n=n,
        random_state=rank_seed,
        subsample_indices=subsamples,
    )


def aggregated_scis_test(
    x: ArrayLike,
    z: ArrayLike,
    time: ArrayLike,
    event: ArrayLike,
    K: int = 3,
    J: int = 5,
    L: int = 6,
    *,
    random_state: int = 1,
    hazard_config: HazardConfig | None = None,
    direction_config: DirectionConfig | RandomForestDirectionConfig | None = None,
    generator_config: GeneratorConfig | None = None,
    test_config: SciTestConfig | None = None,
) -> AggregatedPValueResult:
    """Run scisTest repeatedly and return its Guo--Shah aggregated p-value.

    Every full-sample split and every subsample split refits all nuisance
    components. Consequently this function can require substantial compute for
    publication-scale ``J`` and ``L``.
    """

    x_array, z_array, time_array, event_array = validate_survival_data(
        x, z, time, event
    )
    K = int(K)
    if K < 2:
        raise ValueError("K must be at least two")
    subsample_size = len(time_array) // K
    if subsample_size < 12:
        raise ValueError(
            "floor(n / K) must be at least 12 because every subsample runs scisTest"
        )

    def statistic(indices: NDArray[np.int64], split_seed: int) -> float:
        result = scis_test(
            x_array[indices],
            z_array[indices],
            time_array[indices],
            event_array[indices],
            random_state=split_seed,
            hazard_config=hazard_config,
            direction_config=direction_config,
            generator_config=generator_config,
            test_config=test_config,
        )
        return result.test_statistic

    return aggregate_p_value(
        len(time_array),
        statistic,
        K,
        J,
        L,
        random_state=random_state,
    )
