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

from .config import DirectionConfig, GeneratorConfig, HazardConfig, SciTestConfig
from .core import sci_test, validate_survival_data

StatisticFunction = Callable[[NDArray[np.int64], int], float]


@dataclass(frozen=True)
class AggregatedPValueResult:
    """Output of rank-transformed subsampling calibration."""

    p_value: float
    p_value_decimal: str
    log_p_value: float
    log10_p_value: float
    empirical_p_value: float
    aggregate_statistic: float
    exceedance_count: int
    empirical_resolution: float
    bandwidth: float
    n: int
    Q: int
    m: int
    J: int
    L: int
    B: int
    blocks_per_permutation: int
    observed_statistics: NDArray[np.float64] = field(repr=False)
    subsample_statistics: NDArray[np.float64] = field(repr=False)
    rank_transformed_statistics: NDArray[np.float64] = field(repr=False)
    calibration_statistics: NDArray[np.float64] = field(repr=False)
    subsample_indices: NDArray[np.int64] | None = field(default=None, repr=False)

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly representation of the result."""

        output = asdict(self)
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


def _resolve_subsample_design(n: int, Q: int | None) -> tuple[int, int]:
    """Return integer ``(Q, m)`` with ``m = floor(n / Q)``.

    When ``Q`` is omitted, the default is ``floor(log(n))`` for the natural
    logarithm, with a minimum of two splits.
    """

    n = int(n)
    if n < 4:
        raise ValueError("n must be at least four")
    if Q is None:
        Q = max(2, math.floor(math.log(n)))
    elif isinstance(Q, (bool, np.bool_)) or int(Q) != Q:
        raise ValueError("Q must be an integer")
    Q = int(Q)
    if Q < 2:
        raise ValueError("Q must be at least two")
    m = n // Q
    if m < 2:
        raise ValueError("Q must be at most floor(n / 2), so that m >= 2")
    return Q, m


def make_subsamples(
    n: int,
    Q: int | None = None,
    J: int = 10,
    *,
    random_state: int = 1,
) -> NDArray[np.int64]:
    """Generate Algorithm 1 subsamples.

    Each of ``J`` independent permutations contributes exactly ``Q``
    disjoint blocks of size ``m = floor(n / Q)``. If ``Q`` is omitted, it is
    set to ``floor(log(n))``, where ``log`` is the natural logarithm.
    """

    n, J = int(n), int(J)
    Q, m = _resolve_subsample_design(n, Q)
    if J < 1:
        raise ValueError("J must be at least one")
    if int(random_state) < 0:
        raise ValueError("random_state must be nonnegative")
    rng = np.random.default_rng(int(random_state))
    return np.vstack([rng.permutation(n)[: Q * m].reshape(Q, m) for _ in range(J)])


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
    Q: int | None = None,
    J: int = 10,
    L: int = 3,
    *,
    n: int,
    random_state: int = 1,
    subsample_indices: NDArray[np.int64] | None = None,
) -> AggregatedPValueResult:
    """Calibrate already-computed split statistics.

    Parameters ``Q``, ``J``, and ``L`` are explicit so that saved statistic
    matrices cannot silently be analyzed under a different aggregation design.
    The subsample size is derived as ``m = floor(n / Q)``. If ``Q`` is omitted,
    it is set to ``floor(log(n))`` for the natural logarithm.
    The aggregated p-value uses the Gaussian-kernel-smoothed right tail from
    the authors' MultiSplit implementation; the unsmoothed empirical tail is
    also returned as a diagnostic.
    """

    n, J, L = int(n), int(J), int(L)
    Q, m = _resolve_subsample_design(n, Q)
    if J < 1 or L < 2:
        raise ValueError("Require J >= 1 and L >= 2")
    observed = np.asarray(observed_statistics, dtype=np.float64)
    subsampled = np.asarray(subsample_statistics, dtype=np.float64)
    if observed.shape != (L,):
        raise ValueError(f"observed_statistics must have shape ({L},)")
    if subsampled.ndim != 2 or subsampled.shape[1] != L:
        raise ValueError(f"subsample_statistics must have shape (B, {L})")
    if subsampled.shape[0] < 2:
        raise ValueError("The number of subsamples B must be at least two")
    if not np.isfinite(observed).all() or not np.isfinite(subsampled).all():
        raise ValueError("All statistics must be finite")

    B = len(subsampled)
    blocks_per_permutation = Q
    expected_B = J * Q
    if B != expected_B:
        raise ValueError(f"Expected B={expected_B} rows from Q={Q}, J={J}; got {B}")
    if subsample_indices is not None:
        subsample_indices = np.asarray(subsample_indices, dtype=np.int64)
        if subsample_indices.shape != (B, m):
            raise ValueError(f"subsample_indices must have shape ({B}, {m})")

    transformed = normal_rank_transform(subsampled, random_state=random_state)
    calibration = transformed.mean(axis=1, dtype=np.float64)
    aggregate_statistic = float(observed.mean(dtype=np.float64))
    exceedance_count = int(np.count_nonzero(calibration > aggregate_statistic))
    bandwidth = _bandwidth_nrd0(calibration)
    log_p_value = float(
        logsumexp(log_ndtr((calibration - aggregate_statistic) / bandwidth)) - math.log(B)
    )
    log_p_value = min(0.0, log_p_value)
    return AggregatedPValueResult(
        p_value=float(math.exp(log_p_value)),
        p_value_decimal=_decimal_from_log(log_p_value),
        log_p_value=log_p_value,
        log10_p_value=log_p_value / math.log(10),
        empirical_p_value=exceedance_count / B,
        aggregate_statistic=aggregate_statistic,
        exceedance_count=exceedance_count,
        empirical_resolution=1 / B,
        bandwidth=bandwidth,
        n=n,
        Q=Q,
        m=m,
        J=J,
        L=L,
        B=B,
        blocks_per_permutation=blocks_per_permutation,
        observed_statistics=observed,
        subsample_statistics=subsampled,
        rank_transformed_statistics=transformed,
        calibration_statistics=calibration,
        subsample_indices=subsample_indices,
    )


def aggregate_p_value(
    n: int,
    statistic: StatisticFunction,
    Q: int | None = None,
    J: int = 10,
    L: int = 3,
    *,
    random_state: int = 1,
) -> AggregatedPValueResult:
    """Compute a Guo--Shah aggregated p-value from a split-statistic callback.

    ``statistic(indices, split_seed)`` must refit the complete base procedure on
    the selected observations and return its one-sided, asymptotically
    standard-normal test statistic. The callback is invoked ``L`` times on the
    full sample and ``L`` times on every rank-calibration subsample.
    """

    n, J, L = int(n), int(J), int(L)
    Q, _ = _resolve_subsample_design(n, Q)
    if L < 2:
        raise ValueError("L must be at least two")
    if int(random_state) < 0:
        raise ValueError("random_state must be nonnegative")
    seed_sequence = np.random.SeedSequence(int(random_state))
    subsample_seed_sequence, fit_seed_sequence, rank_seed_sequence = seed_sequence.spawn(3)
    subsample_seed = int(subsample_seed_sequence.generate_state(1)[0])
    subsamples = make_subsamples(n, Q, J, random_state=subsample_seed)
    B = len(subsamples)
    fit_seeds = [int(child.generate_state(1)[0]) for child in fit_seed_sequence.spawn((B + 1) * L)]

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
        Q=Q,
        J=J,
        L=L,
        n=n,
        random_state=rank_seed,
        subsample_indices=subsamples,
    )


def aggregated_sci_test(
    x: ArrayLike,
    z: ArrayLike,
    time: ArrayLike,
    event: ArrayLike,
    Q: int | None = None,
    J: int = 10,
    L: int = 3,
    *,
    random_state: int = 1,
    hazard_config: HazardConfig | None = None,
    direction_config: DirectionConfig | None = None,
    generator_config: GeneratorConfig | None = None,
    test_config: SciTestConfig | None = None,
) -> AggregatedPValueResult:
    """Run sciTest repeatedly and return its Guo--Shah aggregated p-value.

    Every full-sample split and every subsample split refits all nuisance
    components. The subsample size is ``m = floor(n / Q)``. When ``Q`` is not
    supplied, ``floor(log(n))`` is used, where ``log`` is the natural logarithm.
    Consequently this function can require substantial compute for
    publication-scale ``J`` and ``L``.
    """

    x_array, z_array, time_array, event_array = validate_survival_data(x, z, time, event)
    resolved_Q, m = _resolve_subsample_design(len(time_array), Q)
    if m < 12:
        raise ValueError(
            "Q is too large: floor(n / Q) must be at least 12 because every subsample runs sciTest"
        )

    def statistic(indices: NDArray[np.int64], split_seed: int) -> float:
        result = sci_test(
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
        Q=resolved_Q,
        J=J,
        L=L,
        random_state=random_state,
    )
