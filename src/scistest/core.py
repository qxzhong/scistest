"""Single-split scisTest for conditional independence with right censoring."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, hstack
from scipy.sparse.linalg import lsqr
from scipy.stats import norm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._models import (
    DirectionNetwork,
    EngressionGenerator,
    HazardNetwork,
    RandomForestDirection,
)
from .config import (
    DirectionConfig,
    GeneratorConfig,
    HazardConfig,
    RandomForestDirectionConfig,
    SciTestConfig,
)

DirectionEstimatorConfig = DirectionConfig | RandomForestDirectionConfig
DirectionModel = DirectionNetwork | RandomForestDirection


@dataclass(frozen=True)
class SciTestResult:
    """Results from one random hunting/test split."""

    p_value: float
    log_p_value: float
    test_statistic: float
    reject_null: bool
    alpha: float
    n_train: int
    n_test: int
    score_mean: float
    score_sd: float
    standardized_score_mean: float
    direction_loss: float
    time_scale: float
    random_state: int
    train_indices: NDArray[np.int64] = field(repr=False)
    test_indices: NDArray[np.int64] = field(repr=False)
    scores: NDArray[np.float64] = field(repr=False)

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly representation of the result."""

        output = asdict(self)
        array_keys = ("train_indices", "test_indices", "scores")
        if include_arrays:
            for key in array_keys:
                output[key] = output[key].tolist()
        else:
            for key in array_keys:
                output.pop(key)
        return output


@dataclass(frozen=True)
class _PreparedData:
    x_train: torch.Tensor
    z_train: torch.Tensor
    time_train: torch.Tensor
    event_train: torch.Tensor
    x_test: torch.Tensor
    z_test: torch.Tensor
    time_test: torch.Tensor
    event_test: torch.Tensor
    train_indices: torch.Tensor
    test_indices: torch.Tensor
    time_scale: float


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a one- or two-dimensional numeric array")
    return array


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    return array


def validate_survival_data(
    x: ArrayLike,
    z: ArrayLike,
    time: ArrayLike,
    event: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate inputs and return canonical NumPy arrays."""

    x_array = _matrix(x, "x")
    z_array = _matrix(z, "z")
    time_array = _vector(time, "time")
    event_array = _vector(event, "event")
    n = len(time_array)
    if not (len(x_array) == len(z_array) == len(event_array) == n):
        raise ValueError("x, z, time, and event must have the same number of rows")
    if n < 12:
        raise ValueError("scisTest requires at least 12 observations")
    if not all(np.isfinite(array).all() for array in (x_array, z_array, time_array, event_array)):
        raise ValueError("Inputs must not contain NaN or infinite values")
    if np.any(time_array < 0) or np.max(time_array) <= 0:
        raise ValueError("time must be nonnegative with at least one positive value")
    if not np.all(np.isin(event_array, (0.0, 1.0))):
        raise ValueError("event must contain only 0 and 1")
    return x_array, z_array, time_array, event_array


def _validate_configs(
    hazard: HazardConfig,
    direction: DirectionEstimatorConfig,
    generator: GeneratorConfig,
    test: SciTestConfig,
) -> None:
    if not 0 < test.train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if not 0 < test.alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    integer_values: dict[str, int] = {
        "hazard epochs": hazard.epochs,
        "hazard batch_size": hazard.batch_size,
        "generator epochs": generator.epochs,
        "generator batch_size": generator.batch_size,
        "generator num_layers": generator.num_layers,
        "generator hidden_dim": generator.hidden_dim,
        "generator noise_dim": generator.noise_dim,
        "integration_points": test.integration_points,
        "generator_draws": test.generator_draws,
        "score_chunk_size": test.score_chunk_size,
    }
    if isinstance(direction, DirectionConfig):
        integer_values["direction epochs"] = direction.epochs
    else:
        integer_values.update(
            {
                "direction n_estimators": direction.n_estimators,
                "direction min_samples_leaf": direction.min_samples_leaf,
            }
        )
        if direction.max_depth is not None:
            integer_values["direction max_depth"] = direction.max_depth
        if (
            direction.solver_max_iterations is not None
            and direction.solver_max_iterations < 1
        ):
            raise ValueError("direction solver_max_iterations must be positive or None")
        if direction.solver_tolerance <= 0:
            raise ValueError("direction solver_tolerance must be positive")
    if any(value < 1 for value in integer_values.values()):
        bad = ", ".join(name for name, value in integer_values.items() if value < 1)
        raise ValueError(f"The following settings must be positive integers: {bad}")
    positive_values: dict[str, float] = {
        "hazard learning_rate": hazard.learning_rate,
        "generator learning_rate": generator.learning_rate,
    }
    if isinstance(direction, DirectionConfig):
        positive_values["direction learning_rate"] = direction.learning_rate
    if any(value <= 0 for value in positive_values.values()):
        bad = ", ".join(name for name, value in positive_values.items() if value <= 0)
        raise ValueError(f"The following settings must be positive: {bad}")
    if hazard.l1_penalty < 0:
        raise ValueError("The hazard L1 penalty must be nonnegative")


def _prepare_data(
    x: NDArray[np.float64],
    z: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    train_fraction: float,
    split_seed: int,
) -> _PreparedData:
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    z_tensor = torch.as_tensor(z, dtype=torch.float32)
    time_tensor = torch.as_tensor(time, dtype=torch.float32)
    event_tensor = torch.as_tensor(event, dtype=torch.float32)

    split_generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(len(time_tensor), generator=split_generator)
    n_train = int(len(time_tensor) * train_fraction)
    if n_train < 2 or len(time_tensor) - n_train < 2:
        raise ValueError("The requested split leaves fewer than two observations in one sample")
    train_indices = permutation[:n_train]
    test_indices = permutation[n_train:]

    # Estimate preprocessing constants on the hunting sample only.
    x_min = x_tensor[train_indices].amin(dim=0)
    x_range = x_tensor[train_indices].amax(dim=0) - x_min
    x_range = torch.where(x_range > 1e-8, x_range, torch.ones_like(x_range))
    z_mean = z_tensor[train_indices].mean(dim=0)
    z_scale = z_tensor[train_indices].std(dim=0, unbiased=False)
    z_scale = torch.where(z_scale > 1e-8, z_scale, torch.ones_like(z_scale))
    time_scale_tensor = time_tensor[train_indices].amax()
    if not torch.isfinite(time_scale_tensor) or time_scale_tensor <= 0:
        raise ValueError("The hunting sample must contain at least one positive observed time")

    x_scaled = (x_tensor - x_min) / x_range
    z_scaled = (z_tensor - z_mean) / z_scale
    time_scaled = time_tensor / time_scale_tensor

    return _PreparedData(
        x_train=x_scaled[train_indices],
        z_train=z_scaled[train_indices],
        time_train=time_scaled[train_indices],
        event_train=event_tensor[train_indices],
        x_test=x_scaled[test_indices],
        z_test=z_scaled[test_indices],
        time_test=time_scaled[test_indices],
        event_test=event_tensor[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        time_scale=float(time_scale_tensor),
    )


def _right_riemann_grid(time: torch.Tensor, points: int) -> torch.Tensor:
    fractions = torch.arange(1, points + 1, dtype=time.dtype, device=time.device) / points
    return time[:, None] * fractions[None, :]


def _l1_norm(module: nn.Module) -> torch.Tensor:
    return sum(parameter.abs().sum() for parameter in module.parameters())


def _fit_hazard(
    z: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    config: HazardConfig,
    integration_points: int,
    seed: int,
) -> HazardNetwork:
    _set_random_seeds(seed)
    model = HazardNetwork(z.shape[1], config.hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(
        TensorDataset(z, time, event),
        batch_size=min(config.batch_size, len(time)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    for _ in range(config.epochs):
        model.train()
        for z_batch, time_batch, event_batch in loader:
            grid = _right_riemann_grid(time_batch, integration_points)
            batch_size = len(time_batch)
            z_grid = (
                z_batch[:, None, :]
                .expand(batch_size, integration_points, z.shape[1])
                .reshape(-1, z.shape[1])
            )
            hazard_grid = model.hazard(grid.reshape(-1), z_grid).reshape(
                batch_size, integration_points
            )
            integrated_hazard = hazard_grid.sum(dim=1) * time_batch / integration_points
            log_likelihood = (
                event_batch * model.log_hazard(time_batch, z_batch) - integrated_hazard
            )
            loss = -log_likelihood.mean() + config.l1_penalty * _l1_norm(model)
            if not torch.isfinite(loss):
                raise RuntimeError("The hazard loss became non-finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _raw_direction_scores(
    direction: DirectionModel,
    hazard: HazardNetwork,
    z: torch.Tensor,
    x: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    integration_points: int,
) -> torch.Tensor:
    grid = _right_riemann_grid(time, integration_points)
    n = len(time)
    z_grid = z[:, None, :].expand(n, integration_points, z.shape[1]).reshape(-1, z.shape[1])
    x_grid = x[:, None, :].expand(n, integration_points, x.shape[1]).reshape(-1, x.shape[1])
    with torch.no_grad():
        hazard_grid = hazard.hazard(grid.reshape(-1), z_grid).reshape(n, integration_points)
    phi_grid = direction(grid.reshape(-1), z_grid, x_grid).reshape(n, integration_points)
    integral = (hazard_grid * phi_grid).sum(dim=1) * time / integration_points
    return event * direction(time, z, x) - integral


def _quadratic_direction_loss(scores: torch.Tensor) -> torch.Tensor:
    """The unpenalized criterion ``sum(V_i^2) - sum(V_i)``."""

    return scores.square().sum() - scores.sum()


def _fit_dnn_direction(
    hazard: HazardNetwork,
    z: torch.Tensor,
    x: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    config: DirectionConfig,
    integration_points: int,
    seed: int,
) -> tuple[DirectionNetwork, float]:
    _set_random_seeds(seed)
    direction = DirectionNetwork(z.shape[1], x.shape[1], config.hidden_dims)
    optimizer = torch.optim.Adam(direction.parameters(), lr=config.learning_rate)

    for _ in range(config.epochs):
        direction.train()
        scores = _raw_direction_scores(
            direction, hazard, z, x, time, event, integration_points
        )
        loss = _quadratic_direction_loss(scores)
        if not torch.isfinite(loss):
            raise RuntimeError("The direction loss became non-finite")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Report the criterion attained by the final fixed-epoch fit.
    with torch.no_grad():
        final_scores = _raw_direction_scores(
            direction, hazard, z, x, time, event, integration_points
        )
        final_loss = _quadratic_direction_loss(final_scores)
    if not torch.isfinite(final_loss):
        raise RuntimeError("The final direction loss is non-finite")
    direction.eval()
    for parameter in direction.parameters():
        parameter.requires_grad_(False)
    return direction, float(final_loss)


def _fit_random_forest_direction(
    hazard: HazardNetwork,
    z: torch.Tensor,
    x: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    config: RandomForestDirectionConfig,
    integration_points: int,
    seed: int,
) -> tuple[RandomForestDirection, float]:
    """Fit forest partitions and solve their leaf values under the paper's loss."""

    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - installation error path
        raise ImportError(
            "The random-forest direction estimator requires scikit-learn"
        ) from exc

    _set_random_seeds(seed)
    n = len(time)
    grid = _right_riemann_grid(time, integration_points)
    z_grid = z[:, None, :].expand(n, integration_points, z.shape[1])
    x_grid = x[:, None, :].expand(n, integration_points, x.shape[1])
    flat_z = z_grid.reshape(-1, z.shape[1])
    flat_x = x_grid.reshape(-1, x.shape[1])
    flat_time = grid.reshape(-1)
    with torch.no_grad():
        hazard_grid = hazard.hazard(flat_time, flat_z).reshape(n, integration_points)
        coefficients = -hazard_grid * time[:, None] / integration_points
        coefficients[:, -1] += event

    features = (
        torch.cat((flat_time[:, None], flat_z, flat_x), dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    coefficient_values = coefficients.detach().cpu().numpy().astype(np.float64)

    # The pointwise coefficients are the negative functional gradient at phi=0;
    # fitting them supplies data-adaptive random-forest partitions. The original
    # subject-level objective, rather than this auxiliary regression loss, is
    # used below to determine all terminal-node values.
    forest = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        bootstrap=config.bootstrap,
        n_jobs=config.n_jobs,
        random_state=seed,
    )
    forest.fit(features, coefficient_values.reshape(-1))

    subject_rows = np.repeat(np.arange(n, dtype=np.int64), integration_points)
    flat_coefficients = coefficient_values.reshape(-1)
    design_blocks = []
    leaf_node_ids: list[NDArray[np.int64]] = []
    for estimator in forest.estimators_:
        nodes = estimator.apply(features)
        unique_nodes, local_columns = np.unique(nodes, return_inverse=True)
        block = coo_matrix(
            (flat_coefficients, (subject_rows, local_columns)),
            shape=(n, len(unique_nodes)),
        ).tocsr()
        design_blocks.append(block / config.n_estimators)
        leaf_node_ids.append(unique_nodes.astype(np.int64, copy=False))

    # sum(V_i^2 - V_i) = sum((V_i - 1/2)^2) - n/4.  For fixed forest
    # partitions, V is linear in the leaf values, so sparse least squares gives
    # the minimizer over the resulting random-forest function class.
    design = hstack(design_blocks, format="csr")
    solution = lsqr(
        design,
        np.full(n, 0.5, dtype=np.float64),
        atol=config.solver_tolerance,
        btol=config.solver_tolerance,
        iter_lim=config.solver_max_iterations,
    )[0]
    if not np.isfinite(solution).all():
        raise RuntimeError("The random-forest direction fit became non-finite")

    leaf_values: list[NDArray[np.float64]] = []
    position = 0
    for estimator, nodes in zip(forest.estimators_, leaf_node_ids, strict=True):
        values = np.zeros(estimator.tree_.node_count, dtype=np.float64)
        values[nodes] = solution[position : position + len(nodes)]
        leaf_values.append(values)
        position += len(nodes)

    direction = RandomForestDirection(forest, leaf_values)
    with torch.no_grad():
        final_scores = _raw_direction_scores(
            direction, hazard, z, x, time, event, integration_points
        )
        final_loss = _quadratic_direction_loss(final_scores)
    if not torch.isfinite(final_loss):
        raise RuntimeError("The final random-forest direction loss is non-finite")
    return direction, float(final_loss)


def _fit_direction(
    hazard: HazardNetwork,
    z: torch.Tensor,
    x: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    config: DirectionEstimatorConfig,
    integration_points: int,
    seed: int,
) -> tuple[DirectionModel, float]:
    if isinstance(config, RandomForestDirectionConfig):
        return _fit_random_forest_direction(
            hazard, z, x, time, event, config, integration_points, seed
        )
    return _fit_dnn_direction(
        hazard, z, x, time, event, config, integration_points, seed
    )


def _orthogonalized_scores(
    hazard: HazardNetwork,
    direction: DirectionModel,
    conditional_generator: EngressionGenerator,
    z: torch.Tensor,
    x: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    integration_points: int,
    generator_draws: int,
    chunk_size: int,
) -> torch.Tensor:
    score_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(time), chunk_size):
            stop = min(start + chunk_size, len(time))
            z_chunk = z[start:stop]
            x_chunk = x[start:stop]
            time_chunk = time[start:stop]
            event_chunk = event[start:stop]
            batch_size = len(time_chunk)

            generated_x, generated_time = conditional_generator.sample(
                z_chunk, generator_draws
            )
            grid = _right_riemann_grid(time_chunk, integration_points)
            z_grid = (
                z_chunk[:, None, :]
                .expand(batch_size, integration_points, z.shape[1])
                .reshape(-1, z.shape[1])
            )
            x_grid = (
                x_chunk[:, None, :]
                .expand(batch_size, integration_points, x.shape[1])
                .reshape(-1, x.shape[1])
            )
            time_grid = grid.reshape(-1)
            hazard_grid = hazard.hazard(time_grid, z_grid).reshape(
                batch_size, integration_points
            )
            observed_phi_grid = direction(time_grid, z_grid, x_grid).reshape(
                batch_size, integration_points
            )

            expanded_time = grid[:, :, None].expand(
                batch_size, integration_points, generator_draws
            )
            expanded_z = z_chunk[:, None, None, :].expand(
                batch_size, integration_points, generator_draws, z.shape[1]
            )
            expanded_x = generated_x[:, None, :, :].expand(
                batch_size, integration_points, generator_draws, x.shape[1]
            )
            generated_phi = direction(
                expanded_time.reshape(-1),
                expanded_z.reshape(-1, z.shape[1]),
                expanded_x.reshape(-1, x.shape[1]),
            ).reshape(batch_size, integration_points, generator_draws)
            at_risk = expanded_time <= generated_time[:, None, :]
            denominator = at_risk.sum(dim=2)
            numerator = (at_risk * generated_phi).sum(dim=2)
            m_phi_grid = torch.where(
                denominator > 0,
                numerator / denominator.clamp_min(1),
                torch.zeros_like(numerator),
            )

            centered_phi_grid = observed_phi_grid.double() - m_phi_grid.double()
            integral = (
                (hazard_grid.double() * centered_phi_grid).sum(dim=1)
                * time_chunk.double()
                / integration_points
            )
            # The last right-Riemann endpoint equals the observed time.
            event_term = event_chunk.double() * (
                direction(time_chunk, z_chunk, x_chunk).double()
                - m_phi_grid[:, -1].double()
            )
            score_chunks.append(event_term - integral)
    return torch.cat(score_chunks)


def scis_test(
    x: ArrayLike,
    z: ArrayLike,
    time: ArrayLike,
    event: ArrayLike,
    *,
    random_state: int = 1,
    hazard_config: HazardConfig | None = None,
    direction_config: DirectionEstimatorConfig | None = None,
    generator_config: GeneratorConfig | None = None,
    test_config: SciTestConfig | None = None,
) -> SciTestResult:
    """Test ``U independent of X given Z`` from right-censored observations.

    ``time`` is the observed time ``min(U, C)`` and ``event`` is one when the
    event is observed. The returned p-value is the one-sided standard-normal
    tail probability used by the paper.
    """

    x_array, z_array, time_array, event_array = validate_survival_data(
        x, z, time, event
    )
    hazard_config = hazard_config or HazardConfig()
    direction_config = direction_config or DirectionConfig()
    generator_config = generator_config or GeneratorConfig()
    test_config = test_config or SciTestConfig()
    _validate_configs(hazard_config, direction_config, generator_config, test_config)
    random_state = int(random_state)
    if random_state < 0:
        raise ValueError("random_state must be nonnegative")
    _set_random_seeds(random_state)

    data = _prepare_data(
        x_array,
        z_array,
        time_array,
        event_array,
        train_fraction=test_config.train_fraction,
        split_seed=random_state + 1,
    )
    hazard = _fit_hazard(
        data.z_train,
        data.time_train,
        data.event_train,
        hazard_config,
        test_config.integration_points,
        random_state + 2,
    )
    direction, direction_loss = _fit_direction(
        hazard,
        data.z_train,
        data.x_train,
        data.time_train,
        data.event_train,
        direction_config,
        test_config.integration_points,
        random_state + 3,
    )

    # The estimator of P_(X,T)|Z is fitted on the held-out test sample.
    _set_random_seeds(random_state + 4)
    conditional_generator = EngressionGenerator(x_dim=data.x_test.shape[1])
    conditional_generator.fit(
        data.z_test, data.x_test, data.time_test, generator_config
    )
    _set_random_seeds(random_state + 5)
    scores = _orthogonalized_scores(
        hazard,
        direction,
        conditional_generator,
        data.z_test,
        data.x_test,
        data.time_test,
        data.event_test,
        integration_points=test_config.integration_points,
        generator_draws=test_config.generator_draws,
        chunk_size=test_config.score_chunk_size,
    ).to(torch.float64)

    n_test = len(scores)
    score_mean = scores.mean()
    # This matches the paper's n^{-1} variance convention (ddof=0).
    score_sd = torch.sqrt(torch.mean((scores - score_mean).square()))
    if not torch.isfinite(scores).all() or not torch.isfinite(score_sd) or score_sd <= 1e-12:
        raise RuntimeError("The held-out scores are non-finite or have zero variance")

    standardized_score_mean = score_mean / score_sd
    test_statistic = math.sqrt(n_test) * float(standardized_score_mean)
    p_value = float(norm.sf(test_statistic))
    log_p_value = float(norm.logsf(test_statistic))
    if not np.isfinite(test_statistic) or not np.isfinite(log_p_value):
        raise RuntimeError("The test returned a non-finite statistic or log p-value")

    return SciTestResult(
        p_value=p_value,
        log_p_value=log_p_value,
        test_statistic=test_statistic,
        reject_null=bool(log_p_value < math.log(test_config.alpha)),
        alpha=test_config.alpha,
        n_train=len(data.time_train),
        n_test=n_test,
        score_mean=float(score_mean),
        score_sd=float(score_sd),
        standardized_score_mean=float(standardized_score_mean),
        direction_loss=direction_loss,
        time_scale=data.time_scale,
        random_state=random_state,
        train_indices=data.train_indices.numpy(),
        test_indices=data.test_indices.numpy(),
        scores=scores.numpy(),
    )


# Camel-case alias for the name used in the paper. PEP 8 users can call scis_test.
scisTest = scis_test
