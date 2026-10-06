"""Configuration objects for scisTest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HazardConfig:
    """Settings for the null conditional-hazard network."""

    hidden_dims: tuple[int, ...] = (32,)
    learning_rate: float = 0.01
    epochs: int = 100
    l1_penalty: float = 0.001
    batch_size: int = 128


@dataclass(frozen=True)
class DirectionConfig:
    """Settings for the default DNN direction estimator."""

    hidden_dims: tuple[int, ...] = (32, 32)
    learning_rate: float = 0.01
    epochs: int = 100


@dataclass(frozen=True)
class RandomForestDirectionConfig:
    """Settings for the optional random-forest direction estimator."""

    n_estimators: int = 100
    max_depth: int | None = 6
    min_samples_leaf: int = 5
    max_features: int | float | str | None = 1.0
    bootstrap: bool = True
    n_jobs: int | None = None
    solver_tolerance: float = 1e-8
    solver_max_iterations: int | None = None


@dataclass(frozen=True)
class GeneratorConfig:
    """Settings passed to the engression conditional generator."""

    learning_rate: float = 0.005
    epochs: int = 100
    batch_size: int = 128
    num_layers: int = 2
    hidden_dim: int = 100
    noise_dim: int = 100


@dataclass(frozen=True)
class SciTestConfig:
    """Numerical and inferential settings for one scisTest split."""

    train_fraction: float = 0.5
    integration_points: int = 100
    generator_draws: int = 100
    score_chunk_size: int = 16
    alpha: float = 0.05


def config_as_dict(config: object) -> dict[str, Any]:
    """Convert a configuration dataclass to a JSON-serializable dictionary."""

    output = asdict(config)
    for key, value in output.items():
        if isinstance(value, tuple):
            output[key] = list(value)
    return output
