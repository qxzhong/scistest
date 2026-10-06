"""Small data-generating mechanisms used by the examples and smoke simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulatedSurvivalData:
    """A simulated right-censored survival sample."""

    x: NDArray[np.float64]
    z: NDArray[np.float64]
    time: NDArray[np.float64]
    event: NDArray[np.float64]
    event_time: NDArray[np.float64]
    censoring_time: NDArray[np.float64]


def simulate_survival_data(
    n: int = 400,
    *,
    effect: float = 0.0,
    censoring_rate: float = 0.65,
    random_state: int = 1,
) -> SimulatedSurvivalData:
    """Simulate nonlinear covariates and exponential event/censoring times.

    When ``effect=0``, the event time is conditionally independent of ``X``
    given ``Z``. A positive effect creates a conditional alternative while
    retaining independent censoring given the observed covariates.
    """

    n = int(n)
    if n < 12:
        raise ValueError("n must be at least 12")
    if censoring_rate <= 0:
        raise ValueError("censoring_rate must be positive")
    rng = np.random.default_rng(int(random_state))
    z = rng.normal(size=(n, 3))
    x_signal = 0.7 * z[:, 0] - 0.4 * z[:, 1] + 0.5 * np.sin(z[:, 2])
    x = (x_signal + rng.normal(scale=0.8, size=n))[:, None]

    log_event_rate = (
        -0.3
        + 0.45 * z[:, 0]
        - 0.35 * z[:, 1]
        + 0.25 * z[:, 0] * z[:, 2]
        + effect * (0.7 * x[:, 0] + 0.3 * x[:, 0] ** 2)
    )
    event_rate = np.exp(np.clip(log_event_rate, -5.0, 5.0))
    event_time = rng.exponential(scale=1.0 / event_rate)

    # C may depend on observed covariates but is drawn independently of U
    # conditional on (X, Z), as required by conditional independent censoring.
    log_censor_rate = math.log(censoring_rate) + 0.15 * z[:, 0] - 0.1 * x[:, 0]
    censor_rate = np.exp(np.clip(log_censor_rate, -5.0, 5.0))
    censoring_time = rng.exponential(scale=1.0 / censor_rate)
    time = np.minimum(event_time, censoring_time)
    event = (event_time <= censoring_time).astype(np.float64)
    return SimulatedSurvivalData(
        x=x,
        z=z,
        time=time,
        event=event,
        event_time=event_time,
        censoring_time=censoring_time,
    )
