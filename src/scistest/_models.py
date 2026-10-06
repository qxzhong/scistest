"""Internal neural-network and engression model definitions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .config import GeneratorConfig


class FeedForwardNetwork(nn.Module):
    """A ReLU feedforward network with a scalar output."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class HazardNetwork(nn.Module):
    """Represent the null hazard as ``exp(g(t, z))``."""

    def __init__(self, z_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.log_hazard_network = FeedForwardNetwork(z_dim + 1, hidden_dims)

    def log_hazard(self, time: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.log_hazard_network(torch.cat((time, z), dim=1))

    def hazard(self, time: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # The clamp is solely a floating-point overflow guard during optimization.
        return torch.exp(torch.clamp(self.log_hazard(time, z), -15.0, 15.0))


class DirectionNetwork(nn.Module):
    """Represent the data-adaptive direction ``phi(t, z, x)``."""

    def __init__(self, z_dim: int, x_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.phi_network = FeedForwardNetwork(z_dim + x_dim + 1, hidden_dims)

    def forward(
        self,
        time: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.phi_network(torch.cat((time, z, x), dim=1))


class RandomForestDirection:
    """Evaluate a forest whose leaf values minimize the direction objective."""

    def __init__(self, forest: Any, leaf_values: list[NDArray[np.float64]]):
        self.forest = forest
        self.leaf_values = leaf_values

    def __call__(
        self,
        time: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        features = torch.cat((time, z, x), dim=1).detach().cpu().numpy()
        predictions = np.zeros(len(features), dtype=np.float64)
        for estimator, values in zip(
            self.forest.estimators_, self.leaf_values, strict=True
        ):
            predictions += values[estimator.apply(features)]
        predictions /= len(self.forest.estimators_)
        return torch.as_tensor(
            predictions,
            dtype=time.dtype,
            device=time.device,
        )


class EngressionGenerator:
    """Estimate and sample from the joint law of ``(X, T) | Z``."""

    def __init__(self, x_dim: int):
        self.x_dim = x_dim
        self.model: Any | None = None

    def fit(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        time: torch.Tensor,
        config: GeneratorConfig,
    ) -> None:
        try:
            from engression import engression
        except ImportError as exc:  # pragma: no cover - installation error path
            raise ImportError(
                "scisTest requires the 'engression' package; install the package dependencies"
            ) from exc

        response = torch.cat((x, time[:, None]), dim=1)
        self.model = engression(
            z,
            response,
            lr=config.learning_rate,
            num_epochs=config.epochs,
            batch_size=config.batch_size,
            num_layer=config.num_layers,
            hidden_dim=config.hidden_dim,
            noise_dim=config.noise_dim,
            device="cpu",
            verbose=False,
        )

    def sample(self, z: torch.Tensor, draws: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return conditional draws with shape ``(subjects, draws, dimension)``."""

        if self.model is None:
            raise RuntimeError("The conditional generator has not been fitted")
        generated = self.model.sample(z, sample_size=draws, expand_dim=True)
        # engression returns (subjects, response dimension, draws).
        if generated.ndim == 2:  # engression squeezes the draw axis when draws == 1
            generated = generated.unsqueeze(-1)
        generated = generated.permute(0, 2, 1)
        generated_x = generated[:, :, : self.x_dim]
        # Engression has an unconstrained output; positivity enforces the time support.
        generated_time = generated[:, :, self.x_dim].abs()
        return generated_x, generated_time
