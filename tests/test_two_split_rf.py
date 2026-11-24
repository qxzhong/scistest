from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.stats import norm

from scistest import GeneratorConfig, SciTestConfig, core, forest_configs, scis_test
from scistest._forest import fit_direction, fit_hazard


@pytest.mark.parametrize("n", [200, 400, 1000])
def test_rf_defaults_match_manuscript_table(n):
    configs = forest_configs(n)
    hazard, direction = configs["hazard_config"], configs["direction_config"]
    assert hazard.n_estimators == 32 and direction.n_estimators == 16
    assert hazard.max_depth == direction.max_depth == 3
    assert hazard.min_samples_leaf == direction.min_samples_leaf == 20
    assert hazard.max_features == direction.max_features == .8
    assert hazard.l1_penalty == 5 / n and direction.l1_penalty == 50 / n
    assert configs["test_config"].integration_points == 100
    assert configs["test_config"].generator_draws == 100


def test_rf_leaf_losses_equal_manuscript_formulas():
    torch.set_num_threads(1)
    torch.manual_seed(91)
    n, points = 60, 10
    z, x = torch.randn(n, 2), torch.randn(n, 1)
    time = torch.linspace(.01, .8, n)
    event = torch.tensor([0., 1.] * (n // 2))
    configs = forest_configs(n)
    hc = replace(configs["hazard_config"], n_estimators=4, min_samples_leaf=5)
    dc = replace(configs["direction_config"], n_estimators=4, min_samples_leaf=5)
    hazard = fit_hazard(z, time, event, hc, points, 4)
    direction, reported = fit_direction(hazard, z, x, time, event, dc, points, 5)
    grid = core._right_riemann_grid(time, points)
    z_grid = z[:, None, :].expand(n, points, 2).reshape(-1, 2)
    integral = hazard.hazard(grid.flatten(), z_grid).reshape(n, points).double().sum(1)
    integral *= time.double() / points
    expected_hazard = (integral - event * hazard.log_hazard(time, z)).mean()
    expected_hazard += hc.l1_penalty * hazard.diagnostics["l1_norm"]
    assert float(expected_hazard) == pytest.approx(hazard.diagnostics["objective"], abs=1e-6)
    scores = core._raw_direction_scores(direction, hazard, z, x, time, event, points).double()
    expected_direction = (scores**2 - scores).sum()
    expected_direction += dc.l1_penalty * direction.diagnostics["l1_norm"]
    assert reported == pytest.approx(float(expected_direction), abs=1e-5)


def test_rf_selector_fits_two_hazards_and_uses_evaluation_hazard(monkeypatch):
    torch.set_num_threads(1)
    rng = np.random.default_rng(14)
    x, z = rng.normal(size=(2, 120, 1))
    time, event = rng.exponential(size=120), np.tile([0, 1], 60)
    fits = []
    original_fit, original_score = core._fit_hazard, core._orthogonalized_scores

    def fit(*args):
        model = original_fit(*args)
        fits.append((model, args[0].clone(), args[3]))
        return model

    def score(hazard, *args, **kwargs):
        assert len(fits) == 2 and hazard is fits[1][0]
        return original_score(hazard, *args, **kwargs)

    monkeypatch.setattr(core, "_fit_hazard", fit)
    monkeypatch.setattr(core, "_orthogonalized_scores", score)
    result = scis_test(
        x, z, time, event, estimator="rf", random_state=42,
        generator_config=GeneratorConfig(epochs=2, hidden_dim=4, noise_dim=4),
        test_config=SciTestConfig(integration_points=10, generator_draws=10),
    )
    assert fits[0][2] == fits[1][2] == forest_configs(120)["hazard_config"]
    assert not torch.equal(fits[0][1], fits[1][1])
    assert set(result.nuisance_diagnostics) == {"hazard", "hazard_evaluation", "direction"}
    assert result.direction_objective == "forest_quadratic"
    assert set(result.train_indices) | set(result.test_indices) == set(range(120))
    assert not set(result.train_indices) & set(result.test_indices)
    expected = np.sqrt(result.n_test) * result.scores.mean() / result.scores.std(ddof=0)
    assert result.test_statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(norm.sf(expected))


def test_centering_and_score_match_direct_risk_set_formula():
    class Hazard:
        def hazard(self, t, z):
            return torch.full_like(t, .7)

    def direction(t, z, x):
        return t + z[:, 0] + 2 * x[:, 0]

    class Generator:
        def sample(self, z, draws):
            assert draws == 3
            return torch.tensor([[[1.], [2.], [4.]]]), torch.tensor([[.1, .5, .7]])

    time, z = torch.tensor([1.]), torch.tensor([[.2]])
    x, event = torch.tensor([[3.]]), torch.ones(1)
    actual = core._orthogonalized_scores(
        Hazard(), direction, Generator(), z, x, time, event, 4, 3, 1
    )
    grid = np.arange(1, 5) / 4
    generated_x, generated_time = np.array([1., 2., 4.]), np.array([.1, .5, .7])
    centered = []
    for s in grid:
        at_risk = generated_time >= s
        projection = (s + .2 + 2 * generated_x[at_risk]).mean() if at_risk.any() else 0.
        centered.append(s + .2 + 6 - projection)
    expected = centered[-1] - .7 * np.mean(centered)
    assert actual.item() == pytest.approx(expected, abs=1e-6)


def test_positive_small_score_variance_preserves_scale_invariance(monkeypatch):
    monkeypatch.setattr(core, "_fit_hazard", lambda *a: object())
    monkeypatch.setattr(core, "_fit_direction", lambda *a: (object(), 0.))
    monkeypatch.setattr(core.EngressionGenerator, "fit", lambda *a: None)
    scores = torch.linspace(-1., 2., 10, dtype=torch.float64) * 1e-15
    monkeypatch.setattr(core, "_orthogonalized_scores", lambda *a, **k: scores)
    result = scis_test(np.arange(20), np.arange(20), np.arange(1, 21), np.ones(20))
    expected = np.sqrt(10) * scores.numpy().mean() / scores.numpy().std(ddof=0)
    assert result.test_statistic == pytest.approx(expected)


def test_invalid_estimator_fails_before_fitting():
    with pytest.raises(ValueError, match="estimator"):
        scis_test(np.ones(12), np.ones(12), np.ones(12), np.ones(12), estimator="invalid")
