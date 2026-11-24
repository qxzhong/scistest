from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scistest import DirectionConfig, GeneratorConfig, HazardConfig, SciTestConfig, core, scis_test
from scistest.manuscript import forest_configs, framingham_configs, simulation_configs
from simulations.protocol import Cell, array_sha256, deterministic_seed, generate_frame


def test_paper_profile_uses_total_n_and_fixed_100_point_grid():
    configs = simulation_configs(1000)
    assert configs["hazard_config"] == HazardConfig((63, 63, 63), 0.002, 115, 0.005, 128)
    assert configs["direction_config"].l1_penalty == 0.005
    assert configs["generator_config"].num_layers == 4
    assert configs["test_config"].integration_points == 100
    assert framingham_configs()["direction_config"].objective == "standardized"


@pytest.mark.parametrize(
    "cell,rep,master,expected_time,expected_seed",
    [
        (
            Cell("case1", 1, "none", 1000, hypothesis=1),
            86,
            2026111703,
            [0.4089170203249339, 0.9890354294677943],
            1482714797,
        ),
        (
            Cell("case2", 2, "none", 1000, a=0),
            96,
            2026111602,
            [0.36697877629252584, 1.3707795785909798],
            3843778849,
        ),
        (
            Cell("case3a", 3, "a", 1000, a=0),
            146,
            2026111602,
            [1.0899334986601117, 0.4030866906054094],
            3277797003,
        ),
    ],
)
def test_dgp_matches_cluster_manifest(cell, rep, master, expected_time, expected_seed):
    frame = generate_frame(cell, rep, master)
    # exp/sin can differ by an ulp between Linux and macOS/NumPy versions.
    np.testing.assert_allclose(frame["T"].iloc[:2], expected_time, rtol=1e-13)
    assert deterministic_seed(cell, rep, 1, master) == expected_seed
    assert array_sha256(frame) == array_sha256(generate_frame(cell, rep, master))


@pytest.mark.parametrize("objective", ["quadratic", "standardized"])
def test_single_split_scores_with_refitted_evaluation_hazard(monkeypatch, objective):
    # Spy on the real fits and score call: catch accidental pilot-hazard reuse.
    fits = []
    original_fit = core._fit_hazard
    original_score = core._orthogonalized_scores

    def fit(*args):
        model = original_fit(*args)
        fits.append((model, args[0].clone()))
        return model

    def score(hazard, *args, **kwargs):
        assert len(fits) == 2
        assert hazard is fits[1][0]
        return original_score(hazard, *args, **kwargs)

    monkeypatch.setattr(core, "_fit_hazard", fit)
    monkeypatch.setattr(core, "_orthogonalized_scores", score)
    rng = np.random.default_rng(91)
    x, z = rng.normal(size=(2, 40, 1))
    t = rng.exponential(size=40)
    event = rng.integers(0, 2, size=40)
    torch.set_num_threads(1)
    result = scis_test(
        x,
        z,
        t,
        event,
        random_state=19,
        hazard_config=HazardConfig((4,), epochs=2),
        direction_config=DirectionConfig((4,), epochs=2, objective=objective),
        generator_config=GeneratorConfig(epochs=2, hidden_dim=4, noise_dim=4),
        test_config=SciTestConfig(integration_points=5, generator_draws=5),
    )
    assert len(fits) == 2
    assert not torch.equal(fits[0][1], fits[1][1])
    assert result.scoring_hazard == "evaluation"
    assert result.direction_objective == objective
    assert np.isfinite(result.test_statistic)
    np.testing.assert_allclose(
        result.test_statistic, np.sqrt(result.n_test) * result.scores.mean() / result.scores.std()
    )
    assert not set(result.train_indices) & set(result.test_indices)


def test_invalid_objective_fails_before_fitting():
    with pytest.raises(ValueError, match="objective"):
        core._validate_configs(
            HazardConfig(),
            replace(DirectionConfig(), objective="invalid"),
            GeneratorConfig(),
            SciTestConfig(),
        )


def test_omitted_configs_produce_same_fit_as_explicit_simulation_profile():
    rng = np.random.default_rng(63)
    x, z = rng.normal(size=(2, 24, 1))
    t = rng.exponential(size=24)
    event = rng.integers(0, 2, size=24)
    torch.set_num_threads(1)
    automatic = scis_test(x, z, t, event, random_state=17)
    explicit = scis_test(x, z, t, event, random_state=17, **simulation_configs(24))
    np.testing.assert_array_equal(automatic.scores, explicit.scores)
    assert automatic.test_statistic == explicit.test_statistic


@pytest.mark.parametrize("estimator", ["nn", "rf"])
def test_partial_override_preserves_explicit_config_and_resolves_other_families(
    monkeypatch, estimator
):
    custom = DirectionConfig((7,), epochs=3, objective="standardized")
    expected = forest_configs(200) if estimator == "rf" else simulation_configs(200)

    class Captured(Exception):
        pass

    def capture(hazard, direction, generator, test):
        assert hazard == expected["hazard_config"]
        assert direction is custom
        assert generator == expected["generator_config"]
        assert test == expected["test_config"]
        raise Captured

    monkeypatch.setattr(core, "_validate_configs", capture)
    with pytest.raises(Captured):
        scis_test(
            np.ones(200), np.ones(200), np.ones(200), np.ones(200),
            estimator=estimator, direction_config=custom,
        )


@pytest.mark.parametrize("estimator", ["nn", "rf"])
def test_aggregation_resolves_defaults_at_each_input_size(monkeypatch, estimator):
    from scistest import aggregation

    seen = []
    custom = GeneratorConfig(epochs=2, hidden_dim=7)

    class Captured(Exception):
        pass

    def capture(hazard, direction, generator, test):
        seen.append((hazard, direction, generator, test))
        raise Captured

    def fake_fit(x, z, time, event, **kwargs):
        with pytest.raises(Captured):
            core.scis_test(x, z, time, event, **kwargs)
        assert kwargs["estimator"] == estimator
        expected = forest_configs(len(time)) if estimator == "rf" else simulation_configs(len(time))
        hazard, direction, generator, test = seen[-1]
        assert hazard == expected["hazard_config"]
        assert direction == expected["direction_config"]
        assert generator is custom
        assert test == expected["test_config"]
        return SimpleNamespace(
            test_statistic=np.random.default_rng(kwargs["random_state"]).normal()
        )

    monkeypatch.setattr(core, "_validate_configs", capture)
    monkeypatch.setattr(aggregation, "scis_test", fake_fit)
    aggregation.aggregated_scis_test(
        np.ones(120),
        np.ones(120),
        np.ones(120),
        np.ones(120),
        K=3,
        J=1,
        L=2,
        generator_config=custom,
        estimator=estimator,
    )
    assert len(seen) == 8
    assert {row[0].l1_penalty for row in seen} == {
        5 / n for n in (120, 40)
    }
