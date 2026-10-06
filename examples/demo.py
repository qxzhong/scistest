"""Minimal runnable example for scisTest and its optional split aggregation."""

from __future__ import annotations

import argparse

from scistest import (
    DirectionConfig,
    GeneratorConfig,
    HazardConfig,
    SciTestConfig,
    aggregated_scis_test,
    scis_test,
)
from scistest.simulation import simulate_survival_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="also run the computationally heavier Guo--Shah aggregation",
    )
    args = parser.parse_args()

    data = simulate_survival_data(n=400, effect=0.8, random_state=8)
    # Small settings keep this demo quick. Use the package defaults for analysis.
    hazard = HazardConfig(hidden_dims=(16,), epochs=15)
    direction = DirectionConfig(hidden_dims=(16,), epochs=15)
    generator = GeneratorConfig(epochs=15, hidden_dim=32, noise_dim=32)
    test = SciTestConfig(integration_points=20, generator_draws=20)

    result = scis_test(
        data.x,
        data.z,
        data.time,
        data.event,
        random_state=8,
        hazard_config=hazard,
        direction_config=direction,
        generator_config=generator,
        test_config=test,
    )
    print("single-split scisTest")
    print(f"  statistic = {result.test_statistic:.4f}")
    print(f"  p-value   = {result.p_value:.6g}")

    if args.aggregate:
        aggregated = aggregated_scis_test(
            data.x,
            data.z,
            data.time,
            data.event,
            3,  # K (default)
            5,  # J (default)
            6,  # L (default)
            random_state=8,
            hazard_config=hazard,
            direction_config=direction,
            generator_config=generator,
            test_config=test,
        )
        print("Guo--Shah aggregated scisTest")
        print(
            f"  K={aggregated.K}, subsample_size={aggregated.subsample_size}, "
            f"J={aggregated.J}, L={aggregated.L}, B={aggregated.B}"
        )
        print(f"  aggregated p-value = {aggregated.aggregated_p_value:.6g}")
        print(f"  empirical p-value = {aggregated.empirical_p_value:.6g}")


if __name__ == "__main__":
    main()
