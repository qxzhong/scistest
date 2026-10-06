"""Run a reproducible null/alternative Monte Carlo experiment for scisTest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scistest import (
    DirectionConfig,
    GeneratorConfig,
    HazardConfig,
    SciTestConfig,
    aggregated_scis_test,
    scis_test,
)
from scistest.simulation import simulate_survival_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--scenario", choices=("null", "alternative", "both"), default="both")
    parser.add_argument("--effect", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--integration-points", type=int, default=100)
    parser.add_argument("--generator-draws", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/simulation.csv"))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--J", type=int, default=5)
    parser.add_argument("--L", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    effects = []
    if args.scenario in ("null", "both"):
        effects.append(("null", 0.0))
    if args.scenario in ("alternative", "both"):
        effects.append(("alternative", args.effect))

    hazard = HazardConfig(epochs=args.epochs)
    direction = DirectionConfig(epochs=args.epochs)
    generator = GeneratorConfig(epochs=args.epochs)
    test = SciTestConfig(
        integration_points=args.integration_points,
        generator_draws=args.generator_draws,
    )
    seed_sequence = np.random.SeedSequence(args.seed)
    replicate_seeds = seed_sequence.generate_state(args.repetitions * len(effects) * 2)
    seed_position = 0
    rows: list[dict[str, object]] = []

    for scenario, effect in effects:
        for replication in range(args.repetitions):
            data_seed = int(replicate_seeds[seed_position])
            fit_seed = int(replicate_seeds[seed_position + 1])
            seed_position += 2
            data = simulate_survival_data(
                args.sample_size,
                effect=effect,
                random_state=data_seed,
            )
            if args.aggregate:
                result = aggregated_scis_test(
                    data.x,
                    data.z,
                    data.time,
                    data.event,
                    args.K,
                    args.J,
                    args.L,
                    random_state=fit_seed,
                    hazard_config=hazard,
                    direction_config=direction,
                    generator_config=generator,
                    test_config=test,
                )
                row = {
                    "scenario": scenario,
                    "replication": replication,
                    "n": args.sample_size,
                    "effect": effect,
                    "event_fraction": float(data.event.mean()),
                    "aggregated_p_value": result.aggregated_p_value,
                    "log_p_value": result.log_p_value,
                    "empirical_p_value": result.empirical_p_value,
                    "test_statistic": result.aggregate_statistic,
                    "K": result.K,
                    "subsample_size": result.subsample_size,
                    "J": result.J,
                    "L": result.L,
                    "B": result.B,
                    "data_seed": data_seed,
                    "fit_seed": fit_seed,
                }
            else:
                result = scis_test(
                    data.x,
                    data.z,
                    data.time,
                    data.event,
                    random_state=fit_seed,
                    hazard_config=hazard,
                    direction_config=direction,
                    generator_config=generator,
                    test_config=test,
                )
                row = {
                    "scenario": scenario,
                    "replication": replication,
                    "n": args.sample_size,
                    "effect": effect,
                    "event_fraction": float(data.event.mean()),
                    "p_value": result.p_value,
                    "log_p_value": result.log_p_value,
                    "test_statistic": result.test_statistic,
                    "data_seed": data_seed,
                    "fit_seed": fit_seed,
                }
            rows.append(row)
            displayed_p_value = (
                row["aggregated_p_value"] if args.aggregate else row["p_value"]
            )
            print(
                f"{scenario} {replication + 1}/{args.repetitions}: "
                f"p={displayed_p_value}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for scenario, _ in effects:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        rejection_rate = np.mean(
            [float(row["log_p_value"]) < np.log(0.05) for row in scenario_rows]
        )
        print(f"{scenario}: rejection rate at 0.05 = {rejection_rate:.3f}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
