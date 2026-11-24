"""Run main6 simulations; invoke from the repository root with python -m simulations.run."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

from scistest import scis_test
from scistest.config import config_as_dict
from scistest.manuscript import simulation_configs

from . import protocol as p


def run_task(task: dict) -> dict:
    torch.set_num_threads(task.get("threads", 1))
    cell = p.cell_from_dict(task["cell"])
    rep = task["repetition"]
    master = 2026111703 if cell.case == 1 else 2026111602
    seed = p.deterministic_seed(cell, rep, 1, master)
    grid = 100 if task["grid"] == "paper" else math.isqrt(100 * cell.n)
    configs = simulation_configs(cell.n, integration_points=grid)
    if task["smoke"]:
        for key in ("hazard_config", "direction_config", "generator_config"):
            if hasattr(configs[key], "epochs"):
                configs[key] = replace(configs[key], epochs=1)
        configs["test_config"] = replace(
            configs["test_config"], integration_points=8, generator_draws=8
        )
    metadata = {
        **cell.as_dict(),
        "repetition": rep,
        "method": task["method"],
        "master_seed": master,
        "method_seed": seed,
        "smoke": task["smoke"],
        "estimator": "neural",
        "configs": {key: config_as_dict(value) for key, value in configs.items()},
    }
    path = Path(task["output"]) / "raw" / task["method"] / f"{cell.slug}_r{rep:03d}.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("settings") != metadata:
            raise ValueError(f"Settings changed; choose a fresh output directory: {path}")
        return previous
    frame = p.generate_frame(cell, rep, master)
    if task["method"] == "scisTest":
        x, z, t, delta = p.frame_tensors(cell, frame)
        result = scis_test(
            x.numpy(), z.numpy(), t.numpy(), delta.numpy(), random_state=seed, **configs
        ).as_dict()
    else:
        from .comparators.runner import compare

        result = compare(task["method"], cell, frame, seed)
    record = {
        "settings": metadata,
        "dataset_sha256": p.array_sha256(frame),
        "censoring_rate": 1 - float(frame.Delta.mean()),
        **result,
    }
    p.atomic_json(path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", choices=["all", "case1", "case2", "case3a", "case3b"], default="all"
    )
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=list(p.N_VALUES))
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--method", choices=["scisTest", "deepTest", "vim", "GCM", "PCM"], default="scisTest"
    )
    parser.add_argument("--integration-grid", choices=["paper", "cluster"], default="paper")
    parser.add_argument("--output", type=Path, default=Path("results/simulations"))
    parser.add_argument("--smoke", action="store_true", help="one cell/rep with reduced training")
    parser.add_argument("--dry-run", action="store_true", help="print task count without fitting")
    args = parser.parse_args()
    if args.repetitions < 1 or args.workers < 1:
        parser.error("repetitions and workers must be positive")
    if not set(args.sample_sizes) <= set(p.N_VALUES):
        parser.error(f"sample sizes must be from {p.N_VALUES}")
    cells = [
        c
        for c in p.all_cells()
        if c.n in args.sample_sizes and (args.group == "all" or c.group == args.group)
    ]
    if args.method in {"deepTest", "vim"}:
        cells = [c for c in cells if c.case == 1]
    elif args.method in {"GCM", "PCM"}:
        cells = [c for c in cells if c.case == 3]
    if not cells:
        parser.error("selected method and group have no applicable settings")
    if args.smoke:
        cells = cells[:1]
    repetitions = 1 if args.smoke else args.repetitions
    tasks = [
        {
            "cell": c.as_dict(),
            "repetition": r,
            "method": args.method,
            "grid": args.integration_grid,
            "smoke": args.smoke,
            "output": str(args.output.resolve()),
        }
        for c in cells
        for r in range(1, repetitions + 1)
    ]
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "grid": args.integration_grid,
                "method": args.method,
                "smoke": args.smoke,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if args.workers == 1:
        for task in tasks:
            run_task(task)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for _ in pool.map(run_task, tasks, chunksize=1):
                pass
    print("Complete. Use python -m simulations.summarize to aggregate checkpoints.")


if __name__ == "__main__":
    main()
