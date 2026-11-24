"""Repeated Framingham aggregate p-values for the Section 5 boxplot.

For each covariate this program first estimates one rank-transformed
subsampling calibration distribution with J=10, m=1552 and L=3.  It then
forms 1,000 full-data aggregates, each averaging three independent two-split
tests, and maps each aggregate
through that fixed calibration distribution.  Holding the calibration fixed
means that the resulting boxplots isolate the residual random-split variation
after aggregating L=3 splits.

Every neural-network fit is checkpointed.  Aggregate tails are evaluated in
the log domain and stored both as log probabilities and 17-significant-digit
decimal strings, including probabilities below the binary64 range.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp, ndtri

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/framingham"
PYTHON_SOURCE = ROOT / "src/scistest/core.py"
BASE_SEED = 498121629
J = 10
M = 1552
L = 3
REPETITIONS = 1000
COVARIATES = ("num_sbp", "num_dbp", "num_scl", "num_age", "num_bmi", "fac_sex")
LABELS = (
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Total cholesterol",
    "Age",
    "Body mass index",
    "Sex",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decimal_from_log(log_value: float, significant_digits: int = 17) -> str:
    if not math.isfinite(log_value):
        raise ValueError("log probability must be finite")
    with localcontext() as context:
        context.prec = 80
        context.Emin = -999999999
        context.Emax = 999999999
        return format(Decimal(str(log_value)).exp(), f".{significant_digits - 1}E")


def seed_for(covariate: int, phase: int, row: int, split: int) -> int:
    return int(
        np.random.SeedSequence([BASE_SEED, covariate, phase, row, split]).generate_state(1)[0]
    )


def ordered_subsamples(n: int, m: int, permutations: int, seed: int) -> np.ndarray:
    if not 1 < m < n:
        raise ValueError("m must be strictly between 1 and n")
    blocks = n // m
    generator = np.random.default_rng(seed)
    return np.vstack(
        [generator.permutation(n)[: blocks * m].reshape(blocks, m) for _ in range(permutations)]
    )


def normal_rank_transform(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("expected a finite B by L matrix")
    flat = values.ravel()
    permutation = np.random.default_rng(seed).permutation(flat.size)
    order = permutation[np.argsort(flat[permutation], kind="stable")]
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(1, flat.size + 1, dtype=np.float64)
    return ndtri((ranks - 0.5) / flat.size).reshape(values.shape)


def bandwidth_nrd0(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = float(np.std(values, ddof=1))
    iqr = float(np.diff(np.quantile(values, [0.25, 0.75]))[0])
    scale = min(standard_deviation, iqr / 1.34)
    if scale == 0:
        scale = standard_deviation or abs(float(values[0])) or 1.0
    return 0.9 * scale * len(values) ** (-0.2)


def calibrate_observed(observed: np.ndarray, null: np.ndarray) -> dict:
    observed = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    if observed.shape != (L,) or null.ndim != 1 or not np.isfinite(observed).all():
        raise ValueError("invalid aggregate inputs")
    aggregate = float(observed.mean(dtype=np.float64))
    bandwidth = bandwidth_nrd0(null)
    log_probability = float(
        logsumexp(log_ndtr((null - aggregate) / bandwidth)) - math.log(len(null))
    )
    log_probability = min(0.0, log_probability)
    exceedances = int(np.count_nonzero(null > aggregate))
    return {
        "aggregate_statistic": aggregate,
        "log_p_kde": log_probability,
        "log10_p_kde": log_probability / math.log(10),
        "p_kde": decimal_from_log(log_probability),
        "p_empirical": exceedances / len(null),
        "exceedance_count": exceedances,
        "kde_bandwidth": bandwidth,
        "reject_5pct": bool(log_probability < math.log(0.05)),
    }


def encode_sex(values: pd.Series) -> pd.Series:
    """Normalize SurvSet releases to the cluster's F=0, M=1 coding."""
    observed = set(values.dropna().tolist())
    if observed <= {"F", "M"}:
        mapping = {"F": 0.0, "M": 1.0}
    elif observed <= {1, 2}:
        mapping = {1: 1.0, 2: 0.0}
    elif observed <= {0, 1}:
        mapping = {0: 0.0, 1: 1.0}
    else:
        raise ValueError(f"unexpected sex categories: {observed}")
    result = values.astype(object).map(mapping).astype(float)
    if result.isna().any():
        raise ValueError("missing or unrecognized sex category")
    return result


def load_and_save_data(output: Path) -> tuple[pd.DataFrame, Path]:
    from SurvSet.data import SurvLoader

    data_path = output / "framingham_complete_cases.csv"
    raw = SurvLoader().load_dataset(ds_name="Framingham")["df"]
    columns = ["event", "time", *COVARIATES]
    data = raw[columns].dropna().copy()
    data["fac_sex"] = encode_sex(data["fac_sex"])
    if data_path.exists():
        saved = pd.read_csv(data_path)
        np.testing.assert_allclose(data.to_numpy(float), saved.to_numpy(float))
    else:
        data.to_csv(data_path, index=False, float_format="%.17g")
    return data, data_path


def initialize_run(output: Path, repetitions: int) -> dict:
    data, data_path = load_and_save_data(output)
    n = len(data)
    if n != 4658 or int(data.event.sum()) != 1465:
        raise ValueError("Framingham data differs from main6 complete cases")
    if n // M != 3:
        raise ValueError(f"expected floor(n/m)=3, got n={n} and m={M}")
    tuples = ordered_subsamples(n, M, J, seed_for(0, 9, 0, 0))
    tuple_path = output / "subsample_indices.npy"
    if tuple_path.exists():
        np.testing.assert_array_equal(tuples, np.load(tuple_path))
    else:
        np.save(tuple_path, tuples)
    metadata = {
        "base_seed": BASE_SEED,
        "n": n,
        "raw_n": 4699,
        "complete_case_exclusions": 4699 - n,
        "events": int(data.event.sum()),
        "censoring_rate": 1.0 - float(data.event.mean()),
        "J": J,
        "m": M,
        "blocks_per_permutation": n // M,
        "B": len(tuples),
        "L": L,
        "repetitions": repetitions,
        "calibration_reuse": (
            "One fixed rank-transformed calibration distribution per covariate is reused "
            "for all full-data repetitions so the boxplots isolate random-split variation."
        ),
        "aggregation": "arithmetic mean of L one-sided standard-normal statistics",
        "tail": "Gaussian-kernel survival using R bw.nrd0 bandwidth, evaluated in log space",
        "test_implementation": str(PYTHON_SOURCE.relative_to(ROOT)),
        "test_settings": {
            "hazard": {
                "hidden_dims": [32],
                "learning_rate": 0.01,
                "epochs": 100,
                "l1_penalty": 0.001,
                "batch_size": 128,
            },
            "direction": {
                "hidden_dims": [32, 32],
                "learning_rate": 0.01,
                "epochs": 100,
                "l1_penalty": 0.001,
                "objective": "standardized",
            },
            "generator": {
                "learning_rate": 0.005,
                "epochs": 100,
                "batch_size": 128,
                "num_layers": 2,
                "hidden_dim": 100,
            },
            "test": {
                "train_ratio": 0.5,
                "integration_points": 100,
                "generator_draws": 100,
                "score_chunk_size": 8,
                "alpha": 0.05,
            },
        },
        "preprocessing": (
            "Within each fit: X is min-max scaled and Z is standardized using the training half; "
            "time is divided by the training-half maximum."
        ),
        "generator_fit_sample": "test half, as specified in main6.tex",
        "scoring_hazard": "evaluation",
        "integration_grid": "fixed 100 points",
        "covariates": list(COVARIATES),
        "labels": list(LABELS),
        "data_sha256": sha256(data_path),
        "implementation_sha256": {
            name: sha256(ROOT / "src/scistest" / name)
            for name in ("core.py", "_models.py", "config.py", "manuscript.py")
        },
    }
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("metadata changed; choose a new output directory")
    else:
        atomic_json(metadata_path, metadata)
    return metadata


_DATA = None
_TUPLES = None
_OUTPUT = None


def initialize_worker(output: str) -> None:
    global _DATA, _TUPLES, _OUTPUT
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _OUTPUT = Path(output)
    _DATA = pd.read_csv(_OUTPUT / "framingham_complete_cases.csv")
    _TUPLES = np.load(_OUTPUT / "subsample_indices.npy")


def checkpoint(output: Path, covariate: int, phase: int, row: int, split: int) -> Path:
    phase_name = "calibration" if phase == 0 else "full"
    return output / "raw" / COVARIATES[covariate] / phase_name / f"r{row:03d}_l{split:02d}.json"


def run_one(job: tuple[int, int, int, int]) -> dict:
    from scistest import scis_test
    from scistest.manuscript import framingham_configs

    covariate, phase, row, split = job
    seed = seed_for(covariate, phase, row, split)
    started = time.monotonic()
    record = {
        "covariate": COVARIATES[covariate],
        "phase": "calibration" if phase == 0 else "full",
        "row": row,
        "split": split,
        "seed": seed,
        "status": "error",
    }
    try:
        frame = _DATA.iloc[_TUPLES[row]] if phase == 0 else _DATA
        x_column = COVARIATES[covariate]
        z_columns = [column for column in COVARIATES if column != x_column]
        result = scis_test(
            frame[[x_column]].to_numpy(),
            frame[z_columns].to_numpy(),
            frame.time.to_numpy(),
            frame.event.to_numpy(),
            random_state=seed,
            **framingham_configs(),
        ).as_dict()
        result.update(p_value_precise=decimal_from_log(result["log_p_value"]), z_columns=z_columns)
        record.update(result, status="ok", n=len(frame))
    except Exception:
        record["error"] = traceback.format_exc()
    record["seconds"] = time.monotonic() - started
    atomic_json(checkpoint(_OUTPUT, *job), record)
    return record


def collect_covariate(output: Path, metadata: dict, covariate: int) -> dict | None:
    calibration = np.empty((metadata["B"], L), dtype=np.float64)
    observed = np.empty((metadata["repetitions"], L), dtype=np.float64)
    records = []
    for phase, rows, target in (
        (0, metadata["B"], calibration),
        (1, metadata["repetitions"], observed),
    ):
        for row in range(rows):
            for split in range(L):
                path = checkpoint(output, covariate, phase, row, split)
                if not path.exists():
                    return None
                record = json.loads(path.read_text())
                if record["status"] != "ok":
                    return None
                if record["seed"] != seed_for(covariate, phase, row, split):
                    raise ValueError(f"seed mismatch in {path}")
                target[row, split] = record["test_statistic"]
                records.append(record)
    transformed = normal_rank_transform(calibration, seed_for(covariate, 8, 0, 0))
    null = transformed.mean(axis=1, dtype=np.float64)
    rows = []
    for repetition in range(metadata["repetitions"]):
        result = calibrate_observed(observed[repetition], null)
        result.update(
            covariate=COVARIATES[covariate],
            label=LABELS[covariate],
            repetition=repetition + 1,
        )
        rows.append(result)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        output / f"{COVARIATES[covariate]}_aggregated_pvalues.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame(records).to_csv(
        output / f"{COVARIATES[covariate]}_fit_records.csv",
        index=False,
        float_format="%.17g",
    )
    np.savez_compressed(
        output / f"{COVARIATES[covariate]}_statistics.npz",
        calibration=calibration,
        ranked_calibration=transformed,
        null_aggregates=null,
        observed=observed,
    )
    log_values = frame.log_p_kde.to_numpy(float)
    summary = {
        "covariate": COVARIATES[covariate],
        "label": LABELS[covariate],
        "repetitions": len(frame),
        "median_log10_p": float(np.median(log_values) / math.log(10)),
        "q1_log10_p": float(np.quantile(log_values, 0.25) / math.log(10)),
        "q3_log10_p": float(np.quantile(log_values, 0.75) / math.log(10)),
        "median_p": decimal_from_log(float(np.median(log_values))),
        "q1_p": decimal_from_log(float(np.quantile(log_values, 0.25))),
        "q3_p": decimal_from_log(float(np.quantile(log_values, 0.75))),
        "minimum_p": decimal_from_log(float(log_values.min())),
        "maximum_p": decimal_from_log(float(log_values.max())),
        "rejections_5pct": int(frame.reject_5pct.sum()),
        "rejection_fraction_5pct": float(frame.reject_5pct.mean()),
    }
    atomic_json(output / f"{COVARIATES[covariate]}_summary.json", summary)
    return summary


def write_combined(output: Path, metadata: dict) -> None:
    frames = []
    summaries = []
    for covariate in COVARIATES:
        values = output / f"{covariate}_aggregated_pvalues.csv"
        summary = output / f"{covariate}_summary.json"
        if values.exists():
            frames.append(pd.read_csv(values, dtype={"p_kde": str}))
        if summary.exists():
            summaries.append(json.loads(summary.read_text()))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            output / "aggregated_pvalues.csv", index=False, float_format="%.17g"
        )
    if summaries:
        pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    if len(summaries) == len(COVARIATES):
        atomic_json(
            output / "RESULTS_COMPLETE.json",
            {
                "covariates": len(summaries),
                "aggregated_pvalues": len(summaries) * metadata["repetitions"],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="print task count without data loading"
    )
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "fits": 6 * (30 + args.repetitions) * L,
                    "J": J,
                    "m": M,
                    "L": L,
                    "repetitions": args.repetitions,
                }
            )
        )
        return
    if args.workers < 1 or args.repetitions < 1:
        raise ValueError("workers and repetitions must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = initialize_run(output, args.repetitions)
        jobs = []
        completed = 0
        failures = 0
        counts = {covariate: 0 for covariate in COVARIATES}
        target = (metadata["B"] + metadata["repetitions"]) * L
        for covariate in range(len(COVARIATES)):
            for phase, rows in ((0, metadata["B"]), (1, metadata["repetitions"])):
                for row in range(rows):
                    for split in range(L):
                        job = (covariate, phase, row, split)
                        path = checkpoint(output, *job)
                        if path.exists() and (
                            json.loads(path.read_text()).get("status") == "ok"
                            and json.loads(path.read_text()).get("scoring_hazard") == "evaluation"
                        ):
                            completed += 1
                            counts[COVARIATES[covariate]] += 1
                        else:
                            jobs.append(job)
            collect_covariate(output, metadata, covariate)
        write_combined(output, metadata)
        total = len(COVARIATES) * target
        if args.summarize_only:
            print(f"Summarized {completed}/{total} fits.")
            return
        print(
            f"START n={metadata['n']} J={J} m={M} B={metadata['B']} L={L} "
            f"R={metadata['repetitions']}; {completed}/{total} fits complete; "
            f"workers={args.workers}",
            flush=True,
        )
        started = time.monotonic()
        fresh = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(str(output),),
        ) as pool:
            futures = {pool.submit(run_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures.pop(future)
                result = future.result()
                name = COVARIATES[job[0]]
                if result["status"] == "ok":
                    completed += 1
                    fresh += 1
                    counts[name] += 1
                else:
                    failures += 1
                    print(f"ERROR {name} phase={job[1]} row={job[2]} split={job[3]}", flush=True)
                elapsed = time.monotonic() - started
                progress = {
                    "completed": completed,
                    "total": total,
                    "failures": failures,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": (total - completed) * elapsed / fresh
                    if fresh
                    else None,
                    "completed_by_covariate": counts,
                }
                atomic_json(output / "progress.json", progress)
                if fresh % 25 == 0 or failures:
                    print("PROGRESS", json.dumps(progress), flush=True)
                if counts[name] == target:
                    summary = collect_covariate(output, metadata, job[0])
                    write_combined(output, metadata)
                    print("COMPLETE", json.dumps(summary), flush=True)
        if failures or completed != total:
            raise RuntimeError(f"only {completed}/{total} fits completed; failures={failures}")
        write_combined(output, metadata)
        atomic_json(
            output / "RUN_COMPLETE.json",
            {"completed": completed, "total": total, "elapsed_seconds": time.monotonic() - started},
        )
        print(f"FINISHED {completed}/{total} fits", flush=True)


if __name__ == "__main__":
    main()
