"""Summarize newly computed checkpoints; no stored results are needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/simulations"))
    args = parser.parse_args()
    records = []
    for path in sorted((args.output / "raw").glob("*/*.json")):
        record = json.loads(path.read_text())
        records.append(
            {**record["settings"], "p_value": record["p_value"], "reject": record["p_value"] < 0.05}
        )
    if not records:
        parser.error("no checkpoints found")
    frame = pd.DataFrame(records)
    summary = (
        frame.groupby(["method", "group", "setting", "n", "a", "hypothesis", "smoke"])["reject"]
        .agg(n_success="count", n_reject="sum", rejection_rate="mean")
        .reset_index()
    )
    summary["mcse"] = (
        summary.rejection_rate * (1 - summary.rejection_rate) / summary.n_success
    ) ** 0.5
    summary.to_csv(args.output / "rejection_rates.csv", index=False)


if __name__ == "__main__":
    main()
