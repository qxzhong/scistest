"""Plot newly computed Framingham p-values on a log scale without underflow."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .run_framingham import COVARIATES, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/framingham"))
    args = parser.parse_args()
    frame = pd.read_csv(args.output / "aggregated_pvalues.csv")
    series = [frame.loc[frame.covariate == name, "log10_p_kde"].to_numpy() for name in COVARIATES]
    if any(len(values) == 0 for values in series):
        parser.error("all six covariates are required")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(series)
    ax.set_xticks(range(1, 7), LABELS, rotation=20)
    ax.axhline(math.log10(0.05), linestyle=":", color="grey")
    ax.set_ylabel("log10 aggregated p-value")
    fig.tight_layout()
    fig.savefig(args.output / "framingham_repeated_aggregated_pvalues.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
