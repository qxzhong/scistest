"""Plot simulation panels from results computed by this repository."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/simulations"))
    args = parser.parse_args()
    data = pd.read_csv(args.output / "rejection_rates.csv")
    if data.smoke.any():
        parser.error("smoke checkpoints cannot be used for manuscript figures")
    styles = {"scisTest": "-", "vim": "--", "deepTest": "-.", "GCM": "--", "PCM": "-."}
    for group, filename in [
        ("case1", "rejection_rates_section421_case1.pdf"),
        ("case3a", "rejection_rates_section422_case3a.pdf"),
        ("case3b", "rejection_rates_section422_case3b.pdf"),
        ("case2", "rejection_rates_nondegeneracy.pdf"),
    ]:
        subset = data[data.group == group]
        if subset.empty:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
        column = "hypothesis" if group == "case1" else "a"
        levels = range(1, 7) if group == "case1" else [0, 0.5, 1, 1.5, 2, 2.5]
        for ax, level in zip(axes.flat, levels, strict=True):
            panel = subset[subset[column] == level]
            for method, rows in panel.groupby("method", sort=False):
                rows = rows.sort_values("n")
                ax.plot(rows.n, rows.rejection_rate, styles[method], marker="o", label=method)
            ax.axhline(0.05, color="grey", linestyle=":")
            ax.set_title(f"H{level}" if group == "case1" else f"a={level:g}")
            ax.set_ylim(0, 1)
            ax.set_xlabel("Sample size")
            ax.set_ylabel("Rejection rate")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(args.output / filename)
        plt.close(fig)


if __name__ == "__main__":
    main()
