"""Small adapters for the comparison implementations used on the cluster."""

from __future__ import annotations

import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


def compare(method, cell, frame, seed):
    if method == "deepTest":
        from .deep_test import NeuralCoxModel

        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        model = NeuralCoxModel(
            frame,
            "T",
            "Delta",
            cell.x_columns,
            num_nodes=[32],
            n_splits=5,
            epochs=64,
            val_size=0.2,
            batch_norm=True,
            dropout=0.1,
            output_bias=False,
            batch_size=516,
            random_state=seed,
        )
        p_value = float(model.compute_cv_pvalue(rho=0.1, rho_seq=[0.1]))
    else:
        # Temporary data only; no source data or precomputed results are shipped.
        with tempfile.TemporaryDirectory(prefix="scistest-r-") as directory:
            directory = Path(directory)
            source, target = directory / "data.csv", directory / "p.txt"
            frame.to_csv(source, index=False, float_format="%.17g")
            subprocess.run(
                [
                    "Rscript",
                    str(Path(__file__).with_name("run.R")),
                    method,
                    str(source),
                    ";".join(cell.x_columns),
                    str(seed),
                    str(target),
                ],
                check=True,
            )
            p_value = float(target.read_text())
    if not np.isfinite(p_value) or not 0 <= p_value <= 1:
        raise RuntimeError(f"Invalid {method} p-value: {p_value}")
    return {"p_value": p_value, "reject_null": p_value < 0.05}
