#!/usr/bin/env python3
"""Main6 data-generating mechanisms and frozen simulation seed mapping."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TUNING_MASTER_SEED = 2026111601
FINAL_MASTER_SEED = 2026111602
N_VALUES = (200, 400, 600, 800, 1000)
A_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
REPETITIONS = 200
CASE1_BETA0 = 0.7818669539671581
CASE2_INTERCEPTS = {
    0.0: 2.4182045707977444,
    0.5: 2.411926391754132,
    1.0: 2.388945196627814,
    1.5: 2.38521382833757,
    2.0: 2.4050402404784346,
    2.5: 2.4702533832794376,
}


@dataclass(frozen=True)
class Cell:
    group: str
    case: int
    mechanism: str
    n: int
    hypothesis: int = 0
    a: float = 0.0

    @property
    def slug(self) -> str:
        if self.case == 1:
            return f"case1_h{self.hypothesis}_n{self.n}"
        return f"case{self.case}_{self.mechanism}_a{self.a:.1f}_n{self.n}".replace(".", "p")

    @property
    def x_columns(self) -> list[str]:
        if self.case == 1:
            return {
                1: ["W1"],
                2: ["W4"],
                3: ["W5"],
                4: ["W1", "W8", "W9", "W10"],
                5: ["W6"],
                6: ["W6", "W7", "W8", "W9", "W10"],
            }[self.hypothesis]
        if self.case == 2:
            return ["X"]
        return ["V1" if self.mechanism == "a" else "V2"]

    @property
    def null_is_true(self) -> bool:
        return self.case != 1 or self.hypothesis >= 5

    def as_dict(self) -> dict:
        return {
            "group": self.group,
            "case": self.case,
            "mechanism": self.mechanism,
            "n": self.n,
            "hypothesis": self.hypothesis,
            "a": self.a,
            "setting": self.slug,
            "x_columns": self.x_columns,
            "null_is_true": self.null_is_true,
        }


def all_cells() -> list[Cell]:
    cells = [Cell("case1", 1, "none", n, hypothesis=h) for n in N_VALUES for h in range(1, 7)]
    cells += [Cell("case2", 2, "none", n, a=a) for n in N_VALUES for a in A_VALUES]
    cells += [
        Cell(f"case3{mechanism}", 3, mechanism, n, a=a)
        for mechanism in ("a", "b")
        for n in N_VALUES
        for a in A_VALUES
    ]
    assert len(cells) == 120 and len({cell.slug for cell in cells}) == 120
    return cells


def cell_from_dict(value: dict) -> Cell:
    return Cell(
        str(value["group"]),
        int(value["case"]),
        str(value["mechanism"]),
        int(value["n"]),
        int(value.get("hypothesis", 0)),
        float(value.get("a", 0.0)),
    )


def deterministic_seed(cell: Cell, repetition: int, stream: int, master_seed: int) -> int:
    mechanism_id = {"none": 0, "a": 1, "b": 2}[cell.mechanism]
    hypothesis_or_a = cell.hypothesis if cell.case == 1 else int(round(10 * cell.a))
    return int(
        np.random.SeedSequence(
            [
                master_seed,
                cell.case,
                mechanism_id,
                hypothesis_or_a,
                cell.n,
                repetition,
                stream,
            ]
        ).generate_state(1)[0]
    )


def generate_frame(cell: Cell, repetition: int, master_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(deterministic_seed(cell, repetition, 0, master_seed))
    n = cell.n
    if cell.case == 1:
        w = (rng.standard_normal((n, 1)) + rng.standard_normal((n, 10))) / np.sqrt(2)
        w = np.sign(w) * np.abs(w) ** 1.5 + 0.5
        event = (
            5 * np.sin(w[:, 0]) + w[:, 1] ** 2 + 4 * w[:, 2] * w[:, 3] + 1.5 * w[:, 4] - 8.03
        ) / 6
        censor = (w[:, 0] + w[:, 1] ** 2 + w[:, 2] * w[:, 3] + w[:, 5] - 3.9) / 6
        log_u = event + rng.standard_normal(n)
        log_c = CASE1_BETA0 + censor + rng.standard_normal(n)
        frame = pd.DataFrame(w, columns=[f"W{j}" for j in range(1, 11)])
    elif cell.case == 2:
        z = rng.standard_normal(n)
        log_x = 5 * np.sin(z) + cell.a * rng.standard_normal(n)
        log_u = 5 * np.sin(z) + cell.a * rng.standard_normal(n)
        log_c = CASE2_INTERCEPTS[cell.a] + rng.standard_normal(n)
        frame = pd.DataFrame({"X": np.exp(log_x), "Z": z})
    else:
        v = rng.standard_normal((n, 6))
        epsilon_u = rng.standard_normal(n)
        epsilon_c = rng.standard_normal(n)
        if cell.mechanism == "a":
            log_u = np.sqrt(cell.a * cell.a + 1) * epsilon_u
            log_c = v[:, 0] + cell.a * epsilon_c
        else:
            log_u = cell.a * v[:, 0] + epsilon_u
            log_c = v[:, 1] + epsilon_c
        frame = pd.DataFrame(v, columns=[f"V{j}" for j in range(1, 7)])
    frame["T"] = np.exp(np.minimum(log_u, log_c))
    frame["Delta"] = (log_u <= log_c).astype(int)
    if not np.isfinite(frame.to_numpy()).all() or not frame["T"].gt(0).all():
        raise ValueError(f"Invalid generated data for {cell.slug}")
    return frame


def frame_tensors(cell: Cell, frame: pd.DataFrame) -> tuple[torch.Tensor, ...]:
    z_columns = [c for c in frame.columns if c not in cell.x_columns + ["T", "Delta"]]
    return tuple(
        torch.tensor(value, dtype=torch.float32)
        for value in (
            frame[cell.x_columns].to_numpy(),
            frame[z_columns].to_numpy(),
            frame["T"].to_numpy(),
            frame["Delta"].to_numpy(),
        )
    )


def array_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame.to_numpy(dtype="<f8")).tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
