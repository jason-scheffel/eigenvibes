# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the evaluation script."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from eigenvibes import synthetic_data
from eigenvibes.synthetic_data import FEATURES, RULES

SCRIPTS: Path = Path(__file__).resolve().parent.parent / "scripts"


def _script() -> ModuleType:
    """Import ``scripts/evaluate.py``."""
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPTS / "evaluate.py")
    assert spec is not None and spec.loader is not None
    script: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_ridge_recovers_the_coefficients() -> None:
    """On almost noiseless data, ridge returns the true intercept and slopes."""
    rng: np.random.Generator = np.random.default_rng(42)
    beta: np.ndarray = rng.standard_normal(12)
    x: np.ndarray = rng.standard_normal((500, 12)) + 3
    x_valid: np.ndarray = rng.standard_normal((100, 12)) + 3
    noise: np.ndarray = 1e-8 * rng.standard_normal(600)
    intercept: float
    fitted: np.ndarray
    intercept, fitted = _script().ridge(
        x, 1.5 + x @ beta + noise[:500], x_valid, 1.5 + x_valid @ beta + noise[500:]
    )
    assert intercept == pytest.approx(1.5, abs=1e-6)
    np.testing.assert_allclose(fitted, beta, atol=1e-6)


def test_main_writes_every_table_and_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run on two datasets with a fake VNN writes every output.

    The fake VNN's IG is the exact IG, so its errors must be 0, like the
    oracle's.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory from pytest.
    monkeypatch : pytest.MonkeyPatch
        Limits the run to two datasets.
    """
    script: ModuleType = _script()
    datasets: tuple[str, ...] = ("evaluation_00", "evaluation_01")
    monkeypatch.setattr(script, "DATASETS", datasets)
    root: Path = tmp_path / "data"
    results: Path = tmp_path / "results"
    seed: int
    for seed in range(len(datasets)):
        synthetic_data.write_dataset(root, seed)
    formula: str
    interaction: float
    dataset: str
    for formula, interaction in RULES.items():
        folder: Path = results / "final" / formula / "vnn"
        folder.mkdir(parents=True)
        for dataset in datasets:
            cohort: pd.DataFrame = pd.read_csv(
                root / formula / dataset / "cohort.csv", float_precision="round_trip"
            )
            ig: pd.DataFrame = pd.DataFrame(
                synthetic_data.integrated_gradients(
                    cohort[list(FEATURES)].to_numpy(), interaction
                ),
                columns=FEATURES,
            )
            ig.assign(prediction=cohort["sus"]).to_csv(
                folder / f"{dataset}_ig.csv", index=False
            )
            (folder / f"{dataset}.json").write_text(
                json.dumps({"test_rmse": 0.0, "test_r2": 1.0, "diverged": False}),
                encoding="utf-8",
            )

    script.main(root, results)

    summary: pd.DataFrame = pd.read_csv(results / "evaluation" / "summary.csv")
    assert len(summary) == 6
    exact: pd.DataFrame = summary[summary["model"].isin(["oracle", "vnn"])]
    np.testing.assert_allclose(
        exact[["behavior_mae", "total_bias", "ig_rmse"]], 0, atol=1e-9
    )
    contributions: pd.DataFrame = pd.read_csv(
        results / "evaluation" / "contributions.csv"
    )
    assert len(contributions) == 2 * 12 * (1 + 3 * len(datasets))
    assert (results / "evaluation" / "summary.tex").exists()
    name: str
    for name in ("contributions.pdf", "contributions.png", "trajectories.png"):
        assert (results / "figures" / name).exists()
