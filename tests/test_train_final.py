# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the final-training script, on tiny data with tiny models."""

import concurrent.futures
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from eigenvibes import training
from eigenvibes.synthetic_data import FEATURES, RULES
from eigenvibes.training import Hyperparameters

SCRIPTS: Path = Path(__file__).resolve().parent.parent / "scripts"


def test_copies_train_every_model_once_and_save_it_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two copies train all models between them, and the saved files restore them.

    With 1 dataset and 1 seed there are 4 models, 2 per copy. Each saved
    model, rebuilt from its ``.pt``, must reproduce its recorded test RMSE
    exactly, and a ``true`` model must hold ``covariance.csv``. Running a copy
    again must train nothing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory from pytest.
    monkeypatch : pytest.MonkeyPatch
        Shrinks the data, the model list, and the training.
    """
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "train_final", SCRIPTS / "train_final.py"
    )
    assert spec is not None and spec.loader is not None
    script: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    rng: np.random.Generator = np.random.default_rng(42)
    root: Path = tmp_path / "data"
    covariance: np.ndarray = np.eye(12) + 0.1
    root.mkdir()
    pd.DataFrame(covariance, columns=list(FEATURES)).to_csv(
        root / "covariance.csv", index=False
    )
    formula: str
    split: str
    rows: int
    for formula in RULES:
        (root / formula / "evaluation_00").mkdir(parents=True)
        for split, rows in (("train", 12), ("validation", 6), ("test", 6)):
            frame: pd.DataFrame = pd.DataFrame(
                rng.standard_normal((rows, 12)), columns=list(FEATURES)
            )
            frame["sus"] = frame[list(FEATURES)].sum(axis=1)
            frame.to_csv(root / formula / "evaluation_00" / f"{split}.csv", index=False)
    monkeypatch.setattr(script, "DATASETS", ("evaluation_00",))
    monkeypatch.setattr(script, "SEEDS", (0,))
    monkeypatch.setattr(script, "COPIES", 2)
    monkeypatch.setattr(training, "MAX_EPOCHS", 3)
    monkeypatch.setattr(training, "PATIENCE", 2)
    hyperparameters: Hyperparameters = Hyperparameters(0.01, 4, (2,), (2,))
    settings: dict[str, tuple[int, Hyperparameters]] = {
        formula: (7, hyperparameters) for formula in RULES
    }
    results: Path = tmp_path / "results"
    counts: dict[str, float] = {"trainings": 0, "training_seconds": 0.0}
    executor: concurrent.futures.ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        script.run(0, settings, executor, counts, root, results)
        script.run(1, settings, executor, counts, root, results)
        assert counts["trainings"] == 4
        script.run(0, settings, executor, counts, root, results)
        assert counts["trainings"] == 4

    paths: list[Path] = sorted(results.glob("*/*/*.pt"))
    assert len(paths) == 4
    path: Path
    for path in paths:
        saved: dict[str, Any] = torch.load(path, weights_only=True)
        model: torch.nn.Module = training.build(
            saved["covariance_matrix"],
            Hyperparameters(**saved["hyperparameters"]),
            saved["seed"],
        )
        model.load_state_dict(saved["state_dict"])
        x_test, y_test = training.load(saved["formula"], saved["dataset"], "test", root)
        rmse: float = (
            (training.predict(model, x_test) - y_test).square().mean().sqrt().item()
        )
        assert rmse == saved["test_rmse"]
        if saved["covariance"] == "true":
            assert np.array_equal(saved["covariance_matrix"].numpy(), covariance)

    summary: dict[str, Any] = json.loads(json.dumps(script.summarize(results)))
    assert summary["linear"]["tuning_trial"] == 7
    assert summary["interaction"]["estimated"]["models"] == 1
    assert len(pd.read_csv(results / "metrics.csv")) == 4
