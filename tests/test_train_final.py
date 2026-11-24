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
from eigenvibes.training import MODELS, Hyperparameters

SCRIPTS: Path = Path(__file__).resolve().parent.parent / "scripts"


def test_copies_train_explain_and_save_every_model_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two copies train all models between them, and the saved files agree.

    With 1 dataset there are 4 models, 2 per copy. Each saved model, rebuilt
    from its ``.pt``, must reproduce its recorded test RMSE and its cohort
    predictions exactly, and each row of its attributions must sum to its
    prediction minus its prediction for 0. Running a copy again must train
    nothing.

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
    formula: str
    split: str
    rows: int
    for formula in RULES:
        (root / formula / "evaluation_00").mkdir(parents=True)
        for split, rows in (
            ("train", 12),
            ("validation", 6),
            ("test", 6),
            ("cohort", 8),
        ):
            frame: pd.DataFrame = pd.DataFrame(
                rng.standard_normal((rows, 12)), columns=list(FEATURES)
            )
            frame["sus"] = frame[list(FEATURES)].sum(axis=1)
            if split == "cohort":
                frame.insert(0, "player", np.repeat(np.arange(4), 2))
                frame.insert(1, "round", np.tile([1, 2], 4))
                frame.insert(2, "impostor", np.repeat([0, 1], 4))
            frame.to_csv(root / formula / "evaluation_00" / f"{split}.csv", index=False)
    monkeypatch.setattr(script, "DATASETS", ("evaluation_00",))
    monkeypatch.setattr(script, "COPIES", 2)
    monkeypatch.setattr(training, "MAX_EPOCHS", 3)
    monkeypatch.setattr(training, "PATIENCE", 2)
    settings: dict[tuple[str, str], tuple[int, Hyperparameters]] = {
        (model, formula): (7, Hyperparameters(0.01, 4, (2,), (2,) * (model == "vnn")))
        for model in MODELS
        for formula in RULES
    }
    results: Path = tmp_path / "results"
    executor: concurrent.futures.ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        assert script.run(0, settings, executor, root, results) == 2
        assert script.run(1, settings, executor, root, results) == 2
        assert script.run(0, settings, executor, root, results) == 0

    paths: list[Path] = sorted(results.glob("*/*/*.pt"))
    assert len(paths) == 4
    path: Path
    for path in paths:
        saved: dict[str, Any] = torch.load(path, weights_only=True)
        hyperparameters: Hyperparameters = Hyperparameters(**saved["hyperparameters"])
        model: torch.nn.Module = (
            training.build(saved["covariance"], hyperparameters, script.SEED)
            if saved["model"] == "vnn"
            else training.build_mlp(hyperparameters, script.SEED)
        )
        model.load_state_dict(saved["state_dict"])
        x_test, y_test = training.load(saved["formula"], saved["dataset"], "test", root)
        rmse: float = (
            (training.predict(model, x_test) - y_test).square().mean().sqrt().item()
        )
        assert rmse == saved["test_rmse"]
        x_cohort, _ = training.load(saved["formula"], saved["dataset"], "cohort", root)
        ig: pd.DataFrame = pd.read_csv(
            path.with_name(f"{saved['dataset']}_ig.csv"), float_precision="round_trip"
        )
        prediction: np.ndarray = training.predict(model, x_cohort).numpy()
        zero: torch.Tensor = training.predict(
            model, torch.zeros(1, 12, dtype=torch.float64)
        )
        assert np.array_equal(ig["prediction"].to_numpy(), prediction)
        gap: np.ndarray = np.abs(
            ig[list(FEATURES)].to_numpy().sum(1) - (prediction - zero.item())
        )
        assert gap.max() < 1e-3
        assert gap.max() == pytest.approx(saved["completeness_gap"], abs=1e-12)

    summary: dict[str, Any] = json.loads(json.dumps(script.summarize(results)))
    assert summary["vnn_linear"]["tuning_trial"] == 7
    assert summary["mlp_interaction"]["models"] == 1
    assert len(pd.read_csv(results / "metrics.csv")) == 4
