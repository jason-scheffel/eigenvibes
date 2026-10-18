# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the tuning script, on tiny data with tiny models."""

import concurrent.futures
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import optuna
import pandas as pd
import pytest

from eigenvibes import training
from eigenvibes.synthetic_data import FEATURES


def _load_script() -> ModuleType:
    """Import ``scripts/tune.py``, which is not part of the package."""
    path: Path = Path(__file__).resolve().parent.parent / "scripts" / "tune.py"
    spec = importlib.util.spec_from_file_location("tune", path)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tuning_fills_the_study_resumes_and_summarizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy runs trials to the limit, fails only its own orphans, and sums up.

    The first run stops at 2 trials. A trial left running by this copy, as a
    killed run leaves it, and one left by another copy are then added. The
    second run, with a limit of 4, must fail only this copy's orphan, count
    the other copy's as taken, and finish 1 more trial. The summary must
    count the finished trials and the recorded hours.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory from pytest.
    monkeypatch : pytest.MonkeyPatch
        Shrinks the data, the search space, and the training.
    """
    tune: ModuleType = _load_script()
    rng: np.random.Generator = np.random.default_rng(42)
    dataset: str
    split: str
    rows: int
    for dataset in ("tuning_30", "tuning_31"):
        for split, rows in (("train", 12), ("validation", 6)):
            frame: pd.DataFrame = pd.DataFrame(
                rng.standard_normal((rows, 12)), columns=list(FEATURES)
            )
            frame["sus"] = frame[list(FEATURES)].sum(axis=1)
            (tmp_path / "data" / "linear" / dataset).mkdir(parents=True, exist_ok=True)
            frame.to_csv(tmp_path / "data" / "linear" / dataset / f"{split}.csv")
    monkeypatch.setattr(tune, "DATASETS", ("tuning_30", "tuning_31"))
    monkeypatch.setattr(tune, "SLOTS", 2)
    monkeypatch.setattr(training, "MAX_EPOCHS", 3)
    monkeypatch.setattr(training, "PATIENCE", 2)
    monkeypatch.setattr(training, "MAX_LAYERS", 1)
    monkeypatch.setattr(training, "MAX_WIDTH", 2)
    monkeypatch.setattr(training, "MAX_TAPS", 2)
    monkeypatch.setattr(training, "BATCH_SIZES", (4,))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results: Path = tmp_path / "results"
    results.mkdir()
    study: optuna.Study = optuna.create_study(
        study_name="linear",
        storage=tune._storage(results / "linear.log"),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    counts: dict[str, float] = {"trials": 0, "trainings": 0, "training_seconds": 0.0}
    executor: concurrent.futures.ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        monkeypatch.setattr(tune, "TRIALS", 2)
        tune.tune(study, "linear", 0, executor, counts, tmp_path / "data")
        study.ask().set_user_attr("copy", 0)
        study.ask().set_user_attr("copy", 1)
        monkeypatch.setattr(tune, "TRIALS", 4)
        tune.tune(study, "linear", 0, executor, counts, tmp_path / "data")

    states: list[optuna.trial.TrialState] = [t.state for t in study.trials]
    assert states == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert counts["trials"] == 3 and counts["trainings"] == 6
    assert sorted(study.trials[0].user_attrs["datasets"]) == ["tuning_30", "tuning_31"]

    tune._write_json(
        {"hours": 0.5, "training_seconds": 3600.0},
        results / "nodes" / "linear_0_1.json",
    )
    summary: dict[str, dict[str, object]] = json.loads(
        json.dumps(tune.summarize(results))
    )
    assert list(summary) == ["linear"]
    assert summary["linear"]["finished_trials"] == 3
    assert summary["linear"]["gpu_hours"] == 0.5
    assert summary["linear"]["training_hours"] == 1.0
    assert summary["linear"]["best_trial"]["number"] in (0, 1, 4)  # type: ignore[index]


def test_a_diverged_training_fails_its_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diverged dataset fails its trial, which still counts toward the limit.

    Training is replaced by a stand-in that diverges on ``tuning_31``. Both of a
    trial's datasets run at once, so the other one may finish after the trial
    has failed; its time must still be counted.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory from pytest.
    monkeypatch : pytest.MonkeyPatch
        Replaces training and shrinks the study.
    """
    tune: ModuleType = _load_script()

    def fake_train(
        formula: str, dataset: str, hyperparameters: object, root: Path
    ) -> tuple[float, int, int, float]:
        return (float("inf"), 0, 0, 1.0) if dataset == "tuning_31" else (0.5, 1, 1, 1.0)

    monkeypatch.setattr(tune, "train_one", fake_train)
    monkeypatch.setattr(tune, "DATASETS", ("tuning_30", "tuning_31"))
    monkeypatch.setattr(tune, "SLOTS", 2)
    monkeypatch.setattr(tune, "TRIALS", 2)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study: optuna.Study = optuna.create_study(
        study_name="linear",
        storage=tune._storage(tmp_path / "linear.log"),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    counts: dict[str, float] = {"trials": 0, "trainings": 0, "training_seconds": 0.0}
    executor: concurrent.futures.ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        tune.tune(study, "linear", 0, executor, counts, tmp_path)
    assert [t.state for t in study.trials] == [optuna.trial.TrialState.FAIL] * 2
    assert all(t.user_attrs["diverged"] for t in study.trials)
    assert counts == {"trials": 2, "trainings": 4, "training_seconds": 4.0}
