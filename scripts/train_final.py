# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Train the final models on the evaluation datasets and explain them.

For each formula, model kind (MLP or VNN) and evaluation dataset, the script
trains one model with seed 42, 120 models in all. Each uses the
hyperparameters of the best trial of the Optuna study ``<model>_<formula>``
in ``results/tuning/<model>_<formula>.log``, which must hold ``tune.TRIALS``
finished trials. A VNN's covariance is estimated from the dataset's training
split, as in tuning.

A model stops early on its dataset's validation split. It is then scored on
the test split, whose sus scores have no noise, by the root mean squared error
and R² of its predictions. Last, Integrated Gradients explains its prediction
for every row of the dataset's cohort.

The models are split among ``COPIES`` copies of this script, one per GPU: copy
``k`` takes ``models()[k::COPIES]``. A copy trains ``SLOTS`` models at a time,
each in its own process, and skips models already saved, so a stopped run
resumes by starting the same copies again. A model is saved in
``results/final/<formula>/<model>/`` as ``<dataset>.pt``, ``<dataset>_ig.csv``
and, written last, ``<dataset>.json``.

When it stops, a copy writes ``results/final/nodes/<copy>_<job>.json`` with
its run time, and rewrites ``results/final/metrics.csv``, one row per saved
model, and ``results/final/summary.json``.

Run it with ``uv run python scripts/train_final.py <copy>``; on the DGX,
submit ``scripts/train_final.sbatch``, which starts all 16 copies.
"""

import concurrent.futures
import itertools
import json
import multiprocessing
import os
import signal
import socket
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import torch
import tune
import typer
from optuna.storages.journal import JournalFileOpenLock

from eigenvibes import explain, training
from eigenvibes.synthetic_data import EVALUATION_SEEDS, FEATURES, RULES
from eigenvibes.training import MODELS, Hyperparameters

RESULTS: Path = Path(__file__).resolve().parent.parent / "results" / "final"
COPIES: int = 16
SLOTS: int = 4
SEED: int = 42
DATASETS: tuple[str, ...] = tuple(f"evaluation_{seed:02d}" for seed in EVALUATION_SEEDS)

type Model = tuple[str, str, str]
type Setting = tuple[int, Hyperparameters]
type Record = dict[str, Any]


def models() -> list[Model]:
    """List every model as ``(formula, model, dataset)``.

    Returns
    -------
    list[Model]
        All 120 models, in the same order every time.
    """
    return list(itertools.product(RULES, MODELS, DATASETS))


def tuned(model: str, formula: str) -> Setting:
    """Return the number and hyperparameters of a study's best trial.

    Parameters
    ----------
    model : str
        ``"mlp"`` or ``"vnn"``.
    formula : str
        ``"linear"`` or ``"interaction"``.

    Returns
    -------
    number : int
        The trial's number in the study ``<model>_<formula>``.
    hyperparameters : Hyperparameters
        Its hyperparameters.

    Raises
    ------
    SystemExit
        If the study has fewer than ``tune.TRIALS`` finished trials.
    """
    name: str = f"{model}_{formula}"
    study: optuna.Study = optuna.load_study(
        study_name=name, storage=tune._storage(tune.RESULTS / f"{name}.log")
    )
    finished: int = sum(
        t.state == optuna.trial.TrialState.COMPLETE
        or t.user_attrs.get("diverged", False)
        for t in study.get_trials(deepcopy=False)
    )
    if finished < tune.TRIALS:
        raise SystemExit(
            f"The {name} study has {finished} of {tune.TRIALS} finished trials"
        )
    best: optuna.trial.FrozenTrial = study.best_trial
    return best.number, training.suggest(best, model)


def train_one(
    job: Model, trial: int, hyperparameters: Hyperparameters, root: Path, results: Path
) -> Record:
    """Train, score, explain and save one final model.

    Parameters
    ----------
    job : Model
        ``(formula, model, dataset)``.
    trial : int
        The number of the tuning trial the hyperparameters come from.
    hyperparameters : Hyperparameters
        The study's tuned hyperparameters.
    root : pathlib.Path
        The data folder.
    results : pathlib.Path
        The folder to save to.

    Returns
    -------
    Record
        What is also written to ``<dataset>.json``: the model, the tuning
        trial, the hyperparameters, the validation MSE, best epoch and epochs,
        the test RMSE and R², the completeness gap of the attributions, whether
        training diverged, and the seconds it all took. If it diverged, the
        scores are ``None`` and only the ``.json`` is saved.

    The ``.pt`` holds the same values, the ``state_dict`` and, for a VNN, the
    ``covariance`` passed to `training.build`. ``<dataset>_ig.csv`` holds, per
    cohort row, its player, round, impostor flag, the model's prediction, and
    the attribution of each feature. The completeness gap is the largest
    difference between a row's attributions summed and its prediction minus
    the prediction for the all-zero input.
    """
    start: float = time.perf_counter()
    formula: str
    kind: str
    dataset: str
    formula, kind, dataset = job
    x: torch.Tensor
    y: torch.Tensor
    x_valid: torch.Tensor
    y_valid: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_cohort: torch.Tensor
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, x_valid, y_valid, x_test, y_test, x_cohort, _ = (
        t.to(device)
        for split in ("train", "validation", "test", "cohort")
        for t in training.load(formula, dataset, split, root)
    )
    covariance: torch.Tensor = torch.cov(x.T)
    model: torch.nn.Module = (
        training.build(covariance, hyperparameters, SEED)
        if kind == "vnn"
        else training.build_mlp(hyperparameters, SEED).to(device)
    )
    record: Record = {
        "formula": formula,
        "model": kind,
        "dataset": dataset,
        "tuning_trial": trial,
        "hyperparameters": asdict(hyperparameters),
    }
    path: Path = results / formula / kind / dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    validation_mse: float
    best_epoch: int
    epochs: int
    try:
        validation_mse, best_epoch, epochs = training.train(
            model,
            x,
            y,
            x_valid,
            y_valid,
            hyperparameters.lr,
            hyperparameters.batch_size,
            SEED,
        )
    except FloatingPointError:
        record |= dict.fromkeys(
            ("validation_mse", "best_epoch", "epochs", "test_rmse", "test_r2")
        )
        record |= {
            "completeness_gap": None,
            "diverged": True,
            "seconds": time.perf_counter() - start,
        }
        tune._write_json(record, path.with_suffix(".json"))
        return record
    error: torch.Tensor = training.predict(model, x_test) - y_test
    spread: torch.Tensor = (y_test - y_test.mean()).square().sum()
    prediction: torch.Tensor = training.predict(model, x_cohort)
    zero: torch.Tensor = training.predict(model, torch.zeros_like(x_cohort[:1]))
    attributions: torch.Tensor = explain.integrated_gradients(model, x_cohort)
    gap: float = (attributions.sum(1) - (prediction - zero)).abs().max().item()
    frame: pd.DataFrame = pd.read_csv(
        root / formula / dataset / "cohort.csv",
        usecols=["player", "round", "impostor"],
        encoding="utf-8",
    )
    frame["prediction"] = prediction.cpu().numpy()
    frame[list(FEATURES)] = attributions.cpu().numpy()
    frame.to_csv(path.parent / f"{dataset}_ig.csv", index=False, lineterminator="\n")
    record |= {
        "validation_mse": validation_mse,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "test_rmse": error.square().mean().sqrt().item(),
        "test_r2": 1 - (error.square().sum() / spread).item(),
        "completeness_gap": gap,
        "diverged": False,
        "seconds": time.perf_counter() - start,
    }
    saved: Record = record | {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}
    }
    if kind == "vnn":
        saved["covariance"] = covariance.cpu()
    torch.save(saved, path.with_suffix(".pt"))
    tune._write_json(record, path.with_suffix(".json"))
    return record


def run(
    copy: int,
    settings: dict[tuple[str, str], Setting],
    executor: concurrent.futures.Executor,
    root: Path = training.DATA,
    results: Path = RESULTS,
) -> int:
    """Train this copy's models that are not saved yet.

    Parameters
    ----------
    copy : int
        This copy's number, from 0 to ``COPIES - 1``.
    settings : dict[tuple[str, str], Setting]
        Each ``(model, formula)``'s tuning trial number and hyperparameters,
        from `tuned`.
    executor : concurrent.futures.Executor
        Runs `train_one`, ``SLOTS`` at a time.
    root : pathlib.Path
        The data folder.
    results : pathlib.Path
        The folder to save to.

    Returns
    -------
    int
        The number of models trained.
    """
    mine: list[Model] = models()[copy::COPIES]
    todo: list[Model] = [
        job
        for job in mine
        if not (results / job[0] / job[1] / f"{job[2]}.json").exists()
    ]
    tune._log(f"{len(mine) - len(todo)} of this copy's {len(mine)} models are saved")
    pending: dict[concurrent.futures.Future[Record], Model] = {
        executor.submit(train_one, job, *settings[job[1], job[0]], root, results): job
        for job in todo
    }
    done: int
    future: concurrent.futures.Future[Record]
    for done, future in enumerate(
        concurrent.futures.as_completed(pending), start=len(mine) - len(todo) + 1
    ):
        record: Record = future.result()
        name: str = " ".join(pending[future])
        if record["diverged"]:
            tune._log(f"{name} diverged ({done}/{len(mine)})")
            continue
        tune._log(
            f"{name} done in {record['seconds'] / 60:.1f} min ({done}/{len(mine)}): "
            f"{record['epochs']} epochs (best {record['best_epoch']}), "
            f"test RMSE {record['test_rmse']:.5f}, R² {record['test_r2']:.5f}, "
            f"IG gap {record['completeness_gap']:.2e}"
        )
    return len(todo)


def summarize(results: Path) -> dict[str, object]:
    """Write ``metrics.csv`` and return the summary of all saved models.

    Parameters
    ----------
    results : pathlib.Path
        The folder holding the models and the ``nodes`` records.

    Returns
    -------
    dict[str, object]
        ``"gpu_hours"`` over all copies and, per ``<model>_<formula>``, its
        tuning trial and hyperparameters, the number of models, how many
        diverged, the mean and standard deviation of the test RMSE and R², and
        the largest completeness gap.
    """
    records: list[Record] = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results.glob("*/*/*.json"))
    ]
    summary: dict[str, object] = {
        "gpu_hours": sum(
            json.loads(path.read_text(encoding="utf-8"))["hours"]
            for path in sorted((results / "nodes").glob("*.json"))
        )
    }
    if not records:
        return summary
    frame: pd.DataFrame = pd.DataFrame(records)
    frame.drop(columns=["tuning_trial", "hyperparameters"]).to_csv(
        results / "metrics.csv", index=False, lineterminator="\n"
    )
    formula: str
    kind: str
    for formula, kind in itertools.product(RULES, MODELS):
        group: pd.DataFrame = frame[
            (frame["formula"] == formula) & (frame["model"] == kind)
        ]
        if group.empty:
            continue
        summary[f"{kind}_{formula}"] = {
            "tuning_trial": int(group["tuning_trial"].iloc[0]),
            "hyperparameters": group["hyperparameters"].iloc[0],
            "models": len(group),
            "diverged": int(group["diverged"].sum()),
            **{
                f"{statistic}_{column}": float(getattr(group[column], statistic)())
                for column in ("test_rmse", "test_r2")
                for statistic in ("mean", "std")
            },
            "max_completeness_gap": float(group["completeness_gap"].max()),
        }
    return summary


def main(copy: int) -> None:
    """Train copy number ``copy``'s share of the final models.

    Parameters
    ----------
    copy : int
        This copy's number, from 0 to ``COPIES - 1``.
    """
    if not 0 <= copy < COPIES:
        raise typer.BadParameter(f"copy must be from 0 to {COPIES - 1}")
    signal.signal(signal.SIGTERM, tune._stop)
    settings: dict[tuple[str, str], Setting] = {
        (model, formula): tuned(model, formula)
        for model, formula in itertools.product(MODELS, RULES)
    }
    job: str = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get(
        "SLURM_JOB_ID", "local"
    )
    began: datetime = datetime.now(UTC).astimezone()
    tune._log(f"copy {copy} on {socket.gethostname()}, job {job}")
    key: tuple[str, str]
    setting: Setting
    for key, setting in settings.items():
        tune._log(f"{'_'.join(key)}: trial {setting[0]}, {setting[1]}")
    trained: int = 0
    executor: concurrent.futures.ProcessPoolExecutor = (
        concurrent.futures.ProcessPoolExecutor(
            SLOTS,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=tune._start_worker,
        )
    )
    try:
        trained = run(copy, settings, executor)
    finally:
        # Stop running trainings too, rather than waiting for them.
        executor.terminate_workers()
        ended: datetime = datetime.now(UTC).astimezone()
        hours: float = (ended - began).total_seconds() / 3600
        tune._write_json(
            {
                "copy": copy,
                "job": job,
                "host": socket.gethostname(),
                "start": began.isoformat(timespec="seconds"),
                "end": ended.isoformat(timespec="seconds"),
                "hours": hours,
            },
            RESULTS / "nodes" / f"{copy}_{job}.json",
        )
        # One copy at a time, so the last to write has seen every model.
        lock: JournalFileOpenLock = JournalFileOpenLock(str(RESULTS / "summary.json"))
        lock.acquire()
        try:
            tune._write_json(summarize(RESULTS), RESULTS / "summary.json")
        finally:
            lock.release()
        tune._log(f"copy {copy} stopped after {hours:.2f} h: {trained} models")


if __name__ == "__main__":
    typer.run(main)
