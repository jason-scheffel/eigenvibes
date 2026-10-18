# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Train the final VNNs on the evaluation datasets with the tuned settings.

Each formula's VNNs use the hyperparameters of the best trial of its Optuna
study in ``results/tuning/<formula>.log``, which must hold ``tune.TRIALS``
finished trials. For each formula, the script trains two kinds of VNN on each
of the 30 evaluation datasets with seeds 0, 1 and 2, 360 models in all:

- ``true``: the covariance is ``data/covariance.csv``, the one the behaviors
  were drawn from;
- ``estimated``: the covariance is estimated from the dataset's training
  split, as in tuning.

A model stops early on its dataset's validation split. It is then scored on
the test split, whose sus scores have no noise, by the root mean squared error
and R² of its predictions.

The models are split among ``COPIES`` copies of this script, one per GPU: copy
``k`` takes every ``COPIES``-th model of `models`, starting at the ``k``-th. A
copy trains ``SLOTS`` models at a time, each in its own process, and skips
models already saved, so a stopped run resumes by starting the same copies
again. Each model is saved to
``results/final/<formula>/<covariance>/<dataset>_seed_<seed>.pt``, and its
scores and training time to the ``.json`` next to it, which is written last.

When it stops, a copy writes ``results/final/nodes/<copy>_<job>.json`` with
its run time and the trainings it ran, and rewrites
``results/final/metrics.csv``, one row per saved model, and
``results/final/summary.json``.

Run it with ``uv run python scripts/train_final.py <copy>``; on the DGX,
submit ``scripts/train_final.sbatch``, which starts all 8 copies.
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

from eigenvibes import training
from eigenvibes.synthetic_data import EVALUATION_SEEDS, FEATURES, RULES
from eigenvibes.training import Hyperparameters

RESULTS: Path = Path(__file__).resolve().parent.parent / "results" / "final"
COPIES: int = 8
SLOTS: int = 4
SEEDS: tuple[int, ...] = (0, 1, 2)
COVARIANCES: tuple[str, ...] = ("true", "estimated")
DATASETS: tuple[str, ...] = tuple(f"evaluation_{seed:02d}" for seed in EVALUATION_SEEDS)

type Model = tuple[str, str, str, int]
type Record = dict[str, Any]


def models() -> list[Model]:
    """List every model as ``(formula, covariance, dataset, seed)``.

    Returns
    -------
    list[Model]
        All 360 models, in the same order every time.
    """
    return list(itertools.product(RULES, COVARIANCES, DATASETS, SEEDS))


def _path(results: Path, model: Model) -> Path:
    """Return a model's file path without its suffix."""
    formula, covariance, dataset, seed = model
    return results / formula / covariance / f"{dataset}_seed_{seed}"


def tuned(formula: str) -> tuple[int, Hyperparameters]:
    """Return the number and hyperparameters of a formula's best tuning trial.

    Parameters
    ----------
    formula : str
        ``"linear"`` or ``"interaction"``.

    Returns
    -------
    number : int
        The trial's number in the study.
    hyperparameters : Hyperparameters
        Its hyperparameters.

    Raises
    ------
    SystemExit
        If the study has fewer than ``tune.TRIALS`` finished trials.
    """
    study: optuna.Study = optuna.load_study(
        study_name=formula, storage=tune._storage(tune.RESULTS / f"{formula}.log")
    )
    finished: int = sum(
        t.state == optuna.trial.TrialState.COMPLETE
        or t.user_attrs.get("diverged", False)
        for t in study.get_trials(deepcopy=False)
    )
    if finished < tune.TRIALS:
        raise SystemExit(
            f"The {formula} study has {finished} of {tune.TRIALS} finished trials"
        )
    best: optuna.trial.FrozenTrial = study.best_trial
    return best.number, training.suggest(best)


def train_one(
    model: Model,
    trial: int,
    hyperparameters: Hyperparameters,
    root: Path,
    results: Path,
) -> Record:
    """Train, score and save one final model.

    Parameters
    ----------
    model : Model
        ``(formula, covariance, dataset, seed)``.
    trial : int
        The number of the tuning trial the hyperparameters come from.
    hyperparameters : Hyperparameters
        The formula's tuned hyperparameters.
    root : pathlib.Path
        The data folder.
    results : pathlib.Path
        The folder to save to.

    Returns
    -------
    Record
        What is also written to the model's ``.json``: the model, the tuning
        trial, the hyperparameters, the validation MSE, best epoch and epochs,
        the test RMSE and R², whether training diverged, and the seconds it
        took, including loading the data. If it diverged, the scores and
        epochs are ``None`` and no ``.pt`` is saved.

    The ``.pt`` holds the same values, plus the ``state_dict`` and, as
    ``covariance_matrix``, the covariance passed to `training.build`, which
    the ``state_dict`` does not include. ``training.build`` with that matrix,
    the hyperparameters and the seed, followed by ``load_state_dict``,
    restores the model.
    """
    start: float = time.perf_counter()
    formula, kind, dataset, seed = model
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, x_valid, y_valid, x_test, y_test = (
        t.to(device)
        for split in ("train", "validation", "test")
        for t in training.load(formula, dataset, split, root)
    )
    covariance: torch.Tensor = (
        torch.tensor(
            pd.read_csv(root / "covariance.csv", float_precision="round_trip")[
                list(FEATURES)
            ].to_numpy(),
            dtype=torch.float64,
            device=device,
        )
        if kind == "true"
        else torch.cov(x.T)
    )
    vnn: torch.nn.Module = training.build(covariance, hyperparameters, seed)
    record: Record = {
        "formula": formula,
        "covariance": kind,
        "dataset": dataset,
        "seed": seed,
        "tuning_trial": trial,
        "hyperparameters": asdict(hyperparameters),
    }
    path: Path = _path(results, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        validation_mse, best_epoch, epochs = training.train(
            vnn,
            x,
            y,
            x_valid,
            y_valid,
            hyperparameters.lr,
            hyperparameters.batch_size,
            seed,
        )
    except FloatingPointError:
        record |= dict.fromkeys(
            ("validation_mse", "best_epoch", "epochs", "test_rmse", "test_r2")
        )
        record |= {"diverged": True, "seconds": time.perf_counter() - start}
        tune._write_json(record, path.with_suffix(".json"))
        return record
    error: torch.Tensor = training.predict(vnn, x_test) - y_test
    spread: torch.Tensor = (y_test - y_test.mean()).square().sum()
    record |= {
        "validation_mse": validation_mse,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "test_rmse": error.square().mean().sqrt().item(),
        "test_r2": 1 - (error.square().sum() / spread).item(),
        "diverged": False,
        "seconds": time.perf_counter() - start,
    }
    temporary: Path = path.with_name(f".{path.name}.pt.{os.getpid()}")
    torch.save(
        record
        | {
            "state_dict": {k: v.cpu() for k, v in vnn.state_dict().items()},
            "covariance_matrix": covariance.cpu(),
        },
        temporary,
    )
    temporary.replace(path.with_suffix(".pt"))
    tune._write_json(record, path.with_suffix(".json"))
    return record


def run(
    copy: int,
    settings: dict[str, tuple[int, Hyperparameters]],
    executor: concurrent.futures.Executor,
    counts: dict[str, float],
    root: Path = training.DATA,
    results: Path = RESULTS,
) -> None:
    """Train this copy's models that are not saved yet.

    Parameters
    ----------
    copy : int
        This copy's number, from 0 to ``COPIES - 1``.
    settings : dict[str, tuple[int, Hyperparameters]]
        Each formula's tuning trial number and hyperparameters, from `tuned`.
    executor : concurrent.futures.Executor
        Runs `train_one`, ``SLOTS`` at a time.
    counts : dict[str, float]
        Updated in place as models finish, so it is current even if the run
        is stopped: the ``"trainings"`` this copy finished and their total
        ``"training_seconds"``.
    root : pathlib.Path
        The data folder.
    results : pathlib.Path
        The folder to save to.
    """
    mine: list[Model] = models()[copy::COPIES]
    todo: list[Model] = [
        model
        for model in mine
        if not _path(results, model).with_suffix(".json").exists()
    ]
    tune._log(f"{len(mine) - len(todo)} of this copy's {len(mine)} models are saved")
    pending: dict[concurrent.futures.Future[Record], Model] = {
        executor.submit(train_one, model, *settings[model[0]], root, results): model
        for model in todo
    }
    done: int
    future: concurrent.futures.Future[Record]
    for done, future in enumerate(
        concurrent.futures.as_completed(pending), start=len(mine) - len(todo) + 1
    ):
        record: Record = future.result()
        counts["trainings"] += 1
        counts["training_seconds"] += record["seconds"]
        name: str = " ".join(map(str, pending[future]))
        if record["diverged"]:
            tune._log(f"{name} diverged ({done}/{len(mine)})")
            continue
        tune._log(
            f"{name} done in {record['seconds'] / 60:.1f} min ({done}/{len(mine)}): "
            f"{record['epochs']} epochs (best {record['best_epoch']}), "
            f"test RMSE {record['test_rmse']:.5f}, R² {record['test_r2']:.5f}"
        )


def summarize(results: Path) -> dict[str, object]:
    """Write ``metrics.csv`` and return the summary of all saved models.

    Parameters
    ----------
    results : pathlib.Path
        The folder holding the models and the ``nodes`` records.

    Returns
    -------
    dict[str, object]
        ``"gpu_hours"`` and ``"training_hours"`` over all copies, and per
        formula, its tuning trial and hyperparameters and, per covariance, the
        number of models, how many diverged, and the mean and standard
        deviation of the test RMSE and R².
    """
    records: list[Record] = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results.glob("*/*/*.json"))
    ]
    nodes: list[Record] = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((results / "nodes").glob("*.json"))
    ]
    summary: dict[str, object] = {
        "gpu_hours": sum(node["hours"] for node in nodes),
        "training_hours": sum(node["training_seconds"] for node in nodes) / 3600,
    }
    if not records:
        return summary
    frame: pd.DataFrame = pd.DataFrame(records)
    frame.drop(columns=["tuning_trial", "hyperparameters"]).to_csv(
        results / "metrics.csv", index=False, lineterminator="\n"
    )
    formula: str
    for formula in RULES:
        rows: pd.DataFrame = frame[frame["formula"] == formula]
        if rows.empty:
            continue
        entry: dict[str, object] = {
            "tuning_trial": int(rows["tuning_trial"].iloc[0]),
            "hyperparameters": rows["hyperparameters"].iloc[0],
        }
        kind: str
        for kind in COVARIANCES:
            group: pd.DataFrame = rows[rows["covariance"] == kind]
            if group.empty:
                continue
            entry[kind] = {
                "models": len(group),
                "diverged": int(group["diverged"].sum()),
                **{
                    f"{statistic}_{column}": float(getattr(group[column], statistic)())
                    for column in ("test_rmse", "test_r2")
                    for statistic in ("mean", "std")
                },
            }
        summary[formula] = entry
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
    settings: dict[str, tuple[int, Hyperparameters]] = {
        formula: tuned(formula) for formula in RULES
    }
    job: str = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get(
        "SLURM_JOB_ID", "local"
    )
    began: datetime = datetime.now(UTC).astimezone()
    tune._log(f"copy {copy} on {socket.gethostname()}, job {job}")
    formula: str
    for formula, (trial, hyperparameters) in settings.items():
        tune._log(f"{formula}: trial {trial}, {hyperparameters}")
    counts: dict[str, float] = {"trainings": 0, "training_seconds": 0.0}
    executor: concurrent.futures.ProcessPoolExecutor = (
        concurrent.futures.ProcessPoolExecutor(
            SLOTS,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=tune._start_worker,
        )
    )
    try:
        run(copy, settings, executor, counts)
    finally:
        # Stop running trainings too, rather than waiting for them.
        executor.terminate_workers()
        ended: datetime = datetime.now(UTC).astimezone()
        tune._write_json(
            {
                "copy": copy,
                "job": job,
                "host": socket.gethostname(),
                "start": began.isoformat(timespec="seconds"),
                "end": ended.isoformat(timespec="seconds"),
                "hours": (ended - began).total_seconds() / 3600,
                **counts,
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
        tune._log(
            f"copy {copy} stopped after {(ended - began).total_seconds() / 3600:.2f} h: "
            f"{counts['trainings']:.0f} trainings"
        )


if __name__ == "__main__":
    typer.run(main)
