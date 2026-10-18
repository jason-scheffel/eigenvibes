# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tune the VNN's hyperparameters with Optuna on one GPU.

Each formula has one Optuna study, stored in ``results/tuning/<formula>.log``,
an Optuna journal file that several copies of this script can share. A trial
trains one set of hyperparameters on each of the 10 tuning datasets, with the
covariance estimated from that dataset's training split and seed 42, and
scores it by the mean validation MSE. A copy keeps ``SLOTS`` trainings running
on its GPU, each in its own process; when one finishes, it starts the next
dataset of an unfinished trial, or asks Optuna for a new trial. It stops once
the study has ``TRIALS`` trials that are finished or running. A trial whose
training diverges on some dataset is marked as failed with the user attribute
``"diverged"`` and counts as finished.

The copy number names the copy. A trial records which copy ran it, and a copy
that starts again marks its own unfinished trials as failed, so a stopped run
resumes by starting the same copies again. Two copies must never share a copy
number.

When it stops, a copy writes ``results/tuning/nodes/<formula>_<copy>_<job>.json``
with its run time and the trainings it ran, and rewrites
``results/tuning/summary.json`` from all such files and both studies.

Run it with ``uv run python scripts/tune.py <formula> <copy>``; on the DGX,
submit ``scripts/tune.sbatch``, which starts 4 copies per formula.
"""

import concurrent.futures
import json
import math
import multiprocessing
import os
import signal
import socket
import statistics
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import optuna
import torch
import typer
from optuna.storages.journal import (
    JournalFileBackend,
    JournalFileOpenLock,
    JournalStorage,
)

from eigenvibes import training
from eigenvibes.synthetic_data import RULES, TUNING_SEEDS
from eigenvibes.training import Hyperparameters

RESULTS: Path = Path(__file__).resolve().parent.parent / "results" / "tuning"
TRIALS: int = 200
SLOTS: int = 4
SEED: int = 42
DATASETS: tuple[str, ...] = tuple(f"tuning_{seed}" for seed in TUNING_SEEDS)

type Run = tuple[float, int, int, float]


def _log(message: str) -> None:
    """Print a line with the time, flushed so Slurm's log shows it at once."""
    print(f"[{datetime.now(UTC).astimezone():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def _storage(path: Path) -> JournalStorage:
    """Open an Optuna journal file with a lock that works on network drives."""
    return JournalStorage(
        JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path)))
    )


def _taken(study: optuna.Study, below: int) -> int:
    """Count trials numbered below ``below`` that are running or finished."""
    return sum(
        t.number < below
        and (
            t.state == optuna.trial.TrialState.RUNNING
            or t.state == optuna.trial.TrialState.COMPLETE
            or t.user_attrs.get("diverged", False)
        )
        for t in study.get_trials(deepcopy=False)
    )


def _start_worker() -> None:
    """Make a training process deterministic and keep it to one CPU thread."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def train_one(
    formula: str, dataset: str, hyperparameters: Hyperparameters, root: Path
) -> Run:
    """Train one model on one tuning dataset.

    Parameters
    ----------
    formula : str
        ``"linear"`` or ``"interaction"``.
    dataset : str
        Folder name, such as ``"tuning_30"``.
    hyperparameters : Hyperparameters
        The trial's hyperparameters.
    root : pathlib.Path
        The data folder.

    Returns
    -------
    validation_mse : float
        The lowest validation mean squared error.
    best_epoch : int
        The epoch with that error.
    epochs : int
        The epochs trained.
    seconds : float
        The time the training took, including loading the data.

    If training diverged, the error is infinite and the epochs are 0.
    """
    start: float = time.perf_counter()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x: torch.Tensor
    y: torch.Tensor
    x_valid: torch.Tensor
    y_valid: torch.Tensor
    x, y = training.load(formula, dataset, "train", root)
    x_valid, y_valid = training.load(formula, dataset, "validation", root)
    x, y, x_valid, y_valid = (t.to(device) for t in (x, y, x_valid, y_valid))
    model: torch.nn.Module = training.build(torch.cov(x.T), hyperparameters, SEED)
    result: tuple[float, int, int]
    try:
        result = training.train(
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
        result = (math.inf, 0, 0)
    return (*result, time.perf_counter() - start)


def tune(
    study: optuna.Study,
    formula: str,
    copy: int,
    executor: concurrent.futures.Executor,
    counts: dict[str, float],
    root: Path = training.DATA,
) -> None:
    """Run trials of ``study`` until it holds ``TRIALS`` finished or running ones.

    Parameters
    ----------
    study : optuna.Study
        The formula's study.
    formula : str
        ``"linear"`` or ``"interaction"``.
    copy : int
        This copy's number.
    executor : concurrent.futures.Executor
        Runs `train_one`, ``SLOTS`` at a time.
    counts : dict[str, float]
        Updated in place as work finishes, so it is current even if the run
        is stopped: ``"trials"`` this copy finished, including diverged ones,
        ``"trainings"`` it finished, and their total ``"training_seconds"``.
    root : pathlib.Path
        The data folder.
    """
    orphans: list[int] = [
        trial.number
        for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,))
        if trial.user_attrs.get("copy") == copy
    ]
    orphan: int
    for orphan in orphans:
        study.tell(orphan, state=optuna.trial.TrialState.FAIL)
    if orphans:
        _log(f"marked this copy's unfinished trials {orphans} as failed")
    open_trials: dict[int, tuple[optuna.Trial, Hyperparameters, list[str]]] = {}
    runs: dict[int, dict[str, dict[str, float | int]]] = {}
    started: dict[int, float] = {}
    pending: dict[concurrent.futures.Future[Run], tuple[int, str]] = {}

    def next_task() -> tuple[int, str] | None:
        number: int
        for number, (_, _, waiting) in open_trials.items():
            if waiting:
                return number, waiting.pop(0)
        if _taken(study, len(study.trials)) >= TRIALS:
            return None
        # A stop between these two writes would leave a running trial that no
        # copy owns, so no restart would ever fail it. Hold SIGTERM until both
        # are done; Windows can't hold signals, but Slurm doesn't run there.
        holding: bool = hasattr(signal, "pthread_sigmask")
        if holding:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            new: optuna.Trial = study.ask()
            new.set_user_attr("copy", copy)
        finally:
            if holding:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
        # Another copy may have asked at the same moment; the lower number wins.
        taken: int = _taken(study, new.number)
        if taken >= TRIALS:
            study.tell(new, state=optuna.trial.TrialState.FAIL)
            return None
        hyperparameters: Hyperparameters = training.suggest(new)
        open_trials[new.number] = (new, hyperparameters, list(DATASETS))
        runs[new.number] = {}
        started[new.number] = time.perf_counter()
        _log(f"start trial {new.number} ({taken + 1}/{TRIALS}): {hyperparameters}")
        return new.number, open_trials[new.number][2].pop(0)

    while True:
        task: tuple[int, str] | None
        while len(pending) < SLOTS and (task := next_task()) is not None:
            pending[
                executor.submit(
                    train_one, formula, task[1], open_trials[task[0]][1], root
                )
            ] = task
        if not pending:
            break
        done: set[concurrent.futures.Future[Run]] = concurrent.futures.wait(
            pending, return_when=concurrent.futures.FIRST_COMPLETED
        ).done
        future: concurrent.futures.Future[Run]
        for future in done:
            number: int
            dataset: str
            number, dataset = pending.pop(future)
            validation_mse, best_epoch, epochs, seconds = future.result()
            counts["trainings"] += 1
            counts["training_seconds"] += seconds
            if number not in open_trials:  # the trial already diverged
                continue
            new_trial: optuna.Trial = open_trials[number][0]
            if not math.isfinite(validation_mse):
                _log(f"trial {number} diverged on {dataset}")
                new_trial.set_user_attr("diverged", True)
                study.tell(number, state=optuna.trial.TrialState.FAIL)
                del open_trials[number]
                counts["trials"] += 1
                continue
            runs[number][dataset] = {
                "validation_mse": validation_mse,
                "best_epoch": best_epoch,
                "epochs": epochs,
                "seconds": seconds,
            }
            _log(
                f"  trial {number} {dataset} done in {seconds / 60:.1f} min: "
                f"{epochs} epochs (best {best_epoch}), val MSE {validation_mse:.5f}"
            )
            if len(runs[number]) == len(DATASETS):
                score: float = statistics.fmean(
                    float(run["validation_mse"]) for run in runs[number].values()
                )
                new_trial.set_user_attr("datasets", runs[number])
                study.tell(number, score)
                del open_trials[number]
                counts["trials"] += 1
                _log(
                    f"trial {number} done in "
                    f"{(time.perf_counter() - started[number]) / 60:.1f} min: "
                    f"mean val MSE {score:.5f}"
                )


def summarize(results: Path) -> dict[str, object]:
    """Collect run times and the best trial of each formula's study.

    Parameters
    ----------
    results : pathlib.Path
        The folder holding the studies and the ``nodes`` records.

    Returns
    -------
    dict[str, object]
        One entry per formula whose study exists.
    """
    summary: dict[str, object] = {}
    formula: str
    for formula in RULES:
        path: Path = results / f"{formula}.log"
        if not path.exists():
            continue
        study: optuna.Study = optuna.load_study(
            study_name=formula, storage=_storage(path)
        )
        trials: list[optuna.trial.FrozenTrial] = study.get_trials(deepcopy=False)
        complete: list[optuna.trial.FrozenTrial] = [
            t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        diverged: int = sum(t.user_attrs.get("diverged", False) for t in trials)
        nodes: list[dict[str, float]] = [
            json.loads(record.read_text(encoding="utf-8"))
            for record in sorted((results / "nodes").glob(f"{formula}_*.json"))
        ]
        runs: list[dict[str, float]] = [
            run for t in complete for run in t.user_attrs["datasets"].values()
        ]
        starts: list[datetime] = [
            t.datetime_start for t in trials if t.datetime_start is not None
        ]
        ends: list[datetime] = [
            t.datetime_complete for t in trials if t.datetime_complete is not None
        ]
        entry: dict[str, object] = {
            "finished_trials": len(complete) + diverged,
            "diverged_trials": diverged,
            "gpu_hours": sum(node["hours"] for node in nodes),
            "training_hours": sum(node["training_seconds"] for node in nodes) / 3600,
        }
        if starts and ends:
            entry["wall_hours"] = (max(ends) - min(starts)).total_seconds() / 3600
        if runs:
            entry["median_training_minutes"] = (
                statistics.median(run["seconds"] for run in runs) / 60
            )
            entry["longest_training_minutes"] = max(run["seconds"] for run in runs) / 60
            entry["median_epochs"] = statistics.median(run["epochs"] for run in runs)
        if complete:
            best: optuna.trial.FrozenTrial = study.best_trial
            entry["best_trial"] = {
                "number": best.number,
                "validation_mse": best.value,
                "hyperparameters": asdict(training.suggest(best)),
            }
        summary[formula] = entry
    return summary


def _write_json(data: object, path: Path) -> None:
    """Write JSON through a temporary file, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stop(signum: int, frame: object) -> None:
    """Exit on SIGTERM, ignoring repeats, so the records still get written.

    Slurm stops a job with SIGTERM, and ``uv run`` forwards its own copy, so a
    second one can arrive while the records are being written.
    """
    signal.signal(signum, signal.SIG_IGN)
    sys.exit(128 + signum)


def main(formula: str, copy: int) -> None:
    """Tune one formula's VNN as copy number ``copy``.

    Parameters
    ----------
    formula : str
        ``"linear"`` or ``"interaction"``.
    copy : int
        This copy's number; see the module docstring.
    """
    if formula not in RULES:
        raise typer.BadParameter(f"formula must be one of {', '.join(RULES)}")
    signal.signal(signal.SIGTERM, _stop)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    RESULTS.mkdir(parents=True, exist_ok=True)
    study: optuna.Study = optuna.create_study(
        study_name=formula,
        storage=_storage(RESULTS / f"{formula}.log"),
        sampler=optuna.samplers.TPESampler(seed=SEED + copy, constant_liar=True),
        load_if_exists=True,
    )
    # The array's job ID, which also names the Slurm log files.
    job: str = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get(
        "SLURM_JOB_ID", "local"
    )
    began: datetime = datetime.now(UTC).astimezone()
    _log(f"copy {copy} tuning {formula} on {socket.gethostname()}, job {job}")
    counts: dict[str, float] = {"trials": 0, "trainings": 0, "training_seconds": 0.0}
    executor: concurrent.futures.ProcessPoolExecutor = (
        concurrent.futures.ProcessPoolExecutor(
            SLOTS,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_start_worker,
        )
    )
    try:
        tune(study, formula, copy, executor, counts)
    finally:
        # Stop running trainings too, rather than waiting for them.
        executor.terminate_workers()
        ended: datetime = datetime.now(UTC).astimezone()
        _write_json(
            {
                "formula": formula,
                "copy": copy,
                "job": job,
                "host": socket.gethostname(),
                "start": began.isoformat(timespec="seconds"),
                "end": ended.isoformat(timespec="seconds"),
                "hours": (ended - began).total_seconds() / 3600,
                **counts,
            },
            RESULTS / "nodes" / f"{formula}_{copy}_{job}.json",
        )
        # One copy at a time, so the last to write has seen every record.
        lock: JournalFileOpenLock = JournalFileOpenLock(str(RESULTS / "summary.json"))
        lock.acquire()
        try:
            _write_json(summarize(RESULTS), RESULTS / "summary.json")
        finally:
            lock.release()
        _log(
            f"copy {copy} stopped after {(ended - began).total_seconds() / 3600:.2f} h: "
            f"{counts['trials']:.0f} trials, {counts['trainings']:.0f} trainings"
        )


if __name__ == "__main__":
    typer.run(main)
