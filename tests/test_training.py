# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for VNN training and hyperparameter search."""

import optuna
import pytest
import torch

from eigenvibes import training
from eigenvibes.training import Hyperparameters
from eigenvibes.vnn import VNN


def _data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``n`` samples of 4 features and a noisy linear target.

    Parameters
    ----------
    n : int
        Number of samples.

    Returns
    -------
    x : torch.Tensor
        Features, shape ``(n, 4)``.
    y : torch.Tensor
        Targets, shape ``(n,)``.
    """
    generator: torch.Generator = torch.Generator().manual_seed(42)
    x: torch.Tensor = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    y: torch.Tensor = x.sum(dim=1) + torch.randn(
        n, generator=generator, dtype=torch.float64
    )
    return x, y


def _interrupt(trial: optuna.Trial) -> float:
    """Stand in for an objective that the user stops with Ctrl-C.

    Parameters
    ----------
    trial : optuna.Trial
        The trial being run.

    Returns
    -------
    float
        Never returns.

    Raises
    ------
    KeyboardInterrupt
        Always, after a hyperparameter has been drawn.
    """
    trial.suggest_float("lr", *training.LEARNING_RATES, log=True)
    raise KeyboardInterrupt


def test_resumed_search_matches_uninterrupted_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that stopping and resuming a search changes none of its trials.

    One study runs 12 trials at once. Another runs 6, then has one trial left
    running, as a killed process leaves it, and one failed by Ctrl-C. It is
    then loaded again from its storage with a fresh `optuna.Study` object and
    resumed to 12 finished trials. The finished trials must have the same
    hyperparameters and bitwise the same values as the uninterrupted ones.
    Twelve trials go past the 10 random start-up trials of `TPESampler`, so
    the last trials come from the fitted density model, which depends on the
    earlier trials. The search space and training are shrunk to keep the test
    fast.
    """
    monkeypatch.setattr(training, "FOLDS", 3)
    monkeypatch.setattr(training, "MAX_EPOCHS", 4)
    monkeypatch.setattr(training, "PATIENCE", 2)
    monkeypatch.setattr(training, "MAX_WIDTH", 4)
    monkeypatch.setattr(training, "MAX_TAPS", 3)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x: torch.Tensor
    y: torch.Tensor
    x, y = _data(30)

    uninterrupted: optuna.Study = optuna.create_study()
    training.search(uninterrupted, x, y, n_trials=12, seed=42)

    storage: optuna.storages.InMemoryStorage = optuna.storages.InMemoryStorage()
    stopped: optuna.Study = optuna.create_study(storage=storage, study_name="resumed")
    training.search(stopped, x, y, n_trials=6, seed=42)
    stopped.ask()
    with pytest.raises(KeyboardInterrupt):
        stopped.optimize(_interrupt, n_trials=1)
    resumed: optuna.Study = optuna.load_study(study_name="resumed", storage=storage)
    training.search(resumed, x, y, n_trials=12, seed=42)

    finished: list[list[optuna.trial.FrozenTrial]] = [
        [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            or trial.user_attrs.get("diverged", False)
        ]
        for study in (resumed, uninterrupted)
    ]
    assert len(finished[0]) == 12
    assert [t.params for t in finished[0]] == [t.params for t in finished[1]]
    assert [t.value for t in finished[0]] == [t.value for t in finished[1]]


def test_early_stopping_restores_best_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that early stopping leaves the model at its best epoch's weights.

    A second model with the same seeds, trained for exactly the returned
    number of epochs, must hold bitwise the same parameters. The learning rate
    is high enough that the validation loss stops improving before
    ``MAX_EPOCHS``, so the restored weights differ from the last epoch's.
    """
    monkeypatch.setattr(training, "MAX_EPOCHS", 60)
    monkeypatch.setattr(training, "PATIENCE", 5)
    x: torch.Tensor
    y: torch.Tensor
    x, y = _data(40)
    hyperparameters: Hyperparameters = Hyperparameters(
        lr=0.1, batch_size=4, widths=(8, 4), taps=(3, 2)
    )
    model: VNN = training.build(x[:30], hyperparameters, seed=0)
    best_epoch: int = training.train_early_stopping(
        model, x[:30], y[:30], x[30:], y[30:], hyperparameters, seed=1
    )
    assert best_epoch < 60 - 5

    replay: VNN = training.build(x[:30], hyperparameters, seed=0)
    optimizer: torch.optim.Adam = torch.optim.Adam(replay.parameters(), lr=0.1)
    generator: torch.Generator = torch.Generator().manual_seed(1)
    for _ in range(best_epoch):
        training._train_epoch(replay, x[:30], y[:30], optimizer, 4, generator)
    name: str
    for name, value in replay.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
