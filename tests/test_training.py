# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for loading data and training models."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from eigenvibes import training
from eigenvibes.synthetic_data import FEATURES
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


def test_load_selects_behaviors_by_name(tmp_path: Path) -> None:
    """Behaviors come back in `FEATURES` order whatever the column order.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory from pytest.
    """
    expected: torch.Tensor = torch.arange(24, dtype=torch.float64).reshape(2, 12) / 7
    frame: pd.DataFrame = pd.DataFrame(expected.numpy(), columns=list(FEATURES))
    frame["sus"] = [0.1, 0.2]
    frame["player"] = [5, 6]
    folder: Path = tmp_path / "linear" / "tuning_30"
    folder.mkdir(parents=True)
    frame[["sus", "player", *reversed(FEATURES)]].to_csv(
        folder / "train.csv", index=False
    )
    x: torch.Tensor
    y: torch.Tensor
    x, y = training.load("linear", "tuning_30", "train", root=tmp_path)
    torch.testing.assert_close(x, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        y, torch.tensor([0.1, 0.2], dtype=torch.float64), rtol=0, atol=0
    )


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
    covariance: torch.Tensor = torch.cov(x[:30].T)
    hyperparameters: Hyperparameters = Hyperparameters(
        lr=0.1, batch_size=4, widths=(8, 4), taps=(3, 2)
    )
    model: VNN = training.build(covariance, hyperparameters, seed=0)
    validation_mse: float
    best_epoch: int
    epochs: int
    validation_mse, best_epoch, epochs = training.train(
        model, x[:30], y[:30], x[30:], y[30:], lr=0.1, batch_size=4, seed=1
    )
    assert epochs == best_epoch + 5 < 60
    assert (
        validation_mse
        == torch.nn.functional.mse_loss(training.predict(model, x[30:]), y[30:]).item()
    )

    replay: VNN = training.build(covariance, hyperparameters, seed=0)
    optimizer: torch.optim.Adam = torch.optim.Adam(replay.parameters(), lr=0.1)
    generator: torch.Generator = torch.Generator().manual_seed(1)
    for _ in range(best_epoch):
        training._train_epoch(replay, x[:30], y[:30], optimizer, 4, generator)
    name: str
    value: torch.Tensor
    for name, value in replay.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_training_that_never_has_a_finite_loss_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose validation loss is never finite raises instead of returning."""
    monkeypatch.setattr(training, "PATIENCE", 2)
    x: torch.Tensor
    y: torch.Tensor
    x, y = _data(20)
    model: VNN = training.build(
        torch.cov(x.T), Hyperparameters(0.01, 4, (2,), (1,)), seed=0
    )
    with pytest.raises(FloatingPointError):
        training.train(
            model, x, y, x, torch.full_like(y, torch.nan), lr=0.01, batch_size=4, seed=0
        )
