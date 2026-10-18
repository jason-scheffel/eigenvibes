# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Train models on the synthetic datasets, and the VNN search space for Optuna.

A model trains with Adam on mean squared error, one step per batch of a fresh
random order of the training rows each epoch, until the validation loss has
not improved for ``PATIENCE`` epochs (at most ``MAX_EPOCHS``). The weights
from the epoch with the lowest validation loss are then restored.

The inputs are not centered: every behavior averages 0 by design, and
Integrated Gradients explains predictions relative to the input 0. A VNN's
covariance matrix is divided by its trace, whether it is the true covariance
or one estimated from training data. All VNN activations are
`torch.nn.PReLU`, one learned slope per layer.

For bit-for-bit reproducible results on a GPU, import ``eigenvibes`` and call
``torch.use_deterministic_algorithms(True)`` first.
"""

from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd
import torch

from eigenvibes.synthetic_data import FEATURES
from eigenvibes.vnn import VNN

DATA: Path = Path(__file__).resolve().parents[2] / "data"
MAX_EPOCHS: int = 1500
PATIENCE: int = 25
LEARNING_RATES: tuple[float, float] = (1e-4, 1e-1)
BATCH_SIZES: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)
MAX_LAYERS: int = 4
MAX_WIDTH: int = 256
MAX_TAPS: int = 12
PREDICT_BATCH: int = 512


@dataclass(frozen=True)
class Hyperparameters:
    """Hyperparameters of one VNN and its training.

    Attributes
    ----------
    lr : float
        Adam learning rate.
    batch_size : int
        Samples per training step.
    widths : tuple[int, ...]
        Output channels of each layer.
    taps : tuple[int, ...]
        Filter taps of each layer.
    """

    lr: float
    batch_size: int
    widths: tuple[int, ...]
    taps: tuple[int, ...]


def suggest(trial: optuna.trial.BaseTrial) -> Hyperparameters:
    """Draw hyperparameters from the search space.

    The learning rate is log-uniform on ``LEARNING_RATES``, the batch size is
    one of ``BATCH_SIZES``, and the model has 1 to ``MAX_LAYERS`` layers, each
    with 1 to ``MAX_WIDTH`` channels and 1 to ``MAX_TAPS`` taps.

    Parameters
    ----------
    trial : optuna.trial.BaseTrial
        A running trial, or a finished one such as ``study.best_trial``, for
        which this returns the hyperparameters it used.

    Returns
    -------
    Hyperparameters
        The drawn hyperparameters.
    """
    lr: float = trial.suggest_float("lr", *LEARNING_RATES, log=True)
    batch_size: int = trial.suggest_categorical("batch_size", BATCH_SIZES)
    layers: int = trial.suggest_int("layers", 1, MAX_LAYERS)
    widths: list[int] = []
    taps: list[int] = []
    layer: int
    for layer in range(layers):
        widths.append(trial.suggest_int(f"width_{layer}", 1, MAX_WIDTH))
        taps.append(trial.suggest_int(f"taps_{layer}", 1, MAX_TAPS))
    return Hyperparameters(lr, batch_size, tuple(widths), tuple(taps))


def load(
    rule: str, dataset: str, split: str, root: Path = DATA
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one split of a synthetic dataset.

    Parameters
    ----------
    rule : str
        ``"linear"`` or ``"interaction"``.
    dataset : str
        Folder name, such as ``"tuning_30"`` or ``"evaluation_07"``.
    split : str
        ``"train"``, ``"validation"``, or ``"test"``.
    root : pathlib.Path
        The folder written by ``scripts/make_datasets.py``.

    Returns
    -------
    x : torch.Tensor
        The 12 behaviors in the order of `FEATURES`, float64, shape ``(n, 12)``,
        on the CPU.
    y : torch.Tensor
        The ``sus`` column, float64, shape ``(n,)``, on the CPU.
    """
    frame: pd.DataFrame = pd.read_csv(
        root / rule / dataset / f"{split}.csv", float_precision="round_trip"
    )
    x: torch.Tensor = torch.tensor(
        frame[list(FEATURES)].to_numpy(), dtype=torch.float64
    )
    y: torch.Tensor = torch.tensor(frame["sus"].to_numpy(), dtype=torch.float64)
    return x, y


def build(covariance: torch.Tensor, hyperparameters: Hyperparameters, seed: int) -> VNN:
    """Create an untrained VNN with ``covariance`` divided by its trace.

    Parameters
    ----------
    covariance : torch.Tensor
        The true covariance, or ``torch.cov(x_train.T)``, shape ``(N, N)``.
    hyperparameters : Hyperparameters
        The model's widths and taps.
    seed : int
        Seed for the initial weights. This reseeds PyTorch's global random
        number generators.

    Returns
    -------
    VNN
        The model, on the device of ``covariance``.
    """
    torch.manual_seed(seed)
    model: VNN = VNN(
        covariance / covariance.trace(),
        hyperparameters.widths,
        hyperparameters.taps,
        torch.nn.PReLU,
    )
    return model.to(covariance.device)


def predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Predict without gradients, in fixed-size chunks.

    A VNN gets chunks of ``model.max_batch`` samples, which keeps every call
    within the ROCm row limit; other models get chunks of ``PREDICT_BATCH``.
    A fixed chunk size makes the result independent of how many samples ``x``
    holds.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    x : torch.Tensor
        Features, shape ``(n, N)``.

    Returns
    -------
    torch.Tensor
        Predictions, shape ``(n,)``.
    """
    chunk: int = model.max_batch if isinstance(model, VNN) else PREDICT_BATCH
    with torch.no_grad():
        return torch.cat([model(part) for part in torch.split(x, chunk)])


def _train_epoch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    generator: torch.Generator,
) -> None:
    """Take one Adam step per batch of a random permutation of the samples."""
    order: torch.Tensor = torch.randperm(len(x), generator=generator).to(x.device)
    batch: torch.Tensor
    for batch in torch.split(order, batch_size):
        optimizer.zero_grad()
        loss: torch.Tensor = torch.nn.functional.mse_loss(model(x[batch]), y[batch])
        torch.autograd.backward(loss)
        optimizer.step()


def train(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_valid: torch.Tensor,
    y_valid: torch.Tensor,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[float, int, int]:
    """Train until the validation loss stops improving, then restore the best.

    Training stops after ``PATIENCE`` epochs without a lower validation mean
    squared error, or after ``MAX_EPOCHS`` epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The model, trained in place, on the device of the data.
    x, y : torch.Tensor
        Training features ``(n, N)`` and targets ``(n,)``.
    x_valid, y_valid : torch.Tensor
        Validation features and targets.
    lr : float
        Adam learning rate.
    batch_size : int
        Samples per training step.
    seed : int
        Seed for the order of the training samples.

    Returns
    -------
    validation_mse : float
        The lowest validation mean squared error, which the model now has.
    best_epoch : int
        The epoch, counted from 1, whose weights the model now holds.
    epochs : int
        The number of epochs trained.

    Raises
    ------
    FloatingPointError
        If the validation loss is never finite, for example when training
        diverges from the first epoch.
    """
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    best_loss: float = float("inf")
    best_epoch: int = 0
    best_state: dict[str, torch.Tensor] | None = None
    epoch: int = 0
    while epoch < MAX_EPOCHS and epoch - best_epoch < PATIENCE:
        epoch += 1
        _train_epoch(model, x, y, optimizer, batch_size, generator)
        loss: float = torch.nn.functional.mse_loss(
            predict(model, x_valid), y_valid
        ).item()
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = {
                name: value.clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise FloatingPointError("The validation loss was never finite")
    model.load_state_dict(best_state)
    return best_loss, best_epoch, epoch
