# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Train VNNs and tune their hyperparameters with Optuna.

Each Optuna trial scores one set of hyperparameters by ``FOLDS``-fold cross
validation. On every fold, the features are centered with the training folds'
mean, the model's covariance matrix is the training folds' covariance divided
by its trace, and the model trains with Adam on mean squared error until the
validation loss has not improved for ``PATIENCE`` epochs (at most
``MAX_EPOCHS``). The weights from the best epoch are restored and scored by
their mean absolute error on the validation fold. A trial's value is the mean
of these errors over the folds.

The final model is then trained on all the data, without early stopping, for
`final_epochs` of the best trial: the median over the folds of the epoch with
the lowest validation loss.

All activations are `torch.nn.PReLU`, one learned slope per layer.

The functions expect float64 tensors on one device: features ``x`` of shape
``(n, N)`` and targets ``y`` of shape ``(n,)``. For bit-for-bit reproducible
results on a GPU, call ``torch.use_deterministic_algorithms(True)`` and set
the environment variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` first.
"""

import statistics
from dataclasses import dataclass

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold

from eigenvibes.vnn import VNN

FOLDS: int = 10
MAX_EPOCHS: int = 1000
PATIENCE: int = 20
LEARNING_RATES: tuple[float, float] = (1e-4, 1e-1)
BATCH_SIZES: tuple[int, ...] = (2, 4, 8, 12, 16, 32, 64)
MAX_LAYERS: int = 3
MAX_WIDTH: int = 256
MAX_TAPS: int = 32


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


def build(x: torch.Tensor, hyperparameters: Hyperparameters, seed: int) -> VNN:
    """Create an untrained model whose covariance matrix comes from ``x``.

    Parameters
    ----------
    x : torch.Tensor
        Training features, shape ``(n, N)``.
    hyperparameters : Hyperparameters
        The model's widths and taps.
    seed : int
        Seed for the initial weights. This reseeds PyTorch's global random
        number generators.

    Returns
    -------
    VNN
        The model, on the device of ``x``, with covariance matrix
        ``cov(x) / trace(cov(x))``.
    """
    covariance: torch.Tensor = torch.cov(x.T)
    torch.manual_seed(seed)
    model: VNN = VNN(
        covariance / covariance.trace(),
        hyperparameters.widths,
        hyperparameters.taps,
        torch.nn.PReLU,
    )
    return model.to(x.device)


def predict(model: VNN, x: torch.Tensor) -> torch.Tensor:
    """Predict without gradients, in chunks of ``model.max_batch`` samples.

    The fixed chunk size keeps every call within the ROCm row limit and makes
    the result independent of how many samples ``x`` holds.

    Parameters
    ----------
    model : VNN
        The model.
    x : torch.Tensor
        Features, shape ``(n, N)``.

    Returns
    -------
    torch.Tensor
        Predictions, shape ``(n,)``.
    """
    with torch.no_grad():
        return torch.cat([model(chunk) for chunk in torch.split(x, model.max_batch)])


def _train_epoch(
    model: VNN,
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


def train_early_stopping(
    model: VNN,
    x: torch.Tensor,
    y: torch.Tensor,
    x_valid: torch.Tensor,
    y_valid: torch.Tensor,
    hyperparameters: Hyperparameters,
    seed: int,
) -> int:
    """Train until the validation loss stops improving, then restore the best.

    Training stops after ``PATIENCE`` epochs without a lower validation mean
    squared error, or after ``MAX_EPOCHS`` epochs.

    Parameters
    ----------
    model : VNN
        The model, trained in place.
    x, y : torch.Tensor
        Training features ``(n, N)`` and targets ``(n,)``.
    x_valid, y_valid : torch.Tensor
        Validation features and targets.
    hyperparameters : Hyperparameters
        The learning rate and batch size.
    seed : int
        Seed for the order of the training samples.

    Returns
    -------
    int
        The epoch, counted from 1, whose weights the model now holds.

    Raises
    ------
    FloatingPointError
        If the validation loss is never finite, for example when training
        diverges from the first epoch.
    """
    optimizer: torch.optim.Adam = torch.optim.Adam(
        model.parameters(), lr=hyperparameters.lr
    )
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    best_loss: float = float("inf")
    best_epoch: int = 0
    best_state: dict[str, torch.Tensor] | None = None
    epoch: int
    for epoch in range(1, MAX_EPOCHS + 1):
        _train_epoch(model, x, y, optimizer, hyperparameters.batch_size, generator)
        loss: float = torch.nn.functional.mse_loss(
            predict(model, x_valid), y_valid
        ).item()
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = {
                name: value.clone() for name, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= PATIENCE:
            break
    if best_state is None:
        raise FloatingPointError("The validation loss was never finite")
    model.load_state_dict(best_state)
    return best_epoch


def cross_validate(
    x: torch.Tensor, y: torch.Tensor, hyperparameters: Hyperparameters, seed: int
) -> tuple[list[float], list[int]]:
    """Score hyperparameters by ``FOLDS``-fold cross validation.

    The folds are shuffled with ``seed``, and fold ``f = 1, ..., FOLDS`` uses
    seed ``seed + f`` for its initial weights and sample order, so every trial
    of a study sees the same folds and seeds.

    Parameters
    ----------
    x, y : torch.Tensor
        Features ``(n, N)`` and targets ``(n,)``.
    hyperparameters : Hyperparameters
        The hyperparameters to score.
    seed : int
        Seed for the folds, the initial weights, and the sample order.

    Returns
    -------
    fold_mae : list[float]
        Each fold's validation mean absolute error.
    fold_best_epochs : list[int]
        Each fold's epoch with the lowest validation loss.
    """
    fold_mae: list[float] = []
    fold_best_epochs: list[int] = []
    folds: KFold = KFold(FOLDS, shuffle=True, random_state=seed)
    fold: int
    train_index: np.ndarray
    valid_index: np.ndarray
    for fold, (train_index, valid_index) in enumerate(
        folds.split(np.arange(len(x))), start=1
    ):
        train: torch.Tensor = torch.as_tensor(train_index, device=x.device)
        valid: torch.Tensor = torch.as_tensor(valid_index, device=x.device)
        mean: torch.Tensor = x[train].mean(dim=0)
        x_train: torch.Tensor = x[train] - mean
        x_valid: torch.Tensor = x[valid] - mean
        model: VNN = build(x_train, hyperparameters, seed + fold)
        fold_best_epochs.append(
            train_early_stopping(
                model,
                x_train,
                y[train],
                x_valid,
                y[valid],
                hyperparameters,
                seed + fold,
            )
        )
        error: torch.Tensor = predict(model, x_valid) - y[valid]
        fold_mae.append(error.abs().mean().item())
    return fold_mae, fold_best_epochs


def search(
    study: optuna.Study, x: torch.Tensor, y: torch.Tensor, n_trials: int, seed: int
) -> None:
    """Run trials in ``study`` until it holds ``n_trials`` finished ones.

    Trials run one at a time. A trial whose training diverges on some fold is
    marked as failed, with the user attribute ``"diverged"``, and the search
    continues; it counts as finished. Before finished trial ``t``, counted
    from 0, the study gets a new ``TPESampler`` with seed ``seed + t``.

    A study can be stopped at any point and resumed from its storage. A trial
    cut short, which a killed process leaves running and Ctrl-C leaves failed
    without ``"diverged"``, is marked as failed and not counted, so the
    resumed study picks the same hyperparameters as one run without stopping.
    Only one process may use a study at a time: a trial that another process
    is still running would be marked as failed.

    Parameters
    ----------
    study : optuna.Study
        A study that minimizes, new or loaded from storage.
    x, y : torch.Tensor
        Features ``(n, N)`` and targets ``(n,)``.
    n_trials : int
        Number of finished trials the study should hold.
    seed : int
        Seed for the sampler and for `cross_validate`.
    """

    def objective(trial: optuna.Trial) -> float:
        fold_mae: list[float]
        fold_best_epochs: list[int]
        try:
            fold_mae, fold_best_epochs = cross_validate(x, y, suggest(trial), seed)
        except FloatingPointError:
            trial.set_user_attr("diverged", True)
            raise
        trial.set_user_attr("fold_mae", fold_mae)
        trial.set_user_attr("fold_best_epochs", fold_best_epochs)
        return statistics.fmean(fold_mae)

    trial: optuna.trial.FrozenTrial
    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    while True:
        finished: int = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            or trial.user_attrs.get("diverged", False)
            for trial in study.get_trials()
        )
        if finished >= n_trials:
            break
        study.sampler = optuna.samplers.TPESampler(seed=seed + finished)
        study.optimize(objective, n_trials=1, catch=(FloatingPointError,))


def final_epochs(trial: optuna.trial.FrozenTrial) -> int:
    """Return the median of a trial's best epochs, rounded down.

    Parameters
    ----------
    trial : optuna.trial.FrozenTrial
        A completed trial from `search`, usually ``study.best_trial``.

    Returns
    -------
    int
        The number of epochs to train the final model for.
    """
    return int(statistics.median(trial.user_attrs["fold_best_epochs"]))


def fit(
    x: torch.Tensor,
    y: torch.Tensor,
    hyperparameters: Hyperparameters,
    epochs: int,
    seed: int,
) -> tuple[VNN, torch.Tensor]:
    """Train a model on all the data for a fixed number of epochs.

    Parameters
    ----------
    x, y : torch.Tensor
        Features ``(n, N)`` and targets ``(n,)``.
    hyperparameters : Hyperparameters
        The hyperparameters, usually ``suggest(study.best_trial)``.
    epochs : int
        Number of epochs, usually ``final_epochs(study.best_trial)``.
    seed : int
        Seed for the initial weights and the sample order.

    Returns
    -------
    model : VNN
        The trained model. It expects centered features: predict with
        ``predict(model, x_new - mean)``.
    mean : torch.Tensor
        The mean of ``x``, shape ``(N,)``.
    """
    mean: torch.Tensor = x.mean(dim=0)
    centered: torch.Tensor = x - mean
    model: VNN = build(centered, hyperparameters, seed)
    optimizer: torch.optim.Adam = torch.optim.Adam(
        model.parameters(), lr=hyperparameters.lr
    )
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        _train_epoch(
            model, centered, y, optimizer, hyperparameters.batch_size, generator
        )
    return model, mean
