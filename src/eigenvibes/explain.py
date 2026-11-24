# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Explain trained models with Integrated Gradients, and split a group effect.

`integrated_gradients` explains each prediction of a network relative to the
all-zero player. `group_effect` fits, for each behavior's attributions, the
regression ``value ~ 1 + round + impostor + impostor * round`` and returns the
impostor-by-round coefficient: how much faster that behavior's attribution
rises for impostors than for crewmates. The coefficient is linear in the
values, so the behaviors' coefficients add up to the same coefficient fitted
to the predictions.
"""

import numpy as np
import torch

from eigenvibes import training

STEPS: int = 512


def integrated_gradients(
    model: torch.nn.Module, x: torch.Tensor, steps: int = STEPS
) -> torch.Tensor:
    """Return the Integrated Gradients of each prediction, from the input 0.

    The path integral uses the midpoint rule with ``steps`` points. The
    gradients are computed in chunks of `training.chunk_size` rows, so the
    result does not depend on how many rows ``x`` holds.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model, on the device of ``x``.
    x : torch.Tensor
        Inputs, float64, shape ``(n, N)``.
    steps : int
        Number of points on the path.

    Returns
    -------
    torch.Tensor
        Attributions, shape ``(n, N)``. Each row sums to approximately the
        model's prediction for it minus its prediction for 0.
    """
    chunk: int = training.chunk_size(model)
    total: torch.Tensor = torch.zeros_like(x)
    step: int
    for step in range(steps):
        alpha: float = (step + 0.5) / steps
        start: int
        for start in range(0, len(x), chunk):
            point: torch.Tensor = (alpha * x[start : start + chunk]).requires_grad_()
            gradient: torch.Tensor = torch.autograd.grad(model(point).sum(), point)[0]
            total[start : start + chunk] += gradient
    return x * total / steps


def group_effect(
    values: np.ndarray, rounds: np.ndarray, impostor: np.ndarray
) -> np.ndarray:
    """Return the impostor-by-round coefficient of a least squares fit per column.

    With every player observed in every round, this equals the fixed-effect
    estimate of a linear mixed model with a random intercept and slope per
    player.

    Parameters
    ----------
    values : numpy.ndarray
        One row per player and round, shape ``(n, k)``.
    rounds : numpy.ndarray
        Each row's round, shape ``(n,)``.
    impostor : numpy.ndarray
        1 for impostors' rows and 0 for crewmates', shape ``(n,)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(k,)``.
    """
    design: np.ndarray = np.column_stack(
        [np.ones(len(rounds)), rounds, impostor, impostor * rounds]
    )
    coefficients: np.ndarray = np.linalg.lstsq(design, values, rcond=None)[0]
    effect: np.ndarray = coefficients[3]
    return effect
