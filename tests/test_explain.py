# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Integrated Gradients and the group effect."""

import numpy as np
import torch

from eigenvibes import explain
from eigenvibes import synthetic_data as sd


class _Sus(torch.nn.Module):
    """The interaction formula's sus score as a PyTorch model."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sus for each row of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Behaviors, shape ``(n, 12)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n,)``.
        """
        beta: torch.Tensor = torch.tensor(sd.BETA, dtype=x.dtype)
        return x @ beta + x[:, 0] * x[:, 1]


def test_integrated_gradients_match_the_exact_ones() -> None:
    """Numerical IG of the true formula equals the exact IG.

    The gradient of a quadratic is linear along the path, so the midpoint rule
    is exact even with few steps.
    """
    x: np.ndarray = np.random.default_rng(42).standard_normal((30, 12))
    numerical: torch.Tensor = explain.integrated_gradients(
        _Sus(), torch.tensor(x), steps=4
    )
    np.testing.assert_allclose(
        numerical.numpy(), sd.integrated_gradients(x, 1.0), atol=1e-12
    )


def test_group_effect_recovers_the_slope_difference() -> None:
    """With exact group trends, the coefficient is the impostors' extra slope."""
    rounds: np.ndarray = np.tile(np.arange(4), 6)
    impostor: np.ndarray = np.repeat(np.arange(6) % 2, 4)
    values: np.ndarray = np.column_stack(
        [1 + 0.3 * rounds + impostor * (2 + 0.7 * rounds), 0.1 * rounds]
    )
    np.testing.assert_allclose(
        explain.group_effect(values, rounds, impostor), [0.7, 0.0], atol=1e-12
    )
