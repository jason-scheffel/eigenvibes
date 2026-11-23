# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the synthetic dataset generator."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eigenvibes import explain
from eigenvibes import synthetic_data as sd

FEATURES: list[str] = list(sd.FEATURES)


def test_covariance_is_a_correlation_matrix() -> None:
    """Every behavior has variance 1, and the matrix is positive definite."""
    np.testing.assert_allclose(np.diag(sd.COVARIANCE), 1, atol=1e-12)
    assert np.linalg.eigvalsh(sd.COVARIANCE).min() > 0


@pytest.mark.parametrize("interaction", sd.RULES.values())
def test_integrated_gradients_sum_to_sus(interaction: float) -> None:
    """Each row's attributions add up to its sus, and sus is 0 at 0.

    Parameters
    ----------
    interaction : float
        Weight of the interaction term.
    """
    x: np.ndarray = np.random.default_rng(42).standard_normal((50, 12))
    np.testing.assert_allclose(
        sd.integrated_gradients(x, interaction).sum(axis=1),
        sd.sus(x, interaction),
        atol=1e-12,
    )
    assert sd.sus(np.zeros((1, 12)), interaction)[0] == 0


def test_true_contributions_match_the_formula() -> None:
    """The true shares are the hand-computed ones, and the decoy gets none.

    Linear: task_inactivity 2 * 0.5 and task_abandonment 0.5 * 0.25. The
    interaction adds 0.5 * 1 (vent_lingering's cohort mean) per round, split
    evenly between task_inactivity and vent_lingering.
    """
    linear: np.ndarray = np.zeros(12)
    linear[[0, 4]] = 1.0, 0.125
    interaction: np.ndarray = linear.copy()
    interaction[[0, 1]] += 0.25
    np.testing.assert_allclose(sd.true_contributions(0.0), linear, atol=1e-12)
    np.testing.assert_allclose(sd.true_contributions(1.0), interaction, atol=1e-12)


def test_cohorts_recover_the_true_contributions_on_average() -> None:
    """The group effect of the exact attributions averages to the true shares.

    Over 200 cohorts, the mean estimate must be within 0.03 of the true value
    for every behavior, about 4 standard errors.
    """
    estimates: list[np.ndarray] = []
    seed: int
    for seed in range(200):
        frame: pd.DataFrame = sd.cohort(np.random.default_rng(seed))
        estimates.append(
            explain.group_effect(
                sd.integrated_gradients(frame[FEATURES].to_numpy(), 1.0),
                frame["round"].to_numpy(),
                frame["impostor"].to_numpy(),
            )
        )
    np.testing.assert_allclose(
        np.mean(estimates, axis=0), sd.true_contributions(1.0), atol=0.03
    )


def test_write_dataset(tmp_path: Path) -> None:
    """Check the files of one evaluation and one tuning dataset.

    Both formulas share behaviors, noise is only in train and validation, the
    cohort has every player in every round with odd players as impostors, and
    tuning datasets have no test split or cohort.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory from pytest.
    """
    sd.write_dataset(tmp_path, 0)
    sd.write_dataset(tmp_path, 30)
    frames: dict[str, dict[str, pd.DataFrame]] = {
        rule: {
            name: pd.read_csv(
                tmp_path / rule / "evaluation_00" / f"{name}.csv",
                float_precision="round_trip",
            )
            for name in ("train", "validation", "test", "cohort")
        }
        for rule in sd.RULES
    }
    name: str
    for name in ("train", "validation", "test", "cohort"):
        pd.testing.assert_frame_equal(
            frames["linear"][name].drop(columns="sus"),
            frames["interaction"][name].drop(columns="sus"),
        )
    rule: str
    interaction: float
    for rule, interaction in sd.RULES.items():
        for name in ("test", "cohort"):
            frame: pd.DataFrame = frames[rule][name]
            np.testing.assert_allclose(
                frame["sus"], sd.sus(frame[FEATURES].to_numpy(), interaction)
            )
        train: pd.DataFrame = frames[rule]["train"]
        noise: np.ndarray = train["sus"] - sd.sus(
            train[FEATURES].to_numpy(), interaction
        )
        assert abs(noise.std() / sd.noise_sd(interaction) - 1) < 0.1
    cohort: pd.DataFrame = frames["linear"]["cohort"]
    assert len(cohort) == sd.PLAYERS * sd.ROUNDS
    assert (cohort["round"].to_numpy().reshape(-1, sd.ROUNDS) == range(sd.ROUNDS)).all()
    assert (cohort["impostor"] == cohort["player"] % 2).all()
    assert sorted(p.name for p in (tmp_path / "linear" / "tuning_30").iterdir()) == [
        "train.csv",
        "validation.csv",
    ]
