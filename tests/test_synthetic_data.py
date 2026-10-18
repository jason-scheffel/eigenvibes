# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the synthetic dataset generator."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eigenvibes import synthetic_data as sd

FEATURES: list[str] = list(sd.FEATURES)


def test_patterns_and_decoy() -> None:
    """The patterns are orthonormal, and the decoy direction leaves sus unchanged."""
    np.testing.assert_allclose(sd.PATTERNS.T @ sd.PATTERNS, np.eye(12), atol=1e-12)
    chosen: np.ndarray = sd.PATTERNS[:, [0, 1, 4]]
    ones: np.ndarray = np.ones(12)
    np.testing.assert_allclose(chosen @ (chosen.T @ ones), ones, atol=1e-12)
    x: np.ndarray = np.random.default_rng(42).standard_normal((50, 12))
    interaction: float
    for interaction in sd.RULES.values():
        np.testing.assert_allclose(
            sd.sus(x + 3 * sd.DECOY_DIRECTION, interaction),
            sd.sus(x, interaction),
            atol=1e-12,
        )


@pytest.mark.parametrize("interaction", sd.RULES.values())
def test_integrated_gradients_sum_to_sus(interaction: float) -> None:
    """Attributions add up to sus, and only patterns 1, 2 and 5 get any.

    Parameters
    ----------
    interaction : float
        Weight of the q1 q2 term.
    """
    x: np.ndarray = np.random.default_rng(42).standard_normal((50, 12))
    basis: np.ndarray
    for basis in (np.eye(12), sd.PATTERNS):
        np.testing.assert_allclose(
            sd.integrated_gradients(x, interaction, basis).sum(axis=1),
            sd.sus(x, interaction),
            atol=1e-12,
        )
    unused: np.ndarray = sd.integrated_gradients(x, interaction, sd.PATTERNS)[
        :, [2, 3, 5, 6, 7, 8, 9, 10, 11]
    ]
    np.testing.assert_allclose(unused, 0, atol=1e-12)


@pytest.mark.parametrize("version", ["base", "no_sus_change"])
def test_expected_contributions_match_simulated_cohorts(version: str) -> None:
    """The closed-form answers match the average over simulated cohorts.

    Parameters
    ----------
    version : str
        Cohort version to simulate.
    """
    rounds: np.ndarray = np.arange(sd.ROUNDS) - (sd.ROUNDS - 1) / 2
    cohorts: list[pd.DataFrame] = [
        sd.cohorts(np.random.default_rng(seed))[version] for seed in range(200)
    ]
    basis: np.ndarray
    for basis in (np.eye(12), sd.PATTERNS):
        slopes: list[np.ndarray] = []
        cohort: pd.DataFrame
        for cohort in cohorts:
            ig: pd.DataFrame = pd.DataFrame(
                sd.integrated_gradients(cohort[FEATURES].to_numpy(), 1.0, basis)
            )
            means: pd.DataFrame = ig.groupby(
                [cohort["round"], cohort["impostor"]]
            ).mean()
            difference: np.ndarray = (
                means.xs(1, level=1) - means.xs(0, level=1)
            ).to_numpy()
            slopes.append(rounds @ difference / (rounds @ rounds))
        expected: np.ndarray = sd.expected_contributions(
            1.0, sd.VERSIONS[version].drift, basis
        )
        np.testing.assert_allclose(np.mean(slopes, axis=0), expected, atol=0.008)


def test_versions_share_draws() -> None:
    """Smaller and larger cohorts reuse the base players, and missed rounds only drop rows."""
    cohorts: dict[str, pd.DataFrame] = sd.cohorts(np.random.default_rng(42))
    base: pd.DataFrame = cohorts["base"]
    pd.testing.assert_frame_equal(cohorts["100_players"], base[base["player"] < 100])
    pd.testing.assert_frame_equal(cohorts["400_players"].iloc[: len(base)], base)
    missed: pd.DataFrame = cohorts["missed_rounds"]
    pd.testing.assert_frame_equal(
        missed, base.merge(missed[["player", "round"]]), check_like=True
    )
    assert (missed.groupby("player")["round"].min() == 0).all()
    assert (missed.groupby("player").size() >= 3).all()
    assert (base["impostor"] == base["player"] % 2).all()


def test_write_dataset_is_exact_and_repeatable(tmp_path: Path) -> None:
    """Files repeat byte for byte, read back exactly, and share draws across rules.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory from pytest.
    """
    sd.write_dataset(tmp_path / "first", 0)
    sd.write_dataset(tmp_path / "second", 0)
    sd.write_dataset(tmp_path / "other", 1)
    files: list[Path] = sorted(
        path.relative_to(tmp_path / "first")
        for path in (tmp_path / "first").rglob("*.csv")
    )
    assert len(files) == 2 * (3 + len(sd.VERSIONS))
    assert all(
        (tmp_path / "first" / file).read_bytes()
        == (tmp_path / "second" / file).read_bytes()
        for file in files
    )
    train: str = "linear/evaluation_00/train.csv"
    assert (tmp_path / "first" / train).read_bytes() != (
        tmp_path / "other" / train.replace("00", "01")
    ).read_bytes()
    split: str
    for split in ("train", "validation", "test", "cohort_base"):
        frames: dict[str, pd.DataFrame] = {
            rule: pd.read_csv(
                tmp_path / "first" / rule / "evaluation_00" / f"{split}.csv",
                float_precision="round_trip",
            )
            for rule in sd.RULES
        }
        pd.testing.assert_frame_equal(
            frames["linear"].drop(columns="sus"),
            frames["interaction"].drop(columns="sus"),
        )
        noise: dict[str, np.ndarray] = {
            rule: (frame["sus"] - sd.sus(frame[FEATURES].to_numpy(), sd.RULES[rule]))
            / sd.label_noise_sd(sd.RULES[rule])
            for rule, frame in frames.items()
        }
        if split in ("train", "validation"):
            np.testing.assert_allclose(noise["linear"], noise["interaction"], atol=1e-9)
            assert noise["linear"].std() > 0.5
        else:
            np.testing.assert_array_equal(noise["linear"], 0)
            np.testing.assert_array_equal(noise["interaction"], 0)


def test_checksums_warn_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first run writes checksums; a later run only warns about differences.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory from pytest.
    monkeypatch : pytest.MonkeyPatch
        Limits the run to one evaluation and one tuning dataset.
    """
    monkeypatch.setattr(sd, "EVALUATION_SEEDS", range(1))
    monkeypatch.setattr(sd, "TUNING_SEEDS", range(30, 31))
    sd.write_datasets(tmp_path)
    listing: Path = tmp_path / sd.CHECKSUMS
    lines: list[str] = listing.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 + 2 * (3 + len(sd.VERSIONS)) + 2 * 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sd.write_datasets(tmp_path)
    listing.write_text(
        "0" * 64 + lines[0][64:] + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8"
    )
    with pytest.warns(UserWarning, match="1 of"):
        sd.write_datasets(tmp_path)
