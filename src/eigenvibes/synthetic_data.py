# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the synthetic Among Us datasets described in the README.

Each row is one player with 12 behaviors ``x ~ N(0, COVARIANCE)`` and a sus
score ``BETA @ x + c * x[0] * x[1]``, where ``c`` is 0 for the linear formula
and 1 for the interaction formula (`RULES`). Train and validation sus scores
also get noise with an SD of 10% of sus's SD; test and cohort sus is exact.

A cohort follows `PLAYERS` players for `ROUNDS` rounds, and odd players are
impostors. Player ``p`` in round ``t`` has the behaviors
``MEAN + b_p + e_pt + impostor_p * t * DRIFT``. The part ``b_p`` is the same in
every round and ``e_pt`` is new each round; both are normal, with covariances
``PERSISTENT * COVARIANCE`` and ``(1 - PERSISTENT) * COVARIANCE``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES: tuple[str, ...] = (
    "task_inactivity",
    "vent_lingering",
    "time_alone",
    "quiet_room_preference",
    "task_abandonment",
    "following_players",
    "distance_walked",
    "room_changes",
    "walking_speed",
    "hallway_time",
    "route_variability",
    "task_station_proximity",
)
# Weight of the x[0] * x[1] term in the sus score.
RULES: dict[str, float] = {"linear": 0.0, "interaction": 1.0}
EVALUATION_SEEDS: range = range(30)
TUNING_SEEDS: range = range(30, 40)
SPLITS: dict[str, int] = {"train": 2000, "validation": 500, "test": 2000}
PLAYERS: int = 200
ROUNDS: int = 4
PERSISTENT: float = 0.6


def _vector(values: dict[str, float]) -> np.ndarray:
    """Return a length-12 vector with the given values by behavior, else 0."""
    vector: np.ndarray = np.zeros(len(FEATURES))
    name: str
    value: float
    for name, value in values.items():
        vector[FEATURES.index(name)] = value
    return vector


BETA: np.ndarray = _vector(
    {"task_inactivity": 2.0, "vent_lingering": 1.0, "task_abandonment": 0.5}
)
# The cohort's average player lingers near vents more than the training data's.
MEAN: np.ndarray = _vector({"vent_lingering": 1.0})
# Impostors' change per round. route_variability is a decoy: sus ignores it.
DRIFT: np.ndarray = _vector(
    {"task_inactivity": 0.5, "task_abandonment": 0.25, "route_variability": 0.35}
)
# How much each behavior follows three hidden play styles: slacking, sneaking
# and moving. The covariance adds independent noise so every variance is 1.
_STYLES: dict[str, tuple[float, float, float]] = {
    "task_inactivity": (0.8, 0.0, 0.0),
    "vent_lingering": (0.0, 0.7, 0.0),
    "time_alone": (0.3, 0.6, 0.0),
    "quiet_room_preference": (0.0, 0.6, 0.0),
    "task_abandonment": (0.6, 0.0, 0.0),
    "following_players": (0.0, 0.5, 0.3),
    "distance_walked": (0.0, 0.0, 0.8),
    "room_changes": (0.0, 0.0, 0.7),
    "walking_speed": (0.0, 0.0, 0.6),
    "hallway_time": (0.0, 0.3, 0.5),
    "route_variability": (0.5, 0.0, 0.3),
    "task_station_proximity": (-0.6, 0.0, 0.0),
}
_loadings: np.ndarray = np.array([_STYLES[name] for name in FEATURES])
COVARIANCE: np.ndarray = _loadings @ _loadings.T + np.diag(
    1 - (_loadings**2).sum(axis=1)
)
_ROOT: np.ndarray = np.linalg.cholesky(COVARIANCE)


def sus(x: np.ndarray, interaction: float) -> np.ndarray:
    """Return the sus score without noise.

    Parameters
    ----------
    x : numpy.ndarray
        Behaviors, shape ``(n, 12)``.
    interaction : float
        Weight of the ``x[0] * x[1]`` term, a value of `RULES`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``.
    """
    score: np.ndarray = x @ BETA + interaction * x[:, 0] * x[:, 1]
    return score


def noise_sd(interaction: float) -> float:
    """Return the SD of the noise in train and validation sus: 10% of sus's SD.

    For ``x ~ N(0, C)``, ``BETA @ x`` and ``x[0] * x[1]`` are uncorrelated, and
    ``x[0] * x[1]`` has variance ``C[0, 0] C[1, 1] + C[0, 1]**2``.

    Parameters
    ----------
    interaction : float
        Weight of the ``x[0] * x[1]`` term, a value of `RULES`.

    Returns
    -------
    float
    """
    c: np.ndarray = COVARIANCE
    variance: float = float(BETA @ c @ BETA) + interaction**2 * (
        c[0, 0] * c[1, 1] + c[0, 1] ** 2
    )
    return 0.1 * float(np.sqrt(variance))


def integrated_gradients(x: np.ndarray, interaction: float) -> np.ndarray:
    """Return the exact Integrated Gradients of sus, from the all-zero player.

    Along the straight path from 0 to ``x``, behavior ``j`` gets
    ``BETA[j] * x[j]``, and the two behaviors in ``x[0] * x[1]`` get half of
    that term each. Each row's attributions sum to its sus.

    Parameters
    ----------
    x : numpy.ndarray
        Behaviors, shape ``(n, 12)``.
    interaction : float
        Weight of the ``x[0] * x[1]`` term, a value of `RULES`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 12)``.
    """
    attributions: np.ndarray = x * BETA
    half: np.ndarray = interaction * x[:, 0] * x[:, 1] / 2
    attributions[:, 0] += half
    attributions[:, 1] += half
    return attributions


def true_contributions(interaction: float) -> np.ndarray:
    """Return each behavior's true share of the impostors' extra sus slope.

    This is the population value of the impostor-by-round coefficient that
    `eigenvibes.explain.group_effect` estimates from a cohort's attributions.
    In each round both groups' behaviors have the same covariance, so the
    groups' mean attributions differ by the attributions' difference at the
    groups' mean behaviors, ``MEAN + t DRIFT`` and ``MEAN``. The value is the
    least squares slope of that difference over the rounds.

    Parameters
    ----------
    interaction : float
        Weight of the ``x[0] * x[1]`` term, a value of `RULES`.

    Returns
    -------
    numpy.ndarray
        Shape ``(12,)``; sums to the impostors' extra sus slope.
    """
    t: np.ndarray = np.arange(ROUNDS)
    gap: np.ndarray = integrated_gradients(
        MEAN + np.outer(t, DRIFT), interaction
    ) - integrated_gradients(MEAN[None, :], interaction)
    centered: np.ndarray = t - t.mean()
    slope: np.ndarray = centered @ gap / (centered @ centered)
    return slope


def cohort(rng: np.random.Generator) -> pd.DataFrame:
    """Draw one cohort, without sus.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of all random draws.

    Returns
    -------
    pandas.DataFrame
        Columns ``player``, ``round``, ``impostor`` and `FEATURES`, one row
        per player and round, sorted by player, then round.
    """
    player: np.ndarray
    t: np.ndarray
    player, t = np.divmod(np.arange(PLAYERS * ROUNDS), ROUNDS)
    impostor: np.ndarray = player % 2
    fixed: np.ndarray = rng.standard_normal((PLAYERS, 12)) @ _ROOT.T
    each_round: np.ndarray = rng.standard_normal((PLAYERS * ROUNDS, 12)) @ _ROOT.T
    x: np.ndarray = (
        MEAN
        + np.sqrt(PERSISTENT) * fixed[player]
        + np.sqrt(1 - PERSISTENT) * each_round
        + np.outer(impostor * t, DRIFT)
    )
    frame: pd.DataFrame = pd.DataFrame(x, columns=FEATURES)
    frame.insert(0, "impostor", impostor)
    frame.insert(0, "round", t)
    frame.insert(0, "player", player)
    return frame


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` without its index, with Unix line endings on every OS."""
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def write_dataset(root: Path, seed: int) -> None:
    """Write one evaluation or tuning dataset under both formulas.

    Both formulas get the same behaviors and noise draws; only sus differs.
    Tuning datasets have only train and validation splits.

    Parameters
    ----------
    root : pathlib.Path
        The ``data`` directory.
    seed : int
        A value of `EVALUATION_SEEDS` or `TUNING_SEEDS`.
    """
    rows_rng: np.random.Generator
    noise_rng: np.random.Generator
    cohort_rng: np.random.Generator
    rows_rng, noise_rng, cohort_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(3)
    )
    x: dict[str, np.ndarray] = {
        split: rows_rng.standard_normal((n, 12)) @ _ROOT.T
        for split, n in SPLITS.items()
    }
    noise: dict[str, np.ndarray] = {
        split: noise_rng.standard_normal(SPLITS[split])
        for split in ("train", "validation")
    }
    players: pd.DataFrame = cohort(cohort_rng)
    tuning: bool = seed in TUNING_SEEDS
    rule: str
    interaction: float
    for rule, interaction in RULES.items():
        directory: Path = (
            root / rule / f"{'tuning' if tuning else 'evaluation'}_{seed:02d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        split: str
        for split in ("train", "validation") if tuning else SPLITS:
            frame: pd.DataFrame = pd.DataFrame(x[split], columns=FEATURES)
            frame["sus"] = sus(x[split], interaction)
            if split in noise:
                frame["sus"] += noise_sd(interaction) * noise[split]
            _write_csv(frame, directory / f"{split}.csv")
        if not tuning:
            _write_csv(
                players.assign(
                    sus=sus(players[list(FEATURES)].to_numpy(), interaction)
                ),
                directory / "cohort.csv",
            )


def design() -> dict[str, object]:
    """Return the settings and true answers written to ``design.json``.

    Returns
    -------
    dict[str, object]
        JSON-serializable.
    """
    return {
        "features": list(FEATURES),
        "sus": {
            "linear": "2 task_inactivity + vent_lingering + 0.5 task_abandonment",
            "interaction": "the linear formula + task_inactivity * vent_lingering",
        },
        "beta": BETA.tolist(),
        "noise_sd": {rule: noise_sd(c) for rule, c in RULES.items()},
        "cohort": {
            "players": PLAYERS,
            "rounds": ROUNDS,
            "impostor": "player % 2",
            "mean": MEAN.tolist(),
            "drift_per_round": DRIFT.tolist(),
            "persistent_share": PERSISTENT,
        },
        "true_contributions": {
            rule: true_contributions(c).tolist() for rule, c in RULES.items()
        },
        "seeds": {
            "evaluation": list(EVALUATION_SEEDS),
            "tuning": list(TUNING_SEEDS),
        },
    }


def write_datasets(root: Path) -> None:
    """Write `COVARIANCE`, `design`, and every evaluation and tuning dataset.

    Parameters
    ----------
    root : pathlib.Path
        The ``data`` directory; created if missing.
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(pd.DataFrame(COVARIANCE, columns=FEATURES), root / "covariance.csv")
    (root / "design.json").write_text(
        json.dumps(design(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    seed: int
    for seed in (*EVALUATION_SEEDS, *TUNING_SEEDS):
        write_dataset(root, seed)
