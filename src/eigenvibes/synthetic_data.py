# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the synthetic Among Us datasets described in the README.

Each row has 12 behaviors ``x``. They are built from 12 pattern scores
``q ~ N(0, I)`` as ``x = PATTERNS @ (sqrt(EIGENVALUES) * q)``, where the
columns of `PATTERNS` are the eigenvectors of `COVARIANCE`, sorted by
eigenvalue from largest to smallest. The sus score is
``q1 + 0.5 q2 + q5 + c q1 q2``, with ``c`` from `RULES`.

Cohorts follow players over rounds ``t = 0, ..., 3``. In pattern scores, a
player's round ``t`` is ``MEAN_SCORES + t DRIFT + b + t u + e``, where only
impostors get the ``DRIFT`` term, and ``b``, ``u`` and ``e`` are independent
normal draws with variances ``1 - nu``, ``SLOPE_SD**2`` and ``nu``. ``b`` and
``u`` are fixed per player and ``e`` is new each round. Impostors' behaviors
also move ``DECOY_STEP`` per round along `DECOY_DIRECTION`, which leaves sus
unchanged.
"""

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
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
# Weight of the q1 q2 term in the sus score.
RULES: dict[str, float] = {"linear": 0.0, "interaction": 1.0}
EVALUATION_SEEDS: range = range(30)
TUNING_SEEDS: range = range(30, 40)
SPLITS: dict[str, int] = {"train": 2000, "validation": 500, "test": 2000}
ROUNDS: int = 4
MEAN_SCORES: np.ndarray = np.eye(12)[1]
DRIFT: np.ndarray = 0.5 * np.eye(12)[0] + 0.25 * np.eye(12)[4]
SLOPE_SD: float = 1 / 6
# Rounds each player keeps in the missed_rounds version, and their chances.
SCHEDULES: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3), (0, 1, 2), (0, 2, 3))
SCHEDULE_CHANCES: tuple[float, ...] = (0.5, 0.25, 0.25)
CHECKSUMS: str = "SHA256SUMS"


@dataclass(frozen=True)
class Version:
    """Settings of one cohort version.

    Attributes
    ----------
    players : int
        Number of players; even players are crewmates, odd ones impostors.
    nu : float
        Share of each pattern score's variance that is redrawn every round.
    missed_rounds : bool
        Whether players keep only the rounds of a random `SCHEDULES` entry.
    drift : bool
        Whether impostors' sus rises over the rounds.
    """

    players: int = 200
    nu: float = 0.4
    missed_rounds: bool = False
    drift: bool = True


VERSIONS: dict[str, Version] = {
    "base": Version(),
    "100_players": Version(players=100),
    "400_players": Version(players=400),
    "steady": Version(nu=0.1),
    "jumpy": Version(nu=0.8),
    "missed_rounds": Version(missed_rounds=True),
    "no_sus_change": Version(drift=False),
}


def _patterns() -> np.ndarray:
    """Return the 12 orthonormal patterns as columns, pattern 1 first.

    Patterns 2 and 5 are fixed. Pattern 1 is the all-ones vector minus its
    parts along patterns 2 and 5, so that all-ones lies in the span of
    patterns 1, 2 and 5: a VNN's output averages over the 12 features, and
    this lets it represent the linear score exactly. The other 9 patterns
    are random (seed 42) and orthogonal to patterns 1, 2 and 5.

    Returns
    -------
    numpy.ndarray
        Shape ``(12, 12)``.
    """
    ones: np.ndarray = np.ones(12)
    v2: np.ndarray = np.r_[0.6, 0.55, -0.5, -0.4, np.zeros(8)]
    v2 /= np.linalg.norm(v2)
    v5: np.ndarray = np.r_[0.0, 0.0, 0.0, 0.0, 0.6, 0.8, np.zeros(6)]
    v1: np.ndarray = ones - (ones @ v2) * v2 - (ones @ v5) * v5
    v1 /= np.linalg.norm(v1)
    chosen: np.ndarray = np.c_[v1, v2, v5]
    other: np.ndarray = np.random.default_rng(42).standard_normal((12, 9))
    other -= chosen @ (chosen.T @ other)
    r: np.ndarray
    other, r = np.linalg.qr(other)
    other *= np.sign(np.diag(r))
    patterns: np.ndarray = np.empty((12, 12))
    patterns[:, [0, 1, 4]] = chosen
    patterns[:, [2, 3, 5, 6, 7, 8, 9, 10, 11]] = other
    return patterns


_raw: np.ndarray = np.r_[3.6, 2.4, 1.5, 1.2, 1.0, 0.8 * 0.25 ** (np.arange(7) / 6)]
# Scaled to sum to 12, so the behaviors' variances average 1.
EIGENVALUES: np.ndarray = _raw * 12 / _raw.sum()
PATTERNS: np.ndarray = _patterns()
COVARIANCE: np.ndarray = (PATTERNS * EIGENVALUES) @ PATTERNS.T
# x = q @ _TO_BEHAVIORS.T and q = x @ _TO_SCORES.
_TO_BEHAVIORS: np.ndarray = PATTERNS * np.sqrt(EIGENVALUES)
_TO_SCORES: np.ndarray = PATTERNS / np.sqrt(EIGENVALUES)
# The part of route_variability orthogonal to patterns 1, 2 and 5, which sus
# ignores; impostors move 0.35 of route_variability's SD along it per round.
_ROUTE: int = FEATURES.index("route_variability")
_decoy: np.ndarray = (
    np.eye(12)[_ROUTE] - PATTERNS[:, [0, 1, 4]] @ PATTERNS[_ROUTE, [0, 1, 4]]
)
DECOY_DIRECTION: np.ndarray = _decoy / np.linalg.norm(_decoy)
DECOY_STEP: float = 0.35 * float(np.sqrt(COVARIANCE[_ROUTE, _ROUTE]))


def sus(x: np.ndarray, interaction: float) -> np.ndarray:
    """Return the sus score without noise.

    Parameters
    ----------
    x : numpy.ndarray
        Behaviors, shape ``(n, 12)``.
    interaction : float
        Weight of the q1 q2 term, a value of `RULES`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``.
    """
    q: np.ndarray = x @ _TO_SCORES
    score: np.ndarray = (
        q[:, 0] + 0.5 * q[:, 1] + q[:, 4] + interaction * q[:, 0] * q[:, 1]
    )
    return score


def label_noise_sd(interaction: float) -> float:
    """Return the SD of the noise added to sus in train and validation rows.

    It is 10% of sus's SD over independent rows, which is
    ``sqrt(2.25 + interaction**2)``.

    Parameters
    ----------
    interaction : float
        Weight of the q1 q2 term, a value of `RULES`.

    Returns
    -------
    float
    """
    return 0.1 * float(np.sqrt(2.25 + interaction**2))


def integrated_gradients(
    x: np.ndarray, interaction: float, basis: np.ndarray
) -> np.ndarray:
    """Return the exact Integrated Gradients of sus, from the all-zero player.

    Sus is ``beta @ x + x @ h @ x / 2``, so along an orthonormal direction
    ``v`` the attribution is ``(v @ x) * v @ (beta + h @ x / 2)``. The
    attributions of each row sum to its sus.

    Parameters
    ----------
    x : numpy.ndarray
        Behaviors, shape ``(n, 12)``.
    interaction : float
        Weight of the q1 q2 term, a value of `RULES`.
    basis : numpy.ndarray
        Orthonormal directions as columns, shape ``(12, k)``: the identity for
        behaviors, `PATTERNS` for patterns.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, k)``.
    """
    w: np.ndarray = _TO_SCORES
    beta: np.ndarray = w[:, 0] + 0.5 * w[:, 1] + w[:, 4]
    h: np.ndarray = interaction * (
        np.outer(w[:, 0], w[:, 1]) + np.outer(w[:, 1], w[:, 0])
    )
    attributions: np.ndarray = (x @ basis) * ((beta + x @ h / 2) @ basis)
    return attributions


def expected_contributions(
    interaction: float, drift: bool, basis: np.ndarray
) -> np.ndarray:
    """Return each direction's share of the impostor-minus-crewmate sus slope.

    This is the population value of the slope difference that a mixed model
    fits to the Integrated Gradients of a cohort in which every player has
    all rounds; with complete rounds, its estimate equals the least squares
    slope. Crewmates' expected attributions are the same every round, and the
    two groups differ only in their mean behaviors, so the value is the least
    squares slope of the attributions at the impostors' mean behaviors. It
    does not apply to the missed_rounds version.

    Parameters
    ----------
    interaction : float
        Weight of the q1 q2 term, a value of `RULES`.
    drift : bool
        Whether impostors' sus rises, as in `Version`.
    basis : numpy.ndarray
        Orthonormal directions as columns, shape ``(12, k)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(k,)``; sums to the sus slope difference.
    """
    t: np.ndarray = np.arange(ROUNDS)
    impostor: np.ndarray = (MEAN_SCORES + np.outer(t, drift * DRIFT)) @ _TO_BEHAVIORS.T
    impostor += np.outer(t * DECOY_STEP, DECOY_DIRECTION)
    centered: np.ndarray = t - t.mean()
    slope: np.ndarray = (
        centered
        @ integrated_gradients(impostor, interaction, basis)
        / (centered @ centered)
    )
    return slope


def cohorts(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Draw one cohort of each version, sharing random draws between them.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of all random draws.

    Returns
    -------
    dict[str, pandas.DataFrame]
        One frame per `VERSIONS` key, with columns ``player``, ``round``,
        ``impostor`` and `FEATURES`, sorted by player, then round.
    """
    n: int = max(version.players for version in VERSIONS.values())
    fixed: np.ndarray = rng.standard_normal((n, 12))
    slope: np.ndarray = SLOPE_SD * rng.standard_normal((n, 12))
    each_round: np.ndarray = rng.standard_normal((ROUNDS, n, 12))
    schedule: np.ndarray = rng.choice(len(SCHEDULES), size=n, p=SCHEDULE_CHANCES)
    player: np.ndarray
    t: np.ndarray
    player, t = np.divmod(np.arange(n * ROUNDS), ROUNDS)
    impostor: np.ndarray = player % 2
    scheduled: np.ndarray = np.array(
        [[r in kept for r in range(ROUNDS)] for kept in SCHEDULES]
    )
    frames: dict[str, pd.DataFrame] = {}
    name: str
    version: Version
    for name, version in VERSIONS.items():
        q: np.ndarray = (
            MEAN_SCORES
            + np.sqrt(1 - version.nu) * fixed[player]
            + t[:, None] * slope[player]
            + np.sqrt(version.nu) * each_round[t, player]
            + version.drift * np.outer(impostor * t, DRIFT)
        )
        x: np.ndarray = q @ _TO_BEHAVIORS.T + np.outer(
            impostor * t * DECOY_STEP, DECOY_DIRECTION
        )
        frame: pd.DataFrame = pd.DataFrame(x, columns=FEATURES)
        frame.insert(0, "impostor", impostor)
        frame.insert(0, "round", t)
        frame.insert(0, "player", player)
        keep: np.ndarray = (player < version.players) & (
            scheduled[schedule[player], t] | (not version.missed_rounds)
        )
        frames[name] = frame[keep].reset_index(drop=True)
    return frames


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` without its index, with Unix line endings on every OS.

    pandas writes each float with the fewest digits that read back exactly.

    Parameters
    ----------
    frame : pandas.DataFrame
        Data to write.
    path : pathlib.Path
        Destination file.
    """
    frame.to_csv(path, index=False, lineterminator="\n")


def write_dataset(root: Path, seed: int) -> None:
    """Write one evaluation or tuning dataset under both rules.

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
        split: rows_rng.standard_normal((n, 12)) @ _TO_BEHAVIORS.T
        for split, n in SPLITS.items()
    }
    noise: dict[str, np.ndarray] = {
        split: noise_rng.standard_normal(SPLITS[split])
        for split in ("train", "validation")
    }
    groups: dict[str, pd.DataFrame] = cohorts(cohort_rng)
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
                frame["sus"] += label_noise_sd(interaction) * noise[split]
            _write_csv(frame, directory / f"{split}.csv")
        if tuning:
            continue
        name: str
        cohort: pd.DataFrame
        for name, cohort in groups.items():
            _write_csv(
                cohort.assign(sus=sus(cohort[list(FEATURES)].to_numpy(), interaction)),
                directory / f"cohort_{name}.csv",
            )


def design() -> dict[str, object]:
    """Return the settings and expected answers written to ``design.json``.

    Returns
    -------
    dict[str, object]
        JSON-serializable.
    """
    expected: dict[str, dict[str, dict[str, list[float]]]] = {
        rule: {
            name: {
                "behaviors": expected_contributions(
                    interaction, version.drift, np.eye(12)
                ).tolist(),
                "patterns": expected_contributions(
                    interaction, version.drift, PATTERNS
                ).tolist(),
            }
            for name, version in VERSIONS.items()
            if not version.missed_rounds
        }
        for rule, interaction in RULES.items()
    }
    return {
        "features": list(FEATURES),
        "eigenvalues": EIGENVALUES.tolist(),
        "patterns": PATTERNS.T.tolist(),
        "pattern_scores": "q = (patterns @ x) / sqrt(eigenvalues)",
        "sus": {
            "linear": "q1 + 0.5 q2 + q5",
            "interaction": "q1 + 0.5 q2 + q5 + q1 q2",
        },
        "label_noise_sd": {
            rule: label_noise_sd(interaction) for rule, interaction in RULES.items()
        },
        "label_noise": "added to sus in train.csv and validation.csv only",
        "cohort": {
            "rounds": ROUNDS,
            "impostor": "player % 2",
            "pattern_scores": (
                "q(round t) = mean_scores + t drift (impostors only) + b + t u + e, with b, u, e"
                " independent normal: Var b = 1 - nu, sd u = slope_sd, Var e = nu; b and u are"
                " fixed per player, e is new each round"
            ),
            "mean_scores": MEAN_SCORES.tolist(),
            "drift": DRIFT.tolist(),
            "slope_sd": SLOPE_SD,
            "versions": {name: asdict(version) for name, version in VERSIONS.items()},
            "missed_rounds": {"kept_rounds": SCHEDULES, "chances": SCHEDULE_CHANCES},
        },
        "decoy": {
            "behaviors": "x(round t) += t step direction, for impostors only, in every version",
            "direction": DECOY_DIRECTION.tolist(),
            "step": DECOY_STEP,
        },
        "seeds": {
            "evaluation": list(EVALUATION_SEEDS),
            "tuning": list(TUNING_SEEDS),
            "patterns": 42,
        },
        "expected_contributions": expected,
        "expected_contributions_definition": (
            "Integrated Gradients of sus from the all-zero player, along each behavior (identity"
            " basis) or each pattern (rows of patterns); the population least squares slope over"
            " rounds 0-3 of impostors' minus crewmates' mean attributions. The values are the same"
            " for every version with the same rule and drift. missed_rounds has none, because"
            " with uneven rounds the mixed model's estimate depends on its fitted weights."
        ),
    }


def _checksums(root: Path) -> dict[str, str]:
    """Return the SHA-256 hash of every file under ``root`` except `CHECKSUMS`.

    Parameters
    ----------
    root : pathlib.Path
        The ``data`` directory.

    Returns
    -------
    dict[str, str]
        Hashes keyed by path relative to ``root``, with ``/`` separators,
        sorted by path.
    """
    paths: list[Path] = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != CHECKSUMS
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def write_datasets(root: Path) -> None:
    """Write every dataset, then check the files against `CHECKSUMS`.

    Writes `COVARIANCE`, `design`, and every evaluation and tuning dataset.
    If ``root`` has no `CHECKSUMS` file, writes one in the format of
    ``sha256sum``. Otherwise, warns about every file whose hash differs, is
    missing, or is not listed.

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
    hashes: dict[str, str] = _checksums(root)
    listing: Path = root / CHECKSUMS
    if not listing.exists():
        listing.write_text(
            "".join(f"{digest}  {path}\n" for path, digest in hashes.items()),
            encoding="utf-8",
            newline="\n",
        )
        return
    expected: dict[str, str] = {
        path: digest
        for digest, path in (
            line.split("  ", 1)
            for line in listing.read_text(encoding="utf-8").splitlines()
        )
    }
    different: list[str] = sorted(
        path
        for path in hashes.keys() | expected.keys()
        if hashes.get(path) != expected.get(path)
    )
    if different:
        warnings.warn(
            f"{len(different)} of {len(expected)} files in {root} differ from {CHECKSUMS}. This is"
            " not an error and can't be controlled from here: the math library picks code for this"
            " machine's CPU, which can change the last digit of some numbers. The data follow the"
            " same design, so conclusions should not change, but trained models and reported"
            " numbers may differ slightly from ours. First differences: "
            + ", ".join(different[:5]),
            stacklevel=2,
        )
