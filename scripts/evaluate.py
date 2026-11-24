# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Make every number and figure for the report from the trained models.

For each formula and evaluation dataset, the script compares these models'
Integrated Gradients (IG) on the dataset's cohort:

- ``oracle``: the exact IG of the true sus formula;
- ``ridge``: ridge regression fitted to the training split, with an
  unpenalized intercept and the penalty from `LAMBDAS` with the lowest
  validation MSE; its IG is ``beta_j * x_j``;
- ``mlp`` and ``vnn``: the networks in ``results/final``, if present.

A network's IG is read from ``results/final/<formula>/<model>/<dataset>_ig.csv``,
one row per row of the cohort, in the same order, with the 12 behaviors'
attributions and the column ``prediction``. Its test RMSE and R² are read
from ``<dataset>.json`` next to it.

Each model's contributions are `eigenvibes.explain.group_effect` of its IG.
A model's error is measured against the oracle on the same cohort, so it holds
only the model's error, not the cohort's sampling error. The script writes,
under ``results/``:

- ``evaluation/contributions.csv``: every contribution, plus the population
  truth from `eigenvibes.synthetic_data.true_contributions` as model
  ``truth``;
- ``evaluation/summary.csv`` and ``evaluation/summary.tex``: per formula and
  model, means over the datasets, with standard errors;
- ``figures/contributions.{pdf,png}`` and ``figures/trajectories.{pdf,png}``.

Run it with ``uv run python scripts/evaluate.py``.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from eigenvibes import explain, synthetic_data, training
from eigenvibes.synthetic_data import EVALUATION_SEEDS, FEATURES, RULES

RESULTS: Path = Path(__file__).resolve().parent.parent / "results"
DATASETS: tuple[str, ...] = tuple(f"evaluation_{seed:02d}" for seed in EVALUATION_SEEDS)
LAMBDAS: np.ndarray = np.logspace(-6, 3, 50)
DECOY: str = "route_variability"
LABELS: tuple[str, ...] = (
    "inactivity",
    "vent",
    "alone",
    "quiet room",
    "abandon",
    "following",
    "distance",
    "room changes",
    "speed",
    "hallway",
    "route var.",
    "station prox.",
)
COLORS: dict[str, str] = {"ridge": "#2a78d6", "mlp": "#eb6834", "vnn": "#1baf7a"}
NAMES: dict[str, str] = {"ridge": "Ridge", "mlp": "MLP", "vnn": "VNN"}


def ridge(
    x: np.ndarray, y: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray
) -> tuple[float, np.ndarray]:
    """Fit ridge regression with the penalty in `LAMBDAS` that fits validation best.

    The intercept is not penalized: the fit centers ``x`` and ``y`` on their
    training means.

    Parameters
    ----------
    x : numpy.ndarray
        Training behaviors, shape ``(n, k)``.
    y : numpy.ndarray
        Training sus, shape ``(n,)``.
    x_valid : numpy.ndarray
        Validation behaviors, shape ``(m, k)``.
    y_valid : numpy.ndarray
        Validation sus, shape ``(m,)``.

    Returns
    -------
    intercept : float
    beta : numpy.ndarray
        Shape ``(k,)``.
    """
    x_mean: np.ndarray = x.mean(axis=0)
    y_mean: float = float(y.mean())
    centered: np.ndarray = x - x_mean
    gram: np.ndarray = centered.T @ centered
    moment: np.ndarray = centered.T @ (y - y_mean)
    best: tuple[float, float, np.ndarray] = (np.inf, 0.0, np.zeros(x.shape[1]))
    penalty: float
    for penalty in LAMBDAS:
        beta: np.ndarray = np.linalg.solve(gram + penalty * np.eye(len(gram)), moment)
        intercept: float = y_mean - float(x_mean @ beta)
        mse: float = float(np.mean((intercept + x_valid @ beta - y_valid) ** 2))
        if mse < best[0]:
            best = (mse, intercept, beta)
    return best[1], best[2]


def _read(path: Path) -> pd.DataFrame:
    """Read a CSV file with full precision."""
    return pd.read_csv(path, float_precision="round_trip")


def evaluate(
    root: Path = training.DATA, results: Path = RESULTS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every model on every dataset, and write ``contributions.csv``.

    Parameters
    ----------
    root : pathlib.Path
        The data folder.
    results : pathlib.Path
        The results folder, which holds ``final``.

    Returns
    -------
    contributions : pandas.DataFrame
        Columns ``formula``, ``model``, ``dataset``, ``behavior``, ``value``.
    scores : pandas.DataFrame
        One row per formula, model and dataset, with ``test_rmse``,
        ``test_r2``, ``behavior_mae`` (mean absolute error of the 12
        contributions), ``decoy`` (the decoy's contribution),
        ``total_bias`` (the error of their sum), and ``ig_rmse`` (RMSE of
        the IG over the cohort's rows and behaviors), all against the oracle.
    trajectories : pandas.DataFrame
        Mean sus (model ``oracle``) and mean prediction of each other model,
        by formula, dataset, impostor and round.
    """
    contributions: list[pd.DataFrame] = []
    scores: list[dict[str, object]] = []
    trajectories: list[pd.DataFrame] = []
    formula: str
    interaction: float
    for formula, interaction in RULES.items():
        networks: list[str] = [
            model
            for model in training.MODELS
            if (results / "final" / formula / model).is_dir()
        ]
        contributions.append(
            pd.DataFrame(
                {
                    "formula": formula,
                    "model": "truth",
                    "dataset": "population",
                    "behavior": FEATURES,
                    "value": synthetic_data.true_contributions(interaction),
                }
            )
        )
        dataset: str
        for dataset in DATASETS:
            folder: Path = root / formula / dataset
            split: dict[str, pd.DataFrame] = {
                name: _read(folder / f"{name}.csv")
                for name in ("train", "validation", "test", "cohort")
            }
            x: dict[str, np.ndarray] = {
                name: frame[list(FEATURES)].to_numpy() for name, frame in split.items()
            }
            cohort: pd.DataFrame = split["cohort"]
            rounds: np.ndarray = cohort["round"].to_numpy()
            impostor: np.ndarray = cohort["impostor"].to_numpy()
            intercept: float
            beta: np.ndarray
            intercept, beta = ridge(
                x["train"],
                split["train"]["sus"].to_numpy(),
                x["validation"],
                split["validation"]["sus"].to_numpy(),
            )
            y_test: np.ndarray = split["test"]["sus"].to_numpy()
            error: np.ndarray = intercept + x["test"] @ beta - y_test
            spread: float = float(np.sum((y_test - y_test.mean()) ** 2))
            ig: dict[str, np.ndarray] = {
                "oracle": synthetic_data.integrated_gradients(x["cohort"], interaction),
                "ridge": x["cohort"] * beta,
            }
            fit: dict[str, tuple[float, float]] = {
                "oracle": (0.0, 1.0),
                "ridge": (
                    float(np.sqrt(np.mean(error**2))),
                    1 - float(np.sum(error**2)) / spread,
                ),
            }
            prediction: dict[str, np.ndarray] = {
                "oracle": cohort["sus"].to_numpy(),
                "ridge": intercept + x["cohort"] @ beta,
            }
            model: str
            for model in networks:
                path: Path = results / "final" / formula / model / dataset
                record: dict[str, Any] = json.loads(
                    path.with_suffix(".json").read_text(encoding="utf-8")
                )
                if record["diverged"]:
                    print(f"{formula} {model} {dataset} diverged; left out")
                    continue
                frame: pd.DataFrame = _read(path.with_name(f"{dataset}_ig.csv"))
                ig[model] = frame[list(FEATURES)].to_numpy()
                fit[model] = (record["test_rmse"], record["test_r2"])
                prediction[model] = frame["prediction"].to_numpy()
            oracle: np.ndarray = explain.group_effect(ig["oracle"], rounds, impostor)
            values: np.ndarray
            for model, values in ig.items():
                effect: np.ndarray = explain.group_effect(values, rounds, impostor)
                contributions.append(
                    pd.DataFrame(
                        {
                            "formula": formula,
                            "model": model,
                            "dataset": dataset,
                            "behavior": FEATURES,
                            "value": effect,
                        }
                    )
                )
                scores.append(
                    {
                        "formula": formula,
                        "model": model,
                        "dataset": dataset,
                        "test_rmse": fit[model][0],
                        "test_r2": fit[model][1],
                        "behavior_mae": float(np.mean(np.abs(effect - oracle))),
                        "decoy": float(effect[FEATURES.index(DECOY)]),
                        "total_bias": float(effect.sum() - oracle.sum()),
                        "ig_rmse": float(
                            np.sqrt(np.mean((values - ig["oracle"]) ** 2))
                        ),
                    }
                )
            means: pd.DataFrame = (
                pd.DataFrame(prediction)
                .assign(impostor=impostor, round=rounds)
                .groupby(["impostor", "round"])
                .mean()
                .reset_index()
                .melt(["impostor", "round"], var_name="model", value_name="sus")
            )
            trajectories.append(means.assign(formula=formula, dataset=dataset))
    table: pd.DataFrame = pd.concat(contributions, ignore_index=True)
    (results / "evaluation").mkdir(parents=True, exist_ok=True)
    table.to_csv(
        results / "evaluation" / "contributions.csv", index=False, lineterminator="\n"
    )
    return table, pd.DataFrame(scores), pd.concat(trajectories, ignore_index=True)


def summarize(scores: pd.DataFrame, results: Path = RESULTS) -> pd.DataFrame:
    """Average the scores over datasets, and write ``summary.csv`` and ``.tex``.

    Parameters
    ----------
    scores : pandas.DataFrame
        The scores from `evaluate`.
    results : pathlib.Path
        The results folder.

    Returns
    -------
    pandas.DataFrame
        One row per formula and model: the mean of each score and, as
        ``<score>_se``, the standard error of the mean for the behavior,
        decoy and total errors.
    """
    keys: list[str] = ["formula", "model"]
    errors: list[str] = ["behavior_mae", "decoy", "total_bias"]
    summary: pd.DataFrame = (
        scores.groupby(keys, sort=False)[["test_rmse", "test_r2", *errors, "ig_rmse"]]
        .mean()
        .join(scores.groupby(keys, sort=False)[errors].sem().add_suffix("_se"))
        .join(scores.groupby(keys, sort=False).size().rename("datasets"))
        .reset_index()
    )
    summary.to_csv(
        results / "evaluation" / "summary.csv", index=False, lineterminator="\n"
    )
    rows: list[str] = [
        f"{row.formula} & {NAMES.get(str(row.model), 'Oracle')} & {row.test_r2:.3f}"
        + "".join(
            f" & ${getattr(row, name):.4f} \\pm {getattr(row, f'{name}_se'):.4f}$"
            for name in errors
        )
        + f" & {row.ig_rmse:.3f} \\\\"
        for row in summary.itertuples()
    ]
    (results / "evaluation" / "summary.tex").write_text(
        "\n".join(
            [
                "\\begin{tabular}{llrrrrr}",
                "\\toprule",
                (
                    "Formula & Model & Test $R^2$ & Behavior MAE & Decoy credit"
                    " & Total bias & IG RMSE \\\\"
                ),
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def _save(figure: Figure, results: Path, name: str) -> None:
    """Save ``figure`` as PDF and PNG under ``results/figures``."""
    (results / "figures").mkdir(parents=True, exist_ok=True)
    suffix: str
    for suffix in ("pdf", "png"):
        figure.savefig(results / "figures" / f"{name}.{suffix}", dpi=200)


def plot_contributions(contributions: pd.DataFrame, results: Path = RESULTS) -> None:
    """Plot each model's mean contributions next to the truth, by formula.

    Parameters
    ----------
    contributions : pandas.DataFrame
        The contributions from `evaluate`.
    results : pathlib.Path
        The results folder.
    """
    models: list[str] = [
        model for model in COLORS if model in set(contributions["model"])
    ]
    width: float = 0.8 / (len(models) + 1)
    position: np.ndarray = np.arange(len(FEATURES))
    figure: Figure = Figure(figsize=(11, 6.5), layout="constrained")
    axes: np.ndarray = figure.subplots(2, 1, sharex=True)
    index: int
    formula: str
    for index, formula in enumerate(RULES):
        ax: Axes = axes[index]
        rows: pd.DataFrame = contributions[contributions["formula"] == formula]
        # The reference is the true formula's IG on the same cohorts, the
        # same reference the errors in summary.csv use.
        truth: pd.DataFrame = (
            rows[rows["model"] == "oracle"]
            .groupby("behavior")["value"]
            .agg(["mean", "sem"])
            .loc[list(FEATURES)]
        )
        ax.bar(
            position - 0.4 + width / 2,
            truth["mean"],
            width,
            yerr=truth["sem"],
            color="#d4d3cf",
            edgecolor="#52514e",
            hatch="//",
            error_kw={"elinewidth": 1, "ecolor": "#0b0b0b"},
            label="True formula",
        )
        offset: int
        model: str
        for offset, model in enumerate(models, start=1):
            stats: pd.DataFrame = (
                rows[rows["model"] == model]
                .groupby("behavior")["value"]
                .agg(["mean", "sem"])
                .loc[list(FEATURES)]
            )
            ax.bar(
                position - 0.4 + width / 2 + offset * width,
                stats["mean"],
                width,
                yerr=stats["sem"],
                color=COLORS[model],
                error_kw={"elinewidth": 1, "ecolor": "#0b0b0b"},
                label=NAMES[model],
            )
        ax.axhline(0, color="#0b0b0b", linewidth=0.8)
        ax.set_title(f"{formula.capitalize()} formula")
        ax.set_ylabel("Contribution to impostors'\nextra sus per round")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(ncols=len(models) + 1, frameon=False)
    axes[1].set_xticks(position, LABELS, rotation=30, ha="right")
    _save(figure, results, "contributions")


def plot_trajectories(trajectories: pd.DataFrame, results: Path = RESULTS) -> None:
    """Plot mean sus by round for each group, with each model's mean prediction.

    Parameters
    ----------
    trajectories : pandas.DataFrame
        The trajectories from `evaluate`.
    results : pathlib.Path
        The results folder.
    """
    means: pd.DataFrame = (
        trajectories.groupby(["formula", "model", "impostor", "round"])["sus"]
        .mean()
        .reset_index()
    )
    figure: Figure = Figure(figsize=(10, 4), layout="constrained")
    axes: np.ndarray = figure.subplots(1, 2, sharey=True)
    colors: dict[str, str] = {"oracle": "#0b0b0b"} | COLORS
    markers: dict[int, str] = {1: "o", 0: "s"}
    index: int
    formula: str
    for index, formula in enumerate(RULES):
        ax: Axes = axes[index]
        model: str
        color: str
        for model, color in colors.items():
            group: int
            for group, marker in markers.items():
                rows: pd.DataFrame = means[
                    (means["formula"] == formula)
                    & (means["model"] == model)
                    & (means["impostor"] == group)
                ]
                ax.plot(
                    rows["round"],
                    rows["sus"],
                    color=color,
                    linestyle="-" if model == "oracle" else "--",
                    marker=marker,
                    markersize=5,
                    linewidth=3.5 if model == "oracle" else 1.5,
                )
        ax.set_title(f"{formula.capitalize()} formula")
        ax.set_xlabel("Round")
        ax.set_xticks(range(synthetic_data.ROUNDS))
        ax.grid(color="#e4e3df", linewidth=0.6)
    axes[0].set_ylabel("Mean sus")
    present: list[str] = [model for model in colors if model in set(means["model"])]
    handles: list[Line2D] = [
        Line2D(
            [],
            [],
            color=colors[model],
            linestyle="-" if model == "oracle" else "--",
            linewidth=3.5 if model == "oracle" else 1.5,
        )
        for model in present
    ] + [
        Line2D([], [], color="#52514e", linestyle="", marker=markers[group])
        for group in (1, 0)
    ]
    labels: list[str] = [
        "True sus" if model == "oracle" else NAMES[model] for model in present
    ] + ["Impostors", "Crewmates"]
    figure.legend(handles, labels, loc="outside right center", frameon=False)
    _save(figure, results, "trajectories")


def main(root: Path = training.DATA, results: Path = RESULTS) -> None:
    """Write every table and figure for the report.

    Parameters
    ----------
    root : pathlib.Path
        The data folder.
    results : pathlib.Path
        The results folder, which holds ``final``.
    """
    contributions: pd.DataFrame
    scores: pd.DataFrame
    trajectories: pd.DataFrame
    contributions, scores, trajectories = evaluate(root, results)
    print(summarize(scores, results).to_string(index=False))
    plot_contributions(contributions, results)
    plot_trajectories(trajectories, results)


if __name__ == "__main__":
    main()
