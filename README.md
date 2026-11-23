<!--
SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# eigenvibes

[![CI](https://github.com/jason-scheffel/eigenvibes/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jason-scheffel/eigenvibes/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![Python version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjason-scheffel%2Feigenvibes%2Fmaster%2Fpyproject.toml)](pyproject.toml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![REUSE status](https://api.reuse.software/badge/github.com/jason-scheffel/eigenvibes)](https://api.reuse.software/info/github.com/jason-scheffel/eigenvibes)

<p align="center"><img src="docs/crewmates.svg" alt="Hand-drawn Among Us fan art: a red crewmate next to a fallen blue crewmate" width="400"></p>

> Course project for CSI 436/536 (Fall 2026, University at Albany).
>
> This project checks whether explanations of neural networks give the right answer. To know the right answer, it uses synthetic data.
>
> The data describe fictional Among Us players. Each player has 12 behavior numbers, such as time spent near vents or time spent alone, and a formula we chose turns them into a **sus score**: how suspicious the player looks. Neural networks learn to predict sus from the 12 numbers. Then we follow a group of players for four rounds. Impostors' sus rises every round, and crewmates' sus stays flat.
>
> We ask each trained network *why* impostors' sus rises, and compare its answer with the truth. The answer is one amount per behavior, and the amounts add up to the rise: for example, 1.0 of the rise came from `task_inactivity`, 0.125 from `task_abandonment`, and so on. It is computed in two steps:
>
> 1. Integrated Gradients [1] explains each single prediction: it splits the predicted sus into how much each behavior contributed.
> 2. A mixed model (a regression for repeated measurements of the same players) combines those per-prediction amounts into amounts for the difference between the impostors' rise and the crewmates' rise.
>
> In [2], this method explained a VNN's brain-age predictions for two groups of patients. On real data, however, nobody knows the true amounts, so there is no way to check whether they are right. Here we wrote the formula, so we know the true amounts exactly.
>
> We compare:
>
> - **Models.** A VNN (coVariance Neural Network [3]) uses the covariance matrix of the 12 behaviors, estimated from its training data, as a graph that links behaviors that vary together, and each layer mixes every behavior with the behaviors it is linked to. An MLP, a standard fully connected network, has no such built-in structure. Ridge regression, a linear model, is the simple baseline.
> - **Two formulas for sus.** One adds up its parts; the other also multiplies two of them.
> - **A decoy.** One behavior rises for impostors but has no effect on sus, so a correct explanation gives it no credit.
>
> [Datasets](#datasets) describes the data in detail.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [GPU benchmark](#gpu-benchmark)
- [Tuning](#tuning)
- [Final training](#final-training)
- [Contributors](#contributors)
- [Use of generative AI](#use-of-generative-ai)
- [License](#license)
- [References](#references)

## Requirements

Python 3.14 and [uv](https://docs.astral.sh/uv/). uv installs Python 3.14 itself if it is missing. Intel Macs are not supported because PyTorch no longer publishes builds for them.

## Installation

Clone the repository, then run one of these commands from its root, depending on your hardware. Each installs the project, its dependencies, and the matching PyTorch build.

```
uv sync --extra cu130   # NVIDIA GPU on Linux or Windows (driver 580 or newer)
uv sync --extra cu126   # NVIDIA GPU with driver 525.60.13 or newer (Windows: 528.33) but older than 580; no RTX 50-series
uv sync --extra rocm    # AMD GPU on Linux
uv sync --extra cpu     # no GPU, macOS, or an AMD GPU on Windows
```

Plain `uv sync` and `uv remove` uninstall PyTorch. Run your command again after either one.

Run scripts with `uv run python <script>`.

## Datasets

`uv run python scripts/make_datasets.py download` downloads every dataset into `data/` (gitignored) from the [Hugging Face dataset](https://huggingface.co/datasets/jason-scheffel/eigenvibes), at the upload our results use. `uv run python scripts/make_datasets.py generate` instead writes them from their seeds in a few seconds. Another CPU may round the last digit of some numbers differently, which does not change any result.

### Files

```
data/
  README.md        the Hugging Face dataset card (in git)
  covariance.csv   how the 12 behaviors vary together
  design.json      the formulas, the cohort settings and the true answers
  linear/          sus from the linear formula
    evaluation_00/ ... evaluation_29/
      train.csv  validation.csv  test.csv  cohort.csv
    tuning_30/ ... tuning_39/
      train.csv  validation.csv
  interaction/     sus from the interaction formula; same layout
```

The number in a folder name is the seed that made the dataset. Tuning datasets are only for choosing hyperparameters, and evaluation datasets are only for final results. Both formulas use the same players, so only the `sus` column differs between `linear/` and `interaction/`.

| File | Rows | Columns |
|---|---:|---|
| `train.csv` | 2,000 | 12 behaviors, `sus` |
| `validation.csv` | 500 | same |
| `test.csv` | 2,000 | same |
| `cohort.csv` | 800 | `player`, `round`, `impostor`, 12 behaviors, `sus` |

Read the files with `pandas.read_csv(..., float_precision="round_trip")` to get every value back exactly.

### Behaviors

Each row is a made-up player with 12 behavior scores. A score of 0 is a typical player, and every behavior has a standard deviation of 1. The behaviors are correlated because they come from three hidden play styles: slacking, sneaking and moving. `covariance.csv` holds the exact correlations.

| # | Behavior | Play style |
|---|---|---|
| 1 | `task_inactivity` | slacking |
| 2 | `vent_lingering` | sneaking |
| 3 | `time_alone` | sneaking, some slacking |
| 4 | `quiet_room_preference` | sneaking |
| 5 | `task_abandonment` | slacking |
| 6 | `following_players` | sneaking, some moving |
| 7 | `distance_walked` | moving |
| 8 | `room_changes` | moving |
| 9 | `walking_speed` | moving |
| 10 | `hallway_time` | moving, some sneaking |
| 11 | `route_variability` | slacking, some moving |
| 12 | `task_station_proximity` | less slacking |

### Sus

- **linear:** sus = 2 `task_inactivity` + `vent_lingering` + 0.5 `task_abandonment`
- **interaction:** the linear formula + `task_inactivity` × `vent_lingering`

The other 9 behaviors do not affect sus. In `train.csv` and `validation.csv`, sus also has noise with a standard deviation of 10% of sus's; in `test.csv` and `cohort.csv`, sus is exact.

### Cohort

The cohort follows 200 players for 4 rounds (0 to 3), and odd-numbered players are impostors. Part of each player's behavior stays the same every round (60% of the variance), and the rest changes from round to round. The cohort's players linger near vents more than the training players: their `vent_lingering` averages 1 instead of 0.

Every round, impostors' `task_inactivity` rises by 0.5, their `task_abandonment` by 0.25 and their `route_variability` by 0.35. `route_variability` is a decoy, because it rises but does not affect sus.

### True answers

Impostors' sus rises faster than crewmates' every round. Integrated Gradients splits each sus score among the behaviors, and the true answer is each behavior's share of that faster rise:

| Behavior | linear | interaction |
|---|---:|---:|
| `task_inactivity` | 1.000 | 1.250 |
| `vent_lingering` | 0 | 0.250 |
| `task_abandonment` | 0.125 | 0.125 |
| `route_variability` (decoy) | 0 | 0 |
| the other 8 | 0 | 0 |
| **total** | **1.125** | **1.625** |

In the interaction formula, `vent_lingering` gets credit even though it does not change. The product term rises because `task_inactivity` rises while `vent_lingering` is high, and Integrated Gradients splits that rise evenly between the two behaviors.

## GPU benchmark

`scripts/benchmark.py` measures how fast VNNs train on a GPU and whether training several at the same time on one GPU gets more done than training them one after another. It only prints its results. On the DGX, run it with `sbatch scripts/benchmark.sbatch`.

Results on one NVIDIA A100-SXM4-80GB (October 2026, torch 2.14.1+cu126), training copies of a 3-layer VNN with 128 channels and 8 filter taps per layer on 2,000 rows. Each value is the total epochs per second of all models together, relative to one model alone:

| Models at once | Batch 2 | Batch 8 | Batch 32 |
|---:|---:|---:|---:|
| 1 | 1.00x | 1.00x | 1.00x |
| 2 | 2.04x | 2.12x | 1.98x |
| 4 | 3.84x | 3.77x | 3.75x |
| 8 | 4.03x | 4.08x | 3.89x |
| 16 | 4.04x | 4.10x | 3.89x |

Our training scripts therefore run 4 models at once on each GPU. Four gets nearly all of the gain. Eight adds at most 8% more throughput but doubles the time each model takes, so each result arrives later; Optuna picks each new trial from the finished ones, so earlier results matter more than a few percent of throughput.

A training step also takes about as long at batch size 2 as at 32 (about 3.3 ms for this model), so an epoch at batch size 2 takes about 16 times longer.

## Tuning

`scripts/tune.py` tunes the VNN's hyperparameters with [Optuna](https://optuna.org/), separately for each formula. A trial trains one set of hyperparameters on each of the 10 tuning datasets, with the covariance estimated from that dataset's `train.csv`, and its score is the mean validation MSE. Training uses Adam on MSE and stops after 25 epochs without a lower validation loss, or at 1,500 epochs, and keeps the weights of the best epoch. Every layer uses PReLU.

| Hyperparameter | Range |
|---|---|
| Learning rate | 1e-4 to 1e-1, log scale |
| Batch size | 2, 4, 8, 16, 32, 64, 128, or 256 |
| Layers | 1 to 4 |
| Channels per layer | 1 to 256 |
| Filter taps per layer | 1 to 12 |

On the DGX, `sbatch scripts/tune.sbatch` starts 8 copies of the script, one A100 each: copies 0–3 tune the linear formula and 4–7 the interaction formula. Each copy trains 4 models at a time (see [GPU benchmark](#gpu-benchmark)). When one finishes, it starts the next dataset of a trial it is running, or else asks Optuna for a new trial. The copies of a formula share one Optuna study and stop when it has 200 trials. To resume a stopped run, submit the job again; each copy marks only its own unfinished trials as failed.

It writes to `results/` (gitignored):

- `results/tuning/<formula>.log`: the Optuna study, as an Optuna journal file, which several machines can share safely. It holds each trial's hyperparameters and score, and for each finished trial, every dataset's validation MSE, epochs, and training time. Open it with `optuna.load_study` and `JournalStorage`.
- `results/tuning/nodes/<formula>_<copy>_<job>.json`: each copy's run time and how many trainings it finished.
- `results/tuning/summary.json`, per formula:
  - **wall hours**, from the first trial's start to the last trial's end, including any wait between a stopped job and its resubmission;
  - **GPU hours**, the total time the copies held a GPU;
  - **training hours**, the total time of all trainings. With 4 trainings per GPU, one GPU hour holds up to 4 training hours;
  - the median and longest training time, the median number of epochs, the number of finished and diverged trials, and the best trial.
- `tune-<job>_<copy>.out` in the clone: a line for every finished training and trial.

From the benchmark, tuning should take roughly 3 to 36 hours, depending mostly on which batch sizes Optuna favors; it has not been measured yet.

## Final training

After tuning, `scripts/train_final.py` trains the final VNNs on the 30 evaluation datasets with the best trial's hyperparameters for each formula. It refuses to start until both studies have 200 finished trials. Each formula gets two kinds of VNN:

- **true covariance**: the VNN uses `data/covariance.csv`, the covariance the behaviors were drawn from, for training and predicting;
- **estimated covariance**: the VNN uses the covariance of its dataset's `train.csv`, as in tuning.

Each kind is trained on each evaluation dataset with seeds 0, 1, and 2, which gives 2 formulas × 2 kinds × 30 datasets × 3 seeds = 360 models. A model trains as in tuning and stops early on its own `validation.csv`. It is then scored on `test.csv`, whose `sus` has no noise, by the RMSE and R² of its predictions.

On the DGX, `sbatch scripts/train_final.sbatch` starts 8 copies of the script, one A100 each. Each copy trains a fixed 45 of the 360 models, 4 at a time. A copy skips models that are already saved, so submitting the job again resumes a stopped run.

It writes to `results/final/` (gitignored):

- `<formula>/<covariance>/<dataset>_seed_<seed>.pt`: the trained model, which `torch.load` reads. It holds the `state_dict`, the covariance matrix the model was built with (as `covariance_matrix`, since the VNN's `state_dict` does not include it), the hyperparameters and tuning trial, and the scores below. `training.build` with that matrix, the hyperparameters, and the seed, followed by `load_state_dict`, restores the model.
- `<formula>/<covariance>/<dataset>_seed_<seed>.json`: the same values without the weights: validation MSE, best epoch, epochs, test RMSE and R², whether training diverged, and the training time.
- `metrics.csv`: one row per model with those values.
- `nodes/<copy>_<job>.json`: each copy's run time and how many models it trained.
- `summary.json`: GPU hours and training hours (as in [Tuning](#tuning)), and per formula, the tuning trial and hyperparameters, and per kind of covariance, the number of models, how many diverged, and the mean and standard deviation of the test RMSE and R².
- `final-<job>_<copy>.out` in the clone: a line for every finished model.

## Contributors

<p align="center"><a href="https://github.com/jason-scheffel/eigenvibes/graphs/contributors"><img src="docs/contributors.svg" alt="One crewmate per contributor, each with their GitHub avatar in the visor and their username below"></a></p>

`uv run python scripts/contributors.py` redraws this image from the repository's contributor list on GitHub. Run it after someone's first pull request is merged, once GitHub's [contributors page](https://github.com/jason-scheffel/eigenvibes/graphs/contributors) shows them; GitHub can take a few hours to update the list.

## Use of generative AI

This project uses generative AI tools (see section 4.8 of the [course syllabus](https://chong-l.github.io/CSI436_536_26F.html)):

- Coding agents help write and edit some of the code, tests, and documentation.
- Inline code completion in our editors suggests code as we type, though not all suggestions are accepted.
- Commit messages and pull request descriptions are sometimes written with AI help.
- GPT-6 Astra, run locally, is the first reviewer: we run it on changes before pushing them.
- CodeRabbit then reviews pull requests.

Coding agents do not act on their own. We tell them specifically what to do, and the only work they do unprompted is routine, such as exploring the codebase for context. Some of the code also follows high-level designs from earlier projects, similar to design patterns.

We read every line before it is merged and stand by all of it. We further verify the work by:

- tests, run in CI on Linux, Windows, and macOS;
- comparisons with known answers, for instance a bit-for-bit comparison of the VNN against the earlier implementation it replaces;
- identities the method must satisfy exactly, for instance the components summing to the total effect.

Some findings and bugs were found by AI tools rather than by us. We checked each one before relying on it:

| Finding | Found by |
| --- | --- |
| On ROCm, GPU matrix products with more than 524,288 rows return wrong values without an error. | Claude Opus 5.5 |
| A covariance matrix that carries gradients leaks them back into the data it was computed from, so the VNN detaches it. | Claude Opus 5.5 |
| Averaging the VNN's last layer over channels rounds differently depending on memory layout, so matching the earlier implementation exactly needs a contiguous copy. | Claude Opus 5.5 |
| Reproducing the earlier implementation's seeded initialization requires one extra set of random draws that it made and discarded. | Claude Opus 5.5 |
| Tiny rounding differences in a model's output depend on the batch size, so batch sizes must stay fixed for results to reproduce exactly. | Claude Opus 5.5 |
| The VNN's prediction and its regional outputs round differently, so a regional residual that should be zero came out around 1e-16 and was normalized into a full-length direction instead of raising an error. | GPT-6 Astra |
| The CI workflow referred to GitHub Actions by release tags, which can be moved to point at different code, and kept the job's token in `.git/config` after checkout. | CodeRabbit |
| The covariance check accepted matrices containing NaN, because comparisons with NaN are false, so the eigendecomposition later failed with a misleading error. | CodeRabbit |
| Prediction split its input into chunks of 524,288 divided by the number of nodes, which is 0 for graphs with more nodes than that and makes the split fail. | CodeRabbit |
| If a benchmark worker failed to start, the workers already running waited forever, and Python hung on exit. | CodeRabbit |
| The cu126 build needs NVIDIA driver 525.60.13 on Linux and 528.33 on Windows, not any 525 release. | CodeRabbit |
| A test required an exact zero from values recomputed after a round trip through CSV, which can fail from last-digit rounding on another platform. | CodeRabbit |

## License

The code is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for the full text. Documentation and other prose, including this README, is under [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt). Config files and the generated data in `data/` are under [CC0 1.0](LICENSES/CC0-1.0.txt).

The crewmate drawings in `docs/crewmates.svg` and `docs/contributors.svg` are fan art and are not under any of these licenses. This project is not affiliated with Among Us or Innersloth LLC; see [the fan art notice](LICENSES/LicenseRef-Among-Us-Fan-Art.txt) for details.

## References

[1] M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic Attribution for Deep Networks," in *Proc. 34th International Conference on Machine Learning (ICML)*, 2017. https://arxiv.org/abs/1703.01365

[2] J. Scheffel and S. Sihag, "Covariance Eigenspace Provides Latent Attribution of Longitudinal Effects in Brain Age Gap," arXiv:2609.40202, 2026. https://arxiv.org/abs/2609.40202

[3] S. Sihag, G. Mateos, C. McMillan, and A. Ribeiro, "coVariance Neural Networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. https://arxiv.org/abs/2205.15856
