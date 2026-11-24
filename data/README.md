---
license: cc0-1.0
pretty_name: eigenvibes synthetic datasets
size_categories:
  - 100K<n<1M
task_categories:
  - tabular-regression
viewer: false
---

<!--
SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
SPDX-License-Identifier: CC0-1.0
-->

# eigenvibes synthetic datasets

Synthetic data for the [eigenvibes](https://github.com/jason-scheffel/eigenvibes) course project (CSI 436/536, Fall 2026, University at Albany), which checks whether explanations of trained models give the right answer. Each row is a made-up Among Us player with 12 behavior scores and a **sus score** (how suspicious the player looks) from a formula we chose, so the true explanation is known exactly.

The [Datasets section of the project README](https://github.com/jason-scheffel/eigenvibes#datasets) describes every file, column and design choice.

## Files

```
covariance.csv            how the 12 behaviors vary together
design.json               the formulas, the cohort settings and the true answers
linear/                   sus from the linear formula
  evaluation_00/ ... evaluation_29/
    train.csv  validation.csv  test.csv  cohort.csv
  tuning_30/ ... tuning_39/
    train.csv  validation.csv
interaction/              sus from the interaction formula; same layout
```

## Download

From a clone of the project repository:

```
uv run python scripts/make_datasets.py download
```

Read the CSV files with `pandas.read_csv(..., float_precision="round_trip")` to get every value back exactly.

## License

CC0 1.0: no rights reserved.
