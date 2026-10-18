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

Synthetic data for the [eigenvibes](https://github.com/jason-scheffel/eigenvibes) course project (CSI 436/536, Fall 2026, University at Albany), which checks whether explanations of neural networks give the right answer. Each row is a fictional Among Us player with 12 behavior numbers and a **sus score** (how suspicious the player looks) from a formula with known parts, so the true explanation is known exactly.

The [Datasets section of the project README](https://github.com/jason-scheffel/eigenvibes#datasets) describes every file, column, and design choice.

## Files

```
SHA256SUMS                SHA-256 hash of every file below
covariance.csv            how the 12 behaviors vary together; the same for every dataset
design.json               settings and expected answers, for scoring only
linear/                   sus from the linear formula
  evaluation_00/ ... evaluation_29/
    train.csv  validation.csv  test.csv
    cohort_base.csv  cohort_100_players.csv  ...  (7 cohort versions)
  tuning_30/ ... tuning_39/
    train.csv  validation.csv
interaction/              sus from the interaction formula; same layout
```

642 CSV and JSON files, 168 MB.

## How these files were made

`scripts/make_datasets.py` in the project repository writes them from fixed seeds. These files were generated on the project's DGX server. Generating them on another CPU gives the same data except for the last digits: compared with a desktop's run, no value differed by more than 3.6e-15, because NumPy's math library rounds slightly differently on different CPUs. These files are the exact ones the project's results use.

## Download

From a clone of the project repository:

```
uv run python scripts/make_datasets.py download
```

It downloads a fixed revision of this repository into `data/` and checks every file against `SHA256SUMS`. Read the CSV files with `pandas.read_csv(..., float_precision="round_trip")` to get every value back exactly.

## License

CC0 1.0: no rights reserved.
