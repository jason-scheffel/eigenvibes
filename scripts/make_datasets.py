# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Write every synthetic dataset to ``data/`` at the repository root.

Run it with ``uv run python scripts/make_datasets.py``. The README describes
the files.
"""

from pathlib import Path

from eigenvibes.synthetic_data import write_datasets

if __name__ == "__main__":
    write_datasets(Path(__file__).resolve().parent.parent / "data")
