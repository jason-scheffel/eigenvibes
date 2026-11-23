# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Put every synthetic dataset in ``data/`` at the repository root.

Run it with ``uv run python scripts/make_datasets.py download`` to get the
exact files the project's results use, or ``... generate`` to write them
from their seeds. The README describes the files.
"""

import enum
from pathlib import Path

import huggingface_hub
import typer

from eigenvibes.synthetic_data import write_datasets

DATA: Path = Path(__file__).resolve().parent.parent / "data"
REPOSITORY: str = "jason-scheffel/eigenvibes"
# The upload the project's results use.
REVISION: str = "dbdc1ef9d1e1fe45294538756ec17e0e7ede074d"


class Source(enum.StrEnum):
    """Where the datasets come from."""

    download = "download"
    generate = "generate"


def main(source: Source) -> None:
    """Download or generate the datasets.

    Parameters
    ----------
    source : Source
        ``download`` fetches the files from the Hugging Face dataset
        ``REPOSITORY`` at ``REVISION``; ``generate`` writes them from their
        seeds.
    """
    if source is Source.generate:
        write_datasets(DATA)
        return
    huggingface_hub.snapshot_download(
        REPOSITORY,
        repo_type="dataset",
        revision=REVISION,
        local_dir=DATA,
        allow_patterns=["*.csv", "*.json"],
    )


if __name__ == "__main__":
    typer.run(main)
