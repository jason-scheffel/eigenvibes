# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Put every synthetic dataset in ``data/`` at the repository root.

Run it with ``uv run python scripts/make_datasets.py download`` to get the
exact files the project's results use, together with the tuning studies and
the trained models in ``results/``, or ``... generate`` to write the
datasets from their seeds. The README describes the files.
"""

import enum
from pathlib import Path

import huggingface_hub
import typer

from eigenvibes.synthetic_data import write_datasets

ROOT: Path = Path(__file__).resolve().parent.parent
DATA: Path = ROOT / "data"
REPOSITORY: str = "jason-scheffel/eigenvibes"
# The upload the project's results use.
REVISION: str = "f44138ac72e6e51e12d252ed67768c11f358b886"


class Source(enum.StrEnum):
    """Where the datasets come from."""

    download = "download"
    generate = "generate"


def main(source: Source) -> None:
    """Download or generate the datasets.

    Parameters
    ----------
    source : Source
        ``download`` fetches the datasets into ``data/`` and the tuning
        studies and trained models into ``results/tuning`` and
        ``results/final``, from the Hugging Face dataset ``REPOSITORY`` at
        ``REVISION``. ``generate`` writes only the datasets, from their seeds.
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
        ignore_patterns=["results/*"],
    )
    huggingface_hub.snapshot_download(
        REPOSITORY,
        repo_type="dataset",
        revision=REVISION,
        local_dir=ROOT,
        allow_patterns=["results/*"],
    )


if __name__ == "__main__":
    typer.run(main)
