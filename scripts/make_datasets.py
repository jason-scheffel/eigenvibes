# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Put every synthetic dataset in ``data/`` at the repository root.

Run it with ``uv run python scripts/make_datasets.py download`` to get the
exact files the project's results use, or ``... generate`` to write them
here. The README describes the files.
"""

import enum
import hashlib
from pathlib import Path

import huggingface_hub
import typer

from eigenvibes.synthetic_data import CHECKSUMS, write_datasets

DATA: Path = Path(__file__).resolve().parent.parent / "data"
REPOSITORY: str = "jason-scheffel/eigenvibes"
# The upload whose hashes data/SHA256SUMS lists.
REVISION: str = "dc6e8b38dd349efb252d8f4a3b267d0e21b69287"


class Source(enum.StrEnum):
    """Where the datasets come from."""

    download = "download"
    generate = "generate"


def main(source: Source) -> None:
    """Download or generate the datasets.

    ``download`` fetches the files generated on the DGX from the Hugging Face
    dataset ``REPOSITORY`` at ``REVISION`` and stops with an error if any
    differs from ``data/SHA256SUMS``. ``generate`` writes them from their seeds
    and only warns about differences, since another CPU can round the last
    digits differently.

    Parameters
    ----------
    source : Source
        ``download`` or ``generate``.
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
    listing: list[str] = (DATA / CHECKSUMS).read_text(encoding="utf-8").splitlines()
    different: list[str] = [
        path
        for digest, path in (line.split("  ", 1) for line in listing)
        if hashlib.sha256((DATA / path).read_bytes()).hexdigest() != digest
    ]
    if different:
        raise SystemExit(
            f"{len(different)} downloaded files differ from {CHECKSUMS}: "
            + ", ".join(different[:5])
        )
    print(f"Downloaded {len(listing)} files; all match {CHECKSUMS}.")


if __name__ == "__main__":
    typer.run(main)
