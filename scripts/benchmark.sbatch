#!/bin/bash
# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause
#
# Usage, from a clone in your lab directory: sbatch scripts/benchmark.sbatch

#SBATCH --job-name=eigenvibes-benchmark
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80gb
#SBATCH --output=benchmark-%j.out

set -euo pipefail
# Home is only 10 GB, so keep uv's files in the clone.
export UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.uv/cache"
export UV_PYTHON_INSTALL_DIR="$SLURM_SUBMIT_DIR/.uv/python"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi
uv sync --locked --extra cu126
uv run python scripts/benchmark.py
