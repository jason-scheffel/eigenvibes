#!/bin/bash
# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause
#
# Usage, from a clone in your lab directory: sbatch scripts/tune.sbatch
# Each copy tunes one study on one A100: copies 0-7 vnn_interaction, 8-13
# vnn_linear, 14 mlp_linear, and 15 mlp_interaction (COPIES in tune.py).
# Submit it again to resume a stopped run.

#SBATCH --job-name=eigenvibes-tune
#SBATCH --array=0-15
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --output=tune-%A_%a.out

set -euo pipefail
# Home is only 10 GB, so keep uv's files in the clone.
export UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.uv/cache"
export UV_PYTHON_INSTALL_DIR="$SLURM_SUBMIT_DIR/.uv/python"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
# One copy at a time updates .venv; the others then find it up to date.
# Download the data first, on the login node: see the README.
mkdir -p .uv
flock .uv/sync.lock uv sync --locked --extra cu126
uv run --no-sync python scripts/tune.py "$SLURM_ARRAY_TASK_ID"
