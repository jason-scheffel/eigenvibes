#!/bin/bash
# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause
#
# Usage, from a clone in your lab directory: sbatch scripts/tune.sbatch
# Copies 0-3 tune the linear formula and copies 4-7 the interaction formula,
# one A100 each. Submit it again to resume a stopped run.

#SBATCH --job-name=eigenvibes-tune
#SBATCH --array=0-7
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
mkdir -p .uv
flock .uv/sync.lock uv sync --locked --extra cu126
formulas=(linear linear linear linear interaction interaction interaction interaction)
uv run --no-sync python scripts/tune.py "${formulas[$SLURM_ARRAY_TASK_ID]}" "$SLURM_ARRAY_TASK_ID"
