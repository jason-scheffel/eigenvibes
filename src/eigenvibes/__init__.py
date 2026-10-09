# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Shared code for the eigenvibes course project.

Importing the package sets the environment variable ``CUBLAS_WORKSPACE_CONFIG``
to ``:4096:8`` unless it is already set. On NVIDIA GPUs,
``torch.use_deterministic_algorithms(True)`` raises an error at the first
cuBLAS call without it, and it must be set before that call, so import
``eigenvibes`` before running anything on the GPU.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
