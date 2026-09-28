# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the package can be imported."""

import eigenvibes


def test_import() -> None:
    """Import the package and check its name."""
    assert eigenvibes.__name__ == "eigenvibes"
