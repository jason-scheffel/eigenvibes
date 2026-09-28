<!--
SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# Agent Instructions

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for licensing, workflow, signing, and commit messages.

## Commands

Use `uv` for everything; do not use `pip` or create other virtual environments.

- `uv sync`: install the project and development tools
- `uv add <package>` / `uv add --dev <package>`: add a dependency; commit `pyproject.toml` and `uv.lock` together
- `uv run pre-commit run --all-files`: run ruff, mypy, and reuse
- `uv run pytest`: run the tests

The checks and tests must pass before committing.

## Code

- Target Python 3.14. Code must run on Linux, Windows, and macOS.
- Type-annotate every function signature and every variable, including locals (`count: int = 0`); mypy runs in strict mode.
- Use current typing syntax: built-in generics (`list[int]`, `dict[str, float]`), `X | None` and `X | Y` instead of `Optional` and `Union`, `collections.abc` for `Callable` and `Iterable`, the `type` statement for aliases, and type parameters (`def f[T](x: T) -> T`) instead of `TypeVar`. Do not use `from __future__ import annotations`; Python 3.14 already defers annotation evaluation.
- Write NumPy-style docstrings for every public module, class, and function, including tests.
- Start every new file with an SPDX header naming its author and license. Code is BSD-3-Clause, documentation is CC-BY-SA-4.0, and config files are CC0-1.0 and listed in `REUSE.toml`.
- Use `pathlib` for paths and pass `encoding="utf-8"` when opening text files.
- Keep the shared package in `src/eigenvibes/`. Changes to it go in their own pull request, not mixed with experiment code.
- Never commit downloaded data.
- Use `torch.float64` for all tensors and model parameters.
- Use a CUDA GPU when one is available and fall back to the CPU otherwise; all code must run on a CPU-only machine.
- Write tests only for behavior that could realistically break. Do not add unit tests for their own sake.
- Keep code small and direct. Do not add abstractions (base classes, wrappers, registries, config layers, single-use helpers) unless they remove duplication that exists now.

## Determinism

Every result must be reproducible bit-for-bit on the same machine and to numerical tolerance across operating systems.

- Pass an explicit seed to every source of randomness. Use `42` when one seed is needed, and `0, 1, 2, ...` when several are. Use `numpy.random.default_rng(seed)` and pass the generator to functions; do not use `np.random.seed` or other global random state.
- For PyTorch, call `torch.manual_seed(seed)` and `torch.use_deterministic_algorithms(True)`.
- Sort anything whose order is not guaranteed before using it, such as directory listings (`Path.glob`, `Path.iterdir`) and sets.
