<!--
SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# Contributing

## License

Code is licensed under BSD-3-Clause, documentation and other prose under CC BY-SA 4.0, and config files under CC0 1.0. By contributing, you agree that your contributions will be licensed under the same terms.

The project follows the [REUSE](https://reuse.software/) specification. Every file you add needs an [SPDX](https://spdx.dev/learn/handling-license-info/) header naming you as the author and giving its license, for example:

```python
# SPDX-FileCopyrightText: 2026 Your Name <you@example.com>
# SPDX-License-Identifier: BSD-3-Clause
```

If you make substantial changes to a file someone else wrote, add your own `SPDX-FileCopyrightText` line below theirs. Files that cannot hold comments are listed in `REUSE.toml` instead. Run `reuse lint` before opening a pull request.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

- `uv sync`: install Python 3.14, the project, and the development tools
- `uv run pre-commit install`: run the checks (ruff, mypy, reuse) on every commit; do this once per clone
- `uv run pre-commit run --all-files`: run the checks on every file
- `uv run pytest`: run the tests

CI runs the same checks and tests on Linux, Windows, and macOS for every pull request.

## Workflow

All changes go through a pull request into `master`. Pull requests are squash-merged, so each one becomes a single commit on `master`.

Start every pull request from a new branch off the latest `master`. Do not reuse a branch after its pull request has been merged.

```sh
git switch master
git pull
git switch -c <type>/<short-description>
# make changes
git add <files>
git commit -s -S -m "<type>: <description>"
git push -u origin <type>/<short-description>
```

Then open a pull request on GitHub into `master`.

## Signing Requirements

All commits must be signed off to indicate you agree to the [Developer Certificate of Origin](https://developercertificate.org/). Use `git commit -s`.

Commits must also be signed with `git commit -S`.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format. This applies to pull request titles too, since the title becomes the commit message on `master` when the pull request is squash-merged.

- Use imperative mood ("add feature" not "added feature")
- Subject line max 50 characters, lowercase after type
- Body wrapped at 74 characters
