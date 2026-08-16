# Contributing

Thanks for considering a contribution to Wind Turbine Fault Detection. This document covers the branching strategy, code style, PR process, and how to report issues.

## Branching strategy

- `main` is protected: no direct commits or pushes. All changes land via a Pull Request.
- Branch off `main` for every change, using a short, descriptive, kebab-case name prefixed by type:
  - `feature/<short-description>` — new functionality (e.g. `feature/real-time-inference-api`)
  - `fix/<short-description>` — bug fixes (e.g. `fix/lof-threshold-quantile`)
  - `docs/<short-description>` — documentation-only changes (e.g. `docs/add-citation`)
  - `chore/<short-description>` — tooling, dependencies, CI (e.g. `chore/bump-torch`)
- Keep branches focused on a single issue or piece of work; avoid bundling unrelated changes into one PR.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use type hints on public function/method signatures (see `src/models/lof.py` for the existing pattern).
- Use **Google-style docstrings** for public classes and functions:

  ```python
  def example(a: int, b: str = "default") -> bool:
      """
      One-line summary of what the function does.

      Args:
          a (int): Description of a.
          b (str): Description of b. Defaults to "default".

      Returns:
          bool: Description of the return value.
      """
  ```
- Keep the two model tracks (LOF and LSTM autoencoder) isolated: shared/wiring code goes in `main.py`, track-specific code stays under its own `lof_*` / `lstm_autoencoder_*` naming in `src/`.

## Pull request process

1. Open an issue first (or link to an existing one) so the change is discussed before work starts.
2. Branch off `main` following the naming convention above.
3. Add or update tests under `tests/lof/` or `tests/lstm/` for any behavior change — see [Testing](README.md#testing) in the README.
4. Run the test suite locally before opening the PR:
   ```bash
   pip install -r requirements-dev.txt
   pytest tests/lof tests/lstm
   ```
5. Open the PR against `main` with a description covering:
   - What changed and why.
   - Which issue it closes (`Closes #<number>`), if applicable.
   - How it was tested.
6. Keep PRs small and reviewable; a PR should be understandable as a single unit of change.

## Issue reporting

Use GitHub Issues for both bugs and feature requests, and pick the type that matches:

- **Bug report** — something doesn't work as documented. Include: steps to reproduce, expected vs. actual behavior, track affected (`lof` and/or `lstm`), and your environment (OS, Python version, whether you're running via Docker).
- **Feature request** — propose new functionality or an enhancement. Include: the problem it solves, a rough description of the expected behavior, and any alternatives you considered.

Label the issue accordingly (`bug`, `enhancement`, `documentation`, etc.) so it's easy to triage.
