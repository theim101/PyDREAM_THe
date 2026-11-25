# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A maintained fork of **PyDREAM** (DREAM(ZS) MCMC sampler: differential evolution with
snooker updates, multi-try, adaptive crossover, parallel tempering). This copy is the
user's own fork (remotes `origin` = GitHub `theim101/PyDREAM_THe`, `my_origin` = TU Delft
GitLab). It is installed in editable mode in the conda environment `work3_14` and is used
by the landfill emission project (`~/data/GitLab/landfill_emission_exchange`), whose
DREAM runs import `pydream.core.run_dream` from here.

Upstream modernisation (Python >= 3.11, NumPy 2.x, typing, tests) is led by Benjamin Zoller;
see `DEVELOPMENT.md` and `gemini.md` for the project goals and agent directives.

## Rules

- **Preserve the statistical semantics** of DREAM: proposal generation, acceptance,
  chain evolution, crossover/gamma adaptation and multiprocessing behaviour must stay
  equivalent unless a change fixes a demonstrated correctness bug. Prefer minimal,
  targeted changes (`gemini.md`).
- **Every bug fix is recorded in `PYDREAM_BUGS_DOCUMENTATION.md`** as
  `## Fixed Bug N: <title>` with the subsections `### Symptom`, `### Root Cause`,
  `### Implementation of Fix` (bullets with **Location** / **Action**, plus semantics and
  what was deliberately not changed) and verification. Update the
  **Current Status** line at the top. Separate sections with `---`.
- A fix that changes behaviour gets a switch that restores the original behaviour when
  possible (e.g. `snooker_tol=0`), and a regression test that reproduces the defect with
  the switch off.

## Layout

```
pydream/
  core.py               run_dream (entry point), worker pool set-up, parallel tempering
  Dream.py              Dream step class: astep, proposals (generate_proposal_points,
                        snooker_update, snooker_degenerate), crossover/gamma adaptation,
                        history sampling (sample_from_history), multi-try evaluation
  Dream_shared_vars.py  shared-memory arrays (history, count, crossover/gamma state)
  model.py, parameters.py  Model and SampledParam (prior) wrappers
  convergence.py        Gelman-Rubin
  tests/                test_dream.py (needs pysb), test_models.py, test_snooker_degenerate.py
  examples/             CORM, Robertson (pysb), mixture model, n-dim Gaussian
```

## Testing

- `python -m pytest -q pydream/tests` (cwd = repository root). `test_dream.py` imports
  pysb examples; without pysb installed only the other test files run, e.g.
  `python -m pytest -q pydream/tests/test_snooker_degenerate.py`.
- MCMC output is stochastic: test invariants, shapes, and approximate moments on simple
  targets, not exact chains (`DEVELOPMENT.md`).
- Lint: `ruff` (line length 150, docstrings 120; config in `pyproject.toml`).

## Changes made in this fork (summary; details in PYDREAM_BUGS_DOCUMENTATION.md)

- Bug 1: `multitry=2` crash (NumPy squeeze) (fixed before this fork was taken).
- Bug 2: nested multiprocessing pools with multitry + parallel (fixed before this fork was taken).
- Bug 3 (2026-11-25): snooker reference point `z` equal to the current point up to
  round-off gave a huge Jacobian term and accepted bad moves; guarded with
  `snooker_tol` (default 1e-8).
- Bug 4 (2026-11-25): a void snooker move (z = q0, exact or round-off) gave a nan
  acceptance ratio (`RuntimeWarning` in `astep`) and a useless likelihood evaluation;
  now skipped (single-try).
