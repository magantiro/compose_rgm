# COMPOSE (RGM) — agent context

**Rewrite Generator Matching for validity-preserving molecular generation.**
A continuous-time Markov chain over molecular graphs whose committed states are
complete, chemically valid, connected molecules and whose transitions are
executable chemical rewrites; Generator Matching learns their contextual firing
rates. Non-monotone, flexible-size, target-free at inference. Paper 1 (methods)
lives in `paper_iclr_stochastic_rewriting/`.

> Not to be confused with the sibling repo `KoshaTx/compose` (a discrete-*diffusion*
> operator-editing framework). Same name lineage, different generative process.
> This repo is the **RGM / CTMC** line (`KoshaTx/compose_rgm`).

This file loads automatically every session. Keep it short and high-signal —
detail lives in the imported context files below and in scoped `CLAUDE.md`s.

## How we work

1. **Context auto-loads.** The `@`-imported files at the bottom are in your context now.
2. **Capture durable learnings** in `.claude/context/learnings.md` when you hit a non-obvious
   gotcha or make a design call. Commit it — that's how it spreads on the next pull.
3. **`.claude/` is code.** Edit the context files in the same PR as the change that makes them
   true; keep them terse. Stale context is worse than none.

## Commands

```bash
uv sync                          # .venv + deps (or: pip install -e ".[dev]"); then activate .venv
export PYTHONPATH=src:scripts    # experiment scripts import from both
export KMP_DUPLICATE_LIB_OK=TRUE # macOS OpenMP guard
pytest tests/                    # full suite (427 tests, pythonpath=["src"] -> `from compose_v4.…`)
pytest tests/test_e0_toy_h_exactness.py -q   # one file
ruff check .                     # lint (line-length 100, target py310)
python scripts/sanity_check.py   # fast local gauntlet — run before any Modal launch
```

Cloud training / rollout evaluation run on **Modal** (`modal_apps/`), against the
`compose-v4-artifacts` volume. Never launch a Modal job before `sanity_check.py` is green.
SMC experiment runs are single-process — the multiprocessing rollout path fork-deadlocks
on some macOS setups; prefer serial locally, Modal for scale (see `learnings.md`).

## Conventions (adopted from `KoshaTx/compose`)

- **Python 3.10+**; `ruff` (line-length 100, py310) and `pytest` config already in `pyproject.toml`.
- `from __future__ import annotations` at module top. Imports grouped stdlib → third-party
  (numpy, rdkit, torch, networkx, scipy) → local (`from compose_v4.*`).
- `snake_case` funcs/vars, `PascalCase` classes, `UPPER_CASE` constants (`NULL_IDX`, `BOND_CLASSES`).
- Rich **module-level docstrings** stating the data structure and the invariants
  maintained/tested. `# ---- Section ----` markers; inline `#` comments. No `__all__`.
- **Commits are atomic**: one self-contained logical change; subject
  `<module-or-filename>: <one-line description>`, imperative, lowercase, no trailing period.
  **Never mention Claude / AI** — no `Co-Authored-By` or "generated with" trailer, in commits
  *or* PR bodies.
- Feature branches → PR to `main`; both gates (`pytest tests/` + `scripts/sanity_check.py`)
  green before merge.

## Map

- `src/compose_v4/chem/` — `MolecularGraph`, padded state, validity/canonicalization.
- `src/compose_v4/rewrite/` — the rewrite kernel + legal-event **fibers** (`fiber.py`,
  `factorized_fiber.py`, `tracelet_fiber.py`, `ring_system_fiber.py`); `kernel.py` = executor.
- `src/compose_v4/experiments/` — Generator Matching training, conditional (`cnof_conditional.py`,
  `tracelet_conditional.py`), value-guided sampling, calibrated/parallel sampling.
- `src/compose_v4/eval/` — `molecular_quality.py` (V/U/N, descriptor Wassersteins, FCD).
- `src/compose_v4/data/` — CNOF corpus loading/splits.
- `scripts/` — experiment drivers. The conditional controller is
  `griddd_value_guided_smc_controller.py` (hard scaffold/similarity/required-/forbidden-SMARTS
  fibers, arbitrary state→float objective, population dump, anytime traces). Paper results:
  `tier1_pathwise_safety.py`, `physchem_box.py`, `e0_toy_h_exactness.py`,
  `doob_guidance_ground_truth.py`, `make_paper_figures.py`.
- `paper_iclr_stochastic_rewriting/` — the paper (`main.tex`, `latexmk -pdf`).
- `diagnostics/` — committed result JSONs; `modal_apps/` — cloud apps; `tests/` — pytest.

Reproducible inputs: `configs/benchmarks/cnof_leads.json`. Base checkpoint (Lineage B) is on
the Modal artifact volume, not in git — see README.

## Read before large unconditional runs

`docs/HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md` — the base is a step-1,000 preview with
an open small-ring defect; the final-backbone decision is not settled.

## Context (auto-loaded)

@.claude/context/conventions.md
@.claude/context/glossary.md
@.claude/context/learnings.md
