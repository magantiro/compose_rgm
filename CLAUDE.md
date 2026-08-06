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

## Reasoning discipline

Separate what is **measured**, what is **inferred**, and what is **assumed**, and say which.
Most bad calls here came from stating an inference in the voice of a measurement.

- **Check why a thing exists before recommending its removal.** A gate, threshold or
  exclusion usually encodes a past failure. Read the decision record first. The per-cell
  capability gates exist because a 16,000-step run destroyed two operator families while
  aggregate loss improved (`docs/HANDOFF_RINGCORE_V1_POSTRUN.md`); removing them
  reintroduces exactly that blindness.
- **Do not derive importance from availability.** "This operator matters because we have
  data for it" is circular. Derive requirements from the experiments, then check whether
  the corpus supports them. A claim-critical operator that is thin is a corpus question,
  not a reason to demote the claim.
- **Measure the substrate before calling a cell under-mined.** A rare cell whose corpus
  share matches how often its substrate occurs is faithful, not defective, and
  reweighting it would teach wrong rates. Compare share against substrate, not against
  other cells.
- **A projection is not a measurement, and a component is not a path.** State which
  components a number covers, and how many samples it rests on.
- **Terminology carries claims.** Do not say "rebuild" for an incremental append, or
  "capability" for an audit label, or "operator" for one context of an operator. Loose
  words silently reintroduce conclusions already disproven.
- **Answer "is X fine?" by naming what is known and what is not** — capacity, exposure,
  competitive pressure and protection are different questions with different evidence.
  "Unknown, and here is what would settle it" is a complete answer; a confident guess is
  not.
- **When you correct yourself, say so plainly and once**, then continue.

## Commands

```bash
uv sync                          # .venv + deps (or: pip install -e ".[dev]"); then activate .venv
export PYTHONPATH=src:scripts    # both for scripts; `src` also lets parallel-rollout worker subprocesses find compose_v4
export KMP_DUPLICATE_LIB_OK=TRUE # macOS OpenMP guard (silences the abort)
export OMP_NUM_THREADS=1         # REQUIRED locally: torch's bundled OpenMP + brew's collide → suite segfaults without it (see learnings)
pytest tests/                    # full suite (494 tests, pythonpath=["src"] -> `from compose_v4.…`)
pytest tests/test_e0_toy_h_exactness.py -q   # one file
ruff check .                     # lint (line-length 100, target py310)
python scripts/prelaunch_gate.py --corpus <name>  # pre-Modal-launch gate: tests+ruff+clean-tree+corpus+hashes
```

Cloud training / rollout evaluation run on **Modal** (`modal_apps/`), against the
`compose-v4-artifacts` volume. Never launch a Modal job before `prelaunch_gate.py` is green.
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
- Feature branches → PR to `main`; both gates (`pytest tests/` + `scripts/prelaunch_gate.py`)
  green before merge.

## Map

- `src/compose_v4/chem/` — `MolecularGraph`, padded state, validity/canonicalization.
- `src/compose_v4/rewrite/` — the rewrite kernel + legal-event **fibers** (`fiber.py`,
  `factorized_fiber.py`, `tracelet_fiber.py`, `ring_system_fiber.py`); `kernel.py` = executor.
- `src/compose_v4/experiments/` — Generator Matching training, conditional (`cnof_conditional.py`,
  `tracelet_conditional.py`), value-guided sampling, calibrated/parallel sampling.
- `src/compose_v4/eval/` — `molecular_quality.py` (V/U/N, descriptor Wassersteins, FCD).
- `src/compose_v4/data/` — corpus loading/splits: `cnof.py` (de-novo CNOF-neutral) and `organic_corpus.py`
  (the LOCKED broad-organic B-edit scope + shared loader + census; see learnings 2026-07-27).
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
