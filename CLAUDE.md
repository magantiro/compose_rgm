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

## Plan of record — read this before any other document

- **[`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)** — the ONLY current
  experiment plan.
- **[`docs/PROJECT_BOARD.md`](docs/PROJECT_BOARD.md)** — the durable task board.
  The in-session task tool has been wiped twice; this file is what persists.
- **[`docs/DECISION_LOG.md`](docs/DECISION_LOG.md)** — what is established, and
  what was tried and refuted. Read it before re-running an experiment.

**Everything else in `docs/` and every `paper*/` directory is HISTORY** unless
it is linked from one of those three. `docs/` holds 84 markdown files accumulated
across several earlier framings; a dozen are stamped ARCHIVED, but absence of a
banner does not mean a document is current. Default to history.

The manuscript's existing experimental section describes a materially different
model and must not be executed or cited.


## How we work

1. **Context auto-loads.** The `@`-imported files at the bottom are in your context now.
2. **Capture durable learnings** in `.claude/context/learnings.md` when you hit a non-obvious
   gotcha or make a design call. Commit it — that's how it spreads on the next pull.
3. **`.claude/` is code.** Edit the context files in the same PR as the change that makes them
   true; keep them terse. Stale context is worse than none.

## Hold the goal

Before any remote run, and before any optimisation, state in one sentence **what question the
output answers**. Then check the thing you are about to launch actually answers it.

This exists because of a concrete failure. The goal was *a balanced training set, to see whether
the model learns*. Over several hours that turned into slice sizing, worker timeouts, progress
logging, symlink resolution and intra-container parallelism — every one a real defect, none of
them the goal — and the run that finally launched sampled **one lane, two of eight families**,
which could not answer the question at all. Nobody chose that; it drifted there one reasonable
step at a time.

- **A fix on the path is not the path.** Fixing a real blocker is right; three fixes deep, re-read
  the goal before the fourth.
- **Optimising throughput for a job that produces the wrong artifact is worse than not optimising.**
  Check *what* is being produced before making it faster or cheaper.
- **Say the cost of the detour out loud.** "This is a 20-minute detour to fix X, then back to the
  balanced set" keeps it bounded and lets the user veto it.
- This is **not** an argument against parallel work or against fixing things you notice. Notice
  them, say them, park them. The failure mode is silently *substituting* them for the goal.

## Scratchpad is for things you expect to throw away

Anything carrying a **theorem, an invariant, or a reusable interface** is written into `src/` with
tests *when it is written* — not promoted later. Scratchpad is for probes whose conclusions you
expect to discard.

The test: *would the paper cite this, or would another experiment import it?* If yes, it is repo
code. Verified numbers produced by an unimportable script are numbers nobody can reproduce.

This exists because of a concrete failure. The exact bridge control, the path-space sampler and the
evaluation-semantics preflight were all built in scratchpad and validated there — one of them was
described in writing as "the preflight every COMPOSE evaluation must pass" while living in a temp
directory where nothing could import it. Two of them were load-bearing for the paper. Promoting
them afterwards worked, but the tests were retrofitted rather than written alongside, which is
exactly when a guard is weakest.

Exploratory training runs, one-off diagnostics and dead ends belong in scratchpad and should stay
there — several of them encode assumptions since disproved.

## Fail closed on catalog drift

Any **claim-bearing** evaluation or **corpus compilation** must abort if the reconstructed
RingCore catalog fingerprint differs from the frozen production value `639ff6078c32d43c`.
`neutralize_catalog_drift()` — which catches the drift error and overwrites the EXPECTED
fingerprint with whatever the local environment produced — is a scratch/dev probe helper only.
It must never appear in code that compiles corpus data or produces a number the paper cites.

The production environment is pinned by the Modal image: **python 3.11, torch 2.4.0,
numpy 1.26.4, scipy 1.13.1, networkx 3.3, rdkit 2024.3.5**. Reproduce it locally rather than
suppressing the check; a matching venv reconstructs the fingerprint on its own.

This is not hypothetical. A local venv on rdkit 2026.03.4 drifted to `82fd910c`, and a 50-state
parity gate against the pinned versions found 2 states whose canonical successor inventory
differed — an aromaticity-perception change, same molecule, different canonical key. Structure
was identical (same mark counts, families, alias multiplicities), so nothing crashed. Compiling
under the wrong environment would have produced a split-brain corpus where half the canonical
keys came from a different chemistry kernel, and the fingerprint check is the only thing that
catches it.

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
