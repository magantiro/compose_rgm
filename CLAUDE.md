# COMPOSE (RGM) — agent context

**Rewrite Generator Matching for validity-preserving molecular generation.**
A stochastic process over molecular graphs whose committed non-null states are
complete, supported, connected molecules and whose transitions are executable
chemical rewrites. The reference process is learned without a downstream task;
task information enters through a separate controller at inference. The
submitted workshop package, current controller work, and historical manuscript
variants are mapped in `docs/PAPER_TO_CURRENT_CODE.md`.

> Not to be confused with the sibling repo `KoshaTx/compose` (a discrete-*diffusion*
> operator-editing framework). Same name lineage, different generative process.
> This repo is the **RGM / CTMC** line (`KoshaTx/compose_rgm`).

This file loads automatically every session. Keep it short and high-signal —
detail lives in the imported context files below and in scoped `CLAUDE.md`s.

## Start here

1. **[`AGENTS.md`](AGENTS.md)** is the repository-wide scientific and execution
   contract. A task-specific frozen contract may narrow it.
2. **[`docs/START_HERE_ICLR.md`](docs/START_HERE_ICLR.md)** is the authoritative
   controller status, current T4 and PMO evidence, and next decision.
3. **[`docs/PAPER_TO_CURRENT_CODE.md`](docs/PAPER_TO_CURRENT_CODE.md)** maps the
   submitted paper implementation to post-submission controller extensions.
4. **[`docs/CONTROLLER_LIVE.md`](docs/CONTROLLER_LIVE.md)** is the chronological
   evidence ledger. It is not the first-read status page.

Do not infer authority from filenames such as `CURRENT`, `PLAN`, or `HANDOFF`.
This repository preserves superseded plans and preregistrations because they are
part of the scientific record. Follow the reading order in `START_HERE_ICLR.md`
and the current task's explicit acceptance criteria.


## How we work

1. **Context auto-loads.** Read `AGENTS.md` and the start page even when this file
   has already been loaded automatically.
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
pytest tests/                    # full suite; use focused tests while iterating
pytest tests/test_e0_toy_h_exactness.py -q   # one file
ruff check .                     # lint (line-length 100, target py310)
python3 tools/preflight.py       # required before current controller launches
```

Cloud training / rollout evaluation run on **Modal** (`modal_apps/`), against the
`compose-v4-artifacts` volume. Never launch a Modal job before the task's current
preflight and authorization contract is green.
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
- `src/compose_v4/experiments/` — Generator Matching training, value-guided
  sampling, scientific drivers, and artifact reductions.
- `src/compose_v4/control/` — current region, option, continuation, frontier,
  and persistent-controller library code.
- `src/compose_v4/eval/` — `molecular_quality.py` (V/U/N, descriptor Wassersteins, FCD).
- `src/compose_v4/data/` — corpus loading/splits: `cnof.py` (de-novo CNOF-neutral) and `organic_corpus.py`
  (the LOCKED broad-organic B-edit scope + shared loader + census; see learnings 2026-07-27).
- `scripts/` — local experiment drivers, readers, audits, and figure builders.
- `paper_gem_neurips2026/` — submitted workshop package.
- `paper_iclr2027/` — full manuscript source from which that package was
  abridged. See the paper-to-current map before editing any manuscript.
- `diagnostics/` — committed result JSONs; `modal_apps/` — cloud apps; `tests/` — pytest.

Reproducible inputs: `configs/benchmarks/cnof_leads.json`. Base checkpoint (Lineage B) is on
the Modal artifact volume, not in git — see README.

## Read before large unconditional runs

`docs/HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md` — the base is a step-1,000 preview with
an open small-ring defect; the final-backbone decision is not settled.

## Context (auto-loaded)

@.claude/context/mission.md
@.claude/context/conventions.md
@.claude/context/glossary.md
@.claude/context/learnings.md
