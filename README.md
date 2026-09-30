# COMPOSE

COMPOSE represents molecular design through executable rewrites between complete,
supported molecular graphs. A learned reference law assigns preferences to legal
edits in reference-guided settings. Task-specific controllers can use that law,
structural constraints, or evaluated endpoint feedback to construct and select
transformations. The reference-guided editing controller uses finite-horizon
future values; the optimization controllers are distinct. Checkpoint sharing and
numerical reference use are experiment-specific, not universal across benchmarks.

## Choose a path

| Goal | Start here | What the checkout currently supports |
| --- | --- | --- |
| Understand the method | [The three layers below](#the-three-layers) | Read the executor, reference, and controller as distinct components |
| Check the submitted paper's evidence | [`experiments/paper/`](experiments/paper/) | Verify the exact PDF and historical result-artifact identities; this is not yet full benchmark replay |
| Reproduce fragment tables | [`experiments/fragments/`](experiments/fragments/) | Hash-check and reduce saved rows; isolated generation requires declared external assets |
| Check the submitted PMO A/B table | [`experiments/pmo/`](experiments/pmo/) | Recompute the reported 14-pair subset from its saved reduction; raw oracle receipts remain external |
| Trace the submitted T4 table | [`experiments/t4/`](experiments/t4/) | Inspect the frozen row source and its unresolved table reconciliation; fresh docking assets are external |
| Develop or test the code | [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Install the pinned core environment and run offline tests |
| Locate a historical experiment | [`experiments/INDEX.md`](experiments/INDEX.md) | Find its entry point, then read that experiment's contract before any launch |

The published T4 and QED result paths still need task-specific offline
reductions and external-asset access checks in this integration branch. The
PMO A/B saved-data check does not replay raw oracle receipts. T4 has an
unresolved paper-row provenance question. Do not interpret a passing
source-identity check as reproduction of a paper table.

## The three layers

| Layer | Question | Where |
|---|---|---|
| Executor | what is *possible* | `src/compose_v4/rewrite`, `src/compose_v4/chem` |
| `R_θ` | what is *plausible* | `src/compose_v4/gm`, `src/compose_v4/model` |
| Controller | what is *purposeful* | `src/compose_v4/control`, `src/compose_v4/policy` |

## Project history and navigation

For installation and local checks, use
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) and the pinned environments under
[`requirements/`](requirements/).

1. **[`docs/START_HERE_ICLR.md`](docs/START_HERE_ICLR.md)**: a dated controller
   campaign handoff. Check its as-of date and the experiment-specific contract
   before treating any operational status as current.
2. **[`docs/PAPER_TO_CURRENT_CODE.md`](docs/PAPER_TO_CURRENT_CODE.md)**:
   the historical workshop-era code map. It is not yet the reproduction guide
   for the submitted ICLR 2027 PMO/T4 results; see the
   [release-readiness audit](docs/REPOSITORY_READINESS_20260929.md).
3. **[`experiments/INDEX.md`](experiments/INDEX.md)** — every runnable
   experiment and its source-derived command. Regenerate it with
   `tools/gen_experiment_index.py` after adding an entrypoint.
4. **[`experiments/region_resampling/`](experiments/region_resampling/)** — a
   historical campaign with its preregistered gates, inputs, outputs and known
   limitations; it is not the submitted PMO/T4 reproduction path.
5. **[`docs/INDEX.md`](docs/INDEX.md)** — historical document census, classified CURRENT /
   SUPERSEDED / HISTORICAL.

## Layout

```
src/compose_v4/     importable domain library and experiment adapters
modal_apps/         experiment entrypoints; obey each experiment's launch contract
experiments/        what each experiment asks, how to rerun it, what it produced
tools/              repo tooling (audit, index generation, dataset rebuild)
tests/              pytest; mirrors src/
docs/               notes, plans, indices, handoffs
diagnostics/        result artifacts written by runs
archive/            superseded material, kept rather than deleted
```

Shared library code is organized by responsibility; experiment instructions and
evidence maps are organized by task. Frozen paths are preserved during migration.

## Running an experiment

Read the experiment contract and its result provenance before choosing a
launcher. The historical T4 launcher in `tools/t4_launch.py` is not a generic
entry point for every T4 result; similarly, PMO and fragment evaluations have
separate task-specific recipes. An offline import or table check does not
authorize a new scored campaign.

For that historical deployed-app workflow, four facts are easy to learn the hard way:

- **Use the campaign's durable deployed-app launcher for long work.** The T4
  launcher persists a function-call ID and volume namespace so the run survives
  the local client.
- **Persist per unit.** A `.map` that returns results only to the client loses
  everything if one unit times out. Write each unit to the volume and harvest
  from there.
- **The image ships `src/` and `configs/` only**, plus the single file
  `modal_apps/run_process_v2_p50_app.py`. A container cannot import any other
  `modal_apps` module, so shared helpers belong in `src/` or in the app itself.
  That file computes `ROOT = parents[1]` and cannot move.
- **That workflow's artifacts live on the `compose-v4-artifacts` volume**, not in
  the repo. Other experiments use other volumes and may require a different
  Modal profile.

## Tools

```bash
python3 tools/repo_audit.py --classify        # what each tracked file is
python3 tools/repo_audit.py --check PATH      # who references PATH (run before moving it)
python3 tools/gen_experiment_index.py --write # regenerate experiments/INDEX.md
python3 tools/build_dude_pool.py              # rebuild the DUD-E holdout pool
```

`--check` exists because this tree hides references: a checkpoint under `docs/`
is load-bearing when an app mounts it with `add_local_file`, though nothing
imports it. Run it before moving any file.

## Notes

- `README_NAVIGATION.md` predates this file and describes an earlier framing;
  it is kept because several documents link to it. Prefer this README.
- `src/compose_v4/experiments/production_successor_kernel.py` must not be
  edited: it fixes `process_identity_sha256`, which every artifact is keyed on.
