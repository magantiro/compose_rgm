# COMPOSE

COMPOSE treats molecular design as **control of a frozen learned process**. A
generator `R_θ` is learned once over executable chemical rewrites; a controller
then steers trajectories without changing it, as a KL-regularised change of path
measure implemented through finite-horizon Doob/committor values. Every
transformation is realised as primitive executable steps through complete,
chemically valid molecules — never as an endpoint that a search must later
justify.

Three layers, kept separate on purpose:

| Layer | Question | Where |
|---|---|---|
| Executor | what is *possible* | `src/compose_v4/rewrite`, `src/compose_v4/chem` |
| `R_θ` | what is *plausible* | `src/compose_v4/gm`, `src/compose_v4/model` |
| Controller | what is *purposeful* | `src/compose_v4/control`, `src/compose_v4/policy` |

## Start here

1. **[`docs/START_HERE_ICLR.md`](docs/START_HERE_ICLR.md)**: authoritative
   controller status, current T4 and PMO evidence, branch layout, and the next
   scientific decision.
2. **[`docs/PAPER_TO_CURRENT_CODE.md`](docs/PAPER_TO_CURRENT_CODE.md)**:
   submitted-paper code and evidence versus post-submission controller
   extensions, plus a fresh-laptop first-hour path.
3. **[`experiments/INDEX.md`](experiments/INDEX.md)** — every runnable
   experiment and its source-derived command. Regenerate it with
   `tools/gen_experiment_index.py` after adding an entrypoint.
4. **[`experiments/region_resampling/`](experiments/region_resampling/)** — the
   current campaign, with its preregistered gates, inputs, outputs and known
   limitations.
5. **[`docs/INDEX.md`](docs/INDEX.md)** — historical document census, classified CURRENT /
   SUPERSEDED / HISTORICAL.

## Layout

```
src/compose_v4/     library. importable, no experiment logic
modal_apps/         experiment entrypoints; obey each experiment's launch contract
experiments/        what each experiment asks, how to rerun it, what it produced
tools/              repo tooling (audit, index generation, dataset rebuild)
tests/              pytest; mirrors src/
docs/               notes, plans, indices, handoffs
diagnostics/        result artifacts written by runs
repro/              what it takes to re-run a published number
archive/            superseded material, kept rather than deleted
```

Two entries do not match their description, and both are load-bearing rather
than untidy. 21 of the 302 modules under `src/compose_v4/experiments/` are
imported by 37 library-core modules, so they are library code filed as
experiments. `diagnostics/` additionally holds sealed source capsules:
byte-exact frozen copies of `src/compose_v4` recording the source that two
scored campaigns executed.

## Before you change anything

This repository is content-addressed. Source files are hashed into launch
contracts and into the Process-V2 identity, so **editing one byte of a pinned
file makes every contract pinning it refuse to load**, and moving one is worse,
because the pin then addresses nothing while still reading as verified.

The constraint is larger than the per-file pin list. Three fingerprints hash
whole DIRECTORIES, so a file that is individually unpinned is still covered and
**adding** a file moves the identity exactly as editing one does:

| fingerprint | covers |
|---|---|
| `train_tracelet_gm.py::_source_fingerprint` | every `.py`/`.json` under `src/`, `scripts/`, `recipes/` |
| `t4_objective_reset_app.py::main` | every `.py` under `src/` |
| `semantic_p50_successor_cache_implementation_sha256` | every `.py` under `src/compose_v4` |

So `src/`, `scripts/` and `recipes/` are read-only **in aggregate**, not merely
per file. Measured: a whitespace-only edit to any file under `scripts/` moves
the training run identity, and adding a module under `src/compose_v4` makes
every existing P50 successor-cache plan raise.

Check one path before touching it:

```bash
python3 tools/repo_pinned_file_set.py --check PATH   # is it the subject of a pin?
python3 tools/repo_audit.py --check PATH             # who references it?
```

[`repro/MIGRATION_CONTRACT.md`](repro/MIGRATION_CONTRACT.md) states the full
rules. The short version: never delete (retire to `archive/` with provenance),
never edit a historical artifact in place, and never re-pin a contract to make a
change succeed.

Three chemistry kernels are in play and **the laptop `.venv` is production for
nothing**: PMO is rdkit 2023.09.6, T4 and editing are rdkit 2024.03.5, the
`.venv` is 2026.03.6. See [`repro/environments_v1.json`](repro/environments_v1.json).

## Running an experiment

Read the experiment contract before choosing a launcher. For the long T4
driver, use `modal deploy modal_apps/genmol_t4_opt_app.py` followed by the
durable entrypoint in `tools/t4_launch.py`; do not substitute an ephemeral
`modal run --detach` invocation.

Four facts that are easy to learn the hard way:

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
- **Artifacts live on the `compose-v4-artifacts` volume**, not in the repo.

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

Preservation tooling, used before and after any structural change:

```bash
python3 tools/repo_pinned_file_set.py --out diagnostics/repo_hygiene/pinned_file_set_v1.json
python3 tools/repo_capability_baseline.py --out AFTER.json --compare BASELINE.json
python3 tools/repo_entry_point_check.py  --out AFTER.json --compare BASELINE.json
python3 tools/repo_pin_resolution_audit.py --out diagnostics/repo_hygiene/pin_resolution_v1.json
python3 tools/repro_index.py --out-dir repro
python3 tools/repro_gate.py --base 061ead93   # every line must read zero
```

A public-symbol superset is necessary and **not sufficient**: an import can
resolve while the mechanism behind it is reached by no production caller, which
has happened six times here. That is why the entry-point, test and
artifact-to-run checks are separate gates rather than corroboration.

## Notes

- `README_NAVIGATION.md` predates this file and describes an earlier framing;
  it is kept because several documents link to it. Prefer this README.
- `src/compose_v4/experiments/production_successor_kernel.py` must not be
  edited: it fixes `process_identity_sha256`, which every artifact is keyed on.
  It is one of 542 pinned modules in `src/compose_v4`; see the section above for
  why the other 53 are effectively pinned too.
