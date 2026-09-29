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
3. **[`experiments/t4/`](experiments/t4/)** and
   **[`experiments/pmo/`](experiments/pmo/)** — the two benchmark task guides.
   Each states what the task asks, how to regenerate its reported numbers
   **without cloud credentials or oracle calls**, its exact input manifest, its
   environment, and what the result does not support. T4's table is frozen;
   PMO is development evidence with no frozen table. **The two tasks run
   different pinned chemistry kernels** (rdkit 2024.3.5 vs 2023.9.6).
4. **[`experiments/INDEX.md`](experiments/INDEX.md)** — every runnable
   experiment and its source-derived command. Regenerate it with
   `tools/gen_experiment_index.py` after adding an entrypoint.
5. **[`experiments/region_resampling/`](experiments/region_resampling/)** — a
   campaign manifest, with its preregistered gates, inputs, outputs and known
   limitations.
6. **[`docs/INDEX.md`](docs/INDEX.md)** — historical document census, classified CURRENT /
   SUPERSEDED / HISTORICAL.

## Verifying a result without running anything

```bash
python3 tools/reproduce_t4_table.py                        # frozen T4 table
PYTHONPATH=src python3 tools/reproduce_pmo_tables.py       # PMO, from charged receipts
python3 tools/verify_experiment_inputs.py                  # inputs match their manifests
python3 tools/preservation_inventory.py                    # what git does NOT protect
```

None of these touch the network, an oracle, or a credential.

## Layout

```
src/compose_v4/     library. importable, no experiment logic
modal_apps/         experiment entrypoints; obey each experiment's launch contract
experiments/        what each experiment asks, how to rerun it, what it produced
tools/              repo tooling (audit, index generation, dataset rebuild)
tests/              pytest; mirrors src/
docs/               notes, plans, indices, handoffs
diagnostics/        result artifacts written by runs
archive/            superseded material, kept rather than deleted
```

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

## Notes

- `README_NAVIGATION.md` predates this file and describes an earlier framing;
  it is kept because several documents link to it. Prefer this README.
- `src/compose_v4/experiments/production_successor_kernel.py` must not be
  edited: it fixes `process_identity_sha256`, which every artifact is keyed on.
