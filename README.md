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

1. **[`experiments/INDEX.md`](experiments/INDEX.md)** — every runnable
   experiment: 193 apps, 266 entrypoints, with the exact command for each.
   Generated from source by `tools/gen_experiment_index.py`, so it cannot drift.
2. **[`experiments/region_resampling/`](experiments/region_resampling/)** — the
   current campaign, with its preregistered gates, inputs, outputs and known
   limitations.
3. **[`docs/INDEX.md`](docs/INDEX.md)** — all documents, classified CURRENT /
   SUPERSEDED / HISTORICAL.

## Layout

```
src/compose_v4/     library. importable, no experiment logic
modal_apps/         experiment entrypoints, run with `modal run`
experiments/        what each experiment asks, how to rerun it, what it produced
tools/              repo tooling (audit, index generation, dataset rebuild)
tests/              pytest; mirrors src/
docs/               notes, plans, indices, handoffs
diagnostics/        result artifacts written by runs
archive/            superseded material, kept rather than deleted
```

## Running an experiment

```bash
modal run modal_apps/<app>.py::<entrypoint> [--flags]
```

Four facts that are easy to learn the hard way:

- **`--detach` for anything long.** An attached run dies when the laptop
  suspends, and it takes the client-side merge with it.
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
