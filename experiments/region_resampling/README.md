# Region resampling — variable-scope executable rewriting

**Status: CLOSED — structural execution QUALIFIED.**

    R_M        256 attempts   2 establishment    0 complete    0/16 regions
    R_M·h_phi  256 attempts 142 establishment   26 complete    7/16 regions

16 known-reachable held-out regions, molecules disjoint from training, 16
particles per (region, arm), matched horizon / ceiling / regions / seeds. N came
from the measured handoff mass via 1-(1-m)^N and kappa=1.0 was declared, both
before the run. This is CONDITIONAL execution efficiency -- regions were chosen
because a rewrite exists -- not a coverage rate.

The remaining open item is the general local→global gate, not more splitting
work.

The question: molecular design as control of a frozen learned process, where a
transformation is a *region* rewrite — a connected mutable region `M`, a
preserved context `C = x \ M`, and a boundary `∂M` — realised as primitive
executable steps through complete valid molecules.

---

## Reproducing, in order

Every command needs `--detach` if it runs more than a few minutes; a laptop
suspend kills an attached run and the client-side merge with it.

### 1. Correctness gate — the law cache must be exact

```
modal run modal_apps/committor_bellman_app.py::verify_law --n 10
```

Compares the cached marked law against a fresh uncached enumeration on the same
state, bitwise on rule names, actions and probabilities. **Must print
`GATE 1 PASS`.** The result also reports how many compared states sat on a
canonical key covering several slot layouts — if that count is 0 the suite
passed vacuously and proves nothing.

Last run: 327 states, 177,360 marks, 0 mismatches, 60 states on colliding keys.

### 2. Mechanism requalification

```
modal run modal_apps/region_sentinels_app.py::requalify --trials 4
modal run modal_apps/region_sentinels_app.py::report        # merge from the volume
```

Does a structurally directed proposal still establish handoff under the
corrected kernel, where the undirected base kernel does not? Not a 12/12
replication — explicit depth-3 lookahead is far too expensive to deploy, so
this only checks the path still exists.

Last run: directed 1/1, undirected 0/12 (`no_connectivity_preserving_handoff`),
saturated undirected 0/12, pendant directed 12/12.

### 3. Screen, collect, fit

```
modal run modal_apps/committor_bellman_app.py::screen --per-molecule 30 --batch 4
modal run modal_apps/committor_bellman_app.py::mixture --neg-per-pos 2 --sat-zero 12
modal run modal_apps/committor_fit_entry.py::main            # gate, then fit
```

`screen` finds which regions are completable within B=5 primitive edits.
`mixture` composes the replay set explicitly — every reachable region as a
positive tube, unreachable regions matched on size *and* saturation as hard
negatives, plus a spread saturated block for genuine zero mass. Intermediate
values are not sampled: they arrive as the stored predecessors and shells, whose
`R_M` backups sit strictly between 0 and 1.

Gate: ≥3 distinct molecules with positive tubes, ≥8 terminal states, ≥1 negative
region per positive molecule.

Last run: 70 units, 7 positive molecules, 337 terminals, 2844 states.

### 4. Held-out pool and the execution race

```
python3 tools/build_dude_pool.py                              # rebuild the pool
modal run modal_apps/committor_bellman_app.py::screen_dude \
    --per-molecule 30 --n-molecules 16 --offset 0 --subdir screen_dude2
modal run modal_apps/committor_bellman_app.py::profile_attempt --n 2
modal run modal_apps/committor_bellman_app.py::race --seeds 2 --max-calls 4150
```

`screen_dude` **must** use `--per-molecule 30` to match the training screen. A
different value samples different regions per molecule and the rates are then
not comparable — that mistake produced a 1/63 figure that looked like a finding
and was not.

`profile_attempt` fixes the per-attempt call ceiling. Freeze it, and never tune
it on whether either arm succeeds.

---

## Preregistered gates

| Stage | Gate | Fixed before |
|---|---|---|
| Law cache | 0 mismatches, >0 states on colliding keys | any timing result |
| Collection | ≥3 positive molecules, ≥8 terminals | any fit |
| Held-out screen | ≥10 reachable across ≥5 molecules | any race result |
| Race | stop on clear separation, else +2 seeds, cap 6/region | any attempt |

## Inputs

| File | What |
|---|---|
| `docs/GENMOL_T4_DEV_SEEDS.json` | 22 DUD-E dev seeds. `[6:22]` train, `[:6]` sentinels. |
| `docs/DUDE_HOLDOUT_POOL.json` | 743 fresh DUD-E actives, canonical-disjoint, heavy-atom matched. |
| `/artifacts/region_committor/committor_bellman_v1.pt` | Fitted committor (Modal volume). |

## Outputs

All under the `compose-v4-artifacts` volume, `/region_committor/`:
`screen/`, `collect/`, `arms/`, `race/`, plus the checkpoint. Local mirrors land
in `diagnostics/`. Every unit persists independently, so a dead client never
destroys finished work.

## The bug that hid the result

Every guided-vs-base zero before 2026-09-02 was `R_M` against `R_M` plus
overhead, and none of them is evidence about the committor. `adaptive_tilt`
bisects a temperature for an ABSOLUTE ESS target and returns the untilted base
whenever the base kernel's own ESS already sits below it. Measured base ESS was
0.061 against a 0.3 target, so the early return fired at all 79 steps, T pinned
to its bound, and `h^(1/T) ≈ 1` collapsed the guided proposal onto the base.

The committor was never the problem: `h_max_progress / h_max_all = 1.000`, so
the progress successor was top-ranked at essentially every step. `kl_tilt`
replaces the absolute target with a base-relative trust region and raised
progress mass from 0.00004 to 0.09359 at kappa=1 from the SAME fitted model.

**Check the reported tilt temperature before trusting any guided-vs-base
result.** If it sits at the bisection bound, guidance was off.

## What this phase established

- Useful splitting paths exist and survive the corrected kernel (22/22 re-screen).
- Naive `R_M` misses them: 0/12, reproduced post-correction.
- Explicit lookahead is unusable — hours per trial once it can no longer take
  false cache hits.
- Budget is not the limiter at fixed breadth: B ∈ {5,10,20} plus a structural
  allowance gave **zero** additional completions, every success landing at step 4.
  The failure is search *direction*, not search *length*.

## Known limitations

- "Reachable" means *completable within B primitive edits*, never "rewriteable".
- The budget probe varied depth at fixed beam 40, so it says nothing about
  whether a wider search finds longer rewrites.
- Saturated splitting is rare at short horizon on arbitrary regions. It is **not**
  an operator-set impossibility: a `cycle_open` → `cycle_close` path was exhibited.
