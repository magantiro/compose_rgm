# COMPOSE ICLR controller: start here

This is the authoritative collaborator handoff for the controller campaign as
of 2026-09-12. Read this page before the chronological logs. The active
integration branch is `compose-iclr`; the Modal session name is
`compose_iclr`.

## Project identity

- Scientific problem: control a frozen stochastic process over executable
  molecular rewrites so that it finds high-value edits efficiently.
- Primary output: complete supported molecular states reached through exact
  primitive transitions, together with their WHERE, WHAT, and HOW ancestry.
- Central controller claim under development: local-to-global region choice,
  option-level transformation choice, and primitive execution can be combined
  without narrowing COMPOSE's molecular support or teleporting to endpoints.
- Validation: source-conditioned T4 lead optimization and PMO optimization,
  with matched endpoint constraints, explicit oracle accounting, candidate
  locks, and proposal-time reporting.
- Causal baselines: the same generator with post-hoc ranking, reference-only
  option selection, and frozen-policy controls. External numbers are
  comparators, not causal ablations.
- Declared support: the production persistent-slot molecular graph support and
  exact executor recorded in `AGENTS.md`. It is not a fragment vocabulary and
  is not universally expressive.

## Current outcome

No scientific job is active. An attempted duplicate T4 launch on 2026-09-12
was stopped after the repository inventory revealed that the exact comparison
had already completed. The duplicate spent zero oracle calls. Do not relaunch
it.

The completed T4 result is the current primary evidence:

| PARP1 seed0, delta 0.4 | Post-hoc | In-loop |
| --- | ---: | ---: |
| Shared warm archive | 51 calls, best -9.7 | 51 calls, best -9.7 |
| New docking attempts | 4 | 8 |
| Eligible endpoints | 4 | 8 |
| Best new score | -9.9 | **-10.0** |
| Distinct docked bundles | 4 | 8 |
| Proposal time, slowest lineage | 793.1 s | 1,065.1 s |

Both best candidates were valid completed six-member pendant-ring options with
cycle-rank and ring-system deltas of +1. The in-loop arm therefore improved
eligible proposal yield and found constructive ring chemistry. It did not
produce a competitive result against the reported GenMol value of -10.6 or the
three released InVirtuoGen values of -13.5, -13.6, and -13.6. This is one
inspected development round, not evidence of generalization or superiority.

The decisive negative finding is ranking: the frozen endpoint predictor ranked
the observed best ring last in both docked batches. WHERE and WHAT received no
task-value contrast; only eight HOW decisions were task-guided. The immediate
bottleneck is therefore delayed value and allocation across complete options,
not basic ring executability.

Authoritative T4 artifacts:

- `diagnostics/t4_frontier_compare/result.json`: machine-readable verified
  review and hashes.
- `diagnostics/t4_frontier_compare/README.md`: concise interpretation.
- `docs/T4_FRONTIER_COMPARISON.md`: prospective contract, recovery history,
  and completed outcome.
- Source run `c663e7ac0532b0664a40b02de1d53868a3c0bf69726f2957c68fc26777cfa843`
  under `/t4_frontier_compare/` on `compose-v4-artifacts`.
- Scientific source revision `2f535a96d734a3fe9f7d6616ad55aea43d0505ac`;
  verified review revision `bcb82254d256cfd047a3b9a84e34e5c1297a389c`.

## PMO status

PMO is secondary and preserved, not discarded. The latest paid locked-pool
development assay improved the observed Perindopril MPO champion from
`0.6835298931` to `0.6948083338` with 95 calls. All three parent improvements
came from `actor_top`; uniform and largest-release strata produced none. This
is a useful local ranking result, not a reproduced full-controller win or a
matched PMO benchmark result.

The option-value gate abstained because the compatible stored continuations had
no positive identified targets. The prepared next data-collection unit is a
balanced, three-option continuation bank over eight exact parents and sixteen
streams. It is committed but not authorized for remote proposal generation or
new oracle calls. See `docs/PMO_OPTION_CONTROLLER_BANK.md` and
`diagnostics/pmo_option_controller_bank/prepared.json`.

## Next scientific decision

The next T4 controller should evaluate and allocate complete semi-Markov option
outcomes, while preserving the qualified WHERE distribution, the balanced WHAT
exploration floor, the frozen reference process, and exact HOW execution. It
should not reward ring count, narrow support to rings, tune macro weights on
the observed docking outcomes, or call an endpoint predictor a future-value
model.

The smallest useful next experiment is a prospective paired test of
option-boundary delayed value and acquisition against the current in-loop
controller. A positive result must improve eligible yield or best observed
score at a declared oracle cap while retaining option and molecular diversity.
A null stops that value construction; it does not justify more rounds of the
same planner. No next paid T4 round is currently authorized.

## Code map

| Concern | Primary files |
| --- | --- |
| WHERE region law | `src/compose_v4/control/region.py`, `region_selector.py` |
| WHAT option law | `option_selector.py`, `molecular_task_search.py` |
| HOW continuation | `option_continuation.py`, ring/carbonyl/replacement option modules |
| Resumable T4 planner | `frontier_search.py`, `t4_frontier_search.py` |
| Paired T4 driver | `t4_frontier_compare.py`, `t4_frontier_audit.py` |
| Persistent option runtime | `option_controller.py`, `option_controller_runtime.py` |
| PMO continuation bank | `pmo_option_controller_bank.py` |
| Remote entrypoints | `modal_apps/genmol_t4_opt_app.py`, `modal_apps/pmo_option_controller_bank_app.py` |

## Working and launch discipline

1. Develop from `compose-iclr`. The former dirty main checkout was preserved
   losslessly on `legacy-main-recovery-20260912` at commit `f20f7bb`. That
   branch is unvalidated quarantine, not current controller work; do not merge
   it wholesale.
2. Start from a fresh clone or clean worktree of `compose-iclr` after this
   integration commit. The branch is local and unpushed until explicitly
   authorized.
3. Run `python3 tools/preflight.py` before any scientific launch.
4. For long T4 work, run `modal deploy modal_apps/genmol_t4_opt_app.py`, then
   use the durable launcher in `tools/t4_launch.py`. Do not use
   `modal run --detach` for the long driver.
5. Inventory the content-addressed Modal prefix before recomputing. Local JSON
   files are summaries; bulk states, checkpoints, locks, and docking receipts
   live on the `compose-v4-artifacts` volume.
6. Use focused tests while iterating. Run the full suite once only at a frozen
   launch or milestone boundary.

The old handoff's statement that `/Users/rmaganti/compose_v2_work` is fully
redundant conflicts with its own note that two ignored InversionGNN `.pt` files
are load-bearing. Those files are absent from this worktree. Do not archive that
directory until their hashes and a durable replacement location are verified.

## Reading order

1. This file.
2. `AGENTS.md` for scientific and authorization constraints.
3. `diagnostics/t4_frontier_compare/README.md` for the current primary result.
4. `docs/OPTION_CONTROLLER_RUNTIME_V1.md` for the newer controller runtime.
5. `docs/CONTROLLER_LIVE.md` only as a chronological evidence ledger.
6. `docs/MACRO_INVENTORY.md` and `docs/CAMPAIGN_LESSONS.md` before changing
   option support or launching expensive work.
