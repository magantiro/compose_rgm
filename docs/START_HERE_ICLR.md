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

New implementation: typed attachment-specific complete programs, multi-site
dependency graphs, conservative conflict serialization, peak-capacity checks,
and uninterrupted serial/independent/joint proposal arms. Every primitive still
passes the existing executor. The first proposal is finite-bank retrieval, not
yet a task-adapted pointer decoder or future-value model. It remains separate
from the deployed broad population controller.

On the corrected answer-known PARP1 diagnostic, joint proposals reconstructed
the demonstrated winner 14/32 times from the seed versus 4/32 serial and 2/32
independent; post-linker counts were 7/32, 3/32 and 1/32. Independent pools were
more diverse. Two initially mismatched serial-horizon comparisons are explicitly
excluded and preserved. This is not autonomous discovery or a docking gain.
See `docs/ATTACHMENT_PROGRAM_PROPOSAL.md` for exact artifacts and limitations.

Active run record: the nineteen-call winner-refinement assay completed three
controls at -13.6, then failed at a stale training-chain identity. Its checked
resume uses the existing qualified inference package and reuses those controls,
allowing at most sixteen additional calls. Clean launch commit: `0d965bf`.
Call: `fc-01M2BJYZ04VGA4S3J4JHSFQPY1`; remote run:
`t4_winner_refinement/d8881faeffb165bcbe59b1a572d5a6c9b2ec394746875b72a1e8a85ec6d55fd1`.
Last observed phase: `option_proposals`, 2026-09-12 19:56:09 UTC. A separate
26-candidate plus two-repeat pool diagnostic is prepared under the new approval;
see `docs/T4_PROGRAM_POOL.md`. Do not conflate these two assays or their budgets.

The primary scientific champion is unchanged until new candidate docking results
are verified. The -13.6 controls are the supplied winner, not a discovered lead.

### Earlier development evidence

Current development direction: use published IVG winners explicitly to learn
coordinated option sequences and attachment/primitive choices, then evaluate
autonomous search separately. Inverse decomposition now supplies 83 verified ring
demonstrations across 64 of 82 supported winners, including decoration restoration.
Preparation took 22.73 seconds with zero oracle calls. These start from
inverse-derived precursors, not benchmark seeds. The earlier raw-witness fit had
no whole-ring labels; that data gap is repaired.

The neural option actor is still **not promoted**: on four source groups excluded
from fitting, negative log likelihood was 3.995 versus 3.813 for the simpler
demonstration marginal (lower is better). A counterfactual on the real PARP1 route
rows shows that the marginal raises the first needed option 12.7-fold but halves
each of the next three. The specified four-option prefix remains about 7.1e-12
under its stated conditioning. Static ring frequency does not coordinate the
route. Next: complete multi-option demonstrations, including core-carbonyl
insertion, with attachment-level learning. Do not scale the rejected actor or
static prior unchanged. See `diagnostics/inverse_ring_proposals/README.md` and
`docs/WINNER_PROPOSAL_DEVELOPMENT.md`. Implementation and diagnostic artifacts
are uncommitted; the full controller milestone is not complete.

An earlier attempted duplicate T4 launch on 2026-09-12
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

For the exact submitted-paper versus post-submission boundary, including the
submission revision and a fresh-laptop path, see
`docs/PAPER_TO_CURRENT_CODE.md`.

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
2. Start from a fresh clone or clean worktree of `origin/compose-iclr`. The
   quarantine branch is durable on `origin` but is not a development base.
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
3. `docs/PAPER_TO_CURRENT_CODE.md` for the paper-to-current implementation map.
4. `diagnostics/t4_frontier_compare/README.md` for the current primary result.
5. `docs/OPTION_CONTROLLER_RUNTIME_V1.md` for the newer controller runtime.
6. `docs/CONTROLLER_LIVE.md` only as a chronological evidence ledger.
7. `docs/MACRO_INVENTORY.md` and `docs/CAMPAIGN_LESSONS.md` before changing
   option support or launching expensive work.
