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

New implementation: a score-adaptive coordinated-program optimizer now dispatches
before the legacy connected-region WHERE stage. It mutates attachments, created
atom classes and chain/ring-segment length, replaces compatible branches, and
retains the broad hierarchy as its third channel. Every full construction passes
the existing executor before endpoint eligibility or scoring. It is a non-neural
optimizer with measured-score replay, not a pointer decoder or future-value model.
The previous population/SMC runtime is preserved as an alternative, not overwritten.

The first local batch expanded the fixed paid bank: 16 new eligible endpoints
from 75 attempts in 4.23 seconds of proposal work, 13 through mutation and three
through recombination. Ten endpoints changed two sites. These endpoints have not
been docked. The local program-development mode lacks the remote reference assets;
its ten broad draws were counted as explicit failures, not silently reallocated.
Twenty-five focused tests passed, including exact broad-branch execution on a
model-free fixture, program mutation, measured-feedback and resume. This is an
implementation/yield result, not a new docking score or a full milestone sign-off.
See `docs/ADAPTIVE_PROGRAM_OPTIMIZER.md` and
`diagnostics/adaptive_program_optimizer/attempt_1/result.json`.

On the corrected answer-known PARP1 diagnostic, joint proposals reconstructed
the demonstrated winner 14/32 times from the seed versus 4/32 serial and 2/32
independent; post-linker counts were 7/32, 3/32 and 1/32. Independent pools were
more diverse. Two initially mismatched serial-horizon comparisons are explicitly
excluded and preserved. This is not autonomous discovery or a docking gain.
See `docs/ATTACHMENT_PROGRAM_PROPOSAL.md` for exact artifacts and limitations.

Completed docking evidence, checked on 2026-09-12:

- Winner-initialized reference refinement: best candidate -13.0, fresh repeats
  -13.0 and -13.0, versus three -13.6 winner controls. Nineteen total calls,
  including the three controls reused after the initial training-chain failure.
  Resume completed in 232.14 seconds at `0d965bf`. This is a negative result.
- Fully scored program pool: best non-winner candidate -13.3, fresh repeats
  -13.3 and -13.3. The joint channel proposed it from the original seed and
  post-linker state; the independent channel also proposed it post-linker.
  All 26 unique non-winner candidates were scored, then the best repeated twice.
  Twenty-eight new calls, 54.67 seconds of driver runtime, 94.90 summed docking
  worker seconds, eight workers plus a driver, CPU only. No surrogate selection.
  Clean source `1762f2d`; call `fc-01M2BM16P93H7P304NPDBN4FN1`; result prefix:
  `t4_program_pool/79c5b633c04b3f5cc1dc30ab729e8522b521a9ebf12118811fddaeaaebfb2563`.

Authoritative review: `diagnostics/t4_program_pool/attempt_1/review.json`, with
all input hashes and the unchanged remote receipt alongside it. Its README
separates the three proposal arms. These are winner-informed development results,
not a matched improvement over the autonomous broad controller or an IVG win.
The supplied -13.6 winner remains better. Both remote assays are finished; there
is no active run or automatic follow-on round.

The next reusable data unit is prepared: 54 complete program/attachment
representations of 27 measured endpoints, including the reused winner, with
source/endpoint/representation-balanced weights. All belong to one inspected
source group; this is not a transfer validation set. Rejected and unscored
attempts retain explicit exclusions, not invented negative scores. No model was
fitted. See `scored_program_replay.json` in the same diagnostic directory.

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

Retain coordinated program proposals and the existing generic broad channel.
Do not scale the same small winner-derived bank: it improves useful proposal
density in this development neighborhood but has only two new eligible seed
endpoints and five post-linker endpoints. The five post-linker variants all score
between -12.6 and -13.3; independent bundles are more varied but mostly weaker.
That supports learning coordinated attachments/parameters from completed scored
programs, not another larger particle population or a relabeled endpoint value.

Current hypothesis: making those programs variable expands the useful neighborhood;
measured endpoint feedback should then improve allocation within the same expanded
support. Variable-program generation and the adaptive archive API are implemented.
The first 16 new eligible endpoints are locked, unscored and resumable. Next paid
decision: compare static and adaptive preferences with identical initial information
and mutation/recombination support. The suggested 108-call maximum needs a new
budget decision beyond the older 66-call cap; no paid follow-on is active.
The 27-endpoint archive has only one source group and cannot validate transfer or
calibration. Preserve the serial baseline, endpoint-only constraints, old champion,
and all failed attempts. Do not promote adaptation based on structural yield alone.

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
| Attachment-specific programs | `edit_program.py`, `edit_program_policy.py` |
| Multi-site dependency graph and proposal arms | `edit_program_graph.py`, `multi_site_program_policy.py` |
| Variable program mutations and branch replacement | `program_mutation.py` |
| Adaptive archive, top-level dispatch and locked score updates | `adaptive_program_optimizer.py`, `molecular_task_search.py` |
| Cache-only batch preparation/resume | `tools/adaptive_program_search.py` |
| Locked program docking and review | `t4_program_pool.py`, `tools/review_t4_program_pool.py` |
| Scored complete-program replay preparation | `tools/prepare_scored_program_replay.py` |
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
