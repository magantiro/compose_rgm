# COMPOSE ICLR controller: start here

This is the authoritative collaborator handoff for the controller campaign as
of 2026-09-13. Read this page before the chronological logs. The active
integration branch is `compose-iclr`; the Modal session name is
`compose_iclr`.

## Partner quick start

Clone or update the shared branch, then read the frozen benchmark contract and
the current controller decision record:

```sh
git clone --branch compose-iclr https://github.com/KoshaTx/compose_rgm.git
cd compose_rgm
git log --oneline -8
```

The paper-result track is currently running on Modal from the immutable clean
commit `c272b881bd23ebf1f46bbd1989e6e2871ad20687`. Later documentation and
controller commits do not alter that deployed code or its sealed recipe.

| Live item | Identity |
| --- | --- |
| Modal app | `compose-t4-frozen-program-benchmark` (`ap-leszzlH1iWePa3nhCUAjPr`) |
| Function call | `fc-01M2CE78Y7ZRQAXMNSVD775CH4` |
| Durable run | `54c3cb6d4a708cecc45c5037d8873a3f051538c6bc787bf84eff161f30be90da` |
| Volume prefix | `t4_frozen_program_benchmark/<durable-run-id>` on `compose-v4-artifacts` |
| Scope | 15 delta=0.4 T4 cells, three predeclared search replicates, at most 1,000 calls per unit |
| Hard limits | 45,030 calls including confirmations, 30 containers, CPU only, $20 reserved ceiling |

The remote zero-oracle preflight passed, including the clean source identity,
all five receptors and QuickVina. At the 2026-09-13 04:34 UTC read-only snapshot,
7,238 docking calls had completed, 35 of 45 units had started, two had completed,
and none had failed. Twenty-one started units were at or better than the
corresponding reported IVG mean. These are provisional per-run values, not the
final three-replicate aggregation or confirmation result. Monitor without
spawning another run:

```sh
MODAL_PROFILE=nitya PYTHONPATH=src:. .venv/bin/python \
  tools/t4_frozen_program_benchmark.py status
```

The status command reads durable volume state and can take about a minute while
many units are active. After the call completes, retrieve exactly once with:

```sh
MODAL_PROFILE=nitya PYTHONPATH=src:. .venv/bin/python \
  tools/t4_frozen_program_benchmark.py retrieve
```

Do not relaunch from the root checkout, change the frozen configuration, retry
ambiguous docking calls, or mix development-controller results into this run.
Start with [`T4_FROZEN_PROGRAM_BENCHMARK.md`](T4_FROZEN_PROGRAM_BENCHMARK.md),
`configs/t4_frozen_program_benchmark_v2.json`, and
`diagnostics/t4_frozen_program_benchmark/launch.json` for exact protocol and
provenance. The v1 source-state preflight failed before any oracle call and is
retained as a negative artifact.

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

Priority is **T4**; PMO is parked by user direction. Two work tracks are separate:

1. The frozen full-suite benchmark above measures the already selected fast,
   shared 146-program controller. It uses the same cold-start structural seed
   in every replicate for parity with the successful development preparation,
   and independent continuation/docking seeds thereafter. Live score-versus-call
   curves and a conservative predeclared competitive-plateau rule are part of
   the frozen contract.
2. Controller development preserves the fast coordinated-program baseline and
   explores stronger composition and proposal learning without touching the
   live run. The first protected two-program structural comparison produced two
   novel eligible BRAF endpoints and none on JAK2. It failed its preregistered
   gate, made zero oracle calls, and remains disabled in the benchmark. See
   [`T4_PROGRAM_COMPOSITION.md`](T4_PROGRAM_COMPOSITION.md).

The newest controller-development result is positive for bounded direct program
retrieval. Across all fifteen exact delta=0.4 sources, an eight-candidate
context-ranked retrieval slice produced 178 eligible endpoints versus 157 for
the forced-mutation cold-start control. It reconstructed 26 additional public
development winners across thirteen cells, with 51.46 versus 53.14 summed
proposal seconds and zero oracle calls. This is answer-known structural evidence,
not autonomous discovery or a new docking result. It remains absent from the
live frozen run. The unresolved 5HT1B seed-2 cell stayed at one endpoint in both
arms. See [`T4_PROGRAM_RETRIEVAL.md`](T4_PROGRAM_RETRIEVAL.md) and
`diagnostics/t4_program_retrieval/attempt_2/result.json`.

Verification boundary: the focused T4/program suite passes 26/26 and all 31
sealed retrieval artifacts validate. The repository-wide suite is not green. Its
ordinary invocation stops on the documented Editing-V2 semantic registry binding
drift; a continue-on-collection audit exposed further clustered legacy failures
before it was terminated at 47% after a long CPU-bound unrelated test. Do not
represent this branch as globally test-clean until that separate lane is repaired.

The bounded fast albuterol
run completed six rounds/110 charged queries before its time stop. Best score
remained 0.350 from initialization; top-ten mean rose 0.264309 to 0.304106.
All six pools filled sixteen slots, with 22.335 total proposal seconds and 0.106
oracle seconds, but synchronous publication made worker wall time 431.895 seconds.
This validates candidate throughput, not a new PMO best or matched benchmark win.
No further PMO campaign or extension is active. Result:
`diagnostics/fast_pmo_run/result.json`; clean launch commit `506961088fbb`.
The initial zero-query container-import failure and the previous interrupted grid
remain separately recorded.

Previous throughput checkpoint: [`FAST_PROGRAM_SEARCH.md`](FAST_PROGRAM_SEARCH.md).
Explicit program-only dispatch retains the exact executor, coordinated mutation,
recombination and current-state edits without evaluating the primitive reference.
Four BRAF/JAK2 pools each fill twelve eligible slots. Binding/caching optimizations
reduce summed proposal time 68.7 to 55.2 seconds with exact output equivalence.
These are unscored candidates, not improved benchmark scores. The untried-mutation
variant remains experimental. PMO initialization/dispatch/isolation repairs were
used in the bounded fast run above; the old mixed grid was not relaunched.

Latest result (2026-09-13): T4 completed with 72 calls and zero oracle failures.
The learned JAK2 champion scored -10.9 and repeated -10.5/-10.7; BRAF did not
improve its incumbent. PMO was interrupted: 647 charged reservations, 646 completed
scores, one unresolved query. Twelve task-dispatch failures occurred before
scoring, and a started-unit guard exception cancelled the remaining mapped work.
The underlying re-entry cause is unconfirmed. No parent-edit campaign containers
are active; no recovery has launched. Full results, implementation and proposed next decision:
[`PARENT_EDIT_CYCLES_REPORT.md`](PARENT_EDIT_CYCLES_REPORT.md).

The following launch paragraph is the pre-execution checkpoint, retained for
lineage rather than current operational status.

Latest authorization: the user approved the paired parent/edit learning cycles,
with 108 docking calls maximum on BRAF/JAK2 and 24,000 PMO queries, $20 combined
reserved compute and at most 30 containers. Inputs and task-independent PMO
initializations are frozen. Both group calls have now been submitted from clean
commit `5553ef0ca97c`; no new scores had been retrieved at that checkpoint.
The launch uses twelve shared CPU workers, two group drivers and at most two
confirmation workers, below the approved ceiling. Forty focused launch/dependency
tests passed; no full-suite or benchmark sign-off is claimed. Recipe and decision:
[`PARENT_EDIT_LEARNING_CYCLES.md`](PARENT_EDIT_LEARNING_CYCLES.md), machine-readable
contract `configs/parent_edit_cycles.json`. Launch and progress receipts will live
under `diagnostics/parent_edit_cycles/`. T4 call:
`fc-01M2C1HEC84P5BQ9VEQYWAGFJV`; PMO call: `fc-01M2C1HTBHN6X7S5A536A36ZP0`.
The deployed app is `compose-parent-edit-cycles`. Monitor these calls; do not
respawn started units or retry ambiguous oracle attempts. This allocation is separate from all
completed allocations below.

Current implementation checkpoint: the parent/edit controller now has verified
real-program decomposition (20/32 measured constructions), fresh-state proposal
budgets with retained ancestry, an explicit existing-atom channel, mutation-context
features, and shared metered learning-cycle plumbing. A real BRAF branch exchange
produced a new executable endpoint while retaining another branch; it is unscored.
The first small selector audit was mixed (three positive, two negative, three
unchanged archive-gain pools), so learned guidance remains unqualified. Next-model
fits retain 81 unique measured endpoints and all 101 observations across four
targets, including repeats, for future evaluation only. Those local repairs used
no new oracle calls; the prospective campaign above is now submitted.
Full details, limits and artifact map:
[`PARENT_EDIT_CONTROLLER.md`](PARENT_EDIT_CONTROLLER.md).

Correction to the BRAF mutation interpretation: the recorded attachment-only
variant is canonically the parent, and extension-only equals the joint candidate.
This is evidence for extension, not an attachment interaction. The existing
observations resolve the four-arm panel without further docking.

The second-generation scoring round is complete: **69 calls in 102.00 driver
seconds**, 293.47 summed worker seconds, no oracle failures. Best new first scores
across the two arms are JAK2 -10.9, FA7 -9.4, BRAF -11.2 and 5HT1B -12.7.
These replace neither historical benchmark entries nor repeat-only estimates.
BRAF's best came from the score-blind arm; ranked BRAF reached only -10.3.
FA7's ranked champion repeated at -9.3/-9.3, while JAK2 repeated at -10.7/-9.6
and the new 5HT1B champion at -7.5/-12.7. Retain the previous 5HT1B incumbent
(fresh -13.2/-13.9). The scored-history ranking arm has not earned promotion.
All eight arm-specific archives have been updated from their own locked queries
and compatible confirmations. That scoring round is no longer active; the new
learning-cycle allocation above is its own authorization.
Four of the 73 newly allocated calls remain unspent; they are not a new assay.
Authoritative receipts, review and decision:
`diagnostics/t4_second_generation/scoring_1/`.
Run source `66bec5c`; call `fc-01M2BW7XCWAB9PCK4J2AQDNTFW`.

### Preparation history

Second-generation preparation is now locked at
`diagnostics/t4_second_generation/attempt_2`: 31 score-ranked candidates and
32 score-blind controls, comprising 49 distinct target/molecule queries.
Both arms reuse the exact same eight measured programs per cell. Whole-program
source/peak/final size is explicit; no shrinkage reward or capacity increase.
Conditional mutation selection avoids unavailable operations, and duplicate
exhaustion no longer erases the 20% endpoint exploration floor. FA7 ranked
selection filled seven of eight slots under its unchanged work cap. Total
preparation took 81.85 seconds, with zero new oracle calls. These candidates
were unscored at preparation. The subsequently approved round above used 69 of
73 allowed calls including fresh champion/incumbent confirmations. See
`docs/T4_SECOND_GENERATION.md`. Do not regenerate or silently extend these pools.

Completed four-target run: 32 locked candidates across JAK2 seed1, FA7 seed0, BRAF
seed1 and 5HT1B seed0, plus four seed controls (36 docking calls). One shared
146-program library now fills all four eight-candidate pools in 25.06 seconds;
the PARP1-only library filled only JAK2. The 40-heavy-atom support is unchanged.
Best new scores were -10.3/-9.2/-11.1/-13.1 respectively, against freshly docked
seeds -7.9/-7.2/-9.4/-5.8. Every query completed; driver time was 61.16 seconds.
These are first measurements, not repeat-confirmed or matched-baseline wins.
All 32 measured programs now populate four protocol-bound optimizer archives.
The best FA7/BRAF/5HT1B candidates shrink their seeds by 5/7/7 atoms; the best
JAK2 candidate grows by six. No support expansion was needed. See
`docs/T4_SHARED_PROGRAM_CONTROLLER.md` and
`diagnostics/t4_program_curriculum/attempt_1/review.json`.
Run source: `1a5c0cc`; call: `fc-01M2BSMX75V6SM1695M6SFNQ62`.
That initialization used 64 of its separate 66-call follow-on allocation.
The completed second-generation comparison now shows that parent-score ranking
alone does not reliably identify the most productive program mutations. Keep
the measured archives, shared program support and exploration; do not scale an
unchanged ranked-parent policy or increase the atom cap in response.

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
is no unfinished work in those two assays. The separate four-target curriculum
above has also completed its bounded first scoring round.

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
are preserved in commit `909380e`; the full controller milestone is not complete.

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

Do not change the live benchmark. For development, retain the fast coordinated-
program baseline, exact-current-state continuation, branch recombination, original-
seed gates and 40-atom support. The failed protected-composition gate says that
more depth alone is not yet productive. The next high-leverage question is how to
make contextually compatible, variable transformations common without exhausting
work on duplicates, infeasible bindings or a slow reference channel.

The proposed development sequence is: improve component/interface coverage and
cheap binding filters; collect actual parent/edit outcome contrasts; then learn
which attachment-specific complete programs to propose. Preserve several measured
structural families, not only the highest-scoring parent. Add composition depth
only when a specific two-component failure motivates it. Do not add another future-
value head, particle increase or expensive selector before candidate support and
throughput justify it.

The full T4 run will reveal cell-specific deficits under one frozen recipe. Those
results may choose the next development target, but must not retroactively alter
the current paper-result track. Any successor becomes a separately frozen version
and reruns the same protocol.

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
| Fast program search and protected composition | `tools/fast_program_search.py`, `adaptive_program_optimizer.py` |
| Full frozen T4 benchmark | `t4_frozen_program_benchmark.py` under `src/compose_v4/experiments/`, `tools/`, and `modal_apps/` |
| Parent/edit campaign review | `tools/parent_edit_cycles.py`, `docs/PARENT_EDIT_CYCLES_REPORT.md` |
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
4. The current frozen T4 run is already deployed. Monitor its existing receipt;
   do not run a launch command. For a future separately authorized T4 campaign,
   use its task-specific app and durable launcher. Do not use `modal run --detach`
   for a long driver.
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
4. `docs/T4_FROZEN_PROGRAM_BENCHMARK.md` for the live paper-result track.
5. `docs/T4_PROGRAM_COMPOSITION.md` for the latest negative/inconclusive R&D gate.
6. `docs/PARENT_EDIT_CYCLES_REPORT.md` for the last completed learning campaign.
7. `docs/OPTION_CONTROLLER_RUNTIME_V1.md` for the newer controller runtime.
8. `docs/CONTROLLER_LIVE.md` only as a chronological evidence ledger.
9. `docs/MACRO_INVENTORY.md` and `docs/CAMPAIGN_LESSONS.md` before changing
   option support or launching expensive work.
