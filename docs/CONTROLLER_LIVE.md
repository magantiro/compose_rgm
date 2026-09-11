# COMPOSE controller: current decision record

Updated 2026-09-11. This is the concise live record; linked experiment artifacts
retain the detailed history. Latest user contract authorizes implementation and
bounded experiments toward reproducible T4 and PMO improvement. Supporting route
reconstruction and valid chemistry do not count as benchmark success.

## Champion and protocol

- T4: `diagnostics/t4_combined_table.json` identifies the workshop table as a
  per-cell selection between compact controller B's feasible-run mean at up to
  100 calls and the older controller's single 200-call result. It selects B in
  11 cells and the older controller in 15, with four failures. Preserve both,
  not just B. The B source is `diagnostics/t4_all_runs.json`, selected by pooled,
  22 actions, no refinement; the older source is `genmol_t4_official_s1.json`,
  not s2. `scripts/t4_combined_table.py` documents the selection explicitly.
  Replicate counts are not uniformly three: B has zero runs in 12 cells, two
  in one, three in 16 and five in one. BRAF s10 d=0.6 uses one feasible run out
  of five. This hybrid statistic is not a single frozen-controller 500-call
  result or a matched mean-of-three comparison. Do not reuse that claim.
  PARP1 s0 d=0.4 B runs score -10.9/-10.6/-10.7 at 100 calls each. The newer
  broad controller has not demonstrated a reproducible improvement over this
  incumbent. Its later warm run `t4_macro_lookahead/8776743bbb813429663f339f60cbce587e59d427248dfaee80020014b11a6b28`
  reached -11.0 after 40 new calls plus 94 historical attempts. Result SHA-256
  `025a5a5586d440b3bfd07b65877998d10e259bae86d2b28f410155b3f1b82728`.
  That molecule redocked at -9.8; a winner-informed route endpoint scored -13.6
  in the separate six-call diagnostic `t4_winner_route_docking/fece76dedf244574e46892c7c8eb4bf3d5a9ae114e3abcd2be9e47bb779f03c9`.
  Neither is a replicated benchmark comparison. Keep docking noise and
  answer-known routes distinct from autonomous search.
- PMO without the newly recovered prescreen bank: best remains
  0.5222329678670935. The task-trained proposal comparison used 67 new physical
  queries in 212.817 seconds and did not improve either arm's best. Top-ten means
  were 0.503193 (baseline) and 0.507380 (learned). It is stopped, not extended.
  Source: `diagnostics/pmo_learned_proposal/report.json`.
- PMO prescreened development: recovered original top-100 bank reports a best
  of 0.5595028849441883 from 249455 prior evaluations. All 100 top-bank labels
  were revalidated against the pinned current oracle, with explicit achiral
  projection. The isolated donor-program batch improved best to
  0.5705974021574557 and top-ten mean from 0.5366184197 to 0.5505935555.
  It used 94 endpoint queries plus 100 historical parity repeats; 97/128 programs
  compiled and replayed. Recorded proposal work was 30.884 seconds on one CPU.
  Source: `diagnostics/pmo_donor_probe/report.json`. This is not an improvement
  under the previous no-prescreen initialization, a full-controller comparison,
  or official PMO AUC. The legacy prescreen's modern provenance is incomplete.
- The integrated matched PMO comparison is complete: hybrid best
  **0.6030226891555273**, baseline **0.5595028849441883**, identical starts.
  Top-ten means 0.557554 versus 0.550680. 156 new queries, 184 workers,
  2443.108 recorded worker-seconds, 0.175 oracle execution seconds. The best
  gain is a four-primitive side-chain replacement, not a ring addition.
  See `diagnostics/pmo_donor_comparison/report.json` and its README.
- Fresh-seed PMO replication also improved: hybrid best **0.5869734364739604**,
  baseline **0.5595028849441883**. Top-ten means 0.549018 versus 0.544569.
  149 new queries, 627.439 seconds, 186 workers, 2365.461 recorded worker-seconds.
  Both runs improve from the same original starts, but this remains exposed
  prescreened development, not external SOTA. Particle decisions replay exactly.
  See `diagnostics/pmo_donor_comparison/replication_report.json`.
- External comparison: match task, oracle, initialization/prescreening, total
  charged task labels and policy-training calls, per-run budget and replicates.
  Known-winner development is not blind discovery. T4 constraints apply to
  returned molecules, not every executable intermediate. The paper's aggregate
  replicate compute must not be presented as a matched per-run budget advantage.

## Current hypothesis and bounded experiment

Current next run: `configs/pmo_archive_branching.json` and
`docs/PMO_ARCHIVE_BRANCHING.md`. Both PMO gains disappeared from the active
population after exactly one follow-up draw. The new comparison uses identical
top-16 exact parents from both completed comparisons (best 0.603023), identical
donor/reference proposals, and contrasts forward-only SMC with archive parent
sampling. One incumbent slot plus 15 rank-weighted/full-support archive draws.
Four rounds, at most 128 new queries, 29 workers plus one driver, $10 reserved
cap. No concurrent T4 job. A new best beyond 0.603023 and the SMC arm earns
replication; merely retaining the incumbent is a null. Ten focused parent,
driver, donor, and restart checks passed in 3.79 seconds.

Use coordinated population-derived graph proposals to avoid sampling each
enabling primitive independently, then retain broad reference editing for local
refinement and open-ended exploration. The isolated donor channel's positive
result earns a matched comparison, not a benchmark-success claim.

Fresh-seed replication completed: `configs/pmo_donor_comparison.json` and
`PMO_DONOR_COMPARISON.md`. Identical 16 original prescreened starts, six option
boundaries, actual-score SMC in both arms. Baseline uses the broad reference
options; hybrid mixes that law 50/50 with donor programs. The probe's new winner
is neither a starting state nor a donor. At most 192 new physical queries,
14 workers plus one driver, no GPU, $10 reserved cap. Expected 4–12 minutes.
Completed initial run: `478421560799673857cd5aa0d2fc36df5b2ee4582828936ca6eb428699bd02fd`,
call `fc-01M28R0185X0BTA6JPN0P1PD3B`, source `7a8cce2bcaab`.
Remote: `compose-v4-artifacts/pmo_donor_comparison/<run_id>`.
Local launch: `/private/tmp/compose-pmo-donor-comparison-runs/<run_id>/spawn.json`.
Deployment took 86.738 seconds; strict preflight passed. Four focused donor/
handoff/reference-component/contract tests passed in 2.80 seconds. Eight existing
driver/proposal checks passed during the combined focused run; the new metadata
assertion was then corrected and the affected test file rerun. No full suite.
The completed raw donor probe is durable on `compose-v4-artifacts` at
`pmo_donor_probe/52cecccbb4b82a55afd7e2f3f409a6d6de52a0c9c6886639bfcdd010f53cf20b`.
The matched run needed recovery after a preemption exposed tuple/list equality
in its JSON identity lock. Same-task recovery reused completed chemistry and
all 72 existing queries, then finished at 156. Submission-to-finish was 752.728
seconds including recovery downtime, not the final session's 184.422 seconds.
The permanent JSON-content comparison repair retains fail-closed drift checks.
Ten focused donor/cache/driver/restart checks pass in 3.84 seconds.
Replication run: `88906fe9b25c5b283d4d68571dd8273aabbb16ba10929f182dc6f19f6df6ea5c`,
call `fc-01M28SEZ6K9JNBB8ENBT9SF2YG`, source `731a62699d02`.
Raw result SHA-256 `92da4b2adf711aa0e0c7146f8a397dde7372fdd66608e1ae9a10b6e71387c4c5`.
No recovery or repeated completed oracle queries were needed.

T4 transfer completed, negative: `configs/t4_donor_probe.json` and `docs/T4_DONOR_PROBE.md`.
One matched proposal batch from the existing exact-state T4 archive, not a full
optimizer. Same eight parents and endpoint gates for both arms, 16 option slots
per arm, at most eight novel dockings per arm plus two seed and two incumbent
repeat controls. No public winner endpoints, docking predictor, or learned
future-value head. At most 14 workers plus one driver, 20 total new calls,
$5 reserved cap. Launch only from the clean committed image. Ten focused
T4 allocation/lock/resume, donor and shared-driver checks passed in 4.27 seconds.
Run `f01772ffc5e8b46e8038de69ae7e09b47e1fde3c6be0ebc976c7e8bedf608166`,
call `fc-01M28THS9HPKKCW7924H6E6STJ`, source `85c948083f30`.
Deployment 114.424 seconds, execution 114.772 seconds. Six docking attempts:
hybrid's two novel eligible proposals scored -7.5/-7.6; seed repeats -7.3/-7.3,
incumbent repeats -11.8/-12.3. Baseline produced no novel eligible endpoint.
31/32 options completed, so compiler failure was not the principal bottleneck.
Of 31 completed endpoints, 22 failed endpoint constraints and seven were already
observed. No extra med-chem-only rejection. The run stopped at six calls rather
than spending the unused budget on unregistered replacement proposals.
Authoritative audit: `diagnostics/t4_donor_probe/report.json`. One additional
report regression covers floating property roundoff versus exact eligibility;
QED differed by 1.1e-16 across ARM/x86, with unchanged decisions.

Scientific change explicitly authorized by the user's full controller/macro
license: donor plans compile under the frozen executor but need not have positive
probability under the frozen neural proposal tables. No original-R_theta Doob,
kappa=1, or representation-invariance claim applies to this new proposal channel.
No executor, model weights, charge/size/slot semantics or benchmark gate changed.

## Decision and next action

The PMO best-score gain repeated across two seeds, but the one-option T4 transfer
failed on feasible novel coverage and score. Stop that unchanged T4 recipe.
The next T4 question is whether coordinated topology-plus-property repair can
finish at useful feasible endpoints, using the saved exact intermediates without
imposing endpoint constraints on their internal path. This is a proposed repair,
not yet a demonstrated controller improvement. No full panel or external
superiority claim follows from these development comparisons.

Future-aware work remains a candidate, but the previous head is rejected for
this process: its training target was a witnessed-path maximum and its primitive
horizon did not match the remaining option decisions. See the saved
`diagnostics/pmo_option_particles/GUIDE_AUDIT.md`. A replacement must learn from
actual policy continuations with matching horizon, not reuse that head as an
unqualified committor. Do not call donor compilation or an endpoint predictor a
future-value model. The current test changes proposals, not the value estimator.

Goal UI: the user cleared the prior PMO-only goal. The replacement T4+PMO goal
is registered and active under the latest user contract.
