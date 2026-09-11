# COMPOSE controller: current decision record

Updated 2026-09-11. This is the concise live record; linked experiment artifacts
retain the detailed history. Latest user contract authorizes implementation and
bounded experiments toward reproducible T4 and PMO improvement. Supporting route
reconstruction and valid chemistry do not count as benchmark success.

## Champion and protocol

- T4: preserve the earlier compact semantic-ring controller B as the incumbent.
  Its linked/fused ring channels compile through the primitive executor; it is
  distinct from the newer broad region/option controller. See
  `T4_CONTROLLER_CONCLUSION_2026-08-27.md` and the supplied workshop paper.
  `diagnostics/genmol_t4_official_s2.json` is an older 500-call bank, not by itself
  a provenance-complete identification of every workshop-table cell. Reconcile
  cell-to-controller/run lineage before claiming a matched improvement.
- PMO development: the last replacement-continuation comparison did not improve
  the starting best, 0.5222329678670935, in either arm. Actual-score particle
  selection improved top-ten mean versus reference sampling (0.4975138439 versus
  0.4234744327), not the best. Source: `diagnostics/pmo_replacement_continuation/report.json`.
  This is warm perindopril development, not official PMO top-ten AUC.
- External comparison: match task, oracle, initialization/prescreening, total
  charged task labels and policy-training calls, per-run budget and replicates.
  Known-winner development is not blind discovery. T4 constraints apply to
  returned molecules, not every executable intermediate. The paper's aggregate
  replicate compute must not be presented as a matched per-run budget advantage.

## Current hypothesis and bounded experiment

Unchanged proposal generation is a bottleneck that endpoint selection alone has
not repaired. Fit WHAT/HOW proposal energies to existing scored complete-option
returns, then compare trained versus unchanged proposals under identical
actual-score SMC. WHERE, broad generic support, executor and R_theta stay fixed.
This is proposal learning, not a validated future-value model or exact Doob
optimization guarantee.

Prepared: 143 completed scored options, 671 decision examples; no new oracle
labels for fitting. The fit improved its training objective only. Model and
training provenance: `diagnostics/pmo_learned_proposal/`.

Active run: not launched yet. Recipe: `configs/pmo_learned_proposal.json` and
`PMO_LEARNED_PROPOSAL.md`. Nine shared starting molecules, four option boundaries,
two arms, at most 72 new physical PMO calls, 18 workers plus one driver, no GPU,
$10 reserved cap. Expected 6–15 minutes from prior boundary timings, with new
rejection overhead still unmeasured. Durable per-primitive progress and round
oracle locks; no retries or automatic budget extension.

## Decision and next action

Launch the frozen proposal comparison after focused checks and clean-source
preflight. Best-score improvement over both start and baseline earns a matched
replication. Top-ten-only improvement is limited evidence for local refinement.
A null result stops this recipe; a computationally truncated comparison is
inconclusive, not a chemistry failure. Next T4 work must compare against its
actual incumbent, not against a weaker recent controller.

Goal UI: the user cleared the prior PMO-only goal. The replacement T4+PMO goal
is registered and active under the latest user contract.
