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
  broad controller has not demonstrated an improvement over this incumbent.
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
- External comparison: match task, oracle, initialization/prescreening, total
  charged task labels and policy-training calls, per-run budget and replicates.
  Known-winner development is not blind discovery. T4 constraints apply to
  returned molecules, not every executable intermediate. The paper's aggregate
  replicate compute must not be presented as a matched per-run budget advantage.

## Current hypothesis and bounded experiment

Use coordinated population-derived graph proposals to avoid sampling each
enabling primitive independently, then retain broad reference editing for local
refinement and open-ended exploration. The isolated donor channel's positive
result earns a matched comparison, not a benchmark-success claim.

Prepared next: `configs/pmo_donor_comparison.json` and
`PMO_DONOR_COMPARISON.md`. Identical 16 original prescreened starts, six option
boundaries, actual-score SMC in both arms. Baseline uses the broad reference
options; hybrid mixes that law 50/50 with donor programs. The probe's new winner
is neither a starting state nor a donor. At most 192 new physical queries,
29 workers plus one driver, no GPU, $10 reserved cap. Expected 4–10 minutes.
No active scientific run at this pre-launch update.

Scientific change explicitly authorized by the user's full controller/macro
license: donor plans compile under the frozen executor but need not have positive
probability under the frozen neural proposal tables. No original-R_theta Doob,
kappa=1, or representation-invariance claim applies to this new proposal channel.
No executor, model weights, charge/size/slot semantics or benchmark gate changed.

## Decision and next action

Best-score improvement over both the shared prescreened start and baseline earns
a fresh-seed replication. Top-ten-only improvement is limited refinement evidence.
A null stops the unchanged mixture; a truncated comparison is inconclusive.
Next T4 work must compare against its actual incumbent, not a weaker recent arm.

Future-aware work remains a candidate, but the previous head is rejected for
this process: its training target was a witnessed-path maximum and its primitive
horizon did not match the remaining option decisions. See the saved
`diagnostics/pmo_option_particles/GUIDE_AUDIT.md`. A replacement must learn from
actual policy continuations with matching horizon, not reuse that head as an
unqualified committor. Do not call donor compilation or an endpoint predictor a
future-value model. The current test changes proposals, not the value estimator.

Goal UI: the user cleared the prior PMO-only goal. The replacement T4+PMO goal
is registered and active under the latest user contract.
