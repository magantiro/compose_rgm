# Project status — 2026-07-18

This is the operational status for the current working tree. It distinguishes
completed implementation, active verification, queued experiments, and legacy
evidence so a collaborator does not infer that training is running when it is
not.

## Completed and retained

- The `compose_v4` working tree is now a Git repository based on the existing
  `KoshaTx/compose_rgm` initial commit, with an ordered import and handoff
  history.
- All three canonical HTML plans are stored in `docs/research_plans/`.
- The complete step-6,250 trajectory archive is stored under
  `docs/trajectory_diagnostics/legacy_prequotient/`; the migrated PNGs are
  byte-identical to the old transient-workspace copies and the JSON uses
  portable relative paths.
- The measured Graft gauge-churn failure is fixed at the successor quotient:
  canonical self-successors are removed and alias rates reaching the same
  molecule are aggregated.
- The factorized ring decoder uses one executor-aware semantic support for
  teacher likelihoods and ancestral sampling. The retained 128-example teacher
  audit reports zero missing teacher marks.

## Active code gate

The first exact-support H100 preflight reached step 50 and confirmed that the
model learns normally: validation loss fell from 56.1582 to 25.1875 and family
accuracy rose from 0.0769 to 0.5897. It also exposed a decisive systems failure:
the first update waited 119.31 seconds for CPU support construction while GPU
forward/backward/optimization used approximately 1.11 seconds. The run was
stopped after preserving its recovery checkpoint; it is diagnostic only.

The working tree now performs exact objective-aware support evaluation. A
ring-grow teacher receives the complete executor-aware template mask. Every
other row receives an exact one-witness family-enablement certificate (or an
exhaustive all-false certificate), because the hierarchical family softmax
depends only on whether that unselected family is enabled. Sampling and ring
teacher likelihoods retain complete semantic support.

On the frozen deterministic 128-example benchmark, the new path is 3.38x faster
end to end and approximately 8.34x faster after worker startup. Two repetitions
produced identical mask/flag hashes. The refreshed 128-example teacher audit
reports zero failures and completed in 23.9 seconds on one process. The full
local suite passes with 242 tests and the Modal-entrypoint suite passes with 8.
The complete measurements and equivalence contract are recorded in
`docs/audits/2026-07-18_objective_aware_ring_support.md`.

A fresh 200-update H100 preflight with cumulative loader telemetry is still
required before declaring the gate complete.

## Not currently running

- No fresh quotient-correct unconditional training job is active.
- No corrected-checkpoint 100-sample preview exists yet.
- No corrected-checkpoint 2,000-sample FCD estimate exists yet.
- The archived step-6,250 samples and FCD dashboard are from the retired
  pre-quotient checkpoint and are diagnostic only.

## Immediate sequence

1. Rerun the exact-support H100 preflight with 8 data workers and prefetch 2;
   verify cumulative throughput, teacher finiteness, and ancestral sampling.
2. Launch fresh flexible-size training from the carbon-tree source with the
   successor quotient, exact semantic ring support, early stopping, and best
   checkpoint restoration.
3. Validate every 250 updates; after the 500-step warmup, persist
   `checkpoint.best_so_far.pt` every 500 updates and launch a separate
   100-sample ancestral evaluation from the first materially improved snapshot.
   Training continues while those molecules and trajectories are inspected.
4. Freeze the selected checkpoint and run 2,000 samples for validity,
   connectivity, uniqueness, novelty, size/ring/operator diagnostics, FCD, and
   the ChemNet mean/covariance decomposition.
5. Once unconditional sufficiency is established, implement valid-rewrite
   recovery and run the matched QED/conditional guidance gate required before
   COMPOSE-Lipid candidate generation.

## Claim boundary still open

The successor quotient resolves the observed presentation-level Graft churn,
but full invariance of learned successor rates to arbitrary atom-slot
permutations remains a formal and empirical audit obligation. Likewise,
validity-closed guidance has a strong architectural advantage—every oracle or
reward query is evaluated on an executable molecule—but better conditional
sample efficiency must be demonstrated against matched filtering/repair
baselines rather than assumed.
