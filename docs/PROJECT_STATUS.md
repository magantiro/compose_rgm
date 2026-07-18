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

The remaining blocker before a fresh quality run is exact ring-support
throughput—not a chemistry or reachability failure.

The current working implementation adds an exact witness-first path:

1. A catalog electronic assignment may short-circuit a template to “supported”
   only after the normal rewrite executor verifies a valid candidate.
2. Catalog absence or witness failure never marks a template unsupported; the
   complete semantic decoder remains the fallback.
3. Catalog-exact scoring still enumerates the complete candidate table; the
   first-witness shortcut is used only for the Boolean semantic-support query.
4. Legacy topology-only catalogs continue to use the semantic fallback.

Focused witness/fallback/legacy tests pass. The complete local suite passes with
236 non-Modal tests and 8 Modal-entrypoint tests, and the refreshed 128-example
teacher audit again reports zero failures. On the frozen v3 validation cache,
the single-process 128-state support scan completed in 105.24 seconds versus a
144.2-second pre-fix profile. This is a CPU preprocessing path designed to run
in parallel DataLoader workers; it is not the measured GPU update time. A fresh
H100 preflight is still required before declaring the gate complete.

## Not currently running

- No fresh quotient-correct unconditional training job is active.
- No corrected-checkpoint 100-sample preview exists yet.
- No corrected-checkpoint 2,000-sample FCD estimate exists yet.
- The archived step-6,250 samples and FCD dashboard are from the retired
  pre-quotient checkpoint and are diagnostic only.

## Immediate sequence

1. Run the exact-support H100 preflight and verify validation/test batch build,
   training-update throughput, teacher finiteness, and ancestral sampling.
2. Launch fresh flexible-size training from the carbon-tree source with the
   successor quotient, exact semantic ring support, early stopping, and best
   checkpoint restoration.
3. Evaluate the first credible checkpoint on 100 ancestral samples; inspect
   molecules and trajectories before spending on a full evaluation.
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
