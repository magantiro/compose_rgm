# COMPOSE final experiment checklist

Use this document as a run ledger. A checked box means the run, evaluator, and provenance have been frozen and archived—not merely started.

## A. Evaluation contract

- [ ] Freeze GuacaMol/MOSES/QM9 versions and canonical splits.
- [ ] Freeze molecular graph grammar `Γ` and RDKit version.
- [ ] Freeze sanitization, canonicalization, duplicate, and novelty policies.
- [ ] Freeze FCD/ChemNet implementation and preprocessing.
- [ ] Freeze fingerprint type, radius, bit length, and chirality setting.
- [ ] Freeze MCS timeout and timeout treatment.
- [ ] Freeze generated sample count per dataset.
- [ ] Freeze checkpoint-selection metric and tie-breaking rule.
- [ ] Freeze seed count and confidence-interval procedure.
- [ ] Freeze throughput hardware, batch size, warm-up, and timing protocol.

## B. Prior and forward process

- [ ] Direct or equilibrated sampler for `π0` implemented.
- [ ] Constructor samples satisfy internal grammar.
- [ ] Prior energy ratios numerically stable.
- [ ] Alias-aggregated proposal probabilities verified.
- [ ] Detailed-balance unit tests pass.
- [ ] Per-family inverse proposal support verified at every scheduled time.
- [ ] Acceptance rate reported by family and time.
- [ ] Prior coarse statistics converge from diverse initial molecules.
- [ ] Uncorrected legal walk baseline completed.
- [ ] MH-versus-no-MH action-degree/size/cycle bias analysis completed.

## C. Source erasure

- [ ] Terminal corruption versus independent-prior classifier AUC.
- [ ] Terminal corruption versus independent-prior MMD.
- [ ] Atom survival curve versus time.
- [ ] Initial–terminal MCS curve versus time.
- [ ] Scaffold retention curve versus time.
- [ ] Initial–terminal property correlations.
- [ ] Mixing from multiple size/composition/ring source strata.
- [ ] Sensitivity to corruption horizon and event budget.

## D. Model training

- [ ] Single-issue COMPOSE trained for at least three seeds.
- [ ] Reverse clock calibration checked.
- [ ] Operator-family calibration checked.
- [ ] Masked distributions sum to one for random valid states.
- [ ] State-transition alias likelihood tested against brute-force small graphs.
- [ ] Holding-time integral estimator validated.
- [ ] Best checkpoint selected without test leakage.

## E. Primary GuacaMol table

- [ ] BWFlow matched or clearly marked reported/unavailable.
- [ ] GraphBSI matched or clearly marked reported/unavailable.
- [ ] DiGress matched.
- [ ] GruM matched.
- [ ] DISCO matched.
- [ ] Cometh matched.
- [ ] DeFoG matched.
- [ ] ConStruct matched where protocol is applicable.
- [ ] CoCoGraph matched with fiber/prior protocol documented.
- [ ] GrIDDD matched.
- [ ] COMPOSE single issue completed.
- [ ] Validity, uniqueness, novelty, raw FCD, KL, all-step validity, and samples/s populated.
- [ ] Baseline provenance appendix populated.

## F. Path validity

- [ ] Internal grammar validity for every committed forward state.
- [ ] Internal grammar validity for every committed reverse state.
- [ ] Connectivity for every committed state.
- [ ] Independent RDKit sanitization for every committed state or a prespecified sample.
- [ ] Failure taxonomy reviewed manually on a sample.
- [ ] Operator proposal, legality, acceptance, and execution counts.
- [ ] Size/composition/cycle-rank excursion statistics.

## G. Core ablations

- [ ] Single-atom prior.
- [ ] Random-tree prior.
- [ ] Coarse Gibbs prior without MH.
- [ ] Coarse Gibbs prior with MH and flat action head.
- [ ] COMPOSE with hierarchical scheduler, single issue.
- [ ] COMPOSE-MI.
- [ ] Results rerun under identical architecture budget and evaluator.

## H. Scheduler analysis

- [ ] Forward proposal schedule plot.
- [ ] Forward legality schedule plot.
- [ ] Forward MH acceptance schedule plot.
- [ ] Reverse execution schedule plot.
- [ ] Net atom-count change by time.
- [ ] Net cycle-rank change by time.
- [ ] Flat head versus hierarchical scheduler quality/compute comparison.

## I. Multiple issue

- [ ] Bundle legality tests.
- [ ] Canonical serialization tests.
- [ ] Average and quantile issue width.
- [ ] NFE/sample comparison.
- [ ] Accepted edits/NFE comparison.
- [ ] Wall-clock speedup.
- [ ] All-step validity comparison.
- [ ] FCD/novelty comparison.
- [ ] Two-sample distribution discrepancy from exact single issue.

## J. Novelty and coverage

- [ ] Exact train-set novelty.
- [ ] Bemis–Murcko scaffold novelty.
- [ ] Nearest-neighbor Tanimoto distribution.
- [ ] MCS ratio distribution.
- [ ] Internal diversity.
- [ ] Duplicate rate.
- [ ] Property coverage.
- [ ] Scaffold coverage.
- [ ] Initial-prior-to-final similarity reported separately from train-set similarity.

## K. Secondary experiments

- [ ] MOSES table, including recent comparable methods where reproducible.
- [ ] QM9 table.
- [ ] Conditional source editing as a secondary capability.
- [ ] Similarity-constrained property optimization.
- [ ] Scaffold-preserving decoration.
- [ ] Formula/size-changing edit task.

## L. Compute and reproducibility

- [ ] Parameter count.
- [ ] Training GPU-hours per seed.
- [ ] Peak memory.
- [ ] NFE/sample.
- [ ] Accepted edits/sample.
- [ ] Samples/second on identical hardware.
- [ ] Environment lock file.
- [ ] Anonymous code archive.
- [ ] Dataset preprocessing script.
- [ ] Evaluation script with one-command reproduction.

## M. Preliminary-experiment disposition

For every existing pilot, assign exactly one status:

- [ ] **Main confirmatory ablation:** rerun under the frozen final pipeline.
- [ ] **Appendix diagnostic:** rerun under the frozen final pipeline.
- [ ] **Hypothesis only:** discard the old number and test afresh.
- [ ] **Omit:** incomparable, obsolete, or irreproducible.

Pilot categories already anticipated in the draft:

- [ ] single-atom start;
- [ ] tree-only start;
- [ ] tree plus random rings;
- [ ] uncorrected legal random walk;
- [ ] alternative operator schedules;
- [ ] early multi-operation prototypes.

## N. Final paper audit

- [ ] All red `TBD` values removed or intentionally replaced by `--`.
- [ ] Abstract contains one supported quantitative headline.
- [ ] No preliminary language remains in final results prose.
- [ ] Main text fits the official ICLR submission style.
- [ ] References precede appendix.
- [ ] Anonymous author and repository information verified.
- [ ] No unsupported “first” claim.
- [ ] LLM usage statement accurately describes assistance.
- [ ] Ethics/reproducibility statements reviewed by all authors.
- [ ] Every theorem assumption matches released code.
