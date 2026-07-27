# REVIEWER_ATTACKS.md

Simulated adversarial review of the manuscript in its current (results-pending) state, then the revisions
made to preempt every honestly-addressable high-severity attack. Columns: **Attack | Severity | Existing
answer | Missing evidence | Paper change | Experiment dependency**. Severity in {low, medium, high}.

## Reviewer A — generative theory / stochastic control

| Attack | Sev | Existing answer | Missing evidence | Paper change | Experiment dependency |
|---|---|---|---|---|---|
| A1. The finite-horizon Doob transform is classical; nothing new. | high | We explicitly do not claim the transform is new; novelty is the instantiation on an operator-closed molecular state space, the budget-indexed embedded-chain form, and the continuation-vs-restart distinctness (Related work; conservative novelty sentence). | none | Confirmed conservative novelty sentence present; MOG-DFM framed orthogonal. | none |
| A2. "Exact control" is vacuous at scale because you never have exact h on real molecules. | high | Exactness is scoped to enumerable spaces; at scale h_phi is an approximation evaluated (calibration, backward residual, TV to exact); **validity is unconditional** (holds for any nonnegative h). Abstract/intro/limitations all scope this. | none (scoping, not a result) | Strengthened: controller and approximation appendix state validity-regardless-of-value-error explicitly. | none |
| A3. Editing discards the hazard; is it still a well-defined generative process? | medium | Yes: the embedded jump chain of the learned CTMC over a fixed budget; we distinguish timed CTMC (de novo) from fixed-step editing throughout, and reserve "CTMC/Gillespie" for de novo. | none | Present in Sec. kernel. | none |
| A4. Why train the total-hazard head at all if editing ignores Lambda? | medium | Editing uses the same conditional mark law p_theta that GM's factorization learns; Lambda governs de-novo timing and terminal detection (Lambda->0). So GM training is not wasted for editing. | none | Noted in Sec. kernel (shared p_theta; hazard for de novo/terminal). | none |
| A5. Self-normalized controlled kernel with approximate h is biased; unquantified. | medium | Bias vanishes as h->exact; measured via calibration + backward residual; twisted-SMC corrector restores asymptotic exactness. No closed-form bound claimed on real spaces. | quantitative calibration curve | Approximation appendix states this; RQ1 will report the curve. | RQ1 (E0_LEARNED_TV, calibration) |
| A6. Support preservation is trivial (multiply by a ratio). | low | Stated as a consequence, not a headline; the point is guidance cannot leave validity, unlike ambient/masked methods. | none | Framed as Cor., not contribution. | none |

## Reviewer B — molecular ML

| Attack | Sev | Existing answer | Missing evidence | Paper change | Experiment dependency |
|---|---|---|---|---|---|
| B1. Graph validity is not synthesizability. | high | Explicit: we say "graph-valid", never "synthesizable"; SA is an objective, not a validity claim. | none | Limitations + ethics say so. | none |
| B2. Every empirical cell is a placeholder; why believe the paper? | high | The contribution here is the framework + proofs + implementation validation; the empirical program is specified and in progress, carried through a result registry so no number is asserted early. This is a submission-readiness matter, not a soundness flaw. | the runs themselves | Intro states status explicitly; abstract is outcome-neutral. | ALL RQs |
| B3. Charged chemistry excluded -> narrow. | high | Charged states are **retained**, not excluded; only charge-**changing** edits are out of scope; charge is preserved. | none | Setup, operator registry, limitations state retain-not-exclude. | none |
| B4. The lead set (an internal "Jin-800"-like set) is treated as a benchmark without provenance. | medium | We require a primary citation and do not call an internal set an established benchmark; H1_SOURCE_PROVENANCE / H1_SOURCE_NAME are placeholders pending verification. | verified provenance | RQ2 protocol demands provenance before use. | RQ2 (H1_SOURCE_NAME) |
| B5. Stereochemistry excluded limits real lead-opt. | medium | Stated representation limitation (Kekule; no stereo/isotope/radical). | none | Limitations. | none |
| B6. A flexible legal editor games property predictors. | medium | RQ5 optimizes one oracle, evaluates with an independent one at matched OOD distance; gap bounds exploitation. | the gap value | RQ5 designed for this. | RQ5 (H5_*) |
| B7. Operator vocabulary can't build some rings (e.g. S/P aromatic). | low | Limitations: operator vocabulary bounds reachable chemistry; support-preserving control cannot manufacture unreachable transformations. | none | Limitations. | none |

## Reviewer C — multi-objective control

| Attack | Sev | Existing answer | Missing evidence | Paper change | Experiment dependency |
|---|---|---|---|---|---|
| C1. Better computational hypervolume != better molecules, and the gain may be a better edit vocabulary, not a better controller. | high | Same-base controller table isolates the control law; external optimizers additionally run on our legal proposal set; "computational front != efficacy" stated. | the numbers | Fairness + RQ2 + limitations. | RQ2 (H1 table) |
| C2. The goal-conditioned controller is not trained; the central object is a design. | high | Honest: exact DP is realized and verified; the amortized controller is under construction and evaluated as an approximation, with its regression target = the theorem's own value (no invented loss). | trained h_phi | Controller section + appendix design state this; zero METHODCHECK. | RQ2-RQ5 |
| C3. HV hygiene (reference point, caps, oracle counting)? | medium | Fixed before results: published reference point, per-method archive cap, count every oracle call incl. intermediates/rejected/SMC. | none | Fairness + HV-hygiene appendix. | none |
| C4. Region-selection Pareto policy is heuristic. | medium | It is a policy, not claimed optimal; HV-AUC vs oracle calls reported, paired with IGD+/coverage. | none | Multi-objective section + appendix. | RQ2 |
| C5. Linear scalarization strawman. | low | We use goal/region-conditioned control, not fixed weights, precisely to cover non-convex fronts. | none | Multi-objective section. | none |

## Reviewer D — reproducibility

| Attack | Sev | Existing answer | Missing evidence | Paper change | Experiment dependency |
|---|---|---|---|---|---|
| D1. Mixture percentages/counts are "proposed, not locked". | high | Honest and by design: ConfigValue marks them proposed; at-scale mining fixes them; a versioned manifest hashes scope/operators/compilers. | at-scale stats | Data appendix + config registry flag proposed. | at-scale mine |
| D2. Implementation-validation numbers (residuals, test counts) reproducible? | medium | Commit-pinned; clean-worktree gates; tolerances stated; evidence commit recorded. | none | Impl-validation appendix + PAPER_STATUS. | none |
| D3. Is any result based on the 53-pair pilot pool? | medium | No: it is a wiring fixture, appendix-only, explicitly not the production corpus. | none | Data appendix says so. | none |
| D4. Anonymity leaks (home paths, repo/profile names). | low | Anonymous-release checklist enumerates the scrub. | none | Repro appendix. | none |
| D5. Registry -> figure/table provenance. | medium | Every cell is a registry key backed by a machine-readable artifact; plots only from artifacts; release includes the registry + artifacts + harness. | the artifacts | Reproducibility statement + macro system. | runs |

## Reviewer E — area chair

| Attack | Sev | Existing answer | Missing evidence | Paper change | Experiment dependency |
|---|---|---|---|---|---|
| E1. Is this a paper or a plan? Empirically it is all pending. | high | The theory (closure, kernel normalization, Doob with corollaries, support preservation) is proved and machine-verified on the enumerable benchmark, and the implementation is validated; the empirical study is specified and in progress. Contributions are framed theory-first with a rigorous evaluation program. Acceptance depends on venue norms for theory-forward submissions. | the empirical study | Contributions/abstract framed accordingly; no unearned empirical claim. | ALL RQs |
| E2. Scope creep: 5 RQs, many baselines. | medium | RQ1 (exactness) and RQ2 (primary application) are the core; RQ3-5 support; de novo is secondary and in the appendix. | none | Experiments ordered by priority; secondary moved to appendix. | none |
| E3. Positioning vs MOG-DFM/GenMol could read as adversarial. | medium | MOG-DFM framed orthogonal (ported over our kernel, not a baseline); GenMol/others acknowledged as precedents; we do not claim valid editing is new. | none | Related work. | none |

## Revisions made in response (this pass)
1. Verified the abstract is outcome-neutral (no claimed hypervolume/efficacy win) and that every "exact"
   is scoped to an exact value function or enumerable space (A2, C1, E1). No change needed beyond
   confirmation; a scan for improve/outperform/guarantee/SOTA found only legitimate uses.
2. Confirmed "validity is unconditional under an approximate controller" is stated in the controller
   section and the approximation appendix (A2, A5).
3. Confirmed charge is framed as *retained, preserved, not designed* in setup, operator registry, and
   limitations, and that no section says charged chemistry is excluded (B3).
4. Confirmed the shared p_theta / hazard-for-de-novo-and-terminal point is present so GM training is not
   "wasted" for editing (A4).
5. Confirmed no result depends on the pilot pool and that mixture shares are marked proposed (D1, D3).
6. Confirmed the internal engineering codenames appear in **no** `.tex` source (checked); they live only
   in PAPER_STATUS.md's private mapping (anonymity; terminology contract).

Every remaining high-severity attack (B2, C2, E1) is the same honest limitation: the at-scale empirical
results are pending. These are not addressable by wording; they are addressed by (a) framing the paper as
theory-forward with a specified, pre-registered evaluation, (b) proving and machine-verifying the
mathematical claims, and (c) the result-registry discipline that prevents any premature empirical claim.

## Strict final score (results-pending draft)
- **Soundness: 3/5.** Theory is sound and machine-verified on the enumerable benchmark; the at-scale
  claims are unproven pending runs (by design).
- **Presentation: 4/5.** Tight, scoped, one-idea-per-figure; careful terminology; honest status.
- **Originality: 3/5.** The intersection (learned generator-matched process over an operator-closed
  molecular state space with exact path control) is novel; individual ingredients are not, and we say so.
- **Significance: 3/5 (conditional).** High if the RQ2/RQ3 results land; the framework and exactness are
  already a contribution.
- **Reproducibility: 4/5.** Registry-backed, commit-pinned, manifest-hashed, with an anonymous-release
  checklist; the empirical artifacts are pending.
- **Reviewer confidence: 4/5.**
- **Overall (current state): borderline; FULL_DRAFT_READY_FOR_RESULTS.** The manuscript is complete,
  compiles, and is honest; its accept/reject turns on the pending RQ2/RQ3 numbers, which the registry is
  built to drop in without further prose surgery.
