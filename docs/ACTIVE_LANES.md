# COMPOSE active execution lanes

**Updated:** 2026-07-20 11:15 EDT  
**Rule:** measured results, running work, and future gates are kept separate.
Four independent lanes share only frozen artifacts and explicit qualification
decisions.

## 1. Conditional generation

**Owner:** conditional-generation worker  
**State:** first valid-successor mechanism test complete; corrected protocol and
backbone interface in progress

Measured:

- Late-window QED guidance produced mean QED 0.7732 versus 0.7096 for control;
  the paired mean change was +0.0636 over ten starts.
- Seven starts improved, one worsened, and two tied. Two guided samples reached
  QED >= 0.90 versus one control.
- Every state and scored proposal was valid and connected, but the confidence
  interval crossed zero and exact per-pair oracle budgets were not equal.
- These checkpoints descend from the full-ring hierarchical run, not the
  calibrated pancake substrate. The result is mechanism evidence, not a final
  GrIDDD comparison.

Running:

- A corrected-backbone interface for a pancake-derived canonical-successor
  model.
- Direct QED target conditioning plus separate controller and combined arms.
- A fixed-call lead-optimization evaluator that cannot underfill when a path
  terminates early.

Next gate:

- Evaluate starts with QED 0.70--0.80 under an exactly matched oracle-call
  budget; success requires QED >= 0.90 and Tanimoto similarity >= 0.40.
- Report all-attempt validity/connectivity, diversity, anytime candidates,
  oracle calls, and wall time. No controller sweep precedes this gate.

## 2. Unconditional generator repair

**Owner:** unconditional-failure worker  
**State:** P1/P2 rejected; minimal successor-level repair in progress

Measured:

- The calibrated pancake sampler gives 600/600 valid, connected, unique, and
  novel endpoints, zero canonical self-events, and immediate backtracking on
  only 0.143% of opportunities.
- P1/P2 reduced triple-containing molecules from 54.0% to 14.29%, but increased
  small-ring molecules from 11.33% to 45.92% and immediate backtracking to
  2.917%. It is not promoted or extended.
- The useful substrate is the calibrated pancake behavior, not the failed
  P1/P2 topology-rate formulation.

Running:

- Canonical-successor distillation or an equivalent quotient-correct objective
  that aggregates syntactic Graft aliases while preserving productive pancake
  family-rate mass.
- Exact small-state equivalence, family-mass preservation, and no-thrashing
  tests before any training launch.

Next gate:

- A cheap smoke pilot must preserve validity, size behavior, productive Graft
  use, and the pancake backtracking rate before a longer run is authorized.
- Ring-context chemistry and small-ring calibration remain separately gated;
  no executor filter or P1/P2 retry is authorized.

## 3. Large COMPOSE-Lipid structural corpus

**Owner:** lipid-corpus worker  
**State:** source inventory frozen; canonical overlap and R0 release running

Measured:

- LNPDB supplies 19,797 rows and 12,837 previously audited canonical ionizable
  lipids.
- Local AGILE assets supply 12,276 virtual candidates and 1,200 labeled rows;
  the labeled set substantially overlaps LNPDB.
- LiON supplies 13,331 measurement rows; LUMI supplies 1,920 full structures.
  LuT lacks full product structures and is excluded from structural
  canonicalization.
- LiGen's 900,351, LipidBERT's 10M, and Ouyang's roughly 20M corpora are not
  public general-purpose dependencies.

Running:

- Hash-locked source manifest, deterministic canonicalization, exact overlap
  matrix, source memberships, and leakage fields.
- An executable R0 release of real and genuinely available structures.
- Primary-source qualification of reaction transforms and a streaming,
  provenance-preserving enumerator.

Next gate:

- Produce a stratified 50k--100k pilot spanning multiple reaction families,
  heads, linkers, tail architectures, branching, unsaturation, degradability,
  charge, and size.
- Scale toward 500k--1M only after route-certificate, diversity, leakage,
  reachability, and throughput gates pass.

## 4. Pan-lung oracle matrix

**Owner:** primary agent  
**State:** qualified classical filtering bundle complete; expanded matrix in
progress

Measured and packaged:

- Eight serialized Ridge, ExtraTrees, and XGBoost members plus three
  applicability-domain models cover qualified A549, LUMI, and LuT-selectivity
  signals.
- The bundle passed clean-runtime verification and is authorized for
  domain-bounded filtering only. Reward fine-tuning remains disabled.

Running:

- Lipid-appropriate representation cells: fingerprints/descriptors,
  component/region features, frozen molecular encoders, end-to-end molecular
  graphs, and formulation-context encoders.
- Model-family cells: regularized linear models, tree ensembles, boosted
  trees, shallow masked multitask networks, graph models, and calibrated
  ensembles where data support them.
- Held-lipid, held-component/linker, held-study, and round/temporal transfer
  remain the selection splits; random folds are diagnostic only.

Next gate:

- Execute only representation/model combinations compatible with each typed
  endpoint, then package newly qualified members behind applicability-domain
  checks.
- Do not collapse airway in-vitro activity, local pulmonary delivery, systemic
  expression, selectivity, and barcoded uptake into one scalar.

## Shared integration gate

Lipid smoke training begins once the R0 structural release and a lipid-sized
COMPOSE kernel are executable. Oracle filtering can then score only admitted
candidates. Reward fine-tuning and prospective candidate ranking require a
frozen preregistration, exact bundle hash, fixed budgets, and a declared
novel-linker applicability policy.
