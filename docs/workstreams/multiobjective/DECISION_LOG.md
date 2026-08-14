# Decision log — multiobjective workstream (Lane 5)

Every material decision, with the evidence available before it, the alternatives
rejected, and whether it changes a frozen object.

---

## 2026-08-13 — Branch created from `f6146d7`

- **Decision.** Lane branch `codex/compose-multiobjective-package`, based on
  `codex/editing-v2-successor-fiber-fastpath` @ `f6146d7` ("Freeze the top-up
  fallback BEFORE source 000's verdict; median source is 5").
- **Evidence.** That branch carries `docs/BASELINE_IMPLEMENTATION_POLICY.md`,
  `docs/workstreams/PARALLEL_WORKSTREAMS_AND_HANDOFF.md`, the frozen
  `diagnostics/pareto_tradeoff_census.json`, and the Pareto docs. Lane 4's
  active branch `codex/compose-pareto-control` is read-only for this lane.
- **Rejected.** Branching from `codex/compose-baseline-qualification`, which has
  the baseline infrastructure but is behind on the Pareto freeze.
- **Changes a frozen object.** No.

## 2026-08-13 — No compute of any kind

- **Decision.** Stage 0 and Stage 1 are executed with zero compute. Four public
  repositories were cloned into a scratch directory and read; nothing was
  installed into any project environment; no baseline was executed.
- **Evidence.** Standing lane instruction: nothing launches until the lead says
  so, per run; CPU only, never GPU; nothing external installed into the project
  environment.
- **Consequence, stated plainly.** Every cost figure in this workstream is a
  **published figure or a projection**, never a measurement by this lane. Where
  no published figure exists the cell says `UNVERIFIED` rather than carrying an
  estimate.
- **Changes a frozen object.** No.

## 2026-08-13 — HN-GFN is Panel B only, and the reason is one line of its code

- **Decision.** HN-GFN cannot enter the source-conditioned panel. It is
  qualified for the global-competence panel.
- **Evidence, read directly.** `main.py:145` — the rollout body opens
  `m = BlockMoleculeDataExtended()`, an empty block molecule, on every
  trajectory. `mol_mdp_ext.py:193-194` — `reset()` returns the same empty
  object. The only trajectory action is fragment append (`add_block_to`). There
  is no code path seeding a trajectory from a supplied SMILES.
- **Rejected.** Writing an adapter that injects a source molecule as the
  starting block state. That would change the method's proposal distribution and
  its trajectory semantics, which is the baseline-policy stop rule, and it would
  produce a comparison against our modification of HN-GFN rather than HN-GFN.
- **Changes a frozen object.** No.

## 2026-08-13 — `surrogate_calls` is added as a first-class resource axis

- **Decision.** The common vocabulary gains an axis none of the three existing
  schemes had: objective information obtained from a **learned model** standing
  in for the oracle. `oracle_efficiency_verdict` returns
  `INCOMPARABLE_SURROGATE_ASYMMETRY` rather than a number when one method has a
  surrogate and the other does not.
- **Evidence.** HN-GFN's true-oracle budget is 200 + 8×100 = 1,000
  (`main_mobo.py:50-52`) and the true oracle is touched only at
  `main_mobo.py:393`; the GFlowNet's training reward comes from the proxy
  (`main_mobo.py:211`). COMPOSE's committed smoke records `verified_pref` at a
  mean 269,884 distinct evaluations per source. On an oracle axis alone HN-GFN
  looks over two orders of magnitude cheaper, and that number is measuring
  amortization.
- **Alternatives rejected.** (a) Reporting the oracle axis anyway with a
  footnote — footnotes do not survive into a reader's summary, and the project
  rule requires the axis to be named, not annotated. (b) Charging HN-GFN's proxy
  calls to its oracle counter — that would be inventing an accounting rule the
  method does not use and would flatter COMPOSE.
- **Note.** The predecessor lane had already recorded this asymmetry as an
  unresolved fairness blocker in
  `docs/workstreams/baseline-qualification/DECISION_LOG.md`. This decision
  resolves it mechanically rather than restating it.
- **Changes a frozen object.** No — it adds an axis; it renames nothing.

## 2026-08-13 — The three counter vocabularies are reconciled, not unified

- **Decision.** `reconcile_ledger` maps Lane 4's `CostLedger`, Lane 4's
  `pareto_oracle_semantics` correction, and Workstream D's `OracleCounts` onto
  one axis vocabulary. No lane's committed counter names are changed.
- **Evidence.** Three schemes exist and overlap: `raw_oracle_calls` /
  `oracle_requests` are one concept under two names;
  `native_oracle_calls` / `unique_valid_canonical_evaluations` are nearly one
  concept but differ on unparseable proposals; the committed JSON emits
  `harness_only_requests` where the module prose says `benchmark_eval_requests`.
- **Rejected.** Renaming any lane's counters. Two of the three schemes are
  embedded in committed artifacts, and one of them —
  `raw_instrument_oracle_requests` — is the exact object a serial-versus-parallel
  parity replay must match. Renaming it would destroy that baseline.
- **Design consequence.** `raw_instrument_oracle_requests` is carried as a
  passthrough record and is deliberately **not** mapped onto an axis, because
  mapping it would attribute our harness overhead to the method.
- **Changes a frozen object.** No.

## 2026-08-13 — OP-GFN is excluded, and the licence is the decisive reason

- **Decision.** OP-GFN does not enter the numerical table. It is cited in
  related work.
- **Evidence.** (a) `multi/gflownet/tasks/seh_frag_moo.py:398,414` set
  `preference_type` to `None` unless `--type pref`, and `--type pref` is the
  PC-GFN baseline it compares against — so OP-GFN itself is not
  preference-conditioned, which is the property under study. (b) `:62` asserts
  the objective set is closed to `{seh, qed, sa, mw}`. (c) `LICENSE.md:1` is
  CC BY-NC-**ND** 4.0.
- **Why (c) is decisive on its own.** NoDerivatives means the thin adapter that
  would make it runnable in our harness is precisely what the licence does not
  permit us to distribute. No amount of engineering resolves that, and the
  charter admitted it only if a thin adapter would suffice.
- **Rejected.** Including it as a third external for breadth. The charter says
  do not build a baseline zoo, and a method that is not preference-conditioned
  adds no distinct objection.
- **Changes a frozen object.** No.

## 2026-08-13 — InversionGNN is conditional, and its code will not run as shipped

- **Decision.** Status `NUMERIC_BASELINE_IF_QUALIFIED`, conditional on two
  things that are not this lane's to decide.
- **Evidence, read directly in the checked-out tree.**
  `molecular/denovo.py:75-76` sets `model_ckpt = ""` and immediately calls
  `torch.load(model_ckpt)`; no checkpoint and no training labels are shipped.
  `molecular/denovo.py:107` passes six arguments to
  `optimize_single_molecule_one_iterate_epo_v3`, defined with four parameters at
  `molecular/inference_utils.py:153`. There is no `LICENSE` file anywhere in the
  tree.
- **Where the policy line falls.** Repairing the arity mismatch so the authors'
  own function is callable is compatibility work in the same family as the MARS
  sklearn fix — it restores intended behaviour. **Training a surrogate the
  authors never released, and choosing its training data, is not**: that is
  supplying the method's learned component, and the result would not be
  InversionGNN.
- **Rejected.** Reconstructing the surrogate to get a table row. If the
  conditions are not met it goes to related work, which the baseline policy
  explicitly names as a real option.
- **Changes a frozen object.** No.

## 2026-08-13 — The DRD2 oracle identity is recorded as a verified fact, with its limit

- **Decision.** Record that HN-GFN's shipped `oracle/scorer/drd2/clf_py36.pkl`
  is the same file COMPOSE's frozen DRD2 oracle was extracted from — sha256
  `dbc473fca922c834dbaee6eaba832caaff26d4f891734078fb1af359a111100f`,
  35,417,609 bytes, matching `pickle_sha256` and `pickle_bytes` in
  `artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json`, with an identical
  fingerprint definition.
- **Evidence.** `shasum -a 256` on the cloned repo file, compared against the
  committed manifest. Both numbers were read, not inferred.
- **What it licenses and what it does not.** It licenses the statement that the
  potency objective can be the same object for both methods. It does **not**
  license a claim of bit-identical scores: COMPOSE consumes extracted numpy
  parameters and HN-GFN calls sklearn `predict_proba`. Bit-level parity is
  recorded as `UNVERIFIED` and would need its own known-answer test against the
  manifest's committed `reference_scores`.
- **Note.** The predecessor lane's `baselines/hn_gfn/environment.lock` marks
  this pickle `DO NOT USE`, correctly — it is a Python 3.6 sklearn pickle and
  COMPOSE deliberately consumes extracted arrays instead. That guidance is not
  contradicted here; the new fact is the shared provenance, not a proposal to
  unpickle it.
- **Changes a frozen object.** No.

## 2026-08-13 — pCoMole is recorded as `UNVERIFIED`, and no summary is substituted for the paper

- **Decision.** All ten schema fields for pCoMole are `UNVERIFIED`. Title,
  authors and venue are recorded from the ICLR 2026 workshop listing; nothing
  else is recorded.
- **Evidence.** The paper is hosted only on OpenReview, which serves a browser
  challenge to non-browser clients; the API returns `403
  ChallengeRequiredError`; the paper is not on arXiv. The primary document was
  not read. Search-engine summaries of the PDF exist and describe specific
  mechanisms.
- **Rejected, deliberately.** Recording the search-summary description as a
  finding. This project has already paid for exactly that error once — a
  committed artifact cited a delegated report that had not arrived — and the
  standing correction is that *a cited-but-absent source is indistinguishable,
  to a later reader, from a verified one.*
- **Escalation.** This is flagged as the highest-priority unresolved item. The
  title alone describes preference-tilted constraint-aware molecule editing
  under a frozen discrete flow, from the same lab. Two fields decide whether it
  is a lineage citation or a prior-art question, and a human with an OpenReview
  login or an author contact must settle them.
- **Changes a frozen object.** No.

## 2026-08-13 — PepTune is added to the lineage audit, unprompted

- **Decision.** Add PepTune (Tang, Zhang, Chatterjee, ICML 2025,
  arXiv:2412.17780) to `SAME_LAB_LINEAGE.md` as `CONCEPTUAL_LINEAGE`.
- **Evidence.** Abstract, read directly: *"we introduce Monte Carlo Tree
  Guidance (MCTG), an inference-time multi-objective guidance algorithm."* Code
  at `github.com/programmablebio/peptune`, Apache-2.0.
- **Why it matters.** It was not on the charter's list, and it is the closest
  lineage neighbour of COMPOSE's future-aware control — the group's only
  rollout/tree-search multiobjective method. It independently confirms, from the
  literature side, the charter's ruling that *"Pareto does not need another
  'planning beats greedy' headline"*: that ground is already occupied by this
  lab's own prior work.
- **Changes a frozen object.** No. It constrains a claim rather than a number.

## 2026-08-13 — OPEN: Panel B's objective pair is not frozen, and this lane does not freeze it

- **Status: OPEN, main's decision.**
- **The tension.** `{DRD2, QED}` admits HN-GFN and COMPOSE's frozen potency axis
  but excludes InversionGNN, which has no DRD2. `{GSK3β, JNK3}` is the pair both
  external papers actually report and admits both externals, but is not
  COMPOSE's frozen axis. `{QED, SA}` admits everything and is scientifically
  weak.
- **This lane's recommendation.** `{GSK3β, JNK3}`, with Panel B framed as a
  competence comparison among externals, COMPOSE entering only if a legitimate
  global arm exists. It keeps every method on its own published ground.
- **Why it is not decided here.** Choosing a common benchmark after seeing which
  choice suits COMPOSE is the failure the freeze discipline exists to prevent.
  It must be frozen before any Panel B outcome exists, by whoever owns the
  claim.

## 2026-08-13 — LANE REDIRECTED: cite published numbers, do not rerun baselines

- **Decision (lead's, recorded here).** Stop dependency archaeology. Panel B
  becomes: pick ONE standardized published multiobjective molecular benchmark,
  run **COMPOSE only** under it, and cite external numbers **as reported**,
  labelled that way and never presented as a rerun.
- **Consequence.** The GPU-authorization question dissolves — COMPOSE is CPU.
  The HN-GFN environment work (BoTorch bisection, checkpoint absence) is
  **abandoned as unnecessary**, not solved.
- **Changes a frozen object.** No.

## 2026-08-13 — GSK3β/JNK3 freeze DECLINED on external protocol overlap

- **Decision.** GSK3β/JNK3 is **not** frozen as a head-to-head Panel B. The
  published values are labelled **contextual, non-head-to-head**.
- **Criterion applied.** External protocol overlap alone. No COMPOSE outcome
  entered the decision, and the in-flight P3/P4 repair was neither consulted nor
  waited for — its verdict can change how much emphasis Panel B needs, never
  which benchmark Panel B uses.
- **Evidence — dimension 1, oracle identity, FAIL.** TDC's `Oracle('GSK3B')` and
  `Oracle('JNK3')`, which InversionGNN calls (`molecular/denovo.py:9,25-26`),
  score with `AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)`
  (`tdc/chem_utils/oracle/oracle.py:682,710`). HN-GFN scores with
  `AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)` (`utils/chem.py:63`)
  against its own RandomForest pickles (`kinase_scorer.py:18`; sha256
  `60b2e6eb…` and `a03dd44a…`). A model trained on 1,024 features cannot consume
  2,048; these are different functions. **The two papers' GSK3β/JNK3 numbers are
  already not head-to-head with each other, before COMPOSE is mentioned.**
- **Evidence — dimension 2, budget, FAIL.** The published budget is 1,000
  true-oracle evaluations (`main_mobo.py:50-52`). Computed from
  `diagnostics/pareto_semantic_oracle_accounting.json`, COMPOSE spends 2,187
  `native_oracle_calls` per preference trajectory on `greedy_pref` and 53,977 on
  `verified_pref`. COMPOSE cannot complete one trajectory inside the entire
  published budget.
- **Evidence — dimension 5, validity, contributory FAIL.** HN-GFN scores
  unparseable molecules as `0.` and keeps them in the batch
  (`oracle/scorer/scorer.py:36`); COMPOSE's support is legal by construction. The
  top-K sets are drawn from populations with different validity semantics.
- **Alternatives rejected.** (a) Rerunning HN-GFN under TDC oracles to
  manufacture alignment — forbidden by the redirect, and it would substitute our
  numbers for the authors'. (b) Reducing COMPOSE's `BUDGET`, shortlist or fiber
  interrogation to fit the benchmark — changing the method to fit the table, and
  the frozen objects are not this lane's. (c) Trying a different objective pair —
  the budget gap is with oracle-budget benchmarking as a class, not with
  GSK3β/JNK3.
- **Changes a frozen object.** No. It declines to create one.

## 2026-08-13 — All six dimensions fail; the pre-committed repair route closes

- **Decision.** Dimensions 3, 4 and 6 resolve to **FAIL**, joining 1, 2 and 5.
  The verdict is unchanged but its basis is now complete rather than partial.
- **Dimension 3, HV reference point.** HN-GFN uses `ref_point=torch.zeros(...)`
  at `dataset.py:37`, `main_mobo.py:397`, `utils/metrics.py:87`, computed over
  the **entire accumulated oracle-scored dataset** (`dataset.py:131-135`).
  InversionGNN states **no molecular reference point**: the paper's only
  occurrence of "reference point" is `(1,1)` at `inv.txt:208`, in the
  **synthetic minimization** task, and there is no HV code in its `molecular/`
  directory. COMPOSE uses a frozen held-in p5 nadir with a p99 normalizer.
- **Dimension 4, weights and top-K.** HN-GFN's MOBO weights are 5 i.i.d.
  Dirichlet draws **resampled every outer round** (`main_mobo.py:483`); its
  synthetic experiment uses 5 evenly spaced (`main.py:493`); InversionGNN uses 5
  uniformly random angles (Algorithm 3, `inv.txt:430`); COMPOSE uses 5 frozen.
  HN-GFN does not agree with itself between its own two experiments, and the
  GSK3β/JNK3 MOBO numbers are the **random-weight** ones.
- **Dimension 6, `±` semantics.** HN-GFN's `±` is over **3 random seeds**
  (`hngfn.txt:491`); InversionGNN's is over **5 weight vectors**
  (`inv.txt:430,433`) and the word "seed" never appears in that paper. These are
  different statistics — one is run-to-run noise, the other is spread across the
  front, which for a preference-conditioned method is signal. Neither paper
  defines `±` as std or CI; that stays `UNVERIFIED` for both.
- **The pre-committed question, now answered.** This log recorded, *before* the
  extraction arrived, that if InversionGNN had rerun HN-GFN under TDC oracles
  then its single table would be a legitimate citation source. **It did rerun
  HN-GFN — at 50K+20K = 70,000 oracle calls** (`inv.txt:395-399`), against the
  **1,000** HN-GFN publishes for itself (`hngfn.txt:489-491`,
  `main_mobo.py:50-52`). A **70×** disagreement. The two reported hypervolumes
  differ accordingly: 0.669 ± 0.061 (HN-GFN's own, Table 5) versus 0.592 ± 0.042
  (InversionGNN's Table 4 attribution).
- **Consequence.** Both routes to a head-to-head table are now closed. Citing
  each paper separately fails dimension 1; citing InversionGNN's internally
  consistent table would misrepresent HN-GFN by 70× in budget. The gap is
  recorded as a **protocol mismatch, not a reproduction failure** — a distinction
  that protects both papers and that we are only entitled to draw because the
  budgets were read from primary sources rather than assumed.
- **Binding citation rule.** Cite each paper's own number **for its own method
  only**, each carrying its own oracle, budget, reference point and `±`
  semantics. Never quote 0.592 as "HN-GFN's hypervolume"; never place 0.669 and
  0.763 side by side as a comparison.
- **Changes a frozen object.** No.

## 2026-08-13 — The budget finding generalizes, and names an existing queued item

- **Observation.** Standard molecular-optimization benchmarks budget oracle calls
  at 10³–10⁴; PMO's convention is 10,000. COMPOSE's primary arm needs ~5.4 × 10⁴
  **per trajectory**. Changing benchmark does not fix this.
- **The only recorded idea that would change the arithmetic** is
  reference-prioritized shortlisting, already in `docs/QUEUED_EXPERIMENTS.md` as
  `QUEUED_CONDITIONAL — NO DESIGN / NO RUN`. This audit **names it as the
  blocking dependency** for any head-to-head oracle-budget comparison and does
  not design, scope or run it.
- **Changes a frozen object.** No.

## 2026-08-13 — PepTune expanded to the full ten-field schema

- **Decision.** PepTune now carries the same schema as MOG-DFM and AReUReDi.
- **Newly verified at `6f5f7cb`.** Its preference mechanism is **Pareto
  dominance filtering**, not scalarization (`pareto_mcts.py:143
  updateParetoFront`), so PepTune is **not preference-conditioned** — the mirror
  image of MOG-DFM and AReUReDi, which scalarize. Its feasibility handling is a
  soft `invalid_penalty * invalid` subtraction from reward
  (`pareto_mcts.py:446-455`), not a constructive guarantee. Licence Apache-2.0.
- **Changes a frozen object.** No.

## 2026-08-13 — OPEN: whether COMPOSE can enter Panel B at all

- **Status: OPEN.**
- COMPOSE's process is defined as editing from a supplied source. A global de
  novo COMPOSE arm is not something that currently exists, and building one to
  fill a table cell would be building a new method. If it does not exist, Panel B
  is a comparison among externals with COMPOSE absent, and the caption says so.
