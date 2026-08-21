# `docs/` index

Every file under `docs/` — 303 of them — with a one-line purpose and a status.
Generated 2026-08-19 against `codex/editing-v2-successor-fiber-fastpath` at `e3b1420`.

**How status was derived.** From the file's own content (supersession banners,
self-declared status fields, what a newer document says about it) and from
`git log -1` dates — **never from the filename**. Several files in this tree have
names that claim more authority than they hold; `CURRENT_MODEL.md`,
`PAPER1_FRAMING_AUTHORITATIVE.md` and `EXPERIMENT_PLAN.md` are all superseded.
Where the evidence did not settle it, the row says **UNKNOWN** rather than
guessing. A wrong `CURRENT` is worse than an admitted gap.

| Status | Meaning | Count |
|---|---|---|
| **CURRENT** | Governs now, or records live state that nothing in the repo contradicts. | 109 |
| **SUPERSEDED** | A newer document, amendment or measurement replaced it, or an audit found its claims false. Kept for provenance — several are load-bearing link targets. | 32 |
| **HISTORICAL** | A dated record that is correct *as history*: a completed audit, a closed line, a preregistration whose experiment has run. Not governing, not contradicted. | 155 |
| **UNKNOWN** | Could not be established from content plus git history. | 7 |

**SUPERSEDED never means "delete".** Nothing in this repository is deleted, and
several superseded documents (notably `PAPER1_FRAMING_AUTHORITATIVE.md`, with
~13 inbound references) must stay exactly where they are.

**Three documents to read before trusting any other index here:**
`REGISTRY_AUDIT_2026-08-19.md`, `REPRODUCIBILITY_HAZARDS_2026-08-19.md`,
`SESSION_RUN_MANIFEST_2026-08-18.md`.

**Bulk artifacts are not in git.** Corpora, embeddings, per-run records and
trained heads live on the Modal volume `compose-v4-artifacts` under
`editing_v2/r_theta_run/`. The `docs/*.json` files below are the small banked
summaries of results whose raw records are on that volume.

| File (under `docs/`) | Purpose | Status | Last commit |
|---|---|---|---|
| `48_HOUR_RESULTS_TRACKER.md` | 48-hour results sprint tracker opened 2026-07-19. <br> *The window closed; still cited by PROJECT_STATUS.md as 'the current decision schedule'.* | **HISTORICAL** | 2026-07-20 |
| `ACTIVE8_WITHIN_FAMILY_SEMANTIC_CENSUS_V2.md` | Within-family semantic census of Active8 training rows; independent marginals only, authorises nothing. | **HISTORICAL** | 2026-07-31 |
| `ACTIVE_LANES.md` | Four active execution lanes as of 2026-07-20. <br> *Lane structure predates editing-V2, the QED/h_phi lane and MOLLEO.* | **SUPERSEDED** | 2026-07-20 |
| `ADMISSION_MASK_BASELINE.json` | Per-state admission-mask timing baseline (close / restate). | **HISTORICAL** | 2026-08-16 |
| `ADMISSION_MASK_OPTIMIZATION.md` | Exact admission-mask memoization: 6.28x/3.24x, bitwise-qualified, and blocked from deployment. <br> *Blocked because the two files it touches are inside the Process-V2 identity hash - the same hash that gates banked artifacts.* | **CURRENT** | 2026-08-16 |
| `ADMISSION_SPEEDUP_MEASURED.json` | Measured admission-mask speedup with a law-parity flag. <br> *The optimisation is exact but blocked - see ADMISSION_MASK_OPTIMIZATION.md.* | **HISTORICAL** | 2026-08-17 |
| `AMENDMENT_EDITFLOWS_GRIDDD_HEADLINE.md` | Edit Flows + GrIDDD are the two headline neighbours; DDSBM leaves the paper. <br> *Supersedes the GrIDDD demotion in `workstreams/EXTERNAL_BASELINE_INDEX.md` and the DDSBM slot in `BENCHMARK_RULINGS.md`.* | **CURRENT** | 2026-08-14 |
| `AMENDMENT_H40_HPHI.md` | Decision to train a budget-extended (H40-aware) h_phi head. <br> *Executed and closed negative: 'H40-aware h_phi does not pay off', four measurements (`76675bd`). The FROZEN H24 head remains the controller.* | **HISTORICAL** | 2026-08-17 |
| `AMENDMENT_HORIZON_DIAGNOSTIC.md` | Horizon reachability diagnostic; amends preregistration §13 which froze H=24. Recorded before the data. <br> *Its bar on reporting any H>24 configuration was lifted for exactly one run by AMENDMENT_VALIDATION_128.md.* | **CURRENT** | 2026-08-17 |
| `AMENDMENT_INVERSIONGNN_2OBJ.md` | Matched 2-objective head-to-head vs InversionGNN (JNK3 + GSK3b); first external multiobjective comparator. <br> *The live lane at HEAD. Records why MOLLEO Task 3 did not fit and this does.* | **CURRENT** | 2026-08-19 |
| `AMENDMENT_MOLLEO_LAZY_GATE.md` | MOLLEO Dev Gate 1 (lazy verification L1 vs L4); replaces the falsified shortlist gate. <br> *Gate ran (L1/L4 did not diverge, 0 resampling events); all four MOLLEO search branches then closed.* | **HISTORICAL** | 2026-08-18 |
| `AMENDMENT_PUBLISHED_NUMBER_FIRST.md` | Project-wide published-number-first comparator policy binding Lanes 2/3/5/6 and the main lane. <br> *The absolute reading of the slogan is corrected by `PARETO_COMPARATOR_MATRIX_REQUIREMENT.md` (one day newer).* | **CURRENT** | 2026-08-13 |
| `AMENDMENT_SHORTLIST_RETENTION.md` | R_theta shortlist-retention diagnostic, preregistered as the MOLLEO prerequisite. <br> *Its premise was FALSIFIED (0/21 hard rescues in top-128); replaced by AMENDMENT_MOLLEO_LAZY_GATE.md.* | **SUPERSEDED** | 2026-08-18 |
| `AMENDMENT_SMC_EFFICIENCY.md` | SMC efficiency study; amends the §13 freeze on particle-count sweeps. Recorded before the data. <br> *Study completed - see LAZY_SAMPLER_RESULT.md, EXECUTION_PARITY.json, SMC_SENTINELS.json.* | **HISTORICAL** | 2026-08-17 |
| `AMENDMENT_VALIDATION_128.md` | Authorises ONE prospective validation on the held-out 128-source panel; freezes the controller beforehand. <br> *Gates the headline number (49.2% @k=8, 54.7% @k=12 vs GrIDDD 45.1% @k=20).* | **CURRENT** | 2026-08-18 |
| `ARCHIVE_3ARM_RESULT.json` | Three-arm archive comparison: cumulative, paired, work, diversity, branches. <br> *The reversal of the v1 archive result.* | **HISTORICAL** | 2026-08-17 |
| `ARCHIVE_AB_RESULT.json` | Persistent-archive A/B by stratum. <br> *Reversed by the three-arm result (`6eb963e`); line closed.* | **HISTORICAL** | 2026-08-17 |
| `ARTIFACT_INDEX.md` | Index separating 'current model artifacts' from historical diagnostics. <br> *1,247 commits behind - the most stale registry. No dead paths, but its 'Canonical'/'Current' labels are contradicted by three newer documents.* | **SUPERSEDED** | 2026-07-19 |
| `ATOM_RESTATE_NEURAL_ORBIT_AUDIT_V1.md` | Prospective train-only diagnostic: can the six-round node encoder distinguish legal generic atom-restatement marks? | **HISTORICAL** | 2026-07-31 |
| `AUDIT_EXISTING_CONTROLLER_LANE.md` | Audit of what controller machinery already existed (griddd_conditional.py, value-guided SMC, RTB trainer, frozen h_phi ensemble). <br> *Commissioned after that machinery was discovered *late*; still the map of what not to rebuild.* | **CURRENT** | 2026-08-14 |
| `AUDIT_REPORT.md` | Universal-edit-prior pre-training correctness audit; adversarial pass, fixes + regression tests, one open item A7. | **HISTORICAL** | 2026-07-26 |
| `BASELINE_IMPLEMENTATION_POLICY.md` | Project-wide policy separating published-number comparisons from re-implemented baselines. | **CURRENT** | 2026-08-13 |
| `BASELINE_PHILOSOPHY_AND_MAIN_LANE_DECISIONS.md` | Baseline philosophy for a methods paper, plus four main-lane decisions. <br> *Its comparator-*selection* guidance is superseded by COMPARATOR_ROLES_CANONICAL.md; that document states the four main-lane decisions and stop rules remain in force.* | **CURRENT** | 2026-08-13 |
| `BASE_ELEMENT_EXPANSION_PLAN.md` | CNOF -> full organic subset element-vocabulary expansion plan. <br> *Carries an ARCHIVED - DO NOT USE banner.* | **SUPERSEDED** | 2026-08-10 |
| `BEDIT_TRAINING_LAUNCH.md` | B-edit broad-organic training launch spec: wired and verified, not launched. <br> *Its own header defers authoritative status to PRODUCTION_PREFLIGHT_REPORT.md.* | **HISTORICAL** | 2026-07-28 |
| `BENCHMARK_RULINGS.md` | Benchmark rulings after the editing-competence and SA audits; binds Lanes 2/3/5/6. <br> *Its DDSBM tier-1 slot (§1) is superseded by AMENDMENT_EDITFLOWS_GRIDDD_HEADLINE.md; the remaining rulings bind.* | **CURRENT** | 2026-08-14 |
| `BROAD_ORGANIC_MINING_REPORT.md` | Broad-organic B-edit mining validation report; returned GO_FOR_FULL_500K_MINING. | **HISTORICAL** | 2026-07-27 |
| `BUDGETED_PREFERENCE_SWEEP_PREREGISTRATION.md` | Preregistration of the one remaining Pareto scalability mechanism (gate R1 & R2 & B). <br> *Ran and failed - see K41_BUDGETED_SWEEP_RESULT.md; the compression branch closed.* | **HISTORICAL** | 2026-08-14 |
| `BUDGET_SENSITIVITY.json` | Region reachability under the old H24 vs the new H40 budget. <br> *Part of the H40-head evaluation, which closed negative.* | **HISTORICAL** | 2026-08-18 |
| `CHECKPOINT_LINEAGE.md` | Checkpoint lineage classification by run-label pattern plus the loader-rejection guarantee. | **HISTORICAL** | 2026-07-28 |
| `CLAIM_LEDGER.md` | Paper-1 claim ledger: formal (F1-F9), support (S1-S16), empirical, barred/retracted, reviewer stress tests. <br> *All artifact paths resolve, but blockers on rows B/C/F/G are stale and the largest measured results (QED 128-source, MOLLEO, EXTENDED_CURVE_64) have no row. No replacement exists. Note `paper/CLAIM_LEDGER.md` is a *different* ledger.* | **SUPERSEDED** | 2026-08-12 |
| `CLAUDE_GENERATOR_HANDOFF_MANIFEST_V1.json` | Machine-readable manifest for the generator-lane handoff: scope, exclusions, remote runs at cut, sha256s. | **HISTORICAL** | 2026-07-20 |
| `CLAUDE_GENERATOR_RUN_LINEAGE_MANIFEST_V2.json` | Sealed evidence manifest for the run-lineage correction (V2). | **HISTORICAL** | 2026-07-21 |
| `CLAUDE_GENERATOR_START_PROMPT.md` | The prompt pasted into a separate generator session, with its worktree and scope. | **HISTORICAL** | 2026-07-20 |
| `CLAUDE_LIPID_HANDOFF_MANIFEST_V1.json` | File manifest for the lipid data/oracle handoff. | **HISTORICAL** | 2026-07-20 |
| `CNOF_CONDITIONAL_GATE.md` | Factorized C/N/O/F conditional Generator Matching gate. | **HISTORICAL** | 2026-07-18 |
| `COMPARATOR_ROLES_CANONICAL.md` | Canonical comparator policy: comparators have ROLES, not rankings. Project-wide. <br> *Cited as live by the InversionGNN amendment at HEAD.* | **CURRENT** | 2026-08-13 |
| `CONDITIONAL_CLAIMS_AND_CONTROLS.md` | Conditional claims -> controls -> metrics experiment matrix (Option A framing). <br> *'SUPERSEDED THESIS - 2026-07-28' banner.* | **SUPERSEDED** | 2026-07-28 |
| `CONDITIONAL_CONTROLLER_DESIGN.md` | Value-guided SMC controller design over the rewrite CTMC (Lineage B era). | **HISTORICAL** | 2026-07-20 |
| `CONDITIONAL_RESULTS_AND_SCOPING.md` | Conditional/control results and scoping for Paper 1. <br> *'SUPERSEDED THESIS - 2026-07-28' banner. Still linked as 'current' from the root `README.md`, which is itself stale.* | **SUPERSEDED** | 2026-07-28 |
| `CORPUS_STATE_2026_08_09.md` | Physical state of the V2 corpus, measured from artifacts, written after a near-loss of the only copy. <br> *The durability lesson behind `src/compose_v4/data/durable_path.py`.* | **HISTORICAL** | 2026-08-09 |
| `CURRENT_MODEL.md` | Model contract distinguishing source distribution, teacher, operator, generator, sampler and evidence. <br> *Describes NullSourcePrior / DegreeBoundedCarbonTreePrior selection - the pre-`run_v2_01` framing. The name overstates its currency; called out by `f413883`'s own commit message.* | **SUPERSEDED** | 2026-07-19 |
| `DDSBM_ENDPOINT_COMPETENCE_PROTOCOL.md` | DDSBM ZINC logP 2->4 endpoint-competence protocol, frozen before any COMPOSE outcome. <br> *DDSBM has since left the paper (AMENDMENT_EDITFLOWS_GRIDDD_HEADLINE.md); the frozen result remains in DDSBM_FROZEN_RESULT.json.* | **HISTORICAL** | 2026-08-14 |
| `DDSBM_FROZEN_RESULT.json` | Frozen DDSBM endpoint-competence analysis vs DDSBM's reported Table 1, with pending cases retained not dropped. <br> *DDSBM has left the paper; the result is preserved.* | **HISTORICAL** | 2026-08-14 |
| `DECISION_LOG.md` | What is established and what was refuted - read before re-running an experiment. <br> *One unresolved conflict: gives R_theta frozen at step 8,500 where PROJECT_BOARD gives 12,500 (REGISTRY_AUDIT C6).* | **CURRENT** | 2026-08-12 |
| `DEVELOPMENT_PLAN.md` | COMPOSE v4 base-model development plan. <br> *ARCHIVED - DO NOT USE banner.* | **SUPERSEDED** | 2026-08-10 |
| `DEVIATION_REGISTER.md` | Deviation register from the same coherence audit (D1..), severity-ranked. | **HISTORICAL** | 2026-07-28 |
| `DIVERSE_64_RESULT.json` | Forced-diverse branching on 64 sources. <br> *Null; branching line closed (`7313879`).* | **HISTORICAL** | 2026-08-17 |
| `EARLY_STOP_PROBE.json` | Early-stop equivalence probe: does stopping early change the outcome, and what it saves. | **HISTORICAL** | 2026-08-17 |
| `EDITING_CYCLE_CLOSE_GLOBAL_EQUIVALENCE_V1.md` | Cycle-close resolver matched a complete global mixed-integer oracle on all 1,707 exact validation sources. | **HISTORICAL** | 2026-08-01 |
| `EDITING_MODEL_FIRST_PRINCIPLES_SPEC.md` | First-principles capability and data specification for the source-agnostic editing prior. <br> *Working design contract of 2026-07-30; the editing-V2 line has since been built and frozen.* | **HISTORICAL** | 2026-07-30 |
| `EDITING_OPERATOR_DATA_AND_GATE_AUDIT_2026-07-29.md` | Measured audit of the completed RingCore-V1 corpus, operator behaviour and launch gates. <br> *Its forward-looking replacement is EDITING_MODEL_FIRST_PRINCIPLES_SPEC.md.* | **HISTORICAL** | 2026-07-30 |
| `EDITING_PROCESS_V2_DECISION.md` | Process V2: one gated admission authority over `atom_delete`; records a prospective support decision and its frozen artifacts. <br> *The self-hashed contract `configs/editing_v2_semantic_process_v2.json` is part of the identity chain that gates banked results.* | **CURRENT** | 2026-08-02 |
| `EDITING_T1_PROSPECTIVE_DECISION_2026-07-30.md` | T1 prospective optimization and capacity thresholds, frozen before any V4 T1 result. | **HISTORICAL** | 2026-07-31 |
| `EDITING_V2_ACTIVE8_FAST_PATH_DECISION.md` | Active8 fast-path decision: family-local teacher admission + restart-safe launcher; the 80-task pilot stopped on straggler cost. <br> *Cites `/private/tmp/process_v2_active8_teacher_support_benchmark.json`, which is gone - see REPRODUCIBILITY_HAZARDS class 2.* | **HISTORICAL** | 2026-08-07 |
| `EDITING_V2_CYCLE_OPEN_MIGRATION_PLAN.md` | Editing V2 semantic cycle-open migration plan. <br> *ARCHIVED - DO NOT USE banner.* | **SUPERSEDED** | 2026-08-10 |
| `EDITING_V2_FORENSIC_STATUS_2026-07-31.md` | Forensic status of the Active8 editing architecture; development evidence only. | **HISTORICAL** | 2026-07-31 |
| `EDITING_V2_P50_V2_DECISION.md` | P50-v2 prospective decision: retain connected-nonleaf atom deletion in the Process-V2 legal support. | **HISTORICAL** | 2026-08-06 |
| `ENTRYPOINT_MATRIX.md` | Entry-point census: 106 scripts + 6 diagnostics + 5 modal apps + tests, and the production call graph. <br> *Counts are from 2026-07-28; the tree now holds 238 scripts and 141 modal apps.* | **HISTORICAL** | 2026-07-28 |
| `ESTATE_REGISTRY.md` | Whitelist registry: 'exactly three files are current', paper-directory map, ARCHIVED census. <br> *Whitelist overtaken by MASTER_PLAN, which it does not name. Doc count (84) and mtime/size claims false. The ARCHIVED census (11 files) is still accurate.* | **SUPERSEDED** | 2026-08-10 |
| `EXECUTION_PARITY.json` | Execution parity 48/48 across families: `ok`, checked count, mismatches. <br> *Gates the lazy sampler. Its app is one of the repo's correct failure handlers (any exception fails the verdict).* | **CURRENT** | 2026-08-17 |
| `EXPERIMENT_INFRASTRUCTURE_PLAN.md` | Experiment infrastructure plan v2 (E1-E7). <br> *ARCHIVED - DO NOT USE banner.* | **SUPERSEDED** | 2026-08-10 |
| `EXPERIMENT_PERSISTENT_TREE.md` | Design (fixed before the data) for a persistent search archive vs independent restarts. <br> *Line closed on measurement: three-arm archive reversal (`6eb963e`) then forced-diverse branching null (`7313879`).* | **HISTORICAL** | 2026-08-17 |
| `EXPERIMENT_PLAN.md` | Self-declared 'canonical' experiment plan: Experiments 1-5, claim 4A rows, superseded-document list. <br> *Overtaken by MASTER_PLAN (REGISTRY_AUDIT §5 C1). Zero mentions of MOLLEO/QED lanes. Still named as plan-of-record by `CLAUDE.md:19` and `ESTATE_REGISTRY.md:11`.* | **SUPERSEDED** | 2026-08-12 |
| `EXTENDED_CURVE_64.json` | 64-source development coverage curve @1-@20 by stratum: ALL 47/64 = 73.4% at 20; reliable 19/19, marginal 7/8, hard 21/37. <br> *Highest-risk provenance in the repo: its producer reads `/tmp/rv5` with no volume fallback (REPRODUCIBILITY_HAZARDS class 1 HIGH / class 2 CRITICAL).* | **CURRENT** | 2026-08-18 |
| `FAMILY_AUDIT.json` | Family tables plus empty-mass median/max. | **CURRENT** | 2026-08-17 |
| `FAMILY_MASS.json` | Per-family probability mass over n states. | **CURRENT** | 2026-08-17 |
| `FCD_TRANSFER_FROM_RANK_D500K.md` | FCD transfer reference from the sibling `rank_d500k` recipe. <br> *FCD is explicitly deferred, not a headline.* | **HISTORICAL** | 2026-07-18 |
| `FRAGMENT_METHOD_BOUNDARY.md` | Why COMPOSE v4 is not a fragment-assembly model (teacher trace vs generative model). <br> *Conceptual boundary argument, unaffected by later runs.* | **CURRENT** | 2026-07-18 |
| `GENERATOR_INTERNALS_FROM_ZERO.md` | The generator explained from zero for a newcomer to the lipid lane. <br> *Still the gentlest available introduction to the model, but written against the 2026-07-20 code.* | **HISTORICAL** | 2026-07-20 |
| `GENERATOR_LINEAGE_MAP.md` | Definitive evidence-backed lineage audit: every checkpoint, SHA-256, commit. <br> *Still the recovery route for checkpoint identity; records that the `/private/tmp` copies are wiped on reboot but exist on the Modal volume.* | **HISTORICAL** | 2026-07-25 |
| `GENERATOR_MATCHING_INTUITION.md` | Generator Matching versus flow matching and diffusion - conceptual explainer. <br> *Conceptual, not tied to a checkpoint; still accurate.* | **CURRENT** | 2026-07-18 |
| `GENERATOR_RESULTS_SUMMARY.md` | Honest proven/measured/pending status of both generator tracks (2026-07-21). | **HISTORICAL** | 2026-07-21 |
| `GRAPH_ONLY_ENCODE_PLAN.md` | Analysis of a graph-only h_phi encode path targeting the 79.1% of a transition spent in encode. <br> *Marked 'ready to implement' 2026-08-15; no document records whether it was implemented, and the lazy sampler landed two days later.* | **UNKNOWN** | 2026-08-15 |
| `GRIDDD_JIN_PROTOCOL.md` | GrIDDD / Jin ZINC-250k constrained-editing protocol: task, cohort, success criterion, 20-candidate budget. <br> *Carries its own 'SUPERSEDED IN PART' banner - Policy B (stochastic one-step control) was refuted 0/320 and is history; the task definition still governs.* | **CURRENT** | 2026-08-14 |
| `H40HEAD_AB_RESULT.txt` | Raw console output of the 12-source H40-head A/B: decision strata 4/8 -> 3/8. <br> *The H40 head was not adopted.* | **HISTORICAL** | 2026-08-18 |
| `HANDOFF.md` | Top-level COMPOSE RGM handoff describing the repository as a lossless working handoff. <br> *Routes readers to PROJECT_STATUS.md, itself superseded.* | **SUPERSEDED** | 2026-07-19 |
| `HANDOFF_CLAUDE_GENERATORS.md` | Lossless handoff of the unconditional + conditional generator lanes to a second session (2026-07-20). | **HISTORICAL** | 2026-07-20 |
| `HANDOFF_CLAUDE_LIPID_DATA_ORACLES.md` | COMPOSE-Lipid data and oracle lane handoff, 2026-07-20. <br> *Paper-2 / lipid program: zero mentions in any current plan and no commits in 200+ - parked or lapsed, unresolved (REGISTRY_AUDIT §7.2).* | **HISTORICAL** | 2026-07-20 |
| `HANDOFF_COMPOSE_TRACEABILITY_2026-07-29.md` | Claim-by-claim traceability analysis of the 2026-07-29 authoritative handoff. | **HISTORICAL** | 2026-07-30 |
| `HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md` | Run-lineage correction and decision package: the two omitted unconditional lineages, and the backbone decision. <br> *Still linked from the root README.md as required reading before large runs; that README is stale.* | **HISTORICAL** | 2026-07-21 |
| `HANDOFF_LIPID_GENERATOR_FROM_GENERATORS.md` | Generator -> lipid lane handoff, rev 2 (base moved from pancake to Lineage B). | **HISTORICAL** | 2026-07-20 |
| `HANDOFF_PROCESS_V2_CONNECTED_NONLEAF_DELETE.md` | Implementation handoff for Process V2 connected-nonleaf atom deletion. Not training authority. | **HISTORICAL** | 2026-08-02 |
| `HANDOFF_PROCESS_V2_EVIDENCE_CHAIN.md` | Implementation handoff for the Process-V2 evidence chain; launches nothing. | **HISTORICAL** | 2026-08-03 |
| `HANDOFF_RINGCORE_V1_POSTRUN.md` | 'START HERE' handoff for the receiving agent after the RingCore-V1 run. <br> *An entry point for a 2026-07-29 handoff, not for this repo today.* | **HISTORICAL** | 2026-07-29 |
| `HORIZON_AMENDMENT_H24.md` | Horizon amendment fixing H=24 after the frozen selection rule did not fire; amends HPHI_V1_CORPUS_PREREGISTRATION. <br> *H24 is the frozen h_phi head still in use.* | **CURRENT** | 2026-08-14 |
| `HORIZON_DIAGNOSTIC.json` | Horizon diagnostic outcome by stratum. | **CURRENT** | 2026-08-17 |
| `HPHI_CORPUS_CENSUS.json` | Read-only census of the frozen corpus: training support, gateways to benchmark, 20 regions. | **CURRENT** | 2026-08-14 |
| `HPHI_CORPUS_FROZEN.json` | The frozen h_phi V1 rollout corpus: artifact path, sha256, bytes, shape, budget check, R_theta model hash. <br> *Near-complete provenance; missing only a git commit and a producing script.* | **CURRENT** | 2026-08-14 |
| `HPHI_COVERAGE_LADDER_BANKED.json` | Coverage ladder rungs 1-2, cumulative and per-source, with a 20-attempt projection and a mechanical qualification record. | **CURRENT** | 2026-08-16 |
| `HPHI_DEV_PANEL_64_BANKED.json` | The 64-source development panel, per-arm and paired source-level, with record checksums. <br> *Self-labelled `CLOSED FOR CONTROLLER TUNING` - do not tune against it.* | **CURRENT** | 2026-08-15 |
| `HPHI_HORIZON_QUALIFICATION.json` | Horizon qualification: nesting verified, qualifying tiers, and the frozen selection rule. <br> *The selection rule did NOT fire - that is what HORIZON_AMENDMENT_H24.md responds to.* | **HISTORICAL** | 2026-08-14 |
| `HPHI_LADDER_RUNG3.json` | Ladder rung 3: 37 of 37 remaining hard sources failed with zero particles ever entering the region. <br> *The measurement that motivated AMENDMENT_HORIZON_DIAGNOSTIC.md.* | **CURRENT** | 2026-08-17 |
| `HPHI_LAW_TRACE.json` | Which FactorizedMarkBatch fields a law call actually reads, and mean law seconds. <br> *The evidence behind LAW_BUILDER_TRACE_FINDING.md (branch killed).* | **HISTORICAL** | 2026-08-16 |
| `HPHI_N_SWEEP.json` | Particle-count (N) sweep summary and arms. <br> ***Untracked** - present on disk, not in git, and no document references it.* | **UNKNOWN** | untracked |
| `HPHI_PILOT_CENSUS.json` | Read-only census of the pilot rollout corpus (same schema, pilot scale). | **HISTORICAL** | 2026-08-14 |
| `HPHI_QED_PREREGISTRATION.md` | Binding implementation protocol (the 'how') for the QED/GrIDDD region-h_phi controller. | **CURRENT** | 2026-08-15 |
| `HPHI_SMC_64_GATE_BANKED.json` | Full-cohort SMC gate: `PASSED - SMC strictly dominates at matched budget`, with exact paired p and extinction rate. | **CURRENT** | 2026-08-16 |
| `HPHI_SMC_PROBE_BANKED.json` | SMC efficacy probe: `CLOSED - SMC EARNED SCALING`, with a matched 4-slot comparison. | **CURRENT** | 2026-08-15 |
| `HPHI_SMC_REFERENCE_BANKED.json` | The trusted executable specification of the SMC primitive. <br> ***The provenance template for this repo** - the only docs JSON with artifact_sha256 + record_sha256 + provenance.git_commit + mechanical_qualification.script.* | **CURRENT** | 2026-08-15 |
| `HPHI_V1_CORPUS_PREREGISTRATION.md` | Region-h_phi V1 rollout corpus preregistration: 1,024 x 8 x H6, frozen before any rollout. <br> *'The original H6 preregistration no longer applies' - HORIZON_AMENDMENT_H24.md.* | **SUPERSEDED** | 2026-08-14 |
| `INDEX.md` | This index. <br> *Generated; regenerate rather than hand-edit when files are added. Companion to `README_NAVIGATION.md` and `REORG_REPORT.md` at the repo root.* | **CURRENT** | untracked |
| `INVERSIONGNN_BLAMBDA_DIAGNOSTIC.json` | Preference regions B_lambda fixed before any label is spent. <br> ***Untracked** - present on disk, not yet committed.* | **CURRENT** | untracked |
| `INVERSIONGNN_FPSI.pt` | Trained F_psi property-predictor weights for the InversionGNN comparison. <br> ***Untracked and gitignored** (`*.pt`). A working artifact of the live lane, not in git.* | **CURRENT** | untracked |
| `INVERSIONGNN_FROZEN_PROTOCOL.json` | Frozen InversionGNN comparison protocol: ZINC pool + sha256, init-bank rule + sha256, disjointness verification, the five recovered preference vectors. <br> *The live lane at HEAD.* | **CURRENT** | 2026-08-19 |
| `INVERSIONGNN_HPHI.pt` | Trained multiobjective h_phi head for the InversionGNN comparison. <br> ***Untracked and gitignored** (`*.pt`). A working artifact of the live lane, not in git.* | **CURRENT** | untracked |
| `INVERSIONGNN_SUPERVISION_10K.json.gz` | The InversionGNN-matched 10K supervision corpus, with the init bank excluded. <br> *Committed 2026-08-19; the largest artifact tracked under docs/.* | **CURRENT** | 2026-08-19 |
| `K41_BUDGETED_SWEEP_RESULT.md` | K41 budgeted preference sweep read once against the frozen gate. Verdict FAIL. <br> *Plan map: block 4C 'failed; closed'.* | **HISTORICAL** | 2026-08-14 |
| `KERNEL_COST_CHARACTERIZATION.md` | What an executable chemistry-native process costs: measured decomposition of one fiber enumeration. <br> *Closes the master plan's standing 'bounded kernel profile' open item.* | **CURRENT** | 2026-08-14 |
| `LAW_BUILDER_TRACE_FINDING.md` | Where a law call spends its time, and why the law-only builder branch is dead. <br> *Branch killed by measurement.* | **HISTORICAL** | 2026-08-16 |
| `LAW_CACHE_PARITY.json` | Cold-vs-warm law-cache parity with trajectory and checksum mismatch counts. <br> *Its loader drops unparseable replicates with no counter (hazards class 5 CRITICAL).* | **HISTORICAL** | 2026-08-17 |
| `LAW_CACHE_PROBE.json` | Law-cache probe: exactness, mean bytes, mean enumerate/load seconds. | **HISTORICAL** | 2026-08-17 |
| `LAW_PARITY_BASELINE.json` | Baseline mean seconds per law call before the admission-mask work. | **HISTORICAL** | 2026-08-16 |
| `LAZY_HEAD_PARITY.json` | Level-1 raw-head parity between the lazy and eager samplers. <br> ***Caveat:** `ok: true` can be written while the `_family_base` comparison silently never ran - REPRODUCIBILITY_HAZARDS class 5 CRITICAL.* | **CURRENT** | 2026-08-17 |
| `LAZY_SAMPLER_BENCH.json` | Lazy vs eager transition timings: mean/median/p90/p95/p99, fallback fraction, resolver calls. <br> *The measurement behind the ~79 ms figure.* | **CURRENT** | 2026-08-17 |
| `LAZY_SAMPLER_RESULT.md` | Lazy family-first sampler: same law, same chemistry, ~79 ms instead of seconds. Banked as the end of transition-level optimisation. <br> *Describes the runtime in use.* | **CURRENT** | 2026-08-17 |
| `LEARNED_REACHABILITY_CONTROLLER.md` | Design + preregistration of the learned reachability controller, written after Policy B failed 0/320. <br> *The design of the h_phi stack now in use.* | **CURRENT** | 2026-08-14 |
| `LEGACY_QUARANTINE.md` | Proof that no legacy path is reachable-and-silently-wrong from the canonical recipe. | **HISTORICAL** | 2026-07-28 |
| `LONGRANGE_AUC.json` | Long-range AUC of the H40 head at every lookahead 4-40. <br> *The H40 head was worse at every lookahead; line closed.* | **HISTORICAL** | 2026-08-17 |
| `MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md` | **The governing experimental plan.** Doctrine, resource-accounting rule, the current experiment map (1A-6), QED/GrIDDD protocol. Lines 1-425 govern; line 426 opens `ARCHIVED PROVENANCE`. <br> *Two `CURRENT AMENDMENT` sections (L1141, L1740) sit *below* the file's own non-governing divider - unresolved, see REGISTRY_AUDIT §7.5.* | **CURRENT** | 2026-08-15 |
| `MEMORY_PROBE.json` | Peak and steady GiB by stage, with a law-ok flag. | **HISTORICAL** | 2026-08-17 |
| `MMP_V2_PACK_CACHE_DECISION_2026-07-30.md` | MMP exact-state pack completion and the V2 reuse decision; authorised as an immutable development artifact. | **HISTORICAL** | 2026-07-30 |
| `MODAL_COST_MODEL.md` | Verified Modal rate card and the runtime class it applies to. <br> *'Use this and nothing else' for pricing a run.* | **CURRENT** | 2026-08-15 |
| `MOLLEO_BASIN_ANCHORS.json` | Basin anchors: 42/96 reverse-verified routes; one-step fiber max median 0.365. | **HISTORICAL** | 2026-08-19 |
| `MOLLEO_BRIDGE_PAIRS.json` | Bridge probe pairs - each random-ZINC start with its most similar held-out active. <br> *Similarity 0.21 -> 0.51; none reached 0.5 JNK3.* | **HISTORICAL** | 2026-08-19 |
| `MOLLEO_DEV_COHORT.json` | Frozen MOLLEO dev cohort: seed 100, 120 molecules, 24 roots by sha256(smiles), with input hashes. | **CURRENT** | 2026-08-18 |
| `MOLLEO_DEV_LABELS.json` | Objective order, labels and preference vectors for the dev cohort. | **CURRENT** | 2026-08-19 |
| `MOLLEO_ENV_GATE.json` | MOLLEO environment gate: pins held, all five oracles score, TDC QED == our RDKit QED exactly. <br> *The frozen SHA-pinned bundle `artifacts/oracles/molleo_task3_v1/` replaced this TDC path; SESSION_RUN_MANIFEST says do not reintroduce that dependency. Also records `tdc: unknown` - the one unpinned package.* | **SUPERSEDED** | 2026-08-18 |
| `MOLLEO_H40_VISITED.json` | The eight molecules visited under the H40 controller during the MOLLEO probe. <br> *Bare JSON list, no provenance.* | **HISTORICAL** | 2026-08-19 |
| `MOLLEO_TASK3_LANE_SUMMARY.md` | MOLLEO Task 3 lane summary and handoff: two mechanism families closed with diagnoses; harness reusable. <br> *'Banked and paused'. Predates the 2026-08-19 bridge result (`b800c6e`) and does not cover it.* | **HISTORICAL** | 2026-08-16 |
| `MONOTONICITY_LOCALIZE.json` | Are monotonicity violations concentrated in failures? No - Spearman 0.996 under isotonic projection. <br> *A negative that closed a proposed repair.* | **CURRENT** | 2026-08-18 |
| `NOVELTY_POSITIONING.md` | Earlier novelty and positioning argument. <br> *Carries a 'SUPERSEDED THESIS - 2026-07-28' banner pointing at PAPER1_FRAMING_AUTHORITATIVE.md, which is itself ARCHIVED.* | **SUPERSEDED** | 2026-07-28 |
| `OPERATOR_ONTOLOGY.md` | Operator-naming ontology for RingCore-V1: two namespaces, homonym deviations. | **HISTORICAL** | 2026-07-29 |
| `ORACLE_BATCH_INVARIANCE_DEFECT.md` | The DRD2 oracle is batch-size dependent - one line, and a reproducibility hazard. <br> *DEFECT RECORDED, NOT FIXED. Fixing it re-baselines frozen results.* | **CURRENT** | 2026-08-13 |
| `ORACLE_BUDGET_SEMANTICS_AND_P0.md` | Project-wide ruling on what a <=10,000 oracle budget counts, plus the P0 amortization/partition audit. <br> *Binds every cost claim.* | **CURRENT** | 2026-08-14 |
| `P0C_VERDICT.md` | P0c verdict: the preference continuum does NOT collapse (outcome 2). <br> *Its consequence, the budgeted preference sweep, then failed - see K41_BUDGETED_SWEEP_RESULT.md.* | **HISTORICAL** | 2026-08-14 |
| `PAPER1_FRAMING_AUTHORITATIVE.md` | Paper 1 authoritative framing record (RGM-first, trans-dimensional). <br> *ARCHIVED banner; named in EXPERIMENT_PLAN's 'do not read, cite or execute' list. ~13 inbound references depend on it, so it stays where it is.* | **SUPERSEDED** | 2026-08-10 |
| `PAPER_MASTER_PLAN.md` | Paper master plan - source-conditioned molecular optimization via valid-path generator matching. <br> *ARCHIVED - DO NOT USE banner.* | **SUPERSEDED** | 2026-08-10 |
| `PAPER_POSITIONING_EXACT_CONTROL.md` | Paper positioning: pathwise-meaningful, validity-closed molecular control. <br> *ARCHIVED - DO NOT USE banner.* | **SUPERSEDED** | 2026-08-10 |
| `PAPER_REFRAME_CONTROL_SUBSTRATE.md` | Paper reframe - RGM as a control-closed molecular substrate (anytime Pareto editing). <br> *ARCHIVED - DO NOT USE banner.* | **SUPERSEDED** | 2026-08-10 |
| `PAPER_REWRITE_BRIEF.md` | Paper rewrite brief for the control-substrate / anytime-Pareto framing. <br> *ARCHIVED - DO NOT USE banner.* | **SUPERSEDED** | 2026-08-10 |
| `PARETO_COMPARATOR_MATRIX_REQUIREMENT.md` | Comparator scope-lock: every claim-bearing experiment needs an explicit comparator rationale recorded before launch. <br> *Corrects the absolute reading of 'published-number-first'.* | **CURRENT** | 2026-08-14 |
| `PARETO_CONTROL_SMOKE_RESULT.md` | Target-free Pareto control, held-in smoke on 12 sources, K=8; every instrument frozen beforehand. | **HISTORICAL** | 2026-08-13 |
| `PARETO_DEVELOPMENT_PREREGISTRATION.md` | Target-free Pareto development preregistration: four frozen questions, instrumentation fixes, escalation rule. <br> *Its arm hierarchy is superseded by PARETO_DEVELOPMENT_REVISED_PREREGISTRATION.md; the four questions and stop rules were carried forward.* | **SUPERSEDED** | 2026-08-13 |
| `PARETO_DEVELOPMENT_REVISED_PREREGISTRATION.md` | Revised Pareto development preregistration after repaired P3/P4. <br> *Marked DESIGN ONLY / NOT LAUNCHED on 2026-08-14. The plan map records 4A/4B as banked but no document ties them to this revision.* | **UNKNOWN** | 2026-08-14 |
| `PARETO_ORACLE_REQUEST_AUDIT.md` | Audit: is gen_rank's doubled raw_oracle_calls algorithmic demand or harness overhead? (Overhead; the earlier framing is withdrawn.) | **HISTORICAL** | 2026-08-13 |
| `PARETO_PARITY_ENVIRONMENT_STATUS.md` | Status of the local parity environment needed to replay the 12-source smoke against the real checkpoint. | **HISTORICAL** | 2026-08-13 |
| `PARETO_SMOKE_FULL_SUMMARY.md` | Final 12-source summary of the same held-in Pareto smoke. | **HISTORICAL** | 2026-08-13 |
| `PARETO_TOPUP_FALLBACK_PREREGISTRATION.md` | Generate-and-rank top-up fallback rule, frozen before source 000's verdict with the volume verified empty. | **HISTORICAL** | 2026-08-13 |
| `PARTICLE_MULTIPLICITY.json` | 94.7% of particle-states are already unique; ceiling 1.13x on hard sources. <br> *Bounds what deduplication could ever buy.* | **CURRENT** | 2026-08-18 |
| `PATHWISE_HELDOUT_CONFIRMATION_PREREGISTRATION.md` | Pathwise control held-out confirmation preregistration: prevalence criterion, noninferiority margin, sample size, arms. <br> *DESIGN ONLY / NOT LAUNCHED; the plan map places Pathwise (block 6) after controller freeze.* | **HISTORICAL** | 2026-08-13 |
| `PATHWISE_STAGE_B_RESULT.md` | Pathwise control Stage B result (cLogP corridor family), explicitly DEVELOPMENTAL pending its own held-out confirmation. | **HISTORICAL** | 2026-08-13 |
| `PHASE0B_DATA_RECIPE.md` | B-edit universal-prior training corpus recipe; percentages explicitly PROPOSED, NOT LOCKED. | **HISTORICAL** | 2026-07-26 |
| `PHASE0_RESULTS.md` | Phase 0 validity-runtime results, run 2026-07-14. | **HISTORICAL** | 2026-07-18 |
| `PLAUSIBILITY_AND_PURPOSE.md` | Canonical framing of the architecture: legal dynamics + goal-independent R_theta + goal-dependent h_phi. <br> *Supersedes any description of the controller as an add-on.* | **CURRENT** | 2026-08-14 |
| `PROCESS_V2_IMPLEMENTATION_REPORT.md` | Process V2 atom-deletion implementation report (branch, base, head, worktree). | **HISTORICAL** | 2026-08-02 |
| `PROCESS_V2_P50_CACHE_ORDER_DIAGNOSTIC.md` | Diagnostic for the one unexplained whole-suite failure carried out of Wave 1 (stale successor-cache plan). | **HISTORICAL** | 2026-08-03 |
| `PROCESS_V2_WAVE1_CHECKPOINT.md` | Process-V2 Wave 1 review checkpoint (sections 3-4 + protocol fixtures + adversarial pass). | **HISTORICAL** | 2026-08-03 |
| `PROCESS_V2_WAVE2A_CHECKPOINT.md` | Process-V2 Wave 2A: compatible evidence boundary and cache-fed rebind. Nothing remote ran. | **HISTORICAL** | 2026-08-03 |
| `PRODUCTION_PREFLIGHT_REPORT.md` | Production preflight for B-edit (max_atoms=40, broad-organic); stops before any A100 optimizer update. | **HISTORICAL** | 2026-07-27 |
| `PROGRAM_COHERENCE_REPORT.md` | Coherence audit verdict `COHESIVE_WITH_QUARANTINED_LEGACY` at HEAD 4204c9d. | **HISTORICAL** | 2026-07-28 |
| `PROGRAM_CONTRACT.md` | The one canonical scientific program (RingCore-V1), derived from committed code at HEAD 4204c9d, with a contract fingerprint. <br> *Pins a 2026-07-28 code state; the frozen line is now `run_v2_01`/Process-V2.* | **HISTORICAL** | 2026-07-28 |
| `PROJECT_BOARD.md` | Durable task board - in flight / landed / blocked / on hold / carried forward. <br> *359 commits behind; two hard falsehoods (C0 shown in-flight though refuted; P1 code-commit blocker false). REGISTRY_AUDIT §4. P2/P3 rows still true.* | **SUPERSEDED** | 2026-08-11 |
| `PROJECT_STATUS.md` | Dated project status of 2026-07-19 with a 'Live override' block about the step-6,250 pancake model. <br> *Describes a retired checkpoint as live work; contradicted by ARTIFACT_INDEX's own lines 29-30 and by CLAIM_LEDGER S2.* | **SUPERSEDED** | 2026-07-20 |
| `PROSPECTIVE_SIGNAL.json` | Prospective hit/miss AUC of the controller signal. | **HISTORICAL** | 2026-08-17 |
| `PROSPECTIVE_TWO_HEAD.json` | Two-head prospective comparison: parity 1.49e-07, transition-level AUC old vs new. | **HISTORICAL** | 2026-08-17 |
| `QED_COVERAGE_LADDER.md` | Ladder method: spend further attempts only on UNSOLVED sources, since the metric is source success. | **CURRENT** | 2026-08-16 |
| `QED_DEVELOPMENT_CADENCE.md` | The development cadence standard from 2026-08-15; retires the slow serial SMC reference. | **CURRENT** | 2026-08-15 |
| `QUARANTINE_PRE_FREEZE_SMOKE.json` | Quarantine record for a pre-freeze smoke artifact: `PRE_FREEZE_SMOKE_DO_NOT_USE`, with disposition and harmlessness argument. <br> *A model quarantine record.* | **CURRENT** | 2026-08-14 |
| `QUEUED_EXPERIMENTS.md` | Ideas recorded at conception with their trigger conditions, so a later run cannot be mistaken for post-hoc design. <br> *Nothing here is designed or running by construction.* | **CURRENT** | 2026-08-14 |
| `RDKIT_VALIDITY_FINDING.md` | Profile finding: the validity check rebuilds the whole molecule per candidate (52.9x available). <br> *'SUPERSEDED IN PART' by LAZY_SAMPLER_RESULT.md - true of the eager path only; ceilings derived from it no longer describe the runtime.* | **SUPERSEDED** | 2026-08-17 |
| `REACHABILITY_GEOMETRY.md` | Canonical framing: what the horizon diagnostic is for (route existence and geometry, not hyperparameter search). | **CURRENT** | 2026-08-14 |
| `RECEDING_HORIZON_AB.json` | Earlier receding-horizon A/B over n=60. <br> *Declared stale by RECEDING_HORIZON_AB_64.json: 'reported n=60 and disagrees at k=1; it is stale'.* | **SUPERSEDED** | 2026-08-17 |
| `RECEDING_HORIZON_AB_64.json` | H24-vs-H40 receding-horizon A/B recomputed over all 64 complete records: 32/64 -> 36/64. <br> *Carries its own supersession notice for the file below. No writer found anywhere in the repo (REPRODUCIBILITY_HAZARDS class 4).* | **CURRENT** | 2026-08-18 |
| `RECORD_DIFF.json` | Per-slot record diff used while chasing a cache-order discrepancy. | **HISTORICAL** | 2026-08-17 |
| `REFERENCE_LAW_ABLATION_PREREGISTRATION.md` | Reference-law ablation preregistration, fixed before any Pareto controller outcome existed. <br> *No result document references it; whether the ablation ran is not recorded.* | **UNKNOWN** | 2026-08-13 |
| `REGISTRY_AUDIT_2026-08-19.md` | Claim-by-claim currency audit of the four registries above; establishes which plan governs. <br> *Read this before trusting any index in this repo.* | **CURRENT** | 2026-08-19 |
| `REJECTION_DIAGNOSTIC.json` | Rate factorization tables behind the rejection sampler. | **CURRENT** | 2026-08-17 |
| `REJECTION_EXACTNESS.json` | Rejection-sampler exactness: worst within-family spread, worst abs log difference, mismatches. | **CURRENT** | 2026-08-17 |
| `RELATED_WORK_MATRIX.md` | Property-by-property related-work ledger; the novelty claim is a conjunction, so it needs rows not prose. <br> *Carries its own verification debt list for `~` cells.* | **CURRENT** | 2026-07-30 |
| `RELATED_WORK_POSITIONING.md` | Positioning against the discrete-edit-process lineage; states Pareto control is NOT the core novelty. | **CURRENT** | 2026-08-13 |
| `REPRODUCIBILITY_HAZARDS_2026-08-19.md` | Five-class hazard audit: ephemeral paths, missing data, unpinned deps, results without provenance, silent-failure handlers. <br> *Read before reproducing any banked number.* | **CURRENT** | 2026-08-19 |
| `RESAMPLE_RATE_BY_STRATUM.json` | Resample rate by stratum with an explicit reading note and WE scope. | **CURRENT** | 2026-08-17 |
| `RETARGETING_SAME_PREFIX_DESIGN.md` | Same-prefix goal intervention: design plus what the census already decided. Steps 1-2 specified, step 3 unrun. <br> *Plan map records 3B Retargeting as banked, but this document was never updated to say whether it is the same object.* | **UNKNOWN** | 2026-08-13 |
| `RINGCORE_V1_ACCURACY_AND_ARCHITECTURE_AUDIT_2026-07-30.md` | Accuracy and architecture audit of the completed source-conditioned editing run. | **HISTORICAL** | 2026-07-30 |
| `RINGCORE_V1_PREFLIGHT.md` | RingCore-V1 preflight mandate: freeze the compositional ring core, bounded preflight, macro decision. | **HISTORICAL** | 2026-07-28 |
| `RINGCORE_V1_SUCCESSOR_LEADERBOARD_READINESS_2026-07-30.md` | Canonical-successor leaderboard readiness: specified and tested, deliberately not authorised to execute. | **HISTORICAL** | 2026-07-30 |
| `RING_COMPOSITIONAL_P1_SPEC.md` | P1 implementation spec for compositional cycle_close/cycle_open (the coupled core). | **HISTORICAL** | 2026-07-27 |
| `RING_OPERATOR_DECISION_CONTRACT_V1.md` | Ring-operator decision contract: `primitive_ringcore` stays primary until a paired comparison beats it. <br> *A fail-closed default recorded 2026-07-30.* | **HISTORICAL** | 2026-07-30 |
| `RING_REWRITE_ARCHITECTURE.md` | The reversible micro instruction set behind ring rewriting. | **HISTORICAL** | 2026-07-18 |
| `SAMPLER_DISTRIBUTION.json` | Sampler distribution check: worst total variation, any illegal draw, n draws. | **CURRENT** | 2026-08-17 |
| `SAMPLER_FLOOR.json` | Per-component cost floor of a full law call, with and without cycle-open. <br> *One row is biased downward by a timing bug - the elapsed time is appended outside the try (hazards class 5 MEDIUM).* | **CURRENT** | 2026-08-17 |
| `SCALE16_GATE.md` | 1,333-molecule C/N/O/F scale gate: does the generator scale past the 128-molecule pilot? | **HISTORICAL** | 2026-07-18 |
| `SCALED_DATA_AND_TRAINING_STAGES.md` | Scaled-data build and training-execution stages; owner mandate of 2026-07-28. <br> *Self-labelled AUTHORITATIVE for the window between the data-starvation finding and the first editing-prior training run - that window has closed.* | **HISTORICAL** | 2026-07-28 |
| `SCOPE_LOCK_AND_KERNEL_COST.md` | Scope lock for experimental closeout - six locked pieces, barred additions - plus the kernel-cost blocker. <br> *Its per-piece state column is stale (P0c closed by P0C_VERDICT.md; the kernel blocker closed by KERNEL_COST_CHARACTERIZATION.md). Whether the lock still binds after the 64-source QED result is not recorded anywhere.* | **UNKNOWN** | 2026-08-14 |
| `SESSION_RUN_MANIFEST_2026-08-18.md` | Every experiment of the 2026-08-18 session: app, invocation, artifact, result, commit. States the git-vs-Modal-volume split. <br> *The best single reproduction entry point for the QED and MOLLEO lanes.* | **CURRENT** | 2026-08-19 |
| `SHORTLIST_RETENTION.json` | R_theta top-K retention by stratum: 0 of 21 hard-source rescue trajectories fully contained in top-128; median worst rank 271. <br> *The measurement that falsified AMENDMENT_SHORTLIST_RETENTION.md's premise.* | **CURRENT** | 2026-08-18 |
| `SHORTLIST_TASKS.json` | First successful candidate per solved dev source under the H40 frozen controller - input to the retention probe. <br> *Assembled by an inline script reading `/tmp` caches; the assembly step is not in the repo (SESSION_RUN_MANIFEST gap 1).* | **CURRENT** | 2026-08-18 |
| `SLACK_WE_RESULT.json` | Slack-stratified weighted-ensemble gate: decision recorded, reliable stratum broken. <br> *The WE / rare-event line closed here (`b535c05`).* | **HISTORICAL** | 2026-08-17 |
| `SMC_EFFICACY_PROBE_SOURCES.json` | The probe source indices, chosen before any SMC outcome existed. <br> *A blinding record: `chosen_before_any_smc_outcome`.* | **CURRENT** | 2026-08-15 |
| `SMC_LADDER_RUNG2.json` | Ladder rung 2 - candidate 2 run only for sources unsolved after candidate 1. | **CURRENT** | 2026-08-16 |
| `SMC_PROBE_MATCHED_BASELINE.json` | Matched 4-slot baseline for the SMC probe over the same 16 sources. | **CURRENT** | 2026-08-15 |
| `SMC_SENTINELS.json` | Extinction/coverage sentinels: purpose, rule, sentinel set. | **CURRENT** | 2026-08-16 |
| `STAGE6_SMOKE_REPORT.md` | Local end-to-end wiring + smoke-training gate; returned GO_FOR_PRODUCTION_PREFLIGHT. | **HISTORICAL** | 2026-07-26 |
| `STOCHASTIC_REWRITING_SCOPE.md` | What of the stochastic-rewriting formalism the base model adopts and what it does not. <br> *Scope statement, still accurate.* | **CURRENT** | 2026-07-18 |
| `SUCCESSOR_FIBER_TRAINING_INTEGRATION_DESIGN_2026-07-30.md` | Successor-fiber training integration design: address/cache/dataset bridge, indexed backend, support-invariance gate. | **HISTORICAL** | 2026-07-30 |
| `SYSTEM_CONTRACT.md` | Exact mathematics + software the code implemented, reconstructed from source with file:line citations (factorized tracelet family). <br> *2026-07-28 reconstruction, predates editing-V2 / Process-V2.* | **HISTORICAL** | 2026-07-28 |
| `TINY_RATE_GATE.md` | First learned whole-graph rate gate - the small enumerable experiment. | **HISTORICAL** | 2026-07-18 |
| `TRACE_COMPILER_PILOT.md` | Broad-corpus trace compiler pilot on 1,000 held-out GuacaMol entries. | **HISTORICAL** | 2026-07-18 |
| `TREE_PANEL_12.json` | 12-source persistent-tree panel by stratum. <br> *Persistent-archive line closed.* | **HISTORICAL** | 2026-08-17 |
| `TREE_SOURCE_TRANSPORT.md` | Decision on structured tree sources and valid-state transport. | **HISTORICAL** | 2026-07-18 |
| `V3_REUSE_LEDGER.md` | What COMPOSE v4 copied from v3 and what it did not. | **HISTORICAL** | 2026-07-18 |
| `VALID128_CURVE.json` | Same 128-source panel extended to k=12: coverage 0.546875 vs GrIDDD's reported 45.1% at k=20. <br> *Producer `scripts/hphi_valid128_read.py`, recoverable only by grep.* | **CURRENT** | 2026-08-18 |
| `VALID128_K8_RESULT.json` | **Headline validation.** Prospective 128-source panel at k=8: coverage 0.4921875, 1,024 runs, 1,167,787 work transitions. <br> *Controller frozen in advance by AMENDMENT_VALIDATION_128.md. No provenance block (no git commit, no producing script recorded in-file).* | **CURRENT** | 2026-08-18 |
| `audits/2026-07-18_objective_aware_ring_support.md` | Objective-aware exact ring support; H100 preflight diagnosis. | **HISTORICAL** | 2026-07-19 |
| `audits/2026-07-19_pancake_to_quotient_run_audit.md` | Separates changes to the scientific generator from changes that only move or cache computation. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_agile_ugi_corpus_bias.md` | AGILE/Ugi structural-bias and layered-sampling audit; cap AGILE as an auxiliary Ugi layer. <br> *Lipid/Paper-2 lane.* | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_calibrated_pancake_ring_underproduction_root_cause.md` | Root cause of ring/cycle-rank underproduction in the calibrated pancake sampler. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_canonical_successor_distillation_design.md` | Canonical-successor distillation repair design. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_conditional_backbone_selection.md` | Conditional-backbone evidence table over frozen artifacts. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_griddd_analytic_zero_sidecar_rewrite_smoke.md` | Qualified analytic zero-sidecar GrIDDD rewrite smoke; execution gate passed, efficacy untested. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_griddd_conditional_execution_contract.md` | Controlled conditional execution contract; interface and bounded smoke complete. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_griddd_qed_benchmark_inputs.md` | GrIDDD QED benchmark input freeze: the public Jin list is recoverable, the released 800-row artifact is not. <br> *Still the authority for why the 800 cannot be labelled 'official GrIDDD 800'; enforced by `scripts/build_griddd_qed_lead_manifest.py`.* | **CURRENT** | 2026-07-20 |
| `audits/2026-07-20_lung_oracle_results.md` | Verified LuT, LUMI and LNPDB lung-oracle baselines. <br> *Lipid/Paper-2 lane.* | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_pan_lung_manifest_status.md` | Pan-lung corpus manifest and oracle-matrix status. <br> *Lipid/Paper-2 lane.* | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_pancake_ring_fusion_loss_decomposition.md` | Fused-ring loss decomposition and root fix; kills earliest commutation as the immediate fix. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_qed_conditional_pilot.md` | Direct-QED conditional pilot audit with checkpoint SHA-256. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_tomorrow_griddd_claim_gate.md` | Claim-and-evidence gate to stop a pilot being presented as a broader conditional-generation win. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_unconditional_chemistry_failure_audit.md` | Read-only attribution of the calibrated step-6,250 unconditional rollout failures. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_unconditional_reporting_and_conditional_transition.md` | What COMPOSE must establish unconditionally before the property-targeting program proceeds. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_unconditional_root_fix_execution.md` | Measured failures, running repairs and future decisions for the unconditional root fix. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-07-20_valid_state_conditional_control_design.md` | Valid-state conditional control architecture decision for the first production QED experiment. | **HISTORICAL** | 2026-07-20 |
| `audits/2026-08-12_sealed67_protocol_deviations.md` | Sealed 67-pair panel: protocol deviations and bookkeeping defects, disclosed for the reproducibility appendix. <br> *Belongs in the paper appendix; neither item changed the controller or a reported outcome.* | **CURRENT** | 2026-08-12 |
| `audits/semantic_ring_teacher_audit_v4.json` | Semantic ring teacher audit v4: examples, failures, per-family counts, ring electronic mode. | **HISTORICAL** | 2026-07-18 |
| `presentation_figures/aim2_e0_doob_exactness_actual.png` | Presentation figure: E0 Doob exactness. <br> *Illustrates a foundation result that still holds (exact conditional to 1.1e-16).* | **CURRENT** | 2026-07-21 |
| `rendered/GRIDDD_JIN_PROTOCOL.html` | Rendered copy of the GrIDDD/Jin protocol. <br> *Render of a source that carries a partial-supersession banner.* | **CURRENT** | 2026-08-15 |
| `rendered/HORIZON_AMENDMENT_H24.html` | Rendered copy of the H24 horizon amendment. | **CURRENT** | 2026-08-15 |
| `rendered/HPHI_QED_PREREGISTRATION.html` | Rendered copy of the QED/h_phi preregistration. | **CURRENT** | 2026-08-15 |
| `rendered/MASTER_PLAN.html` | Rendered copy of the governing plan (2026-08-15). <br> *A render, not the source. If MASTER_PLAN.md changes, this goes stale silently.* | **CURRENT** | 2026-08-15 |
| `reports/README.md` | Explains that docs/reports holds the source of rendered status pages published as private artifacts. | **CURRENT** | 2026-08-13 |
| `reports/four_lane_status_2026-08-13.html` | Rendered four-lane status page for 2026-08-13. <br> *A dated snapshot of the parallel-lane period.* | **HISTORICAL** | 2026-08-13 |
| `research_plans/README.md` | Calls the three standalone HTML documents 'the canonical project strategy artifacts'. <br> *Contradicted by MASTER_PLAN and EXPERIMENT_PLAN; part of REGISTRY_AUDIT contradiction C1.* | **SUPERSEDED** | 2026-07-18 |
| `research_plans/compose_lipid_pretraining_corpus_audit.md` | COMPOSE-Lipid structural pretraining corpus audit; freeze a measured R0. <br> *Lipid lane.* | **HISTORICAL** | 2026-07-20 |
| `research_plans/compose_two_paper_execution_plan.html` | Integrated two-paper execution and dependency plan. <br> *Same 2026-07-20 framing.* | **SUPERSEDED** | 2026-07-20 |
| `research_plans/defensible_lipid_corpus_spec.md` | Executable design spec for a defensible lipid structural corpus. <br> *Lipid lane.* | **HISTORICAL** | 2026-07-20 |
| `research_plans/lipid_dataset_intake.md` | Lipid dataset intake evidence ledger. <br> *Lipid lane.* | **HISTORICAL** | 2026-07-20 |
| `research_plans/pan_lung_oracle_corpus.md` | Pan-lung oracle corpus and steering plan. <br> *Lipid lane.* | **HISTORICAL** | 2026-07-20 |
| `research_plans/paper1_compose_methods.html` | COMPOSE RGM methods-paper plan (standalone styled HTML). <br> *2026-07-20 framing; the governing plan is MASTER_PLAN.* | **SUPERSEDED** | 2026-07-20 |
| `research_plans/paper2_compose_lipid.html` | COMPOSE-Lipid translational (Paper 2) plan. <br> *Zero mentions of 'lipid' in any current plan and no commits in 200+. Parked or lapsed - REGISTRY_AUDIT could not determine which.* | **UNKNOWN** | 2026-07-20 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/README.md` | Index of the saved 2026-07-18 trajectory-diagnostic bundle for the step-6,250 pre-quotient model. <br> *ARTIFACT_INDEX warns these must not be presented as results of the corrected model.* | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/eval2000_summary_step6250.png` | Eval-2000 summary plot at step 6,250. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32.json` | Full ancestral trajectory, index 32 - the gauge-churn failure case. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32.png` | Rendered trajectory, index 32. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32_page01.png` | Paginated render of trajectory 32, page 1. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32_page02.png` | Paginated render of trajectory 32, page 2. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32_page03.png` | Paginated render of trajectory 32, page 3. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32_page04.png` | Paginated render of trajectory 32, page 4. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index75.json` | Full ancestral trajectory, index 75. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index75.png` | Rendered trajectory, index 75. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index75_page01.png` | Paginated render of trajectory 75, page 1. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index75_page02.png` | Paginated render of trajectory 75, page 2. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_summary.json` | Machine-readable summary of the legacy trajectory bundle. | **HISTORICAL** | 2026-07-18 |
| `trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/trajectory_event_audit_step4750.png` | Trajectory event audit at step 4,750. | **HISTORICAL** | 2026-07-18 |
| `workstreams/AGENT_HANDOFF_TEMPLATE.md` | Template to copy to `docs/workstreams/<workstream>/HANDOFF.md`. <br> *A template - no currency to lose.* | **CURRENT** | 2026-08-12 |
| `workstreams/EXTERNAL_BASELINE_INDEX.md` | Where every external-baseline finding lives, after consolidation into the main lane. <br> *Its GrIDDD demotion is superseded by AMENDMENT_EDITFLOWS_GRIDDD_HEADLINE.md.* | **CURRENT** | 2026-08-14 |
| `workstreams/PARALLEL_WORKSTREAMS_AND_HANDOFF.md` | Charter for the parallel experimental lanes and the handoff protocol; states the paper-level causal chain. <br> *The four-lane structure ran 2026-08-13/14; work has since consolidated into the main lane.* | **HISTORICAL** | 2026-08-13 |
| `workstreams/baseline-qualification/COMPARATOR_MATRIX.md` | Rendered comparator registry v3 (generated - edit the JSON and re-run the renderer). <br> *Generated file; do not hand-edit.* | **CURRENT** | 2026-08-14 |
| `workstreams/baseline-qualification/CONVENTIONAL_SUITE.md` | The conventional optimization suite, frozen 2026-08-13 before any COMPOSE-versus-baseline outcome. | **HISTORICAL** | 2026-08-14 |
| `workstreams/baseline-qualification/DECISION_LOG.md` | Lane D decision log: each decision, the evidence before it, and whether it changes a frozen object. | **HISTORICAL** | 2026-08-14 |
| `workstreams/baseline-qualification/FAIRNESS_CONTRACT.md` | What 'matched budget' means per MUST_RUN comparator, and where each comparison stops being fair. | **CURRENT** | 2026-08-14 |
| `workstreams/baseline-qualification/FAIRNESS_MATRIX.md` | The final fairness matrix, method x task, frozen before any outcome. | **CURRENT** | 2026-08-14 |
| `workstreams/baseline-qualification/HANDOFF.md` | Lane D handoff: external-baseline qualification and adapter readiness (claim C4a-C4d). | **HISTORICAL** | 2026-08-14 |
| `workstreams/baseline-qualification/PROTOCOL.md` | Lane D scientific protocol - the contract under which external comparators may later be run. <br> *DESIGN_ONLY by construction; not overtaken, and the comparator roles it implements are still canonical.* | **CURRENT** | 2026-08-14 |
| `workstreams/baseline-qualification/STATUS.md` | Lane D status: DESIGN_ONLY plus one SMOKE_HELD_IN instrument check; HOLDING by instruction. | **HISTORICAL** | 2026-08-14 |
| `workstreams/baseline-qualification/comparator_registry_v3.json` | Machine-readable comparator registry v3: adapter rules, evidence standard, fairness rules, cost ledgers. <br> *Source of truth for COMPARATOR_MATRIX.md.* | **CURRENT** | 2026-08-14 |
| `workstreams/baseline-qualification/handoff.json` | Machine-readable Lane D handoff manifest. | **HISTORICAL** | 2026-08-14 |
| `workstreams/constraints-hard/BASELINE_TASK_MATRIX.md` | External baseline task matrix for hard structural constraints; unconfirmed cells marked UNVERIFIED. | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/CONSTRAINTS_SECTION_TRILEMMA.md` | The constraint trilemma - the organizing claim for the constraints section; every satisfaction rate carries its denominator. | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/CONSTRAINT_SEMANTICS.md` | Stage 0 executor audit of constraint semantics; local CPU-only probe, no checkpoint needed. | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/DECISION_LOG.md` | Lane 6 decision log, all decisions 2026-08-13 at base `f6146d7`. | **HISTORICAL** | 2026-08-14 |
| `workstreams/constraints-hard/EXTERNAL_HARD_CONSTRAINT_AUDIT.md` | External hard-constraint benchmark audit; every capability cell cites a paper section or repo file. <br> *One of the three findings accepted into project record.* | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/HANDOFF.md` | Lane 6 handoff, headed by the 2026-08-13 note that the lane was narrowed and is COMPLETE. | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/LANE2_FIVE_QUESTIONS.md` | Five framing questions for the hard-constraint lane. <br> *'SUPERSEDED IN PART' banner; the authoritative version is SA_CENSUS_PROTOCOL.md §7 and §10.* | **SUPERSEDED** | 2026-08-14 |
| `workstreams/constraints-hard/PROTOCOL.md` | Lane 6 internal Experiment A design. <br> *Carries a `SUPERSEDED` banner: 'withdrawn and must not be implemented'.* | **SUPERSEDED** | 2026-08-14 |
| `workstreams/constraints-hard/SAME_LAB_LINEAGE.md` | Same-lab lineage and positioning for the constraints lane. | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/SA_CENSUS_APPLICABILITY.md` | Model-free half of the SA census: source applicability, held-out never opened. | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/SA_CENSUS_PROTOCOL.md` | Held-in feasibility census for the externally defined SA constraint; fixes the tau-selection rule before any measurement. <br> *The authoritative version of LANE2_FIVE_QUESTIONS.* | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/SCAFFOLD_FEASIBILITY.md` | Stage 1 protected-core feasibility census. <br> *Explicitly stands as a measurement even though the experiment it fed (PROTOCOL.md) is withdrawn.* | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/STATUS.md` | Lane 6 status: COMPLETE. Scope narrowed to an external audit; nothing internal designed or run. <br> *The lane's closing record.* | **CURRENT** | 2026-08-14 |
| `workstreams/constraints-hard/handoff.json` | Machine-readable Lane 6 handoff manifest (`status: DESIGN_ONLY`). | **HISTORICAL** | 2026-08-14 |
| `workstreams/constraints-hard/probes/gate0_local_authentication_repro.py` | Probe reproducing the Gate-0 local authentication path. | **HISTORICAL** | 2026-08-14 |
| `workstreams/constraints-hard/probes/identity_probe.py` | Probe checking identity fields without the R_theta checkpoint. | **HISTORICAL** | 2026-08-14 |
| `workstreams/multiobjective/BASELINE_TASK_MATRIX.md` | Comparator matrix organized by ROLE; every cost figure a published figure or a projection, never a measurement. | **CURRENT** | 2026-08-14 |
| `workstreams/multiobjective/BENCHMARK_ALIGNMENT_AUDIT.md` | Published-protocol alignment audit for GSK3b / JNK3; every cell resolves to a file:line or paper section. <br> *Directly relevant to the live InversionGNN 2-objective lane.* | **CURRENT** | 2026-08-14 |
| `workstreams/multiobjective/DECISION_LOG.md` | Lane 5 decision log with pre-decision evidence and rejected alternatives. | **HISTORICAL** | 2026-08-14 |
| `workstreams/multiobjective/EDITING_COMPETENCE_AUDIT.md` | Ignoring Pareto entirely: is COMPOSE a credible source-conditioned molecular editor? | **CURRENT** | 2026-08-14 |
| `workstreams/multiobjective/FRAMEWORK_NEIGHBOR_GRIDDD.md` | Qualification of GrIDDD as a FRAMEWORK_NEIGHBOR under the canonical comparator roles. <br> *GrIDDD is now one of the two headline neighbours.* | **CURRENT** | 2026-08-14 |
| `workstreams/multiobjective/HANDOFF.md` | Lane 5 handoff: multiobjective evidence package, external-comparator half only. | **HISTORICAL** | 2026-08-14 |
| `workstreams/multiobjective/PROTOCOL.md` | Lane 5 scientific contract for the external-baseline and fairness half of the Pareto block. | **CURRENT** | 2026-08-14 |
| `workstreams/multiobjective/SAME_LAB_LINEAGE.md` | Same-lab lineage audit and the binding complementary-framing rule. | **CURRENT** | 2026-08-14 |
| `workstreams/multiobjective/SOURCE_CONDITIONING_AUDIT.md` | Source-conditioned versus global Pareto - the fairness audit. | **CURRENT** | 2026-08-14 |
| `workstreams/multiobjective/STATUS.md` | Lane 5 status: redirected 2026-08-13; Stage 0/1 accepted; alignment audit complete on the deciding dimensions. | **HISTORICAL** | 2026-08-14 |
| `workstreams/multiobjective/handoff.json` | Machine-readable Lane 5 handoff manifest. | **HISTORICAL** | 2026-08-14 |
