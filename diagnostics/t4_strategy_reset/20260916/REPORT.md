# T4 strategy reset: keep Dynamic search, distill productive moves, allocate by measured utility

Date: 2026-09-16. This is a retrospective forensic audit and a proposed development plan, not a new controller result. No docking calls, model fits, or Modal worker launches were made. Existing experimental artifacts and runtime code were not modified. Historical artifacts were recovered by read-only Modal volume access.

## Recommendation

Build **route-informed, objective-driven Dynamic program search**, not another offline model tasked with predicting an entire winning patch before seeing any objective feedback.

Reuse the exact v0 proposal mechanisms, the useful v1 operators, protected execution, the corrected region realizer, and the shared program archive. Change three things: remove artificial proposal-support truncation/repeated cold starts; learn a compact, joint context-conditioned prior over productive generic program constructions; and use actual within-run docking outcomes for quality/diversity-aware parent and proposal allocation. Keep an explicit exploratory lane.

The first causal test should distinguish **search allocation** from **distilled proposal quality**. Merely mixing two generators with a bandit is not new: v2.1 already did that, incompletely and with an ill-aligned reward. No evidence currently guarantees this recommendation will match or beat IVG.

## 1. What was recovered and verified

The audit read the clean controller/evidence tree at revision `2114c405` and recovered 71 historical checkpoints. All normalized observations retain their original receipt identity and reference the exact source state, instantiated program, primitive trace, endpoint, target/cell, and oracle protocol in the recovered checkpoint.

| Historical arm | Scored observations | Exact call indices recovered | Parent score/provenance |
| --- | ---: | ---: | ---: |
| Full-146, 15 cells × 3 replicates | 25,084 | 3,274 | 24,622 |
| v0, original three δ=.4 cells | 2,336 | 2,336 | 2,321 |
| v1, three δ=.4 cells | 1,659 | 1,659 | 1,644 |
| v2.1, five δ=.4 cells | 2,936 | 2,936 | 2,896 |
| v0, fifteen δ=.6 cells | 3,880 | 3,880 | 3,845 |
| Total | **35,895** | **14,085** | **35,328** |

These are distinct receipt observations, not 35,895 independent molecules or sources. Replicates and related adaptive descendants are correlated. All rows have one unambiguous stored program entry. Failed charged calls are excluded from the numeric-score count, but remain in the query totals and historical artifacts. The other 21,810 Full-146 observations have scores and programs; their round histories were not downloaded in this bounded audit, so their exact call order is not inferred.

Additional scored artifacts were inventoried separately: program curricula, program-pool development, winner refinement, paired parent-edit cycles, matched reference/committor panels, and later utility acquisitions. Some contain inherited or replayed labels. They are **not added** to the receipt total without reconciliation. This is a verified lower bound on usable history, not a claim to have exhausted every historical label.

Consequently, the earlier 51-row utility experiment was a small selected dataset, **not the available T4 data universe**. The limiting issues are selection bias, source diversity, label quality, and appropriate conditioning, not merely an absence of scores.

Evidence: [audit.json](audit.json), [scored_rows.jsonl](scored_rows.jsonl), [search_audit.json](search_audit.json), recovery manifests, and [verification.json](verification.json).

## 2. Actual historical performance, including v1 and v2.1

Best score / charged calls at the recovered endpoint of each δ=.4 run, replicate 0; lower is better. Different stopping points make this a status table, not a matched-budget causal comparison.

| Cell | Full-146 | Dynamic-v0 | Dynamic-v1 | Dynamic-v2.1 |
| --- | --- | --- | --- | --- |
| 5HT1B seed 0 | −13.3 / 504 | −12.6 / 786 | **−14.0 / 500** | −13.7 / 503 |
| BRAF seed 1 | −11.0 / 511 | **−11.2 / 576** | −10.4 / 287* | −10.7 / 501 |
| JAK2 seed 1 | **−11.6 / 883** | −10.3 / 1,000 | −10.1 / 890* | −10.7 / 1,000 |
| PARP1 seed 0 | **−14.3 / 960** | Not in original panel | Not in panel | −11.7 / 1,000 |
| FA7 seed 0 | **−9.7 / 500** | Not in original panel | Not in panel | No score / 0 |

*V1 BRAF/JAK2 final-result files were unavailable; these are checkpoint observations, not claimed completed results.* V2's failed preflight is not a scored optimizer comparison. No complete δ=.4 fifteen-cell v0 result was recovered.

The early difference is real, not just a late-budget comparison:

| Cell, call 20 | Full-146 | v0 | v1 | v2.1 |
| --- | ---: | ---: | ---: | ---: |
| 5HT1B-0 | −13.1 | −12.1 | −11.7 | −12.3 |
| BRAF-1 | −10.5 | −10.4 | −9.6 | −10.3 |
| JAK2-1 | **−11.3** | −9.9 | −9.3 | −10.0 |
| PARP1-0 | **−13.5** | — | — | −9.8 |

JAK2 Full-146 was already −9.9 at call 1, versus v0 −9.6. The poor raw macro −6.8 and later structural-selector −8.3 were therefore weak even relative to historical early proposals. Iterative feedback is necessary, but cannot explain away a poor initial proposal distribution.

## 3. Dynamic-v0 is not a black box

The audited configuration, not the class defaults, determines these facts.

| Mechanism | Actual behavior | Implication |
| --- | --- | --- |
| Parent selection | `parent_allocation="score_blind"`; equal endpoint quality before duplicate penalties, with 20% exploration | Recorded scores did not preferentially allocate parents to good endpoints |
| Fresh synthesis | 13 generic families; 1–3 modules, ordinary probabilities .4/.4/.2 | Useful broad chemistry, but short generic compositions and mostly unguided parameters |
| Scheduler | 25% archive-program composition; remaining branch uses mutation/recombination 7:2 | Composition is a meaningful part of v0, not interchangeable with v1/v2.1 |
| Mutation | Half of v0 mutation dispatch is fresh generic synthesis; otherwise mutate an archived constructor/binding | A child often edits the parent's original constructor source, not its measured endpoint |
| Recombination | Transfers dependency-connected program branches, with context-compatible binding search | Preserves useful coordinated fragments, including created-handle dependencies |
| Length | Up to 32 primitives and 8 blocks per proposal; fresh generic module horizon 3 | v0 is not simply a three-primitive optimizer |
| WHERE/HOW | Generic compatibility sampling and bounded binding candidates; no learned docking-aware site distribution | Correct operation classes can still have very low effective proposal probability |
| Execution | Exact primitive legality; eligibility on the completed program | Protected execution already exists; do not rebuild it |
| Archive | Successful eligible novel scored endpoints are retained, including worse-scoring ones | Allows nonmonotone lineages; greedy admission would remove useful ancestors |
| Diversity | Endpoint deduplication, duplicate penalties, explicit exploration | No active quality/diversity allocation under the actual score-blind setting |
| Startup | At high atom count, deletion-heavy bootstrap; repeated empty rounds restart the same seed | Some cells never reach an archive from which general search can begin |
| Stopping | 128-round limit, three empty rounds, per-batch wall limit, call ceiling, competitive plateau | An early stop is not evidence that the whole chemical support was searched |

Source anchors: `control/dynamic_program_synthesis.py`; `control/adaptive_program_optimizer.py:309` (selection); `experiments/t4_frozen_program_benchmark.py:430` (runner). Exact file hashes are in the verification artifact.

### Search allocation is a measured problem, but not yet a proven causal fix

Among scored proposals with a recorded parent, JAK2 v0 spent **895/982** on parents more than 1 docking-score unit worse than the best already observed; v2.1 spent **934/968**. For v0 5HT1B this was **753/780**. This is consistent with the score-blind configuration. It does not prove that choosing only the best parent would improve outcomes: constructor mutation is not always a direct endpoint edit, and successful genealogies contain worse-scoring ancestors.

The existing `niche_score`/`score_rank` selection machinery offers a small, testable intervention. Keep an exploration floor and all provenance; change allocation, not scientific eligibility or archive truth.

## 4. What the successful lineages actually used

**BRAF v0, −11.2 at call 320.** The main genealogy contains deletion, ring append, repeated constructor mutation, then a program-composition step and mutation of that composition. The final proposal has three primitives: a carbonyl insertion and ring-state operations. A −8.8 ancestor occurs after a −10.7 ancestor. This directly motivates retaining composition, recombination, and non-greedy exploration.

**5HT1B v1, −14.0 at call 163.** The recorded main genealogy is double deletion → recombination → current-state bond rerouting → reroute mutation. It includes −8.7 and −8.3 ancestors. It does **not** establish that the newly added protected ring-construction operators caused this win. The simple local search/refinement capabilities matter.

**JAK2 v2.1, −10.7 at call 324.** The genealogy crosses shallow and structured channels; its later productive proposal combines substituent deletion with substituted-ring construction. Thus richer operators can help, but the measured result remains below Full-146.

**JAK2 Full-146, −11.6 at call 871.** A bank-initialized 15-primitive constructor is subsequently mutated and recombined. Full-146's advantage is not only literal replay: a productive initial program family gives subsequent search something valuable to vary.

These are genealogies, not claims that every listed edge is a direct edit of the preceding endpoint. `direct_edit_of_measured_parent` is recorded per edge in `audit.json`.

## 5. Teacher transformations versus Dynamic support

Across all 77 exact witness traces:

- Region counts: 26 one-region, 35 two-region, 13 three-region, 3 four-region; total 147.
- Primitive length: 5–32, median 17.
- Median changed original atoms: 12; median net deletions: 5; median net creations: 9.
- Median span between changed original atoms: 8 source-graph bonds. This is a span, not the radius of a single patch.
- 65/77 reuse a created handle later.
- Under the local recorded RDKit version, 76/77 have an ineligible internal prefix while all 77 final endpoints pass δ=.4 eligibility.

By contrast, measured v0 δ=.4 proposals have median 5 primitives, 3 changed original atoms, 1 deletion and 2 creations. V1/v2.1 scored proposals have median 2 primitives. These distributions are selection-biased toward successfully executed, eligible, scored proposals, and cover different cells. They show a scale mismatch in effective proposals, not a formal impossibility theorem.

For teacher roots with same-target, same-input scored histories, exact teacher endpoints appeared in **0/19 v0**, **0/19 v1**, and **0/25 v2.1** teacher comparisons. Median closest endpoint fingerprint similarities were .569, .569, and .625. Other teachers have no same-input history in these limited panels; they are abstentions, not failures. Alternative high-utility endpoints remain successes regardless of teacher identity.

### A decisive actual-sampler probe: JAK2

I reused the existing generic three-module decomposition of the Full-146 −11.6 transformation: pendant deletion → substituted-ring construction → ring-path remodeling.

With the targets supplied, the generic v1 compilers reach the exact endpoint. Through the **actual v1 candidate-panel mechanism**, conditional on the same exact prefix and context:

| Stage | Actual conditional panel support |
| --- | --- |
| Required deletion | Rank 3/5; selection probability .1306 |
| Required substituted ring | **Absent from all four fixed panels**, each containing 16 candidates |
| Required remodeling | Rank 13/16; selection probability .0215 |

The ring panel cache is not merely a speed cache: only four deterministically seeded panels are possible for that exact state/context. Revisiting it does not explore a new panel. This is a concrete autonomous-support loss despite successful teacher-forced compilation. It is **not** proof that every alternative route to the same molecule is impossible.

### Actual archive/refinement probes

Five historical champions, with their genuine old scores, were injected **only for diagnostics** through the unchanged v0 archive interface. All five passed exact admission. Bounded refinement produced respectively 2, 4, 3, 4 and 2 novel eligible candidates for Full-JAK2, v0-BRAF, v1-5HT1B, Full-PARP1 and Full-FA7. None was docked. This verifies integration and feasible refinement, not improved utility or autonomous entry into those neighborhoods.

I also hash-verified the existing 77/77 complete-region runtime receipts and their zero teacher-action fallback. I did not rerun or modify the compiler. Exhaustive equivalent-endpoint support/probability under v0 is **not established for all 77**. Trace length or failed sampling alone cannot establish its absence.

## 6. Zero yield and v2.1's limitations

V2.1 FA7 logged 120 rejected bootstrap attempts but only **40 distinct attempts, repeated three times**; every one failed QED. Five δ=.6 v0 cells made zero calls. Their logged/unique attempt counts were 87/30, 144/48, 84/28, 89/31 and 72/24, with repeated identical prefixes. Wall limits explain small length differences. These are deterministic retry/support problems, not evidence of chemical impossibility.

V0 JAK2's composition channel made 1,470 attempts: **1,390 ineligible (94.6%)**, 67 eligible, 13 duplicates. Protected execution was enabled. Increasing unconditioned composition volume is unlikely to be efficient.

V2.1 already generated/filter-pooled both channels and used independent RNGs plus a UCB-like allocator. However:

- It removed v0's 25% archive-program-composition dispatch and added current-state editing. Its shallow lane was not the complete historical v0 proposal law.
- Its parents remained score-blind.
- Channel reward used positive gains over the selected parent, not improvement over the incumbent or a quality/diversity niche; worsening outcomes did not subtract the corresponding magnitude.
- Context features were recorded but not used in the channel score.
- It selected eligible candidates in within-channel generation order, not by endpoint utility.

A new 'mixture of experts' slogan would repeat this experiment. The improvement must be in actual support, objective-aligned credit, and parent/refinement allocation.

## 7. Structural proximity is not a dependable utility label

The audit grouped by oracle protocol **and replicate**, deduplicated endpoints within that group using mean observed score, and computed radius-2 Morgan fingerprint distances. Against the post-hoc best endpoint, distance-versus-score-gap Spearman correlations in the five δ=.4 replicate-0 development cells were approximately: BRAF .13, 5HT1B −.09, JAK2 .43, PARP1 .12, FA7 .24.

More concrete checks:

- JAK2: 357 other candidates with the exact same program input and changed-original-atom set as the best endpoint were median 1.4 worse.
- PARP1 seed 2: 77 neighbors at endpoint Tanimoto ≥.8 were median 2.3 worse. All 24 at ≥.9 were at least 1.0 worse.
- δ=.6 5HT1B seed 0: 16 neighbors at ≥.8 were median 3.25 worse.

These are adaptive, post-hoc associations, not calibrated predictive tests. They support using structure as a proposal prior, **not assigning docking labels to unscored neighbors**. Dependency-topology-specific predictive validation remains unperformed; block labels, net WHERE, scaffold and endpoint fingerprints do not fully substitute for it.

There is also substantial score-repeat variation: 512 same-canonical-endpoint/same-protocol/same-replicate groups have median score range .2 and 90th percentile 2.9. Restricting to groups where every score is ≤−8 gives 318 groups, median .1 and 90th percentile .7. The most extreme ranges involve positive docking outliers. This is an observed reproducibility issue, not proof of a particular pipeline cause or of physical binding-energy cliffs. Preserve failures/outliers explicitly and do not reward enormous apparent improvements from pathological parent scores without robust clipping/ranking.

## 8. Best diagnosis, mapped to the requested alternatives

- **A, missing operations:** present in specific v0-to-JAK2 decompositions; the corrected region runtime covers the supplied targets. No universal v0 endpoint-impossibility claim.
- **B, poor joint composition:** strongly supported by the JAK2 94.6% ineligibility funnel and the teacher/program-size mismatch.
- **C, WHERE/parameter support:** directly demonstrated by fixed ring panels excluding the required target and by restricted high-capacity bootstrap.
- **D, missing protection:** not the main v0 problem. Protection already exists and must survive. Most teacher traces need it.
- **E, search allocation:** directly measured score-blind allocation and repeated empty bootstrap. A causal performance benefit from repairing these remains to be tested.
- **Additional issue:** earlier v1/v2.1 changes removed/reweighted successful mechanisms, so their results were not clean tests of strictly additive capability.

My inference is **B+C+E, with specific A gaps**, not 'the compiler is broken' or 'another bigger model is required.'

## 9. The one controller to implement next

**Route-informed, quality/diversity-aware Dynamic program evolution.** One shared implementation and configuration; no protein/seed dispatch table.

1. **Keep the real v0 lane.** Preserve fresh synthesis, constructor mutation, dependency-aware recombination and protected archive-program composition, with independent RNG state. Retain v1's useful current-state edits and generic ring/remodeling capabilities as additional bounded proposal sources, not replacements hidden inside 'shallow'.
2. **Repair effective support first.** Persist bootstrap RNG/cursors across empty rounds. Generate bounded compensating complete programs at tight capacity. Progressively refresh/enumerate binding and parameter panels rather than choosing among four permanently frozen samples. Do not relax final eligibility or primitive legality. Keep proposal and wall budgets explicit.
3. **Distill a small joint move prior, not a whole endpoint lookup.** Fit a source-context-conditioned, smoothed distribution over generic module sequences, region roles and their conditional parameters from successful teacher and Dynamic programs. Train on measured high-utility examples with cell/source balancing; retain non-winning scored examples for contrast. Preserve dependencies and parameter coupling, not independent atom/topology marginals. Runtime contains generic coefficients/statistics and operators, not teacher endpoints, complete winner programs, absolute addresses or target-to-route maps. Use the complete-region representation for training/realization, not as an obligation to invent an entire arbitrary graph from scratch on every turn.
4. **Allocate by actual objective feedback.** Reuse existing niche/score-aware parent allocation, with a declared exploration floor. Retain all observations, but focus repeated mutation/refinement on productive measured endpoints and constructors while keeping diverse worse-score lineages reachable. Maintain distinct statistics for constructor mutation versus actual endpoint edits. Channel credit should use bounded incumbent/niche improvement and novel eligible yield, not unbounded positive gain from an arbitrarily poor parent.
5. **Score completed candidates, then refine.** Filter/deduplicate complete programs before charging. Share the archive across proposal sources. A productive jump earns follow-up refinement; it is not merely logged once and lost among hundreds of uniformly selected parents.

The recovered corpus can support a small within-cell utility ranker, but it is **not a prerequisite** for testing these repairs. If trained, evaluate with source-grouped, lineage-aware splits and time-respecting replay; compare actual top-K docking utility/regret, not component similarity. Keep one same-generator post-hoc ranking control. Never label unscored mutations as bad or good merely because they differ from a winner.

Deprioritize primitive beam search, compiler development, whole-template retrieval, raw structural Recall@K optimization, and another large de-novo patch generator. Keep their artifacts as diagnostics. Do not delete user work or modify frozen experiments.

## 10. Fast falsifiable ladder

These are **proposed criteria for a new revision**, not changes to any existing frozen gate.

### First zero-oracle experiment: actual proposal-access audit

Use the five recovered champion contexts above plus their original roots, with genuine known programs only as probes. Freeze 256 attempted proposals per context and a fixed small set of RNG seeds for uniform versus distilled-prior proposals. Test the actual production WHERE, joint parameter sampler, bindings, STOP/budgets, compiler and filters, not an alternate teacher-only path.

Record analytic support when available, conditional ranks, unique eligible yield, realized complete transformations, and proximity to **historically scored** good alternatives. For JAK2 specifically, verify the required ring parameters are accessible through the runtime sampler rather than only `compile_substituted_ring`.

**Promote:** no deterministic support exclusions for the declared probe family; repeated empty batches explore new candidates; known supplied programs retain exact realization; prior improves generation/ranking of measured productive classes or preserves yield while broadening useful support. Exact teacher identity is not required from every small sample.

**Stop:** a known necessary factor still has zero runtime support, teacher information enters the runtime input, or the new law destroys the proven v0 lane. Do not spend docking calls to diagnose these engineering failures. Failure to sample an exact endpoint in 256 draws alone is not a zero-support proof.

### Historical replay

Split original source groups and all related lineages before fitting a prior/utility model. At held historical decision points, compare rankings only on candidates that actually have scores. Use best/top-K observed utility and regret, valid yield, diversity, and proposal work. No invented score for a counterfactual proposal, no off-policy claim that replay equals a new optimization trajectory.

**Promote:** improvements are visible at small K and do not depend solely on teacher injection or one dominant source. **Stop the learned prior/ranker:** it loses to the matched generic prior on measured utility and diversity. Keep the verified search repairs separate so one failed model does not block testing them.

### First scored experiment: 20 calls, three cells, three ablations

δ=.4 cells: **BRAF-1** (preserve a proven success), **JAK2-1** (hard high-value route), **FA7-0** (bootstrap failure). Three frozen arms, same generic proposal support/eligibility/execution and cheap-work budgets:

- A: repaired-support Dynamic portfolio, score-blind parent allocation, generic prior.
- B: same portfolio and prior, objective/diversity-aware parent/refinement allocation.
- C: B plus the route-distilled joint proposal prior.

At most **3 × 3 × 20 = 180 new charged calls**, with candidate locks before each charged batch. Nine independent trajectories can run concurrently. Historical v0/v1/v2.1/Full-146 curves remain contextual references; A/B/C isolate the new mechanisms. The control is not mislabeled as byte-identical historical v0 because the support repairs are explicit.

Inspect at calls 5, 10 and 20; do not kill based on call 1. Do not backfill poor candidates or silently reset optimizer seeds. Report absent yield and failed calls, not only successful scores.

**Extend to 50 calls** if the measured-feedback arm improves a hard cell by at least .5 over A without a >.5 BRAF regression, or has comparable utility with clearly better feasible yield. If C adds no value over B, drop the learned prior from the next scored revision rather than protecting it for architectural reasons. If both B/C are ≥1.0 worse on BRAF and JAK2 at 20 calls and yield remains poor, stop this revision. These are development screens, not statistically conclusive superiority tests.

### Five-cell qualification and full benchmark

After 50-call evidence, include 5HT1B-0 and PARP1-0; use two optimizer seeds and matched curves at 20/50/100/200 before considering 1,000-call runs. Check that the strong 5HT1B/BRAF behavior is retained and the JAK2/PARP1 gaps shrink. Repeatability of apparent wins matters given the historical score variation; any confirmation calls are predeclared and charged, never selected by best-of-retries.

A plausible progression toward IVG would be repeatable qualification trajectories approaching the observed reference regime (roughly JAK2 ≤−11.1 and PARP1 ≤−13.0, while BRAF stays near −11 and 5HT1B near or beyond −13.3), plus recovered FA7 yield. These are **aspirational go/no-go targets**, not predictions, not achieved results, and not substitutes for the IVG comparison.

If support is repaired but bounded objective-driven search still cannot enter productive JAK2/PARP1 neighborhoods, abandon this particular proposal prior. That would localize failure to high-value basin entry; do not respond with a wider beam, a relaxed gate, or a new compiler.

Only then freeze one controller for full δ=.4 and δ=.6 evaluation under the actual benchmark protocol and charged-call budget. Training on all T4 winners is trained-on-T4 evaluation, not held-source generalization. These extensively inspected development cells cannot later be called untouched.

## Checks, limits and handoff

- Ran the retrospective audit, known-answer runtime probes, all-77 route comparison, and search/data audit locally.
- Verified all material input hashes, distinct receipt identities, call-index availability, five diagnostic archive admissions, and the 77 existing runtime realization receipts. No new compiler gate was run.
- Ruff check/format passed on audit scripts. No repository-wide suite was run: no runtime implementation or scientific-launch milestone is being declared complete.
- New descriptors use local Python 3.12.9 / RDKit 2026.03.6 / NumPy 2.5.3 on macOS arm64; historical docking came from the frozen historical environments. Prefix-eligibility recomputation is explicitly local, not a replacement of pinned benchmark receipts.
- Missing terminal artifacts, unrecovered call order, unresolved equivalent-program support and untested causal improvements are preserved as limitations.
- This audit and report are local, uncommitted artifacts. Existing user-owned changes remain untouched. The next safe action is the bounded support/search implementation above, not a full docking launch.
