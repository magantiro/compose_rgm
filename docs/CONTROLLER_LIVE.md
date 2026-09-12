# COMPOSE controller: current decision record

Updated 2026-09-12. Active goal: reproducible improvement over the actual T4 and
PMO champions, then verified matched external comparisons. Supporting chemistry,
known-winner reconstruction and mathematics are not benchmark success.

Current active run: none. The option-controller V1 reusable core is implemented
offline: compositional WHAT features, a policy-identified distributional
best-improvement value, a conservative learned proposal with explicit base-law
probability, persistent option-boundary SMC with `B/q` correction, and locked
joint-posterior batch acquisition. WHERE remains the qualified external region
controller and HOW remains the existing exact option continuation kernel. No
reference model, executor, molecular support, `Q(M)`, docking label or Modal job
changed in this milestone.

Latest discriminating result: the only eligible stored balanced-reference PMO
continuations contain zero positive cells among 288 identified horizon-threshold
targets (600 total cells; 48% identified after right censoring; 50 boundary rows,
four source groups, eight terminal lineages). Fitting would therefore create a
trivial all-negative head, not a validated future-value model. The gate abstains
and prescribes the smallest missing evidence: policy-congruent option
continuations containing both improving and non-improving outcomes. Audit:
`diagnostics/option_controller_v1/report.json`. New oracle calls: zero.

Latest PMO mechanism: witnessed three-option recoveries exist (51 among 903
temporary-loss branches), but their future-return lift is directional rather than
root-bootstrap robust, so no learned Doob twist is justified. A signed endpoint-
delta ranker over complete executable plans does pass the earlier 58-call temporal
holdout: RMSE 0.119 versus 0.252 for parent retention, score Spearman 0.722, gain
Spearman 0.402, and mean selected score 0.639 versus 0.548 uniform across nine
parent choice sets. Predictions are frozen for all 95 unscored locked plan-pool
candidates. Scoring that full assay requires explicit authorization for exactly
95 new PMO oracle calls; none has been spent on it.

Latest comparison completed and rejected: the updating complete-plan proposal
policy tied its frozen-policy control at best score 0.6835298931, final top-ten
mean 0.6787222097, and mean round-end top-ten mean 0.6781711014. The actor changed
after every round and later plan-pool total variation was about 0.13; final
archives still differed by 12 molecules per arm. The null is therefore not a
failure to execute the intervention. It is evidence that sparse endpoint-gain
updates over sampled complete donor plans do not solve the missing proposal and
delayed-credit problem. Stop this unchanged recipe. Cost: 58 new calls, 276.953
seconds wall time, and 243.536 summed proposal seconds. Current champion remains
0.6835298931. Protocol and audit: `docs/PMO_PLAN_POLICY.md` and
`diagnostics/pmo_plan_policy/`. No job is active.

Current hypothesis: competitive improvement requires coupling multi-step proposal
reachability to delayed value, while retaining broad local-to-global reference
traffic. The next smallest experiment must distinguish whether a rollout-derived
continuation value can enrich completed improving plans before paying for a broad
run. Do not call an endpoint predictor future value, and do not scale the rejected
complete-plan actor.

Zero-new-call failure separation completed. Across 46 recorded plan pools, only
76/890 distinct products (8.5%) had pre-run labels. Four pool instances, covering
three distinct parent structures, contained a known improvement; behavior assigned
0.89% mean mass to known improvers. This is evidence of ranking headroom but not a
valid estimate of total proposal recall because 91.5% of distinct products are
counterfactually unlabeled. The frozen and learning arms continued 6 and 8 scored
temporary-loss states and recovered 0 above the respective pre-loss score.
Therefore endpoint-only continuation is inadequate, while rollout-value headroom
remains unmeasured. Audit: `diagnostics/pmo_plan_policy/failure_separation.json`.
Next safe action is a locked, bounded pool-prevalence assay on strong parents;
that requires a new paid-label authorization and must be specified before calls.

Operational status: offline complete-edit context learner implemented and tested;
**rejected**, no new oracle calls or live job. Preparation plus fit/evaluation took
4.6562 seconds. Training retained damaging edits and separately labeled failures.
The cut-context model missed the only calibration-pool improvement; the same-data
endpoint-only model found it. In the later audit, only 1/12 eligible logged pools
contained any improvement even with perfect ranking. Next: use complete planned
molecular endpoints to improve proposal construction before compilation, not
another unchanged cut-kernel or long mixture run. This next change is untested.
Artifact: `diagnostics/pmo_edit_chooser/README.md` and `report.json`.

Previous fresh-seed replication completed and audited on 2026-09-11.
Both arms remained at 0.6747477698. Stop this unchanged mixture: the first best-score
advantage did not repeat. Cost: 76 new calls, 403.8898 seconds, no live job.
Same original starts/model/mixture, with paid-cache and exclusion roles kept
separate. The first run and its best molecule are preserved. The full
competitive-controller objective remains unmet; no broader run is authorized.

Completed run `83d84e96ce3d961d09816b3a7829103a1b608380147416c6b0996e9643806b77`,
call `fc-01M29BN6BVQQF5PDR2A9PNKKP7`, source `c7818ce74c2d`.
Deployment `compose-pmo-local-guidance-replication` completed in 93.321 seconds;
strict preflight passed with a clean tree. Durable local receipt is under
`/private/tmp/compose-pmo-local-guidance-replication-runs/<run_id>/spawn.json`.
Audit: `diagnostics/pmo_local_guidance_replication/report.json` and README.
Do not launch it again or start another task without a bounded new recipe.

## Champion and exact regime

- Current observed best: **0.6835298930947339**, from the completed four-round
  local-guided broad controller, versus **0.6803013430498075** for its matched
  fixed-memory baseline. Both started at 0.6747477698. Eighty-five new calls,
  497.011 seconds; positive on best score but lower guided top-ten mean and
  diversity. Its advantage did not replicate. Exact four-edit winning suffix: local ring closure,
  then three generic atom restatements. `diagnostics/pmo_local_guidance/README.md`.
  Fresh-seed replication: both 0.6747477698; top-ten means 0.6724616451 baseline
  and 0.6728082152 guided. No promotion of this unchanged local-guided mixture.
- Earlier fresh-seed memory comparison: both fixed and evolving memory
  reached **0.672021505032247**, with identical top-ten mean 0.6532973102.
  The first evolving-memory lead below did not replicate. Stop the unchanged
  memory comparison and retain simpler fixed memory. Replication cost: 81 new
  calls, 398.839 seconds, 329.686 proposal seconds. Audit:
  `diagnostics/pmo_evolving_memory/replication_report.json`.
- Previous observed endpoint **0.6747477697966318** is from a separate cached
  primitive-neighborhood diagnostic, not a full-controller comparison. One atom
  deletion improves the previous champion by 0.0027263, at 1024 new queries.
  Of 1580 supported neighbors, 1044 were scored (20 cached), 30 improve the
  originating 0.649519 parent, and one exceeds 0.672022. Local pass: 5.273 seconds,
  reusing 16.936 seconds of generation. Poor oracle efficiency; do not promote
  exhaustive local scoring. `diagnostics/pmo_cached_neighborhood/report.json`.
- PMO perindopril, exposed prescreened development: new best **0.672021505032247**,
  versus **0.6580812002646271** in the latest fixed-memory control, starting
  identically from 0.649519052838329. Six rounds, 109 new calls, 619.577 seconds.
  Top-ten means 0.6502614449 versus 0.6482515168. First-run criterion passed;
  its lead did not replicate, as recorded above. The best sequence was hydroxyl addition then generic
  ring opening, not a newly selected donor. All six completed new-donor uses
  were worse than their parents. Do not claim a demonstrated transfer mechanism.
  Audit: `diagnostics/pmo_evolving_memory/report.json` and README.
- Previous component batch best **0.649519052838329**,
  previously 0.6030226891555273. Uniform donor proposals found it in the latest
  locked batch; contextual replay reached 0.6365315497232666. This is a warm
  proposal-component result, not replicated full-controller improvement or
  official PMO top-ten AUC. The new molecule uses a 39-primitive scaffold
  replacement, +8 heavy atoms and +1 cycle rank. Exact graph, SMILES, operands,
  costs and receipt audit: `diagnostics/pmo_edit_replay/report.json`.
- Strongest matched full-controller PMO evidence: broad-reference/donor mixture
  reached 0.6030226892 versus 0.5595028849 for the unchanged reference baseline.
  A fresh seed reached 0.5869734365 versus the same baseline. Identical original
  starts and donors, six option rounds, 156 and 149 new calls. First gain:
  four-edit side-chain replacement; replication: 47-edit structural replacement.
  `diagnostics/pmo_donor_comparison/README.md` and its two reports.
- Historical accounting: 249455 prescreen evaluations plus **2275 development
  physical calls** through the plan-policy comparison. The prescreen job retained
  only each task's top 100 and discarded 249355 per-molecule labels; it is not a
  249455-row training bank. The retained bank is our historical asset, not
  external labels, and its legacy provenance remains incomplete. Do not call this
  no-prescreen. The separate no-new-prescreen controller remained at
  0.5222329679 (`diagnostics/pmo_learned_proposal/report.json`).
- T4: preserve both historical controller lineages. The workshop table selects
  compact B in 11 cells and an older 200-call controller in 15, with four failures.
  It is not one frozen 500-call controller or matched mean-of-three result.
  `diagnostics/controller_baselines/t4_lineage.json` and
  `diagnostics/t4_combined_table.json` document this. PARP1 s0 d=0.4 B scores
  -10.9/-10.6/-10.7 at 100 calls each. The broader warm run reached -11.0 at
  40 new plus 94 historical attempts, without replicated superiority. Incumbent
  redocks: -9.8 in one diagnostic, -11.8/-12.3 in the latest. Docking noise and
  the answer-known -13.6 reconstruction remain separate evidence.

## Previous experiments and decisions

`docs/PMO_CONTEXTUAL_EDIT_REPLAY.md`: nonparametric complete-edit proposal from
33 positive historical donor edges spanning 29 parents. Same 16 exact current
parents, original 100 donors and 64 attempts per arm. Replay retains 20% uniform
exploration. No endpoint predictor, public winner, or reference retraining.

| Metric | Uniform | Replay |
|---|---:|---:|
| Best | 0.649519 | 0.636532 |
| Archive top-ten mean | 0.597691 | 0.605168 |
| Unique improving offspring | 3 | 9 |
| Completed / distinct molecules | 40 / 40 | 46 / 29 |
| Proposal seconds | 20.358 | 26.104 |

63 new calls, 50.413 seconds total, 0.024 actual oracle seconds. Local CPU,
no Modal worker or GPU. All unique compiled endpoints scored after locking.
Audit verifies sealed receipts, locks, exact graph identities, arm membership,
charged query union and reproduced best/top-ten values. Three focused tests
passed in 2.71 seconds; touched-code lint passed. No full suite or milestone
completion claimed. Executed local source is hash-bound; implementation and
reports were subsequently preserved in commit f47e852.

**Primary replay-vs-control criterion failed.** The secondary top-ten gain does
not authorize promoting or replicating this unchanged recipe. Retain the useful
new states. Raw result: `/private/tmp/compose-pmo-edit-replay-20260911a/result.json`.
Source and receipts: content-addressed volume prefix in
`diagnostics/pmo_edit_replay/README.md`.

Stopped earlier comparisons: archive parent selection retained quality but no
arm exceeded 0.603023 (89 calls, 349.619 seconds). Refit endpoint preferences
avoided damage but selected no improving offspring (30 calls, 46.672 seconds).
A separate 51-query audit scored that entire frozen pool: best 0.593296, so no
selector could recover a new champion from it. This is a pool-specific result,
not a global ceiling on uniform proposals. See `diagnostics/pmo_archive_branching`
and `diagnostics/pmo_program_choice`.

T4 donor transfer stopped: 31/32 programs completed; 22 endpoints failed benchmark
gates and seven were known. Two novel candidates docked -7.5/-7.6. Four repeat
controls bring cost to six calls, 114.772 seconds. Useful feasible coverage failed,
not primarily compilation. `diagnostics/t4_donor_probe/report.json`.

## Next decision and constraints

Completed PMO call: `fc-01M291A3D6D26JKTAH17S4CCPB`, run
`e111b2aac585acaf2c5c8d6338599f6a53856db9e3c5a7132b9cd77e53292189`.
Deployed clean source `eda6e8963d7f`, strict preflight passed. Deployment 89.413s.
Durable prefix `compose-v4-artifacts/pmo_evolving_memory/<run>`; local spawn
`/private/tmp/compose-pmo-evolving-memory-runs/<run>/spawn.json`.
Goal active and unmet. No job currently running. Replication completed:
`fc-01M292KFN8Y77CD4DYA92F74MV`, run
`f6f3264e4271ba044265c4b782c7d4aea154ad12ad027d6627efb44e962651e9`,
clean source `cc15546c6fac`, deployment 87.222 seconds. Durable volume prefix as
above; local `/private/tmp/compose-pmo-evolving-memory-replication-runs/<run>`.
Latest local diagnostic: `/private/tmp/compose-pmo-cached-neighborhood-20260911b`.
No concurrent T4 job.

Completed hypothesis: fixed donor memory cannot directly reuse newly evolved
components across branches. Compare evolving proposal memory to frozen memory,
with identical starts, actual-score parent allocation, and broad reference
editing in both. Only an arm's own requested discoveries update its memory,
at round boundaries. Implemented and launched under
`docs/PMO_EVOLVING_MEMORY.md` and `configs/pmo_evolving_memory.json`.
Same top 16 exact parents, 116 common initial donors, six rounds, at most 192 new
calls. Both use 50% broad reference proposals and 50% rank/uniform donor programs;
only donor-memory updates differ. At most 29 workers plus one driver, $10 cap.
Ten focused memory/worker/driver tests passed in 4.10 seconds; lint passed.
The first run exceeded initial 0.649519 and concurrent fixed-memory best.
Replication contract 4af1aa1662cd04165e5463ada7eddc017345f53e3d456cf43e5a579b7b0ec3a9
uses seed 20261003. Exact initial parent/donor equality and the 109 cache-only
labels were verified; three memory/worker/round-lock tests passed in 2.37 seconds,
with touched-code formatting and lint passing. A repeated lead supports the
comparison; null or reversal leaves the first gain seed-dependent.

Current hypothesis: paid local labels can improve query selection. Run the
fixed-recipe endpoint ranking check and, only if informative, a 32-call maximum
locked comparison on previously unscored neighbors. Preserve broad options;
this tests local interpolation, not future value or cross-parent generalization.
`docs/PMO_LOCAL_QUERY_SELECTION.md` governs the next bounded component test.

Completed: local query selection passed its component criterion. Sixteen predicted
choices averaged 0.646121 versus 0.552757 for sixteen uniform choices, with two
versus one improvements over the originating parent. No new champion. Thirty-two
new queries, 7.008 seconds preparation plus 2.402 seconds replay/scoring.
`diagnostics/pmo_local_query_selection/report.json`. Next: test the unchanged
selector on new parent neighborhoods before integrating it into the broad loop.
The exact kernel repair is 30.56x faster on a recorded 256-molecule fixture,
3.575s to 0.117s, with zero numerical error; the 1580-state kernel took 5.586s.

Parallel component completed: persistent two-option donor planning versus
immediate continuation. Actual witnessed paths were retained. Both bests remain
0.674748; no temporary-loss branch recovered above its root. Fourteen new calls,
15.501s total, 11.903s proposals. Of 32 unique attempted tasks, 17 compiled,
12 failed size bounds, two were unresolved and one was a self-proposal.
Stop this unchanged schedule/proposal recipe. The 574 replayed primitives are
supporting work, not benchmark improvement. `diagnostics/pmo_persistent_lookahead`.
No experiment remains active. Goal is unmet.

Cross-parent selection completed: the same fitted endpoint model found 4/16 and
2/16 parent-improving products, against 0/16 in both uniform controls. Mean scores
were 0.667499/0.611814 and 0.659941/0.596829. Neither new parent was a training
endpoint, but this remains related exposed chemistry. Best stays 0.674748.
Sixty-four calls, 52.328 seconds preparation plus 3.549 scoring/replay. Prediction
replay error is zero; all selected primitive endpoints and paid receipts audited.
`diagnostics/pmo_cross_parent_selection/report.json`.

Next bounded run is prepared under `docs/PMO_LOCAL_GUIDANCE.md`: compare the
retained fixed-memory broad controller to its mixture with the tested local
selector. Same sixteen exact initial states, 116 donors, frozen model, four
rounds and at most 128 new queries. Baseline 50% donor/50% reference; intervention
50% donor/25% reference/25% local selector. A new champion above the simultaneous
baseline earns replication; null/reversal stops the unchanged mixture. This is
not future-value or exact Doob control. No scientific job launched yet.

Launch status: clean source `a215fe7992a5` passed strict preflight. The focused
integration/cache/legacy-driver checks passed (13 nodes initially; the final
driver fixture passed after correcting its two-slot/shared-channel setup).
Deployment was rejected by the approval reviewer because the recorded repository
milestone remains the older scoped T4 authorization. No Modal job started and no
new experiment cost was incurred. Resolve the milestone explicitly before retrying
the same four-round PMO deployment (128-call, 30-container, $10 maximum).

Authorization resolved on 2026-09-11: the user explicitly said "ok well lets just
go ahead" and "what are you waiting for" after that bounded launch request.
The scoped amendment is recorded in both the primary and execution-worktree
AGENTS.md. Proceed with this single prepared comparison; no additional second-task
experiment or extra label budget is bundled into the approval. No scientific
recipe, model coefficient, executor rule or input artifact changed.

Completed run: `pmo_local_guidance/2f58bc8ace9d1591520d0c2a670a0b04cb1bd1cad4879f1ca4420361c8570e72`,
durable call `fc-01M297KAGZKERT3EE8P8SJW9MM`, exact deployed revision
`797d7085e737`. Both arms started at 0.6747477698. Guided finished at
0.6835298931 versus baseline 0.6803013430, passing the predeclared best-score
criterion. Guided top-ten mean was lower (0.6743329457 versus 0.6747775400),
as was archive diversity. Eighty-five new calls, 497.011 seconds total and
470.540 seconds proposals. Development physical-call total is now 2141, plus
249455 historical prescreen calls. No scientific job remains active.
Local source snapshot and spawn receipt are under
`/private/tmp/compose-pmo-local-guidance-runs/2f58bc8ace9d1591520d0c2a670a0b04cb1bd1cad4879f1ca4420361c8570e72/`.
The source snapshot is also preserved on the same remote run namespace.

The new winner comes from a local ring closure followed by three generic atom
restatements, all four exact primitives independently replayed. The local model
still misranked an immediately better move at 110/1618 in the matched first-round
neighborhood. This is a positive mixture result, not learned future-value
validation or external benchmark success. Audit and decision:
`diagnostics/pmo_local_guidance/{report,evidence}.json` and README.md.
Next: unchanged fresh-seed replication from the same original starts and model,
not the new winner. This has now been authorized; no second task is running.
Do not tune this mixture on the new scores. Overall goal remains unmet.

Replication preparation finding (read-only inspection, no new model/oracle
calls): `pmo_option_particles.py` currently uses `data["observed"]` both as the
paid-score lookup and as the initial local exclusion set. Copying the older
memory-replication pattern, which merges previous-run labels into `observed`,
would therefore change this local-guidance proposal law. The saved round-1
selection and oracle ledger confirm that its winning precursor would be removed.
Repair implemented: original prepared inputs and proposal exclusions remain
fixed; 85 additional paid labels occupy a separate requested-only score cache.
The completed first run is unaffected. Do not modify its model or use its new
winner as an initial state. The loader also pins the first run's seed, so an
explicit replication contract is required rather than editing an old run in place.

Replication contract `configs/pmo_local_guidance_replication.json`:
`8c5554584534f3d535edd870c5fa7ce489625ae848b586ed706151b8cfede439`.
Fourteen focused driver, selector, report and cache/exclusion checks passed
(3.53 seconds for thirteen, 1.43 seconds for the replication input check).
Touched-code formatting/lint and diff checks passed. Original prepared inputs,
model and contract are unmodified. Qualified prior law-cache metadata is uploaded.
Launch completed from clean commit c7818ce after strict preflight and deployment.
No overall milestone/full-suite completion is claimed.

The three completed local batches (cached neighborhood, local query selection,
persistent lookahead), including executed source snapshots and all raw receipts,
are preserved on volume `compose-v4-artifacts` at
`controller_local/9bcf2a80cc1496a1433c53a9790b747922d8aee2211de730ceb99e561e0a5b8b/artifacts.tar.gz`.
The archive SHA-256 is the directory name. Local copy:
`/private/tmp/compose-controller-local-20260911.tar.gz`. Upload completed;
individual audited summaries remain in their diagnostic directories.

Preserve generic/reference chemistry and local/global editing. Donor programs
use the unchanged valid-state executor but need not have positive probability
in the frozen neural tables: no original-R_theta Doob, kappa=1 or canonical-law
invariance claim applies to that channel. No endpoint gates on internal states.

The earlier future-value head used witnessed-path maxima and mismatched
primitive/option horizons and remains rejected
(`diagnostics/pmo_option_particles/GUIDE_AUDIT.md`). A replacement must estimate
actual continuation value under its stated policy/horizon. An endpoint predictor
or compiled program is not future value.

Match task, oracle, initialization, all history/training labels, budgets and
replicates for external comparisons. Known-winner development is not blind
discovery. At most 30 Modal containers within declared caps; scientific launches
use clean committed source, deploy and durable spawn. Reuse paid trajectories;
focused checks, no repeated full suites, unrelated cleanup or excessive commits.
