# Fragment and QED ablation report, 2026-09-25

This report answers the later fragment/QED instructions. It binds the manuscript fragment rows to named runs, distinguishes finite-panel program scoring from primitive-reference sampling, specifies the primary nonlearned intervention before interpreting its outcome, and records the actual 800-source QED sampler and fixed-size status. It is not the earlier six-part inspection of all COMPOSE macro families.

## 1. Manuscript row lineage and current completed runs

The current local manuscript is `COMPOSE_ICLR_2027_Overleaf/main.tex`, around lines 525-608. Its COMPOSE fragment cells do not all come from the same 3-seed campaign.

| Task | Values currently in manuscript, quality / uniqueness / diversity / validity | Exact row source | Current named run status |
| --- | --- | --- | --- |
| Motif extension | 43.0 / 97.5 / 0.669 / 100.0 | `diagnostics/fragment_motif_focused_dev_v1/summary.json`, SHA-256 `71c962306249da7c5a040aba1b505086eb1c7bbe5732645643bf843e5747a88a`; 200 attempts, one development seed | `configs/fragment_motif_official_v1.json` has now completed 3,000 attempts, seeds 2/3/4. Its separate `diagnostics/fragment_motif_official_v1/summary.json`, SHA-256 `fc6658b10f83c44aa2fb6aa447492e7ce0852654899507bef3c887576ce5d33d`, records 2,999 outputs and 2,999 valid, prompt-faithful outputs. Official means are 42.6333 / 93.6316 / 0.664021 / 99.9667. These are not the numbers currently printed in the manuscript. |
| Superstructure generation | 39.03 / 97.33 / 0.725 / 100.0 | Completed `configs/fragment_superstructure_official_v2.json` and `diagnostics/fragment_superstructure_official_v2/result.json`, SHA-256 `8406f57dc5ff6736110593d309f207d7a5d0e878eeb1b7a2f2c423fe2d9b3652`; 3,000 attempts | The manuscript row matches this completed 10-prompt, 3-seed run. All 3,000 attempts committed a valid, prompt-compliant output. |
| Linker design | 35.0 / 91.5 / 0.548 / 100.0 | `diagnostics/fragment_training_linker_metric_pilot_v2/summary.json`, SHA-256 `2eaa9c1782b2f520df0af4d004e238f36c407f517579aa83656aaeced9352ffa`; 200 attempts, one development seed | Completed `configs/fragment_linker_official_v1.json` instead records 28.6 / 72.4 / 0.562149 / 100.0 over 3,000 outputs in `diagnostics/fragment_linker_official_v1/summary.json`, SHA-256 `ff6f0fad7f1338540751b1a353c71fce5984108c63225ad80dea542563e87595`. |
| Scaffold morphing | Same as linker | The evaluated morphing block reuses linker outputs | It is not an independent sampling experiment. The current manuscript numbers inherit the earlier linker development row. |
| Scaffold decoration | 22.5 / 99.0 / 0.628 / 100.0 | `diagnostics/fragment_pendant_decoration_dev_v1/summary.json`, SHA-256 `e4d064a9786854ae9e5187654d3be7c1c65efd139aea3bacc458825ab897cf88`; 200 attempts, one development seed | No later joint-mass or breadth pilot has been substituted for this row. A matched, completed 3-seed official decoration row was not established in this inspection. |

The motif contract's serialized `prepared_pending_payload_specific_scored_evaluation_authorization` status was a stale preparation flag, not proof that its run was absent. The completed summary now supplies the run-level evidence: 3,000 attempt slots, 24,000 offered program draws, 2,999 outputs, zero task-oracle calls, contract hash `c22e64562dacb57bf6b497a408ffe7030df50c83d5c2fb47787bdeb88670aabe`, and source revision `ac36485bcfeebe08074f8f8035409020768a1e26`. One failed output slot remains in the denominator. Contract status and measured completion must be checked separately.

## 2. What the fragment samplers actually do

**Motif and linker:** `sample_motif_panel` in `src/compose_v4/benchmark/fragment_motif_focused_programs.py:30` and `sample_linker_panel` in `src/compose_v4/benchmark/fragment_linker_sampler.py:251` each construct eight task-specific complete-program offers. Motif's training-derived attachment catalog and joint final-size/ring allocation generate complete regions. Linker uses a training-derived two-boundary connector and compiles the connector plus second core into an ordered two-block program. The exact executor, structural locks, endpoint checks, canonical endpoint deduplication, and native model-fiber membership precede selection. `learned_program_scores` in `fragment_program_adapter.py:433` calculates

`s(pi) = (1/L) sum_{ell=1}^L log p_theta^native(a_ell | x_{ell-1}, t=0)`.

`select_learned_program` at line 496 samples one distinct supported endpoint with probabilities proportional to `exp(s(pi))`. This is a length-mean **native-mark** score followed by a finite-panel softmax. It is not a product path probability under canonical `R_theta`, and the reference does not construct the motif or linker catalog. No upstream reference-ranked shortlist was found in these named paths.

**Superstructure:** `sample_completion` in `fragment_conditioned_sampler.py:779` samples primitive marks during the trajectory under retained-structure and attachment controls. It does not use the eight-offer motif/linker panel. The learned mark law therefore changes the transition-generating distribution itself. This is the appropriate primary task for a learned-transition-reference ablation.

**Decoration:** the manuscript row comes from `sample_pendant_panel` in `fragment_pendant_programs.py:83`, which builds eight complete pendant-decoration offers by filling declared scaffold interfaces with training-observed pendant content, exact-compiles them, then applies the same mean native-mark score and finite-panel selection. Later joint-mass pilots change construction and are distinct experiments, not the source of the printed 22.5 row.

For motif/linker, structural compilation, endpoint deduplication, and native action masks depend on state and frozen operative flags, not checkpoint tensor values. The native scorer still rejects a nonfinite score, which is recorded as an abstention. A learned-versus-uniform choice on the **same recorded model-supported panel** is therefore a valid conditional panel-selection comparison; it does not remove the training catalog or test the primitive transition law. It does not require the unrelated Editing-V2 empirical-family artifact.

The completed optional linker replay is `diagnostics/fragment_common_panel_linker_v1/result.json`. On the same 3,000 saved panels, the learned selector exactly reproduced the official linker summary. Learned versus uniform selection yielded quality 28.6% versus 23.3333%, uniqueness 72.4% versus 73.3667%, and diversity 0.562149 versus 0.573898. This is a paired saved-panel analysis, not a second generation campaign or a causal test of the complete reference model. No optional motif panel replay was launched.

## 3. Checkpoint and operative-configuration comparison

All named motif, linker, decoration, and superstructure fragment contracts pin RingCore checkpoint SHA-256 `24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`. The loaded model has 118 parameter tensors, width 256, six message-passing steps, hierarchical rate factorization, enabled primitive cycle operations, and disabled ring-grow macro. The constructors and reference-use stages above still differ by task. The file's `best.pt` name alone is not proof of a selected production checkpoint; this is the 16,000-step RingCore diagnostic lineage.

The QED H40 sampler uses Editing-V2 checkpoint SHA-256 `c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8`. Inspection found 120 selected parameter tensors. Of 117 same-named, shape-compatible tensors compared with RingCore, none was numerically equal. Sorted tensor-byte SHA-256 digests are `c5923a3948199cbd9666004a34438e4161eb4517cb0141971e21098fb88ca61e` for RingCore and `078e6a5678ca0a3ea09fc0dac1fc3447bf2937dfc0a922bdeaa9e3639d9e0eb1` for Editing-V2. Shared fragment checkpoint identity is established; a single numerical `R_theta` across fragment and QED results is not. This comparison used loaded parameters and operative flags, not filenames or file hashes alone.

## 4. Primary superstructure transition-reference ablation

The exact law was frozen before the comparison in `docs/FRAGMENT_SUPERSTRUCTURE_REFERENCE_ABLATION_LAW_2026-09-25.md` and self-hashed `configs/fragment_superstructure_reference_ablation_v1.json`, payload SHA-256 `e5ca9f7e95974d87ca019e385619397aede74e2d36c429bc111fe4507db08df7`. At each state, the nonlearned arm uses the **same native action masks and available families** as the RingCore sampler. It chooses uniformly among available families and uniformly among masked native action coordinates within the chosen family. It replaces the learned family and operand/payload probabilities, including the initial atom-insertion-conditioned draw. The frozen learned total hazard remains in both arms to retain the operational-time stopping rule. This is uniform-family, uniform-native-mark sampling, **not** uniform over distinct canonical successors; mark multiplicity is retained in both arms.

The executor, retained-region and effective-chemistry locks, attachment redirection, 24 mark attempts per event, 32 committed-event cap, operational horizon 16, ten prompts, 100 attempts per prompt, seeds 0/1/2, no-replacement failure accounting, and official evaluator were held fixed. The runner verified the learned result SHA, its contract, all 14 material hashes, and every one of the 30 new shards before reducing them. The new result is `diagnostics/fragment_superstructure_reference_ablation_v1/result.json`, SHA-256 `007f59645ffda58f1722eb846444d4f604307468016d4f813740b3ab9c9b0b5d`; it records 3,000/3,000 committed, fragment-preserving and prompt-compliant outputs. The learned arm also has 3,000/3,000 on each count.

| Arm | Quality % | Uniqueness % | Diversity | Validity % | Prompt-compliant / attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned RingCore mark law | 39.0333 | 97.3333 | 0.725074 | 100.0 | 3,000 / 3,000 |
| Uniform native-mark law | 25.4000 | 90.1667 | 0.708503 | 100.0 | 3,000 / 3,000 |
| Learned minus uniform | +13.6333 points | +7.1667 points | +0.016571 | 0 | 0 |

The quality comparison by prompt, averaging the three seeds within each prompt, is:

| Prompt | Learned % | Uniform % | Learned minus uniform, points |
| --- | ---: | ---: | ---: |
| BARICITINIB | 46.67 | 25.67 | +21.00 |
| CYCLOTHIAZIDE | 0.00 | 0.00 | 0.00 |
| ELIGLUSTAT | 66.67 | 41.67 | +25.00 |
| ERLOTINIB | 37.00 | 43.67 | -6.67 |
| FUTIBATINIB | 43.00 | 10.67 | +32.33 |
| LESINURAD | 49.33 | 36.00 | +13.33 |
| LIOTHYRONINE | 44.00 | 50.33 | -6.33 |
| LOVASTATIN | 0.00 | 0.00 | 0.00 |
| MARIBAVIR | 73.33 | 36.33 | +37.00 |
| SPIRAPRIL | 30.33 | 9.67 | +20.67 |

The learned arm improved quality on six prompts, trailed on two, and tied at zero on two. The result supports a learned-mark-probability contribution for this specific constrained primitive sampler and benchmark measure. It does not isolate each individual probability head, establish a result for motif/linker panel scoring, or show that every source benefits. Seeds match but the two samplers consume random draws differently, so identical seed labels do not imply paired molecular trajectories. Per-prompt/seed results and all 30 shard records remain in the result directory.

## 5. Actual 800-source QED result and fixed-size status

The paper-era QED result is from `scripts/hphi_official800_launch.py`, which launches `run_source` in `modal_apps/hphi_h40head_ab_app.py:180` with the restart arm. It uses the Editing-V2 reference above, the `hphi_v2` future-value head with budget input capped at 24, horizon 40, 32 SMC particles, and eight independently seeded terminal returns per source (`seed_for("restart", source, k)`). The generic option/macro implementation is not this run's sampler.

The local rederivation `diagnostics/qed_griddd_arm_a_rederivation_v1.json` contains 798 scored sources and 446 solved sources. Source indices 135 and 408 lack scored records. Thus 446/800 is 55.75%, rounded to the manuscript's 55.8%, but the available records alone do not constitute a completely scored 800-source paired baseline. The alternative 446/798 rate is 55.89%; the denominator must be stated. These two records must be recovered or accounted for under the declared failure rule before a full paired fixed-size estimate is reported.

The fixed-size intervention has been implemented in the same H40 `run_source` path, not in the generic option/macro controller. It removes every size-changing primitive family before each family draw, renormalizes over remaining families, asserts no heavy-atom-count change after execution, and treats an empty restricted successor fiber as a stopped particle without replacing an output slot. The frozen head, reference checkpoint, horizon, particle count, resampling, source panel, and eight-terminal-return rule are unchanged. Focused fixed-size filter tests passed (4 tests). **No 800-source fixed-size result exists yet.** The exact H40 head files (`hphi_v2/head.pt`, `norm.json`), `RUN_PATHS.json`, materialized scorer, full-arm source shards, and missing two source records must be staged with their exact identities in an accessible workspace before launch. Prior read-only access to the Rahul volume was denied by the sandbox, so that path was not retried. The existing endpoint archives also lack enough atom-count ancestry to report successful paths that changed size and later returned to the initial size; authentic trajectory logging is needed on a parity-checked full-arm execution.

## 6. Verification and remaining manuscript corrections

Focused fragment tests passed (8 tests), QED fixed-size mask tests passed (4 tests), touched-file Ruff passed, and `git diff --check` passed before the new result was added. The primary superstructure runner exited successfully after verifying all 30 shards and publishing the reduced result. Repository-wide verification is not green: the pinned chemistry environment lacked Modal and stopped at 61 collection errors, the development environment had five collection errors including missing pandas, and repository-wide Ruff reported pre-existing errors outside these edits. These checks must not be described as passing.

The current manuscript still prints the older 200-attempt motif and linker numbers and the 200-attempt decoration result next to the completed 3,000-attempt superstructure result. Its statement that only the structural specification changes across the five fragment tasks conflicts with their distinct active constructors. Its broad language about the same frozen numerical `R_theta` across fragment and QED settings conflicts with the parameter comparison above. No manuscript cell or prose was changed as part of this report. The next safe paper edit is to choose the completed run-level rows explicitly, keep morphing as a linker-output alias, and qualify the shared-process claim by executable semantics versus actual checkpoint sharing.
