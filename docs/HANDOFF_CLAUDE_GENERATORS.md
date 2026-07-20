# COMPOSE generator lanes: lossless Claude handoff

**Handoff cut:** 2026-07-20, approximately 14:26 EDT

**Scope:** Paper 1 unconditional and conditional molecular generators

**Repository:** `https://github.com/KoshaTx/compose_rgm`

**Intended Claude branch:** `claude/generator-cond-uncond`

This document is the canonical index for a second Claude Code session that owns
the COMPOSE unconditional and conditional generator work. It is intentionally
more detailed than a chat handoff, but it is not a substitute for the governing
HTML plans or the machine-readable artifacts. Read the listed sources in full,
verify hashes and live jobs, and preserve negative results rather than rewriting
history.

## 1. Ownership and coordination

Claude-generator owns after this snapshot:

- the validity-closed COMPOSE/RGM implementation;
- unconditional de-novo molecular generation;
- carbon-tree source/coupling and certified teacher paths;
- canonical-successor quotienting and Graft behavior;
- whole-ring-system operators and ring topology/electronics;
- conditional property targeting, valid-successor control, and molecular
  optimization;
- the GrIDDD/FreeGress-style public benchmark implementation and evidence;
- generator tests, diagnostics, recipes, Modal launch code, and Paper 1 result
  audits.

Claude-lipid remains in the separate worktree/branch
`compose_rgm_claude_lipid` / `claude/lipid-corpus-oracle` and owns the lipid
pretraining corpus and pan-lung oracle matrix. Do not edit that worktree.

Codex will keep the cloud calls already listed below under read-only observation
until ownership is acknowledged. To prevent conflicting edits, generator code
changes after the handoff should be made in the Claude-generator worktree and
returned as commits. Do not work in the Codex checkout.

## 2. Mandatory read order

Read every file completely before changing the scientific plan or launching a
large job:

1. `docs/research_plans/compose_two_paper_execution_plan.html`
2. `docs/research_plans/paper1_compose_methods.html`
3. `docs/research_plans/paper2_compose_lipid.html`
4. `docs/audits/2026-07-20_unconditional_reporting_and_conditional_transition.md`
5. `docs/audits/2026-07-20_unconditional_chemistry_failure_audit.md`
6. `docs/audits/2026-07-20_unconditional_root_fix_execution.md`
7. `docs/audits/2026-07-20_pancake_ring_fusion_loss_decomposition.md`
8. `docs/audits/2026-07-19_pancake_to_quotient_run_audit.md`
9. `docs/audits/2026-07-20_canonical_successor_distillation_design.md`
10. `docs/audits/2026-07-20_conditional_backbone_selection.md`
11. `docs/audits/2026-07-20_valid_state_conditional_control_design.md`
12. `docs/audits/2026-07-20_qed_conditional_pilot.md`
13. `docs/audits/2026-07-20_griddd_qed_benchmark_inputs.md`
14. `docs/audits/2026-07-20_griddd_conditional_execution_contract.md`
15. `docs/audits/2026-07-20_tomorrow_griddd_claim_gate.md`
16. `docs/audits/2026-07-20_griddd_analytic_zero_sidecar_rewrite_smoke.md`
17. `docs/CLAUDE_GENERATOR_HANDOFF_MANIFEST_V1.json`

The three HTML plans govern the paper boundary, claims, experiment order, and
activation gates. A dated audit may record a later negative result or an
implementation correction, but do not silently rewrite an HTML plan. Propose a
redline with evidence if the plan must change.

## 3. Scientific thesis and claim boundary

### 3.1 Core thesis

COMPOSE is **Rewrite Generator Matching**: Generator Matching is specialized to
the admissible matches of a stochastic molecular rewrite system. Qualitative
rules and structural application conditions define which jumps exist. A neural
marked-rate model learns when, where, and with which chemical mark a rule fires.
All descriptions that reach the same canonical molecular successor are
aggregated at that successor.

The model is a flexible-size, non-monotone CTMC over complete connected
molecular states. Every committed intermediate is a sanitizable, valence-valid,
connected molecular graph. Sampling from the unconditional model is target-free
ancestral CTMC sampling, not beam search. Conditional control is a distinct arm
and must retain exact oracle accounting.

### 3.2 Novelty framing

Do not claim that stochastic rewriting, CTMCs, flexible-size jumps, Generator
Matching, molecular grammars, insertion/deletion, or chemical graph rewriting
are individually new. The owned contribution is the structured combination and
its consequences:

- Generator Matching written over executable molecular rewrite matches;
- a validity-closed, connected, flexible-size, non-monotone molecular process;
- canonical-successor aggregation of rule/match aliases;
- transport-aware carbon-tree teachers with derived Graft and whole-ring
  transactions;
- one model spanning de-novo generation, targeting, optimization, and editing;
- empirical testing of whether valid intermediates improve usable-oracle
  efficiency, constraint adherence, anytime candidates, and matched-budget
  molecular design.

Morph owns its own combination of flexible-size 3D geometric graphs, Edit
Flows/Generator Matching, insertion/deletion, continuous coordinates, and
unbalanced optimal transport. COMPOSE differs through 2D complete-molecule
validity closure, executable chemical rules, canonical successors, coordinated
Graft/ring transactions, and pathwise-valid conditional editing. CocoGraph fixes
the molecular formula and is not flexible-size. GrIDDD is a conditional graph
diffusion model and does not require valid connected intermediate molecules.
These distinctions are hypotheses to test, not permission to overstate
superiority before matched experiments.

### 3.3 Paper strategy

The unconditional generator must be credible and reportable, but FCD is not the
paper's sole battleground. Establish fidelity, diversity, flexible size,
revision behavior, and 100% committed-state validity/connectivity. The headline
is conditional targeting and similarity-constrained optimization under matched
budgets. Do not wait for endless FCD shaving before developing conditional
generation, but do not promote visibly pathological unconditional chemistry.

## 4. State space, operators, and training semantics

### 4.1 State and prior

- States are padded 2D molecular graphs with explicit atom and bond types.
- Bond representation is aromatic-aware; aromatic bonds are not forced into a
  fragile Kekule-only generation representation.
- The de-novo source is a directly samplable degree-bounded carbon tree.
- The current source-size prior is empirical over sizes 4--40.
- The production transport uses two tree couplings per target and
  `flexible_size_graft`, so size can grow or shrink around the source.
- A carbon tree is not a fragment prior: the process may insert, delete,
  retype, reorder bonds, Graft subtrees, and grow/delete whole ring systems.

### 4.2 Rewrite vocabulary

The conceptual production vocabulary is:

- atom grow/insert;
- atom shrink/delete;
- atom restate/retype;
- bond reorder;
- subtree **Graft** (detach and reconnect in one connected valid transition);
- whole `RingSystemGrow`;
- whole `RingSystemDelete`;
- ring-system electronic restatement where supported.

Some code and diagnostics retain legacy names such as `bond_reroute`,
`cycle_attach`, or `cycle_insert`. Read the executor and family mapping before
renaming anything. Graft and ring-system operations are verified derived
rewrites, not unstructured arbitrary multi-edit bundles.

One connected cyclic component is one atomic ring-system action. A fused,
bridged, or spiro system is installed as one coordinated connected system;
disjoint cyclic components are separate actions. Do not regress to primitive
ring closure/opening as the main ring mechanism and do not install every ring as
aromatic. Saturated and nonaromatic-unsaturated rings are legitimate corpus
chemistry.

### 4.3 Stochastic rewriting and Generator Matching

The legal marked-event set is built from rule matches and application
conditions. The model predicts a total hazard/family law and conditional mark
factors for site, template, placement, and electronic decoration. The training
loss is the factorized Generator-Matching/Poisson-Bregman objective over
certified teacher events. The unconditional model never receives the final
target molecule at sampling time.

Sampling is ancestral: at a state/time, compute the legal rates, sample a jump
time/event, execute it, and continue. No beam search is used. Valid-successor
control may sample multiple legal proposals and tilt/resample them using a
property potential; this is a controlled CTMC arm whose oracle calls must be
counted exactly.

### 4.4 Canonical-successor quotient

Several syntactic Graft/match descriptions may reach the same canonical
molecular state. Their rates must be summed rather than treated as distinct
chemistry. Molecular self-transitions are virtualized. The analytic pancake
quotient adapter has passed held-out rate and rollout gates and prevents
committed self/backtracking artifacts without changing the intended molecular
event law. Exact tree symmetries are a computational and statistical issue, not
an excuse to discard valid transitions.

## 5. Code map

Read these implementation files before changing behavior:

### Chemistry and state

- `src/compose_v4/chem/molecular_graph.py`
- `src/compose_v4/chem/state.py`
- `src/compose_v4/chem/aromaticity.py`
- `src/compose_v4/chem/source_prior.py`

### Rewrite semantics

- `src/compose_v4/rewrite/operators.py`
- `src/compose_v4/rewrite/kernel.py`
- `src/compose_v4/rewrite/compiler.py`
- `src/compose_v4/rewrite/tree_transport.py`
- `src/compose_v4/rewrite/tracelet_compiler.py`
- `src/compose_v4/rewrite/tracelet_fiber.py`
- `src/compose_v4/rewrite/factorized_fiber.py`
- `src/compose_v4/rewrite/typed_ring_catalog.py`
- `src/compose_v4/rewrite/ring_system_fiber.py`
- `src/compose_v4/rewrite/ring_junctions.py`
- `src/compose_v4/rewrite/commuting_schedule.py`

### Model, objective, and sampling

- `src/compose_v4/model/factorized_tracelet_rate_model.py`
- `src/compose_v4/gm/loss.py`
- `src/compose_v4/experiments/factorized_mark_conditional.py`
- `src/compose_v4/experiments/tracelet_conditional.py`
- `src/compose_v4/experiments/canonical_successor_distillation.py`
- `src/compose_v4/experiments/calibrated_rewrite_sampling.py`
- `src/compose_v4/experiments/guided_rewrite_sampling.py`
- `src/compose_v4/experiments/molecular_property_conditioning.py`
- `src/compose_v4/experiments/griddd_conditional.py`
- `src/compose_v4/experiments/training_support_cache.py`
- `src/compose_v4/experiments/training_support_compiler.py`

### Launch and evaluation

- `scripts/train_tracelet_cnof_gate.py`
- `modal_apps/train_tracelet_gm.py`
- `modal_apps/train_qed_frozen_residual.py`
- `scripts/evaluate_tracelet_rollouts.py`
- `scripts/evaluate_qed_successor_guidance.py`
- `scripts/evaluate_qed_controlled_rollouts.py`
- `scripts/run_griddd_real_rewrite_smoke.py`
- `scripts/run_griddd_analytic_zero_sidecar_smoke.py`

## 6. Unconditional lane: retained incumbent

### 6.1 Checkpoint

- Selected checkpoint: step 6,250 pancake model.
- Local path: `/private/tmp/pancake_checkpoint/checkpoint.recovery.pt`.
- SHA-256:
  `47716924f7798ed24556c5aa8fb10c533c55dbd1f02f8f53a463cf2ad80ae2bf`.
- Persistent compatible source:
  `/artifacts/compose-v4-stage3-full-ring-hierarchical-v1/checkpoint.recovery.pt`.
- Sampling calibration: atom-delete log-rate adjustment `-0.5`; small
  three/four-ring log-rate adjustment `-1.5`.
- Canonical-history safety: exact thinning; no beam search.

### 6.2 Frozen 600-sample evidence

Primary artifact:
`diagnostics/pancake6250_calibration_eval600_metrics.json`.

Positive evidence:

- 600/600 valid final molecules;
- 600/600 connected trajectories and all committed states valid;
- unique fraction 1.0; internal diversity 0.89467;
- novel-to-training fraction 1.0 under the recorded audit;
- no event-budget exhaustion;
- no canonical self events;
- no delete-to-one/regrow collapse;
- mean atoms 27.10 versus 26.35 reference;
- immediate backtracking is low: 0.1433% per opportunity and 3.67% of
  trajectories.

Negative evidence:

- mean cycle rank 2.458 versus 3.353 reference;
- mean rings 2.465 versus 3.374 reference;
- fused endpoints 28.5% versus 57.99%;
- bridged endpoints 1.667% versus 4.007%;
- spiro endpoints 0.667% versus 3.479%;
- small three/four-ring molecules 11.33% versus 6.03%;
- aromatic atom fraction 0.320 versus 0.481;
- generated mean QED 0.439 versus 0.602;
- SA score 4.488 versus 2.875;
- triple-containing molecules 54.0% versus 6.91% matched reference;
- excessive ring heteroatom and `[nH]` correlations documented in the
  chemistry failure audit.

This checkpoint is sufficient as a qualified conditional backbone but is not
the final publication-quality unconditional generator.

## 7. Unconditional experiments and causal conclusions

### 7.1 Failed fresh quotient routes

Fresh quotient/superposed runs underperformed the retained pancake lineage and
are not the default. Do not restart them merely because canonical quotienting
is theoretically attractive. Preserve the analytic quotient adapter around the
productive checkpoint unless a new formulation passes a bounded gate.

### 7.2 Chemistry P1/P2 pilot

The 500-step chemistry pilot completed all updates in 809.54 seconds and
improved validation loss/family metrics, but its rollout failed promotion.
Directional 98-sample evidence:

- triple-containing molecules fell from 54.0% to 14.29%;
- triple bonds per molecule fell from 0.802 to 0.153;
- ring >=3-hetero and O--O improved modestly;
- N--N and adjacent aromatic `[nH]` did not improve;
- small-ring molecules worsened to 45.92%;
- fused prevalence fell to 21.43%;
- Graft/reroute backtracking worsened sharply.

Conclusion: the corpus-based chemistry marks contain useful signal, but the
combined P1/P2 checkpoint is not promotable. Preserve stable pancake family
rates/quotient behavior, isolate chemistry marks, and model joint ring context
rather than independent atom-by-role correlations.

### 7.3 Ring scheduling falsification

Earliest legal commutation preserves all 32 endpoints/actions, and 29/32 ring
actions move earlier, but it fails the preregistered support gate:

- median fused admitted-prior mass increases only 1.186x versus required 2x;
- degenerate <=5-support rows remain 0 to 0;
- earlier states increase placement competition (median 6 to 18) and reduce
  exact-placement probability (0.590 to 0.175).

Do not pursue rigid early-ring scheduling as the immediate fix.

### 7.4 Fused/spiro topology decomposition

Whole-ring execution is correct. Across 9,946 teacher ring actions:

- single: 72.49%;
- bridged/fused: 26.20%;
- spiro: 1.287%;
- macrocycle: 0.020%.

Across 1,274 committed calibrated rollout ring actions:

- single: 84.69%;
- bridged/fused: 14.99%;
- spiro: 0.314%;
- macrocycle: 0%.

The checkpoint's global catalog has broad support, and the flat selector can
assign broad fused mass on late teacher states, but it does not generalize that
topology distribution to its own rollout states. Only 26/32 audited held-out
patterns have an exact production-template representative; median exact-teacher
template probability on covered rows is 0.00213.

Smallest current causal fix: a hierarchy over existing
`(topology_class, cycle_sizes)` groups before conditional exact-template
selection. The implementation mode is
`ring_template_factorization="topology_cycle_hierarchical"`. Its group head is
zero-initialized. `ring_topology_only` freezes encoder, family timing, total
hazard, Graft, exact-template keys, placement, and electronics. Old checkpoints
default exactly to `flat`.

Recipe: `recipes/tree_fcd_transfer_unconditional_ring_topology.json`.

## 8. Unconditional live execution at the handoff cut

### 8.1 Current CPU path/evaluation compile

- App: `ap-R32pM7XOzNxeptbjNmWfuM`.
- Function call: `fc-01KY0BAF1GKK50RNV86YQK77M6`.
- Run label: `compose-v4-unconditional-ring-topology-paths-20260720-v1`.
- Artifact:
  `/artifacts/compose-v4-unconditional-ring-topology-paths-20260720-v1`.
- State at approximately 14:27 EDT: detached, one task, train shard 8/25.
- Progress: 16,384/50,000 training targets; 32,768 transport records.
- The first eight shards remained healthy; shard 8 took 138.61 seconds.
- Mean rate remains approximately 28 records/second.
- Total planned corpus: 50k train + 2k validation + 2k test, two couplings
  each = 108,000 transport records.
- This is CPU-only. No topology GPU optimizer update exists yet.

The first detached training attempt `...ring-topology-pilot-...-v2`, app
`ap-2ABsijfQ5AfHB0KUSBCwrv`, call
`fc-01KY0B5EE4SM5JRJVKRHNFHZN9`, reached CUDA but correctly failed before step
1 because the old path-manifest configuration did not match. The failed app is
stopped.

### 8.2 Old v5 versus required v7 cache

The retained pancake path manifest is complete format v5 and has
transport-support projection v2. Its scientific scalar fields and split inputs
match current requirements. Current v7 additionally binds ring-system
electronic-alias version 1 and removes gauge-only Graft molecular self-events.
The frozen v5/checkpoint catalog reports electronic-alias version 0 with no
electronic aliases. These are scientifically material certificate/witness and
Graft-quotient changes, even though the structural catalog is expected to stay
ordered identically.

The old manifest and retained checkpoint catalogs are byte/pickle identical:
4,096 raw templates with matching counts/order, catalog SHA
`351164b4f5871d1d9444d1b723b28b8e2c98e1ad4ecac7b0af446ae122cac551`.
The production checkpoint selector has 2,074 deduplicated keys; shape-only
weight transfer is not sufficient evidence of semantic alignment. Applying the
unchanged structural deduplicator to both the v5 manifest catalog and retained
checkpoint catalog gives:

- ordered 2,074 selector templates SHA-256
  `d6f4d0d5a101f8654d34f88032c864cf68823cc8e8eee4809fbea1e7b6bd5262`;
- ordered alias groups SHA-256
  `b30c0f94850059f4e824840abdcb827590366658b278097f1dd3baa8da71ee54`;
- ordered 130 topology-cycle groups SHA-256
  `4eaabeecb80eae6ac67ba9555b0b1cbfb9fb59ec40f1c3388069cea96d4742ba`.

The finished v7 artifact must reproduce these structural hashes before weight
transfer, in addition to satisfying its new electronic-alias semantics.

Do not reuse/adapt v5 as though it were lossless. It covers only:

- 46,568/50,000 train targets;
- 1,800/2,000 validation targets;
- 1,799/2,000 test targets;
- 91,800 train support keys rather than exact 2x multiplicity.

Deleting self-Grafts cannot reconstruct missing targets/couplings or the exact
current deterministic row stream. The current decision is fail-closed: finish
v7 compilation, then compare exact catalog identity/key order/signature before
GPU training.

### 8.3 Required cache and GPU sequence

1. Finish v7 path and evaluation-cache compilation.
2. Verify complete manifest, exact 2x multiplicity, endpoint replay, v7 path
   fingerprint, evaluation signature/content, raw catalog identity/order, and
   the 2,074 canonical selector-key order against the checkpoint.
3. Compile only the required 500-step training-support range on CPU.
4. The default 16,000-row support shard equals 250 steps, allowing only two
   containers. A 3,200-row/50-step content-addressed layout would permit up to
   seven containers (98 reserved CPUs). Treat this as a cache-only
   infrastructure change and test ordered equivalence before using it.
5. Launch the 500-step A100 topology-only pilot only after all caches validate.
6. Evaluate the earliest credible checkpoint; do not select on teacher loss
   alone.

Promotion requires matched rollouts showing bridged/fused action share moving
toward 26.20%, no reduction in total ring actions, perfect pathwise/final
validity and connectivity, and no regression in small rings, spiro, aromatic
roles, bond-order/electronic chemistry, event budget, or throughput.

## 9. Conditional lane: developmental evidence

### 9.1 Step-500 direct conditioning

The developmental carbon-tree QED-conditioned model learned a real but weak
bidirectional target response:

- target means are monotonic from requested QED 0.3 to 0.9;
- QED-0.9 mean 0.6418 versus classifier-free 0.5745 in the ten-attempt pilot;
- all attempts valid and unique;
- no target-0.7/0.9 output reached QED >=0.9 in the original 20 high-target
  attempts;
- condition dose 1.1 increased mean QED to 0.7003, while 1.3 reversed,
  demonstrating saturation rather than unlimited extrapolation.

This authorizes bounded development, not a GrIDDD optimization claim.

### 9.2 Step-1000 continuation

Continuation improved Generator-Matching loss but not high-QED rollout means:

- selected validation loss 11.0227;
- validation top-1 60.21%, top-3 84.82%;
- frozen test top-1 58.56%, top-3 86.19%;
- target-0.7 and 0.9 means were worse than the step-500 pilot;
- validity and uniqueness remained 100%.

Do not continue ordinary direct training merely because validation loss falls.

### 9.3 Valid-successor control

On 100 saved step-1000 states, four valid proposals per state with beta 8 and
`U(q)=-|q-0.9|` gave:

- all 400 proposals valid and connected;
- soft expected QED improvement 0.00956;
- offline best-of-four improvement 0.02906;
- maximum selected QED 0.8864.

A ten-seed controlled rollout over early events improved mean QED by 0.0237 but
was noisy. Moving the same fixed proposal ceiling to events 13--24 improved mean
QED from 0.7096 to 0.7732, paired delta 0.0636; seven seeds improved, one
worsened, two tied, and two guided outputs reached QED 0.9. The bootstrap
interval remained wide, and the 48-call allowance was a ceiling rather than
fully consumed. This is positive mechanism evidence, not a matched GrIDDD
claim.

## 10. GrIDDD comparison contract

The primary comparable task is:

- starting QED in `[0.70, 0.80]`;
- 20 candidate attempts per start;
- candidate QED >=0.90;
- Morgan radius-2, 2,048-bit Tanimoto similarity >=0.40 to the lead;
- every start and failed/invalid/stalled/exhausted candidate retained in the
  denominator;
- native direct conditioning as the primary comparison;
- controller and combined arms reported separately because they consume
  inference-time oracle calls.

GrIDDD reports 45.1% success and diversity 0.283 for its optimization setting.
Do not claim an exact official-800 comparison unless the exact released lead
list is available and provenance-audited. The repository contains the exact
public Jin 800 and a separately labeled GrIDDD-code reconstruction. Neither may
be silently renamed the official GrIDDD list.

Exact matched control accounting at the full protocol is 981 QED calls per
start/seed/controlled arm: one lead eligibility call plus 20 candidates times
48 guidance + one terminal call. Productive unique-state calls, selection
calls, padding, and wall time must be separated.

## 11. Qualified conditional backbone

`diagnostics/canonical_successor_analytic_backbone_qualification.json` passes.
It binds:

- the retained checkpoint SHA;
- a 24-state panel and 12-state held-out rate gate;
- a 20-vs-20 rollout gate;
- analytic pancake quotient execution;
- canonical molecular successors and virtual self events;
- empirical mark prior `none`;
- ring family mass mode `boolean`;
- P1/P2 not imported;
- atom-delete -0.5 and small-ring -1.5 calibration.

This is sufficient to develop the conditional sidecar without waiting for the
unconditional topology pilot. It does not make the incumbent publication-ready.

## 12. Real three-arm execution smoke

Durable artifact:
`diagnostics/griddd_analytic_zero_sidecar_rewrite_smoke.json`, SHA prefix
`5aa1753f...`.

- lead QED 0.743501, within required start interval;
- target QED 0.90;
- direct, controller, and combined arms each executed four real rewrites;
- all finals valid and connected;
- final lead similarity 0.428571 >=0.40;
- each arm used exactly 6/6 QED calls;
- direct selection calls 0; controller/combined selection calls 2 and padding
  calls 2;
- final QED 0.396054 and zero successes in every arm;
- zero-initialized target-present/missing-condition rates equal the qualified
  base exactly.

The smoke validates execution, constraint/accounting plumbing, and zero-init
identity. It is deliberately not efficacy evidence.

## 13. Conditional frozen-residual live execution

The intended adapter is a zero-initialized, target-present-aware QED sidecar on
the frozen qualified backbone. Protocol-A inputs are target QED, target-present
mask, valid state graph, and model time. Current/successor QED, Tanimoto, and
oracle tilt are prohibited inputs for the native direct arm.

### 13.1 Failed attempts that must remain in history

- Raced apps `ap-mg...`, `ap-fY...`, and `ap-Fw...` were cancelled before
  artifacts.
- `ap-Hhfeh6k91K5Y0Fyd1vELUl` v2 failed before artifact creation because the
  run label did not exactly match the frozen config.
- `ap-gO8EJsECHmwZc36Gi5h30e` v1 was launched without `--detach`; the parent
  exited with empty `RemoteError`, zero tasks, and no artifact.
- v3 `ap-Mr0Ky0cTaKR0a4GNipkaXZ` wrote `run_config.json` then failed because the
  typed ring catalog did not support the flexible-Graft trace directly.
- v4 `ap-Af6Sg4dTViRtjjEvV5907p` used strict exact catalog identity and retained
  0/32 validation paths, so it failed correctly before training.

### 13.2 Completed v5 at handoff cut

- App: `ap-rPVKcg27IM4dLFfUz5EEhW`.
- Function call: `fc-01KY0BYYJEBXCBYHNMZBQ61EWY`.
- Run label/artifact:
  `compose-v4-griddd-qed-frozen-residual-pilot-20260720-v5-canonical`.
- Config SHA:
  `6a63c154bdab9d97c82c8ce391e3e4e0a8589385993ff2948ce2396fc19fba01`.
- State: completed successfully at step 500; no early stop fired.
- Support projection uses the production
  `structured_ring_trace_supported` predicate rather than strict serialized
  template identity.
- Paths retained: train 227/256, validation 27/32; gates were 128 and 16.
- Persisted step-0 artifacts: `evaluations/step_0000.json` and
  `checkpoint.best.pt`.
- Step-0 validation loss 15.2566204552.
- Step-0 family top-1 0.375 (12/32).
- Mean absolute family log-hazard residual 0.0.
- Maximum missing-condition identity error 0.0.
- Step 0 is a zero-init baseline, not learned performance.

Metrics observed before the snapshot was committed:

| Step | Validation loss | Family top-1 | Mean absolute family log-hazard residual | Identity error | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0 | 15.2566205 | 0.3750 | 0.0000 | 0.0 | exact zero-init baseline |
| 100 | 14.1706439 | 0.3750 | 0.0397 | 0.0 | learned response begins |
| 200 | 12.9446303 | 0.4167 | 0.0839 | 0.0 | improving validation |
| 300 | **12.1176692** | 0.4167 | 0.1762 | 0.0 | best observed checkpoint |
| 400 | 12.3019806 | 0.3750 | 0.2649 | 0.0 | transient regression |
| 500 | **11.1448012** | 0.4167 | 0.4406 | 0.0 | final and best checkpoint |

The final validation loss was 26.95% below step 0. Total elapsed training time
was 425.60 seconds. Single sampled-example training losses were very noisy and
were not used as the selector. The base was frozen; base weights are not stored
inside the sidecar checkpoint. Durable artifacts include `checkpoint.best.pt`,
recovery and step-0100 through step-0500 checkpoints, evaluations, metrics, and
status under the run-label directory. `protocol_a_launched=false` and
`protocol_b_ready_for_post_training_evaluation=true`.

Next gate: select the persisted best checkpoint by validation, confirm the final
task state, and rerun the real lead smoke with the saved trained weights. Only a
positive molecule-level QED effect authorizes a small fixed-lead panel. Do not
launch 800x20 from teacher loss, family accuracy, or residual magnitude alone.

## 14. Conditional files added in the current snapshot

- `modal_apps/train_qed_frozen_residual.py`
- `modal_apps/inspect_function_call.py`
- `configs/experiments/griddd_qed_frozen_residual_pilot_v3_canonical.json`
- `configs/experiments/griddd_qed_frozen_residual_pilot_v4_canonical.json`
- `configs/experiments/griddd_qed_frozen_residual_pilot_v5_canonical.json`
- `scripts/run_griddd_analytic_zero_sidecar_smoke.py`
- `diagnostics/griddd_analytic_zero_sidecar_rewrite_smoke.json`
- `docs/audits/2026-07-20_griddd_analytic_zero_sidecar_rewrite_smoke.md`

The failed v3/v4 configs are historical evidence. Do not delete or relabel them.
`diagnostics/pancake_ring_topology_pilot_status.json` durably records the v1
non-detached failure, the v2 path-signature failure before optimizer step 1,
and the replacement CPU-only v7 compile state at the handoff cut. Do not infer
that topology training started from the existence of the failed v2 manifests.

## 15. Publication evidence contract

### Unconditional sufficiency

Report at minimum:

- final and pathwise validity/connectivity;
- uniqueness, novelty, internal diversity, scaffold diversity;
- atom count, element, bond-order, cycle-rank, and ring-size distributions;
- fused, bridged, spiro, aromatic, saturated, heterocycle, and small-ring
  diagnostics;
- QED/SA/logP/MW/TPSA and FCD only under frozen sample/reference contracts;
- event-family rates, Graft backtracking, self events, collapse, stalls, and
  event-budget exhaustion;
- multiple seeds for final paper claims.

Do not die on FCD, but do not hide obvious chemical pathologies behind 100%
RDKit validity.

### Conditional and optimization

Report:

- direct property targeting MAE/tolerance success, diversity, and all-attempt
  validity;
- QED, penalized logP, DRD2, and similarity-constrained optimization under
  matched starts, thresholds, candidates, oracle calls, and compute;
- direct, controller, and combined arms separately;
- usable oracle calls, padding, unique-state calls, wall time, anytime success,
  scaffold/linker preservation, and pathwise violations;
- paired uncertainty/bootstrap comparisons.

### Mechanism and formal contract

Retain ablations for whole-ring versus primitive ring paths, Graft versus
primitive rerouting, hard legal support versus rejection, base rates versus
learned residuals, and valid-successor control. Exhaustively verify small-state
canonical-successor generator equivalence and executable sampled events.

## 16. Tests and validation

The repository-wide snapshot immediately before this handoff had 411 passed and
one skipped. The current unconditional focused changes passed 59 tests and
Ruff/diff checks. The current conditional implementation passed 23 focused
tests and Ruff before the v5 launch. Rerun the full suite after checking out the
Claude-generator worktree; do not assume live Modal success substitutes for
local tests.

Useful commands:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts modal_apps tests
.venv/bin/modal app list
.venv/bin/modal volume ls compose-v4-artifacts /RUN_LABEL
```

Do not expose Modal credentials, `.env` files, tokens, or private data in Git.

## 17. Immediate execution queue

1. Verify the current Git branch/worktree and read the mandatory documents.
2. Read-only inspect the two live Modal calls; do not duplicate them.
3. Conditional: record the final v5 state and best checkpoint; rerun the real
   lead smoke with the saved trained checkpoint and judge molecule-level QED
   efficacy rather than residual/family metrics alone.
4. Unconditional: let v7 CPU compilation finish, validate catalog/key identity,
   multiplicity, evaluation cache, and path signature.
5. Implement/test a smaller content-addressed support-shard layout only if it
   preserves exact row identity and ordered merge; otherwise use two safe
   containers.
6. Launch topology GPU training only after all cache gates pass.
7. Evaluate actual rollouts at the first credible checkpoint. Kill the topology
   mode if it does not repair fused/spiro action shares without regressions.
8. If topology passes, combine it carefully with the isolated successful
   chemistry-mark correction while preserving stable pancake behavioral rates
   and quotient handling.
9. Move from smoke to a small fixed-lead GrIDDD-style efficacy panel; scale to
   800x20 only after positive paired evidence and exact input provenance.
10. Commit every logical milestone with raw artifacts, negative results, exact
    commands, hashes, and stop/go decisions.

## 18. Return contract

For every milestone return:

- exact command, environment, branch, commit, and run label;
- Modal app and function-call IDs;
- checkpoint/config/data/cache hashes;
- artifact paths and counts before/after every filter;
- elapsed time, throughput, resource use, and failures;
- full-denominator metrics and uncertainty;
- comparison against frozen incumbent artifacts;
- explicit claim authorization or prohibition;
- code/config/test changes committed to `claude/generator-cond-uncond` and
  pushed for review.

Never turn a smoke test into an efficacy result, a reconstructed benchmark into
an official one, a lower teacher loss into rollout promotion, or a valid RDKit
graph into evidence of realistic chemistry.
