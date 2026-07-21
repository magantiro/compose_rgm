# COMPOSE generator run-lineage correction and decision package (V2)

**Created:** 2026-07-20 EDT  
**Scope:** unconditional-generator evidence, its implications for conditional work, and the code/artifacts needed to choose the next backbone  
**Target worktree:** `/Users/rmaganti/Documents/Codex/2026-07-14/ok-so/compose_rgm_claude_generators`  
**Target branch at creation:** `claude/generator-cond-uncond`  
**Branch HEAD observed at creation:** `517889f38fe2f18a493bed92b96d20a563ddb3a4`

## 0. Precedence and correction

Read this document **before** `docs/HANDOFF_CLAUDE_GENERATORS.md`. It corrects
one material omission in the original handoff: the old handoff treated the
calibrated step-6,250 pancake checkpoint as the retained unconditional
incumbent without putting two distinct later branches into the canonical
decision table.

The omitted evidence is:

1. the quotient-correct, flexible-Graft, whole-ring-system step-1,000 preview,
   which nearly matched the held-out fused and spiro prevalences and eliminated
   measured canonical self-Grafts and delete-to-one collapse, but badly
   overproduced small and bridged rings; and
2. the later 2,000-sample factorized-tree run, which matched fused prevalence
   and nearly matched small-ring prevalence, but overproduced spiro/bridged
   systems and learned a delete-most-then-rebuild trajectory.

The original handoff was therefore not lossless about **run lineage**. Raw
artifacts were present, but the mandatory reading order and incumbent decision
did not surface them. Do not inherit the earlier incumbent choice. Reconstruct
the decision from the evidence below.

This correction does **not** assert that rings are solved. It asserts that
different runs solved different failure modes and that choosing a backbone
requires comparing topology, chemistry, and edit dynamics together.

## 1. Scientific invariants that do not change

Whichever backbone is selected must preserve the COMPOSE thesis:

- target-free unconditional sampling is ancestral CTMC sampling, never beam
  search;
- every committed molecular state is sanitizable, valence-valid, connected,
  and non-null;
- the process remains flexible-size and non-monotone;
- insertion, deletion, retyping/restating, topology revision/Graft, bond or
  electronic revision, and coordinated ring changes remain expressible;
- canonical molecular successors, not syntactic atom-slot descriptions, are
  the physical jump states;
- stochastic rewrite rules/application conditions define executable support;
  Generator Matching learns context-dependent rates over that support;
- conditional work must count every oracle call and compare against matched
  direct-conditioning and filter/rejection controls.

Do not collapse the model into monotone autoregressive construction merely to
obtain a quick metric. Do not retain an expensive mechanism solely because it
was expensive to build.

## 2. The three evidence-bearing unconditional lineages

### 2.1 Lineage A — legacy pre-quotient “pancake” checkpoint

**Identity**

- selected step: 6,250;
- checkpoint SHA-256:
  `47716924f7798ed24556c5aa8fb10c533c55dbd1f02f8f53a463cf2ad80ae2bf`;
- main evaluation denominator: 2,000 generated molecules;
- main sources:
  - `diagnostics/pancake_step6250_eval2000_event_audit.json`;
  - `diagnostics/pancake6250_calibration_eval600_metrics.json`;
  - `docs/audits/2026-07-19_pancake_to_quotient_run_audit.md`;
  - `results/diagnostics/step1000_ring_topology_comparison.json` for the
    frozen cross-run topology comparison.

**Strengths**

- neutral-CNOF matched FCD: **11.288** on 2,000 samples;
- final validity, uniqueness, and novelty: 100% in the frozen audit;
- mean final atoms 25.07 versus 26.61 in the matched reference;
- it is the strongest observed FCD among the three lineages and the checkpoint
  already qualified for the current conditional/canonical-successor adapter.

**Failures**

- fused prevalence 26.45% versus 52.18% reference;
- spiro prevalence 0.55% versus 2.74%;
- bridged prevalence 2.8% versus 3.2%;
- small 3/4-member-ring prevalence 29.9% versus 5.26%;
- mean cycle rank 2.40 versus 3.41;
- Graft/bond-reroute is 76.07% of 143,561 events;
- one archived full trajectory contains 89 canonical molecular self-events;
- rings are almost completely deferred: first ring event at normalized
  position 0.978 and 99.31% of ring events in the last decile.

**Interpretation**

This is the best *distributional/FCD incumbent*, not the best mechanistic
editor. Its apparent family accuracy and FCD cannot erase the self-transition,
phase-collapse, and fused-ring failures. It remains a useful conditional base
and comparison arm, not the automatically correct scientific endpoint.

### 2.2 Lineage B — quotient-correct flexible-Graft + whole-ring-system branch

**Identity**

- training run:
  `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1`;
- step-1,000 preview run:
  `compose-v4-stage3-step1000-preview100-v1`;
- step-1,000 checkpoint SHA-256:
  `c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c`;
- preview denominator: 100 generated molecules;
- primary sources:
  - `results/diagnostics/README.md`;
  - `results/diagnostics/step1000_ring_topology_comparison.json`;
  - `results/diagnostics/step1000_preview100_grid.png`;
  - `docs/audits/2026-07-19_pancake_to_quotient_run_audit.md`;
  - recipes `tree_fcd_transfer_stage3_flexible_graft.json`,
    `tree_fcd_transfer_stage3_short_continuation.json`, and
    `tree_fcd_transfer_stage3_legacy_transfer_gate.json`.

**Step-1,000 strengths**

- 100/100 valid, connected, and unique;
- zero canonical molecular self-events over 2,750 committed events;
- zero delete-to-one collapse over 100 trajectories;
- pendant ring-system prevalence 48.0% versus 49.96% reference;
- fused prevalence 48.0% versus 52.18%;
- spiro prevalence 3.0% versus 2.74%.

This is the strongest evidence that canonical-successor quotienting, flexible
Graft, and coordinated whole-ring-system actions can remove the pancake’s main
mechanistic pathology while restoring fused/spiro support.

**Step-1,000 failures and uncertainty**

- only 100 samples: topology estimates are noisy and not publication-grade;
- full-reference FCD 28.688 and matched neutral-CNOF FCD 23.044;
- bridged prevalence 9.0% versus 3.2%;
- small-ring prevalence 51.0% versus 5.26%;
- family accuracy was not directly comparable to the pancake objective/data;
- the 3,000-step cosine schedule decayed while validation loss was still
  improving, confounding scientific and optimization conclusions.

**Later evidence on the same branch**

- at step 1,500, small-ring prevalence improved to 42.2%;
- in the partial step-2,500 set it improved to 36.25%, while fused prevalence
  was 32.5%;
- the completed step-2,500 n=100 set had 40% small rings, full FCD 24.708, and
  matched neutral-CNOF FCD 19.180;
- the bounded fresh-optimizer continuation worsened validation loss and was
  rejected;
- the legacy-transfer gate also lost to the selected quotient checkpoint;
- exact replay localized every newly created small ring to
  `ring_system_grow`, not to Graft;
- the late-state support often contained only rare small-ring templates, so a
  Boolean “ring family enabled” decision renormalized the residual support to
  probability one.

**Interpretation**

This branch is the best *mechanistic editor/topology-support candidate*, but it
is not a finished unconditional generator. “Fused and spiro are fixed” is too
strong; the correct statement is that a 100-sample checkpoint nearly matched
those two marginals while badly missing other ring marginals. The architecture
must remain in the candidate set because it uniquely removed measured
self-thrashing and collapse.

### 2.3 Lineage C — factorized-tree `tree_fcd_transfer_stage1_factorized_v1`

**Identity**

- run label: `tree_fcd_transfer_stage1_factorized_v1`;
- selected step: 7,200 (run trained through step 8,000);
- evaluation denominator: 2,000 generated molecules against 5,000 held-out
  reference molecules;
- source prior: empirical-size carbon tree, four couplings per target;
- teacher ordering: sequential;
- training backend: `factorized_marks`;
- action families in the stored rollouts: atom delete, atom insert, atom
  restate, cycle attach, cycle insert, and ring-ear insert;
- primary sources:
  - `results/tree_fcd_transfer_stage1_factorized_v1/metrics.json`;
  - `results/tree_fcd_transfer_stage1_factorized_v1/audit_v1/audit.json`;
  - `results/tree_fcd_transfer_stage1_factorized_v1/manifest.json`;
  - recipe `recipes/tree_fcd_transfer_stage1.json`.

**Strengths**

- validity 100%; uniqueness 99.5%; novelty 100%; internal diversity 0.8999;
- FCD **14.577** on 2,000 samples;
- selected validation family accuracy 89.50%; final test family accuracy 90.20%;
- fused prevalence 52.2% versus 52.18% reference;
- small-ring prevalence 6.45% versus 5.26%;
- cycle rank 2.662 versus 2.624 in the recorded rollout reference;
- element-distribution TV 0.0122 and bond-order TV 0.0183.

**Failures**

- spiro prevalence 6.65% versus 2.74%;
- bridged prevalence 11.75% versus 3.2%;
- aromatic-ring fraction and mean aromatic rings are low;
- mean rings 2.6995 versus 3.378;
- FCD remains worse than the pancake’s 11.288;
- atom deletion is 69.66% of all events and insertion 20.86%;
- the mean initial delete run is 19.18 events: the model effectively destroys
  most of the carbon-tree prior and rebuilds, defeating the intended editing
  and useful-prior story;
- its ring-ear/cycle action substrate is not the same as the quotient branch’s
  atomic whole-ring-system substrate, so the topology match is not evidence
  that the quotient branch’s ring-rate problem is solved.

**Interpretation**

This is the best *large-sample balanced ring-marginal and family-accuracy
result*, but not the best editor and not the best FCD result. It proves that
the model family can learn fused and small-ring marginals; it also identifies
delete-collapse and excess bridge/spiro construction as serious failures.

## 3. Side-by-side decision table

| Criterion | Pancake 6,250 | Quotient step 1,000 | Factorized-tree step 7,200 |
|---|---:|---:|---:|
| Samples | 2,000 | 100 | 2,000 |
| FCD | 11.288 matched CNOF | 23.044 matched CNOF; 28.688 full | 14.577 full |
| Valid | 100% | 100% | 100% |
| Unique | 100% frozen audit | 100% | 99.5% |
| Fused | 26.45% | 48.0% | 52.2% |
| Fused reference | 52.18% | 52.18% | 52.18% |
| Spiro | 0.55% | 3.0% | 6.65% |
| Spiro reference | 2.74% | 2.74% | 2.74% |
| Bridged | 2.8% | 9.0% | 11.75% |
| Bridged reference | 3.2% | 3.2% | 3.2% |
| Small 3/4 ring | 29.9% | 51.0% | 6.45% |
| Small-ring reference | 5.26% | 5.26% | 5.26% |
| Canonical self-Grafts | observed/severe | 0/2,750 | not the same Graft substrate |
| Delete-to-one/delete-most collapse | not dominant | 0/100 | severe delete-most collapse |
| Main value | best FCD/conditional incumbent | best non-thrashing editor evidence | best large-sample fused/small-ring calibration |

FCD values marked “matched CNOF” and “full” are not interchangeable. Sample
size also differs. Never rank these runs from one column alone.

## 4. Current descendant experiments in the Claude worktree

At branch HEAD `517889f`, Claude has already run bounded transfers from the
pancake base. Preserve them as negative/partial evidence:

- `diagnostics/unconditional_p1context_trained_nobandage_eval200.json`:
  fused 9.0%, spiro 0%, bridged 1.0%, small rings 52.5%; chemistry remains
  strongly mismatched.
- `diagnostics/unconditional_topology_head_trained_nocalib_eval185.json`:
  fused 16.76%, spiro 0.54%, bridged 3.24%, small rings 35.14%; triple bonds are
  severely overproduced.
- `diagnostics/unconditional_combined_chem_topology_nocalib_eval180.json`:
  fused 21.11%, spiro 0%, bridged 11.11%, small rings 31.11%; heteroatom,
  aromaticity, QED, and SA distributions remain poor.

These arms show that limited head-level fine-tuning from the pancake checkpoint
does not automatically recover the quotient branch’s fused/spiro behavior or
the factorized-tree branch’s small-ring calibration. Do not promote them over
the three primary lineages without a matched panel.

## 5. Code map

### 5.1 State, executable rewrites, and transport

- `src/compose_v4/rewrite/operators.py`: primitive graph actions.
- `src/compose_v4/rewrite/tracelets.py`: coordinated tracelet actions,
  including lowering/application/inversion of `ring_system_grow`.
- `src/compose_v4/rewrite/tree_transport.py`: carbon-tree coupling, Graft, and
  atomic ring-system transport construction.
- `src/compose_v4/rewrite/ring_system_fiber.py`: semantic ring-system actions,
  local electronics, enumeration, executable support, and canonical keys.
- `src/compose_v4/rewrite/typed_ring_catalog.py`: typed ring proposal catalog.
- `src/compose_v4/rewrite/ring_junctions.py`: fused/bridged/spiro topology.
- `src/compose_v4/rewrite/kernel.py`: rule registration/application.
- `src/compose_v4/rewrite/commuting_schedule.py`: dependency/commutation-aware
  teacher scheduling.
- `src/compose_v4/rewrite/factorized_fiber.py` and `fiber.py`: executable
  teacher/action fibers.

### 5.2 Models, loss, and canonical successors

- `src/compose_v4/model/tracelet_rate_model.py`: earlier tracelet rate model.
- `src/compose_v4/model/factorized_tracelet_rate_model.py`: hierarchical
  factorized marked-rate model, family/action heads, ring support, and
  factorized likelihood.
- `src/compose_v4/experiments/canonical_successor_distillation.py`:
  canonical-successor aggregation, analytic pancake-to-quotient adapter, and
  distillation loss.
- `src/compose_v4/experiments/factorized_mark_priors.py`: empirical mark/base
  priors.
- `src/compose_v4/experiments/factorized_mark_conditional.py`: trainable scopes
  and conditional residual machinery.

### 5.3 Sampling, training, and evaluation

- `scripts/train_tracelet_cnof_gate.py`: principal training/evaluation CLI.
- `modal_apps/train_tracelet_gm.py`: Modal training/compilation entrypoint.
- `scripts/evaluate_tracelet_rollouts.py` and
  `modal_apps/evaluate_rollout_shards.py`: rollout evaluation.
- `src/compose_v4/experiments/parallel_tracelet_sampling.py` and
  `tracelet_sampling_worker.py`: ancestral rollout execution.
- `src/compose_v4/experiments/calibrated_rewrite_sampling.py`: explicit
  calibration/support-policy experiments; do not confuse inference bandages
  with trained solutions.
- `src/compose_v4/eval/molecular_quality.py`: FCD/descriptor/quality metrics.
- `src/compose_v4/eval/ring_taxonomy.py` and `ring_calibration.py`: ring
  prevalence and calibration.
- `scripts/analyze_unconditional_sufficiency.py` and
  `scripts/diagnostics/audit_pancake_rollouts.py`: cross-run diagnostics.

### 5.4 Conditional backbone

- `src/compose_v4/experiments/griddd_conditional.py`: qualified
  canonical-successor conditional execution contract.
- `modal_apps/train_qed_frozen_residual.py`: QED residual training.
- `scripts/evaluate_qed_controlled_rollouts.py` and
  `scripts/evaluate_property_conditioned_rollouts.py`: molecule-level
  conditional evaluation.

The current conditional adapter is tied to the pancake-derived qualified
canonical-successor backbone. Do not silently swap in Lineage B or C; define
and qualify a new adapter/checkpoint contract and rerun the exact execution and
zero-sidecar equivalence gates first.

## 6. Relevant tests

Before any backbone promotion, run at minimum:

- `tests/test_tree_transport.py`
- `tests/test_ring_system_grow.py`
- `tests/test_ring_system_fiber.py`
- `tests/test_ring_system_polycyclic_parity.py`
- `tests/test_typed_ring_superposition.py`
- `tests/test_commuting_schedule.py`
- `tests/test_factorized_mark_rate_model.py`
- `tests/test_factorized_fiber.py`
- `tests/test_parallel_tracelet_sampling.py`
- `tests/test_calibrated_rewrite_sampling.py`
- `tests/test_canonical_successor_distillation.py`
- `tests/test_ring_taxonomy.py`
- `tests/test_ring_calibration.py`
- `tests/test_griddd_conditional.py`

Then run the full suite before a milestone commit.

## 7. Reusable data and remote artifacts

### Factorized-tree lineage

The checked-in manifest records the remote paths:

- checkpoint: `/artifacts/tree_fcd_transfer_stage1_factorized_v1/checkpoint.pt`;
- recovery checkpoint:
  `/artifacts/tree_fcd_transfer_stage1_factorized_v1/checkpoint.recovery.pt`;
- compiled paths:
  `/artifacts/tree_fcd_transfer_stage1_factorized_v1/compiled_paths.pt`;
- rollout cache:
  `/artifacts/tree_fcd_transfer_stage1_factorized_v1/rollouts.pt`;
- metrics:
  `/artifacts/tree_fcd_transfer_stage1_factorized_v1/metrics.json`.

Do not assume those paths still exist: verify the Modal volume before launch.

### Quotient lineage

The local repository retains metrics, images, recipes, and the checkpoint hash,
but the checkpoint payload is not represented as a checked-in binary. Locate it
by run ID and SHA on the COMPOSE Modal artifact volume before declaring it
unavailable. Never substitute a same-step checkpoint with a different hash.

### Pancake lineage

The current qualified conditional code expects checkpoint SHA
`47716924...ae2bf`. Its qualification manifests and panels are retained under:

- `diagnostics/canonical_successor_analytic_backbone_qualification.json`;
- `diagnostics/canonical_successor_analytic_rate_equivalence_panel.json`;
- `artifacts/canonical_successor_backbone_qualification/`;
- `configs/experiments/griddd_qed_frozen_residual_*`.

## 8. Decision Claude must make

Do not decide from “best rings,” “best FCD,” or “best family accuracy” alone.
Construct one matched, fixed-seed evaluation panel over every retrievable
candidate with:

1. at least an initial 200-sample gate, then 2,000 samples for finalists;
2. identical held-out reference and chemistry subset;
3. validity, connectivity, non-null rate, uniqueness, novelty, and FCD;
4. atom count, element, bond order, cycle rank, aromaticity, QED, SA, and
   scaffold distributions;
5. fused, spiro, bridged, pendant, macrocycle, 3/4-ring, and ring-size metrics;
6. event-family fractions and time positions;
7. canonical self-events and immediate inverse/reversal events;
8. minimum atom count reached, longest delete run, and fraction deleting to
   one or deleting most of the source;
9. wall-clock sampling throughput and event count;
10. one blinded molecule grid plus trajectory panels selected by fixed rules.

The likely research direction—but not a preselected answer—is to preserve the
quotient branch’s canonical, non-thrashing flexible-Graft editor and transfer
the factorized run’s successful topology calibration without importing its
delete-collapse or ring-ear excesses. A simpler alternative may win. Claude is
explicitly authorized to choose another branch if the matched evidence is
better.

## 9. Bounded experiments to distinguish the choices

Run the cheapest falsifiers first:

1. **Artifact recovery:** verify whether the exact quotient step-1,000 and
   selected step-2,500 checkpoints remain on the volume.
2. **Matched reevaluation:** rerun both on the current frozen 200-sample panel;
   the original n=100 prevalence is not enough for selection.
3. **Quotient ring-hazard repair:** replace Boolean ring-family enablement with
   masked, unnormalized topology-group intensities so loss of common support
   lowers ring hazard rather than transferring it to rare small rings. Preserve
   rare support; do not hard-ban rings as the claimed solution.
4. **Factorized delete-collapse ablation:** determine whether the initial
   delete run is caused by teacher ordering, source/target coupling, or learned
   family rates. Test a bounded non-collapse/commuting-teacher arm without
   changing ring operators simultaneously.
5. **Cross-lineage hybrid only after isolation:** import one proven component
   at a time. Do not conflate quotienting, ring operators, teacher order,
   schedule, and model capacity in one long run.

Every arm needs a written expected benefit, time/cost estimate, abort rule, and
which failure it tests before launch.

## 10. Conditional implication

Conditional work can continue on the qualified pancake-derived canonical
adapter while the unconditional backbone decision is resolved. Do not claim
that the unconditional backbone is frozen forever. If Lineage B, C, or a hybrid
wins, qualify it as a new conditional base through:

1. exact canonical-successor execution;
2. zero-sidecar equivalence to its unconditional law;
3. frozen fixed-lead QED panel;
4. identical oracle accounting and constraints;
5. matched comparison to the existing pancake-derived base.

The paper can use different selected checkpoints for unconditional evidence and
conditional experiments only if this is disclosed and scientifically justified.

## 11. Required acknowledgement

Before launching more generator work, report:

1. whether each exact checkpoint payload was found;
2. a reproduced table of the three lineages with denominators and reference
   definitions;
3. which current claims are retracted or narrowed;
4. the matched evaluation plan and its compute estimate;
5. the selected next experiment and why it distinguishes competing causes;
6. confirmation that no large run starts before those checks.

