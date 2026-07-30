# COMPOSE editing operator, data-law, and launch-gate audit

The forward-looking replacement design derived from this audit is
`docs/EDITING_MODEL_FIRST_PRINCIPLES_SPEC.md`. This file remains the measured audit of the completed
RingCore-V1 corpus and operator behavior; the first-principles specification is the proposed contract for
the next editing model.

**Date:** 2026-07-29
**Scope:** source-conditioned broad-organic editing prior only
**Status:** design audit before checkpoint selection or repaired training
**Completed run:** `compose-v4-ringcore-v1-scientific-a7546e2-v1`

This document distinguishes three things throughout:

- **Measured:** established by frozen artifacts, production code, or an exact corpus census.
- **Inferred:** a design consequence supported by the measurements but not yet an experimental result.
- **Pending:** a question that must be resolved by a named test before changing production semantics.

It does not select a checkpoint, use the inspected test partition for selection, or authorize a new long
training run.

## 1. Executive decision

The state/executor/corpus infrastructure is strong enough to support a serious editing model, but the
**current effective learning distribution is not adequate for the full editing thesis**. In particular, raw
trace counts conceal a large teacher-rate and path-ordering imbalance:

- **Measured:** 69.34% of the expected selected-mark log-probability coefficient comes from the MMP layer.
- **Measured:** every MMP path begins with `atom_delete` and ends with `atom_insert`.
- **Measured:** after the actual record sampler, time/progress law, and teacher rate are integrated,
  `atom_delete` receives 51.18% of the selected-mark coefficient.
- **Measured:** `bond_reroute`, `cycle_attach`, and `cycle_insert` receive 3.45%, 2.71%, and 2.71%,
  respectively.
- **Measured:** a draw lands on a terminal state with probability 58.35%; only 41.65% of landings carry a
  selected-mark target.

Those quantities explain why another run cannot be authorized by dataset size, raw family counts, or
aggregate validation loss alone. They do not, by themselves, prove that the imbalance caused every observed
failure.

The current operator recommendation is:

| Operator | Recommendation | Reason |
|---|---|---|
| `atom_insert` | keep zero-/one-neighbor support | cardinality growth; sufficient for the observed corpus; multi-neighbor birth is not yet justified |
| `atom_delete` | keep | cardinality shrinkage and substituent removal |
| `atom_restate` | keep | direct element/valence movement without destructive delete/rebuild paths |
| `bond_reorder` | keep and strengthen supervision | functional-group and electronic-state editing |
| `bond_reroute` | keep and make load-bearing | positional-isomer movement, linker relocation, and compositional middle insertion |
| `cycle_insert` | keep | primitive cycle closure |
| `cycle_attach` | keep | primitive cycle opening and dynamic topology reversal |
| `ring_system_restate` | provisionally keep | coordinated aromatic/electronic ring changes that are not reliably one local reorder |
| `ring_system_delete` | **candidate for removal from the primary model** | active action is an opening-only one-/two-bond macro, overlaps primitive opening, and has no enabled inverse macro |
| `ring_system_grow` | keep disabled | finite-catalog growth is not the definition of RingCore support |

No production operator is changed by this document. `ring_system_delete` remains pending a bounded
reachability/path-cost comparison with and without the macro.

## 2. Exact frozen-corpus findings

The exact model-independent audit is implemented in `modal_apps/audit_editing_corpus.py`. It reads the
packed stores, applies the frozen three-record representability overlay, and integrates the actual
hierarchical record sampler and production time law.

Audit identity:

```text
schema:
  compose.diagnostics.editing_corpus_operator_audit

content_sha256:
  13f801fcc3510380cfb754ecf31ad981a1d63e8b755f1ad7d77b20ec43f5a822

remote immutable artifact:
  /artifacts/_frozen_corpus_audits/
  ringcore-v1-editing-corpus-audit-2026-07-29.json

unified_manifest_checksum:
  5c5c254e1054c081

representability_overlay_checksum:
  32372dc5d73139a7
```

Exact partition totals after the overlay:

| Partition | Records | Rewrite steps |
|---|---:|---:|
| train | 661,105 | 2,429,648 |
| validation | 33,404 | 115,231 |
| test | 31,159 | 100,261 |

### 2.1 What “effective supervision” means here

For a path with families \(a_0,\ldots,a_{K-1}\), the production GM loss gives the selected mark at progress
\(k\) a coefficient equal to the teacher rate \(K-k\). The expected coefficient of a family therefore
depends on:

1. the probability of drawing its record;
2. the probability of landing at each progress state under the time law;
3. its position in the path;
4. the teacher rate at that position.

The audit computes:

\[
\Pr(\text{record})\,
\mathbb E_t[\Pr(N_t=k)]\,
(K-k).
\]

This is the expected coefficient of the **selected-mark log-probability term**. It is not a claim about the
norm or direction of the complete neural-network parameter gradient.

### 2.2 Exact expected training-law shares

| Layer | Share of selected-mark coefficient |
|---|---:|
| MMP analogues | 69.34% |
| general corruption | 25.25% |
| cycle operations | 5.41% |

| Family | Share of selected-mark coefficient |
|---|---:|
| `atom_delete` | 51.18% |
| `atom_insert` | 25.11% |
| `atom_restate` | 7.49% |
| `bond_reorder` | 3.50% |
| `bond_reroute` | 3.45% |
| `cycle_attach` | 2.71% |
| `cycle_insert` | 2.71% |
| `ring_system_restate` | 2.59% |
| `ring_system_delete` | 1.26% |

The cycle layer is exactly direction-balanced in record count and expected coefficient: 127,584 opening and
127,584 closing records. Therefore the full-run ring-opening failure is **not explained by a simple
open-versus-close count imbalance**.

### 2.3 Path construction matters

The MMP layer contains 330,991 training paths and 1,955,849 steps:

- 978,498 `atom_delete` steps;
- 977,351 `atom_insert` steps;
- every path starts with deletion;
- every path ends with insertion;
- path lengths range from 2 to 16;
- every source-to-target cycle-rank delta is zero.

The raw insertion/deletion counts are nearly balanced, yet the integrated teacher-rate coefficient is not.
Early path positions carry larger \(K-k\), so the fixed path order strongly favors deletion. This is a
concrete example of why a count-based “balanced dataset” check would have approved a materially imbalanced
training law.

The general-corruption layer is much broader but synthetic:

| Family | Raw train steps |
|---|---:|
| `atom_delete` | 34,524 |
| `atom_insert` | 25,871 |
| `atom_restate` | 65,100 |
| `bond_reorder` | 30,361 |
| `bond_reroute` | 29,905 |
| `ring_system_delete` | 10,646 |
| `ring_system_restate` | 22,224 |

The one-step cycle layer teaches cycle-rank changes of exactly \(+1\) or \(-1\), but does not teach
multi-step ring-size changes, topology addition followed by property repair, or topology reversal after a
goal switch.

## 3. Semantic capability coverage, not merely operator coverage

The earlier project plan defined a five-part editing-process recipe:

1. data-backed analogue pairs at several molecular-graph distances;
2. reversible synthetic legal perturbations in both directions;
3. an edit-distance curriculum from precise local edits through ordinary lead optimization to
   scaffold-scale modification;
4. source/protected-subgraph/edit-budget conditioning;
5. a small full-support legal component.

That list needs two corrections under the current architecture:

- the source, protected mask, and remaining budget belong to the controller or augmented controlled state,
  not as mandatory inputs to the source-agnostic base prior;
- the full-support legal component is a declared kernel/process choice, not a category of training example.

The first three items remain binding data-design principles. They imply a semantic coverage matrix broader
than MMP `pair_type` or operator counts.

### 3.1 Required transformation regimes

Every revised corpus census must distinguish at least:

| Regime | Examples of the intended capability | Load-bearing operators |
|---|---|---|
| local substituent / bioisostere | terminal group replacement, atom identity change, small functional-group move | restate, insert/delete, reorder |
| linker and attachment relocation | linker lengthening/shortening, positional isomer, branch relocation | insert/delete, graft, reorder |
| cardinality adaptation | genuine molecular growth and shrinkage, including a later target-size switch | insert, delete |
| electronic / bond-state adaptation | bond-order and aromatic/electronic changes without destructive rebuild | reorder, atom/ring restate |
| ring/topology adaptation | open/close, ring-size change, fused-system modification, topology reversal | cycle open/close, possibly coordinated restate |
| longer scaffold restructuring | compositional movement between more distant analogues while retaining a declared core when required | multiple families |

These categories may overlap. The audit must report their **joint** coverage; for example, a corpus can have
many ring examples and many real analogues while having zero real ring-changing analogues.

### 3.2 Required context axes

For every regime, measure:

- compiled path length and shortest-known path overhead;
- source/target atom-count and graph-cycle-rank deltas;
- source/target similarity and maximum intermediate departure;
- protected or constant-core retention;
- attachment count and variable-fragment size;
- ring complexity, ring-system type, and ring-size change;
- element/valence classes, charge state, and rare broad-organic targets;
- operator sequence, direction, reversals, and repeated states;
- source scaffold, analogue series, and transformation signature concentration;
- train/validation/test coverage under source-, scaffold-, series-, and transformation-held-out rules.

Marginal histograms are insufficient. The gate needs a sparse joint table such as:

```text
transformation regime
  x path-scale bin
  x topology/cardinality delta
  x chemistry stratum
  x split
  -> records, unique sources, unique scaffolds, effective teacher coefficient
```

### 3.3 What the present corpus is known to cover

**Measured**

- The one-cut MMP layer supplies real-to-real analogue endpoints at several path lengths, but every
  compiled path uses only deletion followed by insertion, has one attachment site, and preserves graph
  cycle rank.
- The general-corruption layer supplies all seven of its enabled semantic operator families and path
  lengths one through five, but its targets are synthetic perturbations.
- The cycle layer supplies exactly balanced one-step opening and closing with cycle-rank deltas \(-1\) and
  \(+1\).
- The four legacy MMP labels (`grow`, `shrink`, `isosteric`, `atom_swap`) describe variable-fragment size
  relationships inside the one-cut compiler; they do not establish linker, topology, or scaffold-scale
  coverage.

**Inferred**

The current mixture is strong in one-cut substituent replacement plus synthetic operator support. It is not
yet an adequate demonstration of real, multi-operator linker relocation, ring-size/topology redesign, or
longer scaffold restructuring. Those are exactly the trajectories that Pareto branching, dynamic
retargeting, and pathwise constraints make scientifically interesting.

**Measured provenance limitation**

The exact packed states and executable actions are sufficient to reconstruct attachment count,
constant-core size, variable-fragment size, variable-ring participation, graph cycle rank, and directed
size class. The full 361,019-row effective MMP census has now done so and is frozen at
`diagnostics/coherence/ringcore_v1_mmp_semantic_coverage_full_local_2026-07-30.json`. The original
363,456-row source pool is nevertheless absent from the connected artifact volume, and the V2 packing step
did not copy its semantic descriptors. Consequently, the reconstruction cannot be compared against the
authoritative compiler metadata. This is not a training-state correctness defect; it is an
auditability/provenance defect. The revised schema should preserve the source pool, carry the descriptors
into the packed derivative, and bind both to the corpus contract.

## 4. Operator-by-operator audit

### 4.1 Atom insertion

**Measured**

- Production represents insertion from null with zero existing neighbors and connected insertion with
  exactly one existing neighbor.
- In the general-corruption training layer there is one root insertion and 25,870 one-neighbor insertions.
- Only three unsupported multi-neighbor insertion traces were found among 725,671 pre-overlay traces.

**Inferred**

One-neighbor insertion is a sensible first production boundary. It gives true cardinality growth without
adding a high-arity decoder. Many “insert in the middle” transformations can be represented by:

1. insert a pendant atom or fragment endpoint;
2. reroute an existing pendant bridge to the new atom;
3. restate/reorder locally if needed.

This makes graft reliability a prerequisite for claiming that one-neighbor insertion is compositionally
adequate.

**Pending**

- A task-level reachability census for linker lengthening, bond subdivision, and ring expansion under the
  actual edit horizons and protected masks.
- A comparison of shortest valid path length with and without two-neighbor insertion.

Do not add multi-neighbor insertion unless that audit finds a meaningful unreachable task class or an
unacceptable path-cost penalty. Three excluded corpus traces are not sufficient justification.

### 4.2 Atom deletion

Keep it. It is the direct trans-dimensional shrinking operation and is necessary for dynamic size
retargeting. The current model should not, however, learn a useful editing kernel in which deletion dominates
simply because it occurs early in long compiled MMP paths.

Required repair: either change the target trace/process construction or introduce an explicitly declared,
importance-aware family objective. Merely sampling the existing raw records “more evenly” is not a complete
mathematical specification.

### 4.3 Atom restatement

Keep it. It prevents many property changes from degenerating into delete-and-reinsert programs and is
especially important when source similarity, path length, or protected topology matters.

Required data: real analogue pairs whose minimal semantic edit is a restatement, with held-scaffold splits
and both chemically meaningful directions where supported.

### 4.4 Bond reordering

Keep it. Bond order is a high-leverage coordinate for functional groups, conjugation, and property movement.
The final mark-level top-1 of zero is not acceptable evidence for a flagship editing prior.

Required diagnostics:

- successor-level recovery and rank;
- candidate-set and alias statistics;
- charged/aromatic teacher round trips;
- tiny-set overfit;
- real-analogue rather than corruption-only coverage.

### 4.5 Bond rerouting / graft

Keep it and treat it as load-bearing. Graft is not a decorative legacy capability:

- it changes attachment position without deleting and rebuilding a whole substituent;
- it gives short paths between positional isomers;
- it supports compositional linker insertion with one-neighbor birth;
- it allows property movement while preserving a large fragment;
- it is useful for branch diversification around an underexplored Pareto region.

The completed run inherited a strong graft signal and subsequently lost it. Whether the next model is warm
started or trained from scratch, a graft successor-level gate is mandatory. From-scratch initialization
removes the specific “retention” obligation but not the “learnability and practical mass” obligation.

### 4.6 Primitive cycle closing and opening

Keep both. Dynamic topology control requires both directions: a controller must be able to add a cycle,
later reverse it when preferences change, and do so through valid intervention points.

The exact corpus census shows equal direction counts and equal expected coefficients. Existing micro-overfit
evidence shows that ring opening can be learned in isolation. The unresolved issue is competition under the
full mixture, not basic representability.

Required gates are successor-level and opportunity-aware:

- teacher successor probability and rank;
- learned-versus-uniform log-likelihood advantage;
- productive family mass on states where that family is legal;
- topology-task success under fixed budgets;
- open/close performance conditioned on raw and canonical candidate count.

### 4.7 Ring-system restatement

Provisionally keep it. The corpus contains coordinated changes spanning two to twenty ring bonds, with most
actions changing three bonds. This family can express aromatic/electronic transformations that are awkward
or invalid as a sequence of independently committed local reorders.

Before freezing it, verify:

- every intermediate that a primitive alternative would require;
- whether the macro changes reachable molecular support or only shortens paths;
- compatibility with protected scaffold and atom-identity constraints;
- successor-level learnability.

### 4.8 Active ring-system deletion

The name is misleading for the active clean-ring action:

- all 10,646 general-corruption instances delete **zero atoms**;
- 10,054 delete one bond;
- 592 delete two bonds;
- all are topology-opening operations;
- no inverse whole-ring growth macro is enabled.

It therefore overlaps primitive `cycle_attach`, but can make a multi-bond jump in one committed event. That
creates three concerns:

1. **Control granularity:** a multi-edge jump removes intervention points that the primitive trajectory
   would expose.
2. **Dynamic reversibility:** the matched inverse macro is disabled, so the edit-budget geometry is
   directionally asymmetric.
3. **Framing:** an opening-only catalog macro is hard to reconcile with the claim that compositional
   primitives define ring support and macros are optional accelerators.

The provisional recommendation is to remove it from the primary repaired model and retain it only as an
explicit optional-accelerator ablation. This is conditional on a bounded audit showing that it adds no
scientifically required reachable states under the production vocabulary and only shortens primitive paths.

### 4.9 Disabled ring-system growth

Keep it disabled. The current evidence neither evaluates nor requires it, and it cannot replace ring opening.
If a future macro is tested, it must be an explicitly paired, support-preserving efficiency proposal rather
than the definition of allowable topology.

## 5. Consequences for Pareto editing and dynamic control

A useful editing prior should not be optimized solely for one-step teacher accuracy. It must provide a
controllable graph of molecular successors.

### 5.1 Pareto exploration

Linear scalarizations alone systematically miss non-convex regions of a Pareto front. The intended workflow
should support:

1. maintain an external nondominated archive;
2. identify an undercovered objective-space cell or a region with high expected hypervolume gain;
3. condition a remaining-budget value on that target region and source/protection context;
4. branch multiple continuations from a valid prefix;
5. update the archive and allocate later oracle calls adaptively.

This favors operators that make distinct semantic moves with useful probability mass. Graft, atom
birth/death, restatement, reorder, and bidirectional cycle operations are complementary coordinates; a
delete/rebuild-dominated kernel wastes edit budget and oracle calls.

### 5.2 Temporary property valleys

Ring-size changes, linker moves, and scaffold-preserving substitutions can require an intermediate whose
one-step property score is worse. Greedy reward tilts cannot be the only controller. Remaining-budget value
control and SMC/Feynman--Kac baselines are needed to cross those valleys while retaining valid
intermediates.

### 5.3 Hard constraints

Hard successor predicates may prune legal base transitions and may produce \(h_b(x)=0\). Such states are
unreachable under that constraint and budget; they must be reported, never patched with epsilon mass.

There are two different constraint classes:

- charge, atom-count, alert, similarity, and substructure-presence predicates can often be evaluated on the
  canonical molecular successor;
- protected **source-atom identity**, provenance, or path statistics require an augmented state such as
  `(molecule, protected mapping, path statistic)`.

The molecular quotient alone can merge mark aliases that produce the same unlabeled molecule but treat
source atoms differently. Operator and controller tests must use the correct state space for the declared
constraint.

## 6. Data redesign

Preserve the existing frozen corpus as a baseline. Build a new versioned training contract rather than
silently mutating it.

### 6.1 Add operator-aware analogue layers

Compile real analogue/topology pairs into shortest or near-shortest **semantic** programs using:

- direct restatement when the atom identity changes locally;
- bond reorder for bond-state changes;
- graft for attachment relocation;
- primitive cycle close/open for topology changes;
- ring-system restatement only where coordinated electronic validity requires it;
- insertion/deletion for genuine cardinality change, not as the universal fallback.

Every layer must be split by source and scaffold before path generation. Include both directions when legal,
and record the exact slot-addressed states.

### 6.2 Add multi-step topology curricula

One-step open/close pairs are necessary but insufficient. Add held-out task families covering:

- ring creation and opening followed by local property repair;
- ring-size changes;
- chain-to-ring and ring-to-chain edits;
- fused-system simplification/modification;
- topology addition followed by reversal after a goal switch;
- spiro/bridged cases only after production reachability is measured.

The curriculum should include valid intermediate trajectories, because that is part of the scientific object.

### 6.3 Correct path-order and sampling effects explicitly

Before generating a new corpus, compute its expected family coefficient using the exact training law. Freeze
both:

- record/landing probabilities;
- selected-mark teacher-rate coefficients.

Possible repairs must be evaluated as distinct mathematical choices:

1. generate alternative valid linearizations where actions commute;
2. change path compilation so semantic operators replace delete/rebuild fallbacks;
3. use importance-weighted family or layer stratification that preserves a declared target process;
4. add a separately weighted successor-level auxiliary objective;
5. deliberately define a utility-oriented editing reference law rather than claiming it is the unmodified
   empirical path law.

Whichever choice is made must be written into the process definition. “Balanced batches” is not a scientific
definition.

### 6.4 Spend batches efficiently without dropping the hazard

The editing deployment uses a fixed-budget embedded jump chain, while the GM objective also trains hazard
and terminal/no-jump behavior. The current time law lands at terminal with probability 58.35%.

A principled repair can oversample nonterminal rows and retain unbiased hazard estimation through explicit
importance weights or a separate calibrated hazard stratum. The exact estimator must be specified and
tested; silently removing terminal rows would change the trained generator.

## 7. No-wasted-compute launch protocol

A long run is forbidden unless every earlier gate produces a machine-readable `PASS`. The same run may
continue through the pilot ladder, preserving optimizer, RNG, dataloader offset, and the **full-horizon**
learning-rate schedule. A recipe change creates a new run identity and restarts at Gate 0.

### Gate -1 — corpus/process preflight (CPU)

Required before model construction:

- manifest, overlay, codec, operator, capability, and split hashes match;
- all exclusions resolve exactly;
- every active family has declared teacher records;
- raw record/step counts are reported;
- expected landing mass and teacher-rate coefficients are reported by layer, family, and path position;
- no family silently falls below a frozen effective-coefficient floor;
- task reachability is audited under production budgets and masks.

### Gate 0 — initialization and support parity (CPU or one forward pass)

For warm start:

- transferred tensors and rows match the source checkpoint within tolerance;
- every inherited family is evaluated on a frozen retention probe;
- every newly initialized row/head is listed.

For from scratch:

- no inherited-parity claim is made;
- the same fixed family probes establish initialization baselines.

For both:

- every teacher is present in the exact legal candidate set;
- every legal mark executes;
- every active family has nonzero candidates on its probe;
- gradients reach every active head;
- the canonical successor pushforward is finite and normalized.

### Gate 1 — per-family successor micro-overfit

Use a fixed 64--128-example development panel per load-bearing family. Train one family/head first, then a
minimal adapter/body scope only if necessary.

Pass requires, with frozen numerical thresholds:

- finite loss and gradients;
- teacher canonical successor present for every example;
- high teacher-successor recovery/probability;
- learned-versus-uniform successor log-likelihood advantage;
- no collapse of another active family on the shared sentinel panel.

Failure blocks all mixed training. It diagnoses representation/teacher/factorization/optimization before
significant compute is spent.

### Gate 2 — 500-update mixed pilot

Evaluate the exact **current** model, not the trainer's historical best state, at short fixed intervals.
Abort immediately on:

- missing teachers or invalid/unnormalized successor rows;
- non-finite losses or gradients;
- an inherited/load-bearing family crossing its frozen retention bound;
- a new/load-bearing family with no positive successor-level movement;
- practical productive family mass collapsing toward zero on opportunity-matched states;
- candidate-space explosion beyond the frozen resource bound.

The report must contain per-family trajectories, not just aggregate loss.

### Gate 3 — 2,000-update pilot

Resume the exact Gate-2 state only after Gate 2 passes. Require:

- positive validation successor-level improvement for every load-bearing family;
- acceptable retention on all protected capabilities;
- positive learned-versus-uniform advantage;
- productive short rollouts for size, graft, and topology fixtures;
- no material source/scaffold/topology validation divergence;
- calibration reported separately from embedded-chain selection.

### Gate 4 — full schedule

Resume the exact Gate-3 state only after Gate 3 passes. Keep the same live alarms. Recovery snapshots must
store `current_state_dict`; checkpoint forensics and launch decisions must explicitly name whether
`current_state_dict` or `best_state_dict` was evaluated.

No aggregate improvement can waive a failed load-bearing-family gate.

## 8. Immediate implementation order

1. Freeze this audit and its exact source identities.
2. Fix shared checkpoint reconstruction so each recovery snapshot is scored using its own
   `current_state_dict`.
3. Finish the production canonical-successor leaderboard on validation only.
4. Implement a machine-readable capability-gate contract and an abort callback in the trainer.
5. Run the remaining per-family successor micro-overfits, especially graft and bond reorder.
6. Complete the `ring_system_delete` reachability/path-cost audit.
7. Design and census the revised operator-aware corpus **before** any mixed pilot.
8. Compare warm-start-with-retention and from-scratch recipes only through the same short gates.
9. Continue 500 → 2,000 → full only for a recipe that passes every gate.

## 9. Claims this audit does not make

- It does not prove that teacher-rate imbalance is the sole cause of the completed run's failures.
- It does not prove that multi-neighbor insertion is never useful.
- It does not prove that `ring_system_delete` is redundant until the reachability audit is complete.
- It does not turn mark-level Generator Matching into successor-level training.
- It does not select a completed-run checkpoint.
- It does not authorize tuning on the inspected final test split.
- It does not require warm start; from-scratch training remains a valid gated candidate.
