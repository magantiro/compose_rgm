# RingCore-V1 accuracy and architecture audit

**Date:** 2026-07-30
**Scope:** completed source-conditioned editing run and the replacement-run decision
**Status:** diagnostic; no replacement training authorized

This audit separates **measured**, **inferred**, and **pending** claims. It does not select a checkpoint and
does not use the inspected final test set for selection or tuning.

## Executive conclusion

No training protocol can guarantee a high-quality model before it is run. The defensible goal is stronger:
make every known failure mode either impossible by construction or detectable in a cheap, predeclared gate.

The completed run does **not** show that executable Rewrite Generator Matching is fundamentally broken. It
also does not establish that the current architecture is sufficient. The strongest current diagnosis is:

1. the headline “accuracy” was an operator-family choice diagnostic, not full-action or molecular-successor
   accuracy;
2. the training objective and effective sampling law were poorly aligned with the fixed-budget editing
   kernel used by the flagship experiments;
3. the full mixture gave radically unequal effective supervision to the required operator families;
4. the inherited graft behavior and the new ring-opening behavior were unstable under joint optimization;
5. a candidate-specific feature bottleneck remains plausible for graft and long-range edits, but has not
   yet been isolated by the required within-family successor-level micro-overfit tests.

The next model should therefore change the **measurement, corpus compiler, training law, and loss alignment**
before increasing backbone size. Architecture changes are gated by targeted failure evidence.

## 1. Immutable evidence used

Completed run:

```text
compose-v4-ringcore-v1-scientific-a7546e2-v1
```

Frozen artifact identities:

```text
inventory:
df815175f8b75a319cc87c01b923db431e53834e28d8b6c5f232ac34e66ddbc0

manifest.json:
e06fb9946a4f25499546b36d8e9ba73250dff30e4e8163351c326bb269955e8d

metrics.json:
fd5798faa816741b3629d991fa34ec39e35f03dd6992007ea8581299582ef3dd

checkpoint.pt:
a25bb66749447f0799003c265482615b33749bde0349d86495246edd33e1bc6c
```

The checkpoint contains 118 parameter tensors, 6,171,899 parameters, and 24,687,596 parameter bytes. Its
recorded architecture and optimization contract are:

| Field | Value |
|---|---:|
| hidden width | 256 |
| message-passing steps | 6 |
| rate factorization | hierarchical |
| family-mass mode | boolean |
| ring electronic mode | factorized local |
| trainable scope | all parameters |
| optimizer | decoupled AdamW |
| peak learning rate | \(3\times10^{-4}\) |
| warmup | 500 updates |
| schedule horizon | 16,000 updates |
| batch size | 64 |

**Measured conclusion:** this was not an obviously tiny network. A blanket “make the model larger” response
is not supported.

## 2. What the reported accuracy actually measured

`factorized_mark_metrics` computes `family_accuracy_X` as:

> among examples whose teacher operator family is \(X\), the fraction for which the model's highest-probability
> **family** is \(X\).

It does not ask whether the model selected:

- the correct atom;
- the correct bond;
- the correct reroute operands;
- the correct raw rewrite mark;
- or the correct canonical molecular successor.

Likewise, `family_top3_accuracy_X` asks only whether family \(X\) is among the three highest-probability
families. The name is now retained only for backward compatibility; new reports also emit the explicit names
`family_choice_accuracy`, `family_choice_top3_recall`, and their balanced/per-family variants.

The completed run did report mean probability of the selected teacher mark, but did not report its rank,
canonical-successor probability, candidate-count-normalized rank, or successor NLL. Consequently, the
headline 0.194 balanced-family “accuracy” is neither a complete measure of edit correctness nor a sufficient
reason to reject the paradigm.

For a stochastic, property-agnostic editing prior, top-1 classification can also be intrinsically low when
one state has several plausible next edits. The primary measurements must be proper scoring rules over the
teacher molecular-successor distribution, followed by target recovery and productive rollout tests.

## 3. Proven training-law and objective mismatches

The frozen corpus audit measured:

- 58.35% of training landings are terminal and contribute hazard-only supervision;
- editing rollouts discard the hazard and use the embedded molecular jump chain;
- MMP paths receive 69.34% of the expected selected-mark coefficient;
- every packed MMP path is serialized as `atom_delete* -> atom_insert*`;
- `atom_delete` alone receives 51.18% of the selected-mark coefficient;
- graft, ring opening, and ring closing receive 3.45%, 2.71%, and 2.71%, respectively;
- paths longer than six steps contribute 85.62% of the conditional MMP selected-mark coefficient even
  though those paths are delete/rebuild serializations rather than diverse mixed-operator chemistry.

The completed objective trains one selected rewrite mark. The paper's editing and control experiments consume
the canonical molecular-successor pushforward. Except for specific graft grouping, equivalent marks are not
generally aggregated in the training objective.

These are direct, sufficient reasons to redesign the replacement training law:

1. compile direct semantic operations before delete/rebuild fallbacks;
2. train teacher **successor-fiber mass**, not an arbitrary alias, wherever exact aggregation is available;
3. separate or independently weight the hazard and successor-identity components;
4. sample semantic capability cells explicitly and report their effective loss coefficients;
5. prevent long traces and large analogue series from dominating merely because they contain more serialized
   positions.

None of these conclusions requires an architectural hypothesis.

## 4. What the checkpoint trajectory adds

The validation history is not consistent with a simple monotone “the capability never existed” story:

- ring-opening family top-3 recall reached 0.92 at step 4,500, then later fell;
- graft family top-3 recall was 0.821 at initialization, fluctuated throughout training, reached 0.679 at
  step 13,500 on the small validation probe, and was 0.393 at step 16,000;
- the inspected final test graft top-3 value was 0.0 on 50 examples.

**Inferred:** capability competition and retention are real concerns, but “monotone catastrophic forgetting”
is not yet proved. Small family panels, checkpoint variance, and validation/test shift can contribute. The
required test is one frozen graft probe evaluated on base B, initialized RingCore, and every current-state
snapshot.

The existing ring-opening micro-overfit establishes that the family logit, mask, teacher routing, and gradient
path can put essentially all family mass on `cycle_attach`. It does **not** establish that the model can rank
the correct ring edge or canonical successor: the reported micro-overfit table omitted within-family
teacher-mark rank and successor NLL. That narrower test remains required.

## 5. Architecture-specific hypotheses

These are hypotheses to test, not findings.

### 5.1 Family gate

The hierarchical family head receives a mean-pooled global graph state and time. It does not directly receive
normalized summaries of each family's candidate table. This may make family competition harder when the
best available edit, candidate entropy, or candidate geometry matters.

Test:

- baseline global-only family gate;
- gate augmented with per-family normalized candidate summaries such as supported count, mean, maximum, and
  log-mean-exp score;
- same data, loss, initialization, and update budget.

Do not use an unnormalized log-sum-exp alone: it can reward a family merely for having more aliases or legal
coordinates.

### 5.2 Graft scorer

The current graft score uses:

```text
node(moved) + node(target) + global_state
```

It does not directly encode:

- the removed neighbor or cut edge;
- the moved fragment size;
- the moved-target pair representation;
- path length or topological relationship;
- attachment-environment deltas.

This is a concrete candidate-specific bottleneck for reroute ranking, especially on graphs whose relevant
context lies beyond six message-passing hops.

Test the current scorer first on a unique-state, high-candidate graft micro-panel. If it fails to memorize
canonical successors, compare one bounded replacement that additionally uses the moved-target pair, cut-edge
pair, fragment size, and shortest-path/topology features. Do not enlarge the whole backbone first.

### 5.3 Cycle and bond scorers

Cycle-open, cycle-close, and bond-reorder heads score the learned pair representation. That representation
already includes node context, bond state, closure topology, and ring-system topology. Their final heads are
linear, but the shared pair projection is nonlinear.

Therefore no missing-feature claim is currently justified. Required tests:

- within-family edge/successor micro-overfit;
- performance stratified by ring size, fused/spiro/bridged context, candidate count, alias multiplicity, and
  graph diameter;
- a small family-specific MLP only if the current head fails while support and labels pass.

### 5.4 Message-passing range

Six message-passing steps can be insufficient for edits depending on graph context more than six bonds away,
but the current results do not isolate that effect. Stratify error by graph diameter and edit-context radius.
Compare a local-feature augmentation with additional message-passing depth only on the failing stratum.

### 5.5 Source and objective conditioning

The base prior is intentionally property-agnostic. Omitting the immutable source, Pareto preference, remaining
budget, and protected mask from the base generator is not an architectural defect. Those variables belong to
the controller/value model or augmented controlled state. Mixing them into the reference prior would make
the learned transport task-dependent and weaken the same-base controller comparisons.

## 6. Replacement-run decision ladder

### A. Metric gate

Before optimization, compute on every development panel:

- teacher canonical-successor NLL and probability;
- successor rank, MRR, top-\(k\), and candidate-normalized lift over uniform;
- raw mark count, canonical successor count, alias multiplicity, and family support;
- family-choice diagnostics under explicit names;
- productive/self/virtual mass;
- hazard metrics separately.

### B. Support and label gate

For every teacher:

- exact action is legal;
- execution equals the stored exact successor;
- canonical teacher fiber is nonempty;
- production segmented aggregation equals the dictionary oracle;
- undirected endpoint conventions are unique;
- no teacher disappears under packing or persistent-slot replay.

### C. True micro-overfit gate

Use 64–128 examples per load-bearing semantic slice. Separate:

1. unique-state deterministic panels, which must memorize the teacher canonical successor;
2. repeated-state panels, which must fit the empirical multi-successor distribution rather than an impossible
   single label.

Report both family mass and within-family/successor learning. A family-probability-only result is insufficient.

### D. Architecture gate

Only change a head when C fails after support and labels pass. Only change backbone capacity when:

- all relevant heads receive nonzero finite gradients;
- training successor NLL remains high;
- a targeted larger/deeper model improves micro-overfit or a held-context stratum;
- the gain is not reproduced by corrected features, loss, or sampling.

### E. Objective and sampler factorial

On the same small development corpus, compare:

- selected-mark versus canonical-successor loss;
- current versus semantic-cell-balanced sampling;
- joint versus separated hazard optimization;
- scratch initialization versus compatible warm start;
- warm start with and without replay/distillation retention.

This bounded factorial identifies cause before a long run. It is more informative than another single 16,000
step trajectory.

### F. Short mixed pilots

At 50–100 updates, require finite nonzero gradients and early movement for every required family. At 500
updates, stop on any required successor slice that collapses or fails to beat uniform. At 2,000 updates,
require held-analogue transport, size adaptation, topology adaptation, and all load-bearing successor NLL
safeguards before authorizing a full schedule.

## 7. Current decision

Do not train a replacement full model yet. The immediate blockers are:

1. complete the current-snapshot canonical-successor leaderboard;
2. build true within-family successor micro-panels for graft, reorder, ring open, and ring close;
3. implement differentiable teacher-successor aggregation for the replacement loss;
4. freeze the semantic-cell sampler and numeric stop/go thresholds after the development census;
5. run the bounded objective/sampler/initialization factorial;
6. pass the 500- and 2,000-step gates.

If the corrected 256-by-6 model passes micro-overfit and short mixed pilots, retain it. If a specific head
fails, repair that head. If the whole model fails to fit training successor distributions after those repairs,
then—and only then—run a controlled capacity comparison or replace the backbone.
