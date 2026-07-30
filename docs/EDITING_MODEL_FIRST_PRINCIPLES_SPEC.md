# COMPOSE editing model: first-principles capability and data specification

**Status:** working design contract, 2026-07-30
**Scope:** the source-agnostic editing prior and the data used to train it
**Not a result:** every empirical adequacy statement below remains gated by measurement

This document derives the editing model backward from the behavior required by the paper. It does not
assume that the completed RingCore-V1 operator set, corpus mixture, warm start, loss, or schedule must be
preserved. It also does not add capabilities merely because they are chemically interesting.

The governing rule is:

> Include the smallest executable rewrite basis and the smallest curated data mixture that give useful
> probability mass to the molecular successors required by the registered editing and control tasks,
> within the declared edit and oracle budgets.

The completed 16k run remains a valid diagnostic baseline. This specification governs the decision about a
replacement training contract.

## 1. What the base prior must and must not learn

The editing prior is a source-agnostic kernel

\[
P_\theta(y\mid x,t),
\]

induced by executable rewrite marks and canonical molecular identity. It should learn which legal molecular
successors are plausible from the current molecule. It is not a property optimizer and is not a
source-to-target translator.

The controller, not the base prior, receives:

- the immutable source lead;
- a goal, preference vector, or objective-space target region;
- the remaining edit budget;
- protected atoms, scaffold, or pharmacophore mapping;
- running path constraints or accumulated path statistics.

This separation is necessary for same-base controller comparisons and dynamic retargeting. It also prevents
the training corpus from becoming an arbitrary mixture of task-specific reward labels.

The base prior nevertheless needs enough non-negligible support for a controller to:

1. make small, high-similarity improvements;
2. grow and shrink a molecule;
3. alter composition and bond electronics without destructive rebuilding when a local rewrite exists;
4. relocate an attachment or preserve a large fragment while changing connectivity;
5. close and open cycles in both directions;
6. reverse or correct an earlier decision after a goal switch;
7. branch from one prefix toward chemically distinct successors;
8. remain useful when hard support masks remove many otherwise plausible transitions.

Legal support alone is insufficient. A transition with astronomically small base probability is formally
reachable but practically unavailable to finite-particle or approximate value control.

## 2. Capability-to-training-evidence matrix

| Required behavior | Minimal training evidence | Load-bearing support | Development gate |
|---|---|---|---|
| local substituent and bioisostere tuning | real close analogues plus reversible local perturbations | restate, reorder, insert/delete | held-transformation successor NLL and target recovery beat uniform legal rewriting |
| size growth and shrinkage | real directed size-changing analogues in both database-independent directions | insert, delete | both directions succeed on held atom-count intervals without collapse toward deletion |
| linker length and attachment relocation | real two-cut/linker or positional-isomer examples, not only random corruption | insert/delete, reroute, reorder | bounded-path recovery and lower path overhead than delete-and-rebuild |
| electronic and bond-state changes | real pairs whose shortest semantic explanation is restate/reorder | atom restate, bond reorder, possibly coordinated ring restate | successor recovery across aromatic, charged-context, and rare-element strata |
| ring and topology adaptation | real topology-changing analogues where available plus balanced executable synthetic paths | cycle close/open; coordinated restate only when required | close/open, ring-size, and topology-reversal tasks pass within budget |
| longer compositional editing | paths through real analogue-series neighbors plus bounded synthetic compositions | multiple families | useful target recovery without excessive reversals, repeated states, or source-similarity collapse |
| Pareto branching | multiple plausible targets or successors per source/context | diversity across the same legal successor fiber | archive coverage and successor entropy do not collapse to one frequent transformation |
| dynamic correction | bidirectional support and explicit reversal/correction examples | inverse-capable core basis | goal-switch continuation outperforms continuing the stale objective at matched post-switch cost |
| hard path constraints | executor-visible protected mappings and enough alternative routes | controller-side support restriction over the same basis | feasible paths exist after masking; endpoint-only filtering wastes more oracle calls |

Counts of operator names do not satisfy this matrix. The audit unit is a joint semantic cell:

```text
capability regime
  x real/synthetic evidence
  x path scale
  x cardinality/topology delta
  x chemistry and charge stratum
  x source/scaffold/series/transformation split
  -> records, unique contexts, candidate difficulty, effective loss coefficient
```

### 2.1 Pareto exploration requires composition and branching

Efficient frontier mapping cannot be learned from isolated one-step edits alone. Different Pareto regions
will generally require different multi-step combinations of composition, size, connectivity, and topology
changes. The base data must therefore include:

- several plausible analogue targets from the same source or series;
- short and medium paths containing more than one operator family;
- alternative valid paths or causal orderings to the same endpoint;
- correction and partial reversal after an earlier structural decision;
- shared prefixes that can continue toward structurally distinct endpoint regions.

The requirement is diversity of useful compositions, not an unrestricted collection of long traces. Very
long arbitrary paths inflate early-step teacher coefficients, increase repeated or destructive edits, and
teach a serialization more readily than a molecular process.

There are three distinct learning problems:

1. **Local successor fidelity:** the base kernel must rank chemically plausible immediate successors.
2. **Compositional transport:** the base kernel must retain usable mass along multi-operator paths and
   alternative branches.
3. **Goal-conditioned allocation:** the learned value/controller is trained on frozen-base rollouts to
   select continuations toward objective-space target regions and under-covered hypervolume cells.

Property-labelled Pareto trajectories are therefore controller-training data, not a substitute for a
property-agnostic compositional edit prior. The development gate must measure both endpoint recovery across
multi-operator paths and branch coverage from a shared source.

### 2.2 What the complete packed-corpus census changes

The complete semantic census has now been run over all 361,019 effective packed MMP records, not a sample.
For the 330,991 training records:

- 54,692 paths have length one or two, 157,743 have length three through six, and 118,556 are longer than
  six;
- every path is `atom_delete* -> atom_insert*`;
- every source and target variable fragment has exactly one attachment;
- every pair preserves graph cycle rank;
- no variable fragment contains a ring atom;
- 320,400 cyclic pairs preserve the same Murcko scaffold, while the remaining 10,591 are both acyclic.

Under the actual progress-state and teacher-rate law, paths longer than six contribute approximately 85.6%
of the conditional MMP selected-mark coefficient. The length-one-or-two slice contributes approximately
1.6%. Thus the present corpus already contains many long paths, but the dominant long-path signal is a
serialized one-cut delete/rebuild program rather than diverse operator composition.

The other layers do not close this gap:

- general corruption contains mixed paths of length one through five over seven families, but its endpoints
  are synthetic and its topology macro only removes cycle rank;
- the cycle layer is balanced between closing and opening, but every example is one step;
- consequently, the current union contains no real mixed-family analogue path and no path in which a
  primitive cycle change composes with subsequent size, electronic, attachment, or property-repair edits.

This is the reason for a V2 corpus. It is not a claim that long traces are intrinsically useful or that the
current 661,105-record training set is small.

The full local result is
`diagnostics/coherence/ringcore_v1_mmp_semantic_coverage_full_local_2026-07-30.json`
(`content_sha256=8f2aee186c8a3d58d9834115598e444206a338e7ae2b7239fd03ccbff1d16966`). It records
the exact packed-manifest and source-pool hashes plus the Python, NumPy, and RDKit runtime used for the
descriptor portion of the census.

## 3. Provisional minimal operator basis

These are provisional decisions, each with an explicit overturning test.

### Keep as core primitives

- **Atom insertion and deletion.** Required for trans-dimensional adaptation. One-neighbor insertion remains
  the default boundary.
- **Atom restatement.** Required for local element/valence changes without deleting molecular identity.
- **Bond reordering.** Required for local electronic and functional-group changes.
- **Bond rerouting/graft.** Required for attachment relocation, fragment preservation, positional isomers,
  and short compositional linker insertion.
- **Primitive cycle closing and opening.** Required for bidirectional topology control.

### Keep only if a primitive-path audit justifies it

- **Ring-system restatement.** Retain if coordinated electronic changes cannot be expressed through valid
  committed primitive intermediates, or if the primitive path is too long for the registered budget.
  Otherwise it is unnecessary model surface.

### Default to disabling

- **Current `ring_system_delete`.** The active corpus realization behaves mainly as a macro ring opener and
  currently appears redundant with primitive cycle opening. Retain it only if it reaches a meaningful
  valid successor class more efficiently than cycle operations.
- **`ring_system_grow`.** Keep disabled. A finite ring catalog must not define RingCore support.

### Do not add without a measured gap

- **Multi-neighbor atom insertion.** Add only if two-neighbor insertion resolves a material unreachable
  class or unacceptable path overhead for linker subdivision or ring expansion.
- **Fragment-replacement macros.** Medicinal-chemistry practice makes fragment replacement a plausible
  accelerator, but it changes the one-step molecular graph and the meaning of an edit budget. First measure
  operator-aware primitive path length and controlled recovery. If the primitive basis is reachability
  complete but unusably slow, add a data-backed macro as an explicitly ablated accelerator, not as the
  definition of support.
- **New ring templates or topology labels.** Add only after the primitive cycle forensic program rules out
  supervision, factorization, features, and optimization as the cause of failure.

### Explicit V1 scope boundaries

- Stereochemical changes are not represented. Stereo-only pairs must be filtered and the paper must claim a
  2D molecular-graph process, not stereochemical lead optimization.
- Formal-charge changes are outside the current charge-preserving edit contract. Charged contexts should be
  represented and audited, but a charge-changing action should not be added without a protonation and
  standardization contract.
- Validity is not synthetic feasibility. Observed analogue transformations and synthesizability diagnostics
  provide a plausibility bias; the current paper does not claim a reaction-plan generator.

## 4. Required training-data lanes

### Lane A — observed local analogue transitions

Use single-cut MMP and matched-series neighbors for high-similarity medicinal-chemistry moves. Preserve:

- the constant-core mapping;
- attachment context;
- variable-fragment sizes;
- transformation signature and frequency;
- document, target, assay, or series provenance when available;
- both endpoint directions to remove arbitrary database-order bias.

Do not let a handful of common transformations or large series dominate by raw pair count.

### Lane B — operator-aware compilation of real endpoints

The current one-cut compiler always emits deletion followed by insertion. The replacement compiler must first
test whether the real endpoint change has a shorter semantic path using:

1. atom restatement;
2. bond reordering;
3. rerouting/graft;
4. insertion/deletion;
5. cycle close/open;
6. coordinated ring restatement only when required.

Delete-and-rebuild is a fallback, not the universal explanation. This lane is required to put real-analogue
supervision on graft, reorder, restate, and topology families rather than teaching those families almost
entirely through synthetic corruption.

### Lane C — linker, positional, and topology analogues

Mine and compile separately:

- two-cut linker lengthening, shortening, and replacement;
- attachment-point relocation and positional isomers;
- ring opening/closure and ring-size changes;
- saturated/aromatic ring-state changes;
- fused, spiro, or bridged changes only where the production executor actually supports them.

The corpus must report failures rather than silently dropping difficult subclasses. Sparse real coverage may
be supplemented with executable synthetic pairs, but real and synthetic metrics remain separate.

### Lane D — longer paths through analogue series

Construct sparse graphs within document-, target-, or series-consistent compound groups. Prefer paths whose
intermediate endpoints are observed compounds. Sample series/groups before raw paths so large cliques do not
dominate.

Long paths exist to teach composition, not to manufacture distant arbitrary molecule pairs. Reject paths
with gratuitous source-similarity loss, repeated states, immediate reversals, or a shorter valid compilation.

### Lane E — reversible synthetic legal walks

Start from real molecules and apply short legal sequences under the exact production executor. Include both
directions when representable. Use this lane to:

- cover rare but load-bearing operator/context cells;
- expose correction and reversal;
- supply controlled path lengths;
- test support beyond transformations already observed in ChEMBL-derived data.

Synthetic walks are support coverage, not evidence that the learned process reproduces real medicinal
chemistry.

### Not a data lane — legal smoothing

A small

\[
P_{\mathrm{ref}}=(1-\epsilon)P_\theta+\epsilon P_{\mathrm{legal}}
\]

component may preserve reachability under an imperfect learned prior. It is a declared kernel wrapper and
must be ablated. It must not be described as additional training examples, and uniformity is over canonical
molecular successors rather than raw marks.

## 5. Path compiler contract

For every endpoint pair:

1. preserve exact slot-addressed states and a verified core/attachment mapping;
2. search for a shortest or near-shortest valid semantic path under the production operators;
3. penalize protected-core changes, excessive source departure, destructive rebuilding, and redundant
   reversals;
4. replay every transition through the production executor;
5. verify every committed state is supported, connected, and sanitizable;
6. verify exact canonical endpoint identity;
7. verify the inverse path where the declared support contains it;
8. retain causal dependencies or multiple equivalent valid paths rather than canonizing an arbitrary
   serialization order;
9. report candidate-set size, canonical successor count, aliases, and path failure reason;
10. reject unsupported pairs explicitly.

The registered edit budget must be checked against the compiled path distribution before training. A
capability that usually needs more steps than the evaluation horizon is not meaningfully present.

## 6. Training law and objective

### Hierarchical sampling

The sampler should select approximately in this order:

```text
data lane
  -> series/scaffold/source group
  -> semantic capability stratum
  -> endpoint pair or path
  -> progress state
```

Exact weights are not chosen in this document. They must be derived from the completed semantic census and
frozen before the corresponding model comparison. The desired law combines:

- an empirical-plausibility bulk component;
- explicit minimum exposure for load-bearing rare capability cells;
- no raw-pair domination by large series, long traces, or common transformations.

The artifact must report the **effective loss coefficient**, not only record counts. Path position, teacher
rate, importance weights, and layer sampling can substantially change what the optimizer sees.

### Canonical-successor Generator Matching

The completed run used selected-mark loss and must continue to be described that way. For a replacement run,
the preferred target is the aggregate rate of the teacher molecular successor:

\[
\lambda_\theta(y^\star\mid x,t)
=
\sum_{a:T(x,a)\cong y^\star}\lambda_\theta(a\mid x,t).
\]

This aligns training with checkpoint selection and successor-level control, and removes penalties for
arbitrary within-fiber aliases. It does not remove executable marks: marks remain the parameterization and
executor interface.

The repository already contains the successor-rate Bregman objective and a tested segmented aggregation
primitive. A scalable implementation should precompute, for each fixed training state and teacher
successor, the coordinates of all teacher-fiber aliases. It need not materialize every successor during
every gradient step.

The first bounded implementation of that bridge now lives in
`src/compose_v4/experiments/factorized_successor_training.py`. It compiles teacher fibers with the production
enumerator, executor, and canonical identity, then performs differentiable `logsumexp` over the stored
factorized coordinates. Its tests verify equality to the probability obtained by summing the production
marked law and verify gradient flow. It is **not yet wired into the full packed trainer**; packed fiber
serialization, throughput, and full-corpus support audits remain prelaunch gates.

If exact teacher-fiber aggregation is too expensive, that is a measured implementation constraint. It is not
permission to claim a successor-trained objective.

### Hazard separation

Editing experiments use the fixed-step embedded molecular jump chain, while de novo generation uses timed
CTMC simulation. Hazard calibration should therefore be monitored and optimized separately from successor
identity. A poorly scaled hazard term must not be allowed to select an inferior editing kernel.

### Path-order invariance audit

The current delete-then-insert MMP serialization and remaining-step teacher rate strongly favor early
deletions. Any replacement scheduler or importance law must demonstrate:

- the intended target expectation mathematically;
- its effective per-family and per-position coefficients on the full corpus;
- robustness to alternative valid serializations of causally independent events.

## 7. Splits and provenance

At minimum, freeze disjoint development and final partitions along:

- exact molecule/source identity;
- scaffold identity;
- analogue series or publication/document;
- transformation signature;
- topology and cardinality strata.

No pair may bridge partitions. Large series must stay intact. Since the completed final test aggregate has
already been inspected, a replacement model requires a new sealed final holdout or external evaluation set.

Every packed derivative must retain or bind to:

- the authoritative source rows;
- standardized endpoint identities;
- exact states and actions;
- semantic descriptors and core mapping;
- compiler/operator/canonicalizer hashes;
- split, sampler, and objective contracts.

## 8. No-waste gates

### Gate D0 — corpus before training

Block training unless:

- every load-bearing capability cell has measured real and/or synthetic coverage;
- effective coefficients, not merely counts, pass a predeclared floor;
- all teachers are legal and their successor fibers are recoverable;
- split leakage is zero under every declared split key;
- path lengths fit the registered budgets;
- deletion-first, source, series, and transformation concentration are reported;
- unsupported and failed pairs are retained as audit records.

### Gate O0 — operator reachability

On frozen development panels, compare shortest valid paths under:

- the provisional minimal basis;
- no graft;
- no cycle operations;
- optional two-neighbor insertion;
- optional ring-system restatement;
- any proposed macro.

Add an operator only for a material reachability or path-efficiency gain.

### Gate T1 — family and semantic micro-overfit

Before a mixed run, each load-bearing semantic slice must memorize 64–128 examples at the canonical-successor
level. Include high-candidate and aliased states. Failure means data, support, factorization, or features are
defective; it blocks training.

The completed ring-opening micro-overfit established that its family route can learn. Because it reported
family mass but not within-family edge rank or canonical-successor NLL, it does not satisfy this stricter
gate.

### Gate T2 — 500-step pilot

Stop automatically for:

- non-finite loss or gradients;
- any missing/zero-gradient family;
- successor performance at or below uniform on a required slice;
- rapid collapse of a previously learnable family;
- severe divergence between mark and successor metrics;
- obvious deletion or size drift in short productive rollouts.

### Gate T3 — 2,000-step pilot

Proceed to a full schedule only if:

- production-weighted and balanced semantic successor NLL improve;
- real-analogue transport beats uniform legal rewriting;
- insertion/deletion, graft, reorder, and cycle open/close all pass their development safeguards;
- cardinality and topology development tasks succeed within budget;
- no required capability depends only on legal smoothing;
- calibration and productive rollouts are stable.

No fixed long schedule is authorized by corpus size alone.

The machine-readable stop/go contract is `configs/editing_training_v2_gate.json`. It deliberately contains
unfrozen numeric thresholds and `full_training_authorized=false`, so its validator blocks a replacement full
run until the development panels are measured and the thresholds are frozen.

## 9. What established practice contributes—and what it does not

The design is consistent with several primary sources:

- [mmpdb](https://doi.org/10.1021/acs.jcim.8b00173) treats matched transformations as reusable medicinal
  chemistry knowledge for hit-to-lead optimization.
- [CReM](https://doi.org/10.1186/s13321-020-00431-w) uses context-matched fragment mutation, growth, and
  linking; explicitly distinguishes small local lead-optimization steps from larger exploratory steps and
  supports protected atoms.
- [The Fragment Network](https://doi.org/10.1021/acs.jmedchem.7b00809) organizes observed ChEMBL paths into
  substituent, linker, and ring replacements rather than treating all structural differences alike.
- [PMO](https://arxiv.org/abs/2206.12411) shows that oracle-call efficiency and assembly strategy materially
  affect optimization, and that atom-by-atom approaches can be inefficient.

These sources support using contextual analogue transformations, series structure, path scale, and
fragment/linker/ring strata. They do **not** determine COMPOSE's mixture weights, prove that a fragment macro
is necessary, or establish that the current corpus is inadequate. Those remain empirical questions answered
by the audits and gates above.

## 10. Comparator strategy

No external method matches all of COMPOSE's state space, executable intermediates, learned transport,
canonical quotient, and finite-horizon control. The comparison must therefore isolate claims rather than
pretend that one method is a complete architectural control.

| Question | Required comparator | What is held fixed |
|---|---|---|
| did the model learn state-dependent transport? | uniform canonical-successor law and a state-independent empirical-family law | executor, legal support, source, budget |
| are birth/death, graft, and cycle operations load-bearing? | fixed-cardinality, no-reroute, and no-cycle support ablations | task, controller class, accounting |
| does finite-horizon control add anything? | unguided, rerank, greedy, local Boltzmann, SMC, learned Doob, and exact Doob on the finite slice | one canonical successor kernel |
| does the learned controller map a Pareto front efficiently? | same-base scalarization, MOG-DFM-style guidance, SMC/Doob, and NSGA-II over COMPOSE successors | proposal support, objectives, oracle budget |
| is the complete system competitive for lead optimization? | current runnable small-molecule methods under a compatible benchmark | sources, oracle/constraints, budget, selection and evaluator |

The compact external run set is:

- [InVirtuoGen](https://openreview.net/forum?id=Qdu92a5DiM), a current fragment-based discrete-flow
  lead-optimization and PMO method;
- [GenMol](https://proceedings.mlr.press/v267/lee25o.html), a fragment-remasking discrete-diffusion
  lead-optimization method;
- [GraphGA](https://doi.org/10.1039/C8SC05372C), a simple strong graph-search baseline;
- [MARS](https://arxiv.org/abs/2103.10432), an iterative fragment-graph editing and multiobjective MCMC
  method;
- [RetMol](https://arxiv.org/abs/2208.11126) where its exact lead task matches;
- [HN-GFN](https://openreview.net/forum?id=uoG1fLIK2s) on a compatible sample-efficient multiobjective task.

MOG-DFM, pCoMole, and AReUReDi are important algorithmic relatives, but their published tasks and
representations are not identical small-molecule graph-editing comparisons. Their control ideas should be
adapted to the same COMPOSE successor kernel where appropriate, and their published results should remain
contextual rather than being presented as matched head-to-head numbers.

Every external adapter must freeze code/checkpoint identity, standardization, source set, objective and
constraint implementations, oracle accounting, seeds, selection, and failure handling. Published numbers
are directly comparable only when all those fields match. The machine-readable plan is
`configs/comparator_registry_v1.json`.

## 11. Immediate decision sequence

1. Treat the full semantic census as complete and freeze its artifact identity.
2. Build a frozen endpoint panel spanning every required capability and path scale.
3. Run the operator reachability/path-cost audit.
4. Recompile a development subset with the operator-aware compiler.
5. Finish packed teacher-fiber serialization and throughput validation for the now-tested differentiable
   successor aggregation bridge.
6. Populate and freeze the unresolved numeric gates in `configs/editing_corpus_v2_contract.json` and
   `configs/editing_training_v2_gate.json`.
7. Compare current and redesigned training laws on true successor micro-overfit and short pilots.
8. Freeze a replacement full-run contract only after D0, O0, T1, T2, and T3 pass.

The measured architecture diagnosis and escalation rules are in
`docs/RINGCORE_V1_ACCURACY_AND_ARCHITECTURE_AUDIT_2026-07-30.md`.
