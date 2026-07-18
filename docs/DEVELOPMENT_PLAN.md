# COMPOSE v4 base-model development plan

## Locked thesis

COMPOSE learns a time-dependent rate measure over executable,
constraint-preserving molecular graph rewrites. Stochastic graph rewriting
provides the process semantics; Generator Matching turns endpoint-conditioned
rewrite processes into a target-free de novo and conditional generator.

The novelty is their trainable synthesis, not the invention of graph rewriting
or Markov generators in isolation.

## Non-negotiable invariants

1. Every committed visible state is a complete valid molecule or the formal
   null source state.
2. Training and inference call the same validators and executors.
3. The production sampler receives no target molecule, target region, or oracle
   alignment.
4. Sampling is ancestral from normalized rates; no beam search is part of the
   generative definition.
5. The production model may parameterize a labelled marked-rewrite lift, but
   its pushforward chemical generator aggregates all aliases that reach the
   same canonical molecule. The exhaustive quotient remains the exact oracle.
6. Atom ordering, equivalent alignments, and resonance encodings cannot define
   different chemical outcomes.

## Model object

For hard condition `c`, let `M_c` be the admissible molecular state space and
`A_c(G)` the valid marked rewrite instances at `G`:

```text
(L_theta,t,c f)(G)
  = integral_[a in A_c(G)] [f(T_a(G)) - f(G)] q_theta,t(da | G, c).
```

The learned rate measure is factorized as:

```text
q_theta,t(da | G, c)
  = Lambda_theta(G,t,c)
    pi_theta(operator | G,t,c)
    p_theta(operands | operator,G,t,c).
```

Hard conditions restrict `A_c(G)` through executor hooks. Soft conditions enter
the neural rate model.

## Phase 0 — executable rewrite semantics

**Purpose:** establish the mathematical state graph before learning rates.

- Minimal micro basis: atom insert/delete, atom restate, bond insert/delete,
  bond reorder.
- Formal null source and padded variable-size state.
- Exact valence plus RDKit transaction boundary.
- Typed rule registry and hard-condition predicates.
- Canonical successor keys and alias-rate aggregation.
- Construction tests: null -> chain, ring closure, inverse edits, invalid-action
  rejection, scaffold-preservation hook.

**Exit gate:** at least 10,000 randomized valid micro rewrites with zero invalid
commits; exact round trips for every reversible test fixture; canonical alias
aggregation invariant under endpoint ordering and slot permutation.

**Status:** passed on the Phase 0 fixture distribution: 10,020 committed
rewrites, 580 exact inverses, zero invalid intermediates, and 580 canonical
slot-permutation matches.

## Phase 1 — legal trace compiler

**Purpose:** obtain tractable endpoint-conditioned teacher processes without
making the trace visible to the learned sampler.

1. Parse and canonicalize target molecules in a Kekule/resonance-consistent
   representation.
2. Sample a random spanning tree and atom construction order.
3. Emit atom insertions for the tree, non-tree bond insertions for closures, and
   restates/reorders for final labels.
4. Compile the inverse deletion trace as well.
5. Randomize across equivalent trees, slot assignments, and commuting edits.
6. Support arbitrary source-target pairs through delete-to-reference followed
   by construct-to-target; optimize paths later only if necessary.

**Exit gate:** 100% exact source-target reconstruction on at least 10,000 held-
out molecules, zero invalid intermediate states, broad fused/bridged/spiro ring
coverage, and trace diversity measurements showing no fixed ordering collapse.

**Status:** randomized spanning-tree/chord orders and the exact arbitrary
source-target baseline bridge are implemented. A 1,000-molecule GuacaMol pilot
gave 985/995 exact eligible round trips with zero invalid commits. The ten
failures are hypervalent-sulfur path-language failures; see
`TRACE_COMPILER_PILOT.md`. The full exit gate is not yet passed.

## Phase 2 — conditional rewrite process and GM loss

**Purpose:** turn a compiled trace into a continuous-time conditional generator.

- Introduce a latent progress process over compiled instructions.
- Use a monotone time scheduler initially; permit forward/reverse correction
  events only after the base objective is verified.
- Project the latent trace to molecular states using auxiliary/latent-process
  Generator Matching theory.
- Implement the rate Bregman/point-process loss as total predicted hazard minus
  weighted log rate on teacher marked events.
- Aggregate aliases before evaluating successor-level diagnostics.

**Exit gate:** closed-form toy systems match exact CTMC marginals; empirical
rollouts recover target endpoint distributions on small graph corpora; loss and
hazard calibration pass numerical tests.

**Status:** the closed-form binomial-progress CTMC satisfies the numerical
Kolmogorov forward equation and empirical marginal test. The complete-successor
Poisson-KL rate objective passes optimum and hazard-penalty tests. On the
enumerable 12-molecule gate, exact marginal generators yield target-free learned
rollouts with total variation 0.0523 at 1,000 samples. Conditional estimation
now passes a first 96/16/16 held-out C/N/O/F corpus gate; the benchmark-scale
gate remains open.

## Phase 3 — neural marked-rate model

**Purpose:** learn target-free marginal rates.

- Compact permutation-equivariant graph transformer.
- Global total-hazard and operator-family heads.
- Atom, pair, and slot operand heads with legality masks.
- Joint atom-state head for element/charge/H state.
- Time and optional condition embeddings.
- Normalize within each legal action fiber so total intensity is tractable.
- Use dense batched action tables and analytic masks in the production loop;
  construct complete successor fibers only for oracle tests and diagnostics.

Start with micro rules only. Preserve the model/executor boundary: the network
scores actions; it never implements chemistry.

**Exit gate:** overfit a tiny corpus, then achieve calibrated held-out action
NLL, zero starvation on ordinary states, zero invalid commits, and target-free
ancestral generation that beats a corpus-marginal rewrite policy.

**Status:** partial pass. The whole-graph rate network fits the tiny exact
marginal generator to mean rate-vector L1 error 0.0146 and 100% top-successor
accuracy. Its target-free sampler has 100% validity/connectedness and recovers
all tiny reference modes. The factorized model preserves the exhaustive
chemical successor set while reducing measured fiber evaluation time by
21-30x. On a 96/16/16 held-out C/N/O/F split, late-time-calibrated conditional
training reaches test GM loss 4.05 and produces 100/100 valid, connected,
non-null target-free rollouts with mean size 10.05 versus 10.50 and mean cycle
rank 1.39 versus 1.19. That pilot's scale and missing matched baseline were
addressed by the subsequent gate.

The subsequent full-scan C/N/O/F gate contains 1,333 eligible molecules. A
matched corpus-marginal rewrite policy and a state-dependent prior-tilted
generator now share the same compiler, legal fiber, CTMC clock, and sampler.
The tilt keeps 100% validity/connectivity/non-null sampling and improves atom-
count TV (0.279 to 0.239), bond-order TV (0.088 to 0.021), ring-size TV (0.274
to 0.252), and SA (5.401 to 5.173). It does not yet improve FCD (20.38 to
21.07) or QED, so the exit gate is a partial rather than decisive pass. See
`SCALE16_GATE.md`.

A topology-aware, trust-regularized follow-up now provides a decisive pass on
the random split. It compiles valid topology before bond-order refinement and
uses bond reorders for 20.8% of sampled events. Against its matched prior it
improves FCD from 22.24 to 19.42, ring-size TV from 0.284 to 0.054, cycle-rank
TV from 0.254 to 0.094, QED from 0.511 to 0.539, and SA from 5.727 to 4.771,
with 100% valid/connected/non-null/unique samples. Mean aromatic rings improve
fivefold but remain 0.130 versus 1.105 in the test set; aromatic ring-system
rewriting is the next measured operator bottleneck.

The production backend now implements the intended hierarchical marked CTMC
directly. On H100, its sustained 32-graph update time is 0.020-0.032 seconds
versus 9-15 seconds for exhaustive successor construction, a roughly 300-500x
speedup. Direct ancestral sampling instantiates only the fired action and keeps
the same executor boundary. The end-to-end preflight produced 16/16 valid,
connected molecules; benchmark-scale quality and FCD are being measured in the
current tree-source run.

## Phase 4 — de novo base model

Compare reference distributions rather than building one into the theory:

- formal null/single-atom source;
- simple valid acyclic/tree reference;
- small-fragment reference mixture.

Primary metrics: strict validity, connectedness, uniqueness, novelty, FCD,
NSPDK/graph statistics, size calibration, ring-system distribution, and
fused/bridged/spiro coverage. Report quality versus number of CTMC events and
wall-clock cost.

**Go/no-go gate:** meaningful held-out distribution learning with 100% committed
state validity, no collapse to acyclic molecules, and competitive quality at a
tractable event horizon.

## Phase 5 — conditional generation

Use the same generator and change the source/condition:

- Scaffold completion: immutable scaffold constraint plus open attachment ports.
- Fragment growing: fragment-preserving action fiber.
- Property conditioning: condition embedding and classifier-free rate guidance.
- Molecular optimization: input molecule as source plus optional edit budget.

For future pocket-conditioned 3D generation, superpose the topology jump
generator with an SE(3)-equivariant coordinate flow; do not burden the first 2D
base-model gate with that extension.

## Phase 6 — architecture-inspired multiscale operators

Add only after a measured micro-model bottleneck:

- RISC layer: universal micro basis.
- CISC layer: verified path/ear insertion, stitch deletion, ring or region
  transaction.
- Compiler lowering: every macro records or verifies its micro expansion.
- Runtime retirement: macro commits atomically after complete validation.

First ablation: micro only versus micro plus path/ear macros. Port the v3
transactional ring-patch executor only if path length, not model quality, is the
demonstrated constraint.

**Status:** the first semantics gate is implemented. Parameterized cycle and
ear tracelets, coordinated ring-system restates, exact inverses, micro
lowerings, a block/ear compiler, and a finite C/N/O/F tracelet fiber pass the
ring fixture panel. The 1,067-molecule training split compiles with 100%
coverage into 11,952 visible events versus 21,909 verified micro lowerings
(1.83x compression). The model/evaluation ablation is not yet run.

The subsequent topology-commitment refinement removes scalar ring closure from
the typed inference fiber. Root rings enter through typed cycle transactions;
fused/bridged systems use ears whose anchors already lie in one cyclic block;
spiro systems use a same-anchor cycle; and separate ring blocks use a new
closed `cycle_attach` transaction. Thus no atom is edited as an acyclic chain
atom and later reinterpreted as a ring atom. Typed cyclic-block interiors are
also protected from incoherent scalar mutation. The revised compiler emits
5,974 acyclic insertions, 977 root cycles, 303 closed-cycle attachments, and
460 ears on the 1,067-molecule training split, and the non-parallel test suite
passes 122 tests. A two-step smoke verifies end-to-end sampling but is not a
quality result.

## Required ablations

- Rewrite GM versus the old destruction-noising model under matched backbone.
- Micro only versus micro plus macros.
- Null versus tree versus fragment reference.
- Single canonical trace versus randomized trace distribution.
- Mark-level versus complete-successor rate training/diagnostics.
- Validity-closed action fiber versus post-hoc rejection.
- Unconditional versus condition-restricted action spaces.

## Immediate implementation order

1. **Done:** replace exhaustive fibers with factorized operator/operand heads
   and exact hierarchical legality masks.
2. **Done:** scale conditional Generator Matching to the 1,333-molecule
   supported C/N/O/F corpus and add random plus zero-overlap scaffold splits.
3. **Done:** add a matched corpus-marginal rewrite policy and a state-dependent
   neural tilt.
4. **Done on the random split:** stabilize the tilt with a prior trust region,
   explicit closure context, and topology-first/bond-order-second teachers. The
   learned generator now beats the matched prior on overall FCD and quality.
5. **Done at the rewrite/fiber level:** implement verified cycle/ear and
   coordinated ring-system tracelets with exact micro lowerings and finite
   teacher support.
6. **Done as a diagnostic:** integrate typed tracelets, superposed family
   hazards, topology commitment, and target-free ancestral sampling.
7. Replace the finite typed catalog with a factorized mark decoder over event
   family, topology/anchors, span, and joint atom/bond payload. The catalog's
   top-128 coverage is 100%/88.0%/92.5% on train/validation/test and is evidence
   for the operator design, not an acceptable final support boundary.
8. Run the topology-committed model versus the earlier generic-carbon and
   scalar-closure models under matched training/sample budgets; include FCD
   only after the structural/ring gates pass.
9. **Reachability done; quality comparison next:** run the checkpoint-distinct
   null-versus-tree source ablation. Primitive tree transport now uses a
   retained-carbon root tracelet, has zero scalar closures, and reaches every
   chemistry-supported target in the 512-attempt audit.
10. Expand the chemistry/path language, beginning with the measured hypervalent-
   sulfur limitation.

The first macro is now justified by a measured correlated-event bottleneck, but
it must be a verified parameterized ring-system transaction rather than a
fragment vocabulary.
