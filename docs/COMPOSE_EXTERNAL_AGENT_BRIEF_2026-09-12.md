# COMPOSE controller: comprehensive external-agent brief

> Historical architecture brief. This document preserves the 2026-09-12 state
> used for external review. For current operational status, the live frozen T4
> run, subsequent coordinated-program results, and collaborator commands, read
> `docs/START_HERE_ICLR.md` first. Historical “current” and “next” statements
> below are not launch authority.

Snapshot: 2026-09-12. Integration branch: `compose-iclr`. Committed launch revision:
`66a5f6eae3eab6361acecb0f2626cf1ae6264f30`. Session label: `compose_iclr`.

Purpose: give another research agent enough context to challenge the current
design and recommend a concrete, efficient intervention. This is a synthesis of
existing code, result artifacts, the supplied manuscript, and development records.
It reports no new experiment. Numbers below retain their original evidence roles.

## 1. The objective and the honest bottom line

We want a genuinely competitive, reusable controller for COMPOSE, initially on
similarity-constrained T4 lead optimization and PMO molecular optimization. The
ultimate goal is reproducible improvement over our strongest actual controller,
then competitive results against verified external baselines under matched
protocols. InVirtuoGen (IVG) is an important comparator, not a guarantee about what
COMPOSE must be able to achieve.

The intended contribution is not a new name for beam search, options, the Doob
transform, or sequential Monte Carlo. It is effective control of a learned,
executable molecular transition process across spatial and temporal scales.

The main distinction to preserve is:

1. A molecule lies within the represented state space.
2. An exact valid primitive route to it exists.
3. The deployed proposal enumerator exposes the necessary actions.
4. The controller gives useful routes appreciable probability.
5. Search preserves useful partial routes and completes them.
6. Oracle allocation recognizes useful completed candidates.
7. The entire optimizer wins under a matched budget and protocol.

We have substantial evidence for the first two, some repaired failures at the
third, and weak results at the fourth through sixth. We have not demonstrated the
seventh against IVG with the new broad controller.

There is real completed work: constructive ring execution, exact winner routes,
broader options, caching and parallel search, candidate provenance, an integrated
semi-Markov particle runtime, and useful negative diagnoses. These are not the
same as benchmark success. The next agent should not spend another iteration
merely proving that we can add a ring.

## 2. What COMPOSE actually generates

### State and support

COMPOSE generates a sequence of complete supported molecular graphs. The current
controller lane uses connected molecules with 1 through 40 heavy atoms, the
declared broad-organic element/valence vocabulary, charge-preserving semantics,
and the production executor and canonicalization. Exact persistent-slot states
preserve atom identity across edits. Padded tensor capacity is a representation
detail, not molecular dimension. A distinguished null source is not a molecule.

Formal-charge design, stereochemistry, isotopes, and radicals are not general
output dimensions of this lane. A chemically valid executor state is not a proof
of synthetic accessibility, stability, drug-likeness, or experimental activity.
Likewise, this is a discrete graph of molecular states, not physical molecular
kinetics or a demonstrated smooth geometric manifold.

The semantic state space is a disjoint union over active atom cardinality:

\[
\mathcal X=\bigsqcup_{n=1}^{40}\mathcal X_n.
\]

The executor admits a marked rewrite only if its successor satisfies the
declared attribute, valence, connectivity, and size conditions. Every committed
non-null state is therefore a complete supported molecule.

Canonical SMILES is useful for molecular identity and reporting. It is not a
replacement for exact persistent-slot replay state. For structural comparison,
parse SMILES into graphs and examine correspondences and graph changes. Raw
SMILES edit distance is not a reliable chemistry distance: atom ordering,
branches, and ring digits are serialization choices.

### Primitive operations

| Primitive | Meaning |
| --- | --- |
| Atom insertion | Add one atom with one existing neighbor, or a root at the null source |
| Atom deletion | Remove an atom while retaining a supported connected remainder |
| Semantic atom restating | Change a supported atom's element/valence/hydrogen state |
| Bond reordering | Change a bond order with corresponding valid hydrogen updates |
| Bond rerouting | Relocate a pendant fragment in one connectivity-preserving transaction |
| Cycle closing | Insert an absent bond between existing atoms, increasing graph cycle rank |
| Cycle opening | Delete a non-bridge cycle edge, decreasing graph cycle rank |
| Ring-system restating | Coordinated supported bond/electronic changes on a cyclic system |

Naming trap: model family `cycle_insert` corresponds to ring closing, historically
executor `bond_insert`. Model family `cycle_attach` corresponds to ring opening,
historically `bond_delete`. `cycle_attach` does not mean attaching a new ring.

Whole-ring construction is a program of primitives, not a new whole-ring birth
operator. Current birth does not silently add an atom with multiple existing
neighbors. A bond-subdivision-like transformation must compile into supported
steps or be declared a support expansion.

There is no universal expressibility claim. A spanning-tree argument establishes
untyped graph topology construction, not a valid, chemically labeled molecular
path within this vocabulary, size limit, and finite budget. Nor is every deletion
invertible by a single permitted insertion.

## 3. The frozen learned reference law

At state \(x\), the executor defines legal marks \(\mathcal A(x)\). A mark includes
its family, operands, and typed payload. The model factorizes their probability:

\[
p_\theta(a\mid x,t)
=p_\theta(k\mid x,t)\,
  p_\theta(\text{operands,payload}\mid k,x,t).
\]

Deployment fixes the registered progress convention, historically \(t=0.5\).
The reference model receives no downstream objective or remaining task budget.
It is frozen while the controller changes.

Different marks can produce the same canonical molecular successor. Define

\[
\mathcal G_y(x)=\{a\in\mathcal A(x):T(x,a)\simeq y\}.
\]

For a committed non-self successor, the molecular kernel is

\[
R_\theta(y\mid x)=
\frac{\sum_{a\in\mathcal G_y(x)}p_\theta(a\mid x)}
{\sum_{y'\ne x}\sum_{a\in\mathcal G_{y'}(x)}p_\theta(a\mid x)}.
\]

Control defined on canonical successors is invariant to re-encodings that
preserve each successor's aggregate mass. Canonicalization does not make
arbitrary mark duplication harmless. Nonlinear mark-level truncation or power
transforms do not inherit this invariance automatically.

Use the production evaluator, not experiment-specific probability reconstructions:
`src/compose_v4/experiments/production_successor_kernel.py`.

The supplied paper describes a GuacaMol-derived, goal-independent transition law
trained on executable teachers. It reports canonical-successor NLL 3.90 versus
5.37 for empirical family frequencies on 3,545 held-out transitions. Its
decomposition attributes 86.7% of the improvement to state-dependent successor
identity. Those are paper-reported results, not new evaluations in this brief.

Important limitations: ring/cycle supervision was largely synthetic perturbation
data rather than observed medicinal ring transformations. Learning conditional
successor probabilities does not calibrate continuous-time holding rates.
Historical mark-level Generator Matching and the successor-trained editing lane
are distinct objectives and must not be merged in provenance or claims.

## 4. The local-to-global hierarchy

The intended controller factorization is

\[
Q(M\mid x,z)\quad\longrightarrow\quad
Q(o\mid x,M,z)\quad\longrightarrow\quad
q(\omega\mid x,M,o,z).
\]

These are WHERE, WHAT, and HOW. An option can additionally have parameters
\(\xi\), such as attachment sites, ring size, heteroatom composition, or an
electronic-state objective. These choices must be accounted for somewhere in the
joint proposal, not treated as free decisions after sampling an option.

### WHERE: region and intended scale

\(M\) is a mutable connected molecular region with preserved context and an
interface. The region selector includes executability/cost considerations and a
scale-balanced exploration floor. It can be tilted by a qualified task-specific
value. The existing T4 comparison did not have such a qualified WHERE value:
using a QED-trained region value as docking guidance was explicitly avoided.

Spatial scope and time horizon are different variables. A large mutable region
does not force a large molecular change. A small region can grow several new
atoms, while a large region can emit a shallow decorative edit.

The current hierarchy also has concrete region-enumeration limits; it is not an
unrestricted rewrite of all 40 atoms on every draw. Report actual supported
regions and horizons for each run.

Track separately:

- intended released fraction of the parent;
- actual changed fraction and insertion/deletion counts;
- coherent structural change, including the largest connected changed component;
- graph cycle rank, ring-system count, and ring perception statistics;
- parent, region, option, primitive depth, and exact/canonical identity.

Graph cycle rank is \(|E|-|V|+c\), not SSSR ring count or ring-system count.

### WHAT: existing macros and compositional options

The original registry contains 14 option families, plus the mandatory generic
channel. An older exported tuple listed only ten, which contributed to confusion.

| Family | Main purpose |
| --- | --- |
| `generic` | Broad ordinary executable rewriting |
| `local` | Atom restating and bond reordering |
| `grow` | Atom insertion |
| `append` | Growth with new-material construction semantics |
| `scaffold_extend` | Backbone-capable growth rather than terminal decoration |
| `decorate` | Terminal substituent chemistry |
| `cyclize` | Supported cycle-closing/bond-insertion moves |
| `append_system` | New ring system from grown material |
| `annulate` | Fuse new material into an existing cyclic system |
| `small_ring` | Small-ring construction channel |
| `aromatize` | Supported aromatic/electronic restructuring |
| `restate` | Atom/ring electronic or identity changes |
| `open` | Ring opening and rerouting |
| `rebuild` | Opening, growth, and rerouting |
| `shrink` | Deletion and opening |

Existing support filters and contracts live in `MACRO_FAMILIES` and
`macro_action_distribution` in `macro_engine.py`. The initial region path
bypassed these channels. Their absence is a historical defect, not an accurate
description of every current controller path.

Newer complete options include parameterized pendant/fused ring construction,
ring expansion, carbonyl addition and core carbonyl insertion. Ring specifications
include topology, five/six-member size, C/N/O composition, electronic character,
and optional refinement. They generate and execute primitive programs rather
than retrieving a stored endpoint molecule.

For simple construction, a pendant L-ring needs L births and a closure; a fused
L-ring across an existing edge needs L-2 births and a closure. Electronic
refinement can add work. A core carbonyl insertion program uses opening, carbon
birth, oxygen birth, and closure rather than illegal multi-neighbor birth.

These ring channels are bounded subfamilies, not unlimited chemistry. Broad
ordinary support remains through `generic` and other options. A finite descriptor
menu in one diagnostic is not a claim that the entire generator is a finite
fragment vocabulary.

The inherited 11-step `BUILD_RING_SYSTEM` program remains disabled in the current
runtime/refinement recipe. Other compound options are already implemented. Do
not confuse this specific program decision with a blanket ban on compound options.

### HOW: primitive execution within a chosen channel

An option restricts admissible primitive support and executes every intermediate
through the same executor. It never teleports to the endpoint. Depending on the
experiment, HOW uses the reference conditional law, a structural committor tilt,
or task guidance. These are different deployments, not one universally enabled
stack.

The historical region controller has the schematic form

\[
q(y\mid x,M,o)\propto R_{M,o}(y\mid x)
\widehat h(y,M,o,b-1)^\eta,
\qquad D_{\rm KL}(q\|R_{M,o})\le 1.
\]

A structural rewrite-completion committor is not a docking future-value model.
Also, conditioning on an option already changes the reference row. A one-nat
constraint relative to \(R_{M,o}\) is not a global one-nat constraint relative
to the unconditioned raw \(R_\theta\).

### Population and oracle allocation

The conserved unit of outer allocation is a `(parent, region, option)` bundle.
Frontier size must not silently grant a long trajectory more WHERE probability.
Intermediates are reusable search states, not automatically extra oracle slots.

The historical T4 architecture used eight lineages, several region bundles per
lineage, parallel proposal generation, canonical deduplication, bundle-diverse
selection, concurrent docking, and archive/resampling after the round. Within a
round, candidates were locked before any of that round's docking results could
change selection. Later rounds could learn from completed earlier rounds.

Round synchrony is an experimental choice, not a theorem that all good molecular
optimizers must be synchronous. Changing it requires explicit accounting for the
resulting policy, data, and comparison.

## 5. Exact Doob control, future value, and the optimization objective

For a nonnegative terminal desirability \(g_z\), the reference future value is

\[
h_0(x,z)=g_z(x),\qquad
h_b(x,z)=\sum_yR_\theta(y\mid x)h_{b-1}(y,z).
\]

Where \(h_b>0\), the exact Doob transform is

\[
P^*(y\mid x,z,b)=R_\theta(y\mid x)
\frac{h_{b-1}(y,z)}{h_b(x,z)}.
\]

Its path probabilities telescope, producing the reference terminal law tilted
by \(g_z\). This is exact only with the correct backward function, horizon,
boundary conditions, and state. A first-hit objective needs appropriate absorbing
semantics; an arbitrary early stop does not inherit an exact terminal tilt.

For \(g_z(x)=\exp(\beta U_z(x))\), the path tilt also solves the appropriate
entropy-regularized path-distribution problem

\[
\max_P\;\mathbb E_P[U_z(X_K)]
-\beta^{-1}D_{\rm KL}(P\|P_R),
\]

under the usual finite-normalizer conditions. This is not a guarantee of finding
the maximum molecule or maximizing a benchmark's top-ten AUC with finite compute.

The following objects must remain distinct:

- a property predictor for the current molecule;
- expected future desirability under a specified policy;
- probability of reaching an improvement threshold under that policy;
- the maximum achieved by an answer-known teacher route;
- the existence of any executable successful route;
- probability that a completed candidate deserves an expensive oracle call.

They are different labels and different learning problems. Several previous
failures involved substituting one for another.

### What the paper supports, and what it does not

The supplied paper reports QED editing success of 446/800 at eight returned
trajectories, versus GrIDDD's reported 45.1% at twenty; bounded target-known
future-aware recovery of 40/65 versus 26/65 for local selection; and positive
retargeting results. These support using executable trajectories and future
reachability as a research direction.

They do not qualify a QED head for docking or PMO. The bounded reachability study
is not evidence that an amortized T4 head knows valuable multi-ring futures.
The paper's diversity value 0.4999 conditions on solved sources with at least two
distinct non-source outputs; including singleton solved sources gives 0.376.
Pathwise validity follows from enforcement, not a learned performance gain.

## 6. The new semi-Markov and particle machinery: what is actually built

An option consumes a variable number of primitives, so option boundaries naturally
define a semi-Markov decision process. This reduces the decision horizon without
pretending that the primitive work disappears.

The augmented state includes exact molecule, option/history context, incumbent,
remaining option clock, and controller identity. If the target concerns best
value already seen, the path incumbent must be included. If the policy depends
on an evolving archive or learned model, those dependencies must also be fixed
or represented in its state/identity.

A convenient explicit reference option-path law is

\[
B(M,o,\xi,\omega,x',\ell\mid s)
=Q_0(M\mid x)\rho_0(o,\xi\mid x,M)
K_0(\omega,x',\ell\mid s,M,o,\xi).
\]

Here \(\ell\) is primitive duration. The option clock, primitive work, and
expensive oracle budget are separate quantities.

The new code contains:

1. Policy-identified, censoring-aware continuation targets.
2. A distributional improvement head monotone in horizon and threshold.
3. A conservative advantage-weighted option actor with a positive base floor.
4. Persistent exact-state particles, proposal/reference probabilities, deterministic
   resampling and restartable state.
5. Canonical candidate locking and joint-posterior batch acquisition.

For example, an improvement value can target

\[
H^\pi(s,b,\Delta)=
\Pr_\pi\!\left[\max_{0\le j\le b}U(X_j)
\ge u_\star+\Delta\mid s\right].
\]

Labels must come from the identified continuation policy and clock. A truncated
suffix is not automatically a failure at an unobserved longer horizon. A teacher
path's success is not the success probability of the deployed stochastic policy.

Persistent SMC records the incremental correction

\[
\log w' = \log w + \log B-\log q
+\log h(s')-\log h(s).
\]

Boundary potentials and terminal semantics determine the actual target. If the
proposal changes while a fixed reference target is claimed, the reference/proposal
ratio matters. Using a guide to select and weight without the appropriate
correction can double-count guidance.

Crucially, correcting a learned proposal back to a fixed reference target does
not itself redefine that target as an improved policy. We must explicitly choose
whether SMC is estimating a fixed tilt, supplying policy-improvement targets, or
serving as a heuristic optimizer. Similarly, a value for one policy is not the
exact Doob value for a different base law.

Status: the integrated mechanical runtime passed 57 focused tests in its recorded
implementation check. It is not a qualified learned T4 controller. The compatible
retrospective bank lacked positive identified improvement targets, so that value
gate abstained. The current learned actor does not yet solve general attachment
and primitive selection. Weighted-ensemble stratification and other suggested
extensions are not all integrated production features.

### 6.1 Necessary preliminaries for understanding this implementation

An option has an initiation condition, an internal primitive policy, and a
termination condition. It is a temporally extended action, not just an alias for
a primitive family. A primitive-family channel and a completed ring-construction
program therefore need different duration and completion semantics even if both
appear in the WHAT interface.

A proposal distribution is where we actually draw the next decision. A reference
distribution is the law relative to which a regularizer or importance ratio is
defined. A target distribution is what weighted particles are intended to
approximate. These can differ; the code records them rather than conflating them.

A twisting function allocates particles toward promising futures before terminal
reward is known. With the appropriate correction and terminal boundary, it can
change computational efficiency without changing the declared terminal target.
It does not create promising branches that the proposal never samples.

An exploration floor prevents an applicable channel from receiving exactly zero
probability. It does not imply practically adequate exploration: a tiny positive
probability can still make discovery impossible at the available compute.

Finally, Markov sufficiency belongs to the augmented controller state, not merely
to canonical molecular identity. Two identical molecules reached with different
remaining budgets, preserved contexts, option phases, or incumbents can require
different continuation laws and values.

### 6.2 Concrete state machine and executable continuation

`MolecularHierarchy` exposes three stages: `where`, `what`, and `how`.
`MolecularSearchState` carries the exact graph, lineage, primitive budget, root
identity, selected region, and active option state when present.

At WHERE, the current implementation enumerates connected regions with sizes
1 through 24 and uses the existing region row with a 20% scale floor. At WHAT,
it evaluates applicability against actual legal region-supported actions, remaining
capacity, and primitive budget. It can use lazy product-applicability checks.
Options longer than the remaining primitive budget are unavailable. The reference
WHAT row combines balanced purpose allocation with a 10% uniform-option floor.

`OptionState` contains the current graph, exact origin, rewrite context, lineage,
option identity, phase, horizon, bundle identity, and typed progress for fused
construction, general ring construction, expansion, carbonyl insertion, or region
replacement. These fields prevent a closure from accidentally acting on the
wrong atoms or violating preserved context.

`OptionContinuationKernel` provides the actual supported within-option process.
`FiniteHorizonContinuation` and `sample_option_trajectory` support reference,
bounded exact, and sampled continuation estimators. Exactness applies only when
the relevant continuation graph is fully evaluated. Sampled estimation and
fallbacks must retain their actual status and work accounting.

Lazy reference rows defer materializing successors until needed; exact-state
memoization avoids repeated work at shared states. These optimizations are intended
to change computation, not the probability law. They do not license top-k
truncation masquerading as an exact sampler.

### 6.3 The current neural inputs are modest, not a learned graph planner

The default boundary molecular representation is a 512-bit radius-two Morgan
fingerprint plus five normalized descriptors: heavy-atom count, cycle rank,
aromatic-ring count, hydrogen-bond donors, and acceptors. This 512-bit modeling
fingerprint is separate from T4's 2048-bit benchmark similarity definition.

Value inputs append current oriented utility, incumbent utility, and remaining
option clock. Actor inputs additionally append six numeric region descriptors:
released fraction, size fraction, boundary-bond count, context-component count,
ring presence, and ring-boundary fraction.

Each option is encoded compositionally with purpose/family indicators, special
operation flags, pendant/fused topology, ring size, C/N/O counts, electronic
character, refinement, and primitive horizon. The actor scores concatenated
state and option features using a two-hidden-layer, width-64 SiLU MLP by default.

This is a real state-conditioned actor, but it is not an atom-attention graph
encoder, an attachment pointer network, a learned graph-delta decoder, or a
multi-option plan generator. The feature callback can be replaced, but richer
representations have not already been trained and validated merely because the
interface permits them.

### 6.4 Actor learning and deployment are separate computations

`fit_option_actor` implements an advantage-weighted likelihood objective with
a KL penalty. Schematically,

\[
\mathcal L(\psi)=\frac1N\sum_i
\left[-w_i\log q_\psi(o_i\mid s_i)
+\lambda D_{\rm KL}(q_\psi(\cdot\mid s_i)\|\rho_i)\right],
\]

where weights combine clipped exponential advantage and any explicitly supplied
off-policy correction. Training mixes a softmax tilt of the base with a positive
base floor. At deployment, `conservative_option_distribution` additionally
solves the declared KL-limited tilt. Current defaults are a 10% base floor and
one-nat option-level KL ceiling. Training's KL penalty is not itself the
deployment hard constraint.

Rows record source, decision, behavior-policy identity, available options,
reference/behavior probabilities, selected action, and advantage. The interface
rejects undeclared policy mixing. Supplying an arbitrary positive importance
weight does not prove that it is a correct likelihood ratio; that remains a
data-contract responsibility.

The winner-imitation experiments use supervised demonstrated choices instead.
They do not fabricate advantages, behavior likelihoods, or docking future labels.
Their rejected actor is therefore distinct from a successfully trained
advantage-weighted task controller.

### 6.5 Distributional value: exact parameterization and limitations

`MonotoneImprovementModel` uses a width-64 MLP and predicts a grid over registered
option horizons and improvement thresholds. Its logits have the form

\[
L_{b,d}(s)=a(s)+\sum_{j\le b}\operatorname{softplus}(v_j(s))
-\sum_{k\le d}\operatorname{softplus}(r_k(s)),
\qquad \widehat H=\sigma(L).
\]

This enforces nondecreasing probability with horizon and nonincreasing probability
with improvement threshold. It is a restricted monotone family, not a universal
return distribution. It is fitted using weighted binary cross-entropy on cells
identified by the observed continuation; unknown cells receive zero loss weight.

The evaluator reports identified-target coverage, positive rate, Brier score,
ranking AUC, decision coverage and improvement precision. Correctly refusing to
label unobserved failures is useful, but does not alone prove unbiased survival
estimation. Censoring and selection effects, especially selectively known positive
cells, still require statistical assessment before claiming calibration.

The current runtime's intermediate potential is specifically an approximation:

\[
\log\widetilde h(s,b)
=\beta\left[u_\star+
\sum_j(\Delta_j-\Delta_{j-1})\widehat H(s,b,\Delta_j)\right].
\]

The sum is a truncated survival-integral approximation to expected improvement.
The terminal log potential is \(\beta u_{\rm best}\), with default \(\beta=10\).
In general, exponentiating expected improvement is not the same as taking the
expectation of exponentiated best utility. This is explicitly an approximate
intermediate twist, not the exact Doob backward function of that terminal reward.

### 6.6 Persistent particle behavior and acquisition

`PersistentOptionPopulation` and `PersistentOptionParticle` serialize exact
states, ancestry, normalized log weights, potentials, RNG state/stream identities,
option boundary, controller snapshot, and frozen control context. Each completed
option records its reference/proposal probabilities across WHERE, WHAT and HOW.

After an update, effective sample size is
\(\mathrm{ESS}=1/\sum_i\bar w_i^2\). Systematic resampling triggers below half
the population size. Descendants receive separate reproducible streams. Failed
particles have zero weight; complete extinction is recorded rather than hidden
by an unaccounted restart. State, actor, value, and context changes invalidate a
frozen run instead of silently changing its policy.

Only completed canonical-unique candidates can be locked for acquisition. The
current acquisition implementation consumes joint posterior draws, estimates
each candidate's marginal probability of being the best, splits tie credit, and
selects the highest ranked candidates. It is not a full joint optimization of
batch expected improvement, information gain, or diversity. Nor does its existence
provide a calibrated molecular posterior; posterior construction is a separate
responsibility.

### 6.7 Utility access is an unresolved task interface, not free docking

The runtime requires a frozen utility callback oriented to [0,1], caches its
outputs, and owns no task oracle. Raw negative docking scores cannot be passed
unchanged. The transformation, label source and applicability must be declared.

If this callback uses a cheap learned property predictor, the continuation model
learns futures according to that predictor, not automatically true docking
improvement. If it calls docking or a benchmark oracle at internal option boundaries,
those calls must be charged. An exact terminal potential is exact relative to
the supplied utility, not necessarily to an unobserved experimental or docking
objective. The new reference-only refinement assay deliberately does not pretend
this task-value interface has already been solved.

### 6.8 Integration inventory

| Component | Built? | Current qualification/deployment boundary |
| --- | --- | --- |
| Exact broad chemistry and frozen successor model | Yes | Existing production substrate |
| WHERE/WHAT/HOW hierarchy and complete-option execution | Yes | Used in broad development experiments |
| Parameterized rings, carbonyls, expansion/replacement | Yes | Enabled options vary by explicit recipe |
| Lazy reference evaluation and exact-state caching | Yes | Preserve tested row semantics; not permission to shrink support |
| Persistent semi-Markov runtime | Yes | Mechanical integration verified |
| Conservative learned WHAT actor | Yes | Winner-informed fit not promoted; no broad benchmark win |
| Censoring-aware distributional improvement learner | Yes | Compatible retrospective gate abstained |
| Exact log-ratio SMC bookkeeping and deterministic restart | Yes | Finite-particle efficiency still needs task evidence |
| Joint-draw probability-of-optimality selector | Yes | No proven calibrated T4 posterior attached |
| Inverse winner decomposition and imitation data pipeline | Yes | Development-only; later preserved in `909380e` |
| Strong learned atom/attachment/payload proposal | Not yet demonstrated | Major missing capability in controller learning |
| Trained, calibrated, end-to-end T4/PMO future-aware stack | No | Infrastructure must not be reported as this result |
| Broad controller beating IVG under matched protocols | No | The actual outstanding goal |

The existing code is therefore considerably more than a raw primitive sampler.
However, it is not yet the complete trained, task-effective system implied by a
diagram containing all these components.

## 7. T4 results and the historical baseline issue

T4 returns the best docking score among molecules satisfying endpoint QED >=0.6,
SA <=4, and Morgan radius-two/2048-bit similarity to the original seed >=delta.
Lower QuickVina2 score is better. Intermediate states need executability, not
these endpoint feasibility thresholds.

### The workshop table is not one current frozen baseline

`diagnostics/controller_baselines/t4_lineage.json` reconciles the table's inputs.
It selects between two historical controller generations and averages only
feasible compact-controller runs. Eleven cells come from the compact 100-call
generation, fifteen from the older 200-call generation, and four have no feasible
result. It is not a single frozen 500-call controller or a uniformly replicated
matched comparison.

For PARP1 seed0, delta=0.4, the compact controller's three recorded 100-call runs
are -10.9, -10.6, and -10.7, reported mean -10.733. Preserve these useful historical
results while correcting their attribution. A manuscript assertion cannot replace
the run ledger.

Also, averaging three independent baseline runs does not mean that one baseline
search had three times the per-run oracle budget. A sixfold efficiency claim
based on that multiplication needs correction or a genuinely matched definition.

### Most recent completed paired broad-controller assay

| PARP1 seed0, delta=0.4 | Post-hoc arm | In-loop arm |
| --- | ---: | ---: |
| Shared warm archive | 51 historical calls, best -9.7 | Same |
| New docking attempts | 4 | 8 |
| Best new score | -9.9 | -10.0 |
| Distinct docked bundles | 4 | 8 |
| Proposal time, slowest lineage | 793.1 s | 1,065.1 s |

Both best candidates completed a six-member pendant-ring option, with cycle-rank
and ring-system changes of +1. In-loop guidance improved eligible yield in this
small development pair, but not enough to establish competitive performance.

The project's verified comparator audit records GenMol -10.6 and three released
IVG winner values -13.5, -13.6, -13.6 for this cell. A separately discussed -14.1
paper value must not be conflated with these released endpoints or a fresh
same-pipeline redock.

The endpoint predictor ranked the observed best ring last in both docked batches.
WHERE and WHAT had no task-value contrast; only eight HOW decisions were guided.
This is direct evidence of poor ranking in those batches, not proof that all
surrogates or fingerprints are incapable of recognizing ring chemistry.

Source: `diagnostics/t4_frontier_compare/` and `docs/T4_FRONTIER_COMPARISON.md`.

## 8. A concrete known winner and its executable route

PARP1 development seed, 19 heavy atoms:

```text
CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3
```

Exact inspected IVG winner, 30 heavy atoms:

```text
O=C1CCCc2cc(CCc3ccc4c(c3)CNC(=O)c3cccn3C4=O)ccc21
```

One saved compiled route has five semantic stages and 21 primitives:

| Stage | Primitive count | One recorded production docking score |
| --- | ---: | ---: |
| Remodel the linker | 4 | -7.7 |
| Construct pendant benzene | 7 | -9.5 |
| Construct fused six-ring | 5 | -11.8 |
| Add ring carbonyl | 1 | -12.5 |
| Insert core carbonyl | 4 | -13.6 |

These are five semantic stages, not proof that all five are already autonomous
single decisions with good learned probabilities. Earlier witnesses had 23,
25, or 27 primitives. None of these counts proves a shortest path.

The endpoint was actually docked in our production pipeline and returned -13.6
in the stored diagnostic. A saved incumbent previously recorded at -11.0 returned
-9.8 in that same diagnostic. These single evaluations establish neither low
noise nor deterministic improvement at every stage. The route is answer-known
reconstruction, not autonomous discovery.

The raw six-call receipt is on `compose-v4-artifacts` at
`t4_winner_route_docking/fece76dedf244574e46892c7c8eb4bf3d5a9ae114e3abcd2be9e47bb779f03c9/result.json`,
SHA-256 `02f9d33a271b1bea6b74252fe107d0e2761bad40b7d9a869f808f2671bb9176d`.

The useful lesson is not simply "add more rings." It is coordinated linker
remodeling, pendant construction, fusion, and carbonyl/core editing with the
correct attachments and preserved context.

## 9. What we learned from using published winners

The current user explicitly permits winner-informed development, including exact
SMILES, routes, reconstruction and starting from winners. Keep it separate from
blind discovery. Previously inspected tasks do not become fresh held-out tasks
because a later fitting split excludes their examples.

### Representability and demonstrations

The route audit has 91 distinct source/winner pairs, with supported saved witnesses
for 82. The nine remaining cases retain explicit representation or unresolved
route exclusions. Do not claim every public winner is covered.

A first primitive-witness imitation dataset replayed 1,576 actions but supplied
no whole-ring training labels. Valid primitive witnesses need not be ordered in
a way that yields contiguous compound-option labels. The actor underperformed
the simpler demonstration-frequency prior and was not promoted.

Inverse decomposition repaired that specific supervision gap. It removes a
peripheral ring using valid deletions, reconstructs it using the existing ring
program, and restores terminal decoration. It produced 83 verified reconstructions
across 64 of the 82 supported winners, 78% coverage of that subset. All 83 accepted
examples reconstruct the canonical endpoint. These precursors are inverse-derived,
not necessarily benchmark seeds.

Preparation cost: 22.732 seconds, 6,047 executor applications, zero reference-model
evaluations and zero oracle calls. After deduplication there are 115 decisions:
80 across eleven fitting sources, 35 across four excluded sources. The fixed
retrospective roles preceded extraction.

### The learned proposal still did not beat the simple prior

| Source-balanced excluded-source diagnostic | Balanced reference | Demonstration marginal | Neural actor |
| --- | ---: | ---: | ---: |
| Negative log likelihood | 5.848 | 3.813 | 3.995 |
| Mean demonstrated-option probability | 0.00934 | 0.05842 | 0.02429 |
| Top-label agreement | 0 | 0.139 | 0 |

The fit took 0.526 seconds. This is not a compute bottleneck. The 213-descriptor
classification menu contains inapplicable distractors, so these are not production
route-recovery metrics. The neural actor remains unpromoted.

### A global ring prior does not coordinate the route

A counterfactual used four actual saved PARP1 option rows and unchanged measured
HOW probabilities. The demonstration marginal increased the first needed pendant
aromatic-ring option by 12.744 times, but reduced each of the next three needed
options to 0.491 times its original probability.

The specified exact four-option prefix probability moved from approximately
4.733e-12 to 7.122e-12, only 1.505 times larger.

Conditioning is essential: this starts after linker remodeling and assumes a
fully mutable region at each option. It is not a seed-to-winner probability,
the total mass of all successful routes, or a bound on nearby high-scoring molecules.

Nevertheless, it is strong diagnostic evidence that merely increasing ring-option
frequency does not fix sequence coordination and low-level choices. Complete
multi-option demonstrations, core-carbonyl decomposition, and attachment-level
learning remain specific missing development work.

Sources: `docs/WINNER_PROPOSAL_DEVELOPMENT.md`,
`diagnostics/winner_option_proposal/`, and `diagnostics/inverse_ring_proposals/`.

## 10. PMO status and the failed future guide

PMO measures a different optimization regime: 23 tasks, ordinarily a 10,000-oracle
budget and top-ten area under the oracle-call curve. A best endpoint score is not
that AUC. Prescreened task labels and historical policy-training evaluations
change the regime and require explicit accounting.

The latest warm-start Perindopril MPO locked-pool assay improved our observed
champion from 0.6835298931 to 0.6948083338 using 95 calls. All three parent
improvements came from the actor-top stratum; uniform and largest-release strata
produced none. This is a useful local ranking result, not a reproduced complete
optimizer improvement or a matched PMO/IVG win.

A public IVG example transcribed from a released figure scores 0.80883 in our
evaluator and has exact reconstructed routes from the declared starts. That
example is not bound by the inspected metadata to the no-prescreen runs. Its
endpoint score must not be compared to a published AUC.

The authors' current results repository explicitly documents separate prescreened
and no-prescreen protocols. The prescreen path scores roughly 250,000 ZINC molecules
across the task oracles. Its README reports Perindopril top-ten AUC 0.753 with
prescreening. Our 0.6948 best endpoint is not the same statistic.
Source: [IVG official repository](https://github.com/invirtuolabs/InVirtuoGen_results).

### Complete-option SMC has already had a negative development result

An eight-particle, six-option-opportunity PMO experiment compared reference,
immediate-score SMC, and an older future-head SMC. All reached the same best
score, approximately 0.522233. Actual executed depths were 14 to 19 primitives
despite a theoretical allowance of 66. Temporal abstraction and particles alone
did not improve the champion.

The older guide was especially problematic. Its first-boundary top candidate had
current score 0.002166, predicted achieved value 0.780725, and a sampled suffix
maximum of only 0.079737. Across eight trajectories, first-boundary correlation
with eventual terminal score was -0.238 for the head versus +0.524 for current
score. It was worse at all five nonterminal boundaries in that audit.

Its target and clock were mismatched: training on best witnessed teacher routes
is not policy-congruent continuation value, and nominal primitive budget is not
the actual remaining option clock. The small correlated sample does not isolate
every cause, and one bad suffix does not bound all possible futures. It does
justify not scaling this head unchanged.

The newer value machinery addresses target semantics but still lacks sufficient
compatible positive continuation evidence. A prepared balanced three-option bank
exists; its remote generation is not authorized merely because the code exists.

Sources: `docs/PMO_PUBLIC_MOLECULES.md`, `docs/PMO_OPTION_CONTROLLER_BANK.md`,
`diagnostics/pmo_option_particles/GUIDE_AUDIT.md`, and current campaign records.

## 11. The failure mechanisms: established, suspected, and unknown

### Established in bounded diagnostics

- The first region path bypassed macro contracts; current paths have reintegrated them.
- Raw primitive probability heavily favored decoration. The historical cycle-close
  mass of 7e-5 omitted another closure family and is not total ring-closing mass.
- Some productive option support was lost through proposal enumeration/caps;
  executor reachability alone did not imply production proposal availability.
- Frontier growth can be very expensive: an early round reached 55,038 candidates
  for twenty docking slots and used 4,991 seconds of proposal time.
- Bundle-diverse selection repaired an early 19/20 same-region collapse.
- Large intended regions frequently yielded small realized changes.
- Endpoint feasibility masks on intermediates excluded useful routes; endpoint-only
  treatment is now explicit in the relevant new recipes.
- The recent T4 predictor misranked the best completed ring in both sampled arms.
- The old PMO future head had poor observed continuation ranking and wrong target semantics.
- Static option frequency and the small demonstration actor did not solve route coordination.

### Plausible explanations, not yet settled causal results

- Task-blind attachment/payload selection may dominate failure after WHAT is fixed.
- Broadening chemistry without learning where to put probability dilutes useful paths.
- Current features may inadequately represent graph edits and attachment context.
- Sparse, biased replay may lack the positive/negative contrasts needed for value learning.
- Repeated selection of immediately good molecules may discard temporarily worse
  but productive precursors.
- Regularization toward an unhelpful reference may conflict with finite-budget
  best-of-run optimization.
- Proposal enumeration/model evaluation, rather than docking alone, may limit
  useful development throughput.

### Still unknown

- Whether a strong transferable attachment-aware proposal closes a meaningful T4 gap.
- Whether qualified future values improve a matched live run beyond good proposals.
- Whether persistent SMC outperforms independent complete-option sampling with
  equally good learned proposals and equal compute.
- Whether starting from the inspected winner yields repeatable further improvement.
- How much of external docking-score disagreement comes from ligand preparation,
  receptor/box setup, stochastic search, or other protocol differences.
- Whether the present frozen reference and representation can match IVG broadly
  within practical data and compute limits. There is no theorem that they must.

## 12. The approved winner-initialized refinement experiment

The user approved this diagnostic to ask whether there is accessible headroom
around an already excellent molecule. It does not ask whether autonomous search
can reach that molecule from the original benchmark seed.

The committed recipe uses:

- the exact 30-heavy-atom PARP1 winner above as the initial molecule;
- one full positive-mass primitive-successor census through the production evaluator;
- sixteen independent streams, at most two complete existing options and fourteen
  primitives each;
- existing WHERE, balanced WHAT, lazy reference HOW, mandatory generic, and enabled
  ring/carbonyl channels;
- no endpoint predictor, learned actor, future head, or winner-distance reward;
- fourteen distinct eligible edited candidates, balancing complete-option and
  ordinary local proposals through option/family grouping and diversity;
- original-seed endpoint QED/SA/similarity and the existing endpoint medchem gate;
- three fresh winner dockings, fourteen first candidate dockings, and two repeat
  dockings of the selected best candidate: at most nineteen new calls;
- explicit QuickVina seeds 1701, 1702, 1703 for corresponding replicate positions;
- unchanged stochastic Open Babel gen3D preparation, with ligand/pose hashes retained;
- at most eight proposal workers or eight docking workers plus a driver, CPU only,
  a one-hour driver ceiling and a reserved $10 cap, with no automatic follow-on.

The first candidate replicate is used for selection. Report repeat-only differences
separately, not just the optimistically selected mean. Three repeats are a
development check, not strong statistical proof of superiority.

Operational status at this historical snapshot: implementation and launch were
committed at `66a5f6e`. Thirty focused tests passed for the bounded change and
dependencies. Clean-source preflight passed, while the first Modal deployment
failed before spawning or spending a docking call. The assay was subsequently
repaired, pushed, run, and reported; consult `docs/START_HERE_ICLR.md` rather than
using this historical paragraph as current status.

The current assay can provide information about local headroom and feasible
proposal yield. It does not by itself validate the proposed future-aware controller
or solve seed-to-winner discovery.

## 13. What we want the external agent to rethink

Do not assume the existing architecture is the answer just because its math is
coherent. Equally, do not recommend a fashionable algorithm without specifying
what concrete failure it repairs. The previous literature synthesis is in
`docs/CONTROLLER_LITERATURE_DECISION_2026-09-12.md`; it is a proposal, not an
experiment showing that its hybrid is optimal.

The central question is:

> Given an exact molecular executor, a frozen learned primitive reference,
> compositional multi-edit options, public successful demonstrations, and a
> limited expensive-oracle budget, how should we learn and allocate probability
> so useful multi-option, attachment-specific trajectories become common enough
> to improve actual optimization performance?

My present working hypothesis is that the highest-leverage missing component is
state- and attachment-conditioned proposal learning across complete transformations,
with task feedback and correctly defined continuation value. More generic particles
cannot reliably rescue trajectories whose relevant choices remain negligibly likely.
This hypothesis is not yet a validated controller result.

Please specifically address:

1. **Representation.** What should represent a molecule, mutable region, attachment,
   proposed graph delta, and history? Would a graph/atom-conditioned parameter
   decoder materially improve on the current coarse descriptor scorer?
2. **Proposal factorization.** Which choices should be learned jointly, and which
   should remain conditional enumeration? Avoid allocating all mass to popular
   option labels while missing the right sites and payloads.
3. **Temporal coordination.** How should a controller discover sequences such as
   linker edit, pendant ring, fusion, carbonyl, and core remodeling without an
   endpoint template or a known winner at inference?
4. **Learning from winners.** How should exact routes and inverse-derived precursors
   train transferable behavior? Which additional examples, perturbations, failed
   alternatives, or complete decompositions are most informative per unit compute?
5. **Future value.** What precisely is the prediction target, behavior policy,
   clock, censoring treatment, and useful ranking/calibration test? How can we
   gather positive contrasts without launching another large ineffective run?
6. **Objective.** Are we sampling a reference tilt, improving a policy, or maximizing
   best/top-k performance under a query budget? State what any importance
   correction preserves and what it does not optimize.
7. **Oracle allocation.** When should we dock directly, learn an endpoint predictor,
   collect information, or rescore the apparent champion? How do we detect harmful
   surrogate ranking before it consumes a campaign?
8. **Evidence.** What single small experiment would distinguish your leading
   explanation from the nearest alternative? Specify expected positive, negative,
   and inconclusive observations and the resulting next action.

A useful answer should give a strongest recommendation, a simpler fallback, the
specific existing code to retain/change, a concrete learning objective and data
recipe, a bounded experiment, and the most likely way the recommendation could
fail. Do not promise certain IVG recovery or confuse target-conditioned route
reconstruction with autonomous benchmark success.

## 14. Code, artifacts, and operating constraints

### Main code map

| Concern | Location |
| --- | --- |
| Exact graph state | `src/compose_v4/chem/molecular_graph.py` |
| Executor/support | `src/compose_v4/rewrite/` |
| Frozen model | `src/compose_v4/model/factorized_tracelet_rate_model.py` |
| Canonical probability evaluator | `src/compose_v4/experiments/production_successor_kernel.py` |
| WHERE | `src/compose_v4/control/region.py`, `region_selector.py` |
| Macro support and WHAT | `macro_engine.py`, `option_selector.py`, `molecular_task_search.py` under `control/` |
| Complete ring/carbonyl execution | `ring_program.py`, `fused_option.py`, `carbonyl_option.py`, `option_continuation.py` under `control/` |
| Boundary policy/value | `option_controller.py`, `option_policy.py`, `improvement_value.py`, `option_features.py` under `control/` |
| Persistent runtime/SMC | `option_controller_runtime.py`, `persistent_option_smc.py` under `control/` |
| Oracle batch allocation | `src/compose_v4/control/batch_acquisition.py` |
| Paper-era reachability/SMC | `src/compose_v4/experiments/hphi_rollout.py`, `hphi_smc.py` |
| Winner demonstrations | `src/compose_v4/control/option_demonstrations.py`, `demonstration_prior.py`; `experiments/inverse_ring_demonstrations.py` |
| New bounded assay | `src/compose_v4/experiments/t4_winner_refinement.py` |
| Remote T4 app and launcher | `modal_apps/genmol_t4_opt_app.py`, `tools/t4_launch.py` |

### Reading and provenance

Read `AGENTS.md`, `docs/START_HERE_ICLR.md`, and `docs/PAPER_TO_CURRENT_CODE.md`.
The workshop-package boundary is commit `34da388`; the post-submission controller
campaign begins at `f0a49ab`. Individual scientific results have their own producer
revisions and input identities. A paper folder is not a single experiment commit.

`docs/HANDOFF_CODEX_2026-09-05.md`, `docs/MACRO_INVENTORY.md`, and
`docs/CAMPAIGN_LESSONS.md` preserve essential history. The last is a point-in-time
memory export, not an automatically updated status source. Older "next run" or
"no run authorized" prose predates the explicitly approved winner-refinement
recipe; consult its scoped contract rather than silently expanding any budget.

Bulk exact states, checkpoints, docking assets, locks, and receipts live on the
`compose-v4-artifacts` Modal volume, not entirely in Git. Historical handoff paths
include `/artifacts/editing_v2/r_theta_run` and
`/region_committor/committor_bellman_v1.pt`. Always verify the specific consuming
recipe's identities. A fresh clone alone cannot reproduce the banked remote runs.

The winner-demonstration implementation and diagnostics were subsequently
preserved in commit `909380e`. They were not part of the older clean launch
revision named by this historical snapshot. Current repository and run status
are maintained in `docs/START_HERE_ICLR.md`.

### Practical discipline

- Use existing compatible trajectories, scores, exact-state caches and receipts.
- Parallelize independent units without changing RNG streams or global selection.
  General approval permits up to thirty containers, but each run has its own
  narrower current cap and budget.
- Record proposal/model work, wall time and oracle calls separately. Large numbers
  of cheap candidates can still cost excessive proposal compute.
- Use focused tests for affected scientific invariants; do not repeatedly gate
  iteration on unrelated full-suite work. Full release qualification remains
  separate; the recorded unrelated Editing-V2 registry mismatch is not waived.
- Launch scientific work from clean committed source. Run preflight, then
  `modal deploy modal_apps/genmol_t4_opt_app.py`, then the appropriate durable
  `python3 tools/t4_launch.py` mode. Do not use `modal run --detach` for a long driver.
- Keep the reference frozen and generic support active unless a new scientific
  decision explicitly changes the contract. Do not silently narrow chemistry,
  relax endpoint gates, invent labels, or turn public examples into held-out evidence.
- Prescreening and winner-informed training are allowed development choices, but
  compare and report them in the corresponding regime with all historical task
  labels and policy-training oracle costs accounted for.
- Do not archive the legacy workspace solely on an old "redundant" statement:
  current onboarding flags potentially load-bearing ignored checkpoint files.

The best next contribution is a measured improvement in useful proposal or search
behavior that survives the actual oracle, not another architectural diagram or
another expensive repetition of an unchanged null approach.
