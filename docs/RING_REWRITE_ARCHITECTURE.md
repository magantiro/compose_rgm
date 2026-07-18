# Ring rewrite architecture

## Current implemented system

The semantic substrate is a small reversible micro instruction set:

- `atom_insert` / `atom_delete`;
- `atom_restate`;
- `bond_insert` / `bond_delete`;
- `bond_reorder`.

The unrestricted micro system can create a ring by inserting a bond between
atoms already connected by a path. That is universal, but it is no longer the
main learned construction path. It has a delayed-topology defect: atoms may be
edited while acyclic and then retrospectively acquire a cyclic role when a
later closure fires. The terminal molecule can be valid while the teacher path
still has poor coordination and credit assignment.

The original compiler constructed a target spanning tree with final bond orders
and then inserted its non-tree chords. This is exact and universal over the
supported chemistry, but it gives weak supervision to bond-order correction and
makes aromaticity depend on coordinating many construction-time bond choices.

The production tree-source path implements six refinements:

1. **Topology-aware rates.** Each action encoding receives current atom ring
   membership, cycle size if a proposed pair were bonded, and the ring-system
   size that closure would create.
2. **Atomic full-system teachers.** The compiler groups every connected
   component of non-bridge target edges—including fused, bridged, and spiro
   combinations—and commits all of its cycle-rank edges, atom labels, bond
   orders, and aromatic semantics in one `ring_system_grow` event.
3. **Verified reversible lowering.** `ring_system_grow` and
   `ring_system_delete` are one learned event each, with exact inverse actions
   and validity-checked lowerings into the universal micro rules.
4. **Structured rather than memorized labels.** The rate model selects a
   complete topology/bond-pattern placement and predicts C/N/O/F labels with
   normalized, valence-masked site factors. A held-out heteroatom arrangement
   need not appear as an exact typed catalog entry.
5. **Resonance quotienting.** Aromatic placement support uses semantic class-4
   edges, while the executor lowers one valid Kekule representative. Alternate
   Kekule programs are not treated as different chemical ring topologies.
6. **Hard topology commitment.** The production family contains no learned
   ring ear or scalar closure. Growth matches only a complete neutral-carbon,
   single-bond acyclic scaffold and cannot overlap an existing cyclic atom;
   deletion matches one exact current cyclic component and restores a valid
   carbon-tree precursor.

These changes preserve pathwise validity. The macro CTMC is intentionally a new
generator learned at macro-event time scale; it is not claimed to reproduce the
waiting-time law of its sequential micro lowering.

## Topology-commitment invariant

No chemically decorated chain is retrospectively reinterpreted by an
independent closure. A neutral carbon scaffold may exist before the ring event,
but the event jointly commits its complete cyclic topology and final chemical
labels. This does not impose a rigid "all rings before all chains" phase;
acyclic branches and Graft remain available before and after a ring event.

This addresses a path-design problem rather than a state-validity problem. A
Markov chain is mathematically allowed to take an unusual valid route to an
endpoint, but such routes create delayed credit, multimodal precursor states,
and unnecessary learning burden. The conditional teacher and inference fiber
therefore share the stronger committed-topology support.

## Is the micro system the best universal substrate?

It is a good semantic substrate but not the most efficient final instruction
set.

Its strengths are completeness, simple inverse rules, exact validation, and no
finite fragment vocabulary. Its costs are long traces, many commuting program
orders, resonance aliases, and a combinatorial legal closure fiber. Aromatic
ring systems are particularly inefficient because several correlated bond
orders must be coordinated.

The appropriate analogy is a universal micro ISA with a small verified macro
layer, not a choice between atom-by-atom generation and fragment assembly.

## What stochastic rewriting contributes

The relevant stochastic-rewriting principles are:

1. **Separate qualitative rules from quantitative rates.** A small reversible
   generator set defines which transitions exist; contextual patterns refine
   their rates. Thermodynamic graph rewriting explicitly separates generating
   rules from connected energy patterns and uses a growth policy for contextual
   rule refinement.
2. **Use application conditions for global invariants.** Restricted rewriting
   theories compile structural constraints into constraint-preserving rule
   conditions. This matches the COMPOSE executor boundary: rates never override
   valence, connectivity, or task constraints.
3. **Treat matches as stochastic events but quotient chemical outcomes.** Rule
   embeddings/matches define marked CTMC events. Symmetric matches and different
   programs may reach the same molecule, so successor rates must be summed at
   the canonical chemical state.
4. **Compose recurring programs, retaining inverses.** A macro rule is justified
   when it represents a frequent correlated valid program and has a verified
   inverse. It should not change the reachable state space supplied by the
   micro generators.
5. **Detailed balance is optional here.** Reversible rules are useful for
   editing and correction, but a finite-time endpoint-conditioned generative
   bridge is not an equilibrium sampler and need not impose thermodynamic
   stationarity.

These principles support context-refined micro rules immediately and a small
verified macro layer after measuring a trace-level bottleneck.

## Production ring instruction set

### 1. Full ring-system grow

`ring_system_grow(system, scaffold, target_pattern, atom_labels)` installs the
entire connected cyclic system in one marked event. One action may add several
cycle-rank edges, so fused and bridged systems are not assembled through a
sequence of independently sampled closures. The v1 tree path matches an
existing acyclic carbon scaffold; the kernel also supports allocating new ring
slots for later direct-growth experiments.

The probability of this structured mark is normalized hierarchically as

`p(family) p(complete topology | state,time)
 p(scaffold match | topology,state,time)
 prod_v p(atom_type_v | match,topology,state,time)`.

The topology partition uses exact tree-subgraph feasibility without expanding
all matches. Only the selected topology's legal scaffold matches are
materialized and normalized. This preserves an exact distribution over valid
rule matches while avoiding the thousands of topology-placement pairs that a
flat action table creates on a 40-atom carbon tree.

The per-atom factors do not make the event non-atomic: all labels and edges are
validated and committed together. They avoid turning complete typed rings into
a memorized fragment vocabulary.

Semantic aromatic topology is quotiented across resonance forms, but executable
Kekule/lowering aliases are retained underneath that topology and their rates
are aggregated. Neutral heteroaromatic atom masks use ring size and local
Kekule valence; the sampler validates the completed macro and internally
resamples before any state transition if a proposed label combination fails a
chemical application condition.

### 2. Full ring-system delete

`ring_system_delete` removes every cycle-rank edge of one exact cyclic
component and restores a valid carbon-tree precursor in one event. It is
constructed by rebuilding the precursor and mechanically inverting the
verified grow program. Generated heteroatom labels are not required to match a
training template.

### 3. Electronic restatement

`ring_system_restate` remains a coordinated accelerator for valid changes of
cyclic bond-order patterns. Ordinary scalar bond opening/closure is retained in
the universal lowering language and legacy diagnostics, but is absent from the
production marked model.

The null-source `cycle_insert`, `cycle_attach`, and ear compiler remain for
historical ablations and semantic regression tests. They are not the ring
families trained by the carbon-tree production backend.

## Coverage across ring kinds

Topology and chemistry are orthogonal axes:

- the compiler groups connected non-bridge edges, so one system can be a
  simple, fused, bridged, spiro, or combined polycyclic topology;
- the topology/bond-pattern placement determines saturated, partially
  unsaturated, aromatic, and mixed electronics;
- factorized atom labels determine carbocycle versus heterocycle without
  changing the coordinated topology;
- disjoint cyclic systems connected directly or through linkers are installed
  as separate complete events; and
- observed topology frequency affects learned rates, while hard support and
  strain/macrocycle policies remain explicit application conditions.

This covers combined fused/bridged/spiro/linker topologies without one operator
per named motif. It does not imply that every graph-valid ring is low-strain,
synthetically accessible, or geometrically realizable; those remain separate
3D and synthesis constraints.

## Why not fire arbitrary edit bundles?

DiGress-style discrete graph diffusion resamples many fixed node/edge variables
per denoising step and permits chemically invalid or disconnected intermediate
tensors. In an exact rewrite CTMC, unrelated simultaneous events have
second-order probability, valid actions can conflict on valence, and rewrite
order can change the successor. Treating every edit subset as one mark also
creates an exponential joint action space.

Parallel score evaluation is useful, and disjoint commuting events may be
committed together when read/write and rate independence are verified. A whole
ring transaction is also a legitimate multi-edit event because it has one
chemical meaning and one validated successor. Arbitrary parallel micro edits
are a different discrete-time model, not a free acceleration of this CTMC.

## Frequency is learned; support is structural

Operator availability and operator probability are deliberately separate.
Topology rules retain structural support while corpus couplings and Generator
Matching determine their contextual hazards. Rule choice, scaffold match, and
atom labels are separately normalized conditional factors, not an exact typed
fragment lookup.

The topology conditional includes a fixed, add-one-smoothed log-frequency base
measure estimated from training rewrite events. The neural score learns a
context-dependent residual on top. Thus rare but valid 3/4-member and complex
polycyclic rules remain reachable without receiving the same initial mass as a
common five- or six-member ring.

On the fixed 1,067/133/133 C/N/O/F split (maximum 16 heavy atoms), 91.6% of
training molecules are cyclic, 71.9% contain an aromatic ring, 66.4% contain a
heterocycle, 34.0% are fused, 3.9% bridged, and 1.2% spiro. Among the 1,759
training SSSR rings, sizes 5 and 6 account for 562 and 1,108 respectively; sizes
3, 4, 7, and 8 are much rarer. The only >=12 macrocycle in the 1,333-molecule
eligible corpus falls outside the training partition. This is exactly why a
smoothed learned rate prior must not be confused with the operator support.

The current full-system transport audit reconstructed 200/200 independently
sampled C/N/O/F targets through valid connected states and emitted 501 atomic
ring systems: 368 single, 123 fused-or-bridged, 9 spiro, and 1 macrocyclic.
Cycle-rank deltas ranged from 1 through 6, with zero model-level ear or scalar
closure actions. A deliberately tiny eight-target training smoke demonstrated
why atom labels must be factorized: exact typed templates supported none of the
held-out paths, whereas structured topology/bond-pattern support covered all
four test paths and two of four validation paths; the two misses had topologies
absent from the tiny training set. Scaled held-out topology coverage remains a
required pretraining gate.

## Approaches not recommended as the base

- **Ring-fragment vocabulary:** efficient for familiar motifs but makes the
  ontology data-dependent, complicates deletion/correction, and weakens the
  universal-operator claim.
- **One primitive per named ring motif:** duplicates semantics; the complete
  cyclic-component transaction already covers the structural cases.
- **Hard ring-size cutoff as the only solution:** useful as a task condition,
  but unsuitable as the universal model because macrocycles can be legitimate.
- **Post-hoc aromatization:** can improve terminal molecules but is not part of
  the learned CTMC and obscures likelihood/rate semantics.
- **Unverified macros:** risk intermediate invalidity and make inverse editing
  ill-defined.

## Relationship to molecular graph grammars

Molecular hypergraph grammars encode valence through grammar structure and can
guarantee valid terminal molecules. Learned graph grammars often promote whole
rings to hyperedges or motifs for data efficiency. Those results motivate
ring-aware context and macro composition, but COMPOSE keeps complete molecules
as stochastic states and learns rates on executable reversible edits rather
than sampling a parse tree or a fixed motif grammar.

## Primary references

- Behr et al., [Rewriting Theory for the Life Sciences: A Unifying Theory of CTMC Semantics](https://arxiv.org/abs/2106.02573), 2021.
- Behr and Krivine, [Compositionality of Rewriting Rules with Conditions](https://arxiv.org/abs/1904.09322), 2019.
- Behr, [Tracelets and Tracelet Analysis of Compositional Rewriting Systems](https://arxiv.org/abs/1904.12829), 2019.
- Andersen et al., [A Software Package for Chemically Inspired Graph Transformation](https://arxiv.org/abs/1208.3153), 2012.
- Danos et al., [Thermodynamic Graph-Rewriting](https://arxiv.org/abs/1503.06022), 2015.
- Kajino, [Molecular Hypergraph Grammar with Its Application to Molecular Optimization](https://proceedings.mlr.press/v97/kajino19a.html), ICML 2019.
- Guo et al., [Data-Efficient Graph Grammar Learning for Molecular Generation](https://arxiv.org/abs/2203.08031), ICLR 2022.
- Sun et al., [Representing Molecules as Random Walks Over Interpretable Grammars](https://proceedings.mlr.press/v235/sun24c.html), ICML 2024.
