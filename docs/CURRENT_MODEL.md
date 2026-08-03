# Current model contract

> [!CAUTION]
> Historical model contract, retained for provenance. Despite the filename,
> this describes an earlier unconditional whole-ring system. It is not the
> current Active8 source-conditioned Editing-V2 model or execution contract.
> Start with [`START_HERE_EDITING_V2.md`](START_HERE_EDITING_V2.md) and
> `AGENTS.md`. Current self-hashed contracts remain authoritative over prose.

This file distinguishes concepts that were previously all called a "prior."

## Source distribution

The state at model time zero is selected explicitly:

- `NullSourcePrior`: delta mass at the formal all-NULL graph; or
- `DegreeBoundedCarbonTreePrior`: a target-independent valid alkane tree with
  size sampled from a declared categorical distribution.

The recommended flexible-size experiment samples source size independently
from the empirical training size distribution. Uniform size is a stress
ablation. Training compiles multiple independent source trees per endpoint.
The source marginal is therefore exactly the same distribution used by
ancestral inference, while paired paths include genuine grow and shrink events.

## Source-target coupling and teacher

The source prior does not determine a path. A compiler couples a sampled source
to a data endpoint and supplies a valid rewrite trace. The null teacher uses
causal topology-committed construction. Primitive tree transport peels the
source to one carbon and converts the null teacher's root instruction into an
existing-anchor atom/ring transaction. It never visits the null state.

Atomic subtree Graft (implemented under the backward-compatible internal
`bond_reroute` name) is now active in the topology-revision recipes. Existing
atom deletion/insertion already proves reachability because source atom
identities are disposable; Graft is the derived path-compression/editor rule
that prevents the compiler from teaching delete-to-one-and-regrow. The
`flexible_size_graft` compiler grows a carbon tail when the sampled source is
smaller, or Grafts the source into a canonical path and removes surplus leaves
when it is larger, before Graft/retype/bond/ring reconstruction.

## Operator support

The executable fiber is the state-dependent instruction set, not a prior. Every
candidate is committed through valence, RDKit, and connectivity validation.
The production marked model has complete `ring_system_grow` and
`ring_system_delete` families and no learned `ring_ear_insert` or scalar ring
closure. A ring mark first selects a complete topology, then a legal
acyclic-scaffold match, then predicts its atom labels through normalized
per-site categorical factors.
Thus the topology catalog is proposal support, not a vocabulary of typed
fragments: a pattern learned from benzene can generate and uninstall an unseen
pyridine atom placement. Aromatic placement keys quotient Kekule resonance
aliases. Hard application conditions reject overlap with existing cyclic atoms,
invalid valence, disconnected lowerings, and incomplete cyclic systems.
Semantic resonance aliases share a learned topology score while executable
Kekule lowerings remain available underneath it. A smoothed empirical topology
base measure calibrates rare versus common ring rules; contextual neural scores
still learn the corpus-dependent residual.

## Learned generator

The production neural model learns a continuous-time rate measure over labelled
rule-match events. It factorizes the total hazard, rewrite family, complete ring
topology, application match, and atom labels. Exact tree-subgraph dynamic
programming masks impossible ring topologies without enumerating all matches;
only one selected/teacher topology's matches are materialized. It receives the
current molecule and time, not the paired target or teacher trace.

Pushing this marked process through the chemical rewrite executor gives the
generator over complete molecules: rates of all marked aliases that reach the
same canonical successor are summed. The exhaustive complete-successor model is
retained as a small-state semantic oracle and diagnostic, not used in the
benchmark training loop. Quality evaluation decomposes FCD into additive
ChemNet mean and covariance components, distinguishing mean mismatch from the
under-dispersion that dominated the sibling `rank_d500k` model's residual.

## Sampling

Unconditional sampling is target-free ancestral CTMC simulation. It is not beam
search. The production sampler draws a hazard, rewrite family, template, and
legal site directly, then instantiates and validates only the fired rewrite.
Null and tree checkpoints must be evaluated from the same source prior used in
their training probability path.

## Guidance boundary

Inference-time property guidance is a later controlled tilt of executable
successor rates. The unguided unconditional checkpoint and metrics remain
separate. The planned first guide preserves total hazard and tilts only the
marked successor distribution; a learned value/Doob approximation comes after
the local valid-successor baseline.

## Evidence status

- Tree reachability and train/sample execution: passed.
- Dense factorized training and direct ancestral sampling: passed; 100% valid
  and connected in the end-to-end preflight.
- Size-matched Graft developmental run: selected step-3,000 checkpoint yielded
  2,000-sample FCD 24.35. Graft dominated its actions and it did not prune to one
  atom, but the old ring-ear family produced severe cage/over-fusion pathology.
- Full-system ring replacement: local full suite, structured held-out-label
  generalization tests, serialized flexible-size path compilation, and remote
  teacher-support audits pass. The current Stage-3 recipe uses a flexible-size
  empirical carbon-tree source, Graft transport, semantic whole-ring actions,
  validation every 250 updates, a 500-update warmup, and six evaluations of
  early-stopping patience.
- The resource gate selected A100 + 32 CPU cores + 24 data workers. Its
  200-step run reached validation GM loss 19.0701, family accuracy 0.7436, and
  54.52 examples/s in the final interval. All 16 samples were non-null, valid,
  connected, unique, and novel; the 7/16 small-ring warning is explicitly
  rechecked in the automatic 100-sample checkpoint preview.
- Production training releases the A100 after writing the validation-selected
  checkpoint. A separate retry-safe CPU evaluator owns the final 2,000
  ancestral samples and FCD, so inference never keeps the training GPU alive.
- The commit-`2be9258` production pipeline is active under
  `compose-v4-stage3-flexible-graft-prod-2be9258-v1`; its current stage is CPU
  compilation before the teacher audit and A100 allocation.
- Null-versus-tree matched-budget FCD comparison: pending.
- Recovery/revision objective and guided Pareto evaluation: pending.
