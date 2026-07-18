# Structured tree source and valid-state transport

## Decision

The null graph remains the checkpoint-compatible baseline. The new transport
mode samples a target-independent, degree-bounded alkane tree and learns a
valid-to-valid rewrite process into the data distribution. This is structured
chemical noise, not a user-provided scaffold: the source need not survive as a
subgraph and can be rerouted, retyped, grown, shrunk, or replaced.

The source distribution is now an explicit object, separate from the empirical
action-rate baseline, the typed ring proposal catalog, and any endpoint
size/property profile. `DegreeBoundedCarbonTreePrior` samples a declared size
categorical followed by a capped Prüfer process. Every sample is a connected,
neutral, single-bonded carbon tree with maximum degree at most four.

## Primitive reachability versus path compression

A standalone deletion of a tree edge is illegal because it exposes two
components. This does **not** freeze the source topology, because source atoms
are unlabeled disposable noise. A terminal atom can be deleted while the
remaining molecule stays connected, and an equivalent carbon can be inserted
elsewhere. Repeating leaf deletion and regrowth can transform any source tree
through the retained one-carbon tree without exposing a disconnected state.

The primitive reachability baseline therefore uses only the existing
`atom_delete` and `atom_insert` rules and supports different source/target
sizes. The quality-bearing topology-revision path uses Graft rather than
discarding the source topology.

For a path-compression and editing ablation, transport mode also defines an
atomic subtree graft (the current internal class is named `BondReroute`):

```text
subtree_graft(old bridge a-b, new cross-component bond u-v, order)
```

The executor removes `a-b` and inserts `u-v` in one transaction. The old edge
must be a bridge, the new endpoints must lie in the two components induced by
the conceptual cut, and the complete successor must pass hydrogen, valence,
RDKit, and connectivity checks. Re-adding the same edge is an identity and is
forbidden. This is a tree edge exchange / subtree reattachment that preserves
an entire component. It is a derived efficiency instruction, not a requirement
for reachability or universality. The paper-facing term is **subtree graft**;
`bond_reroute` remains a backward-compatible implementation identifier.

## Flexible-size Graft compiler

`flexible_size_graft` draws the tree size independently from the declared
source prior, so it does not leak the paired target size and exactly matches
the ancestral source marginal. It then resizes without leaving the molecular
state space:

- if the source is smaller, it inserts a carbon tail one valid leaf at a time;
- if the source is larger, bounded Grafts turn it into a canonical path whose
  surplus slots are removable leaves, then deletes exactly those leaves; and
- once cardinality matches, the standard Graft/restate/bond/ring compiler
  reaches the exact target.

This is not a monotone construction process: the trained action space still
contains grow, shrink, retype, bond revision, Graft, and coordinated ring
events in both the model and target-free ancestral sampler.

Standalone `bond_delete` remains useful for non-bridge edges, especially ring
opening during revision. It is not a tree-rewiring instruction.

## Reachability gate

`scripts/audit_tree_source_reachability.py` samples several independent trees
per held-out molecule and reports exact endpoint reachability, valid rewrite
length, operator counts, and compiler failure classes.

The primitive compiler supports size-mismatched sources by peeling to one
retained carbon and regrowing. The equal-size Graft audit isolates topology;
the flexible-size Graft audit jointly tests topology and cardinality. A
stationary local size-jitter coupling is a lower-variance follow-up to the
implemented independent-size coupling, not required for source-marginal
correctness.

## Ring commitment from a tree scaffold

The production Graft compiler transports the source to a target spanning-tree
scaffold while leaving every future cyclic-system member as neutral carbon.
It then groups each connected component of target non-bridge edges and emits
one `ring_system_grow` per complete system. The action jointly installs all
cycle-rank edges, target atom labels, executable bond orders, and semantic
aromatic edges. No `ring_ear_insert` or scalar `bond_insert` appears in the
learned path.

The flexible-size compiler performs its target-independent grow/shrink phase
before the same Graft-plus-full-system suffix. Full-system deletion is the exact
inverse and restores a valid carbon-tree precursor.

Tree-source quality training is gated on exact dataset reachability and teacher
support in the topology-committed inference fiber.

## Paper claim if the gate succeeds

> COMPOSE learns a continuous-time generator that transports a tractable
> structured distribution of valid molecular trees into the data distribution
> through executable, chemistry-preserving rewrite instructions.

The null-source result remains an ablation. The structured source becomes the
headline only if it matches or improves unconditional distribution quality and
measurably exercises primitive deletion/regrowth, retyping, and size changes.
