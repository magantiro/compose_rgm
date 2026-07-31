# Atom-restatement neural-orbit audit V1

This is a prospective, train-only, non-authorizing diagnostic. It asks whether
the current six-round additive node encoder can distinguish the legal generic
atom-restatement marks that lead to different canonical molecular successors.
It does not choose or apply a support policy.

The current behavior remains broad-organic generic restatement at every real
site, including cyclic sites, subject to the existing executable mask. Two
explicitly hypothetical views are reported for comparison: restricting cyclic
targets to the declared `ORGANIC_RING_ELEMENTS` set, and retaining acyclic
sites only. That declared set is not authoritative for generic atom
restatement, and it must not be confused with the narrower currently wired
macro ring head.

## Exact certificate

The initial node label is:

```text
(atom type, formal charge, implicit H count, atom topology)
```

Each round refines it with:

```text
(self label,
 multiset of neighbor labels,
 multiset of incident neural edge classes)
```

The neighbor-to-edge pairing is intentionally absent. This matches the current
encoder, which separately sums transformed neighbor states and edge embeddings.
After six rounds, the candidate certificate adds the target atom-valence class.
Equal certificates therefore guarantee equal atom-restatement logits for every
parameter setting of the current information path. They do not assert that all
unequal certificates are distinguishable.

For each certificate class, the audit executes every mark with the production
rewrite system, canonicalizes every successor, and records successor
multiplicity. If a teacher successor occupies `t` of `m` marks in a tied class,
that class contributes a family-conditional ceiling of `t/m`. The overall
ceiling is the maximum such fraction among classes containing the teacher
successor.

## Implementation boundary

The tested core and prospective contract are:

```text
src/compose_v4/experiments/atom_restate_neural_orbit_audit.py
configs/editing_atom_restate_neural_orbit_audit_v1.json
tests/test_atom_restate_neural_orbit_audit.py
```

The core requires an immutable train packed-trace progress address, exact source
and stored-successor states, and an executable teacher action. Row aggregation
retains every address whose current-policy ceiling is below one.

A full-corpus driver is intentionally not included in V1. The exact admitted
Active8 train derivative and its physical shard, manifest, overlay, and
admission identities must be bound before such a driver is scientifically
valid. Running this core over arbitrary legacy train shards would create a
plausible-looking but lineage-invalid number. Adding the driver is the next
bounded task once those exact input bindings are frozen.
