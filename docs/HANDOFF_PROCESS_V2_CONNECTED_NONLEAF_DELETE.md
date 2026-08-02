# COMPOSE Process V2 connected-nonleaf atom deletion

## Status and authority

This is an implementation handoff, not training authority.

The implementation agent must read `AGENTS.md` completely before inspecting or editing code. The
project-specific contract and any stricter self-hashed contract remain authoritative.

Implementation may begin only after the user explicitly records this prospective decision:

> I approve COMPOSE Process V2 expanding the learned marked fiber to executor-verified,
> charge-preserving, non-aromatic, non-articulation connected-nonleaf atom deletions, invalidating V1
> downstream artifacts.

The approved work is restricted to the semantic-process implementation, its prospective decision
artifact, compatibility/rebinding proof infrastructure, tests, and documentation. It does not
authorize Active8 materialization, Gate 0, T1, P50, later pilots, a long editing run, or a de novo run.

## Scientific decision

Adopt Process V2 for the source-conditioned editing lane.

For `atom_delete`, preserve existing root, singleton, and leaf behavior. Additionally admit an active
atom as a connected-nonleaf deletion candidate only when all of the following hold:

1. the deleted atom is a real element under the authoritative element predicate;
2. it is non-aromatic under the frozen production representation;
3. it is not a graph articulation point;
4. the operation preserves the current charge policy;
5. the unchanged production atom-delete executor accepts the operation;
6. the resulting exact persistent-slot state is connected, valence-valid, canonicalizable, and within
   the declared broad-organic, at-most-40-active-atom support.

The executor remains the legality authority. Do not duplicate a weaker approximate valence test and
call it executor equivalence.

Continue to exclude aromatic connected-nonleaf deletion. The current executor can produce
representation-sensitive open-chain successors from Kekule encodings. Aromatic deletion requires a
separate future semantic decision and resolver.

## Why this is the principled choice

The V1 dense production mask excludes every cyclic atom before executor validation. This creates a
real support mismatch: valid non-aromatic, non-articulation deletions are legal under the production
executor and independent fiber, but the learned model assigns them zero probability.

Verified examples include:

- deleting any atom of cyclohexane, for which the unchanged executor yields pentane;
- deleting a non-articulation atom from a saturated oxygen-containing ring, for which the unchanged
  executor yields a valid connected successor;
- excluding benzene-like aromatic deletion, whose current successor is representation-sensitive;
- excluding articulation-point deletion and invalid-valence successors.

Process V2 supports direct, valid size and topology correction. The narrower V1 alternative would
require some transformations to use ring opening followed by leaf deletion, increasing path length and
weakening E3 size adaptation, E4 topology adaptation, dynamic retargeting, and finite-budget Pareto
search. This decision does not add arbitrary multi-neighbor insertion and does not claim one-step
inverse closure.

## Frozen non-goals

Do not:

- change atom-delete executor semantics;
- change persistent-slot identity or canonicalization;
- change formal-charge policy;
- admit aromatic connected-nonleaf deletion;
- add multi-neighbor atom insertion;
- enable `ring_system_delete` or `ring_system_grow`;
- change any other Active8 operator semantics;
- weaken required semantic-cell gates;
- relabel V1 artifacts as V2;
- reconstruct exact states from SMILES;
- launch Modal jobs or training;
- use inspected final-test aggregates for any design or threshold decision.

## Required implementation sequence

### Phase 1: record the prospective support decision

1. Preserve the V1 semantic-process contract and identity unchanged.
2. Add a new self-hashed Process V2 support decision and semantic-process contract. Do not edit a V1
   identity in place.
3. Bind at minimum:
   - precise connected-nonleaf admission semantics;
   - unchanged executor implementation identity;
   - exact atom-delete mask/enumerator implementation identity;
   - action codec, persistent-slot, canonicalizer, charge-policy, vocabulary, and size identities;
   - explicit V1 downstream invalidation;
   - explicit absence of training and experiment authority.
4. Use deterministic serialization and validate the physical file hash and semantic self-hash.
5. Give the new process a distinct schema/version/identity. Historical V1 records remain readable only
   under V1 identity.

### Phase 2: align the learned candidate mask with executor legality

1. Inspect `src/compose_v4/model/factorized_tracelet_rate_model.py`, especially
   `_graph_application_masks` and construction of `atom_delete_mask`.
2. Replace the V1 blanket cyclic-atom exclusion with an exact executor-verified V2 candidate decision.
3. Reuse the production executor/legal-enumerator path wherever possible. If the current batch interface
   cannot carry exact CPU-derived masks without duplicating scientific logic, stop and report the design
   boundary before introducing an approximation.
4. Keep masks boolean, deterministic, slot-addressed, and independent of canonical SMILES atom order.
5. Ensure inference, training, direct collation, and multiworker collation consume the same V2 mask.
6. Bind the current process identity into batches, checkpoints, evaluators, and artifacts. Reject mixed
   V1/V2 batches or checkpoints.

### Phase 3: add bounded compatibility and regression evidence

At minimum add behavior-level tests for:

- non-aromatic carbocycle connected-nonleaf deletion is exposed by the model and accepted by executor;
- saturated heterocycle connected-nonleaf deletion is exposed when executor-valid;
- aromatic connected-nonleaf deletion remains excluded;
- articulation-point deletion remains excluded;
- invalid-valence and disconnected successors remain excluded;
- existing leaf, singleton-to-null, and root-boundary behavior remains unchanged;
- SCAR and non-element slots are never admitted;
- every admitted delete executes to the exact expected persistent-slot successor;
- direct and multiworker collation produce identical masks;
- model mask and an independently invoked production executor agree on a bounded representative panel;
- other seven Active8 masks and logits are unchanged on a frozen fixture;
- Process V2 identity differs from V1 deterministically;
- V1 artifacts cannot masquerade as V2 after self-rehashing;
- missing or mismatched Process V2 identity fails loudly.

Use the independent oracle only in bounded tests. Production experiment scripts must not reconstruct the
successor kernel independently.

### Phase 4: implement the V1-payload to V2-proof rebind tool

The completed V1 migration payload may be reused only as immutable chemical data under an explicit V2
compatibility proof. Do not relabel records byte-for-byte.

Implement restart-safe, content-addressed range tasks that:

1. validate each V1 record, receipt, schema, hash, and exact process identity;
2. codec-roundtrip every semantic action;
3. replay every teacher transition through the unchanged executor;
4. require exact persistent-slot successor-array equality and canonical-key agreement;
5. evaluate the V2 atom-delete candidate mask at every progress state against an independent bounded
   legality oracle;
6. preserve source, split, lane, provenance, and evidence-profile identities;
7. record all unsupported teachers and mismatches with reason codes;
8. emit new V2 receipts, manifests, semantic hashes, physical hashes, and completion identity;
9. publish only atomically after complete validation;
10. refuse to publish on any mismatch and never fall back to partial trace admission.

The reducer must require a complete deterministic task range, reject duplicates and gaps, and emit no
training authority. Do not launch the proof scan in this implementation task.

## Required verification

Run the narrowest tests while iterating, then before handoff run:

```text
.venv/bin/python -m pytest <all focused Process-V2 and rebind tests>
.venv/bin/python -m ruff check <all touched Python files and tests>
git diff --check
```

Run a relevant broader model/data suite if it is bounded locally. Do not claim the repository-wide suite
passes unless it was actually run and its exact pass, fail, and skip counts are recorded.

Inspect every generated contract and artifact, the complete diff, and `git status` before committing.

## Commit and delivery discipline

Work only in:

```text
/private/tmp/compose-process-v2-atom-delete
```

Branch:

```text
codex/editing-v2-process-v2-atom-delete
```

Expected starting commit:

```text
d89079d5bd7eaf790e3c6a2369692d43fc0d6e93
```

Do not edit the dirty main checkout and do not touch running Modal jobs.

Use separate, reviewable commits for:

1. prospective Process V2 contract and identity;
2. exact atom-delete candidate-mask implementation and tests;
3. proof-bound V1-to-V2 rebind infrastructure and tests;
4. documentation or generated deterministic artifacts when separation improves reviewability.

Use repository-style commit messages. Do not mention an AI assistant in commits or pull-request text.
Do not push until focused verification is complete. At handoff, report commits, tests actually run,
unresolved failures, changed identities, and whether the branch is pushed.

## Stop conditions

Stop and report rather than improvising if:

- exact executor legality cannot be represented in the batch mask without changing executor semantics;
- aromaticity or charge behavior is ambiguous;
- a supposedly unchanged operator changes on a frozen fixture;
- V1 and V2 identities cannot coexist without ambiguity;
- exact transition replay differs for any payload claimed compatible;
- implementing the rebind requires changing split or provenance identity;
- any requested action would authorize or launch a scientific job.
