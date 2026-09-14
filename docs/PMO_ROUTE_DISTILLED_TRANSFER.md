# PMO Route-Distillation Export and Transfer Preparation

## Problem and output

The locked Practical Molecular Optimization (PMO) development artifacts contain
exact, answer-known source-to-endpoint programs, but their task labels, endpoints
and bindings cannot enter a generic runtime controller. This revision exports one
training-only dataset whose prediction targets are generic primitive action
families, structural action roles, relative created-handle dependencies and the
same molecular and stage descriptors used by the T4 route-distilled policy.

The output is supervision for a future shared proposal policy. It is not a fitted
policy, optimizer result or autonomous route-recovery result.

## Frozen source census

The exporter admits exactly:

- 75 complete programs from the target-program wave;
- 45 complete programs from property-program wave curriculum v2;
- 30 complete programs from the property-panel refinement;
- 20 complete programs from formula/median curriculum v2;
- 11 complete programs from the Perindopril winner curriculum; and
- five exact Perindopril public-winner witnesses.

This is 181 complete programs plus five witnesses. The 181 programs contain 5,997
primitive transitions and 178 distinct endpoints. The five witnesses contain 215
additional transitions. Two exact program traces duplicated between the property
and panel sources are retained in the provenance manifest but collapsed before
supervision weighting.

| Task | Route instances | Primitive transitions | Complete route at most 32 | Long, local-only |
| --- | ---: | ---: | ---: | ---: |
| Albuterol similarity | 15 | 463 | 15 | 0 |
| Celecoxib rediscovery | 15 | 461 | 15 | 0 |
| GSK3B | 25 | 728 | 24 | 1 |
| C7H8N2O2 isomers | 10 | 131 | 10 | 0 |
| JNK3 | 25 | 1,262 | 0 | 25 |
| Median1 | 10 | 175 | 10 | 0 |
| Mestranol similarity | 15 | 415 | 15 | 0 |
| Perindopril MPO | 16 | 680 | 0 | 16 |
| QED | 25 | 736 | 18 | 7 |
| Thiothixene rediscovery | 15 | 638 | 0 | 15 |
| Troglitazone rediscovery | 15 | 523 | 0 | 15 |

Counts in this table retain route multiplicity. The training export separately
collapses two exact duplicates, leaving 184 traces and 6,143 decisions.

Older property/formula curricula, the derivative winner-imitation result, route
support re-encodings, autonomous PMO trajectories and all live PMO/T4 runs are
excluded. Their exclusion is part of the frozen manifest rather than an implicit
file-system choice.

## Evidence and support boundaries

Every admitted route must have one more exact persistent-slot state than action,
replay through the production semantic rewrite system, remain within 48 slots and
40 active atoms, and end at its recorded exact endpoint. All source envelopes and
byte hashes must match the sealed contract.

The future runtime retains the existing limit of 32 primitives, eight blocks and
40 active atoms. The audit reports complete routes at or below 32 primitives
separately from longer routes. A longer route may provide explicitly labeled local
primitive supervision, but it cannot count as a runtime-supported complete route.

The current generic recognizer maps all 6,212 admitted PMO transitions to primitive
fallback decisions and recognizes no compound stage. This is a negative result and
a hard semantic boundary. PMO rows must not determine high-level module-count or
multi-primitive binding targets unless a later, task-independent stage-recognition
revision is separately authorized and validated.

## Generic supervision schema

For each deduplicated transition, export:

- the 517-dimensional `molecule_features` vector of the exact parent;
- the option label from the frozen generic descriptor menu;
- the 18 fields in `STAGE_DESCRIPTOR_NAMES`;
- the exact generic executor and model family;
- action parameters with every atom address replaced by a structural role;
- whether each operand is an existing source atom or a handle created earlier in
  the same route;
- created-handle ordinal and dependency lag, never its persistent slot; and
- whether the containing complete route is inside the 32-primitive runtime support.

Training provenance is stored in a separate section keyed by dataset row index. It
may name the task, source artifact, curriculum member, lineage and primitive index.
It does not duplicate endpoint strings, source graphs, assignments or executable
actions. The generic row section contains none of those provenance labels.

No exported artifact is a runtime checkpoint. A future checkpoint must use the
existing `route_distilled_program_actor_v1` schema and contain only learned numeric
parameters, generic option/reference data, generic structural prototypes,
module-count probabilities and a hashed training identity.

## Grouping and weights

Task-family labels are frozen before fitting:

- `rediscovery`: Celecoxib, Thiothixene and Troglitazone;
- `similarity`: Albuterol and Mestranol;
- `bioactivity`: GSK3B and JNK3;
- `druglikeness`: QED;
- `formula`: C7H8N2O2 isomers;
- `multi_property`: Median1; and
- `mpo`: Perindopril MPO.

Programs sharing an exact source and declared base action trace form one lineage.
The winner curriculum has member-specific bound slots, so its lineage hashes the
common abstract input topology and base marks instead. Each saved Perindopril
witness is its own lineage. Sibling suffix variants remain together. Exact
duplicate traces are reconciled before rows are emitted. The frozen census has 18
lineages.

The PMO-only export assigns each decision

\[
w_i = \frac{1}{N_{domain}}
      \frac{1}{N_{family\mid domain}}
      \frac{1}{N_{lineage\mid family}}
      \frac{1}{N_{decision\mid lineage}}.
\]

The weights must sum to one globally and within each hierarchy level to numerical
tolerance. A future combined export must give T4 and PMO equal domain mass, then
apply the same family, lineage and decision balancing within each domain. Random
row or endpoint splitting is prohibited because sibling routes share long exact
prefixes.

## Zero-oracle acceptance gates

1. **Input and manifest:** every source byte hash and envelope self-hash matches;
   admitted and excluded assets exactly match the contract.
2. **Replay:** all 186 route instances have contiguous exact ancestry and replay
   precision 1.0; duplicate reconciliation is explicit.
3. **Descriptors:** every generic row has finite, fixed-width state/stage/action-role
   descriptors and no absolute atom address.
4. **Support:** route counts and transitions are reported separately for complete
   routes at or below 32 primitives and long-route local-only supervision.
5. **Balance:** domain, family, lineage and decision factors are recorded and the
   normalized weights satisfy the frozen hierarchy.
6. **Leakage:** generic rows are scanned to exclude task names, route identifiers,
   endpoints, SMILES, assignments, source graphs, target maps and executable
   programs. Training provenance is explicitly non-runtime.
7. **Negative result:** zero compound PMO stages is recorded, and no actor,
   module-count target or binding-prototype target is emitted.

## Prohibited interpretations and next revision

These routes are answer-known, panel-informed or winner-informed development
evidence. They are not held-out PMO evidence and scores are not pathwise value
labels. Terminal scores remain provenance only and do not weight imitation rows.

Training the PMO-only and balanced T4+PMO actors is a later revision. It must keep
their checkpoints and results distinct, compare against the existing T4-only actor
at fixed lineage-held-out diagnostics, scan runtime payloads for forbidden teacher
content, and require a separate authorization before any oracle or docking call.
