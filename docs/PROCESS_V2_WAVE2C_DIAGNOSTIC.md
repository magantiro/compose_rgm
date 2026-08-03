# Process-V2 Wave 2C: diagnostic checkpoint, NOT the production path

**This branch is a diagnostic record. It must not be integrated as the production
execution path.** The owner has directed a clean reimplementation of the
Process-V2 Active8 and Gate-0 orchestration as a minimal vertical pipeline from
the stable scientific base `0e3f24a`, reusing the executor, legal enumerators,
exact-state codec, canonicalizer, production candidate evaluator, frozen corpus
inputs and scientific registries unchanged.

Nothing remote ran at any point: no Modal job, Gate-0 execution, T1, P50 or
training.

## Why this branch is being retired rather than finished

It accumulated four defects that share one cause: **orchestration was layered on
top of orchestration.** Each layer was individually reasonable and each was
verified against the layer below rather than against the corpus. The four are
recorded below because the rebuild must not reproduce them, not because they are
still live here.

## The four defects, all measured

### 1. Gate 0 published a decision while assigning nothing

Gate 0 handed the index a TRANSFORMED view and asked
`validate_accepted_transition` about it. The index's first check is an exact
field-set match against the raw schema, which a view never satisfies, so it
raised on every transition; Gate 0 caught it, counted a classification failure
and continued.

```
before:  0 structural assignments, 15 classification failures
after :  15 structural assignments, 0 classification failures
```

It survived because the end-to-end assertion was
`decision["structural_result"] in {"PASS", "FAIL"}` -- satisfied by a gate that
does nothing. Fixed at `fbd7edf`; the assertion now requires nonzero
assignments, nonzero observed cells and zero classification failures, and is
proven to fail under the old ordering with "Gate 0 assigned nothing".

### 2. `terminal_assignment_count_is_zero` was unsatisfiable for ANY corpus

`_structural_view` set `terminal` from `successor_is_terminal`, which is true for
the last step of every accepted trace. So `terminal_assignments` equalled the
number of accepted decision-eligible traces and the check was **false for any
non-empty corpus, independent of coverage** -- a guaranteed FAIL that no amount
of cell coverage could clear.

`terminal` describes the teacher's SOURCE progress position, and a published
transition exists precisely because its source has an outgoing step. Corroborated
by the Active8 stage's own vocabulary
(`editing_v2_process_v2_active8_mapreduce.py:1225`, `"terminal": index ==
address.path_length`). Measured: **terminal assignments 10 -> 0**.

### 3. The traversal was one chunk decode per trace, and the earlier measurement hid it

The audit's cost model `decodes = T + 2*sum(L)` reproduces exactly: `10 + 2*15 =
40`. The old cost was **4 decodes per eligible trace** -- ten point lookups plus
thirty validations, because `validate_accepted_transition` is itself a point
lookup.

An earlier measurement of 25 was an artefact of defect 1: every transition was
refused before the index reached a chunk, so the loop looked cheap. **Making
validation correct is what made the cost visible.**

```
chunk decodes            40 -> 5
eligible chunks           5      (tasks holding >=1 eligible trace)
sealed-role chunk reads   0 -> 0
```

### 4. A tautological check, and a declared requirement nothing enforced

`_structural_view` derived each stratum FROM the count, and `_teacher_failures`
compared `stratum_for(count, bins)` against it -- both sides from one input.
Measured: an alias count crossed a bin boundary and zero failures were recorded.
Replaced by one authoritative derivation plus a separate proof that the bins tile
`[1, inf)` exactly once, with six negative controls.

Separately, `require_every_teacher_supported: true` was declared in the frozen
contract and load-bearing enough that Gate 0 refuses a contract omitting it --
and nothing read it. Now enforced through `exact_successor_mark_count`.

## The independent tampering audit: evidence integrity is self-hash only

A read-only audit resealed accepted decision rows and re-ran the full chain. All
four asked-about fields were **accepted**, with the tampered values reaching
published Gate-0 evidence:

| tamper | outcome |
|---|---|
| `raw_mark_count` 279 -> 999 | accepted; cell sum 2790 -> 3510, census 33460 -> 34180 |
| `candidate_totals` +1000, per-action counts untouched | accepted; census 33460 -> 34460 |
| `canonical_successor_key` -> a non-molecule literal | accepted; flows out of both index APIs |
| `supported` true -> false | accepted; still receives a capability cell |
| `exclusion_reason` set on an accepted row | accepted; inert |

Controls confirm the instrument: a no-op reseal reproduced the reference evidence
byte-identically, an unresealed mutation was refused at the reducer, and
`target_state_sha256` / `action_sha256` ARE re-derived and refuse with distinct
messages.

**The guard on all four is the row's own self-hash**, which proves the row was
not edited after sealing and says nothing about whether it was true when sealed.
V1 re-derives the row census from its actions and refuses; the Process-V2 reducer
replaced that with `counts.update(row["candidate_totals"])`.

**Claim boundary, stated because the first framing overstated it:** content
hashes are not signatures. A writer able to replace every artifact and trusted
root cannot be defeated by more hashing. The threat this addresses is
**corrupted, stale, incorrectly produced or incorrectly sealed** evidence -- a
row that was wrong when written. That is a scientific-validity property.

## What genuinely worked, and should be carried forward

**Genuine per-cell chemistry, discovered rather than fabricated.** 22 of 22
registered capability cells and 17 of 17 required, built by enumerating legal
marks through the production fibers, executing them through the exact 8-rule
Active8 registry, encoding with ActionCodecV4 and bucketing by whatever cell the
classifier returned. **Cells are an output, never an input.** ~6.6 s, no
committed cache, no staleness risk. `tests/process_v2_genuine_transitions.py`.

With the terminal fix, that fixture reaches a genuine **PASS**: 22/22 eligible
transitions assigned, 0 classification failures, 22/22 cells observed, 17/17
required, 0 terminal assignments.

**The end-to-end chain test.** Every defect above surfaced through the chain
running, not through per-stage suites. Three of them survived three green suites.

## Measured Gate-0 cost, with its limits stated

Complete Gate-0 pass, corpus-scaling work only, contract preloaded:

```
wall 0.2521 s   CPU 0.2519 s   peak Python allocation 1529.8 KiB
80 resolved rows (all roles), 10 eligible traces, 15 transitions,
5 decodes, 0 successor evaluations, 16.79 ms/transition
```

**Covers:** the all-role resolved-trace scan, role-filtered chunk decode, the
structural view (ActionV4 decode plus capability-cell classification against
exact source/successor states), aggregation.

**Does NOT cover, and therefore NO full-corpus projection is offered:** a
production chunk. `RECORDS_PER_CHUNK = 2` here against production 2048;
fixture-sized molecules understate the per-state RDKit round-trip; the
decision-shard parse is 16 rows per call, not 2048.

## Structural findings the rebuild needs

1. **The Active8/cache task boundary IS the right map boundary, because a task
   carries exactly one `split`.** Roles and chunks do not interleave within a
   task, so role filtering is a pure task-level predicate and held-out sealing
   costs nothing at runtime. **Preserve that property**: the moment a chunk can
   hold two roles, "never opened" becomes unachievable and the only option left
   is decode-and-discard.
2. **The unit of work is the eligible CHUNK, not the eligible task.** Ten
   eligible tasks, only five hold an accepted trace. Plan map tasks from the
   decision shards -- readable without opening any chunk -- so empty containers
   are never launched.
3. **`supported` and `exclusion_reason` do not cross the seam.**
   `ACCEPTED_TRANSITION_FIELDS` omits both, so the audit's `supported` mutation
   is invisible to Gate 0 **by construction**; only the index can refuse it.
   Publish them on the accepted transition, or no downstream gate can enforce the
   clause its own contract declares.
4. **A consumer must not re-validate what a stream already validated.** Thirty of
   the forty old decodes were validation.
5. **Reducer order-independence must be stated, not inherited.** Three inventory
   hashes are running SHA-256 over `iter_resolved_traces()` order; under a
   map/reduce that is deterministic only if the reducer folds per-task digests in
   `reduction_order`.
6. **Read decision metadata for every role; decode only eligible roles.** Decision
   shards are separate artifacts from chunk caches, so a complete census and
   complete sealed-role hashes cost nothing.
7. **`iter_accepted_transitions()` has no role filter.** Measured: it decodes 20
   chunks and yields 60 transitions, **45 of them held-out**. Consuming it as-is
   weakens "never opened" to "never counted".

## State at this checkpoint

Focused: `tests/test_process_v2_end_to_end_gate_zero.py` **19 passed**.
`tests/test_process_v2_gate_zero.py` + `..._stage_modal_surfaces.py`: **76 passed,
7 failed.** The seven, with causes:

- `test_the_runner_does_not_import_the_v1_concrete_decision_source` -- the known
  classifier-extraction blocker;
- `test_a_transition_the_index_refuses_becomes_a_typed_receipt` -- unreachable by
  design now that a refusal inside the stream raises;
- `test_two_v1_tasks_may_share_a_trace_id_and_resolve_independently`,
  `test_gate_zero_never_opens_a_sealed_trace`,
  `test_the_adapter_refuses_a_raw_transition_it_cannot_view` (x2),
  `test_every_frozen_requirement_is_detected_on_a_real_view[has_exact_successor_support]`
  -- stand-in lag behind the bulk-stream protocol.

**No full repository suite was run at this head, and none is claimed.**

Both scientific identities are unchanged: V1
`6c4721f0dd37132aae657e7aa5f1bfc01cef270662f228171c4587eb7dd48491`, V2
`0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd`.

## Not completed here

The classifier extraction and its binding cascade, and the Active8 write-time
evidence derivation, were both in flight when this branch was retired. Their
findings are in the workstream branches `w2c/classifier` and `w2c/evidence`.
