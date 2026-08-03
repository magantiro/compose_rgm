# Process-V2 Wave 2B: runnable Active8 and Gate 0

Checkpoint for Codex review. **Nothing remote ran.** No Modal job, corpus
materialization, Active8 execution, Gate 0, T1, P50 or training run was started
at any point.

## 1. Outcome

The wave's stated outcome is met:

> A deterministic end-to-end local fixture runs
> `cache -> rebind -> Active8 decisions -> Process-V2 index -> Gate 0`

```
tests/test_process_v2_end_to_end_gate_zero.py   4 passed
```

**One blocker stands between this and the execution pass**, and it is a single
owner decision rather than a set of defects. See §5.

| | |
|---|---|
| branch | `codex/editing-v2-process-v2-wave2b-active8-gate0` |
| base | `0e3f24a` |
| head | `a90484e` (34 commits) |
| worktree | `/private/tmp/compose-process-v2-wave2b-active8-gate0` |

## 2. What was built

| deliverable | module | state |
|---|---|---|
| A: cache-fed source | `editing_v2_process_v2_active8_source.py` | complete |
| B: Active8 policy / runtime / mapreduce / index | four `editing_v2_process_v2_active8_*` modules | complete |
| C: Gate 0 | `editing_v2_process_v2_gate_zero.py` | complete except §5 |
| D: two unlaunched Modal surfaces | `run_process_v2_active8_decisions_app.py`, `run_process_v2_gate_zero_app.py` | complete |
| seam | `editing_v2_process_v2_active8_interfaces.py` | complete |

The V1 loader edge is gone: `resolve_editing_v2_semantic_active8_sources` has no
importer among Process-V2 modules, and the strict `xfail` that recorded that
boundary is retired to an ordinary passing gate.

## 3. Verification

```
end-to-end chain                     4 passed
all Wave-2B test files             185 passed, 7 failed
identities V1 6c4721f0.. V2 0c938177..   unchanged
frozen files                         byte-identical to base
chain verifier                       AGREES
ruff, touched files                  clean
ruff, repository-wide                64 = base 64
git diff --check                     clean
```

**The complete repository suite has NOT been run at head.** Base `0e3f24a` is
`16 failed, 3417 passed, 1 skipped, 1 xfailed`, measured in an isolated
worktree. A head run was not taken because the 7 known failures below make the
comparison uninformative until they are resolved. Do not read the numbers above
as a full-suite result.

## 4. What the end-to-end test found, and why it exists

Three stages were built in parallel against a frozen seam. All three suites were
green. **The chain did not join.** Three separate defects, each invisible to
every per-stage suite:

1. **The seam froze the join key and the categories but not the ROW.** The
   producer published `split` / `category`; the consumer required
   `partition_role` / `rejection_category`. Fixed by freezing
   `RESOLVED_TRACE_ROW_FIELDS` and having the index build its row set FROM the
   frozen minimum rather than alongside it.
2. **The seam never froze the TRANSITION.** Gate 0 required
   `capability_cell_id`, `family_context`, `assignment_sha256` and three strata
   that no index publishes. The index publishes raw evidence, which is the
   correct split: §5 gives the Active8 stage the job of preserving counts, §6
   gives Gate 0 the job of evaluating cells and publishing strata. Gate 0 now
   performs that evaluation.
3. **Gate 0's classification import broke its own V1 isolation** -- §5 below.

A fourth was caught by review before integration: `accepted_transitions_for`
took a bare `trace_id`, contradicting the join key the same file froze. A trace
id is unique within a V1 task and nothing guarantees it across tasks, so a
bare-id lookup merges two traces into an answer that merely looks longer.

**A seam that names a concept but not the field carrying it has not named it.**
That is the lesson, and it cost three rounds.

Two properties are now pinned that no per-stage suite could express:

* no raw packed shard is opened after cache publication, measured with a
  positive control proving the recorder can see that read when it happens;
* Gate 0 does not decode one chunk per trace -- **25 decodes for 40 accepted
  traces**. The index's own docstring warns that its point lookup in a loop
  "would reinstate exactly the triangular rescan the chunk cache exists to
  remove"; whether Gate 0 does that is a property of the join.

## 5. THE BLOCKER: the pure classifiers are trapped behind a V1 import

Gate 0 must classify each transition into a capability cell and stratify its
counts. The functions that do this -- `classify_action_family_context`,
`_stratum`, `_action_audit_axes`, `_ring_system_topology`,
`_component_after_cut`, `_state_graph`, `_cycle_rank`, `_real_atom_count`,
`_minimum_cycle_length_for_edge`, `CountBin` -- are **pure graph math with zero
V1 coupling**, verified by AST scan, not by inspection.

They live in `src/compose_v4/data/editing_v2_semantic_capability_cells.py`,
which imports the V1 concrete decision source at line 59 for the one function
that genuinely needs it. So Gate 0 importing them acquires a V1 edge, and
`test_the_runner_does_not_import_the_v1_concrete_decision_source` correctly
refuses it.

**Extracting them is the right fix and is an authorized two-lineage identity
move, which is why it was not taken here.** That file's physical hash is pinned
in two frozen lineages:

* `configs/editing_v2_semantic_capability_cells_v1.json:15`
  (`classifier_implementation_sha256`) -- the **V1** registry;
* `configs/editing_v2_process_v2_capability_cells.json:189`
  (`classifier_implementation`) -- the Process-V2 chain.

The extraction was built and reverted. It works: the leaf module imports
cleanly, the original re-exports with no duplication, and Gate 0's V1 edge
disappears. It then invalidates both pins, and the Process-V2 chain regenerates.
Wave-2B scope forbids adapting a V1 artifact, and this repository's own
learnings record that re-pinning a content-addressed identity has a transitive
blast radius and that the naive method produces a silent stale pin.

**Decision needed:** authorize the extraction plus the two-lineage re-pin, or
name a different owner for the classification.

## 6. The 7 known failures

**1 is the blocker above.**

**6 are stand-in fixture shape.** `tests/test_process_v2_gate_zero.py` and
`tests/test_process_v2_stage_modal_surfaces.py` publish PRE-CLASSIFIED
transitions; production now publishes raw evidence and classifies at the
boundary. 26 of the original 32 were resolved by making the view an explicit
parameter with a guard asserting the production default is the real adapter. The
remaining six reach production through the Modal app, which correctly uses the
real adapter, so they need genuine molecular states their synthetic fixture
cannot fabricate.

They are left failing rather than papered over. **The stand-in was green against
a shape no real index produces** -- which is the same defect class as everything
in §4, and worth fixing properly rather than by substitution.

## 7. Deferred findings

1. `accepted_transitions_for` is a point lookup costing one chunk decode.
   `iter_accepted_transitions()` exists for a full pass. Gate 0 currently uses
   the point lookup and measures within bounds at fixture scale (25 decodes / 40
   accepted); at production scale the bulk API should be used.
2. The Active8 stage names its policy/mapreduce/index modules as constants in
   both Modal apps, resolved at execution and refused by exact missing name.
3. Two `local_entrypoint`s per app means `modal run` requires `::main` or
   `::describe`.
4. Untouched by scope: `cache_materialized: True`, the verifier near-miss
   threshold, the Gate-0 model contract's bare process hash, RingCore/E6
   staleness.

## 8. Process notes for the next wave

* **The base changed under us.** `625473e` repaired the oracle-import false
  positive and broke `test_the_owning_oracle_gate_exists_and_covers_the_
  process_v2_directories`, which verified it by substring-matching its source.
  Repaired here by reading the oracle from the constants the gate declares. Base
  failures went 16 -> 15.
* **A count comparison would have missed it.** W2A and W2B bases both show 16
  failures; the SETS differ by a one-for-one swap.
* Two coordination errors of mine are recorded so the next wave avoids them: the
  eight declared families are `ACTIVE8_FAMILIES` (`editing_corpus_contract.py`),
  NOT `ACTIVE8_EXECUTOR_RULES`; and agent corrections must be addressed by
  verified ID, not by remembered role.
