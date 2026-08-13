# STATUS — hard structural constraints (Lane 6)

**Status:** Stage 0 complete. Stage 1 half complete, half blocked. Stage 2 design
frozen, **not run**. Ready for handoff.

**Branch:** `codex/compose-constraints-hard` **Base commit:** `f6146d7`

**Held-out data opened:** **NO.** Held-in `training_source_keys` only;
`reserve_source_keys` is deleted from the loaded payload before any sweep.

**Currently running:** nothing. No Modal launch, no GPU, no claim-bearing
compute. Nothing installed into the project environment.

---

## Last completed gate

**Stage 0 — executor semantics.** Verdict:
**`LABELED_SUBGRAPH_PRESENCE_INVARIANT`**. `IDENTITY_INVARIANT` is not provable
and is barred. Four probes, all reproducible on CPU without the checkpoint:
`docs/workstreams/constraints-hard/probes/identity_probe.py`.

**Stage 1a — model-free protected-core census.** `SMOKE_HELD_IN`. Across the
full 96,094-source held-in pool the Bemis–Murcko core is a **median 78.95%** of
heavy atoms, leaving a **median 5** editable heavy atoms; **22.03%** of sources
are eligible under Lane 2's reused bands.

## Blocked

**Stage 1b — fiber census.** Gate-0 authentication fails locally on the
already-diagnosed non-portable `source_index_sha256`, and the model is
constructed *from* the authenticated source, so there is no partial local path.
Reproduced with `probes/gate0_local_authentication_repro.py`; **not** routed
around. Needs a Modal run and the lead's per-run authorization.

## Next action — a scope decision, not more work

**The lead must decide whether Experiment A's overlap with Lane 2's Stage A is
the intended division of labour.** Lane 2 already ran this arm structure at
n = 6 and recorded the primary contrast under `endpoint_only_return_failure`.
This lane's recommendation is in `PROTOCOL.md` §13: fold the yield contrast into
Lane 2's existing harness rather than standing up a parallel one, and keep this
lane's distinct contribution to the semantics freeze, the census, and the
baseline matrix.

## Deliverables

| file | status |
|---|---|
| `CONSTRAINT_SEMANTICS.md` | Stage 0 verdict + evidence |
| `SCAFFOLD_FEASIBILITY.md` | Stage 1a results, Stage 1b blocker |
| `PROTOCOL.md` | Stage 2 design + costed plan |
| `BASELINE_TASK_MATRIX.md` | primary-source qualification |
| `SAME_LAB_LINEAGE.md` | framing freeze |
| `DECISION_LOG.md` | every material decision |
| `HANDOFF.md` / `handoff.json` | handoff packet |
| `probes/` | 2 reproducible probes |
| `scripts/constraints_hard_scaffold_census.py` | census producer |
| `src/compose_v4/experiments/constraints_hard_mask.py` | injection point, `DESIGN_ONLY` |
| `tests/test_hard_scaffold_constraint.py` | 13 tests, all passing |
| `diagnostics/constraints_hard_scaffold_census.json` | `SMOKE_HELD_IN` |

## Top risks

1. **Overlap with Lane 2** — needs a lead decision (HIGH).
2. **Only trivial edits outside the core** — median 5 editable atoms (HIGH).
3. **Applicability is 22.03%** — stop rule already triggered; disclosed, not
   hidden.
4. **`fragment_smarts` reuse** depends on the unmerged
   `codex/compose-pathwise-constraints`.
