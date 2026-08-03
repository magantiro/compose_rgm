"""Every shape that crosses a stage boundary in the Process-V2 vertical pipeline.

WHY THIS IS COMPLETE AND NOT MINIMAL
------------------------------------
The previous attempt froze a protocol, the join key and two rejection categories,
and left the resolved ROW and the accepted TRANSITION unstated.  Three stages
were then built in parallel, each green against the contract written for it, and
the chain did not join: the producer spelled the partition role ``split`` while
the consumer required ``partition_role``; the consumer required a capability cell
and three strata that no producer emitted.  A seam that names a concept but not
the field carrying it has not named it.

So this module names every field of every artifact that crosses a boundary, and
nothing else.  It holds no machinery, computes nothing, and owns no artifact.

THE PIPELINE
------------
    plan -> Active8 map (one task per source chunk) -> Active8 reduce + sentinel
         -> Gate-0 map (one task per nonempty train-role shard) -> Gate-0 reduce

CLASSIFICATION IS DERIVED ANNOTATION
------------------------------------
Active8 assigns ``capability_cell_id`` at WRITE time, while the exact source
state, the action, the candidate fiber and the executor result are already in
memory.  It is an ANNOTATION: it must never affect admission.  The admission
decision is a function of the candidate evidence alone, and a row records both
so a reader can check that independence rather than trust it.

The raw structural axes are stored BESIDE the cell so a consumer can audit the
assignment without the classifier.  That is what lets Gate 0 aggregate
authenticated evidence while importing no classifier, reconstructing no model,
decoding no molecular state and re-enumerating no successor fiber.
"""

from __future__ import annotations

# ---- Schemas and versions ----

PIPELINE_NAMESPACE = "compose.data.editing_v2_process_v2_pipeline"

PLAN_SCHEMA = f"{PIPELINE_NAMESPACE}.plan"
PLAN_SCHEMA_VERSION = 1

ACTIVE8_TASK_SCHEMA = f"{PIPELINE_NAMESPACE}.active8_task_result"
ACTIVE8_TASK_SCHEMA_VERSION = 1

ACTIVE8_COMPLETION_SCHEMA = f"{PIPELINE_NAMESPACE}.active8_completion"
ACTIVE8_COMPLETION_SCHEMA_VERSION = 1

SENTINEL_SCHEMA = f"{PIPELINE_NAMESPACE}.active8_release_sentinel"
SENTINEL_SCHEMA_VERSION = 1

GATE_ZERO_TASK_SCHEMA = f"{PIPELINE_NAMESPACE}.gate_zero_task_result"
GATE_ZERO_TASK_SCHEMA_VERSION = 1

GATE_ZERO_DECISION_SCHEMA = f"{PIPELINE_NAMESPACE}.gate_zero_decision"
GATE_ZERO_DECISION_SCHEMA_VERSION = 1

#: No stage in this pipeline authorizes anything, in either outcome.
PIPELINE_STATUS_NO_AUTHORITY = "PROCESS_V2_PIPELINE_EVIDENCE_ONLY_NO_AUTHORITY"

# ---- Addressing ----

#: A cache row joins to its rebind decision by the V1 task identity and the
#: GLOBAL entry index.  Never by chunk position: position is a function of the
#: chunk size, so the same corpus chunked differently holds the same rows at
#: different offsets and a positional join silently pairs the wrong decision.
JOIN_KEY_FIELDS: tuple[str, ...] = ("v1_task_identity_sha256", "entry_index")

#: Addressing ONE trace needs the trace id as well.  A trace id is unique within
#: a V1 task and nothing guarantees it across tasks, so a bare-id lookup merges
#: two traces into an answer that merely looks longer than it should be.
TRACE_KEY_FIELDS: tuple[str, ...] = (*JOIN_KEY_FIELDS, "trace_id")

#: An action within a trace.
ACTION_KEY_FIELDS: tuple[str, ...] = (*TRACE_KEY_FIELDS, "step_index")

# ---- Candidate evidence: what admission is a function of ----

#: Emitted per action by the production evaluator while the fiber is in memory.
#: `supported` and `exclusion_reason` are PUBLISHED, unlike the previous
#: attempt, where they never crossed the seam and Gate 0 therefore could not
#: enforce the clause its own contract declared.
CANDIDATE_EVIDENCE_FIELDS: tuple[str, ...] = (
    "supported",
    "exclusion_reason",
    "action_sha256",
    "source_state_sha256",
    "target_state_sha256",
    "canonical_successor_key",
    "raw_mark_count",
    "canonical_successor_count",
    "matching_mark_count",
    "exact_successor_mark_count",
    "successor_alias_count",
)

#: Arithmetic every accepted action's evidence satisfies.  Stated as data so the
#: writer, the reader and the sentinel check one list rather than three copies.
ACCEPTED_EVIDENCE_INVARIANTS: tuple[str, ...] = (
    "matching_mark_count == 1",
    "exact_successor_mark_count == 1",
    "successor_alias_count >= 1",
    "exact_successor_mark_count <= successor_alias_count",
    "matching_mark_count <= raw_mark_count",
    "canonical_successor_count <= raw_mark_count",
    "successor_alias_count <= raw_mark_count",
)

# ---- Classification: derived annotation, beside the evidence ----

#: The capability cell and the RAW STRUCTURAL AXES it was derived from.  The
#: axes travel with the cell so a consumer can audit the assignment without
#: importing the classifier, which is what keeps Gate 0 free of it.
#:
#: `classification_affects_admission` is always False and is published rather
#: than assumed: it is the field a reader checks to confirm the annotation did
#: not participate in the decision.
CLASSIFICATION_FIELDS: tuple[str, ...] = (
    "capability_cell_id",
    "model_family",
    "family_context",
    "audit_axes",
    "audit_cycle_rank_delta",
    "audit_touches_ring_system",
    "audit_is_terminal_source",
    "classification_affects_admission",
)

# ---- The Active8 decision row ----

#: One published row per resolved trace.  `partition_role`, not `split`: this is
#: a policy artifact a gate reads to decide eligibility, and the vocabulary it
#: carries is REQUIRED_PARTITION_ROLES.  `split` is the data path's name for the
#: same value and is correct in the cache and the rebind.
ACTIVE8_ROW_FIELDS: tuple[str, ...] = (
    *TRACE_KEY_FIELDS,
    "task_identity_sha256",
    "data_lane",
    "partition_role",
    "admission_status",
    "rejection_category",
    "upstream_rejection_code",
    "path_length",
    "actions",
    "candidate_totals",
    "action_family_histogram",
    "row_sha256",
)

#: A trace the rebind already refused: never candidate-evaluated, present for
#: the census.  Distinct from a trace Active8 evaluated and excluded -- merging
#: them makes "we did not look" indistinguishable from "we looked and said no".
UPSTREAM_REJECTED = "upstream_rebind_rejected"
ACTIVE8_EXCLUDED = "active8_candidate_excluded"
REJECTION_CATEGORIES: tuple[str, ...] = (UPSTREAM_REJECTED, ACTIVE8_EXCLUDED)

#: Recomputed from the action evidence at every boundary, never carried forward
#: as a trusted aggregate.  The previous attempt published these and re-read
#: them; a resealed row could move a census by 1000 undetected.
CANDIDATE_TOTAL_FIELDS: tuple[str, ...] = (
    "raw_candidate_marks",
    "canonical_candidate_successors",
    "matching_candidate_marks",
    "exact_successor_marks",
    "successor_aliases",
)

# ---- The accepted transition Gate 0 aggregates ----

#: Everything Gate 0 needs, and nothing it must compute.  It carries the cell,
#: the axes, the evidence and the counts, so Gate 0 opens no molecular state.
#:
#: `terminal` describes the teacher's SOURCE progress position and is therefore
#: False for every accepted action, INCLUDING a final action whose successor is
#: terminal: a transition exists precisely because its source has an outgoing
#: step.  Reading it from the successor made `terminal_assignment_count_is_zero`
#: unsatisfiable for any non-empty corpus.
ACCEPTED_TRANSITION_FIELDS: tuple[str, ...] = (
    *ACTION_KEY_FIELDS,
    "task_identity_sha256",
    "data_lane",
    "partition_role",
    "executor_rule",
    "progress_index",
    "terminal",
    "capability_cell_id",
    "model_family",
    "family_context",
    "audit_axes",
    "candidate_evidence",
    "assignment_sha256",
)

# ---- Census ----

#: source == upstream_rejected + active8_accepted + active8_excluded.
ACTIVE8_CENSUS_FIELDS: tuple[str, ...] = (
    "source_entries",
    "upstream_rejected_entries",
    "active8_accepted_entries",
    "active8_excluded_entries",
)

# ---- The Active8 task receipt: metadata a gate reads without opening rows ----

#: Published per task BESIDE the transition rows, never inside them.  The split
#: is load-bearing: Gate 0 reads every role's receipt to build the complete
#: census and the sealed-role digests, and only constructs a transitions path
#: inside the eligible-role loop.  Put the rows in the receipt and reading the
#: census opens held-out molecular content, so "never opened" degrades to
#: "never counted".
#:
#: `partition_role`, NOT `split`.  A receipt is exactly the artifact a gate
#: reads to decide eligibility, so it carries the policy vocabulary
#: (`REQUIRED_PARTITION_ROLES`).  `split` is the data path's name for the same
#: value and stays correct in the cache and the rebind.  This field was left
#: unnamed in the first version of this module and the producer and consumer
#: each picked a spelling -- the exact failure the docstring above describes,
#: reproduced one layer down, which is why receipt fields are now named too.
ACTIVE8_RECEIPT_FIELDS: tuple[str, ...] = (
    "schema",
    "schema_version",
    "status",
    "task_identity_sha256",
    "partition_role",
    "data_lane",
    "source_chunk_identity_sha256",
    *ACTIVE8_CENSUS_FIELDS,
    "transition_count",
    "decision_shard_sha256",
    "receipt_sha256",
)

#: Layout. The seam names fields; without naming these it does not name where
#: they live, and two agents chose two layouts.
ACTIVE8_TASKS_DIRNAME = "tasks"
ACTIVE8_RECEIPT_FILENAME = "RECEIPT.json"
ACTIVE8_DECISION_SHARD_FILENAME = "transitions.jsonl.gz"
GATE_ZERO_DECISION_FILENAME = "DECISION.json"

# ---- The release sentinel ----

#: Below this many unique accepted (source_state_sha256, action_sha256) pairs,
#: the sentinel checks ALL of them.
SENTINEL_EXHAUSTIVE_THRESHOLD = 6_784

#: Above it: the deduplicated union of this many SHA-256-ranked examples per
#: required capability cell, and this many globally ranked.
SENTINEL_PER_CELL_EXAMPLES = 128
SENTINEL_GLOBAL_EXAMPLES = 4_608

#: Of the selected set, this many are additionally compared against the
#: independent dictionary successor oracle.  The oracle is a BOUNDED test
#: instrument and never appears in the production path.
SENTINEL_ORACLE_EXAMPLES = 256

#: Prospectively frozen: ranking must not be choosable after seeing results.
SENTINEL_SALT = "process_v2_active8_release_sentinel_v1"

SENTINEL_RESULT_FIELDS: tuple[str, ...] = (
    "schema",
    "schema_version",
    "status",
    "salt",
    "unique_accepted_pairs",
    "selection_mode",
    "selected_pairs",
    "per_cell_examples",
    "global_examples",
    "oracle_examples",
    "evidence_mismatches",
    "cell_mismatches",
    "oracle_mismatches",
    "sentinel_sha256",
)

#: The sentinel runs AFTER task publication and BEFORE authoritative completion,
#: and ANY mismatch blocks completion. It is the only thing that converts the
#: residual -- a per-action count moved within its arithmetic bounds and
#: propagated consistently through every aggregate -- from undetectable to
#: sampled.
SENTINEL_BLOCKS_COMPLETION = True

__all__ = [
    "GATE_ZERO_DECISION_FILENAME",
    "ACTIVE8_TASKS_DIRNAME",
    "ACTIVE8_RECEIPT_FILENAME",
    "ACTIVE8_RECEIPT_FIELDS",
    "ACTIVE8_DECISION_SHARD_FILENAME",
    "ACCEPTED_EVIDENCE_INVARIANTS",
    "ACCEPTED_TRANSITION_FIELDS",
    "ACTION_KEY_FIELDS",
    "ACTIVE8_CENSUS_FIELDS",
    "ACTIVE8_COMPLETION_SCHEMA",
    "ACTIVE8_COMPLETION_SCHEMA_VERSION",
    "ACTIVE8_EXCLUDED",
    "ACTIVE8_ROW_FIELDS",
    "ACTIVE8_TASK_SCHEMA",
    "ACTIVE8_TASK_SCHEMA_VERSION",
    "CANDIDATE_EVIDENCE_FIELDS",
    "CANDIDATE_TOTAL_FIELDS",
    "CLASSIFICATION_FIELDS",
    "GATE_ZERO_DECISION_SCHEMA",
    "GATE_ZERO_DECISION_SCHEMA_VERSION",
    "GATE_ZERO_TASK_SCHEMA",
    "GATE_ZERO_TASK_SCHEMA_VERSION",
    "JOIN_KEY_FIELDS",
    "PIPELINE_NAMESPACE",
    "PIPELINE_STATUS_NO_AUTHORITY",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "REJECTION_CATEGORIES",
    "SENTINEL_BLOCKS_COMPLETION",
    "SENTINEL_EXHAUSTIVE_THRESHOLD",
    "SENTINEL_GLOBAL_EXAMPLES",
    "SENTINEL_ORACLE_EXAMPLES",
    "SENTINEL_PER_CELL_EXAMPLES",
    "SENTINEL_RESULT_FIELDS",
    "SENTINEL_SALT",
    "SENTINEL_SCHEMA",
    "SENTINEL_SCHEMA_VERSION",
    "TRACE_KEY_FIELDS",
    "UPSTREAM_REJECTED",
]
