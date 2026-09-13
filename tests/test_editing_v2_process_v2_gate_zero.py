"""Gate 0 for the Process-V2 pipeline: the post-Active8 structural reducer.

The stand-in producer here writes exactly the seam's ``ACCEPTED_TRANSITION_FIELDS``
and ``ACTIVE8_CENSUS_FIELDS`` -- ``_transition`` asserts the field set against the
frozen tuple before self-hashing, so a seam change breaks the fixture loudly
instead of leaving the suite green while the chain fails to join.  Its canonical
serialization is written here independently rather than imported from the module
under test, because producer and consumer agreeing on it IS the join.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import pathlib
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_process_v2_gate_zero import (
    GateZeroContracts,
    ProcessV2GateZeroError,
    ProcessV2GateZeroIncomplete,
    build_gate_zero_contracts,
    load_gate_zero_contracts,
    read_active8_decision_index,
    reduce_gate_zero,
    reduction_order,
    run_gate_zero,
)
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FIELDS,
    COMPLETION_FILENAME,
)
from compose_v4.data.editing_v2_process_v2_active8_sentinel import (
    SELECTION_EXHAUSTIVE,
    SENTINEL_PASSED,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    CAPABILITY_CELLS,
    DEVELOPMENT_CELL_ROLES,
    GATE_ZERO_STRUCTURAL,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACCEPTED_TRANSITION_FIELDS,
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_DECISION_SHARD_FILENAME,
    ACTIVE8_RECEIPT_FIELDS,
    ACTIVE8_RECEIPT_FILENAME,
    ACTIVE8_TASK_SCHEMA,
    ACTIVE8_TASK_SCHEMA_VERSION,
    ACTIVE8_TASKS_DIRNAME,
    ACTIVE8_COMPLETION_SCHEMA,
    ACTIVE8_COMPLETION_SCHEMA_VERSION,
    CANDIDATE_EVIDENCE_FIELDS,
    CANDIDATE_TOTAL_FIELDS,
    GATE_ZERO_DECISION_FILENAME,
    GATE_ZERO_DECISION_SCHEMA,
    PIPELINE_STATUS_NO_AUTHORITY,
    SENTINEL_RESULT_FIELDS,
    SENTINEL_ORACLE_EXAMPLES,
    SENTINEL_SALT,
    SENTINEL_SCHEMA,
    SENTINEL_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_process_v2_schema import authority_false_block

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The frozen Process-V2 identity, spelled literally so the binding is asserted
#: against a value this repository froze, not against whatever recomputes today.
FROZEN_V2_IDENTITY = "f2739338c561cdc38d550aba08e5ec756005beb9b6a97aa9b4177bd57b270682"

#: Producer-side knowledge: which executor rule carries each scoring family.
_EXECUTOR_RULE = {
    "atom_insert": "atom_insert",
    "atom_delete": "atom_delete",
    "atom_restate": "atom_restate_semantic",
    "bond_reorder": "bond_reorder",
    "bond_reroute": "bond_reroute",
    "cycle_insert": "cycle_close",
    "cycle_attach": "cycle_open",
    "ring_system_restate": "ring_system_restate",
}


# ---- The stand-in producer ---------------------------------------------------


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "supported": True,
        "exclusion_reason": None,
        "action_sha256": _identity("action"),
        "source_state_sha256": _identity("source"),
        "target_state_sha256": _identity("target"),
        "source_canonical_key": "CC",
        "canonical_successor_key": "CCO",
        "teacher_coordinate_legal": True,
        "teacher_executes_to_exact_successor": True,
        "productive_canonical_successor": True,
    }
    evidence.update(overrides)
    assert set(evidence) == set(CANDIDATE_EVIDENCE_FIELDS)
    return evidence


def _transition(
    *,
    task_identity: str,
    partition_role: str,
    data_lane: str,
    v1_task: str,
    entry_index: int,
    trace_id: str,
    step_index: int,
    cell: str,
    terminal: bool = False,
    evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    _namespace, family, context = cell.split(":")
    row: dict[str, object] = {
        "v1_task_identity_sha256": v1_task,
        "entry_index": entry_index,
        "trace_id": trace_id,
        "step_index": step_index,
        "task_identity_sha256": task_identity,
        "data_lane": data_lane,
        "partition_role": partition_role,
        "executor_rule": _EXECUTOR_RULE[family],
        "progress_index": step_index,
        # The teacher's SOURCE progress position: False for every accepted
        # action, including the final one whose successor is terminal.
        "terminal": terminal,
        "capability_cell_id": cell,
        "model_family": family,
        "family_context": context,
        "audit_axes": {
            "audit_cycle_rank_delta": 0,
            "audit_touches_ring_system": False,
            "audit_is_terminal_source": False,
        },
        "candidate_evidence": evidence if evidence is not None else _evidence(),
    }
    assert set(row) == set(ACCEPTED_TRANSITION_FIELDS) - {"assignment_sha256"}
    row["assignment_sha256"] = _digest(row)
    assert set(row) == set(ACCEPTED_TRANSITION_FIELDS)
    return row


def _write_shard(
    run_root: Path,
    *,
    label: str,
    partition_role: str,
    data_lane: str = "observed_local_analogue",
    cells: list[str],
    rows_per_trace: int = 2,
    upstream_rejected: int = 3,
    excluded: int = 2,
    mutate=None,
    receipt_mutate=None,
    accepted_entries: int | None = None,
    write_shard_object: bool = True,
) -> str:
    """Write one Active8 decision shard: RECEIPT.json plus transitions.jsonl.gz."""

    task_identity = _identity(label)
    v1_task = _identity(f"v1:{label}")
    rows: list[dict[str, object]] = []
    traces = 0
    entry_index = 0
    pending = list(cells)
    while pending:
        batch, pending = pending[:rows_per_trace], pending[rows_per_trace:]
        trace_id = f"{label}-trace-{traces:03d}"
        for step_index, cell in enumerate(batch):
            rows.append(
                _transition(
                    task_identity=task_identity,
                    partition_role=partition_role,
                    data_lane=data_lane,
                    v1_task=v1_task,
                    entry_index=entry_index,
                    trace_id=trace_id,
                    step_index=step_index,
                    cell=cell,
                )
            )
        traces += 1
        entry_index += 1
    if mutate is not None:
        rows = mutate(rows)
    accepted = traces if accepted_entries is None else accepted_entries
    census = {
        "source_entries": accepted + upstream_rejected + excluded,
        "upstream_rejected_entries": upstream_rejected,
        "active8_accepted_entries": accepted,
        "active8_excluded_entries": excluded,
    }
    assert set(census) == set(ACTIVE8_CENSUS_FIELDS)
    directory = run_root / ACTIVE8_TASKS_DIRNAME / task_identity
    directory.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical(row) + b"\n" for row in rows)
    shard_bytes = gzip.compress(payload, mtime=0)
    if write_shard_object:
        (directory / ACTIVE8_DECISION_SHARD_FILENAME).write_bytes(shard_bytes)
    # The census fields are TOP-LEVEL, not nested under a `census` key.
    body = {
        "schema": ACTIVE8_TASK_SCHEMA,
        "schema_version": ACTIVE8_TASK_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        "task_identity_sha256": task_identity,
        "partition_role": partition_role,
        "data_lane": data_lane,
        "source_chunk_identity_sha256": _identity(f"chunk:{label}"),
        **census,
        "transition_count": len(rows),
        "decision_shard_sha256": hashlib.sha256(shard_bytes).hexdigest(),
    }
    receipt = {**body, "receipt_sha256": _digest(body)}
    assert set(receipt) == set(ACTIVE8_RECEIPT_FIELDS)
    (directory / ACTIVE8_RECEIPT_FILENAME).write_bytes(_canonical(receipt) + b"\n")
    _publish_completion(run_root)
    if receipt_mutate is not None:
        receipt = receipt_mutate(receipt)
        (directory / ACTIVE8_RECEIPT_FILENAME).write_bytes(_canonical(receipt) + b"\n")
    return task_identity


def _reseal(receipt: dict[str, object], **changes: object) -> dict[str, object]:
    """Apply changes and re-seal, so the self-hash is not what fails."""

    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    body.update(changes)
    return {**body, "receipt_sha256": _digest(body)}


def _publish_completion(run_root: Path) -> dict[str, object]:
    """Publish an exact context-bound stand-in completion for the reducer tests."""

    receipts = []
    for directory in sorted((run_root / ACTIVE8_TASKS_DIRNAME).iterdir()):
        receipt = json.loads((directory / ACTIVE8_RECEIPT_FILENAME).read_bytes())
        receipts.append(receipt)
    inventory = [
        [str(receipt["task_identity_sha256"]), str(receipt["receipt_sha256"])]
        for receipt in receipts
    ]
    task_inventory_sha256 = _digest([item[0] for item in inventory])
    result_inventory_sha256 = _digest(inventory)
    binding_sha256 = _identity("binding")
    plan_sha256 = _identity("plan")
    run_identity_sha256 = _identity("run")
    transition_count = sum(int(receipt["transition_count"]) for receipt in receipts)
    sentinel_body = {
        "schema": SENTINEL_SCHEMA,
        "schema_version": SENTINEL_SCHEMA_VERSION,
        "status": SENTINEL_PASSED,
        "salt": SENTINEL_SALT,
        "binding_sha256": binding_sha256,
        "plan_sha256": plan_sha256,
        "run_identity_sha256": run_identity_sha256,
        "task_inventory_sha256": task_inventory_sha256,
        "result_inventory_sha256": result_inventory_sha256,
        "unique_accepted_pairs": transition_count,
        "selection_mode": SELECTION_EXHAUSTIVE,
        "selected_pairs": transition_count,
        "selected_pairs_sha256": _digest(
            [["source", str(index)] for index in range(transition_count)]
        ),
        "per_cell_examples": 0,
        "global_examples": 0,
        "oracle_examples": min(transition_count, SENTINEL_ORACLE_EXAMPLES),
        "evidence_mismatches": [],
        "cell_mismatches": [],
        "oracle_mismatches": [],
    }
    sentinel = {**sentinel_body, "sentinel_sha256": _digest(sentinel_body)}
    assert tuple(sentinel) == SENTINEL_RESULT_FIELDS
    census = {
        field: sum(int(receipt[field]) for receipt in receipts)
        for field in ACTIVE8_CENSUS_FIELDS
    }
    body = {
        "schema": ACTIVE8_COMPLETION_SCHEMA,
        "schema_version": ACTIVE8_COMPLETION_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "accepted_transitions": transition_count,
        "action_family_histogram": {},
        "active8_exclusion_histogram": {},
        "binding_sha256": binding_sha256,
        "candidate_totals": dict.fromkeys(CANDIDATE_TOTAL_FIELDS, 0),
        "capability_cell_histogram": {},
        "census": census,
        "classification_affects_admission": False,
        "plan_sha256": plan_sha256,
        "result_inventory": inventory,
        "result_inventory_sha256": result_inventory_sha256,
        "run_artifact_root": str(run_root),
        "run_identity_sha256": run_identity_sha256,
        "sentinel": sentinel,
        "task_inventory_sha256": task_inventory_sha256,
    }
    completion = {**body, "completion_sha256": _digest(body)}
    assert set(completion) == COMPLETION_FIELDS
    (run_root / COMPLETION_FILENAME).write_bytes(_canonical(completion) + b"\n")
    return completion


def _rewrite_completion(run_root: Path, mutate) -> dict[str, object]:
    """Mutate and fully reseal a stand-in completion."""

    path = run_root / COMPLETION_FILENAME
    completion = json.loads(path.read_bytes())
    body = {key: value for key, value in completion.items() if key != "completion_sha256"}
    body = mutate(body)
    rewritten = {**body, "completion_sha256": _digest(body)}
    path.write_bytes(_canonical(rewritten) + b"\n")
    return rewritten


def _contracts() -> GateZeroContracts:
    return load_gate_zero_contracts(repo_root=REPO_ROOT)


def _build_run(root: Path, *, contracts: GateZeroContracts, **overrides) -> dict[str, str]:
    """A run whose train role covers every required cell, plus sealed roles."""

    required = list(contracts.required_cell_ids)
    assert len(required) == 17
    groups = [required[:6], required[6:12], required[12:]]
    identities = {
        f"train-{index}": _write_shard(
            root,
            label=f"train-{index}",
            partition_role="train",
            cells=group,
            **overrides,
        )
        for index, group in enumerate(groups)
    }
    identities["train-empty"] = _write_shard(
        root,
        label="train-empty",
        partition_role="train",
        cells=[],
        accepted_entries=0,
    )
    for role in contracts.sealed_roles:
        identities[role] = _write_shard(
            root, label=f"sealed-{role}", partition_role=role, cells=required[:4]
        )
    return identities


# ---- Contract binding --------------------------------------------------------


def test_contracts_bind_the_frozen_process_v2_identity_and_registry() -> None:
    contracts = _contracts()
    assert contracts.process_identity_sha256 == FROZEN_V2_IDENTITY
    assert len(contracts.active_families) == 8
    assert len(contracts.required_cell_ids) == 17
    assert len(contracts.conditional_cell_ids) == 3
    assert len(contracts.separate_lane_cell_ids) == 2
    assert len(contracts.registered_cell_ids) == 22
    assert contracts.decision_eligible_roles == ("train",)
    assert set(contracts.sealed_roles) == {"validation", "controller_validation", "final_test"}


# ---- The decision ------------------------------------------------------------


def test_complete_train_coverage_passes_and_authorizes_nothing(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts)
    decision = run_gate_zero(
        active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT
    )

    assert decision["decision"] == "PASS"
    assert decision["schema"] == GATE_ZERO_DECISION_SCHEMA
    assert decision["status"] == PIPELINE_STATUS_NO_AUTHORITY
    assert decision["process_identity_sha256"] == FROZEN_V2_IDENTITY
    assert all(value is False for key, value in decision.items() if key.endswith("_authorized"))
    assert decision["missing_required_cells"] == []
    assert decision["missing_active8_families"] == []
    assert decision["total_violations"] == 0
    assert decision["accounting"]["decision_shards_read"] == 3
    assert decision["accounting"]["transitions"] == 17
    assert decision["accounting"]["distinct_action_keys"] == 17
    assert decision["enforced_structural_clauses"] == {
        "require_every_active8_family": True,
        "require_every_teacher_supported": True,
        "require_one_assignment_per_accepted_action": True,
        "require_productive_nonself_successor": True,
        "require_teacher_coordinate_legal": True,
        "require_teacher_executes_to_exact_successor": True,
        "require_zero_terminal_assignments": True,
    }
    assert (tmp_path / "gate0" / GATE_ZERO_DECISION_FILENAME).is_file()


def test_every_role_is_censused_while_only_train_is_read(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts)
    index = read_active8_decision_index(active8, contracts=contracts)

    assert len(index.shards) == 7
    assert len(index.eligible_task_identities) == 3
    assert sorted(index.role_census) == sorted(
        ["train", "validation", "controller_validation", "final_test"]
    )
    for role in contracts.sealed_roles:
        assert index.role_census[role]["shards"] == 1
        assert index.role_census[role]["active8_accepted_entries"] > 0
        assert index.sealed_role_metadata[role]["metadata_resolved"] == 1
        # Nothing has been opened yet, so the index must not claim a count.
        assert "decision_shards_opened" not in index.sealed_role_metadata[role]
    # The empty train shard is censused but never becomes eligible.
    assert index.role_census["train"]["shards"] == 4
    assert all(
        shard["rows_read"] is False
        for shard in index.shards
        if shard["census"]["active8_accepted_entries"] == 0
    )


def test_gate_zero_refuses_a_run_without_the_sentinel_gated_completion(
    tmp_path: Path,
) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts)
    (active8 / COMPLETION_FILENAME).unlink()
    with pytest.raises(ProcessV2GateZeroIncomplete, match="sentinel-gated completion"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)


def test_gate_zero_refuses_a_resealed_pass_sentinel_with_a_mismatch(
    tmp_path: Path,
) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts)

    def mutate(completion: dict[str, object]) -> dict[str, object]:
        sentinel = dict(completion["sentinel"])
        sentinel_body = {
            key: value for key, value in sentinel.items() if key != "sentinel_sha256"
        }
        sentinel_body["evidence_mismatches"] = ["resealed-mismatch"]
        completion["sentinel"] = {
            **sentinel_body,
            "sentinel_sha256": _digest(sentinel_body),
        }
        return completion

    _rewrite_completion(active8, mutate)
    with pytest.raises(ProcessV2GateZeroIncomplete, match="sentinel-gated completion"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)


def test_an_injected_index_cannot_bypass_a_later_failed_completion(
    tmp_path: Path,
) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts)
    index = read_active8_decision_index(active8, contracts=contracts)

    def fail(completion: dict[str, object]) -> dict[str, object]:
        sentinel = dict(completion["sentinel"])
        sentinel_body = {
            key: value for key, value in sentinel.items() if key != "sentinel_sha256"
        }
        sentinel_body["status"] = "SENTINEL_FAILED"
        sentinel_body["evidence_mismatches"] = ["late-failure"]
        completion["sentinel"] = {
            **sentinel_body,
            "sentinel_sha256": _digest(sentinel_body),
        }
        return completion

    _rewrite_completion(active8, fail)
    with pytest.raises(ProcessV2GateZeroIncomplete, match="sentinel-gated completion"):
        reduce_gate_zero(
            active8,
            gate_zero_root=tmp_path / "gate0",
            contracts=contracts,
            index=index,
        )


# ---- The three defects the previous attempt shipped --------------------------


def test_terminal_is_the_source_position_so_a_final_action_still_passes(
    tmp_path: Path,
) -> None:
    """A trace's last action has a terminal SUCCESSOR and terminal=False.

    Reading `terminal` off the successor made `terminal_assignment_count_is_zero`
    unsatisfiable for any non-empty corpus.  The fixture's traces each end in
    such an action, and the gate passes with a zero terminal count.
    """

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts)
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    assert decision["accounting"]["accepted_traces"] == 9
    assert decision["accounting"]["transitions"] > decision["accounting"]["accepted_traces"]
    assert decision["violation_counts"]["terminal_assignment"] == 0
    assert decision["checks"]["terminal_assignment_count_is_zero"] is True


def test_a_terminal_source_position_fails_the_gate(tmp_path: Path) -> None:
    def flip(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        body = {key: value for key, value in rows[0].items() if key != "assignment_sha256"}
        body["terminal"] = True
        return [{**body, "assignment_sha256": _digest(body)}, *rows[1:]]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts, mutate=flip)
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    assert decision["decision"] == "FAIL"
    assert decision["violation_counts"]["terminal_assignment"] == 3
    assert decision["checks"]["terminal_assignment_count_is_zero"] is False
    assert all(value is False for key, value in decision.items() if key.endswith("_authorized"))


def test_require_every_teacher_supported_is_actually_enforced(tmp_path: Path) -> None:
    """The clause the previous attempt declared, refused a contract without, and
    then never read.  The seam publishes `supported`, so it is read here."""

    def unsupport(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        body = {key: value for key, value in rows[0].items() if key != "assignment_sha256"}
        body["candidate_evidence"] = _evidence(
            supported=False, exclusion_reason="outside_declared_support"
        )
        return [{**body, "assignment_sha256": _digest(body)}, *rows[1:]]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts, mutate=unsupport)
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    assert decision["decision"] == "FAIL"
    assert decision["checks"]["every_teacher_supported"] is False
    assert decision["violation_counts"]["unsupported_teacher"] == 3
    categories = {receipt["category"] for receipt in decision["negative_receipts"]}
    assert categories == {"unsupported_teacher"}
    assert "outside_declared_support" in decision["negative_receipts"][0]["detail"]


def test_a_canonical_self_transition_fails_the_productive_successor_gate(
    tmp_path: Path,
) -> None:
    """Whole-corpus evidence must prove a productive nonself successor."""

    def make_self(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        body = {key: value for key, value in rows[0].items() if key != "assignment_sha256"}
        body["candidate_evidence"] = _evidence(
            source_canonical_key="CCO", canonical_successor_key="CCO"
        )
        return [{**body, "assignment_sha256": _digest(body)}, *rows[1:]]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts, mutate=make_self)
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    assert decision["decision"] == "FAIL"
    assert decision["violation_counts"]["self_transition_teacher"] == 3
    assert decision["checks"]["productive_nonself_successor"] is False


# ---- Every declared structural clause has a behavioural enforcement site ------


@pytest.mark.parametrize(
    ("overrides", "violation"),
    [
        ({"teacher_coordinate_legal": False}, "illegal_teacher_coordinate"),
        (
            {"teacher_executes_to_exact_successor": False},
            "inexact_teacher_successor",
        ),
        (
            {"productive_canonical_successor": False},
            "nonproductive_teacher_successor",
        ),
        (
            {"source_canonical_key": "CCO", "canonical_successor_key": "CCO"},
            "self_transition_teacher",
        ),
    ],
)
def test_evidence_clauses_fail_the_gate(
    tmp_path: Path, overrides: dict[str, object], violation: str
) -> None:
    def corrupt(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        body = {key: value for key, value in rows[0].items() if key != "assignment_sha256"}
        body["candidate_evidence"] = _evidence(**overrides)
        return [{**body, "assignment_sha256": _digest(body)}, *rows[1:]]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts, mutate=corrupt)
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    assert decision["decision"] == "FAIL"
    assert decision["violation_counts"][violation] >= 1


def test_a_missing_family_and_cell_fail_the_gate(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    required = [
        cell for cell in contracts.required_cell_ids if "ring_system_restate" not in cell
    ]
    _write_shard(active8, label="train-0", partition_role="train", cells=required)
    for role in contracts.sealed_roles:
        _write_shard(
            active8, label=f"sealed-{role}", partition_role=role, cells=list(required[:2])
        )
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    assert decision["decision"] == "FAIL"
    assert decision["missing_active8_families"] == ["ring_system_restate"]
    assert len(decision["missing_required_cells"]) == 2
    assert decision["checks"]["every_active8_family_present"] is False
    assert decision["checks"]["every_required_cell_present"] is False
    assert all(value is False for key, value in decision.items() if key.endswith("_authorized"))


def test_a_repeated_action_assignment_fails_the_gate(tmp_path: Path) -> None:
    def duplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return [*rows, dict(rows[0])]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    required = list(contracts.required_cell_ids)
    _write_shard(
        active8, label="train-0", partition_role="train", cells=required, mutate=duplicate
    )
    for role in contracts.sealed_roles:
        _write_shard(active8, label=f"sealed-{role}", partition_role=role, cells=required[:2])
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    assert decision["decision"] == "FAIL"
    assert decision["violation_counts"]["duplicate_action_assignment"] == 1
    assert decision["checks"]["one_assignment_per_accepted_action"] is False


def test_an_unregistered_capability_cell_fails_the_gate(tmp_path: Path) -> None:
    def relabel(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        body = {key: value for key, value in rows[0].items() if key != "assignment_sha256"}
        body["capability_cell_id"] = "editing_v2_active8_v1:atom_insert:invented_birth"
        body["family_context"] = "invented_birth"
        return [{**body, "assignment_sha256": _digest(body)}, *rows[1:]]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts, mutate=relabel)
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    assert decision["decision"] == "FAIL"
    assert decision["violation_counts"]["unregistered_capability_cell"] == 3
    assert decision["violation_counts"]["unknown_family_context"] == 3
    assert decision["checks"]["every_cell_registered"] is False


# ---- The seam actually joining -----------------------------------------------


def test_a_row_spelling_split_instead_of_partition_role_raises(tmp_path: Path) -> None:
    """The exact way the previous chain failed to join, made loud."""

    def rename(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        body = {key: value for key, value in rows[0].items() if key != "assignment_sha256"}
        body["split"] = body.pop("partition_role")
        return [{**body, "assignment_sha256": _digest(body)}, *rows[1:]]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts, mutate=rename)
    with pytest.raises(ProcessV2GateZeroError, match="ACCEPTED_TRANSITION_FIELDS"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)
    assert not (tmp_path / "gate0" / GATE_ZERO_DECISION_FILENAME).exists()


def test_tampered_teacher_evidence_breaks_the_row_self_hash(tmp_path: Path) -> None:
    def tamper(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        row = dict(rows[0])
        evidence = dict(row["candidate_evidence"])
        evidence["teacher_coordinate_legal"] = False
        row["candidate_evidence"] = evidence
        return [row, *rows[1:]]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts, mutate=tamper)
    with pytest.raises(ProcessV2GateZeroError, match="assignment_sha256"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)


def test_a_shard_census_that_disagrees_with_its_rows_raises(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _write_shard(
        active8,
        label="train-0",
        partition_role="train",
        cells=list(contracts.required_cell_ids),
        accepted_entries=99,
    )
    for role in contracts.sealed_roles:
        _write_shard(
            active8,
            label=f"sealed-{role}",
            partition_role=role,
            cells=list(contracts.required_cell_ids[:2]),
        )
    with pytest.raises(ProcessV2GateZeroError, match="accepted traces"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)


def test_an_absent_eligible_shard_publishes_nothing(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    identities = _build_run(active8, contracts=contracts)
    (
        active8
        / ACTIVE8_TASKS_DIRNAME
        / identities["train-1"]
        / ACTIVE8_DECISION_SHARD_FILENAME
    ).unlink()
    with pytest.raises(ProcessV2GateZeroIncomplete):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)
    assert not (tmp_path / "gate0" / GATE_ZERO_DECISION_FILENAME).exists()


# ---- Determinism -------------------------------------------------------------


def test_reduction_order_is_stated_not_inherited() -> None:
    shards = [{"task_identity_sha256": _identity(str(index))} for index in range(8)]
    forward = reduction_order(shards)
    backward = reduction_order(list(reversed(shards)))
    assert forward == backward
    assert [shard["task_identity_sha256"] for shard in forward] == sorted(
        shard["task_identity_sha256"] for shard in shards
    )
    with pytest.raises(ProcessV2GateZeroError):
        reduction_order([shards[0], dict(shards[0])])


def test_a_reversed_filesystem_enumeration_yields_the_same_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts)
    forward = run_gate_zero(active8, gate_zero_root=tmp_path / "a", repo_root=REPO_ROOT)

    real_iterdir = pathlib.Path.iterdir
    monkeypatch.setattr(
        pathlib.Path, "iterdir", lambda self: iter(sorted(real_iterdir(self), reverse=True))
    )
    reversed_decision = run_gate_zero(active8, gate_zero_root=tmp_path / "b", repo_root=REPO_ROOT)

    assert reversed_decision == forward
    assert (tmp_path / "a" / GATE_ZERO_DECISION_FILENAME).read_bytes() == (
        tmp_path / "b" / GATE_ZERO_DECISION_FILENAME
    ).read_bytes()


def test_the_reduction_digest_witnesses_the_stated_order(tmp_path: Path) -> None:
    """Computed here from the shard identities and their bytes, in the stated
    ascending-identity order, so a digest folded in any other order fails."""

    contracts = _contracts()
    active8 = tmp_path / "active8"
    identities = _build_run(active8, contracts=contracts)
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    eligible = sorted(identities[f"train-{index}"] for index in range(3))
    expected = _digest(
        [
            [
                identity,
                hashlib.sha256(
                    (
                        active8
                        / ACTIVE8_TASKS_DIRNAME
                        / identity
                        / ACTIVE8_DECISION_SHARD_FILENAME
                    ).read_bytes()
                ).hexdigest(),
            ]
            for identity in eligible
        ]
    )
    assert decision["reduction"]["digest_sha256"] == expected
    assert decision["reduction"]["order"] == "ascending_task_identity_sha256"


# ---- What Gate 0 opens, measured ----------------------------------------------


class _OpenSpy:
    """Records every file this process opens, through `io.open` and `builtins.open`."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = io.open

        def spy(file, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            self.paths.append(str(file))
            return real(file, *args, **kwargs)

        monkeypatch.setattr(io, "open", spy)
        monkeypatch.setattr("builtins.open", spy)


def test_gate_zero_opens_zero_molecular_chunks_and_zero_sealed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    identities = _build_run(active8, contracts=contracts)
    sealed_identities = {identities[role] for role in contracts.sealed_roles}

    # A molecular cache chunk, in the production naming, next to the run.  The
    # gate has no reason to open it; the positive control below proves the spy
    # would see it if the gate did.
    chunk = tmp_path / "cache" / "chunk-000000.jsonl.gz"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(gzip.compress(b"{}\n", mtime=0))

    spy = _OpenSpy()
    spy.install(monkeypatch)
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)
    observed = list(spy.paths)

    # Positive control: the same instrument, still installed, sees a genuine
    # chunk open.  Without this the "zero" below could be an un-wired probe.
    chunk.read_bytes()
    assert sum("chunk-000000.jsonl.gz" in path for path in spy.paths) == 1

    chunk_opens = [path for path in observed if "chunk-" in path and path.endswith(".jsonl.gz")]
    assert chunk_opens == []
    assert decision["accounting"]["decision_shards_opened_by_role"] == {"train": 3}

    sealed_opens = [
        path
        for path in observed
        if path.endswith(ACTIVE8_DECISION_SHARD_FILENAME)
        and any(identity in path for identity in sealed_identities)
    ]
    assert sealed_opens == []
    for role in contracts.sealed_roles:
        assert decision["sealed_role_metadata"][role]["decision_shards_opened"] == 0

    # ...while every role's metadata WAS resolved, sealed roles included.
    sealed_metadata_opens = [
        path
        for path in observed
        if path.endswith(ACTIVE8_RECEIPT_FILENAME)
        and any(identity in path for identity in sealed_identities)
    ]
    assert len(sealed_metadata_opens) == len(sealed_identities) == 3

    # The empty train shard is eligible by role but not by census, so its rows
    # are never opened either.
    empty_opens = [
        path
        for path in observed
        if path.endswith(ACTIVE8_DECISION_SHARD_FILENAME) and identities["train-empty"] in path
    ]
    assert empty_opens == []

    shard_opens = [path for path in observed if path.endswith(ACTIVE8_DECISION_SHARD_FILENAME)]
    assert len(shard_opens) == 3 == decision["accounting"]["decision_shards_read"]
    assert len(shard_opens) < decision["accounting"]["transitions"]


def test_shard_reads_do_not_scale_with_transitions(tmp_path: Path) -> None:
    """One read per eligible shard, whatever the shard holds."""

    contracts = _contracts()
    required = list(contracts.required_cell_ids)
    counts = []
    for factor, name in ((1, "small"), (6, "large")):
        active8 = tmp_path / name
        _write_shard(
            active8, label="train-0", partition_role="train", cells=required * factor
        )
        for role in contracts.sealed_roles:
            _write_shard(active8, label=f"sealed-{role}", partition_role=role, cells=required[:2])
        decision = run_gate_zero(
            active8, gate_zero_root=tmp_path / f"gate0-{name}", repo_root=REPO_ROOT
        )
        counts.append((decision["accounting"]["decision_shards_read"], decision["accounting"]))

    assert counts[0][0] == counts[1][0] == 1
    assert counts[1][1]["transitions"] == 6 * counts[0][1]["transitions"] == 102
    assert counts[1][1]["accepted_traces"] > counts[0][1]["accepted_traces"]


# ---- Refusals ----------------------------------------------------------------


def test_a_sealed_shard_can_never_reach_the_reduction(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    for role in contracts.sealed_roles:
        _write_shard(
            active8,
            label=f"sealed-{role}",
            partition_role=role,
            cells=list(contracts.required_cell_ids),
        )
    index = read_active8_decision_index(active8, contracts=contracts)
    assert index.eligible_task_identities == ()
    with pytest.raises(ProcessV2GateZeroIncomplete):
        reduce_gate_zero(
            active8, gate_zero_root=tmp_path / "gate0", contracts=contracts, index=index
        )
    assert not (tmp_path / "gate0" / GATE_ZERO_DECISION_FILENAME).exists()


def test_a_row_claiming_another_active8_task_raises(tmp_path: Path) -> None:
    def steal(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        body = {key: value for key, value in rows[0].items() if key != "assignment_sha256"}
        body["task_identity_sha256"] = _identity("some-other-task")
        return [{**body, "assignment_sha256": _digest(body)}, *rows[1:]]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts, mutate=steal)
    with pytest.raises(ProcessV2GateZeroError, match="another Active8 task"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)


def test_absent_structural_axes_fail_the_gate(tmp_path: Path) -> None:
    """The axes travel with the cell precisely so the assignment is auditable
    without the classifier; a row that drops them is not auditable."""

    def strip(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        body = {key: value for key, value in rows[0].items() if key != "assignment_sha256"}
        body["audit_axes"] = {}
        return [{**body, "assignment_sha256": _digest(body)}, *rows[1:]]

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _build_run(active8, contracts=contracts, mutate=strip)
    decision = run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)

    assert decision["decision"] == "FAIL"
    assert decision["violation_counts"]["missing_audit_axes"] == 3
    assert decision["checks"]["audit_axes_present"] is False


# ---- The receipt binds to the seam, not to a local copy ----------------------


def test_a_receipt_spelling_split_instead_of_partition_role_raises(tmp_path: Path) -> None:
    """The adjudicated case, locked in.

    The receipt is exactly the artifact this gate reads to decide eligibility, so
    it carries the policy vocabulary: `partition_role`, not `split`.  `split`
    stays correct in the cache and the rebind.  Accepting it here silently would
    make the seam's field-naming rule advisory.
    """

    contracts = _contracts()
    active8 = tmp_path / "active8"
    _write_shard(
        active8,
        label="train-0",
        partition_role="train",
        cells=list(contracts.required_cell_ids),
        receipt_mutate=lambda receipt: _reseal(
            {k: v for k, v in receipt.items() if k != "partition_role"},
            split="train",
        ),
    )
    with pytest.raises(ProcessV2GateZeroError, match="ACTIVE8_RECEIPT_FIELDS"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)
    assert not (tmp_path / "gate0" / GATE_ZERO_DECISION_FILENAME).exists()


def test_a_receipt_whose_self_hash_disagrees_raises(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _write_shard(
        active8,
        label="train-0",
        partition_role="train",
        cells=list(contracts.required_cell_ids),
        receipt_mutate=lambda receipt: {**receipt, "active8_excluded_entries": 99},
    )
    with pytest.raises(ProcessV2GateZeroError, match="receipt_sha256"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)


def test_shard_bytes_are_authenticated_against_the_receipt(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    identity = _write_shard(
        active8,
        label="train-0",
        partition_role="train",
        cells=list(contracts.required_cell_ids),
        receipt_mutate=lambda receipt: _reseal(receipt, decision_shard_sha256=_identity("other")),
    )
    assert identity
    _publish_completion(active8)
    with pytest.raises(ProcessV2GateZeroError, match="decision_shard_sha256"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)


def test_a_shard_shorter_than_its_declared_transition_count_raises(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _write_shard(
        active8,
        label="train-0",
        partition_role="train",
        cells=list(contracts.required_cell_ids),
        receipt_mutate=lambda receipt: _reseal(receipt, transition_count=999),
    )
    _publish_completion(active8)
    with pytest.raises(ProcessV2GateZeroError, match="its receipt declares"):
        run_gate_zero(active8, gate_zero_root=tmp_path / "gate0", repo_root=REPO_ROOT)


# ---- Contract cross-validation, with reachable witnesses ---------------------


def _payloads() -> tuple[dict, dict, dict]:
    return (
        json.loads((REPO_ROOT / GATE_ZERO_STRUCTURAL).read_bytes()),
        json.loads((REPO_ROOT / CAPABILITY_CELLS).read_bytes()),
        json.loads((REPO_ROOT / DEVELOPMENT_CELL_ROLES).read_bytes()),
    )


def test_the_frozen_payloads_build_the_bound_contracts() -> None:
    structural, cells, roles = _payloads()
    contracts = build_gate_zero_contracts(
        structural, cells, roles, live_process_identity_sha256=FROZEN_V2_IDENTITY
    )
    assert contracts.binding_sha256 == _contracts().binding_sha256


def test_a_contract_omitting_a_clause_gate_zero_enforces_is_refused() -> None:
    for clause in (
        "require_every_teacher_supported",
        "require_zero_terminal_assignments",
        "require_every_active8_family",
        "require_one_assignment_per_accepted_action",
    ):
        structural, cells, roles = _payloads()
        structural["structural_checks"].pop(clause)
        with pytest.raises(ProcessV2GateZeroError, match=clause):
            build_gate_zero_contracts(
                structural, cells, roles, live_process_identity_sha256=FROZEN_V2_IDENTITY
            )


def test_roles_that_do_not_partition_the_corpus_roles_are_refused() -> None:
    structural, cells, roles = _payloads()
    structural["structural_checks"]["sealed_nondecision_partition_roles"] = ["validation"]
    with pytest.raises(ProcessV2GateZeroError, match="partition REQUIRED_PARTITION_ROLES"):
        build_gate_zero_contracts(
            structural, cells, roles, live_process_identity_sha256=FROZEN_V2_IDENTITY
        )

    structural, cells, roles = _payloads()
    structural["structural_checks"]["decision_eligible_partition_roles"] = ["train", "validation"]
    with pytest.raises(ProcessV2GateZeroError, match="both decision-eligible and sealed"):
        build_gate_zero_contracts(
            structural, cells, roles, live_process_identity_sha256=FROZEN_V2_IDENTITY
        )


def test_a_registry_that_is_not_the_family_by_context_product_is_refused() -> None:
    structural, cells, roles = _payloads()
    roles["required_cell_ids"] = roles["required_cell_ids"][:-1]
    roles["partition_policy"]["required_cell_count"] -= 1
    with pytest.raises(ProcessV2GateZeroError, match="family x context registry"):
        build_gate_zero_contracts(
            structural, cells, roles, live_process_identity_sha256=FROZEN_V2_IDENTITY
        )

    structural, cells, roles = _payloads()
    roles["conditional_cell_ids"] = list(roles["required_cell_ids"][:1])
    with pytest.raises(ProcessV2GateZeroError, match="two development roles"):
        build_gate_zero_contracts(
            structural, cells, roles, live_process_identity_sha256=FROZEN_V2_IDENTITY
        )


def test_a_contract_pinning_another_process_identity_is_refused() -> None:
    structural, cells, roles = _payloads()
    superseded_v1 = "6c4721f0dd37132aae657e7aa5f1bfc01cef270662f228171c4587eb7dd48491"
    structural["process_identity"]["process_identity_sha256"] = superseded_v1
    with pytest.raises(ProcessV2GateZeroError, match="pins process identity"):
        build_gate_zero_contracts(
            structural, cells, roles, live_process_identity_sha256=FROZEN_V2_IDENTITY
        )


def test_an_unknown_partition_role_or_lane_raises(tmp_path: Path) -> None:
    contracts = _contracts()
    active8 = tmp_path / "active8"
    _write_shard(
        active8,
        label="train-0",
        partition_role="train",
        data_lane="a_lane_nobody_bound",
        cells=list(contracts.required_cell_ids),
    )
    with pytest.raises(ProcessV2GateZeroError, match="data_lane"):
        read_active8_decision_index(active8, contracts=contracts)
