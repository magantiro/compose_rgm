"""The Process-V2 Active8 stage, driven end to end on a real payload.

Every test here runs the production modules over a real chunk cache, a real
Process-V2 rebind and a real semantic model.  Nothing reimplements an
expectation the code under test could also produce: the classifier-independence
test compares two REAL runs against each other, the determinism test compares
two REAL reductions, and the dictionary oracle is checked against the
repository's own ``reference_successor_kernel``, which production code is
forbidden to import.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from compose_v4.data.editing_process_v2_rebind import (
    MANIFEST_FILENAME as REBIND_MANIFEST_FILENAME,
)
from compose_v4.data.editing_process_v2_admitted_source import (
    resolve_process_v2_admitted_source,
)
from compose_v4.data.editing_v2_process_v2_active8_map import (
    ACCEPTED,
    ACTION_FIELDS,
    NOT_EVALUATED,
    RECEIPT_FILENAME,
    ROWS_FILENAME,
    SUMMARY_FILENAME,
    TRANSITIONS_FILENAME,
    ProcessV2Active8MapError,
    derive_candidate_totals,
    execute_process_v2_active8_task,
    read_task_rows,
    read_task_transitions,
    require_accepted_evidence_invariants,
    summarize_task,
    validate_process_v2_active8_task_result,
)
from compose_v4.data.editing_v2_process_v2_active8_plan import (
    IMPLEMENTATION_FILES,
    PLAN_FILENAME,
    ProcessV2Active8PlanError,
    build_process_v2_active8_binding,
    load_process_v2_active8_plan,
    model_runtime_descriptor,
    plan_process_v2_active8,
    validate_process_v2_active8_plan,
    write_process_v2_active8_plan,
)
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FILENAME,
    ProcessV2Active8Incomplete,
    completed_process_v2_active8_task_ids,
    load_process_v2_active8_completion,
    reduce_process_v2_active8,
    run_process_v2_active8_tasks,
    task_output_path,
)
from compose_v4.data.editing_v2_process_v2_active8_sentinel import (
    SELECTION_EXHAUSTIVE,
    SENTINEL_PASSED,
    ProcessV2Active8SentinelError,
    _dictionary_successor_oracle,
    require_sentinel_passed,
    sentinel_rank,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACCEPTED_TRANSITION_FIELDS,
    ACTIVE8_DECISION_SHARD_FILENAME,
    ACTIVE8_RECEIPT_FIELDS,
    ACTIVE8_RECEIPT_FILENAME,
    ACTIVE8_ROW_SHARD_FILENAME,
    ACTIVE8_TASKS_DIRNAME,
    ACTIVE8_EXCLUDED,
    ACTIVE8_ROW_FIELDS,
    CANDIDATE_EVIDENCE_FIELDS,
    CLASSIFICATION_FIELDS,
    SENTINEL_RESULT_FIELDS,
    SENTINEL_SALT,
    UPSTREAM_REJECTED,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    canonical_bytes,
    canonical_sha256,
)
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.data.editing_v2_process_v2_active8_admission import (
    ProductionProcessV2BatchedTeacherSupportChecker,
    ProductionProcessV2SemanticExactCandidateChecker,
    SemanticActive8AdmissionError,
    build_process_v2_semantic_active8_admission_policy,
    validate_process_v2_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_process_v2_teacher_admission import (
    ProductionProcessV2FamilyTeacherAdmissionChecker,
)
from compose_v4.data.editing_v2_semantic_active8_admission import (
    SemanticActive8AdmissionError as V1SemanticActive8AdmissionError,
    build_semantic_active8_admission_policy,
    validate_semantic_active8_admission_policy,
)
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    FrozenGateZeroSemanticContract,
    build_gate_zero_process_v2_contract,
)
from compose_v4.experiments.production_successor_kernel import canonical_successor_result
from compose_v4.experiments.reference_successor_kernel import (
    enumerate_uniform_marked_masses,
    reference_successor_batch,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchModelConfig,
    build_semantic_scratch_runtime,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_process_v2_identity,
    editing_v2_process_identity,
)
from compose_v4.rewrite.action_codec_v4 import ACTIVE8_EXECUTOR_RULES
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (str(_REPO_ROOT / "tests"), str(_REPO_ROOT / "scripts")):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import process_v2_genuine_transitions as genuine_fixture  # noqa: E402
import test_process_v2_chunk_fed_rebind as rebind_fixture  # noqa: E402

ROOT = _REPO_ROOT

FROZEN_V1_PROCESS_IDENTITY = (
    "6c4721f0dd37132aae657e7aa5f1bfc01cef270662f228171c4587eb7dd48491"
)
FROZEN_V1_ACTIVE8_POLICY = (
    "34f3f7ed77bbf0a007399a4e39d2361c9351f34ae1c641377ce2f8d3e09c02a0"
)
FROZEN_V2_PROCESS_IDENTITY = (
    "0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd"
)


# ---- Fixture -------------------------------------------------------------------


def _runtime():
    payload = build_gate_zero_process_v2_contract()
    contract = FrozenGateZeroSemanticContract(
        source=Path("configs/editing_v2_semantic_process_v2.json"),
        payload=payload,
        file_sha256="0" * 64,
    )
    config = SemanticScratchModelConfig(
        initialization_seed=104729,
        max_atoms=40,
        hidden_dim=32,
        message_passing_steps=2,
        mark_dim=32,
        dtype="torch.float32",
        atom_vocabulary_class_count=15,
        catalog_fingerprint="639ff6078c32d43c",
    )
    return build_semantic_scratch_runtime(config, contract)


class _Stage:
    def __init__(self, tmp_path: Path, *, prefix: str = "/artifacts/active8_fixture"):
        payload, _binding, cache_plan, cache_completion = rebind_fixture._cached_payload(
            tmp_path
        )
        rebind_plan = rebind_fixture._cache_fed_plan(payload, cache_plan)
        rebind_fixture._execute_and_reduce(payload, rebind_plan)
        rebind_completion = json.loads(
            (
                payload.artifact_root
                / str(rebind_plan["run_artifact_root"]).removeprefix("/artifacts/")
                / "PROCESS_V2_REBIND_COMPLETE.json"
            ).read_bytes()
        )
        admitted = resolve_process_v2_admitted_source(
            rebind_plan, artifact_root=payload.artifact_root, repo_root=ROOT
        )
        self.artifact_root = payload.artifact_root
        self.rebind_plan = rebind_plan
        self.runtime = _runtime()
        self.binding = build_process_v2_active8_binding(
            cache_completion=cache_completion,
            rebind_plan=rebind_plan,
            rebind_completion=rebind_completion,
            admitted_source_identity=admitted.identity(),
            model_runtime=model_runtime_descriptor(self.runtime),
            repo_root=ROOT,
        )
        self.plan = plan_process_v2_active8(
            self.binding, rebind_plan=rebind_plan, output_artifact_prefix=prefix
        )

    def run(self, *, reuse: bool = True):
        return run_process_v2_active8_tasks(
            self.plan,
            runtime=self.runtime,
            artifact_root=self.artifact_root,
            repo_root=ROOT,
            reuse=reuse,
        )

    def reduce(self, *, publish: bool = True):
        return reduce_process_v2_active8(
            self.plan,
            runtime=self.runtime,
            artifact_root=self.artifact_root,
            repo_root=ROOT,
            publish=publish,
        )

    def rows(self) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for task in self.plan["tasks"]:
            collected.extend(
                read_task_rows(task_output_path(self.plan, task, artifact_root=self.artifact_root))
            )
        return collected

    def transitions(self) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for task in self.plan["tasks"]:
            collected.extend(
                read_task_transitions(
                    task_output_path(self.plan, task, artifact_root=self.artifact_root)
                )
            )
        return collected


@pytest.fixture(scope="module")
def stage(tmp_path_factory) -> _Stage:
    built = _Stage(tmp_path_factory.mktemp("active8") / "artifacts")
    built.run()
    return built


# ---- The frozen identities -----------------------------------------------------


def test_both_process_identities_are_the_frozen_values() -> None:
    assert (
        editing_v2_process_identity()["process_identity_sha256"]
        == FROZEN_V1_PROCESS_IDENTITY
    )
    assert (
        editing_process_v2_identity()["process_identity_sha256"]
        == FROZEN_V2_PROCESS_IDENTITY
    )


def test_active8_names_the_receipt_bound_payload_identity_when_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical cache rows are read under their identity, never the live one.

    The fixture's V1 identity happens to remain current, so an omitted
    ``expected_process_identity`` would otherwise stay invisible.  This guard
    makes the production call name the complete object carried by the exact
    rebind receipt, which is the path required by a genuinely superseded
    production payload.
    """

    import compose_v4.data.editing_v2_process_v2_active8_map as map_module

    built = _Stage(tmp_path / "explicit-payload-identity")
    expected = built.rebind_plan["pinned_process_identity"]
    original = map_module.read_process_v2_chunk_target
    observed: list[dict[str, Any]] = []

    def _require_explicit_identity(*args, expected_process_identity=None, **kwargs):
        assert expected_process_identity == expected
        observed.append(dict(expected_process_identity))
        return original(
            *args,
            expected_process_identity=expected_process_identity,
            **kwargs,
        )

    monkeypatch.setattr(
        map_module, "read_process_v2_chunk_target", _require_explicit_identity
    )
    built.run()
    assert len(observed) == len(built.plan["tasks"])


def test_release_sentinel_names_the_receipt_bound_payload_identity_when_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded re-enumeration uses the same historical identity contract."""

    import compose_v4.data.editing_v2_process_v2_active8_sentinel as sentinel_module

    built = _Stage(tmp_path / "sentinel-explicit-payload-identity")
    built.run()
    expected = built.rebind_plan["pinned_process_identity"]
    original = sentinel_module.read_process_v2_chunk_target
    observed: list[dict[str, Any]] = []

    def _require_explicit_identity(*args, expected_process_identity=None, **kwargs):
        assert expected_process_identity == expected
        observed.append(dict(expected_process_identity))
        return original(
            *args,
            expected_process_identity=expected_process_identity,
            **kwargs,
        )

    monkeypatch.setattr(
        sentinel_module, "read_process_v2_chunk_target", _require_explicit_identity
    )
    built.reduce()
    assert observed


def test_the_published_layout_is_the_one_gate_zero_pins(stage: _Stage) -> None:
    """Metadata and rows are separate files, at the seam's own pinned names."""

    assert RECEIPT_FILENAME == ACTIVE8_RECEIPT_FILENAME
    assert TRANSITIONS_FILENAME == ACTIVE8_DECISION_SHARD_FILENAME
    assert ROWS_FILENAME == ACTIVE8_ROW_SHARD_FILENAME
    for task in stage.plan["tasks"]:
        output = task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
        assert output.parent.name == ACTIVE8_TASKS_DIRNAME
        assert output.name == str(task["task_identity_sha256"])
        receipt = json.loads((output / RECEIPT_FILENAME).read_bytes())
        # Exactly the frozen set, not a superset: Gate 0 raises on an extra key.
        assert tuple(sorted(receipt)) == tuple(sorted(ACTIVE8_RECEIPT_FIELDS))
        assert receipt["partition_role"] == task["split"]
        assert receipt["data_lane"] == task["data_lane"]
        assert receipt["task_identity_sha256"] == task["task_identity_sha256"]
        assert isinstance(receipt["active8_accepted_entries"], int)
        assert receipt["decision_shard_sha256"] == hashlib.sha256(
            (output / TRANSITIONS_FILENAME).read_bytes()
        ).hexdigest()
        # The rows are a sibling file, never inside the receipt: a gate reading
        # the census must not be able to open held-out molecular content.
        assert "transitions" not in receipt and "actions" not in receipt
        assert (output / TRANSITIONS_FILENAME).is_file()
    census = {
        field: 0
        for field in (
            "source_entries",
            "upstream_rejected_entries",
            "active8_accepted_entries",
            "active8_excluded_entries",
        )
    }
    transitions = 0
    for task in stage.plan["tasks"]:
        output = task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
        receipt = json.loads((output / RECEIPT_FILENAME).read_bytes())
        for field in census:
            census[field] += int(receipt[field])
        transitions += int(receipt["transition_count"])
    assert census["source_entries"] == len(stage.rows())
    assert transitions == len(stage.transitions())


def test_the_stage_binds_the_process_v2_policy_and_the_v1_policy_is_untouched() -> None:
    v1 = build_semantic_active8_admission_policy()
    v2 = build_process_v2_semantic_active8_admission_policy()
    assert v1.policy_sha256 == FROZEN_V1_ACTIVE8_POLICY
    assert v1.process_identity_sha256 == FROZEN_V1_PROCESS_IDENTITY
    assert v2.process_identity_sha256 == FROZEN_V2_PROCESS_IDENTITY
    assert dict(v1.required_model_modes)["editing_process_semantics"] == (
        "semantic_editing_v2_v1"
    )
    assert dict(v2.required_model_modes)["editing_process_semantics"] == (
        "semantic_editing_v2_v2"
    )
    assert dict(v2.required_model_modes)["atom_delete_action_semantics"] == (
        "process_v2_uniform_gated_atom_delete_v2"
    )
    # The support statement is shared; schema, identity, implementation and the
    # two version-specific model modes deliberately belong to separate modules.
    v1_body = {
        key: value
        for key, value in v1.as_payload().items()
        if key
        not in {
            "schema",
            "schema_version",
            "status",
            "process_identity_sha256",
            "required_model_modes",
            "candidate_evaluator",
            "implementation_file_sha256",
            "policy_sha256",
        }
    }
    v2_body = {
        key: value
        for key, value in v2.as_payload().items()
        if key
        not in {
            "schema",
            "schema_version",
            "status",
            "process_identity_sha256",
            "required_model_modes",
            "candidate_evaluator",
            "implementation_file_sha256",
            "policy_sha256",
        }
    }
    assert v1_body == v2_body
    assert v2.schema_version == 3
    assert v2.candidate_evaluator.endswith(
        ".ProductionProcessV2FamilyTeacherAdmissionChecker"
    )
    with pytest.raises(SemanticActive8AdmissionError):
        validate_process_v2_semantic_active8_admission_policy(v1)
    with pytest.raises(V1SemanticActive8AdmissionError):
        validate_semantic_active8_admission_policy(v2)


def test_the_process_v2_policy_module_does_not_import_the_v1_policy_module() -> None:
    path = ROOT / "src/compose_v4/data/editing_v2_process_v2_active8_admission.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "compose_v4.data.editing_v2_semantic_active8_admission" not in imported


# ---- The published shapes are exactly the frozen seam --------------------------


def test_every_published_shape_is_exactly_the_frozen_seam(stage: _Stage) -> None:
    rows = stage.rows()
    assert rows
    for row in rows:
        assert set(row) == set(ACTIVE8_ROW_FIELDS)
        for action in row["actions"]:
            assert set(action) == set(ACTION_FIELDS)
            assert set(CLASSIFICATION_FIELDS) <= set(action)
            if action["candidate_evidence"] is not None:
                assert set(action["candidate_evidence"]) == set(CANDIDATE_EVIDENCE_FIELDS)
    transitions = stage.transitions()
    assert transitions
    for transition in transitions:
        assert set(transition) == set(ACCEPTED_TRANSITION_FIELDS)
        assert set(transition["candidate_evidence"]) == set(CANDIDATE_EVIDENCE_FIELDS)


def test_the_stage_publishes_both_rejection_categories(stage: _Stage) -> None:
    """The fixture is built to exercise both, so both must appear."""

    categories = {row["rejection_category"] for row in stage.rows()}
    assert UPSTREAM_REJECTED in categories
    assert None in categories
    upstream = [row for row in stage.rows() if row["rejection_category"] == UPSTREAM_REJECTED]
    assert upstream
    for row in upstream:
        assert row["admission_status"] == NOT_EVALUATED
        assert row["upstream_rejection_code"]
        assert row["actions"] == []
    for row in stage.rows():
        if row["admission_status"] == ACCEPTED:
            assert row["rejection_category"] is None
            assert row["upstream_rejection_code"] is None
        elif row["rejection_category"] == ACTIVE8_EXCLUDED:
            assert row["admission_status"] == "excluded"


def test_supported_and_exclusion_reason_cross_the_seam(stage: _Stage) -> None:
    published = [
        action["candidate_evidence"]
        for row in stage.rows()
        for action in row["actions"]
        if action["candidate_evidence"] is not None
    ]
    assert published
    for evidence in published:
        assert isinstance(evidence["supported"], bool)
        assert (evidence["exclusion_reason"] is None) is evidence["supported"]


def test_the_cell_and_the_raw_axes_travel_together(stage: _Stage) -> None:
    transitions = stage.transitions()
    assert transitions
    assert all(transition["capability_cell_id"] for transition in transitions)
    for transition in transitions:
        assert transition["capability_cell_id"] == (
            f"editing_v2_active8_v1:{transition['model_family']}:"
            f"{transition['family_context']}"
        )
        assert transition["terminal"] is False
        assert transition["progress_index"] == transition["step_index"]
    rows = {(row["v1_task_identity_sha256"], row["entry_index"]): row for row in stage.rows()}
    for transition in transitions:
        row = rows[(transition["v1_task_identity_sha256"], transition["entry_index"])]
        action = row["actions"][transition["step_index"]]
        assert action["audit_axes"] == transition["audit_axes"]
        assert isinstance(action["audit_cycle_rank_delta"], int)
        assert isinstance(action["audit_touches_ring_system"], bool)
        assert action["audit_is_terminal_source"] is False


# ---- Classification is derived annotation --------------------------------------


def test_a_constant_classifier_does_not_move_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proof, run twice for real: only the annotation half may differ."""

    honest = _Stage(tmp_path / "honest", prefix="/artifacts/active8_honest")
    honest.run()
    honest_rows = honest.rows()

    import compose_v4.data.editing_v2_process_v2_active8_map as map_module

    monkeypatch.setattr(
        map_module,
        "classify_action_family_context",
        lambda source, successor, step: ("atom_insert", "root_birth"),
    )
    stubbed = _Stage(tmp_path / "stubbed", prefix="/artifacts/active8_stubbed")
    stubbed.run()
    stubbed_rows = stubbed.rows()

    assert len(honest_rows) == len(stubbed_rows)
    annotation = set(CLASSIFICATION_FIELDS) - {"model_family"}
    for left, right in zip(honest_rows, stubbed_rows, strict=True):
        assert left["admission_status"] == right["admission_status"]
        assert left["rejection_category"] == right["rejection_category"]
        assert left["upstream_rejection_code"] == right["upstream_rejection_code"]
        assert left["candidate_totals"] == right["candidate_totals"]
        assert left["action_family_histogram"] == right["action_family_histogram"]
        assert len(left["actions"]) == len(right["actions"])
        for a, b in zip(left["actions"], right["actions"], strict=True):
            assert a["candidate_evidence"] == b["candidate_evidence"]
            assert a["executor_rule"] == b["executor_rule"]
            assert a["classification_affects_admission"] is False
            assert b["classification_affects_admission"] is False
            assert {key: a[key] for key in ACTION_FIELDS if key not in annotation} == {
                key: b[key] for key in ACTION_FIELDS if key not in annotation
            }
    # And the stub really did reach the annotation, so this is not a null test.
    stubbed_cells = {
        action["capability_cell_id"]
        for row in stubbed_rows
        for action in row["actions"]
        if action["capability_cell_id"] is not None
    }
    honest_cells = {
        action["capability_cell_id"]
        for row in honest_rows
        for action in row["actions"]
        if action["capability_cell_id"] is not None
    }
    assert honest_cells
    assert stubbed_cells != honest_cells
    assert stubbed_cells <= {"editing_v2_active8_v1:atom_insert:root_birth"}


def test_a_raising_classifier_still_publishes_the_admission_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import compose_v4.data.editing_v2_process_v2_active8_map as map_module

    def _boom(source, successor, step):
        raise map_module.SemanticCapabilityCellError("annotation refused")

    monkeypatch.setattr(map_module, "classify_action_family_context", _boom)
    broken = _Stage(tmp_path / "broken", prefix="/artifacts/active8_broken")
    broken.run()
    rows = broken.rows()
    assert any(row["admission_status"] == ACCEPTED for row in rows)
    assert all(
        action["capability_cell_id"] is None
        for row in rows
        for action in row["actions"]
    )


# ---- Aggregates are derived, never carried -------------------------------------


def test_a_resealed_row_whose_total_moved_is_refused(stage: _Stage) -> None:
    task = stage.plan["tasks"][0]
    output = task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
    original = (output / ROWS_FILENAME).read_bytes()
    rows = read_task_rows(output)
    target = next(
        index for index, row in enumerate(rows) if row["candidate_totals"]["evaluated_teachers"]
    )
    tampered = dict(rows[target])
    totals = dict(tampered["candidate_totals"])
    totals["evaluated_teachers"] -= 1
    tampered["candidate_totals"] = totals
    body = {key: value for key, value in tampered.items() if key != "row_sha256"}
    tampered["row_sha256"] = canonical_sha256(body)
    rows[target] = tampered
    try:
        with (output / ROWS_FILENAME).open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                for row in rows:
                    handle.write(canonical_bytes(row) + b"\n")
        with pytest.raises(ProcessV2Active8MapError):
            read_task_rows(output)
    finally:
        (output / ROWS_FILENAME).write_bytes(original)


@pytest.mark.parametrize(
    "field,value",
    (
        ("teacher_coordinate_legal", False),
        ("teacher_executes_to_exact_successor", False),
        ("productive_canonical_successor", False),
        ("source_canonical_key", "K"),
    ),
)
def test_each_frozen_evidence_invariant_is_enforced(field: str, value: object) -> None:
    """Fed a violating payload, not compared against a recomputed expectation."""

    payload = {
        "supported": True,
        "exclusion_reason": None,
        "action_sha256": "a" * 64,
        "source_state_sha256": "b" * 64,
        "target_state_sha256": "c" * 64,
        "source_canonical_key": "SOURCE",
        "canonical_successor_key": "K",
        "teacher_coordinate_legal": True,
        "teacher_executes_to_exact_successor": True,
        "productive_canonical_successor": True,
    }
    require_accepted_evidence_invariants(payload)
    with pytest.raises(ProcessV2Active8MapError, match="frozen invariant"):
        require_accepted_evidence_invariants({**payload, field: value})


def test_the_totals_equal_the_evidence_they_are_derived_from(stage: _Stage) -> None:
    for row in stage.rows():
        assert row["candidate_totals"] == derive_candidate_totals(row["actions"])


# ---- Reduction: deterministic, complete, restart safe ---------------------------


def test_the_reduction_is_independent_of_task_order(tmp_path: Path) -> None:
    forward = _Stage(tmp_path / "forward", prefix="/artifacts/active8_forward")
    forward.run()
    first = forward.reduce(publish=False)

    reversed_stage = _Stage(tmp_path / "reversed", prefix="/artifacts/active8_forward")
    for task in reversed(reversed_stage.plan["tasks"]):
        execute_process_v2_active8_task(
            reversed_stage.plan,
            task["task_identity_sha256"],
            runtime=reversed_stage.runtime,
            artifact_root=reversed_stage.artifact_root,
            repo_root=ROOT,
        )
    second = reversed_stage.reduce(publish=False)
    # The two payloads address different artifact roots but the same corpus, so
    # everything except the per-run receipt inventory must agree exactly.
    ignored = {"result_inventory", "result_inventory_sha256", "completion_sha256"}
    assert {k: v for k, v in first.items() if k not in ignored} == {
        k: v for k, v in second.items() if k not in ignored
    }


def test_an_incomplete_reduction_publishes_nothing(tmp_path: Path) -> None:
    stage = _Stage(tmp_path / "incomplete", prefix="/artifacts/active8_incomplete")
    stage.run()
    victim = stage.plan["tasks"][-1]
    shutil.rmtree(task_output_path(stage.plan, victim, artifact_root=stage.artifact_root))
    with pytest.raises(ProcessV2Active8Incomplete):
        stage.reduce()
    run_root = stage.artifact_root / str(stage.plan["run_artifact_root"]).removeprefix(
        "/artifacts/"
    )
    assert not (run_root / COMPLETION_FILENAME).exists()


def test_a_restart_reuses_exact_outputs_and_refuses_a_corrupt_target(
    tmp_path: Path,
) -> None:
    stage = _Stage(tmp_path / "restart", prefix="/artifacts/active8_restart")
    first = stage.run()
    assert completed_process_v2_active8_task_ids(
        stage.plan, artifact_root=stage.artifact_root
    ) == set(first)

    reused_task = stage.plan["tasks"][0]
    damaged_task = stage.plan["tasks"][-1]
    damaged = task_output_path(stage.plan, damaged_task, artifact_root=stage.artifact_root)
    receipt = json.loads((damaged / RECEIPT_FILENAME).read_bytes())
    receipt["source_entries"] = 999
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = canonical_sha256(body)
    (damaged / RECEIPT_FILENAME).write_bytes(canonical_bytes(receipt) + b"\n")
    damaged_bytes = {
        name: (damaged / name).read_bytes()
        for name in (
            ROWS_FILENAME,
            TRANSITIONS_FILENAME,
            RECEIPT_FILENAME,
            SUMMARY_FILENAME,
        )
    }
    assert str(damaged_task["task_identity_sha256"]) not in (
        completed_process_v2_active8_task_ids(stage.plan, artifact_root=stage.artifact_root)
    )

    reused_dir = task_output_path(stage.plan, reused_task, artifact_root=stage.artifact_root)
    before = (reused_dir / ROWS_FILENAME).stat().st_mtime_ns
    with pytest.raises(
        ProcessV2Active8MapError,
        match="refusing to overwrite an invalid existing Active8 task output",
    ):
        stage.run()
    assert (reused_dir / ROWS_FILENAME).stat().st_mtime_ns == before
    assert {
        name: (damaged / name).read_bytes()
        for name in (
            ROWS_FILENAME,
            TRANSITIONS_FILENAME,
            RECEIPT_FILENAME,
            SUMMARY_FILENAME,
        )
    } == damaged_bytes
    with pytest.raises(ProcessV2Active8MapError):
        validate_process_v2_active8_task_result(damaged)


def test_one_chunk_reports_progress_and_an_exact_restart_reuses_it(
    stage: _Stage,
) -> None:
    task = next(
        task
        for task in stage.plan["tasks"]
        if sum(
            validate_process_v2_active8_task_result(
                task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
            )[1]["action_family_histogram"].values()
        )
        > 0
    )
    progress: list[tuple[int, int]] = []
    execute_process_v2_active8_task(
        stage.plan,
        task["task_identity_sha256"],
        runtime=stage.runtime,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        reuse=False,
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )
    assert progress
    assert progress[-1][0] == progress[-1][1]
    assert all(
        0 < processed <= total == progress[-1][1] for processed, total in progress
    )

    reused_progress: list[tuple[int, int]] = []
    execute_process_v2_active8_task(
        stage.plan,
        task["task_identity_sha256"],
        runtime=stage.runtime,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        progress_callback=lambda processed, total: reused_progress.append(
            (processed, total)
        ),
    )
    assert reused_progress == []


def test_the_completion_is_published_only_after_the_sentinel_passes(
    tmp_path: Path,
) -> None:
    stage = _Stage(tmp_path / "complete", prefix="/artifacts/active8_complete")
    stage.run()
    completion = stage.reduce()
    assert completion["sentinel"]["status"] == SENTINEL_PASSED
    assert completion["census"]["source_entries"] == (
        completion["census"]["upstream_rejected_entries"]
        + completion["census"]["active8_accepted_entries"]
        + completion["census"]["active8_excluded_entries"]
    )
    reloaded = load_process_v2_active8_completion(
        str(stage.plan["run_artifact_root"]), artifact_root=stage.artifact_root
    )
    assert reloaded == completion


# ---- The release sentinel -------------------------------------------------------


def test_the_sentinel_is_exhaustive_on_a_small_corpus_and_reports_the_seam_fields(
    tmp_path: Path,
) -> None:
    stage = _Stage(tmp_path / "sentinel", prefix="/artifacts/active8_sentinel")
    stage.run()
    completion = stage.reduce(publish=False)
    sentinel = completion["sentinel"]
    assert set(sentinel) == set(SENTINEL_RESULT_FIELDS)
    assert sentinel["salt"] == SENTINEL_SALT
    assert sentinel["selection_mode"] == SELECTION_EXHAUSTIVE
    assert sentinel["selected_pairs"] == sentinel["unique_accepted_pairs"] > 0
    assert sentinel["oracle_examples"] == sentinel["selected_pairs"]
    assert sentinel["evidence_mismatches"] == []
    assert sentinel["cell_mismatches"] == []
    assert sentinel["oracle_mismatches"] == []


def test_a_resealed_false_state_identity_is_caught_by_the_sentinel(
    tmp_path: Path,
) -> None:
    """The residual the sentinel exists for: internally consistent but false.

    The tampered row satisfies every logical invariant, its own self-hash, its
    derived totals and its receipt summary, so the whole reduction reconciles.
    Only the sentinel's molecular re-entry sees it.
    """

    stage = _Stage(tmp_path / "residual", prefix="/artifacts/active8_residual")
    stage.run()
    task = next(
        task
        for task in stage.plan["tasks"]
        if any(
            row["admission_status"] == ACCEPTED
            for row in read_task_rows(
                task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
            )
        )
    )
    output = task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
    _replace_target_state_sha256(output)
    # The task still reconciles completely from its own bytes ...
    validate_process_v2_active8_task_result(output)
    # ... and only the sentinel refuses it.
    with pytest.raises(ProcessV2Active8SentinelError, match="release sentinel"):
        stage.reduce()
    run_root = stage.artifact_root / str(stage.plan["run_artifact_root"]).removeprefix(
        "/artifacts/"
    )
    assert not (run_root / COMPLETION_FILENAME).exists()


def _replace_target_state_sha256(output: Path) -> None:
    """Replace one accepted action's target-state digest and reseal everything."""

    rows = read_task_rows(output)
    transitions = read_task_transitions(output)
    target = next(
        index for index, row in enumerate(rows) if row["admission_status"] == ACCEPTED
    )
    row = json.loads(json.dumps(rows[target]))
    action = row["actions"][0]
    original_target = action["candidate_evidence"]["target_state_sha256"]
    action["candidate_evidence"]["target_state_sha256"] = (
        "0" * 64 if original_target != "0" * 64 else "1" * 64
    )
    row["candidate_totals"] = derive_candidate_totals(row["actions"])
    body = {key: value for key, value in row.items() if key != "row_sha256"}
    row["row_sha256"] = canonical_sha256(body)
    rows[target] = row
    resealed: list[dict[str, Any]] = []
    for transition in transitions:
        if (
            transition["v1_task_identity_sha256"] == row["v1_task_identity_sha256"]
            and transition["entry_index"] == row["entry_index"]
            and transition["step_index"] == action["step_index"]
        ):
            transition = json.loads(json.dumps(transition))
            transition["candidate_evidence"] = dict(action["candidate_evidence"])
            transition_body = {
                key: value
                for key, value in transition.items()
                if key != "assignment_sha256"
            }
            transition["assignment_sha256"] = canonical_sha256(transition_body)
        resealed.append(transition)
    receipt = json.loads((output / RECEIPT_FILENAME).read_bytes())
    summary = json.loads((output / SUMMARY_FILENAME).read_bytes())
    _rewrite(output / ROWS_FILENAME, rows)
    _rewrite(output / TRANSITIONS_FILENAME, resealed)
    measured = summarize_task(
        rows,
        resealed,
        entry_start=int(summary["entry_start"]),
        entry_stop=int(summary["entry_stop"]),
        task_identity_sha256=str(receipt["task_identity_sha256"]),
    )
    receipt = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt.update(
        {field: measured[field] for field in ACTIVE8_RECEIPT_FIELDS if field in measured}
    )
    receipt["decision_shard_sha256"] = hashlib.sha256(
        (output / TRANSITIONS_FILENAME).read_bytes()
    ).hexdigest()
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (output / RECEIPT_FILENAME).write_bytes(canonical_bytes(receipt) + b"\n")
    summary = {key: value for key, value in summary.items() if key != "summary_sha256"}
    summary.update(
        {field: measured[field] for field in tuple(summary) if field in measured}
    )
    summary["receipt_sha256"] = receipt["receipt_sha256"]
    summary["rows_file_sha256"] = hashlib.sha256(
        (output / ROWS_FILENAME).read_bytes()
    ).hexdigest()
    summary["rows_stream_sha256"] = _stream_sha256(rows)
    summary["transitions_stream_sha256"] = _stream_sha256(resealed)
    summary["summary_sha256"] = canonical_sha256(summary)
    (output / SUMMARY_FILENAME).write_bytes(canonical_bytes(summary) + b"\n")


def _rewrite(path: Path, rows) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in rows:
                handle.write(canonical_bytes(row) + b"\n")


def _stream_sha256(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_bytes(row) + b"\n")
    return digest.hexdigest()


def test_the_sentinel_rank_is_the_frozen_salted_digest() -> None:
    import hashlib

    expected = hashlib.sha256(
        b"process_v2_active8_release_sentinel_v1\x00" + b"a" * 64 + b"\x00" + b"b" * 64
    ).hexdigest()
    assert sentinel_rank("a" * 64, "b" * 64) == expected
    assert SENTINEL_SALT == "process_v2_active8_release_sentinel_v1"


def test_require_sentinel_passed_refuses_resealed_semantic_mutations(stage: _Stage) -> None:
    completion = stage.reduce(publish=False)
    original = completion["sentinel"]
    require_sentinel_passed(
        original,
        binding_sha256=completion["binding_sha256"],
        plan_sha256=completion["plan_sha256"],
        run_identity_sha256=completion["run_identity_sha256"],
        task_inventory_sha256=completion["task_inventory_sha256"],
        result_inventory_sha256=completion["result_inventory_sha256"],
    )
    mutations = (
        {"schema_version": 999},
        {"salt": "another-salt"},
        {"selection_mode": "unknown-selection"},
        {"evidence_mismatches": ["real mismatch"]},
        {"plan_sha256": "f" * 64},
        {"selected_pairs": int(original["unique_accepted_pairs"]) + 1},
    )
    for changes in mutations:
        body = {key: value for key, value in original.items() if key != "sentinel_sha256"}
        body.update(changes)
        sentinel = {**body, "sentinel_sha256": canonical_sha256(body)}
        with pytest.raises(ProcessV2Active8SentinelError):
            require_sentinel_passed(
                sentinel,
                binding_sha256=completion["binding_sha256"],
                plan_sha256=completion["plan_sha256"],
                run_identity_sha256=completion["run_identity_sha256"],
                task_inventory_sha256=completion["task_inventory_sha256"],
                result_inventory_sha256=completion["result_inventory_sha256"],
            )


# ---- The bounded dictionary oracle is checked by the repository's own oracle ----


def test_the_bounded_dictionary_oracle_agrees_with_the_reference_oracle(
    stage: _Stage,
) -> None:
    """The instrument is verified by the oracle production may not import."""

    checker = ProductionProcessV2SemanticExactCandidateChecker(
        stage.runtime.model, policy=build_process_v2_semantic_active8_admission_policy()
    )
    states = []
    for task in stage.plan["tasks"]:
        for row in read_task_rows(
            task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
        ):
            if row["admission_status"] == ACCEPTED:
                states.append((task, row))
    assert states
    checked = 0
    for task, row in states[:2]:
        for read in _chunk_reads(stage, task):
            if int(read.entry_index) != int(row["entry_index"]) or read.addressed is None:
                continue
            state = read.addressed.path.state_at(0)
            result = canonical_successor_result(
                checker.model, state, checker.time, system=checker.system
            )
            instrument = _dictionary_successor_oracle(result, state, system=checker.system)
            reference = reference_successor_batch(
                state,
                enumerate_uniform_marked_masses(
                    [(mark.executor_rule_name, mark.action) for mark in result.marked_law.marks]
                ),
                system=checker.system,
            )
            assert instrument == {
                successor.key: successor.alias_count for successor in reference.successors
            }
            checked += 1
    assert checked


def _chunk_reads(stage: _Stage, task):
    from compose_v4.data.editing_v2_process_v2_active8_map import _chunk_target
    from compose_v4.data.editing_v2_process_v2_chunk_cache import (
        read_process_v2_chunk_target,
    )
    from compose_v4.data.editing_process_v2_rebind import mounted_process_v2_artifact_path

    target = _chunk_target(task)
    return read_process_v2_chunk_target(
        mounted_process_v2_artifact_path(
            target.source_artifact_path,
            artifact_root=stage.artifact_root,
            field="task.cache_source_artifact_path",
        ),
        target=target,
        sentinel_replay_entries=0,
        recover_row_errors=False,
        repo_root=ROOT,
    )


def _accepted_teacher_queries(stage: _Stage):
    accepted = {
        (str(row["v1_task_identity_sha256"]), int(row["entry_index"]))
        for row in stage.rows()
        if row["admission_status"] == ACCEPTED
    }
    queries = []
    for task in stage.plan["tasks"]:
        task_key = str(task["v1_task_identity_sha256"])
        for read in _chunk_reads(stage, task):
            if (
                (task_key, int(read.entry_index)) not in accepted
                or read.addressed is None
            ):
                continue
            queries.extend(
                (read.addressed, step_index)
                for step_index in range(len(read.addressed.trace.steps))
            )
    return tuple(queries)


def _genuine_teacher_queries() -> tuple[tuple[AddressedPackedTrace, int], ...]:
    """One real, executed one-step teacher for every discovered capability cell."""

    discovered = genuine_fixture.discover_genuine_transitions()
    assert len(discovered) == 22
    queries: list[tuple[AddressedPackedTrace, int]] = []
    for entry_index, cell in enumerate(sorted(discovered, reverse=True)):
        item = discovered[cell]
        step = RewriteStep(item.executor_rule, item.action)
        trace = RewriteTrace(
            source=item.source_state,
            target=item.successor_state,
            steps=(step,),
            metadata={"fixture": "process-v2-genuine-capability-cell", "cell": cell},
        )
        addressed = AddressedPackedTrace(
            address=PackedTraceAddress(
                packed_shard_content_sha256="e" * 64,
                packed_shard_name="process-v2-genuine-cells.jsonl.gz",
                entry_index=entry_index,
                trace_id=f"genuine-cell-{entry_index}",
                layer="reversible_synthetic_walk",
                partition="validation",
                source_key=canonical_state_key(item.source_state),
                target_key=item.canonical_successor_key,
                path_length=1,
            ),
            trace=trace,
            path=PackedTraceProgress(
                trace,
                (item.source_state, item.successor_state),
            ),
        )
        queries.append((addressed, 0))
    return tuple(queries)


def test_batched_teacher_support_matches_the_slow_full_quotient(stage: _Stage) -> None:
    queries = _accepted_teacher_queries(stage)
    assert queries
    policy = build_process_v2_semantic_active8_admission_policy()
    fast = ProductionProcessV2BatchedTeacherSupportChecker(
        stage.runtime.model,
        policy=policy,
        batch_size=4,
    ).evaluate_many(queries)
    slow_checker = ProductionProcessV2SemanticExactCandidateChecker(
        stage.runtime.model,
        policy=policy,
    )
    slow = tuple(slow_checker.evaluate(*query) for query in queries)
    assert len(fast) == len(slow)
    for observed, expected in zip(fast, slow, strict=True):
        evidence = expected.evidence
        assert observed.supported == evidence.supported
        assert observed.exclusion_reason == evidence.exclusion_reason
        assert observed.action_sha256 == evidence.action_sha256
        assert observed.source_state_sha256 == evidence.source_state_sha256
        assert observed.target_state_sha256 == evidence.target_state_sha256
        assert observed.canonical_successor_key == evidence.canonical_successor_key
        assert observed.raw_mark_count == evidence.raw_mark_count
        assert observed.matching_mark_count == evidence.matching_mark_count
        assert observed.exact_successor_mark_count == expected.exact_successor_mark_count


def test_batched_teacher_support_matches_all_genuine_cells_in_reversed_order(
    stage: _Stage,
) -> None:
    queries = _genuine_teacher_queries()
    assert len(queries) == 22
    policy = build_process_v2_semantic_active8_admission_policy()
    observed = ProductionProcessV2BatchedTeacherSupportChecker(
        stage.runtime.model,
        policy=policy,
        batch_size=3,
    ).evaluate_many(queries)
    slow_checker = ProductionProcessV2SemanticExactCandidateChecker(
        stage.runtime.model,
        policy=policy,
    )
    expected = tuple(slow_checker.evaluate(*query) for query in queries)

    assert {query[0].trace.steps[0].rule_name for query in queries} == set(
        ACTIVE8_EXECUTOR_RULES
    )
    assert len(observed) == len(expected)
    for fast, slow in zip(observed, expected, strict=True):
        evidence = slow.evidence
        assert fast.supported == evidence.supported
        assert fast.action_sha256 == evidence.action_sha256
        assert fast.source_state_sha256 == evidence.source_state_sha256
        assert fast.target_state_sha256 == evidence.target_state_sha256
        assert fast.source_canonical_key == evidence.source_canonical_key
        assert fast.canonical_successor_key == evidence.canonical_successor_key
        assert fast.raw_mark_count == evidence.raw_mark_count
        assert fast.matching_mark_count == evidence.matching_mark_count
        assert fast.exact_successor_mark_count == slow.exact_successor_mark_count
        if evidence.matching_mark_count == 0:
            assert evidence.exclusion_reason == (
                "teacher_action_absent_from_production_marked_law"
            )
            assert fast.exclusion_reason == (
                "teacher_action_not_a_unique_process_v2_coordinate"
            )
        else:
            assert fast.exclusion_reason == evidence.exclusion_reason


def test_family_teacher_admission_matches_all_genuine_cells(
    stage: _Stage,
) -> None:
    queries = _genuine_teacher_queries()
    assert len(queries) == 22
    policy = build_process_v2_semantic_active8_admission_policy()
    observed = ProductionProcessV2FamilyTeacherAdmissionChecker(
        stage.runtime.model,
        policy=policy,
    ).evaluate_many(queries)
    expected = ProductionProcessV2BatchedTeacherSupportChecker(
        stage.runtime.model,
        policy=policy,
        batch_size=3,
    ).evaluate_many(queries)

    assert len(observed) == len(expected)
    for query, direct, exhaustive in zip(queries, observed, expected, strict=True):
        rule = query[0].trace.steps[0].rule_name
        assert direct.supported == exhaustive.supported, rule
        assert direct.exclusion_reason == exhaustive.exclusion_reason, rule
        assert direct.action_sha256 == exhaustive.action_sha256
        assert direct.source_state_sha256 == exhaustive.source_state_sha256
        assert direct.target_state_sha256 == exhaustive.target_state_sha256
        assert direct.source_canonical_key == exhaustive.source_canonical_key
        assert direct.canonical_successor_key == exhaustive.canonical_successor_key
        assert direct.teacher_coordinate_legal == exhaustive.teacher_coordinate_legal
        assert (
            direct.teacher_executes_to_exact_successor
            == exhaustive.teacher_executes_to_exact_successor
        )
        assert (
            direct.productive_canonical_successor
            == exhaustive.productive_canonical_successor
        )


def _replace_one_step_query(
    query: tuple[AddressedPackedTrace, int],
    *,
    step: RewriteStep | None = None,
    target=None,
) -> tuple[AddressedPackedTrace, int]:
    addressed, step_index = query
    assert step_index == 0
    selected_step = step or addressed.trace.steps[0]
    selected_target = target or addressed.path.state_at(1)
    trace = RewriteTrace(
        source=addressed.path.state_at(0),
        target=selected_target,
        steps=(selected_step,),
        metadata=dict(addressed.trace.metadata),
    )
    return (
        AddressedPackedTrace(
            address=addressed.address,
            trace=trace,
            path=PackedTraceProgress(trace, (trace.source, trace.target)),
        ),
        0,
    )


def test_family_teacher_admission_rejects_an_exact_successor_mismatch(
    stage: _Stage,
) -> None:
    queries = _genuine_teacher_queries()
    original = queries[0]
    replacement_target = next(
        query[0].path.state_at(1)
        for query in queries[1:]
        if query[0].path.state_at(1).n_atoms == original[0].path.state_at(1).n_atoms
    )
    query = _replace_one_step_query(original, target=replacement_target)
    observed = ProductionProcessV2FamilyTeacherAdmissionChecker(
        stage.runtime.model,
    ).evaluate(*query)
    assert observed.supported is False
    assert observed.teacher_coordinate_legal is True
    assert observed.teacher_executes_to_exact_successor is False
    assert observed.exclusion_reason == "teacher_action_does_not_reproduce_exact_successor"


def test_family_teacher_admission_rejects_a_noncanonical_insert_slot(
    stage: _Stage,
) -> None:
    original = next(
        query
        for query in _genuine_teacher_queries()
        if query[0].trace.steps[0].rule_name == "atom_insert"
    )
    action = original[0].trace.steps[0].action
    assert isinstance(action, AtomInsert)
    null_slots = tuple(
        int(slot)
        for slot in np.flatnonzero(original[0].path.state_at(0).atom_types == 0)
    )
    assert len(null_slots) >= 2 and action.slot == null_slots[0]
    changed = AtomInsert(
        slot=null_slots[1],
        atom_type=action.atom_type,
        formal_charge=action.formal_charge,
        implicit_h_count=action.implicit_h_count,
        neighbors=action.neighbors,
    )
    query = _replace_one_step_query(
        original,
        step=RewriteStep("atom_insert", changed),
    )
    observed = ProductionProcessV2FamilyTeacherAdmissionChecker(
        stage.runtime.model,
    ).evaluate(*query)
    assert observed.supported is False
    assert observed.teacher_coordinate_legal is False
    assert observed.exclusion_reason == "teacher_action_not_a_unique_process_v2_coordinate"


def test_family_teacher_admission_reports_bounded_monotone_progress(
    stage: _Stage,
) -> None:
    queries = tuple(_genuine_teacher_queries()) * 4
    progress: list[tuple[int, int]] = []
    checker = ProductionProcessV2FamilyTeacherAdmissionChecker(stage.runtime.model)
    expected = checker.evaluate_many(queries)
    observed = checker.evaluate_many(
        queries,
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )

    assert observed == expected
    assert progress == [(64, len(queries)), (len(queries), len(queries))]


def test_batched_teacher_support_is_a_callable_checker(stage: _Stage) -> None:
    query = _accepted_teacher_queries(stage)[0]
    checker = ProductionProcessV2BatchedTeacherSupportChecker(
        stage.runtime.model,
        batch_size=2,
    )
    assert checker(*query) == checker.evaluate(*query)


def test_batched_teacher_support_reports_monotone_progress_without_changing_results(
    stage: _Stage,
) -> None:
    queries = _accepted_teacher_queries(stage)[:7]
    assert len(queries) == 7
    checker = ProductionProcessV2BatchedTeacherSupportChecker(
        stage.runtime.model,
        batch_size=3,
    )
    expected = checker.evaluate_many(queries)
    progress: list[tuple[int, int]] = []
    observed = checker.evaluate_many(
        queries,
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )

    assert observed == expected
    assert progress[-1] == (len(queries), len(queries))
    assert [processed for processed, _total in progress] == sorted(
        processed for processed, _total in progress
    )
    assert all(total == len(queries) for _processed, total in progress)


@pytest.mark.parametrize("table_name", ("ring_system_grow", "ring_system_delete"))
def test_batched_teacher_support_refuses_nonzero_disabled_family_masks(
    stage: _Stage,
    monkeypatch: pytest.MonkeyPatch,
    table_name: str,
) -> None:
    query = _accepted_teacher_queries(stage)[0]
    model_type = type(stage.runtime.model)
    real_action_tables = model_type._action_tables

    def action_tables_with_disabled_support(model, *args, **kwargs):
        masks, logits, partitions = real_action_tables(model, *args, **kwargs)
        masks = dict(masks)
        injected = masks[table_name].clone()
        injected.reshape(-1)[0] = True
        masks[table_name] = injected
        return masks, logits, partitions

    monkeypatch.setattr(model_type, "_action_tables", action_tables_with_disabled_support)
    checker = ProductionProcessV2BatchedTeacherSupportChecker(
        stage.runtime.model,
        batch_size=2,
    )
    with pytest.raises(SemanticActive8AdmissionError, match="disabled"):
        checker.evaluate_many((query,))


def test_batched_teacher_support_never_builds_a_canonical_quotient(
    stage: _Stage, monkeypatch
) -> None:
    queries = _accepted_teacher_queries(stage)[:4]
    assert queries

    def forbidden(*_args, **_kwargs):
        raise AssertionError("whole-corpus teacher support built a canonical quotient")

    monkeypatch.setattr(
        "compose_v4.data.editing_v2_process_v2_active8_admission.canonical_successor_result",
        forbidden,
    )
    observed = ProductionProcessV2BatchedTeacherSupportChecker(
        stage.runtime.model,
        batch_size=4,
    ).evaluate_many(queries)
    assert all(item.supported for item in observed)


# ---- The plan ------------------------------------------------------------------


def test_the_plan_is_one_task_per_source_chunk(stage: _Stage) -> None:
    assert len(stage.plan["tasks"]) == len(stage.rebind_plan["tasks"])
    chunks = {
        (task["cache_source_task_identity_sha256"], task["chunk_index"])
        for task in stage.plan["tasks"]
    }
    assert len(chunks) == len(stage.plan["tasks"])
    validate_process_v2_active8_plan(stage.plan, repo_root=ROOT)


def test_the_plan_addresses_every_declared_input(stage: _Stage) -> None:
    binding = stage.plan["binding"]
    assert binding["process_v2_identity_sha256"] == FROZEN_V2_PROCESS_IDENTITY
    assert binding["active8_policy_sha256"] == (
        build_process_v2_semantic_active8_admission_policy().policy_sha256
    )
    for field in (
        "cache_completion_sha256",
        "rebind_completion_sha256",
        "admitted_source_sha256",
        "code_revision_sha256",
        "capability_cell_registry_sha256",
        "cell_role_policy_sha256",
    ):
        assert len(binding[field]) == 64
    assert binding["model_runtime"] == model_runtime_descriptor(stage.runtime)


def test_the_plan_publishes_immutably_and_reopens_exactly(stage: _Stage) -> None:
    path = write_process_v2_active8_plan(
        stage.plan,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
    )
    assert path.name == PLAN_FILENAME
    assert load_process_v2_active8_plan(path, repo_root=ROOT) == stage.plan
    assert (
        write_process_v2_active8_plan(
            stage.plan,
            artifact_root=stage.artifact_root,
            repo_root=ROOT,
        )
        == path
    )


def test_the_plan_refuses_an_existing_different_payload(stage: _Stage) -> None:
    path = write_process_v2_active8_plan(
        stage.plan,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
    )
    path.write_bytes(b"{}\n")
    with pytest.raises(ProcessV2Active8PlanError, match="immutable.*publication"):
        write_process_v2_active8_plan(
            stage.plan,
            artifact_root=stage.artifact_root,
            repo_root=ROOT,
        )


def test_a_moved_implementation_revision_refuses_the_plan(
    stage: _Stage, tmp_path: Path
) -> None:
    """Validated against a tree where one bound module differs by one byte."""

    fake_root = tmp_path / "moved_revision"
    for relative in IMPLEMENTATION_FILES:
        target = fake_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    victim = fake_root / IMPLEMENTATION_FILES[0]
    victim.write_bytes(victim.read_bytes() + b"\n# moved\n")
    with pytest.raises(ProcessV2Active8PlanError, match="implementation has moved"):
        validate_process_v2_active8_plan(stage.plan, repo_root=fake_root)


def test_a_task_whose_chunk_binding_moved_cannot_read_its_chunk(stage: _Stage) -> None:
    """Caught twice: by the plan self-hash, and by the task identity beneath it."""

    plan = json.loads(json.dumps(stage.plan))
    plan["tasks"][0]["chunk_file_sha256"] = "0" * 64
    with pytest.raises(ProcessV2SchemaError, match="plan_sha256"):
        validate_process_v2_active8_plan(plan, repo_root=ROOT)
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha256(body)
    with pytest.raises(ProcessV2Active8PlanError, match="task identity disagrees"):
        validate_process_v2_active8_plan(plan, repo_root=ROOT)


def test_the_upstream_rejection_code_is_the_rebind_code(stage: _Stage) -> None:
    published = {
        (row["v1_task_identity_sha256"], row["entry_index"]): row["upstream_rejection_code"]
        for row in stage.rows()
        if row["rejection_category"] == UPSTREAM_REJECTED
    }
    assert published
    upstream: dict[tuple[str, int], str] = {}
    for task in stage.rebind_plan["tasks"]:
        output = stage.artifact_root / str(task["output_artifact_path"]).removeprefix(
            "/artifacts/"
        )
        manifest = json.loads((output / REBIND_MANIFEST_FILENAME).read_bytes())
        for rejection in manifest["rejected_traces"]:
            upstream[
                (str(task["v1_task_identity_sha256"]), int(rejection["entry_index"]))
            ] = str(rejection["exclusion_code"])
    assert published == {key: upstream[key] for key in published}
