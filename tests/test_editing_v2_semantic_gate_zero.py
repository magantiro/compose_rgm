"""Focused semantic Gate 0 structural-evidence tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_active8_admission import (
    build_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    SemanticStructuralCapabilityAssignment,
    load_semantic_capability_cell_registry,
)
from compose_v4.experiments import editing_v2_semantic_gate_zero as gate_zero

ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return payload + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _runtime_identity(contract) -> dict[str, object]:
    decision_runtime = json.loads(
        (ROOT / contract.payload["parents"]["decision_runtime"]["path"]).read_text()
    )
    semantic = json.loads(
        (ROOT / contract.payload["parents"]["semantic_model_process"]["path"]).read_text()
    )
    required = contract.payload["required_architecture"]
    body = {
        "schema": "compose.data.semantic_active8_exact_model_runtime",
        "schema_version": 2,
        "runtime_contract_sha256": decision_runtime["runtime_contract_sha256"],
        "semantic_model_process_contract_sha256": semantic["contract_sha256"],
        "semantic_model_identity": semantic["model_identity"],
        "process_identity_sha256": semantic["process_identity_sha256"],
        "architecture": {
            "max_atoms": required["max_atoms"],
            "hidden_dim": required["hidden_dim"],
            "message_passing_steps": required["message_passing_steps"],
            "mark_dim": required["mark_dim"],
            "dtype": required["dtype"],
            "parameter_dtypes": [required["dtype"]],
            "atom_vocabulary_class_count": required["atom_vocabulary_class_count"],
            "catalog_fingerprint": required["catalog_fingerprint"],
            "operator_capability_fingerprint": semantic["model_identity"][
                "operator_capability_fingerprint"
            ],
        },
        "initialization_seed": required["initialization_seed"],
        "initial_model_state_sha256": "a" * 64,
        "software": decision_runtime["software"],
        "producer_source_revision_sha256": "b" * 64,
        "execution_source_revision_sha256": "c" * 64,
    }
    return {**body, "identity_sha256": _sha(body)}


@dataclass(frozen=True)
class _FakeTransition:
    family: str
    step_index: int = 0

    @property
    def source_progress_address(self):
        return SimpleNamespace(terminal=False)

    @property
    def action_decision(self):
        return SimpleNamespace(classification=SimpleNamespace(model_family=self.family))


def _assignment(
    family: str,
    entry_index: int,
    *,
    partition_role: str = "train",
):
    registry = load_semantic_capability_cell_registry()
    context = registry.contexts_by_family[family][0]
    policy = build_semantic_active8_admission_policy()
    digest = hashlib.sha256(f"{family}-{entry_index}".encode()).hexdigest()
    provisional = SemanticStructuralCapabilityAssignment(
        decision_source_inventory_sha256="c" * 64,
        decision_sha256=hashlib.sha256(f"decision-{family}".encode()).hexdigest(),
        trace_address_sha256=hashlib.sha256(f"address-{family}".encode()).hexdigest(),
        source_progress_address_sha256=hashlib.sha256(
            f"source-progress-{family}".encode()
        ).hexdigest(),
        successor_progress_address_sha256=hashlib.sha256(
            f"successor-progress-{family}".encode()
        ).hexdigest(),
        trace_id=f"trace-{family}",
        progress_index=0,
        packed_shard_content_sha256=hashlib.sha256(f"shard-{family}".encode()).hexdigest(),
        packed_shard_name="semantic.jsonl.gz",
        packed_entry_index=entry_index,
        action_sha256=hashlib.sha256(f"action-{family}".encode()).hexdigest(),
        source_state_sha256=digest,
        target_state_sha256=hashlib.sha256(f"target-{family}".encode()).hexdigest(),
        source_canonical_key=f"source-{family}",
        successor_canonical_key=f"successor-{family}",
        model_family=family,
        family_context=context,
        capability_cell_id=f"{registry.namespace}:{family}:{context}",
        data_lane="reversible_synthetic_walk",
        partition_role=partition_role,
        raw_mark_count=20,
        raw_mark_count_stratum="marks_017_064",
        canonical_successor_count=10,
        canonical_successor_count_stratum="successors_005_016",
        successor_alias_multiplicity=2,
        successor_alias_multiplicity_stratum="aliases_002_004",
        matching_mark_count=1,
        atom_element_transition=(),
        minimum_edited_cycle_length=None,
        source_edge_aromatic=None,
        successor_edge_aromatic=None,
        registry_sha256=registry.registry_sha256,
        process_identity_sha256=registry.process_identity_sha256,
        corpus_contract_file_sha256=registry.corpus_contract_file_sha256,
        classifier_implementation_sha256=registry.classifier_implementation_sha256,
        active8_policy_sha256=policy.policy_sha256,
        assignment_sha256="0" * 64,
    )
    payload = provisional.as_payload()
    return replace(
        provisional,
        assignment_sha256=_sha(
            {key: value for key, value in payload.items() if key != "assignment_sha256"}
        ),
    )


@dataclass
class _FakeIndex:
    assignments: dict[str, SemanticStructuralCapabilityAssignment]
    decision_plan_file_sha256: str
    decision_plan_sha256: str
    decision_run_identity_sha256: str
    model_runtime_identity_sha256: str
    process_identity_sha256: str
    policy_sha256: str
    inventory_sha256: str = "c" * 64

    @property
    def count_map(self):
        count = len(self.assignments)
        return {
            "accepted_traces": count,
            "excluded_traces": 0,
            "actions": count,
            "progress_rows": count * 2,
        }

    @property
    def action_family_histogram(self):
        return tuple((family, 1) for family in self.assignments)

    @property
    def active8_exclusion_reason_histogram(self):
        return ()

    def iter_resolved_traces(self):
        for family, assignment in self.assignments.items():
            address = SimpleNamespace(
                packed_shard_content_sha256=assignment.packed_shard_content_sha256,
                packed_shard_name=assignment.packed_shard_name,
                entry_index=assignment.packed_entry_index,
                trace_id=assignment.trace_id,
                layer=assignment.data_lane,
                partition=assignment.partition_role,
                source_key=assignment.source_canonical_key,
                target_key=assignment.successor_canonical_key,
                path_length=1,
            )
            resolved = SimpleNamespace(
                addressed_trace=SimpleNamespace(address=address),
                decision_sha256=assignment.decision_sha256,
                action_decisions=(
                    SimpleNamespace(classification=SimpleNamespace(model_family=family)),
                ),
                progress_addresses=(
                    SimpleNamespace(terminal=False),
                    SimpleNamespace(terminal=True),
                ),
            )
            yield True, resolved

    def accepted_transitions_for(self, resolved):
        family = resolved.action_decisions[0].classification.model_family
        yield _FakeTransition(family)

    def identity_payload(self):
        return {
            "schema": "fixture.decision_source",
            "training_authorized": False,
            "gate_zero_authorized": False,
            "t1_authorized": False,
            "bounded_p50_authorized": False,
            "long_training_authorized": False,
            "checkpoint_selection_authorized": False,
            "final_test_selection_authorized": False,
            "inventory_sha256": self.inventory_sha256,
            "model_runtime_identity_sha256": self.model_runtime_identity_sha256,
            "process_identity_sha256": self.process_identity_sha256,
            "policy_sha256": self.policy_sha256,
            "counts": self.count_map,
            "action_family_histogram": dict(self.action_family_histogram),
            "active8_exclusion_reason_histogram": {},
        }


def _fixture_index_and_plan(
    tmp_path: Path,
    *,
    families=ACTIVE8_FAMILIES,
    partition_by_family: dict[str, str] | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    contract = gate_zero.load_semantic_gate_zero_structural_contract(repo_root=ROOT)
    runtime = _runtime_identity(contract)
    plan = {
        "model_runtime_identity": runtime,
        "plan_sha256": "d" * 64,
        "run_identity_sha256": "e" * 64,
    }
    raw = _canonical(plan, newline=True)
    path = tmp_path / "PLAN.json"
    path.write_bytes(raw)
    policy = build_semantic_active8_admission_policy()
    selected_partitions = partition_by_family or {}
    assignments = {
        family: _assignment(
            family,
            index,
            partition_role=selected_partitions.get(family, "train"),
        )
        for index, family in enumerate(families)
    }
    index = _FakeIndex(
        assignments=assignments,
        decision_plan_file_sha256=hashlib.sha256(raw).hexdigest(),
        decision_plan_sha256=plan["plan_sha256"],
        decision_run_identity_sha256=plan["run_identity_sha256"],
        model_runtime_identity_sha256=runtime["identity_sha256"],
        process_identity_sha256=policy.process_identity_sha256,
        policy_sha256=policy.policy_sha256,
    )
    return contract, index, path


def _patch_structural_projection(monkeypatch, index: _FakeIndex) -> None:
    def classify(observed_index, transition, *, registry):
        assert observed_index is index
        assignment = index.assignments[transition.family]
        assert assignment.registry_sha256 == registry.registry_sha256
        return assignment

    monkeypatch.setattr(gate_zero, "classify_verified_structural_transition", classify)


def test_contract_binds_exact_parents_and_grants_no_authority() -> None:
    contract = gate_zero.load_semantic_gate_zero_structural_contract(repo_root=ROOT)
    assert len(contract.sha256) == 64
    assert all(contract.payload[field] is False for field in gate_zero._AUTHORITY)
    checks = contract.payload["structural_checks"]
    assert checks["legacy_action_v2_evidence"] == "forbidden"
    assert checks["require_exactly_one_matching_mark"] is True
    assert checks["require_one_assignment_per_accepted_action"] is True
    assert checks["require_zero_terminal_assignments"] is True


def test_structural_pass_counts_verified_assignments_without_authorizing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract, index, plan_path = _fixture_index_and_plan(tmp_path)
    _patch_structural_projection(monkeypatch, index)
    evidence = gate_zero.build_semantic_gate_zero_structural_evidence(
        index,  # type: ignore[arg-type]
        decision_plan_path=plan_path,
        contract=contract,
        repo_root=ROOT,
    )
    assert evidence["structural_result"] == "PASS"
    assert evidence["counts"]["accepted_actions"] == 8
    assert evidence["counts"]["structural_assignments"] == 8
    assert evidence["counts"]["terminal_assignments"] == 0
    assert evidence["counts"]["accepted_progress_rows"] == 16
    assert evidence["checks"]["one_structural_assignment_per_decision_eligible_action"]
    assert evidence["checks"]["every_decision_eligible_teacher_matches_exactly_one_action_v4_mark"]
    assert evidence["decision_eligible_teacher_counts_by_family"] == {
        family: 1 for family in ACTIVE8_FAMILIES
    }
    assert all(evidence[field] is False for field in gate_zero._AUTHORITY)
    decision = gate_zero.structural_decision_from_evidence(
        evidence,
        index=index,  # type: ignore[arg-type]
        contract=contract,
        decision_plan_path=plan_path,
        repo_root=ROOT,
    )
    assert decision["structural_result"] == "PASS"
    assert decision["next_authorized_stage"] is None
    assert all(decision[field] is False for field in gate_zero._AUTHORITY)


def test_missing_family_and_rehashed_pass_claim_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract, index, plan_path = _fixture_index_and_plan(
        tmp_path,
        families=ACTIVE8_FAMILIES[:-1],
    )
    _patch_structural_projection(monkeypatch, index)
    evidence = gate_zero.build_semantic_gate_zero_structural_evidence(
        index,  # type: ignore[arg-type]
        decision_plan_path=plan_path,
        contract=contract,
        repo_root=ROOT,
    )
    assert evidence["structural_result"] == "FAIL"
    decision = gate_zero.structural_decision_from_evidence(
        evidence,
        index=index,  # type: ignore[arg-type]
        contract=contract,
        decision_plan_path=plan_path,
        repo_root=ROOT,
    )
    assert decision["required_next_action"] == (
        "repair_missing_structural_coverage_and_rerun_gate_zero"
    )

    forged = json.loads(json.dumps(evidence))
    forged["structural_result"] = "PASS"
    forged["checks"]["all_active8_families_have_decision_eligible_teachers"] = True
    forged["evidence_sha256"] = _sha(
        {key: value for key, value in forged.items() if key != "evidence_sha256"}
    )
    with pytest.raises(
        gate_zero.SemanticGateZeroStructuralError,
        match="recomputed exact source snapshot",
    ):
        gate_zero.structural_decision_from_evidence(
            forged,
            index=index,  # type: ignore[arg-type]
            contract=contract,
            decision_plan_path=plan_path,
            repo_root=ROOT,
        )


def test_final_test_teacher_cannot_satisfy_train_family_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    held_out_family = ACTIVE8_FAMILIES[-1]
    contract, index, plan_path = _fixture_index_and_plan(
        tmp_path,
        partition_by_family={held_out_family: "final_test"},
    )
    _patch_structural_projection(monkeypatch, index)
    evidence = gate_zero.build_semantic_gate_zero_structural_evidence(
        index,  # type: ignore[arg-type]
        decision_plan_path=plan_path,
        contract=contract,
        repo_root=ROOT,
    )
    assert evidence["structural_result"] == "FAIL"
    assert evidence["decision_eligible_teacher_counts_by_family"][held_out_family] == 0
    assert "partition_counts" not in evidence["capability_cell_counts"][0]
    assert evidence["checks"]["all_active8_families_have_decision_eligible_teachers"] is False


def test_classification_failure_is_published_as_bounded_negative_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract, index, plan_path = _fixture_index_and_plan(tmp_path)

    def classify(observed_index, transition, *, registry):
        assert observed_index is index
        if transition.family == ACTIVE8_FAMILIES[0]:
            raise gate_zero.SemanticCapabilityCellError("unsupported fixture teacher")
        return index.assignments[transition.family]

    monkeypatch.setattr(gate_zero, "classify_verified_structural_transition", classify)
    evidence = gate_zero.build_semantic_gate_zero_structural_evidence(
        index,  # type: ignore[arg-type]
        decision_plan_path=plan_path,
        contract=contract,
        repo_root=ROOT,
    )
    assert evidence["structural_result"] == "FAIL"
    assert evidence["classification_failure_count"] == 1
    assert len(evidence["classification_failure_receipts"]) == 1
    assert evidence["classification_failure_receipts"][0]["reason"] == (
        "unsupported fixture teacher"
    )


def test_tampered_runtime_and_counts_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract, index, plan_path = _fixture_index_and_plan(tmp_path)
    _patch_structural_projection(monkeypatch, index)
    plan = json.loads(plan_path.read_text())
    plan["model_runtime_identity"]["architecture"]["hidden_dim"] = 999
    runtime_body = {
        key: value
        for key, value in plan["model_runtime_identity"].items()
        if key != "identity_sha256"
    }
    plan["model_runtime_identity"]["identity_sha256"] = _sha(runtime_body)
    raw = _canonical(plan, newline=True)
    plan_path.write_bytes(raw)
    index.decision_plan_file_sha256 = hashlib.sha256(raw).hexdigest()
    index.model_runtime_identity_sha256 = plan["model_runtime_identity"]["identity_sha256"]
    with pytest.raises(
        gate_zero.SemanticGateZeroStructuralError,
        match="architecture/process",
    ):
        gate_zero.build_semantic_gate_zero_structural_evidence(
            index,  # type: ignore[arg-type]
            decision_plan_path=plan_path,
            contract=contract,
            repo_root=ROOT,
        )

    contract, index, plan_path = _fixture_index_and_plan(tmp_path / "fresh")
    _patch_structural_projection(monkeypatch, index)
    evidence = gate_zero.build_semantic_gate_zero_structural_evidence(
        index,  # type: ignore[arg-type]
        decision_plan_path=plan_path,
        contract=contract,
        repo_root=ROOT,
    )
    evidence["counts"]["accepted_actions"] = 999
    evidence["structural_result"] = "PASS"
    evidence["evidence_sha256"] = _sha(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    with pytest.raises(
        gate_zero.SemanticGateZeroStructuralError,
        match="recomputed exact source snapshot",
    ):
        gate_zero.structural_decision_from_evidence(
            evidence,
            index=index,  # type: ignore[arg-type]
            contract=contract,
            decision_plan_path=plan_path,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "accepted_trace_stream_sha256",
        "structural_assignment_inventory_sha256",
        "active8_policy",
        "capability_registry",
        "model_runtime_identity",
    ),
)
def test_rehashed_embedded_identity_or_stream_forgery_fails_exact_rebuild(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    contract, index, plan_path = _fixture_index_and_plan(tmp_path)
    _patch_structural_projection(monkeypatch, index)
    evidence = gate_zero.build_semantic_gate_zero_structural_evidence(
        index,  # type: ignore[arg-type]
        decision_plan_path=plan_path,
        contract=contract,
        repo_root=ROOT,
    )
    forged = json.loads(json.dumps(evidence))
    if mutation in {
        "accepted_trace_stream_sha256",
        "structural_assignment_inventory_sha256",
    }:
        forged[mutation] = "f" * 64
    elif mutation == "active8_policy":
        forged[mutation]["status"] = "FORGED"
    elif mutation == "capability_registry":
        forged[mutation]["registry_sha256"] = "f" * 64
    else:
        forged[mutation]["initial_model_state_sha256"] = "f" * 64
    forged["evidence_sha256"] = _sha(
        {key: value for key, value in forged.items() if key != "evidence_sha256"}
    )
    with pytest.raises(
        gate_zero.SemanticGateZeroStructuralError,
        match="recomputed exact source snapshot",
    ):
        gate_zero.structural_decision_from_evidence(
            forged,
            index=index,  # type: ignore[arg-type]
            contract=contract,
            decision_plan_path=plan_path,
            repo_root=ROOT,
        )


def test_absent_contract_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(
        gate_zero.SemanticGateZeroStructuralError,
        match="absent or invalid",
    ):
        gate_zero.load_semantic_gate_zero_structural_contract(
            tmp_path / "missing.json",
            repo_root=ROOT,
        )


def test_substituted_self_hashed_contract_id_is_rejected(tmp_path: Path) -> None:
    payload = json.loads((ROOT / gate_zero.CONTRACT_RELATIVE_PATH).read_text())
    payload["contract_id"] = "substituted_contract"
    payload["contract_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "contract_sha256"}
    )
    path = tmp_path / "contract.json"
    path.write_bytes(_canonical(payload, newline=True))
    with pytest.raises(
        gate_zero.SemanticGateZeroStructuralError,
        match="identity, self-hash, or authority",
    ):
        gate_zero.load_semantic_gate_zero_structural_contract(
            path,
            repo_root=ROOT,
        )


def test_run_publishes_completion_last_without_downstream_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract, index, plan_path = _fixture_index_and_plan(tmp_path / "fixture")
    _patch_structural_projection(monkeypatch, index)
    monkeypatch.setattr(
        gate_zero,
        "resolve_editing_v2_semantic_active8_decision_source",
        lambda **_kwargs: index,
    )
    artifact_root = tmp_path / "artifacts"
    output = artifact_root / "gate-zero"
    result = gate_zero.run_semantic_gate_zero_structural_evidence(
        migration_completion_path=tmp_path / "migration.json",
        chunk_cache_plan_path=tmp_path / "cache-plan.json",
        chunk_cache_global_completion_path=tmp_path / "cache-complete.json",
        decision_plan_path=plan_path,
        decision_completion_path=tmp_path / "decision-complete.json",
        artifact_root=artifact_root,
        repo_root=ROOT,
        output_directory=output,
        contract_path=contract.source,
    )
    completion = json.loads((output / gate_zero.COMPLETION_FILENAME).read_text())
    completion_body = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    assert completion["completion_sha256"] == _sha(completion_body)
    assert completion["structural_result"] == "PASS"
    assert all(completion[field] is False for field in gate_zero._AUTHORITY)
    assert result["completion"] == completion
    assert (output / gate_zero.EVIDENCE_FILENAME).is_file()
    assert (output / gate_zero.DECISION_FILENAME).is_file()
    loaded = gate_zero.load_semantic_gate_zero_structural_artifacts(
        output_directory=output,
        index=index,  # type: ignore[arg-type]
        decision_plan_path=plan_path,
        contract=contract,
        repo_root=ROOT,
    )
    assert loaded == result
