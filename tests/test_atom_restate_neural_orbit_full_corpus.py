from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.active8_trace_inventory import (
    Active8SourceShard,
    ExactCandidateEvidence,
    build_active8_trace_inventory,
)
from compose_v4.data.packed_trace_store import write_packed_shard
from compose_v4.experiments.atom_restate_neural_orbit_audit import (
    AtomRestateNeuralOrbitAuditError,
    enumerate_current_broad_restate_candidates,
    file_sha256,
)
from compose_v4.experiments.atom_restate_neural_orbit_full_corpus import (
    ROW_EVIDENCE_NAME,
    SUMMARY_NAME,
    AtomRestateFullCorpusAuditError,
    FullCorpusAuditInputs,
    execute_bound_full_corpus_audit,
    load_full_corpus_driver_contract,
    map_resumable_full_corpus_shard,
    plan_resumable_full_corpus_audit,
    reduce_resumable_full_corpus_audit,
    resolve_bound_train_shards,
)
from compose_v4.experiments.factorized_successor_training import (
    rewrite_action_codec_sha256,
)
from compose_v4.rewrite.action_codec import encode_action
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomDelete
from compose_v4.rewrite.trace_shard import encode_state


ROOT = Path(__file__).resolve().parents[1]
DRIVER_CONTRACT = ROOT / "configs/editing_atom_restate_neural_orbit_full_corpus_v1.json"
ORBIT_CONTRACT = ROOT / "configs/editing_atom_restate_neural_orbit_audit_v1.json"


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _accept_exact_teacher(addressed, step_index: int) -> ExactCandidateEvidence:
    step = addressed.trace.steps[step_index]
    return ExactCandidateEvidence(
        supported=True,
        action_sha256=rewrite_action_codec_sha256(
            step.rule_name,
            step.action,
        ),
    )


def _packed_entry(
    *,
    wrong_stored_successor: bool,
    atom_restate_teacher: bool,
) -> dict[str, object]:
    if not atom_restate_teacher:
        source = pad_molecular_graph(smiles_to_molecular_graph("C"), 12)
        action = AtomDelete(v=0)
        successor = de_novo_rewrite_system().apply(source, "atom_delete", action)
        rule_name = "atom_delete"
    else:
        source = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 12)
        candidate = next(
            candidate
            for candidate in enumerate_current_broad_restate_candidates(source)
            if candidate.target_class[0] == ELEMENT_TO_IDX["N"]
        )
        action = candidate.action
        successor = de_novo_rewrite_system().apply(source, "atom_restate", action)
        rule_name = "atom_restate"
        if wrong_stored_successor:
            successor = next(
                de_novo_rewrite_system().apply(source, "atom_restate", other.action)
                for other in enumerate_current_broad_restate_candidates(source)
                if other.action != action and other.target_class[0] == ELEMENT_TO_IDX["N"]
            )
    return {
        "trace": {
            "trace_id": "restate-trace",
            "layer": "corruption",
            "partition": "train",
            "source_key": canonical_state_key(source),
            "target_key": canonical_state_key(successor),
            "path_length": 1,
            "steps": [
                {
                    "action": encode_action(
                        rule_name,
                        action,
                    )
                }
            ],
            "metadata": {"fixture": "atom-restate-full-corpus"},
        },
        "states": [encode_state(source), encode_state(successor)],
    }


def _fixture(
    tmp_path: Path,
    *,
    wrong_stored_successor: bool = False,
    atom_restate_teacher: bool = True,
) -> tuple[FullCorpusAuditInputs, dict[str, object]]:
    audit_root = tmp_path / "audit_layers"
    mmp_root = tmp_path / "mmp_layer"
    mmp_root.mkdir(parents=True)
    shard = audit_root / "train" / "fixture.jsonl.gz"
    write_packed_shard(
        shard,
        [
            _packed_entry(
                wrong_stored_successor=wrong_stored_successor,
                atom_restate_teacher=atom_restate_teacher,
            )
        ],
        provenance={"capability_hash": "orbit-full-corpus-fixture"},
        deterministic_gzip=True,
    )
    unified = {
        "artifact": "unified_packed_corpus_manifest",
        "roots": {
            "audit_layers": str(audit_root),
            "mmp_layer": str(mmp_root),
        },
        "layers": {
            "general_corruption": {
                "train": ["train/fixture.jsonl.gz"],
            }
        },
        "totals": {"shards": 1},
    }
    unified_path = tmp_path / "UNIFIED_PACKED_MANIFEST.json"
    unified_path.write_text(json.dumps(unified, sort_keys=True))
    inventory_dir = tmp_path / "active8"
    inventory = build_active8_trace_inventory(
        (
            Active8SourceShard(
                manifest_layer="general_corruption",
                envelope_layer="corruption",
                partition="train",
                relative_path="train/fixture.jsonl.gz",
                path=shard,
            ),
        ),
        exact_candidate_checker=_accept_exact_teacher,
        source_manifest_path=unified_path,
        source_manifest=unified,
        support_contract_sha256="d" * 64,
        output_dir=inventory_dir,
    )
    inventory_path = inventory_dir / "ACTIVE8_TRACE_INVENTORY.json"
    orbit_payload = json.loads(ORBIT_CONTRACT.read_text())
    driver_contract = {
        "status": "FROZEN_TRAIN_ONLY_FULL_CORPUS_DIAGNOSTIC_NO_TRAINING_AUTHORITY",
        "authorizes_audit_execution": True,
        "required_core_revision": "f921443193038be423383074a96f834de72f5123",
        "orbit_audit_contract": {
            "file_sha256": file_sha256(ORBIT_CONTRACT),
            "contract_sha256": orbit_payload["contract_sha256"],
        },
        "frozen_active8_parent": {
            "inventory_manifest_file_sha256": file_sha256(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "effective_source_corpus_cache_sha256": inventory["source_identity"][
                "effective_source_corpus_cache_sha256"
            ],
            "support_contract_sha256": "d" * 64,
            "unified_packed_manifest_sha256": file_sha256(unified_path),
        },
        "contract_sha256": "e" * 64,
    }
    inputs = FullCorpusAuditInputs(
        driver_contract_path=DRIVER_CONTRACT,
        orbit_contract_path=ORBIT_CONTRACT,
        active8_inventory_path=inventory_path,
        unified_manifest_path=unified_path,
        audit_root=audit_root,
        mmp_root=mmp_root,
        output_dir=tmp_path / "output",
        repository_root=ROOT,
        max_problem_address_examples=2,
    )
    return inputs, driver_contract


def _source_revision() -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "worktree_clean": True,
        "required_core_revision": "f921443193038be423383074a96f834de72f5123",
        "required_core_revision_is_ancestor": True,
    }


def test_full_corpus_driver_streams_exact_admitted_train_rows_atomically(
    tmp_path: Path,
) -> None:
    inputs, contract = _fixture(tmp_path)
    summary = execute_bound_full_corpus_audit(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
    )

    assert summary["authorizes_training"] is False
    assert summary["selects_support_policy"] is False
    assert summary["counts"]["shards"] == 1
    assert summary["counts"]["traces"] == 1
    assert summary["counts"]["atom_restate_teachers"] == 1
    assert summary["row_evidence"]["row_count"] == 1
    assert summary["row_evidence"]["file_sha256"] == file_sha256(
        inputs.output_dir / ROW_EVIDENCE_NAME
    )
    persisted = json.loads((inputs.output_dir / SUMMARY_NAME).read_text())
    unhashed = dict(persisted)
    expected = unhashed.pop("summary_sha256")
    assert _canonical_sha256(unhashed) == expected
    with gzip.open(inputs.output_dir / ROW_EVIDENCE_NAME, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 1
    assert rows[0]["address"]["partition"] == "train"
    assert (
        rows[0]["address"]["packed_shard_content_sha256"]
        == summary["inputs"]["train_shards"][0]["packed_shard_content_sha256"]
    )


def test_driver_rejects_inventory_or_source_byte_substitution(
    tmp_path: Path,
) -> None:
    inventory_inputs, inventory_contract = _fixture(tmp_path / "inventory")
    inventory_inputs.active8_inventory_path.write_text("{}")
    with pytest.raises(AtomRestateFullCorpusAuditError, match="inventory file differs"):
        resolve_bound_train_shards(
            inventory_inputs,
            driver_contract=inventory_contract,
        )

    source_inputs, source_contract = _fixture(tmp_path / "source")
    with source_inputs.audit_root.joinpath("train/fixture.jsonl.gz").open("ab") as handle:
        handle.write(b"substitution")
    with pytest.raises(
        AtomRestateFullCorpusAuditError,
        match="physical or semantic verification",
    ):
        execute_bound_full_corpus_audit(
            source_inputs,
            driver_contract=source_contract,
            source_revision=_source_revision(),
        )
    assert not source_inputs.output_dir.exists()


def test_failed_teacher_execution_publishes_no_partial_artifact(
    tmp_path: Path,
) -> None:
    inputs, contract = _fixture(tmp_path, wrong_stored_successor=True)
    with pytest.raises(
        AtomRestateNeuralOrbitAuditError,
        match="exact stored successor",
    ):
        execute_bound_full_corpus_audit(
            inputs,
            driver_contract=contract,
            source_revision=_source_revision(),
        )
    assert not inputs.output_dir.exists()
    assert not list(inputs.output_dir.parent.glob(f".{inputs.output_dir.name}.*.tmp"))


def test_production_driver_contract_preserves_validation_only_blocker() -> None:
    payload = load_full_corpus_driver_contract(DRIVER_CONTRACT)
    assert payload["status"] == "BLOCKED_MISSING_FROZEN_ACTIVE8_TRAIN_PARENT"
    assert payload["partition_role"] == "train_only"
    assert payload["authorizes_audit_execution"] is False
    assert payload["authorizes_training"] is False
    assert payload["selects_support_policy"] is False
    assert payload["allows_arbitrary_or_legacy_shards"] is False
    assert payload["required_core_revision"] == ("f921443193038be423383074a96f834de72f5123")
    assert payload["frozen_active8_parent"] is None
    assert payload["observed_validation_only_parent"]["partitions"] == ["validation"]
    assert payload["observed_validation_only_parent"]["train_shards"] == 0


def test_blocked_production_contract_cannot_plan_full_corpus(
    tmp_path: Path,
) -> None:
    inputs, _ = _fixture(tmp_path)
    payload = load_full_corpus_driver_contract(DRIVER_CONTRACT)
    with pytest.raises(
        AtomRestateFullCorpusAuditError,
        match="no frozen admitted Active8 train parent",
    ):
        plan_resumable_full_corpus_audit(
            inputs,
            driver_contract=payload,
            source_revision=_source_revision(),
        )


def test_resumable_map_reduce_reuses_verified_shard_receipts(
    tmp_path: Path,
) -> None:
    inputs, contract = _fixture(tmp_path)
    output_root = tmp_path / "resumable"
    plan = plan_resumable_full_corpus_audit(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
    )
    assert plan["expected_map_tasks"] == 1
    task_identity = plan["tasks"][0]["task_identity_sha256"]

    first_map = map_resumable_full_corpus_shard(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
        output_root=output_root,
        task_identity_sha256=task_identity,
    )
    second_map = map_resumable_full_corpus_shard(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
        output_root=output_root,
        task_identity_sha256=task_identity,
    )
    assert first_map["reused"] is False
    assert second_map["reused"] is True
    assert first_map["summary_sha256"] == second_map["summary_sha256"]

    first_final = reduce_resumable_full_corpus_audit(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
        output_root=output_root,
    )
    second_final = reduce_resumable_full_corpus_audit(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
        output_root=output_root,
    )
    assert first_final["reused"] is False
    assert second_final["reused"] is True
    assert first_final["counts"]["shards"] == 1
    assert first_final["counts"]["atom_restate_teachers"] == 1
    assert first_final["row_evidence"]["row_count"] == 1
    assert first_final["summary_sha256"] == second_final["summary_sha256"]


def test_resumable_map_refuses_corrupt_receipt_instead_of_recomputing(
    tmp_path: Path,
) -> None:
    inputs, contract = _fixture(tmp_path)
    output_root = tmp_path / "resumable"
    plan = plan_resumable_full_corpus_audit(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
    )
    task_identity = plan["tasks"][0]["task_identity_sha256"]
    map_resumable_full_corpus_shard(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
        output_root=output_root,
        task_identity_sha256=task_identity,
    )
    task_dir = output_root / "runs" / plan["run_identity_sha256"] / "map" / task_identity
    with (task_dir / ROW_EVIDENCE_NAME).open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(
        AtomRestateFullCorpusAuditError,
        match="row evidence disagrees",
    ):
        map_resumable_full_corpus_shard(
            inputs,
            driver_contract=contract,
            source_revision=_source_revision(),
            output_root=output_root,
            task_identity_sha256=task_identity,
        )


def test_resumable_map_persists_verified_empty_shard_receipt(
    tmp_path: Path,
) -> None:
    inputs, contract = _fixture(tmp_path, atom_restate_teacher=False)
    output_root = tmp_path / "resumable"
    plan = plan_resumable_full_corpus_audit(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
    )
    receipt = map_resumable_full_corpus_shard(
        inputs,
        driver_contract=contract,
        source_revision=_source_revision(),
        output_root=output_root,
        task_identity_sha256=plan["tasks"][0]["task_identity_sha256"],
    )
    assert receipt["counts"]["atom_restate_teachers"] == 0
    assert receipt["row_evidence"]["row_count"] == 0
    assert receipt["policy_aggregates"] == {}


def test_resumable_plan_rejects_local_to_remote_source_snapshot_drift(
    tmp_path: Path,
) -> None:
    inputs, contract = _fixture(tmp_path)
    source_revision = _source_revision()
    source_revision["implementation_sha256"] = "0" * 64
    with pytest.raises(
        AtomRestateFullCorpusAuditError,
        match="runtime implementation bytes differ",
    ):
        plan_resumable_full_corpus_audit(
            inputs,
            driver_contract=contract,
            source_revision=source_revision,
        )
