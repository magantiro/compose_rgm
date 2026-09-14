from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_complete_program_decoder import (
    LEARNED,
    UNIFORM,
    DecoderConfig,
    build_candidate_case,
    build_candidate_lock,
    candidate_lock_from_cases,
    decode_source,
    seal_candidate_lock,
    source_manifest_from_panels,
    validate_candidate_lock,
    validate_fold_checkpoints,
)
from compose_v4.experiments.pmo_dependency_region_policy_comparison import (
    CHECKPOINT_SCHEMA as DEPENDENCY_CHECKPOINT_SCHEMA,
)
from compose_v4.experiments.pmo_dependency_region_policy_comparison import (
    HEAD_SUPPORT,
    POLICIES,
)
from compose_v4.experiments.pmo_legal_action_policy import (
    CHECKPOINT_SCHEMA as LEGAL_CHECKPOINT_SCHEMA,
)
from compose_v4.experiments.pmo_legal_action_policy import (
    RULES,
    enumerate_rule_successors,
)
from compose_v4.rewrite.trace_shard import encode_state
from modal_apps.pmo_complete_program_decoder_app import (
    _replace_json,
    _validate_existing_shard,
)
from tools.pmo_complete_program_decoder import sha256_file
from tools.pmo_complete_program_decoder_parallel import _read_progress, _volume_path

ROOT = Path(__file__).resolve().parents[1]


def _methane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[0] = 2
    hydrogens[0] = 4
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def _sequence_checkpoint(fold: int = 0) -> dict:
    rule_probability = 1 / len(RULES)
    return {
        "fold_index": fold,
        "policies": {
            POLICIES[0]: {
                "conditioned": False,
                "heads": {
                    "primitive_rule": {
                        "classes": list(HEAD_SUPPORT["primitive_rule"]),
                        "marginal_probabilities": [rule_probability] * len(RULES),
                    },
                    "program_control": {
                        "classes": list(HEAD_SUPPORT["program_control"]),
                        "marginal_probabilities": [0.5, 0.5],
                    },
                },
            }
        },
    }


def _runtime_checkpoints() -> tuple[dict, dict]:
    dependency = {
        "schema_version": DEPENDENCY_CHECKPOINT_SCHEMA,
        "new_oracle_calls": 0,
        "folds": [_sequence_checkpoint(fold) for fold in range(3)],
    }
    legal = {
        "schema_version": "pmo_legal_action_fold_checkpoints_v1",
        "new_oracle_calls": 0,
        "folds": [
            {
                "fold": fold,
                "checkpoint": {
                    "schema_version": LEGAL_CHECKPOINT_SCHEMA,
                    "coefficients": [0.0] * 581,
                    "feature_scale": [1.0] * 581,
                },
            }
            for fold in range(3)
        ],
    }
    return dependency, legal


def test_source_manifest_is_task_and_teacher_blind():
    panels = []
    for case in range(9):
        for duplicate in range(2):
            panels.append(
                {
                    "test_fold": case % 3,
                    "source_state": {"synthetic_source": case},
                    "task_family": f"forbidden_family_{case}",
                    "lineage_identity": f"forbidden_lineage_{case}_{duplicate}",
                    "teacher_endpoint_states": [{"teacher": True}],
                }
            )
    corpus = {
        "split": {"split_identity": "frozen-split"},
        "source_conditioned_panels": panels,
    }
    manifest = source_manifest_from_panels(corpus, split_identity="frozen-split")
    assert manifest["source_case_count"] == 9
    assert manifest["panel_count"] == 18
    serialized = repr(manifest)
    assert "forbidden_family" not in serialized
    assert "forbidden_lineage" not in serialized
    assert "teacher_endpoint_states" not in serialized
    assert all("teacher" not in repr(case) for case in manifest["source_cases"])


def test_uniform_and_learned_decoders_share_rule_and_stop_law_and_replay_exactly():
    source = _methane()
    sequence = _sequence_checkpoint()
    legal = {"schema_version": LEGAL_CHECKPOINT_SCHEMA}
    config = DecoderConfig(
        beam_widths=(1,),
        snapshot_depths=(1,),
        output_cutoffs=(1,),
        successors_per_rule=1,
        maximum_primitives=1,
        maximum_components=8,
        maximum_active_atoms=40,
    )

    def one_successor(graph, rule):
        if rule != "atom_insert":
            return ()
        return enumerate_rule_successors(graph, rule)[:1]

    outputs = [
        decode_source(
            source,
            sequence,
            legal,
            decoder=decoder,
            beam_width=1,
            config=config,
            successor_enumerator=one_successor,
            learned_scorer=lambda graph, candidate, checkpoint: 0.0,
        )
        for decoder in (UNIFORM, LEARNED)
    ]
    for output in outputs:
        candidates = output["snapshots"]["1"]["ranked_routes"]
        assert len(candidates) == 1
        assert candidates[0]["primitive_count"] == 1
        assert output["telemetry"]["exact_execution_precision"] == 1.0
    assert (
        outputs[0]["snapshots"]["1"]["ranked_routes"][0]["route_identity"]
        == outputs[1]["snapshots"]["1"]["ranked_routes"][0]["route_identity"]
    )
    assert (
        outputs[0]["snapshots"]["1"]["ranked_routes"][0]["score"]
        == outputs[1]["snapshots"]["1"]["ranked_routes"][0]["score"]
    )


def test_candidate_lock_mutation_fails_before_teacher_evaluation():
    payload = {
        "schema_version": "pmo_complete_program_decoder_candidate_lock_v1",
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
    }
    sealed = seal_candidate_lock(payload)
    assert validate_candidate_lock(sealed) == payload
    mutated = copy.deepcopy(sealed)
    mutated["payload"]["new_oracle_calls"] = 1
    with pytest.raises(ValueError, match="sealed or was modified"):
        validate_candidate_lock(mutated)


def test_checkpoint_validation_requires_balanced_marginal_and_matching_folds():
    dependency, legal = _runtime_checkpoints()
    validate_fold_checkpoints(dependency, legal)
    dependency["folds"][0]["policies"].pop(POLICIES[0])
    with pytest.raises(ValueError, match="balanced generic marginal"):
        validate_fold_checkpoints(dependency, legal)


def test_independent_case_generation_merges_to_the_serial_candidate_identity():
    dependency, legal = _runtime_checkpoints()
    state = encode_state(_methane())
    state_identity = identity(state)
    source = {
        "source_case_id": identity(
            {
                "schema_version": "pmo_decoder_source_case_v1",
                "fold": 0,
                "state_identity": state_identity,
            }
        ),
        "fold": 0,
        "source_state": state,
        "source_state_identity": state_identity,
    }
    manifest = {
        "schema_version": "pmo_complete_program_decoder_sources_v1",
        "split_identity": "frozen-split",
        "source_cases": [source],
        "panel_count": 2,
        "source_case_count": 1,
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
    }
    config = DecoderConfig(
        beam_widths=(1,),
        snapshot_depths=(1,),
        output_cutoffs=(1,),
        successors_per_rule=1,
        maximum_primitives=1,
        maximum_components=8,
        maximum_active_atoms=40,
    )
    progress = []
    case = build_candidate_case(
        source,
        dependency,
        legal,
        config,
        progress_callback=progress.append,
    )
    assert len(progress) == 2
    assert candidate_lock_from_cases(manifest, [case], config) == build_candidate_lock(
        manifest, dependency, legal, config
    )
    with pytest.raises(ValueError, match="census incomplete"):
        candidate_lock_from_cases(manifest, [], config)


def test_completed_source_shard_requires_matching_durable_receipt(tmp_path):
    task = {
        "run_id": "sealed-task",
        "source_case_index": 3,
        "source_case_id": "case-three",
    }
    payload = {"task_run_id": task["run_id"], "source_case": {"result": "fixed"}}
    shard_path = tmp_path / "case_shard.json"
    shard = {"payload": payload, "payload_sha256": identity(payload)}
    _replace_json(shard_path, shard)
    assert _validate_existing_shard(tmp_path, task) is None

    receipt_payload = {
        "task_run_id": task["run_id"],
        "source_case_index": task["source_case_index"],
        "source_case_id": task["source_case_id"],
        "case_shard_sha256": sha256_file(shard_path),
        "case_shard_payload_sha256": shard["payload_sha256"],
        "new_oracle_calls": 0,
    }
    _replace_json(
        tmp_path / "generation_receipt.json",
        {"payload": receipt_payload, "payload_sha256": identity(receipt_payload)},
    )
    assert _validate_existing_shard(tmp_path, task) == payload

    corrupt = {**receipt_payload, "case_shard_sha256": "0" * 64}
    _replace_json(
        tmp_path / "generation_receipt.json",
        {"payload": corrupt, "payload_sha256": identity(corrupt)},
    )
    with pytest.raises(ValueError, match="receipt is invalid"):
        _validate_existing_shard(tmp_path, task)


def test_missing_remote_progress_is_an_unstarted_case_not_a_status_failure():
    class MissingVolume:
        def read_file(self, path):
            raise FileNotFoundError(path)

    assert (
        _read_progress(MissingVolume(), {"output": "/artifacts/missing/case"}) is None
    )


def test_modal_volume_paths_are_relative_to_the_artifact_mount():
    assert (
        _volume_path("/artifacts/pmo_complete_program_decoder/attempt_1/result.json")
        == "/pmo_complete_program_decoder/attempt_1/result.json"
    )


def test_contract_is_sealed_and_pins_authoritative_sharded_execution():
    envelope = json.loads(
        (ROOT / "configs/pmo_complete_program_decoder_v1.json").read_text()
    )
    payload = envelope["payload"]
    assert envelope["contract_sha256"] == identity(payload)
    assert payload["authoritative_execution_enabled"] is True
    assert payload["inputs"]["legal_action_authoritative"]["path"].endswith(
        "attempt_2/runtime_fold_checkpoints.json.gz"
    )
    assert payload["execution"]["mode"] == "durable_source_case_shards"
    assert payload["execution"]["source_case_shards"] == 9
    assert payload["execution"]["maximum_concurrent_single_cpu_workers"] == 9
    assert payload["execution"]["automatic_retries"] == 0
    assert (
        payload["inputs"]["legal_action_preview_fixture_only"]["authoritative"] is False
    )
