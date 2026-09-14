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

ROOT = Path(__file__).resolve().parents[1]
from compose_v4.experiments.pmo_legal_action_policy import (
    RULES,
    enumerate_rule_successors,
)


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
                    "coefficients": [0.0],
                    "feature_scale": [1.0],
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


def test_contract_is_sealed_and_authoritative_execution_is_disabled():
    envelope = json.loads(
        (ROOT / "configs/pmo_complete_program_decoder_v1.json").read_text()
    )
    payload = envelope["payload"]
    assert envelope["contract_sha256"] == identity(payload)
    assert payload["authoritative_execution_enabled"] is False
    assert payload["inputs"]["legal_action_authoritative"] is None
    assert (
        payload["inputs"]["legal_action_preview_fixture_only"]["authoritative"] is False
    )
