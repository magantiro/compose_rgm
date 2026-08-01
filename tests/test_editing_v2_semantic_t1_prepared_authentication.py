"""Adversarial authentication of semantic T1 prepared partitions."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
)
from compose_v4.experiments import editing_v2_semantic_t1_prepared_inputs as prepared
from compose_v4.experiments.editing_v2_semantic_t1_capacity_policy import (
    load_semantic_t1_capacity_policy,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    LoadedSemanticT1PreparedInputs,
    SemanticT1PreparedInputError,
    authenticate_semantic_t1_prepared_inputs,
    build_semantic_t1_prepared_inputs,
    compiled_successor_map_from_payload,
)
from compose_v4.experiments.editing_v2_semantic_t1_successor_cache import (
    SemanticT1SuccessorCache,
)
from compose_v4.experiments.factorized_successor_training import (
    CanonicalSuccessorAliasGroup,
    CompiledStateSuccessorMap,
    CompiledSuccessorMark,
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs/editing_v2_semantic_t1_capacity_policy_v1.json"


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 12)


def _source_revision() -> dict[str, object]:
    body: dict[str, object] = {"worktree_clean": True, "commit": "4" * 40}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "source_revision_sha256": hashlib.sha256(encoded).hexdigest()}


def _fixture():
    source = _state("C")
    teacher_target_sha256 = persistent_slot_state_sha256(_state("CC"))
    other_target_sha256 = persistent_slot_state_sha256(_state("N"))
    source_sha256 = persistent_slot_state_sha256(source)
    teacher_alias = TeacherSuccessorAlias(
        family_name="atom_insert",
        table_name="grow_connected",
        coordinate=(0, 0, 0),
    )
    other_alias = TeacherSuccessorAlias(
        family_name="atom_restate",
        table_name="restate",
        coordinate=(0, 1),
    )
    support = StateProductiveSupport("C", source_sha256, ())
    teacher_action_sha256 = "b" * 64
    teacher = TeacherSuccessorFiber(
        "C",
        "CC",
        teacher_target_sha256,
        (teacher_alias,),
        support,
    )
    record = SuccessorFiberCacheRecord(
        address=SuccessorFiberCacheAddress(
            packed_shard_content_sha256="a" * 64,
            packed_shard_name="train.jsonl.gz",
            entry_index=0,
            layer="observed_local_analogue",
            partition="train",
            trace_id="trace-0",
            trace_source_key="C",
            trace_target_key="CC",
            progress_index=0,
            path_length=1,
        ),
        state_support=support,
        teacher_fiber=teacher,
    )
    partition = CompiledStateSuccessorMap(
        state_support=support,
        successor_groups=(
            CanonicalSuccessorAliasGroup(
                target_key="CC",
                marks=(
                    CompiledSuccessorMark(
                        alias=teacher_alias,
                        successor_state_sha256=teacher_target_sha256,
                        action_sha256=teacher_action_sha256,
                    ),
                ),
            ),
            CanonicalSuccessorAliasGroup(
                target_key="N",
                marks=(
                    CompiledSuccessorMark(
                        alias=other_alias,
                        successor_state_sha256=other_target_sha256,
                        action_sha256="c" * 64,
                    ),
                ),
            ),
        ),
        virtual_marks=(),
    )
    panel_id = "d" * 64
    panel_entry = SimpleNamespace(
        panel_entry_sha256=panel_id,
        model_family="atom_insert",
        capability_cell_id="cell:atom_insert:connected",
        support_time_hex=(0.5).hex(),
        source_state_sha256=source_sha256,
        source_canonical_key="C",
        successor_canonical_key="CC",
        representative_target_state_sha256=teacher_target_sha256,
        representative_action_sha256=teacher_action_sha256,
        objective_coefficient=1,
        raw_mark_count=2,
        canonical_successor_count=2,
        production_successor_alias_multiplicity=1,
    )
    panel = SimpleNamespace(
        artifact_sha256="2" * 64,
        request=SimpleNamespace(support_time_hex=(0.5).hex()),
        entries=(panel_entry,),
    )
    cache = SemanticT1SuccessorCache(
        completion=MappingProxyType(
            {
                "completion_sha256": "3" * 64,
                "decision_source_inventory_sha256": "4" * 64,
                "initial_model_state_sha256": "5" * 64,
            }
        ),
        manifest=MappingProxyType(
            {
                "manifest_sha256": "6" * 64,
                "panel_binding": {"panel_artifact_sha256": panel.artifact_sha256},
                "semantic_model_process_contract": {"contract_sha256": "7" * 64},
                "source_revision": {"source_revision_sha256": "8" * 64},
            }
        ),
        records=(record,),
        records_by_address_sha256=MappingProxyType({}),
        records_by_panel_entry_sha256=MappingProxyType({panel_id: record}),
    )
    policy = load_semantic_t1_capacity_policy(POLICY_PATH)
    artifact = build_semantic_t1_prepared_inputs(
        panel=panel,
        cache=cache,
        source_states_by_panel_entry_sha256={panel_id: source},
        successor_partitions_by_panel_entry_sha256={panel_id: partition},
        capacity_policy=policy,
        capacity_policy_file_sha256=hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        source_revision=_source_revision(),
        repo_root=ROOT,
    )
    return artifact, source, panel, cache, policy, panel_id


def _reseal(artifact: dict[str, object]) -> None:
    entry = artifact["entries"][0]
    entry_body = dict(entry)
    entry_body.pop("entry_sha256")
    entry["entry_sha256"] = prepared._sha(entry_body)
    artifact["entry_inventory_sha256"] = prepared._sha(artifact["entries"])
    body = dict(artifact)
    body.pop("artifact_sha256")
    artifact["artifact_sha256"] = prepared._sha(body)


def _loaded(
    artifact: dict[str, object], source: object, panel_id: str
) -> LoadedSemanticT1PreparedInputs:
    entry = artifact["entries"][0]
    return LoadedSemanticT1PreparedInputs(
        artifact=MappingProxyType(artifact),
        states_by_panel_entry_sha256=MappingProxyType({panel_id: source}),
        partitions_by_panel_entry_sha256=MappingProxyType(
            {panel_id: compiled_successor_map_from_payload(entry["successor_partition"])}
        ),
    )


def test_authentication_accepts_exact_reopened_panel_and_cache() -> None:
    artifact, source, panel, cache, _policy, panel_id = _fixture()
    loaded = _loaded(artifact, source, panel_id)
    assert (
        authenticate_semantic_t1_prepared_inputs(
            loaded,
            panel=panel,
            cache=cache,
        )
        is loaded
    )


def test_authentication_rejects_rehashed_teacher_alias_forgery() -> None:
    artifact, source, panel, cache, policy, panel_id = _fixture()
    forged = copy.deepcopy(artifact)
    teacher_alias = forged["entries"][0]["successor_partition"]["successor_groups"][0]["marks"][0][
        "alias"
    ]
    teacher_alias["coordinate"] = [0, 0, 1]
    _reseal(forged)
    prepared.validate_semantic_t1_prepared_inputs(
        forged,
        expected_capacity_policy_sha256=policy["policy_sha256"],
        expected_panel_artifact_sha256=panel.artifact_sha256,
        expected_cache_completion_sha256=cache.completion["completion_sha256"],
        expected_initial_model_state_sha256=cache.completion["initial_model_state_sha256"],
        repo_root=ROOT,
    )
    with pytest.raises(SemanticT1PreparedInputError, match="teacher target, aliases"):
        authenticate_semantic_t1_prepared_inputs(
            _loaded(forged, source, panel_id),
            panel=panel,
            cache=cache,
        )


def test_authentication_rejects_rehashed_canonical_grouping_forgery() -> None:
    artifact, source, panel, cache, policy, panel_id = _fixture()
    forged = copy.deepcopy(artifact)
    groups = forged["entries"][0]["successor_partition"]["successor_groups"]
    groups[0]["marks"], groups[1]["marks"] = groups[1]["marks"], groups[0]["marks"]
    _reseal(forged)
    prepared.validate_semantic_t1_prepared_inputs(
        forged,
        expected_capacity_policy_sha256=policy["policy_sha256"],
        expected_panel_artifact_sha256=panel.artifact_sha256,
        expected_cache_completion_sha256=cache.completion["completion_sha256"],
        expected_initial_model_state_sha256=cache.completion["initial_model_state_sha256"],
        repo_root=ROOT,
    )
    with pytest.raises(SemanticT1PreparedInputError, match="teacher target, aliases"):
        authenticate_semantic_t1_prepared_inputs(
            _loaded(forged, source, panel_id),
            panel=panel,
            cache=cache,
        )


def test_builder_and_authentication_reject_mark_census_forgery() -> None:
    artifact, source, panel, cache, _policy, panel_id = _fixture()
    forged = copy.deepcopy(artifact)
    forged["entries"][0]["raw_mark_count"] = 3
    _reseal(forged)
    with pytest.raises(SemanticT1PreparedInputError, match="reopened panel entry"):
        authenticate_semantic_t1_prepared_inputs(
            _loaded(forged, source, panel_id),
            panel=panel,
            cache=cache,
        )

    panel.entries[0].raw_mark_count = 3
    policy = load_semantic_t1_capacity_policy(POLICY_PATH)
    with pytest.raises(SemanticT1PreparedInputError, match="full mark census"):
        build_semantic_t1_prepared_inputs(
            panel=panel,
            cache=cache,
            source_states_by_panel_entry_sha256={panel_id: source},
            successor_partitions_by_panel_entry_sha256={
                panel_id: _loaded(artifact, source, panel_id).partitions_by_panel_entry_sha256[
                    panel_id
                ]
            },
            capacity_policy=policy,
            capacity_policy_file_sha256=hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
            source_revision=_source_revision(),
            repo_root=ROOT,
        )
