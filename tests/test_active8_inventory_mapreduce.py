from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.active8_inventory_mapreduce import (
    Active8MapReduceError,
    Active8MapReduceIncomplete,
    map_active8_source_decisions,
    plan_active8_mapreduce,
    reduce_active8_mapreduce,
    verified_completed_task_identities,
)
from compose_v4.data.active8_trace_inventory import (
    Active8SourceShard,
    ExactCandidateEvidence,
    accepted_trace_keys,
    load_active8_trace_admission,
)
from compose_v4.data.packed_trace_store import write_packed_shard
from compose_v4.experiments.factorized_successor_training import (
    rewrite_action_codec_sha256,
)
from compose_v4.rewrite.action_codec import encode_action
from compose_v4.rewrite.operators import AtomDelete
from compose_v4.rewrite.trace import RewriteStep
from compose_v4.rewrite.trace_shard import encode_state
from compose_v4.rewrite.tracelets import RingSystemDelete


def _ring_delete() -> RingSystemDelete:
    return RingSystemDelete(
        system_atoms=(0, 1),
        retained_system_atoms=(0,),
        bond_deletions=(),
        atom_deletions=(1,),
        atom_payloads=(),
        bond_reorders=(),
        source_aromatic_edges=(),
        aromatic_edges=(),
        topology_class="fixture",
    )


def _entry(trace_id: str, steps: tuple[RewriteStep, ...]) -> dict:
    state = pad_molecular_graph(smiles_to_molecular_graph("CC"), 8)
    return {
        "trace": {
            "trace_id": trace_id,
            "layer": "corruption",
            "partition": "train",
            "source_key": "CC",
            "target_key": "CC",
            "path_length": len(steps),
            "steps": [
                {"action": encode_action(step.rule_name, step.action)}
                for step in steps
            ],
            "metadata": {},
        },
        "states": [encode_state(state) for _ in range(len(steps) + 1)],
    }


def _fixture(tmp_path: Path, *, shards: int = 1):
    declared = []
    for index in range(shards):
        shard = tmp_path / "packed" / "train" / f"shard_{index:04d}.jsonl.gz"
        write_packed_shard(
            shard,
            [
                _entry(
                    f"accepted-{index}",
                    (RewriteStep("atom_delete", AtomDelete(1)),),
                ),
                _entry(
                    f"excluded-{index}",
                    (
                        RewriteStep("atom_delete", AtomDelete(1)),
                        RewriteStep("ring_system_delete", _ring_delete()),
                        RewriteStep("atom_delete", AtomDelete(0)),
                    ),
                ),
            ],
            provenance={"capability_hash": "active8-mapreduce-fixture"},
            deterministic_gzip=True,
        )
        declared.append(
            Active8SourceShard(
                manifest_layer="general_corruption",
                envelope_layer="corruption",
                partition="train",
                relative_path=f"train/{shard.name}",
                path=shard,
            )
        )
    source_manifest = {"artifact": "fixture-unified", "shards": shards}
    source_manifest_path = tmp_path / "UNIFIED_PACKED_MANIFEST.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, sort_keys=True)
    )
    plan = plan_active8_mapreduce(
        tuple(declared),
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        support_contract_sha256="a" * 64,
    )
    return plan


def _accept(addressed, step_index):
    step = addressed.trace.steps[step_index]
    return ExactCandidateEvidence(
        supported=True,
        action_sha256=rewrite_action_codec_sha256(
            step.rule_name,
            step.action,
        ),
    )


def test_map_is_resumable_and_reduce_publishes_compatible_inventory(
    tmp_path: Path,
) -> None:
    plan = _fixture(tmp_path)
    output = tmp_path / "output"
    first = map_active8_source_decisions(
        plan["tasks"][0],
        exact_candidate_checker=_accept,
        output_root=output,
    )
    second = map_active8_source_decisions(
        plan["tasks"][0],
        exact_candidate_checker=_accept,
        output_root=output,
    )
    assert first["reused"] is False
    assert second["reused"] is True
    assert first["decision_object_sha256"] == second["decision_object_sha256"]
    assert first["counts"]["accepted_traces"] == 1
    assert first["counts"]["excluded_traces"] == 1
    assert verified_completed_task_identities(
        plan,
        output_root=output,
    ) == {plan["tasks"][0]["task_identity_sha256"]}

    complete = reduce_active8_mapreduce(plan, output_root=output)
    assert complete["status"] == "COMPLETE"
    assert complete["training_authorized"] is False
    assert complete["counts"]["traces"] == 2
    assert complete["counts"]["accepted_progress_rows"] == 2
    manifest = output / complete["inventory_manifest"]
    assert len(accepted_trace_keys(manifest)) == 1
    admission = load_active8_trace_admission(
        manifest,
        expected_manifest_file_sha256=complete[
            "inventory_manifest_file_sha256"
        ],
        expected_inventory_sha256=complete["inventory_sha256"],
        expected_effective_source_corpus_cache_sha256=complete[
            "effective_source_corpus_cache_sha256"
        ],
    )
    assert admission.counts["accepted_traces"] == 1
    assert admission.counts["excluded_traces"] == 1

    resumed = reduce_active8_mapreduce(plan, output_root=output)
    assert resumed["reused"] is True
    assert resumed["inventory_sha256"] == complete["inventory_sha256"]


def test_reducer_refuses_even_one_missing_expected_source_decision(
    tmp_path: Path,
) -> None:
    plan = _fixture(tmp_path, shards=2)
    output = tmp_path / "output"
    map_active8_source_decisions(
        plan["tasks"][0],
        exact_candidate_checker=_accept,
        output_root=output,
    )
    with pytest.raises(Active8MapReduceIncomplete, match="1 expected"):
        reduce_active8_mapreduce(plan, output_root=output)
    assert not (
        output
        / "runs"
        / plan["run_identity_sha256"]
        / "COMPLETE.json"
    ).exists()


def test_immutable_decision_object_collision_is_rejected(
    tmp_path: Path,
) -> None:
    plan = _fixture(tmp_path)
    output = tmp_path / "output"
    receipt = map_active8_source_decisions(
        plan["tasks"][0],
        exact_candidate_checker=_accept,
        output_root=output,
    )
    decision = output / receipt["decision_object"]
    receipt_path = (
        output
        / "runs"
        / plan["run_identity_sha256"]
        / "receipts"
        / f"{plan['tasks'][0]['task_identity_sha256']}.json"
    )
    receipt_path.unlink()
    decision.write_bytes(b"collision")
    with pytest.raises(Active8MapReduceError, match="collision"):
        map_active8_source_decisions(
            plan["tasks"][0],
            exact_candidate_checker=_accept,
            output_root=output,
        )


def test_modal_surface_is_bounded_and_orders_map_before_reduce() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "modal_apps"
        / "build_active8_trace_inventory_app.py"
    ).read_text()
    assert "_MAX_MAP_CONTAINERS = 16" in source
    assert "max_containers=_MAX_MAP_CONTAINERS" in source
    assert "map_source.starmap" in source
    assert source.index("map_source.starmap") < source.index(
        "return reduce_inventory.remote"
    )
    assert "driver.spawn" in source
    assert '"training_launched": False' in source
