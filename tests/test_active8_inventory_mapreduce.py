from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import compose_v4.data.active8_inventory_mapreduce as mapreduce
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
    implementation_identity as active8_implementation_identity,
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
            "steps": [{"action": encode_action(step.rule_name, step.action)} for step in steps],
            "metadata": {},
        },
        "states": [encode_state(state) for _ in range(len(steps) + 1)],
    }


def _fixture(
    tmp_path: Path,
    *,
    shards: int = 1,
    entries_per_shard: int = 2,
    target_entries_per_range: int = 500,
    worker_resources: dict[str, object] | None = None,
    source_revision: dict[str, object] | None = None,
):
    declared = []
    for index in range(shards):
        shard = tmp_path / "packed" / "train" / f"shard_{index:04d}.jsonl.gz"
        entries = []
        for entry_index in range(entries_per_shard):
            if entry_index % 2 == 0:
                steps = (RewriteStep("atom_delete", AtomDelete(1)),)
                verdict = "accepted"
            else:
                steps = (
                    RewriteStep("atom_delete", AtomDelete(1)),
                    RewriteStep("ring_system_delete", _ring_delete()),
                    RewriteStep("atom_delete", AtomDelete(0)),
                )
                verdict = "excluded"
            entries.append(
                _entry(
                    f"{verdict}-{index}-{entry_index}",
                    steps,
                )
            )
        write_packed_shard(
            shard,
            entries,
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
    source_manifest_path.write_text(json.dumps(source_manifest, sort_keys=True))
    plan = plan_active8_mapreduce(
        tuple(declared),
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        support_contract_sha256="a" * 64,
        source_revision=source_revision or _source_revision(),
        target_entries_per_range=target_entries_per_range,
        worker_resources=worker_resources,
    )
    return plan


def _source_revision() -> dict[str, object]:
    active8_identity = active8_implementation_identity()
    mapreduce_identity = mapreduce._implementation_identity()
    return mapreduce._source_revision_binding(
        commit="1" * 40,
        tree="2" * 40,
        worktree_clean=True,
        active8_identity=active8_identity,
        mapreduce_identity=mapreduce_identity,
    )


def _accept(addressed, step_index):
    step = addressed.trace.steps[step_index]
    return ExactCandidateEvidence(
        supported=True,
        action_sha256=rewrite_action_codec_sha256(
            step.rule_name,
            step.action,
        ),
    )


def _map_all(plan: dict[str, object], output: Path) -> None:
    for task in plan["tasks"]:
        map_active8_source_decisions(
            task,
            exact_candidate_checker=_accept,
            output_root=output,
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
        expected_manifest_file_sha256=complete["inventory_manifest_file_sha256"],
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


def test_publication_does_not_require_hard_links(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _fixture(tmp_path)
    output = tmp_path / "modal-compatible-output"

    def reject_hard_link(*args, **kwargs):
        raise PermissionError("Modal Volumes do not support hard links")

    monkeypatch.setattr("os.link", reject_hard_link)
    _map_all(plan, output)
    complete = reduce_active8_mapreduce(plan, output_root=output)

    assert complete["status"] == "COMPLETE"
    assert complete["counts"]["traces"] == 2


def test_plan_partitions_one_source_into_contiguous_immutable_ranges(
    tmp_path: Path,
) -> None:
    plan = _fixture(
        tmp_path,
        entries_per_shard=7,
        target_entries_per_range=3,
    )
    assert plan["expected_source_decisions"] == 1
    assert plan["expected_map_tasks"] == 3
    assert [(task["entry_start"], task["entry_stop"]) for task in plan["tasks"]] == [
        (0, 3),
        (3, 6),
        (6, 7),
    ]
    assert [task["range_index"] for task in plan["tasks"]] == [0, 1, 2]
    assert {task["range_count"] for task in plan["tasks"]} == {3}
    assert len({task["task_identity_sha256"] for task in plan["tasks"]}) == 3
    assert plan["range_policy"] == {
        "algorithm": "fixed_target_entries_contiguous_v1",
        "target_entries_per_range": 3,
    }
    assert plan["worker_resources"] == {
        "cpu": 2.0,
        "memory_mb": 8192,
        "max_containers": 64,
        "omp_num_threads": 1,
    }

    output = tmp_path / "ranged-output"
    _map_all(plan, output)
    complete = reduce_active8_mapreduce(plan, output_root=output)
    assert complete["expected_source_decisions"] == 1
    assert complete["expected_map_tasks"] == 3
    assert complete["counts"]["traces"] == 7
    manifest = json.loads((output / complete["inventory_manifest"]).read_text())
    assert len(manifest["shards"]) == 1
    assert manifest["shards"][0]["map_range_count"] == 3
    assert manifest["shards"][0]["counts"]["accepted_traces"] == 4
    assert manifest["shards"][0]["counts"]["excluded_traces"] == 3
    receipt_path = (
        output
        / "runs"
        / plan["run_identity_sha256"]
        / "receipts"
        / f"{plan['tasks'][0]['task_identity_sha256']}.json"
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["range_policy"] == plan["range_policy"]
    assert receipt["worker_resources"] == plan["worker_resources"]
    assert receipt["source_revision"] == plan["source_revision"]
    assert receipt["source_revision_sha256"] == plan["source_revision_sha256"]


def test_range_policy_and_worker_resources_change_run_and_task_identity(
    tmp_path: Path,
) -> None:
    baseline = _fixture(
        tmp_path / "baseline",
        entries_per_shard=7,
        target_entries_per_range=3,
    )
    capped = _fixture(
        tmp_path / "capped",
        entries_per_shard=7,
        target_entries_per_range=3,
        worker_resources={
            "cpu": 2.0,
            "memory_mb": 8192,
            "max_containers": 32,
            "omp_num_threads": 1,
        },
    )
    resized = _fixture(
        tmp_path / "resized",
        entries_per_shard=7,
        target_entries_per_range=2,
    )

    assert baseline["source_bindings"] == capped["source_bindings"]
    assert baseline["run_identity_sha256"] != capped["run_identity_sha256"]
    assert (
        baseline["tasks"][0]["task_identity_sha256"] != capped["tasks"][0]["task_identity_sha256"]
    )
    assert baseline["run_identity_sha256"] != resized["run_identity_sha256"]
    assert (
        baseline["tasks"][0]["task_identity_sha256"] != resized["tasks"][0]["task_identity_sha256"]
    )


def test_plan_rejects_dirty_and_off_revision_source_bindings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    active8_identity = active8_implementation_identity()
    mapreduce_identity = mapreduce._implementation_identity()
    dirty = mapreduce._source_revision_binding(
        commit="1" * 40,
        tree="2" * 40,
        worktree_clean=False,
        active8_identity=active8_identity,
        mapreduce_identity=mapreduce_identity,
    )
    with pytest.raises(Active8MapReduceError, match="clean committed source revision"):
        _fixture(
            tmp_path / "dirty",
            source_revision=dirty,
        )

    valid = _source_revision()
    off_revision = copy.deepcopy(active8_identity)
    off_revision["implementation_sha256"] = "0" * 64
    monkeypatch.setattr(
        mapreduce,
        "active8_implementation_identity",
        lambda *, repo_root=None: off_revision,
    )
    with pytest.raises(Active8MapReduceError, match="differs from the live implementation"):
        _fixture(
            tmp_path / "off-revision",
            source_revision=valid,
        )


def test_repository_source_revision_rejects_dirty_or_changed_snapshot(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    active8_identity = active8_implementation_identity(repo_root=repository_root)
    mapreduce_identity = mapreduce._implementation_identity(repo_root=repository_root)
    copied_root = tmp_path / "git-snapshot"
    for relative in sorted(set(active8_identity["sources"]) | set(mapreduce_identity["sources"])):
        source = repository_root / relative
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    subprocess.run(("git", "init", "-q"), cwd=copied_root, check=True)
    subprocess.run(("git", "add", "."), cwd=copied_root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Active8 Test",
            "-c",
            "user.email=active8-test@example.invalid",
            "commit",
            "-qm",
            "freeze source snapshot",
        ),
        cwd=copied_root,
        check=True,
    )

    revision = mapreduce.repository_source_revision(repo_root=copied_root)
    assert revision["worktree_clean"] is True
    changed_source = copied_root / "src/compose_v4/rewrite/operators.py"
    changed_source.write_bytes(changed_source.read_bytes() + b"\n# off-revision probe\n")
    with pytest.raises(Active8MapReduceError, match="clean committed source worktree"):
        mapreduce.repository_source_revision(repo_root=copied_root)
    with pytest.raises(Active8MapReduceError, match="differs from the live implementation"):
        mapreduce._validate_source_revision_binding(
            revision,
            active8_identity=active8_implementation_identity(repo_root=copied_root),
            mapreduce_identity=mapreduce._implementation_identity(repo_root=copied_root),
        )


def test_ranged_reduce_is_decision_object_equivalent_to_serial(
    tmp_path: Path,
) -> None:
    serial_plan = _fixture(
        tmp_path,
        entries_per_shard=7,
        target_entries_per_range=100,
    )
    source_task = serial_plan["tasks"][0]
    source_manifest_path = tmp_path / "UNIFIED_PACKED_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source = Active8SourceShard(
        manifest_layer=source_task["manifest_layer"],
        envelope_layer=source_task["envelope_layer"],
        partition=source_task["partition"],
        relative_path=source_task["relative_path"],
        path=Path(source_task["source_path"]),
    )
    ranged_plan = plan_active8_mapreduce(
        (source,),
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        support_contract_sha256="a" * 64,
        source_revision=serial_plan["source_revision"],
        target_entries_per_range=2,
    )
    serial_output = tmp_path / "serial-output"
    ranged_output = tmp_path / "ranged-output"
    _map_all(serial_plan, serial_output)
    _map_all(ranged_plan, ranged_output)
    serial_complete = reduce_active8_mapreduce(
        serial_plan,
        output_root=serial_output,
    )
    ranged_complete = reduce_active8_mapreduce(
        ranged_plan,
        output_root=ranged_output,
    )
    serial_manifest = json.loads(
        (serial_output / serial_complete["inventory_manifest"]).read_text()
    )
    ranged_manifest = json.loads(
        (ranged_output / ranged_complete["inventory_manifest"]).read_text()
    )
    assert serial_complete["counts"] == ranged_complete["counts"]
    assert (
        serial_manifest["shards"][0]["inventory_shard_sha256"]
        == ranged_manifest["shards"][0]["inventory_shard_sha256"]
    )


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
    assert not (output / "runs" / plan["run_identity_sha256"] / "COMPLETE.json").exists()


def test_reducer_rejects_overlapping_plan_ranges_and_extra_receipts(
    tmp_path: Path,
) -> None:
    plan = _fixture(
        tmp_path,
        entries_per_shard=5,
        target_entries_per_range=2,
    )
    overlapped = copy.deepcopy(plan)
    overlapped["source_range_partitions"][0]["ranges"][1]["entry_start"] = 1
    identity_fields = (
        "schema",
        "schema_version",
        "source_manifest_sha256",
        "source_manifest_semantic_sha256",
        "support_contract_sha256",
        "source_revision",
        "source_revision_sha256",
        "active8_families",
        "active8_implementation_sha256",
        "mapreduce_implementation_sha256",
        "source_bindings",
        "range_policy",
        "range_policy_sha256",
        "source_range_partitions",
        "worker_resources",
        "worker_resources_sha256",
    )
    overlapped["run_identity_sha256"] = mapreduce._sha256_payload(
        {field: overlapped[field] for field in identity_fields}
    )
    with pytest.raises(
        Active8MapReduceError,
        match="not deterministic contiguous full coverage",
    ):
        reduce_active8_mapreduce(
            overlapped,
            output_root=tmp_path / "overlap-output",
        )

    output = tmp_path / "extra-output"
    _map_all(plan, output)
    receipts = output / "runs" / plan["run_identity_sha256"] / "receipts"
    (receipts / f"{'f' * 64}.json").write_text("{}")
    with pytest.raises(
        Active8MapReduceError,
        match="unexpected task identities",
    ):
        reduce_active8_mapreduce(plan, output_root=output)


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
        Path(__file__).resolve().parents[1] / "modal_apps" / "build_active8_trace_inventory_app.py"
    ).read_text()
    assert "_MAX_MAP_CONTAINERS_LIMIT = 64" in source
    assert '"COMPOSE_ACTIVE8_MAX_MAP_CONTAINERS", "64"' in source
    assert "_MAP_CPU = 2.0" in source
    assert "_MAP_MEMORY_MB = 8192" in source
    assert "_MAP_OMP_NUM_THREADS = 1" in source
    assert "_DEFAULT_TARGET_ENTRIES_PER_RANGE = 500" in source
    assert "create_if_missing=False" in source
    assert "source_revision = _local_source_revision()" in source
    assert "source_revision=source_revision" in source
    assert "max_containers=_MAX_MAP_CONTAINERS" in source
    assert "cpu=_MAP_CPU" in source
    assert "memory=_MAP_MEMORY_MB" in source
    assert "target_entries_per_range=target_entries_per_range" in source
    assert '"worker_resources": plan["worker_resources"]' in source
    assert "map_source.starmap" in source
    assert source.index("map_source.starmap") < source.index("return reduce_inventory.remote")
    assert "driver.remote" in source
    assert "driver.spawn" not in source
    assert '"training_launched": False' in source
