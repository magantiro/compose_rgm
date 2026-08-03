"""Equivalence on a production-shaped source, not only on bounded fixtures.

The twenty-cell fixture in ``test_process_v2_chunk_fed_rebind`` proves the two
geometries agree in *breadth*: every lane and role, both admission outcomes, a
rejected trace, a multi-chunk address space.  Its rows are small -- sixteen slots
and short paths -- so it does not exercise the axis a production shard differs
on, which is row and state size.

This module closes that gap on the axis that actually differs.  Its corpus is
built by the benchmark's own production-shaped builder: forty slots, eight
steps, and a persisted row within a few percent of the measured production mean
of ~8.7 kB.  The record counts here are deliberately small, because a Process-V2
proof costs about a second per record at this molecule size on this hardware --
a measured fact reported with the benchmark, not an accident of the fixture --
so the module states which property each test covers rather than implying scale
it does not have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_process_v2_rebind import (
    MANIFEST_FILENAME,
    RECEIPT_FILENAME,
    TASK_DIRNAME,
    bind_process_v2_chunk_cache_generation,
    build_process_v2_rebind_source_revision,
    execute_process_v2_rebind_task,
    plan_process_v2_rebind,
    read_v1_records_through_the_bounded_raw_oracle,
    reduce_process_v2_rebind,
    write_process_v2_rebind_plan,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    execute_process_v2_chunk_cache_task,
    open_process_v2_chunk_cache,
    plan_process_v2_chunk_cache,
    read_process_v2_chunk_target,
    reduce_process_v2_chunk_cache,
    write_process_v2_chunk_cache_plan,
)
from compose_v4.data.semantic_packed_trace_store import semantic_packed_builder_identity
from compose_v4.rewrite.editing_v2_process_identity import editing_v2_process_identity

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (str(_REPO_ROOT / "tests"), str(_REPO_ROOT / "scripts")):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import benchmark_process_v2_chunk_sizes as bench  # noqa: E402
import test_process_v2_chunk_target_reader as reader_fixture  # noqa: E402

ROOT = _REPO_ROOT
_row_projection = reader_fixture._row_projection

# The production mean, measured in Wave 1 on one real shard: 281.8 MB
# decompressed over 32,339 rows.
PRODUCTION_BYTES_PER_ROW = bench.PRODUCTION_SHARD_DECOMPRESSED_BYTES / (
    bench.PRODUCTION_SHARD_ENTRIES
)


def _production_shaped_cache(tmp_path: Path, *, entries: int, records_per_chunk: int):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    source = bench.build_benchmark_payload(artifact_root, entries=entries)
    binding = bench._binding(artifact_root, source)
    cache_plan = plan_process_v2_chunk_cache(
        binding,
        output_artifact_prefix="/artifacts/production_shape_cache",
        records_per_chunk=records_per_chunk,
    )
    write_process_v2_chunk_cache_plan(cache_plan, artifact_root=artifact_root)
    for task in cache_plan["tasks"]:
        execute_process_v2_chunk_cache_task(
            cache_plan, task["task_identity_sha256"], artifact_root=artifact_root
        )
    completion = reduce_process_v2_chunk_cache(cache_plan, artifact_root=artifact_root)
    generation = open_process_v2_chunk_cache(completion, artifact_root=artifact_root)
    return artifact_root, source, binding, cache_plan, generation


def _oracle_rows(artifact_root: Path, source: dict[str, Any], target) -> list[Any]:
    v1_task_dir = (
        artifact_root
        / "benchmark_v1_payload"
        / TASK_DIRNAME
        / str(source["task_identity_sha256"])
    )
    receipt = json.loads((v1_task_dir / "RECEIPT.json").read_bytes())
    task = {
        "v1_semantic_shard_sha256": receipt["semantic_shard_sha256"],
        "v1_semantic_manifest_sha256": receipt["semantic_manifest_sha256"],
        "v1_source_binding": receipt["source_binding"],
        "entry_start": target.entry_start,
        "entry_stop": target.entry_stop,
    }
    return list(
        read_v1_records_through_the_bounded_raw_oracle(
            task,
            v1_task_dir=v1_task_dir,
            pinned_process_identity=json.loads(json.dumps(editing_v2_process_identity())),
            pinned_builder_identity=json.loads(json.dumps(semantic_packed_builder_identity())),
        )
    )


# ---- Decoded rows, at production row size, over a real chunk partition --------


def test_a_production_shaped_source_decodes_identically(tmp_path: Path) -> None:
    """Covers: decoded semantic rows, at production row and state size.

    Not the proof path -- that is the next test, which is bounded by cost.
    """

    artifact_root, source, _binding, _cache_plan, generation = _production_shaped_cache(
        tmp_path, entries=96, records_per_chunk=16
    )
    observed_bytes_per_row = int(source["decompressed_bytes"]) / int(source["entries"])
    assert 0.75 * PRODUCTION_BYTES_PER_ROW <= observed_bytes_per_row <= (
        1.25 * PRODUCTION_BYTES_PER_ROW
    ), (
        "the corpus must be production-shaped on the axis this module exists for; "
        f"observed {observed_bytes_per_row:.0f} B/row against {PRODUCTION_BYTES_PER_ROW:.0f}"
    )

    targets = generation.targets()
    assert len(targets) == 6
    compared = 0
    for target in targets:
        source_dir = artifact_root / target.source_artifact_path.removeprefix("/artifacts/")
        cached = [
            _row_projection(row)
            for row in read_process_v2_chunk_target(
                source_dir,
                target=target,
                expected_process_identity=json.loads(
                    json.dumps(editing_v2_process_identity())
                ),
                recover_row_errors=True,
            )
        ]
        oracle = [_row_projection(row) for row in _oracle_rows(artifact_root, source, target)]
        assert cached == oracle
        compared += len(cached)
    assert compared == 96
    # Chunks past the first start above zero, so a chunk-local index would be
    # visibly wrong here rather than coincidentally right.
    assert targets[-1].entry_start == 80


# ---- Proof outcomes, at production row size ----------------------------------


@pytest.mark.parametrize("entries,records_per_chunk", [(16, 8)])
def test_a_production_shaped_source_proves_identically(
    tmp_path: Path, entries: int, records_per_chunk: int
) -> None:
    """Covers: proof decisions, whole-trace admission, reason codes, censuses.

    Deliberately small.  One Process-V2 proof costs about a second per record at
    this molecule size, so a larger corpus would buy minutes of runtime and no
    additional property: what differs from the twenty-cell fixture is the size
    of each record, not how many there are.
    """

    artifact_root, source, binding, cache_plan, _generation = _production_shaped_cache(
        tmp_path, entries=entries, records_per_chunk=records_per_chunk
    )
    process_identity = json.loads(json.dumps(editing_v2_process_identity()))
    builder_identity = json.loads(json.dumps(semantic_packed_builder_identity()))
    revision = build_process_v2_rebind_source_revision(
        commit="a" * 40, tree="b" * 40, repo_root=ROOT, worktree_clean=True
    )
    cache_binding = bind_process_v2_chunk_cache_generation(
        str(cache_plan["run_artifact_root"]), artifact_root=artifact_root, repo_root=ROOT
    )

    def _run(prefix: str, *, cached: bool):
        plan = plan_process_v2_rebind(
            binding.v1_payload_binding,
            source_revision=revision,
            repo_root=ROOT,
            pinned_process_identity=process_identity,
            pinned_builder_identity=builder_identity,
            cache_binding=cache_binding if cached else None,
            output_artifact_prefix=prefix,
            entries_per_task=records_per_chunk,
        )
        write_process_v2_rebind_plan(plan, artifact_root=artifact_root, repo_root=ROOT)
        for task in plan["tasks"]:
            execute_process_v2_rebind_task(
                plan,
                task["task_identity_sha256"],
                artifact_root=artifact_root,
                repo_root=ROOT,
            )
        completion = reduce_process_v2_rebind(
            plan, artifact_root=artifact_root, repo_root=ROOT
        )
        documents = []
        for task in sorted(plan["tasks"], key=lambda row: int(row["entry_start"])):
            output = artifact_root / str(task["output_artifact_path"]).removeprefix("/artifacts/")
            documents.append(
                {
                    "receipt": json.loads((output / RECEIPT_FILENAME).read_bytes()),
                    "manifest": json.loads((output / MANIFEST_FILENAME).read_bytes()),
                }
            )
        return completion, documents

    cached_completion, cached_documents = _run("/artifacts/production_shape_cached", cached=True)
    oracle_completion, oracle_documents = _run("/artifacts/production_shape_oracle", cached=False)

    assert len(cached_documents) == entries // records_per_chunk
    for cached, oracle in zip(cached_documents, oracle_documents, strict=True):
        assert cached["receipt"]["proof_stream_sha256"] == (
            oracle["receipt"]["proof_stream_sha256"]
        )
        assert cached["receipt"]["counts"] == oracle["receipt"]["counts"]
        assert cached["manifest"]["v1_record_inventory_sha256"] == (
            oracle["manifest"]["v1_record_inventory_sha256"]
        )
        assert cached["manifest"]["rejected_traces"] == oracle["manifest"]["rejected_traces"]
        assert cached["manifest"]["process_v2_atom_delete_census"] == (
            oracle["manifest"]["process_v2_atom_delete_census"]
        )
    for field in (
        "counts",
        "family_histogram",
        "teacher_census",
        "process_v2_atom_delete_census",
        "rejected_traces_by_code",
        "unsupported_teacher_steps_by_code",
    ):
        assert cached_completion[field] == oracle_completion[field], field
    assert cached_completion["counts"]["source_entries"] == entries
    # Whatever the fixture chemistry decides, both geometries decided it, and at
    # least one record was actually accounted for on each side.
    assert (
        cached_completion["counts"]["admitted_entries"]
        + cached_completion["counts"]["rejected_entries"]
        == entries
    )
