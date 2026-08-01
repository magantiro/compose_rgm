"""Whole-shard semantic migration and decision-accounting tests."""

from __future__ import annotations

import gzip
import hashlib
import json

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.packed_trace_store import (
    build_packed_entry,
    manifest_path_for,
    write_packed_shard,
)
from compose_v4.data.semantic_packed_trace_store import (
    MANIFEST_FILENAME,
    SHARD_FILENAME,
    read_semantic_packed_artifact,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    DECISION_FILENAME,
    SEMANTIC_ARTIFACT_DIRNAME,
    SemanticTraceMigrationTask,
    materialize_frozen_packed_shard,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, BondInsert
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import encode_trace_record


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_trace(source_smiles: str, rule_name: str, action) -> RewriteTrace:
    source = pad_molecular_graph(smiles_to_molecular_graph(source_smiles), 16)
    target = de_novo_rewrite_system().apply(source, rule_name, action)
    return RewriteTrace(
        source,
        target,
        (RewriteStep(rule_name, action),),
        {"scientific_metadata": "preserve-me"},
    )


def _source(tmp_path):
    accepted = _legacy_trace("CCCCCC", "bond_insert", BondInsert(0, 5, 1))
    rejected = _legacy_trace(
        "CC",
        "atom_insert",
        AtomInsert(
            slot=2,
            atom_type=2,
            formal_charge=0,
            implicit_h_count=2,
            neighbors=((0, 1), (1, 1)),
        ),
    )
    entries = []
    for index, trace in enumerate((accepted, rejected)):
        progress = TraceProgressCTMC(trace)
        envelope = encode_trace_record(
            trace,
            n_slots=16,
            seed=11 + index,
            trace_id=f"legacy-{index}",
            partition="train",
            layer="legacy_layer",
            extra=dict(trace.metadata),
        )
        entries.append(build_packed_entry(envelope, progress))
    source = tmp_path / "source.jsonl.gz"
    write_packed_shard(
        source,
        entries,
        provenance={"fixture": True},
        deterministic_gzip=True,
    )
    return source


def test_whole_shard_migration_accounts_for_acceptance_and_rejection(tmp_path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "migrated"
    task = SemanticTraceMigrationTask(
        source_path=source,
        source_shard_sha256=_sha256(source),
        source_manifest_sha256=_sha256(manifest_path_for(source)),
        source_overlay_sha256=None,
        source_unified_manifest_sha256="7" * 64,
        source_entry_count=2,
        implementation_revision="a" * 40,
        data_lane="operator_aware_real_endpoint",
        split="train",
        output_dir=output,
    )
    receipt = materialize_frozen_packed_shard(task)
    assert receipt["counts"] == {"admitted": 1, "rejected": 1, "source": 2}
    assert receipt["rejections_by_code"] == {"semantic_action_rejected": 1}

    with gzip.open(output / DECISION_FILENAME, "rt") as handle:
        decisions = [json.loads(line) for line in handle if line.strip()]
    assert [row["decision"] for row in decisions] == ["admitted", "rejected"]
    assert [row["source_address"]["entry_index"] for row in decisions] == [0, 1]
    assert all(len(row["decision_sha256"]) == 64 for row in decisions)

    semantic = output / SEMANTIC_ARTIFACT_DIRNAME
    rows = list(
        read_semantic_packed_artifact(
            semantic,
            expected_shard_sha256=receipt["semantic_shard_sha256"],
            expected_manifest_sha256=receipt["semantic_manifest_sha256"],
        )
    )
    assert len(rows) == 1
    assert rows[0].trace.steps[0].rule_name == "cycle_close"
    assert rows[0].trace.metadata == {"scientific_metadata": "preserve-me"}
    assert rows[0].address.trace_id.startswith("semantic-")
    manifest = json.loads((semantic / MANIFEST_FILENAME).read_text())
    assert manifest["training_authorized"] is False
    assert _sha256(semantic / SHARD_FILENAME) == receipt["semantic_shard_sha256"]


def test_whole_shard_migration_reuses_identical_complete_output(tmp_path) -> None:
    source = _source(tmp_path)
    task = SemanticTraceMigrationTask(
        source_path=source,
        source_shard_sha256=_sha256(source),
        source_manifest_sha256=_sha256(manifest_path_for(source)),
        source_overlay_sha256=None,
        source_unified_manifest_sha256="8" * 64,
        source_entry_count=2,
        implementation_revision="a" * 40,
        data_lane="operator_aware_real_endpoint",
        split="train",
        output_dir=tmp_path / "migrated",
    )
    first = materialize_frozen_packed_shard(task)
    second = materialize_frozen_packed_shard(task)
    assert first == second
