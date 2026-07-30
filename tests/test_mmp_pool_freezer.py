"""V2 MMP mining must be partition-invariant and freeze only complete, coherent pools."""
from __future__ import annotations

import errno
import json
import os
import stat
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "modal_apps"))

from mine_edit_traces import MiningConfig, compile_mmp, dedup_and_cap  # noqa: E402
from mine_edit_traces_app import (  # noqa: E402
    MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS,
    v2_run_subdir,
)
from pack_mmp_pool_app import build_pack_request, preflight_frozen_input  # noqa: E402

from compose_v4.data.mmp_pool_freezer import (  # noqa: E402
    REDUCTION_SCHEMA,
    MMPPoolFreezeError,
    build_compile_shard_manifest,
    build_compiler_identity,
    canonical_sha256,
    file_sha256,
    freeze_reduced_mmp_pool,
    partition_independent_mining_config,
    run_write_once_storage_preflight,
    validate_compile_inventory,
    write_bytes_if_absent,
)
from compose_v4.experiments.analogue_prior import ANALOGUE_SUPPORT_CONTRACT  # noqa: E402


def _jsonl_bytes(rows) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _compiled_row(pair_index: int, direction: str) -> dict:
    forward = direction == "forward"
    return {
        "pair_index": pair_index,
        "direction": direction,
        "source_smiles": "shared-source" if forward else f"target-{pair_index}",
        "target_key": f"target-{pair_index}" if forward else "shared-source",
        "target_smiles": f"target-{pair_index}" if forward else "shared-source",
        "n_slots": 40,
        "path_length": 1,
        "operator_histogram": {"atom_restate": 1},
        "steps": [{"rule": "atom_restate", "v": 0}],
        "layer": "mmp_one_cut",
        "metadata": {
            "support_contract": ANALOGUE_SUPPORT_CONTRACT,
            "compiler_path_class": "direct_atom_restate",
        },
        "diagnostics": {
            "support_contract": ANALOGUE_SUPPORT_CONTRACT,
        },
    }


def _write_run(
    artifact_root: Path,
    run_name: str,
    n_shards: int,
    *,
    corpus_sha256: str = "1" * 64,
) -> tuple[Path, dict]:
    run = artifact_root / run_name
    compiled = run / "compiled"
    compiled.mkdir(parents=True)
    pairs = [[f"S{i}", f"T{i}", f"C{i}"] for i in range(7)]
    pairs_path = run / "pairs.jsonl"
    pairs_path.write_bytes(_jsonl_bytes(pairs))
    pairs_sha256 = file_sha256(pairs_path)

    config = asdict(MiningConfig(
        corpus_id="fixture-corpus",
        corpus_path="/guacamol/fixture.smi",
        corpus_sha256=corpus_sha256,
        train_size=100,
        validation_size=10,
        test_size=10,
        max_pairs_per_source=2,
        split_workers=n_shards + 1,
        n_shards=n_shards,
        shard_index=0,
        out_dir=f"/artifacts/{run_name}",
    ))
    scientific_config = partition_independent_mining_config(config)
    config_sha256 = canonical_sha256(scientific_config)
    compiler = build_compiler_identity(
        {
            "scripts/build_analogue_trace_pool.py": "2" * 64,
            "scripts/mine_edit_traces.py": "3" * 64,
        },
        runtime_versions={
            "networkx": "fixture",
            "numpy": "fixture",
            "python": "fixture",
            "rdkit": "fixture",
        },
    )

    for shard_index in range(n_shards):
        rows = [
            _compiled_row(pair_index, direction)
            for pair_index in range(len(pairs))
            if pair_index % n_shards == shard_index
            for direction in ("forward", "reverse")
        ]
        name = f"shard_{shard_index:04d}.jsonl"
        shard_path = compiled / name
        shard_path.write_bytes(_jsonl_bytes(rows))
        manifest = build_compile_shard_manifest(
            shard_path=shard_path,
            shard_artifact_path=f"/artifacts/{run_name}/compiled/{name}",
            shard_index=shard_index,
            n_shards=n_shards,
            total_pairs=len(pairs),
            source_pairs_path=f"/artifacts/{run_name}/pairs.jsonl",
            source_pairs_sha256=pairs_sha256,
            mining_config_sha256=config_sha256,
            compiler_identity=compiler,
        )
        (compiled / (name + ".manifest.json")).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    mine_meta = {
        "config": config,
        "provenance": {
            "corpus_id": "fixture-corpus",
            "corpus_sha256": corpus_sha256,
            "corpus_scope": "broad_organic_v1",
            "corpus_scope_hash": "scope-fixture",
            "compiler_identity_v2": compiler,
        },
        "pairs": {
            "path": f"/artifacts/{run_name}/pairs.jsonl",
            "sha256": pairs_sha256,
            "records": len(pairs),
        },
        "partition_independent_mining_config": scientific_config,
        "mining_config_sha256": config_sha256,
        "n_pairs": len(pairs),
    }
    mine_meta_path = run / "mine_meta.json"
    mine_meta_path.write_text(json.dumps(mine_meta, indent=2, sort_keys=True) + "\n")

    inventory = validate_compile_inventory(
        artifact_root=artifact_root,
        source_run_root=f"/artifacts/{run_name}",
        expected_shards=n_shards,
        total_pairs=len(pairs),
        source_pairs_path=f"/artifacts/{run_name}/pairs.jsonl",
        source_pairs_sha256=pairs_sha256,
        mining_config_sha256=config_sha256,
        compiler_identity=compiler,
    )
    pool, cap_report = dedup_and_cap(list(inventory.rows), MiningConfig(**config))
    pool_path = run / "edit_pool_full.jsonl"
    pool_path.write_bytes(_jsonl_bytes(pool))

    summary = {
        "schema": REDUCTION_SCHEMA,
        "REDUCTION_COMPLETE": True,
        "source_run_root": f"/artifacts/{run_name}",
        "summary_artifact_path": f"/artifacts/{run_name}/mining_summary_full.json",
        "config": config,
        "partition_independent_mining_config": scientific_config,
        "mining_config_sha256": config_sha256,
        "compiler_identity": compiler,
        "source_inputs": {
            "corpus": {
                "id": "fixture-corpus",
                "path": "/guacamol/fixture.smi",
                "sha256": corpus_sha256,
                "scope_name": "broad_organic_v1",
                "scope_hash": "scope-fixture",
            },
            "pairs": mine_meta["pairs"],
            "mine_meta": {
                "path": f"/artifacts/{run_name}/mine_meta.json",
                "sha256": file_sha256(mine_meta_path),
            },
        },
        "compile": {
            "expected_shards": n_shards,
            "records": inventory.record_count,
            "records_sha256": inventory.records_sha256,
            "shards": list(inventory.shards),
            "coverage": {
                "rule": "global_pair_index_mod_n_shards",
                "total_pairs": len(pairs),
                "unique_assigned_pairs": len(pairs),
                "missing_pairs": 0,
                "overlapping_pairs": 0,
            },
        },
        "global_reduce": cap_report,
        "pool": {
            "path": f"/artifacts/{run_name}/edit_pool_full.jsonl",
            "sha256": file_sha256(pool_path),
            "records": len(pool),
        },
    }
    summary_path = run / "mining_summary_full.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path, summary


def test_global_pair_indices_make_pool_bytes_invariant_to_compile_shard_count(tmp_path):
    summary_two, payload_two = _write_run(tmp_path, "run_two", 2)
    summary_three, payload_three = _write_run(tmp_path, "run_three", 3)
    pool_two = summary_two.parent / "edit_pool_full.jsonl"
    pool_three = summary_three.parent / "edit_pool_full.jsonl"

    assert pool_two.read_bytes() == pool_three.read_bytes()
    assert file_sha256(pool_two) == file_sha256(pool_three)
    assert (
        payload_two["partition_independent_mining_config"]
        == payload_three["partition_independent_mining_config"]
    )
    assert payload_two["mining_config_sha256"] == payload_three["mining_config_sha256"]


def test_write_once_falls_back_when_modal_volume_rejects_hard_links(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "content-addressed" / "artifact.bin"
    content = b"frozen-content\n"

    def unsupported_hard_link(_source, _destination):
        raise OSError(errno.EPERM, "Modal Volume does not support hard links")

    monkeypatch.setattr(os, "link", unsupported_hard_link)
    assert write_bytes_if_absent(destination, content) is True
    assert destination.read_bytes() == content
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert not list(destination.parent.glob("*.tmp"))


def test_modal_fallback_reuses_identical_bytes_and_rejects_collisions(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    original = b"authoritative bytes"

    def unsupported_hard_link(_source, _destination):
        raise OSError(errno.EPERM, "hard links unsupported")

    monkeypatch.setattr(os, "link", unsupported_hard_link)
    destination.write_bytes(original)

    assert write_bytes_if_absent(destination, original) is False
    with pytest.raises(MMPPoolFreezeError, match="different bytes"):
        write_bytes_if_absent(destination, b"collision")
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_write_once_does_not_mask_non_capability_link_failures(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"

    def io_failure(_source, _destination):
        raise OSError(errno.EIO, "storage failure")

    monkeypatch.setattr(os, "link", io_failure)
    with pytest.raises(OSError, match="storage failure"):
        write_bytes_if_absent(destination, b"content")
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_once_rejects_non_regular_preexisting_destination(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    destination.mkdir()

    def unsupported_hard_link(_source, _destination):
        raise OSError(errno.EPERM, "hard links unsupported")

    monkeypatch.setattr(os, "link", unsupported_hard_link)
    with pytest.raises(MMPPoolFreezeError, match="not a regular file"):
        write_bytes_if_absent(destination, b"content")
    assert destination.is_dir()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_once_storage_preflight_is_bounded_idempotent_and_isolated(
    tmp_path,
    monkeypatch,
):
    scientific = tmp_path / "scientific-run"
    scientific.mkdir()
    marker = scientific / "do-not-touch"
    marker.write_bytes(b"owned")

    def unsupported_hard_link(_source, _destination):
        raise OSError(errno.EPERM, "hard links unsupported")

    monkeypatch.setattr(os, "link", unsupported_hard_link)
    first = run_write_once_storage_preflight(tmp_path)
    second = run_write_once_storage_preflight(tmp_path)

    assert first["status"] == second["status"] == "PASS"
    assert first["preflight_identity"] == second["preflight_identity"]
    assert first["probe_created_this_invocation"] is True
    assert second["probe_created_this_invocation"] is False
    assert marker.read_bytes() == b"owned"
    scratch = Path(first["scratch_namespace"])
    assert "edit_mining_v2_run_" not in str(scratch)
    assert set(path.name for path in scratch.iterdir()) == {
        "PREFLIGHT_PASS.json",
        "probe.bin",
    }
    receipt = json.loads((scratch / "PREFLIGHT_PASS.json").read_text())
    assert receipt["receipt_sha256"] == canonical_sha256({
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    })
    assert not list(scratch.glob("*.tmp"))


def test_v2_modal_storage_preflight_precedes_expensive_pair_mining():
    source = (REPO / "modal_apps" / "mine_edit_traces_app.py").read_text()
    assert source.index("storage_preflight.remote") < source.index("mine_pairs.remote")
    assert MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS == 5
    compile_surface = source[
        source.index("@app.function", source.index("def storage_preflight")) :
        source.index("def compile_shard")
    ]
    assert "max_containers=MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS" in compile_surface


def test_local_compiler_preserves_caller_global_pair_index():
    records = compile_mmp(
        [("Cc1ccccc1", "CCc1ccccc1", "c1ccccc1")],
        MiningConfig(),
        pair_indices=[17],
    )
    assert records
    assert {record["pair_index"] for record in records} == {17}
    with pytest.raises(ValueError, match="length"):
        compile_mmp(
            [("Cc1ccccc1", "CCc1ccccc1", "c1ccccc1")],
            MiningConfig(),
            pair_indices=[],
        )


def test_freezer_publishes_packer_contract_in_distinct_immutable_namespace(tmp_path):
    summary_path, _ = _write_run(tmp_path, "run", 3)

    first = freeze_reduced_mmp_pool(summary_path, artifact_root=tmp_path)
    second = freeze_reduced_mmp_pool(summary_path, artifact_root=tmp_path)

    assert first == second
    assert first["namespace"].startswith("mmp_pool_v2_")
    assert "/mmp_pool_frozen_v2/mmp_pool_v2_" in first["pool_path"]
    contract_path = (
        tmp_path / Path(first["contract_path"]).relative_to("/artifacts")
    )
    contract = json.loads(contract_path.read_text())
    assert contract["MMP_POOL_COMPLETE"] is True
    assert contract["pool_sha256"] == first["pool_sha256"]
    assert contract["pool_records"] == first["pool_records"]
    assert contract["analogue_support_contract"] == ANALOGUE_SUPPORT_CONTRACT
    assert "split_workers" not in contract["partition_independent_mining_config"]
    assert "n_shards" not in contract["partition_independent_mining_config"]

    request = build_pack_request(
        pool_path=first["pool_path"],
        expected_pool_sha256=first["pool_sha256"],
        expected_pool_records=first["pool_records"],
        pool_contract_path=first["contract_path"],
        expected_pool_contract_sha256=first["contract_sha256"],
        expected_analogue_support_contract=ANALOGUE_SUPPORT_CONTRACT,
        launch_commit="a7546e2",
        gate_sha256="0123456789abcdef",
    )
    observed = preflight_frozen_input(request, artifact_root=tmp_path)
    assert observed["pool_sha256"] == first["pool_sha256"]
    assert observed["row_support_contract_verified"] is True


def test_missing_compile_shard_is_rejected(tmp_path):
    summary_path, _ = _write_run(tmp_path, "run", 3)
    (summary_path.parent / "compiled" / "shard_0002.jsonl").unlink()

    with pytest.raises(MMPPoolFreezeError, match="partial or unexpected"):
        freeze_reduced_mmp_pool(summary_path, artifact_root=tmp_path)


def test_overlapping_compile_assignment_is_rejected(tmp_path):
    summary_path, _ = _write_run(tmp_path, "run", 2)
    manifest_path = summary_path.parent / "compiled" / "shard_0001.jsonl.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shard_index"] = 0
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(MMPPoolFreezeError, match="disagrees"):
        freeze_reduced_mmp_pool(summary_path, artifact_root=tmp_path)


def test_partial_reduction_and_mixed_contract_rows_are_rejected(tmp_path):
    summary_path, summary = _write_run(tmp_path, "partial", 2)
    summary["REDUCTION_COMPLETE"] = False
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MMPPoolFreezeError, match="partial"):
        freeze_reduced_mmp_pool(summary_path, artifact_root=tmp_path)

    summary_path, summary = _write_run(tmp_path, "mixed", 2)
    pool_path = summary_path.parent / "edit_pool_full.jsonl"
    rows = [json.loads(line) for line in pool_path.read_text().splitlines() if line]
    rows[0]["metadata"]["support_contract"] = "stale-contract"
    pool_path.write_bytes(_jsonl_bytes(rows))
    summary["pool"]["sha256"] = file_sha256(pool_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MMPPoolFreezeError, match="mixed or missing"):
        freeze_reduced_mmp_pool(summary_path, artifact_root=tmp_path)


def test_existing_frozen_contract_cannot_be_overwritten_or_reblessed(tmp_path):
    summary_path, _ = _write_run(tmp_path, "run", 2)
    frozen = freeze_reduced_mmp_pool(summary_path, artifact_root=tmp_path)
    contract_path = tmp_path / Path(frozen["contract_path"]).relative_to("/artifacts")
    contract_path.write_text('{"tampered":true}\n')

    with pytest.raises(MMPPoolFreezeError, match="different bytes"):
        freeze_reduced_mmp_pool(summary_path, artifact_root=tmp_path)


def test_freeze_identity_binds_source_and_compiler_evidence(tmp_path):
    summary_a, _ = _write_run(tmp_path, "run_a", 2, corpus_sha256="1" * 64)
    summary_b, _ = _write_run(tmp_path, "run_b", 2, corpus_sha256="9" * 64)
    frozen_a = freeze_reduced_mmp_pool(summary_a, artifact_root=tmp_path)
    frozen_b = freeze_reduced_mmp_pool(summary_b, artifact_root=tmp_path)
    assert frozen_a["freeze_identity"] != frozen_b["freeze_identity"]


def test_compiler_identity_is_full_hash_bound():
    identity = build_compiler_identity(
        {"compiler.py": "a" * 64},
        runtime_versions={"python": "3.11.fixture"},
    )
    assert len(identity["identity_sha256"]) == 64
    assert identity["analogue_support_contract"] == ANALOGUE_SUPPORT_CONTRACT
    assert identity["runtime_versions"] == {"python": "3.11.fixture"}
    assert identity["identity_sha256"] == canonical_sha256({
        key: value for key, value in identity.items() if key != "identity_sha256"
    })
    with pytest.raises(MMPPoolFreezeError, match="runtime versions"):
        build_compiler_identity(
            {"compiler.py": "a" * 64},
            runtime_versions={},
        )


def test_v2_mining_run_namespace_is_computed_and_never_defaults_to_v1():
    config = asdict(MiningConfig(corpus_sha256="1" * 64))
    two = v2_run_subdir(
        config,
        source_commit="a" * 40,
        expected_corpus_sha256="1" * 64,
        n_compile_shards=2,
    )
    three = v2_run_subdir(
        config,
        source_commit="a" * 40,
        expected_corpus_sha256="1" * 64,
        n_compile_shards=3,
    )
    assert two.startswith("edit_mining_v2_run_")
    assert two != three
    assert "v1" not in two
    with pytest.raises(ValueError, match="full lowercase"):
        v2_run_subdir(
            config,
            source_commit="a" * 40,
            expected_corpus_sha256="unknown",
            n_compile_shards=2,
        )
    with pytest.raises(ValueError, match="full lowercase 40-character"):
        v2_run_subdir(
            config,
            source_commit="a7546e2",
            expected_corpus_sha256="1" * 64,
            n_compile_shards=2,
        )
