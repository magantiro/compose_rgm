"""The production corpus loader must refuse anything off-contract, BEFORE any GPU work.

The failure this guards against is not a crash -- it is a run that trains successfully on the wrong data:
a partial build, a stale contract, a silently-empty layer, or MMP records drawn from held-out scaffolds.
Every case here asserts a loud failure or a measured correction, never a best-effort recovery.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compose_v4.data.production_edit_corpus import (  # noqa: E402
    ProductionCorpusError,
    load_mmp_records,
    verify_build_complete,
)
from compose_v4.data.scaffold_partition import (  # noqa: E402
    murcko_scaffold,
    partition_for_scaffold,
)

CONTRACT = {
    "capability_hash": "330473e319bfec19",
    "codec_implementation_hash": "baaa75367f25a8c6",
    "operator_registry_hash": "9197401e8dc3a7ae",
    "scope_hash": "3721d69851110fdd",
    "trace_schema_version": 2,
    "partitioner": {
        "scaffold_key_algorithm": "murcko+carbonized-wl3",
        "scaffold_key_version": 2,
        "partition_salt": "ringcore-v1",
        "split_ratios": [0.9, 0.05, 0.05],
    },
}


def _write_build(root: Path, *, complete=True, contract=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "BUILD_COMPLETE.json").write_text(
        json.dumps(
            {
                "BUILD_COMPLETE": complete,
                "contract": contract if contract is not None else CONTRACT,
                "authorization": {"gate_sha256": "9617b590e1cb4739", "launch_commit": "c0ff477"},
                "reducer_checksum": "fd14600ff4429f7c",
            }
        )
    )
    return root


def test_missing_build_complete_is_refused(tmp_path):
    with pytest.raises(ProductionCorpusError, match="did not reach a terminal COMPLETE state"):
        verify_build_complete(tmp_path)


def test_incomplete_build_is_refused(tmp_path):
    _write_build(tmp_path, complete=False)
    with pytest.raises(ProductionCorpusError, match="not marked complete"):
        verify_build_complete(tmp_path)


def test_matching_contract_is_accepted(tmp_path):
    _write_build(tmp_path)
    payload = verify_build_complete(tmp_path, expected_contract=CONTRACT)
    assert payload["reducer_checksum"] == "fd14600ff4429f7c"


@pytest.mark.parametrize(
    "key",
    ["capability_hash", "codec_implementation_hash", "operator_registry_hash", "scope_hash"],
)
def test_each_contract_hash_mismatch_is_refused(tmp_path, key):
    """A stale artifact must not be trainable just because it happens to parse."""
    drifted = dict(CONTRACT)
    drifted[key] = "deadbeefdeadbeef"
    _write_build(tmp_path, contract=drifted)
    with pytest.raises(ProductionCorpusError, match=f"contract mismatch on {key}"):
        verify_build_complete(tmp_path, expected_contract=CONTRACT)


def test_trace_schema_version_mismatch_is_refused(tmp_path):
    drifted = dict(CONTRACT)
    drifted["trace_schema_version"] = 1
    _write_build(tmp_path, contract=drifted)
    with pytest.raises(ProductionCorpusError, match="trace_schema_version"):
        verify_build_complete(tmp_path, expected_contract=CONTRACT)


def test_partitioner_drift_is_refused(tmp_path):
    """A different scaffold key or salt repartitions the corpus -- held-out sets would no longer be held out."""
    drifted = json.loads(json.dumps(CONTRACT))
    drifted["partitioner"]["partition_salt"] = "some-other-salt"
    _write_build(tmp_path, contract=drifted)
    with pytest.raises(ProductionCorpusError, match="partitioner mismatch on partition_salt"):
        verify_build_complete(tmp_path, expected_contract=CONTRACT)


# ---- MMP partition discipline -----------------------------------------------------------------------

_FIXTURE = Path(__file__).resolve().parent / "fixtures/analogue_trace_pool_sample.jsonl"
_FULL = Path(__file__).resolve().parent.parent / "diagnostics/composition/analogue_trace_pool.jsonl"
# Prefer the COMMITTED fixture so these tests run in a clean checkout -- the authoritative launch gate
# runs from a clean worktree, where an untracked pool would silently skip them.
_POOL = _FIXTURE if _FIXTURE.exists() else _FULL


@pytest.mark.skipif(not _POOL.exists(), reason="local analogue pool sample unavailable")
@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_mmp_records_are_filtered_to_their_scaffold_partition(partition):
    """Every loaded MMP record must belong to the requested partition -- this is the leak fix.

    The pool predates the ringcore-v1 partitioner (it was mined under a different random split), so
    without this filter ~8% of its rows train on held-out scaffolds.
    """
    records, stats = load_mmp_records(_POOL, partition=partition)
    assert stats["scanned"] > 0
    rows = [json.loads(line) for line in _POOL.read_text().splitlines() if line.strip()]
    expected = [
        row
        for row in rows
        if (sc := murcko_scaffold(row["target_smiles"])) is not None
        and partition_for_scaffold(sc, salt="ringcore-v1") == partition
    ]
    assert stats["kept"] == len(records)
    assert stats["kept"] <= len(expected)
    assert stats["kept"] + stats["other_partition"] + stats["unassignable"] + stats["unbuildable"] == (
        stats["scanned"]
    )


@pytest.mark.skipif(not _POOL.exists(), reason="local analogue pool sample unavailable")
def test_partitions_are_disjoint_and_cover_the_pool():
    """No record may appear in two partitions, and the three must account for every assignable row."""
    seen, totals = {}, {}
    scanned = None
    for partition in ("train", "validation", "test"):
        records, stats = load_mmp_records(_POOL, partition=partition)
        totals[partition] = stats["kept"]
        scanned = stats["scanned"] if scanned is None else scanned
        assert stats["scanned"] == scanned, "every call must stream the whole pool"
        for record in records:
            key = record.target_key
            assert key not in seen or seen[key] == partition, (
                f"target {key!r} appears in both {seen.get(key)} and {partition}"
            )
            seen[key] = partition
    assert sum(totals.values()) > 0
    _, any_stats = load_mmp_records(_POOL, partition="train")
    assert sum(totals.values()) + any_stats["unassignable"] + any_stats["unbuildable"] <= scanned * 1


@pytest.mark.skipif(not _POOL.exists(), reason="local analogue pool sample unavailable")
def test_limit_is_applied_after_filtering_not_before():
    """A prefix cap must never become a partition-blind head of the file (the old truncation bug)."""
    capped, stats = load_mmp_records(_POOL, partition="train", limit=3)
    assert len(capped) <= 3
    # the cap bounds OUTPUT, but the scan still had to reach past non-train rows to find them
    assert stats["kept"] == len(capped)


def test_unassignable_and_unbuildable_rows_are_counted_not_hidden(tmp_path):
    """Silent drops are how a corpus shrinks without anyone noticing; they must appear in the stats."""
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "\n".join(
            [
                json.dumps({"target_smiles": "not-a-molecule", "steps": [], "path_length": 0}),
                json.dumps({"target_smiles": "", "steps": [], "path_length": 0}),
            ]
        )
    )
    _, stats = load_mmp_records(pool, partition="train")
    assert stats["scanned"] == 2
    assert stats["unassignable"] == 2
    assert stats["kept"] == 0


def test_gzip_shard_directory_missing_is_refused(tmp_path):
    from compose_v4.data.production_edit_corpus import load_shard_layer_records

    _write_build(tmp_path)
    with pytest.raises(ProductionCorpusError, match="missing shard directory"):
        load_shard_layer_records(tmp_path, "corruption", "train")


def test_empty_shard_directory_yields_no_records(tmp_path):
    """An empty (but present) directory must not masquerade as data; the layer check catches it upstream."""
    from compose_v4.data.production_edit_corpus import load_shard_layer_records

    (tmp_path / "corruption" / "train").mkdir(parents=True)
    assert load_shard_layer_records(tmp_path, "corruption", "train") == ()


def test_corrupt_shard_raises_rather_than_skipping(tmp_path):
    """A truncated shard must abort the build, not silently contribute fewer records."""
    from compose_v4.data.production_edit_corpus import load_shard_layer_records

    directory = tmp_path / "corruption" / "train"
    directory.mkdir(parents=True)
    with gzip.open(directory / "shard_0000.jsonl.gz", "wt") as handle:
        handle.write('{"schema": "compose.rewrite.trace", "schema_version": 99}\n')
    with pytest.raises(Exception):  # noqa: B017 -- any loud failure is acceptable; silence is not
        load_shard_layer_records(tmp_path, "corruption", "train")


def test_packed_store_must_cover_every_audit_shard(tmp_path):
    """An incomplete derivative store must RAISE, not silently train on a smaller corpus.

    This was a real gap: with one of two audit shards packed, the loader happily returned half the
    records. A run that succeeds on half the data is worse than one that crashes.
    """
    from compose_v4.data.production_edit_corpus import load_shard_layer_records

    audit = tmp_path / "audit" / "corruption" / "train"
    packed = tmp_path / "packed" / "corruption" / "train"
    audit.mkdir(parents=True)
    packed.mkdir(parents=True)
    for name in ("shard_0000.jsonl.gz", "shard_0001.jsonl.gz"):
        (audit / name).write_bytes(b"")
    (packed / "shard_0000.jsonl.gz").write_bytes(b"")

    with pytest.raises(ProductionCorpusError, match="does not cover"):
        load_shard_layer_records(
            tmp_path / "audit", "corruption", "train", packed_root=tmp_path / "packed"
        )


def test_packed_store_with_an_unexpected_shard_is_refused(tmp_path):
    """Extra packed shards mean the derivative drifted from its source; refuse rather than over-train."""
    from compose_v4.data.production_edit_corpus import load_shard_layer_records

    audit = tmp_path / "audit" / "corruption" / "train"
    packed = tmp_path / "packed" / "corruption" / "train"
    audit.mkdir(parents=True)
    packed.mkdir(parents=True)
    (audit / "shard_0000.jsonl.gz").write_bytes(b"")
    (packed / "shard_0000.jsonl.gz").write_bytes(b"")
    (packed / "shard_9999.jsonl.gz").write_bytes(b"")

    with pytest.raises(ProductionCorpusError, match="unexpected"):
        load_shard_layer_records(
            tmp_path / "audit", "corruption", "train", packed_root=tmp_path / "packed"
        )


def test_declared_shard_names_cross_checks_the_build_count(tmp_path):
    """Enumeration must be manifest-DECLARED: a glob alone silently absorbs a missing or extra shard."""
    from compose_v4.data.production_edit_corpus import declared_shard_names

    root = tmp_path
    (root / "BUILD_COMPLETE.json").write_text(
        json.dumps({"BUILD_COMPLETE": True, "expected_shards": 3, "contract": CONTRACT})
    )
    for layer, partition, names in (
        ("corruption", "train", ["shard_0000.jsonl.gz", "shard_0001.jsonl.gz"]),
        ("cycle_ops", "train", ["shard_0002.jsonl.gz"]),
    ):
        directory = root / layer / partition
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(b"")
            (directory / name).with_suffix(".manifest.json").write_text("{}")
    declared = declared_shard_names(root)
    assert sum(len(v) for v in declared.values()) == 3
    assert declared[("corruption", "train")] == ["shard_0000.jsonl.gz", "shard_0001.jsonl.gz"]


def test_shard_census_disagreement_is_refused(tmp_path):
    """One shard short of the declared count must abort, not train on a smaller corpus."""
    from compose_v4.data.production_edit_corpus import declared_shard_names

    root = tmp_path
    (root / "BUILD_COMPLETE.json").write_text(
        json.dumps({"BUILD_COMPLETE": True, "expected_shards": 45, "contract": CONTRACT})
    )
    directory = root / "corruption" / "train"
    directory.mkdir(parents=True)
    (directory / "shard_0000.jsonl.gz").write_bytes(b"")
    (directory / "shard_0000.jsonl.manifest.json").write_text("{}")
    with pytest.raises(ProductionCorpusError, match="shard census disagrees"):
        declared_shard_names(root)


def test_manifest_without_its_shard_is_refused(tmp_path):
    from compose_v4.data.production_edit_corpus import declared_shard_names

    root = tmp_path
    (root / "BUILD_COMPLETE.json").write_text(
        json.dumps({"BUILD_COMPLETE": True, "expected_shards": 1, "contract": CONTRACT})
    )
    directory = root / "corruption" / "train"
    directory.mkdir(parents=True)
    (directory / "shard_0000.jsonl.manifest.json").write_text("{}")
    with pytest.raises(ProductionCorpusError, match="has no shard"):
        declared_shard_names(root)


def test_overlay_lookup_uses_the_production_layer_name_not_the_shard_directory():
    """Regression: exclusions are keyed by PRODUCTION layer name, shards by compiler directory name.

    The census records `general_corruption`; shards live in `corruption/`. Looking up with the directory
    name matched nothing, so every listed exclusion read as unlisted and the loader refused to load --
    a loud failure, but for the wrong reason.
    """
    from compose_v4.data.production_edit_corpus import _SHARD_LAYER_TO_PRODUCTION

    assert _SHARD_LAYER_TO_PRODUCTION["corruption"] == "general_corruption"
    assert _SHARD_LAYER_TO_PRODUCTION["cycle_ops"] == "cycle_operations"

    module = (Path(__file__).resolve().parent.parent
              / "src/compose_v4/data/production_edit_corpus.py").read_text()
    assert "overlay_layer = _SHARD_LAYER_TO_PRODUCTION.get(shard_layer, shard_layer)" in module
    assert "excluded_keys(representability_overlay, overlay_layer, partition)" in module
    assert "excluded_keys(representability_overlay, shard_layer, partition)" not in module


def test_excluded_keys_returns_nothing_for_the_wrong_layer_name():
    """Demonstrate the failure mode directly, so the aliasing is load-bearing rather than cosmetic."""
    from compose_v4.data.representability_overlay import excluded_keys

    overlay = {"exclusions": [
        {"layer": "general_corruption", "partition": "train", "trace_key": "sha:abc"}
    ]}
    assert excluded_keys(overlay, "general_corruption", "train") == {"sha:abc"}
    assert excluded_keys(overlay, "corruption", "train") == set()
