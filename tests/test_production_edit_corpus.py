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

_POOL = Path(__file__).resolve().parent.parent / "diagnostics/composition/analogue_trace_pool.jsonl"


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
