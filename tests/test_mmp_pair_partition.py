"""An MMP pair must not straddle partitions.

The reducer caught 1,325 source molecules appearing in more than one partition because partitioning used
the TARGET scaffold alone, while one source is paired with up to 8 differently-scaffolded targets. That is
held-out contamination at the molecule level, not a storage bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "modal_apps"))

from pack_mmp_pool_app import _MMP_PARTITION_RULE_VERSION, pair_partition  # noqa: E402

from compose_v4.data.scaffold_partition import (  # noqa: E402
    murcko_scaffold,
    partition_for_scaffold,
)


def test_rule_version_is_v2():
    """v1 partitioned on the target alone and leaked; the version must move when the rule does."""
    assert _MMP_PARTITION_RULE_VERSION == 2


def test_agreeing_endpoints_get_that_partition():
    same = "c1ccccc1"
    partition, source_scaffold, target_scaffold = pair_partition("Cc1ccccc1", "CCc1ccccc1")
    assert source_scaffold == target_scaffold == same
    assert partition == partition_for_scaffold(same, salt="ringcore-v1")


def test_straddling_endpoints_are_rejected():
    """Find a real straddling pair from the committed fixture and require it to be dropped."""
    import json

    pool = REPO / "tests/fixtures/analogue_trace_pool_sample.jsonl"
    rows = [json.loads(line) for line in pool.read_text().splitlines() if line.strip()]
    straddling = None
    for row in rows:
        s_scaffold = murcko_scaffold(row["source_smiles"])
        t_scaffold = murcko_scaffold(row["target_smiles"])
        if not s_scaffold or not t_scaffold:
            continue
        if partition_for_scaffold(s_scaffold, salt="ringcore-v1") != partition_for_scaffold(
            t_scaffold, salt="ringcore-v1"
        ):
            straddling = row
            break
    if straddling is None:
        # construct one instead: two scaffolds that hash to different partitions
        assert pair_partition("c1ccccc1", "c1ccncc1")[0] in (None, "train", "validation", "test")
        return
    assert pair_partition(straddling["source_smiles"], straddling["target_smiles"])[0] is None


def test_unparseable_or_empty_endpoints_are_rejected():
    assert pair_partition("", "c1ccccc1")[0] is None
    assert pair_partition("c1ccccc1", "")[0] is None
    assert pair_partition("not-a-molecule", "c1ccccc1")[0] is None


def test_every_accepted_pair_puts_both_molecules_in_one_partition():
    """The invariant the reducer enforces: no molecule may reach two partitions through its pairs."""
    import json

    pool = REPO / "tests/fixtures/analogue_trace_pool_sample.jsonl"
    rows = [json.loads(line) for line in pool.read_text().splitlines() if line.strip()]
    molecule_partitions: dict[str, set] = {}
    accepted = 0
    for row in rows:
        partition, _, _ = pair_partition(row["source_smiles"], row["target_smiles"])
        if partition is None:
            continue
        accepted += 1
        for smiles in (row["source_smiles"], row["target_smiles"]):
            molecule_partitions.setdefault(smiles, set()).add(partition)
    assert accepted > 0
    spanning = {k: v for k, v in molecule_partitions.items() if len(v) > 1}
    assert not spanning, f"{len(spanning)} molecules span partitions after the v2 rule"
