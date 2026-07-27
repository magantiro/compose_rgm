"""Step 1b: the shardable global-grouping edit-trace miner's correctness invariants.

The load-bearing claim is that mining is a genuine map -> GLOBAL-group -> compile -> global-reduce, NOT
independent per-shard mining: two molecules sharing an MMP core must be paired even when they land in
DIFFERENT shards. We also pin the deterministic shard partition and that every compiled pool record is
re-consumable (replays through the executor to its recorded endpoint).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_analogue_trace_pool import rewrite_trace_from_record  # noqa: E402
from mine_edit_traces import (  # noqa: E402
    MiningConfig,
    ShardEmission,
    characterize_corruption,
    compile_mmp,
    group_mmp_pairs,
    map_shard,
    shard_of,
)

from compose_v4.rewrite.kernel import canonical_state_key  # noqa: E402

_CFG = MiningConfig(max_atoms=32, corruption_sample_size=3)
# toluene and ethylbenzene share the benzene core under a single acyclic cut (CH3 vs C2H5 R-group).
_TOLUENE = "Cc1ccccc1"
_ETHYLBENZENE = "CCc1ccccc1"


def test_global_grouping_finds_cross_shard_pairs() -> None:
    # place the two core-sharing molecules in SEPARATE shards.
    shard_a = map_shard([_TOLUENE], _CFG)
    shard_b = map_shard([_ETHYLBENZENE], _CFG)
    # per-shard grouping sees one molecule each -> zero pairs (no within-shard partner).
    assert group_mmp_pairs(shard_a, _CFG) == []
    assert group_mmp_pairs(shard_b, _CFG) == []
    # the GLOBAL reduce merges every shard's emissions -> the cross-shard pair appears.
    merged = ShardEmission(_CFG.corpus_id, None, _CFG.n_shards)
    merged.merge(shard_a)
    merged.merge(shard_b)
    pairs = group_mmp_pairs(merged, _CFG)
    assert len(pairs) == 1, pairs
    members = {pairs[0][0], pairs[0][1]}
    assert members == {_TOLUENE, _ETHYLBENZENE}
    # the shared key is the benzene core (identical canonical fragment from both molecules).
    assert pairs[0][2] == "c1ccccc1"


def test_shard_stride_is_a_deterministic_partition() -> None:
    train = tuple(f"C{'C' * (i % 5)}O" for i in range(37))  # arbitrary deterministic list
    n_shards = 4
    covered: list[str] = []
    for k in range(n_shards):
        cfg = MiningConfig(n_shards=n_shards, shard_index=k, shard_rule="stride")
        covered.extend(shard_of(train, cfg))
    ordered = sorted(train)
    # every stride-shard is disjoint and their union is exactly the canonical-ordered train list.
    assert sorted(covered) == ordered
    assert len(covered) == len(ordered)


def test_compiled_pool_records_are_reconsumable() -> None:
    merged = ShardEmission(_CFG.corpus_id, None, _CFG.n_shards)
    merged.merge(map_shard([_TOLUENE, _ETHYLBENZENE], _CFG))
    records = compile_mmp(group_mmp_pairs(merged, _CFG), _CFG)
    assert records, "expected at least one compiled MMP record"
    for record in records:
        trace = rewrite_trace_from_record(record)  # rebuild + re-execute from the serialized steps
        assert canonical_state_key(trace.target) == record["target_key"]
        assert record["direction"] in ("forward", "reverse")
        assert record["layer"] == "mmp_one_cut"


def test_corruption_drift_tanimoto_is_a_recognizable_variant() -> None:
    # the corrupted source must be a variant of the lead, not identical and not unrelated.
    stats = characterize_corruption((_TOLUENE, _ETHYLBENZENE, "O=C(Nc1ccccc1)c1ccncc1"), _CFG)
    tan = stats["source_target_tanimoto"]["mean"]
    assert tan is not None and 0.0 < tan < 1.0, stats
    assert stats["trim_yield"] > 0
