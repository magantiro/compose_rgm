"""Step 1b: the shardable global-grouping edit-trace miner's correctness invariants.

The load-bearing claim is that mining is a genuine map -> GLOBAL-group -> compile -> global-reduce, NOT
independent per-shard mining: two molecules sharing an MMP core must be paired even when they land in
DIFFERENT shards. We also pin the deterministic shard partition and that every compiled pool record is
re-consumable (replays through the executor to its recorded endpoint).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_analogue_trace_pool import (
    ANALOGUE_SUPPORT_CONTRACT,
    _compile_verify,
    _pool_record,
    rewrite_trace_from_record,
)
from mine_edit_traces import (
    MiningConfig,
    ShardEmission,
    characterize_corruption,
    compile_mmp,
    group_mmp_pairs,
    map_shard,
    shard_of,
)

from compose_v4.experiments.analogue_prior import (
    rewrite_trace_from_record as production_rewrite_trace_from_record,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import (
    decode_trace_record,
    encode_trace_record,
)

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


def test_terminal_atom_swap_prefers_real_atom_restate_supervision() -> None:
    result = _compile_verify(
        "Fc1ccccc1",
        "Clc1ccccc1",
        n_slots=16,
        max_variable_atoms=8,
    )
    assert result["outcome"] == "ok"
    assert result["path_length"] == 1
    assert result["operator_histogram"] == {"atom_restate": 1}
    assert result["meta"]["compiler_path_class"] == "direct_atom_restate"

    record = _pool_record(0, "forward", "Fc1ccccc1", result, 16)
    rebuilt = rewrite_trace_from_record(record)
    assert tuple(step.rule_name for step in rebuilt.steps) == ("atom_restate",)
    assert canonical_state_key(rebuilt.target) == record["target_key"]
    production_rebuilt = production_rewrite_trace_from_record(record)
    assert tuple(step.rule_name for step in production_rebuilt.steps) == (
        "atom_restate",
    )
    assert canonical_state_key(production_rebuilt.target) == record["target_key"]


@pytest.mark.parametrize(
    ("source_smiles", "target_smiles", "family"),
    (
        ("OCCc1ccccc1", "O=CCc1ccccc1", "bond_reorder"),
        ("CCCc1ccccc1", "CC(C)c1ccccc1", "bond_reroute"),
    ),
)
def test_unique_production_edit_precedes_delete_rebuild_fallback(
    source_smiles: str,
    target_smiles: str,
    family: str,
) -> None:
    result = _compile_verify(
        source_smiles,
        target_smiles,
        n_slots=20,
        max_variable_atoms=8,
    )
    assert result["outcome"] == "ok"
    assert result["path_length"] == 1
    assert result["operator_histogram"] == {family: 1}
    assert result["meta"]["compiler_path_class"] == f"direct_{family}"
    assert result["meta"]["direct_path_match_count"] == 1

    source, target = result["states"]
    assert np.array_equal(source.atom_types, target.atom_types)
    assert np.array_equal(source.formal_charges, target.formal_charges)

    record = _pool_record(0, "forward", source_smiles, result, 20)
    assert record["metadata"]["support_contract"] == ANALOGUE_SUPPORT_CONTRACT
    assert record["diagnostics"]["support_contract"] == ANALOGUE_SUPPORT_CONTRACT
    for loader in (rewrite_trace_from_record, production_rewrite_trace_from_record):
        rebuilt = loader(record)
        assert tuple(step.rule_name for step in rebuilt.steps) == (family,)
        assert canonical_state_key(rebuilt.target) == record["target_key"]

    trace = production_rewrite_trace_from_record(record)
    packed = encode_trace_record(
        trace,
        n_slots=20,
        seed=0,
        trace_id="support-contract-roundtrip",
        partition="train",
        layer="mmp_analogue",
        extra={**trace.metadata, "row_id": 0},
    )
    assert (
        decode_trace_record(packed).metadata["support_contract"]
        == ANALOGUE_SUPPORT_CONTRACT
    )

    wrong_contract = {
        **record,
        "metadata": {
            **record["metadata"],
            "support_contract": "unknown-contract",
        },
    }
    missing_contract = {
        **record,
        "metadata": {
            key: value
            for key, value in record["metadata"].items()
            if key != "support_contract"
        },
    }
    for invalid in (wrong_contract, missing_contract):
        for loader in (rewrite_trace_from_record, production_rewrite_trace_from_record):
            with pytest.raises(
                ValueError,
                match="not bound to the current editing-support contract",
            ):
                loader(invalid)


def test_same_family_aliases_do_not_force_delete_rebuild() -> None:
    result = _compile_verify(
        "CCOc1ccccc1",
        "CCc1ccccc1O",
        n_slots=20,
        max_variable_atoms=8,
    )
    assert result["outcome"] == "ok"
    assert result["path_length"] == 1
    assert result["operator_histogram"] == {"bond_reroute": 1}
    assert result["meta"]["compiler_path_class"] == "direct_bond_reroute"
    assert result["meta"]["direct_path_status"] == "direct_bond_reroute"
    assert result["meta"]["direct_path_match_count"] == 2


def test_charge_changing_endpoint_pair_is_rejected() -> None:
    result = _compile_verify(
        "NCc1ccccc1",
        "[NH3+]Cc1ccccc1",
        n_slots=20,
        max_variable_atoms=8,
    )
    assert result["outcome"] == "charge_change_unsupported"
    assert result["steps"] is None


def test_fallback_cannot_modify_a_preserved_charged_center() -> None:
    result = _compile_verify(
        "C[NH2+]Cc1ccccc1",
        "C[NH2+]CCc1ccccc1",
        n_slots=20,
        max_variable_atoms=8,
    )
    assert result["outcome"] == "charge_policy_violation"
    assert result["states"] is None


def test_multistep_size_change_remains_delete_insert_fallback() -> None:
    result = _compile_verify(
        _TOLUENE,
        _ETHYLBENZENE,
        n_slots=20,
        max_variable_atoms=8,
    )
    assert result["outcome"] == "ok"
    assert result["path_length"] == 3
    assert result["operator_histogram"] == {"atom_delete": 1, "atom_insert": 2}
    assert result["meta"]["compiler_path_class"] == "delete_insert_fallback"
    assert result["meta"]["direct_path_status"] == "direct_cardinality_mismatch"
    assert result["meta"]["direct_path_match_count"] == 0


def test_corruption_drift_tanimoto_is_a_recognizable_variant() -> None:
    # the corrupted source must be a variant of the lead, not identical and not unrelated.
    stats = characterize_corruption((_TOLUENE, _ETHYLBENZENE, "O=C(Nc1ccccc1)c1ccncc1"), _CFG)
    tan = stats["source_target_tanimoto"]["mean"]
    assert tan is not None and 0.0 < tan < 1.0, stats
    assert stats["trim_yield"] > 0
