from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.control.target_conditioned_utility_selector import (
    MeasuredEndpoint,
    UtilityRanker,
    evaluate_scores,
    fit_utility_ranker,
    strict_pairs,
)
from tools.t4_target_conditioned_utility_selector import (
    CONTRACT,
    _leakage_filter,
    _state_hash,
    canonical_json_hash,
    run,
)

ROOT = Path(__file__).resolve().parents[1]


def _row(identity: int, *, target: str = "jak2", score: float) -> MeasuredEndpoint:
    source_idx = identity % 3
    return MeasuredEndpoint(
        row_id=f"row-{identity}",
        artifact="fixture",
        target=target,
        cell=f"{target}_{source_idx}",
        source_idx=source_idx,
        delta=0.6,
        oracle_protocol="protocol",
        docking_seed=1701,
        docking_score=score,
        source_smiles="CC",
        endpoint_smiles="CCC" if identity % 2 else "CCO",
        source_state_sha256=f"{identity:064x}",
        endpoint_state_sha256=f"{identity + 100:064x}",
        scaffold="acyclic",
        lineage_ids=(f"lineage-{identity}",),
        provenance={"fixture": True},
    )


def test_address_free_state_identity_ignores_persistent_slot_permutation() -> None:
    first = {
        "n_slots": 2,
        "atom_types": [2, 0],
        "formal_charges": [0, 0],
        "implicit_h_counts": [4, 0],
        "bonds": [],
    }
    second = {
        "n_slots": 2,
        "atom_types": [0, 2],
        "formal_charges": [0, 0],
        "implicit_h_counts": [0, 4],
        "bonds": [],
    }
    assert _state_hash(first) == _state_hash(second)


def test_pairs_never_cross_comparability_strata() -> None:
    rows = [
        _row(0, score=-8.0),
        _row(3, score=-9.0),
        _row(1, score=-12.0),
    ]
    assert strict_pairs(rows) == [("row-3", "row-0")]


def test_connected_global_graph_groups_are_removed_across_folds() -> None:
    first = _row(0, score=-8.0)
    second = replace(_row(1, target="braf", score=-9.0), endpoint_smiles="CCO")
    connected = replace(_row(3, score=-10.0), lineage_ids=first.lineage_ids)
    kept, exclusions, audit = _leakage_filter([first, second, connected], [])
    assert kept == []
    assert len(exclusions) == 3
    assert audit["conflicting_connected_components"] == 1
    assert audit["post_exclusion_conflicting_group_count"] == 0


def test_shared_ranker_checkpoint_contains_no_graph_or_score_rows() -> None:
    rows = [
        _row(0, score=-8.0),
        _row(3, score=-9.0),
        _row(1, target="parp1", score=-7.0),
        _row(4, target="parp1", score=-10.0),
    ]
    ranker, audit = fit_utility_ranker(
        rows,
        conditioned=True,
        updates=5,
        learning_rate=0.03,
        base_l2=0.02,
        target_interaction_l2=0.1,
        seed=17,
    )
    checkpoint = ranker.checkpoint()
    restored = UtilityRanker.from_checkpoint(checkpoint)
    assert restored.conditioned
    assert audit["strict_pairs"] == 2
    serialized = json.dumps(checkpoint, sort_keys=True)
    assert "smiles" not in serialized
    assert "docking_score" not in serialized
    scores = {row.row_id: restored.score(row) for row in rows}
    assert evaluate_scores(rows, scores)["strict_pairs"] == 2


def test_stale_contract_hash_is_rejected_before_extraction(tmp_path: Path) -> None:
    document = json.loads((ROOT / CONTRACT).read_text())
    assert canonical_json_hash(document["payload"]) == document["contract_sha256"]
    document["payload"]["mode"] = "tampered"
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        run(ROOT, tmp_path / "result", stale)


def test_repaired_coverage_gate_abstains_without_fitting(tmp_path: Path) -> None:
    result = run(ROOT, tmp_path / "result", CONTRACT)
    assert result["status"] == "abstain_data_gate_failed"
    assert result["fit_performed"] is False
    audit = json.loads((tmp_path / "result" / "data_audit.json").read_text())
    assert audit["pre_leakage_census"]["artifacts"]["second_generation"] == 49
    assert audit["pre_leakage_census"]["artifacts"]["program_pool"] == 14
    assert audit["input_reconciliation"]["totals"] == {
        "observed_physical_units": 99,
        "finite_measured_labels": 97,
        "failed_or_null_physical_units": 2,
        "invalid_or_out_of_cell_physical_labels": 13,
        "admitted_labels_before_group_filter": 84,
        "cross_fold_group_removals": 37,
        "final_labels": 47,
    }
    assert audit["gate_checks"]["at_least_three_supported_targets"] is False
    assert audit["gate_checks"]["at_least_five_supported_strata"] is False
    assert audit["support"]["supported_targets"] == ["jak2"]
    assert not list((tmp_path / "result").glob("fold_*_checkpoint.json"))
