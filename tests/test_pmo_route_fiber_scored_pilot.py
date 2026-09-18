import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.control.fiber_control import SearchState
from compose_v4.experiments.pmo_route_fiber_scored_pilot import (
    LOCK_MANIFEST,
    _recursive_keys,
    _selection_seed,
    pmo_program_features,
    prepare_runtime_bundles,
    run_locked_unit,
    unseal,
)
from compose_v4.experiments.pmo_route_fiber_scored_pilot_contract import (
    CONTRACT,
    TASKS,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_reward(smiles: str) -> float:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return min(1.0, molecule.GetNumHeavyAtoms() / 48)


def test_runtime_bundles_are_fold_specific_and_teacher_free(tmp_path, monkeypatch):
    import compose_v4.experiments.pmo_route_fiber_scored_pilot as pilot

    monkeypatch.setattr(pilot, "RUNTIME_DIRECTORY", str(tmp_path))
    bundles = prepare_runtime_bundles(ROOT)
    assert set(bundles) == {0, 2}
    forbidden = {
        "task",
        "task_family",
        "route_id",
        "lineage_id",
        "source_graph",
        "endpoint",
        "teacher_action",
    }
    for fold, envelope in bundles.items():
        assert envelope["payload"]["fold_index"] == fold
        assert envelope["payload"]["new_oracle_calls"] == 0
        assert not (forbidden & _recursive_keys(envelope["payload"]))


def test_feature_adapter_is_fixed_size_and_task_free():
    candidate = {
        "selection_features": {
            "route_lane_indicator": 1,
            "primitive_count": 8,
            "dependency_region_count": 2,
            "created_atom_count": 2,
            "deleted_atom_count": 1,
            "created_dependency_count": 1,
            "cycle_dependency_count": 0,
            "generic_rule_histogram": {
                "atom_delete": 1,
                "atom_insert": 2,
                "atom_restate_semantic": 0,
                "bond_reorder": 1,
                "bond_reroute": 1,
                "cycle_close": 1,
                "cycle_open": 1,
                "ring_system_restate": 1,
            },
        }
    }
    features = pmo_program_features(
        candidate,
        SearchState(archive={"parent": -0.8}),
        parent_reward=0.7,
    )
    assert features.shape == (18,)
    assert np.isfinite(features).all()
    assert features[-1] == 1


def test_round_one_selection_seed_is_shared_inside_each_proposal_pair():
    contract_id = json.loads((ROOT / CONTRACT).read_text())["payload_sha256"]
    for fold in (0, 2):
        for source in ("old_v0", "additive_route"):
            assert _selection_seed(
                contract_id, fold, source, 1, "blind"
            ) == _selection_seed(contract_id, fold, source, 1, "fiber_control")
            assert _selection_seed(
                contract_id, fold, source, 2, "blind"
            ) != _selection_seed(contract_id, fold, source, 2, "fiber_control")


def test_real_protocol_cannot_run_without_sealed_launch_receipt(tmp_path):
    with pytest.raises(PermissionError, match="launch receipt"):
        run_locked_unit(
            ROOT,
            tmp_path / "run",
            task="gsk3b",
            arm="old_v0_blind",
            oracle_protocol="native-pmo:gsk3b",
            evaluate=_synthetic_reward,
        )


def test_candidate_manifest_is_complete_score_blind_and_exact():
    manifest = unseal(ROOT / LOCK_MANIFEST)
    assert manifest["pool_count"] == 16
    assert manifest["all_locked_before_scoring"] is True
    assert manifest["new_oracle_calls"] == 0
    for row in manifest["pools"]:
        pool = unseal(ROOT / row["path"])
        assert pool["candidate_count"] >= 8
        assert pool["score_fields_present"] is False
        assert pool["new_oracle_calls"] == 0
        assert all(candidate["exact_replay"] for candidate in pool["candidates"])
        assert not {
            "reward",
            "control_score",
            "winner_identity",
            "comparator_outcome",
        } & _recursive_keys(pool)


def test_synthetic_interruption_resume_matches_uninterrupted(tmp_path):
    task = "perindopril_mpo"
    arm = "additive_route_fiber_control"
    resumed = tmp_path / "resumed"
    resumed_locks = tmp_path / "resumed_locks"
    partial = run_locked_unit(
        ROOT,
        resumed,
        task=task,
        arm=arm,
        oracle_protocol="synthetic:heavy-atom-v1",
        evaluate=_synthetic_reward,
        stop_after_round=2,
        query_lock_root=resumed_locks,
    )
    assert partial["charged_calls"] == 32
    completed = run_locked_unit(
        ROOT,
        resumed,
        task=task,
        arm=arm,
        oracle_protocol="synthetic:heavy-atom-v1",
        evaluate=_synthetic_reward,
        query_lock_root=resumed_locks,
    )
    uninterrupted = run_locked_unit(
        ROOT,
        tmp_path / "uninterrupted",
        task=task,
        arm=arm,
        oracle_protocol="synthetic:heavy-atom-v1",
        evaluate=_synthetic_reward,
        query_lock_root=tmp_path / "uninterrupted_locks",
    )
    assert completed["charged_calls"] == 48
    assert (
        completed["selected_candidate_ids"] == uninterrupted["selected_candidate_ids"]
    )
    assert completed["score_curve"] == uninterrupted["score_curve"]
    assert completed["auc_top10_48"] == uninterrupted["auc_top10_48"]
    assert set(TASKS) == {"gsk3b", "perindopril_mpo"}
