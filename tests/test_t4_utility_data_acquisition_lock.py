import json
from pathlib import Path

from tools.t4_utility_data_acquisition_lock import canonical_hash, run

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "diagnostics/t4_utility_data_acquisition_lock/attempt_1"


def load(name):
    return json.loads((OUTPUT / name).read_text())


def test_contract_is_self_hashed_and_scoring_is_not_authorized():
    document = json.loads(
        (ROOT / "configs/t4_utility_data_acquisition_lock_v1.json").read_text()
    )
    assert canonical_hash(document["payload"]) == document["contract_sha256"]
    authorization = document["payload"]["authorization"]
    assert authorization == {
        "docking_status": "UNAUTHORIZED_NOT_LAUNCHED",
        "call_ceiling_if_later_exactly_authorized": 4,
        "automatic_retries": 0,
        "replacement_or_backfill": False,
        "required_later_authorization": (
            "A separate explicit authorization must name the final physical and "
            "payload SHA-256 hashes of both candidate_lock.json and request_lock.json "
            "and authorize exactly four scored calls."
        ),
    }


def test_locked_candidates_are_exact_eligible_and_independently_grouped():
    candidates = load("candidate_lock.json")["payload"]
    assert candidates["new_docking_calls"] == candidates["new_oracle_calls"] == 0
    assert len(candidates["candidates"]) == 4
    assert candidates["counts_per_cell"] == {"fa7_1": 2, "parp1_2": 2}
    expected = {
        "705e30bec99bf7ae6a51a7f862d6c4ff06bcb89f9f3964c78edbb5337b22af12": (
            22,
            "7556c312e3b076681c88482750993bca91e46ee9b9dc8bd8a1c61637a316bf68",
        ),
        "d7c7b5625111f0c41fa24873e5dc4f0a3048f3d6a5670dc93f44268f645e3345": (
            11,
            "b1b7182a33284ea0ee4bf0d70e30be7ea459d527a1aa8b3e74c5947b003ff136",
        ),
        "6ab68c2df5b637214cdb695ad1fa0b8302f5f17ad9fd38641f29f1355a4950fb": (
            1,
            "d5de86ecbeb01a5ea886ff54b486dfa8d4e0986ee756d31d08fac2af7318a2d4",
        ),
        "fd4d754151f7312e8c6d4df7f79e83f77eaae6262f48d3943856138ffd40036c": (
            1,
            "8c1fdbba554949afa493fda4d1e9f263f5639f4efc83e56e8c900e8c21ce72a4",
        ),
    }
    assert {row["candidate_id"] for row in candidates["candidates"]} == set(expected)
    for row in candidates["candidates"]:
        assert (row["action_count"], row["action_list_sha256"]) == expected[
            row["candidate_id"]
        ]
        assert row["strict_eligibility"]["oracle_eligible"] is True
        assert row["strict_eligibility"]["endpoint_exclusion_reasons"] == []
        assert row["docking_score"] is None
        assert row["scoring_status"] == "UNAUTHORIZED_NOT_LAUNCHED"
        assert row["lineage_ids"]


def test_unchanged_v3_gate_passes_only_conditionally_on_four_strict_scores():
    gate = load("gate_simulation.json")["payload"]
    assert (
        gate["evidence_status"]
        == "CONDITIONAL_STRUCTURAL_SIMULATION_NOT_MEASURED_UTILITY"
    )
    assert gate["hypothetical_scores_persisted"] is False
    assert gate["baseline_admitted_rows_before_grouping"] == 84
    assert gate["baseline_final_rows"] == 47
    assert gate["original_cross_fold_group_removals"] == 37
    assert gate["locked_rows_surviving_grouping"] == 4
    assert gate["post_exclusion_conflicting_groups"] == 0
    assert gate["supported_targets"] == ["fa7", "jak2", "parp1"]
    assert gate["supported_target_count"] == 3
    assert gate["supported_strata"] == 7
    assert gate["passes_if_condition_holds"] is True


def test_request_lock_is_four_calls_and_byte_deterministic(tmp_path):
    request = load("request_lock.json")["payload"]
    assert len(request["requests"]) == 4
    assert request["call_ceiling_if_later_exactly_authorized"] == 4
    assert request["automatic_retries"] == 0
    assert request["replacement_or_backfill"] is False
    assert request["scoring_status"] == "UNAUTHORIZED_NOT_LAUNCHED"
    assert request["training_data_acquisition_only"] is True
    assert request["prospective_evaluation_separate"] is True
    assert len({row["request_id"] for row in request["requests"]}) == 4
    assert all(row["docking_seed"] == 1701 for row in request["requests"])

    first, second = tmp_path / "first", tmp_path / "second"
    run(ROOT, first)
    run(ROOT, second)
    for path in sorted(first.iterdir()):
        assert path.read_bytes() == (second / path.name).read_bytes()
        assert path.read_bytes() == (OUTPUT / path.name).read_bytes()
