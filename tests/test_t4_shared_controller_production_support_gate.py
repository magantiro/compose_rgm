from pathlib import Path

from compose_v4.experiments.t4_shared_controller_production_support_gate import (
    BRAF_CANDIDATE_SHA256,
    run_gate,
    scientific_projection,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_shared_controller_production_support_gate_v1.json"


def test_zero_oracle_gate_uses_exact_runtime_admission_and_is_deterministic():
    first = run_gate(ROOT, CONTRACT)
    second = run_gate(ROOT, CONTRACT)

    assert scientific_projection(first) == scientific_projection(second)
    assert first["gate"] == {
        "braf_unseen_route_survives": True,
        "protonation_unseen_candidates_survive": True,
        "passed": True,
    }
    assert first["runtime_semantics"]["preflight_and_runtime_share_exact_helpers"]
    assert first["costs"] == {
        "docking_calls": 0,
        "oracle_calls": 0,
        "modal_launches": 0,
        "live_run_reads": 0,
    }


def test_gate_excludes_stale_braf_rows_but_keeps_unseen_route():
    result = run_gate(ROOT, CONTRACT)["cells"]["braf_0_d06"]

    assert result["required_candidate"] == {
        "canonical_smiles_sha256": BRAF_CANDIDATE_SHA256,
        "survived": True,
        "proposal_experts": ["route_complete_region"],
        "route_proposal_rank": 38,
        "realized_primitive_band": "large",
    }
    assert result["stale_candidates_excluded"] == 3
    assert result["candidate_union_unique"] == 1
    assert (
        result["runtime_admission"]["selection_ready_by_expert"][
            "route_complete_region"
        ]
        == 1
    )


def test_gate_keeps_all_sealed_eligible_protonation_candidates_unseen():
    result = run_gate(ROOT, CONTRACT)["cells"]["5ht1b_2_d06"]

    assert result["proposer_receipt_candidates"] == 828
    assert result["proposer_receipt_fully_eligible"] == 14
    assert result["candidate_union_unique"] == 14
    assert result["all_expected_eligible_survived"]
    assert result["all_survivors_unseen"]
    assert (
        result["runtime_admission"]["selection_ready_by_expert"][
            "protonation_aware_retained_subgraph"
        ]
        == 14
    )
