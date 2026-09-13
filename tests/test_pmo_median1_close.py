from __future__ import annotations

from pathlib import Path

from compose_v4.control.edit_program import EditProgram, execute_bound_program
from compose_v4.experiments.pmo_ivg_oracle_parity import load_contract as load_parity
from compose_v4.experiments.pmo_ivg_oracle_parity import score_values
from compose_v4.experiments.pmo_median1_close import build_query_lock, load_contract
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def test_contract_preserves_one_query_and_claim_boundary() -> None:
    contract = load_contract(ROOT)
    assert contract["oracle"]["new_query_ceiling"] == 1
    assert (
        contract["candidate"]["source_role"] == "pre_score_panel_series_extrapolation"
    )
    assert contract["information_regime"]["held_out_claim"] is False
    assert contract["information_regime"]["t4_affected"] is False


def test_query_lock_contains_one_exact_replay_program() -> None:
    contract = load_contract(ROOT)
    lock = build_query_lock(ROOT, contract)
    source = decode_state(lock["source"]["state"])
    program = EditProgram.from_payload(lock["program"])
    _, receipt = execute_bound_program(
        source,
        program,
        tuple(lock["assignment"]),
        max_primitives=contract["support"]["max_primitives_per_complete_program"],
        max_blocks=contract["support"]["max_blocks_per_complete_program"],
    )
    assert receipt == lock["receipt"]
    assert receipt["endpoint"] == contract["candidate"]["smiles"]
    assert receipt["primitive_edits"] == 14


def test_expected_series_score_would_close_prescreen_gap() -> None:
    contract = load_contract(ROOT)
    prior = __import__("json").loads(
        (ROOT / contract["prior_result"]["path"]).read_text()
    )["payload"]
    values = [row["parity_score"] for row in prior["results"]["median1"]["rows"]]
    metrics = score_values(values + [0.39488251846572103], load_parity(ROOT))
    assert (
        metrics["auc_top10_official_10k"]
        > contract["baseline_and_ablation"]["ivg_prescreen_auc_top10"]
    )
