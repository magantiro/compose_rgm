from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/t4_target_conditioned_utility_v0_qualification_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_contract_self_hash_and_immutable_bindings() -> None:
    contract = _load(CONTRACT_PATH)
    payload = contract["payload"]
    assert contract["contract_sha256"] == _payload_sha256(payload)

    frozen = payload["frozen_inputs"]
    integration = frozen["macro_archive_v0_integration_revision"]
    assert integration["git_revision"] == "2a41b086fb4f109a09bee15c17f1c6ec6551e29d"
    assert integration["git_tree"] == "5e24f7a1a114796f8c0938623d30b3f5c046e68c"

    for key in (
        "integration_contract",
        "selector_binding",
        "integration_implementation",
        "dynamic_v0_implementation",
    ):
        bound = integration[key]
        assert _sha256(ROOT / bound["path"]) == bound["sha256"]

    target_selector = frozen["target_conditioned_utility_selector_design"]
    assert target_selector == {
        "git_revision": "fe9545eae766f3cd8558af1aaa5576f1c11a6be3",
        "path": "configs/t4_target_conditioned_utility_selector_v1.json",
        "sha256": "f031fed9e6bd43bd7b6188b106986a6ee863f5ff259d14fe409c2e9348e022b2",
        "payload_sha256": "266a8e02b34155b946bfa71cb059272443356d4dc81eb556d47e6e9bd683c743",
        "runtime_status": "design_bound_but_authoritative_result_and_fold_checkpoint_hashes_pending",
        "required_split": "use the frozen fold whose held target-local source index matches the development cell; no row from that held source may fit features, preprocessing, pairs or coefficients",
    }

    reference = frozen["historical_matched_call_reference"]
    reference_path = ROOT / reference["path"]
    reference_payload = _load(reference_path)
    assert _sha256(reference_path) == reference["sha256"]
    assert reference_payload["payload_sha256"] == reference["payload_sha256"]


def test_reference_checkpoints_abstentions_and_promotion_rules_are_frozen() -> None:
    payload = _load(CONTRACT_PATH)["payload"]
    reference = _load(
        ROOT / payload["frozen_inputs"]["historical_matched_call_reference"]["path"]
    )["payload"]
    checkpoints = payload["prospective_design"]["report_at_charged_calls"]
    assert checkpoints == [1, 5, 10, 20, 50, 100]
    assert payload["cells"] == ["5ht1b_0", "braf_1", "jak2_1", "parp1_0", "fa7_0"]
    assert "only for the frozen five-level target-conditioning input" in payload[
        "contract_precedence"
    ]
    assert "Dynamic-v0 refinement" in payload["contract_precedence"]

    contract_curves = payload["historical_references"]["by_cell"]
    for cell in ("5ht1b_0", "braf_1", "jak2_1"):
        unit_id = f"{cell}_r0"
        for arm in ("dynamic_v0", "full_146"):
            source_points = {
                row["query"]: row["best_score"]
                for row in reference[arm][unit_id]["curve"]
                if row["query"] in checkpoints
            }
            bound_points = {
                int(call): score
                for call, score in contract_curves[cell][arm][
                    "best_score_by_call"
                ].items()
            }
            assert source_points == bound_points

    for cell in ("parp1_0", "fa7_0"):
        assert contract_curves[cell]["dynamic_v0"]["status"] == (
            "abstain_missing_historical_curve"
        )
        assert contract_curves[cell]["full_146"]["status"] == (
            "abstain_missing_historical_curve"
        )

    design = payload["prospective_design"]
    ceiling = design["initial_qualification_ceiling_if_separately_authorized"]
    assert ceiling["maximum_new_charged_calls"] == 505
    assert ceiling["plateau_stopping"] is False
    assert ceiling["later_calls_authorized"] is False
    assert payload["authorized_actions"] == {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "live_run_reads": 0,
        "frozen_artifact_mutations": 0,
    }

    gates = payload["promotion_gates"]
    assert "Only direct measured docking utility" in gates["authoritative_endpoint"]
    assert gates["integrated_controller_preserves_early_v0"]["calls"] == [1, 5, 10, 20]
    hard_gate = gates["integrated_controller_reduces_hard_case_full_route_gap"]
    assert hard_gate["reference_hard_case"] == "jak2_1"
    assert hard_gate["matched_calls"] == [50, 100]
    assert hard_gate["fa7_status"].startswith("abstain_missing_historical")
