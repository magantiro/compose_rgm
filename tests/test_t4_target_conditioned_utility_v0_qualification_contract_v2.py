from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/t4_target_conditioned_utility_v0_qualification_v2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _payload_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return _sha256_bytes(encoded)


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _git_text(*args: str) -> str:
    return _git(*args).decode("utf-8").strip()


def _assert_bound_git_blob(binding: dict, revision: str) -> None:
    object_spec = f"{revision}:{binding['path']}"
    assert _git_text("rev-parse", object_spec) == binding["git_blob_sha1"]
    assert _sha256_bytes(_git("show", object_spec)) == binding["sha256"]


def _without_git_objects(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_git_objects(item)
            for key, item in value.items()
            if key not in {"git_blob_sha1", "git_tree"}
        }
    if isinstance(value, list):
        return [_without_git_objects(item) for item in value]
    return value


def test_v2_self_hash_and_v1_lineage_are_immutable() -> None:
    contract = _load(CONTRACT_PATH)
    payload = contract["payload"]
    assert contract["contract_sha256"] == _payload_sha256(payload)

    predecessor = payload["predecessor_v1"]
    revision = predecessor["git_revision"]
    assert revision == "b372f6c67873a4b656066df2361b0d99751be392"
    assert _git_text("rev-parse", f"{revision}^{{tree}}") == predecessor["git_tree"]
    assert predecessor["preservation"].startswith("immutable")

    for key in ("contract", "documentation", "focused_test"):
        binding = predecessor[key]
        _assert_bound_git_blob(binding, revision)
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]

    v1 = _load(ROOT / predecessor["contract"]["path"])
    assert v1["contract_sha256"] == predecessor["contract"]["payload_sha256"]
    assert v1["contract_sha256"] == _payload_sha256(v1["payload"])


def test_v2_binds_the_same_integration_selector_and_historical_curves() -> None:
    payload = _load(CONTRACT_PATH)["payload"]
    frozen = payload["frozen_inputs"]

    integration = frozen["macro_archive_v0_integration_revision"]
    integration_revision = integration["git_revision"]
    assert integration_revision == "2a41b086fb4f109a09bee15c17f1c6ec6551e29d"
    assert _git_text("rev-parse", f"{integration_revision}^{{tree}}") == integration["git_tree"]
    for key in (
        "integration_contract",
        "selector_binding",
        "integration_implementation",
        "dynamic_v0_implementation",
    ):
        binding = integration[key]
        _assert_bound_git_blob(binding, integration_revision)
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]

    selector = frozen["target_conditioned_utility_selector_design"]
    selector_revision = selector["git_revision"]
    assert _git_text("rev-parse", f"{selector_revision}^{{tree}}") == selector["git_tree"]
    _assert_bound_git_blob(selector, selector_revision)

    historical = frozen["historical_matched_call_reference"]
    historical_revision = historical["git_revision"]
    assert _git_text("rev-parse", f"{historical_revision}^{{tree}}") == historical["git_tree"]
    _assert_bound_git_blob(historical, historical_revision)
    assert _sha256(ROOT / historical["path"]) == historical["sha256"]
    source = _load(ROOT / historical["path"])
    assert source["payload_sha256"] == historical["payload_sha256"]

    v1 = _load(ROOT / payload["predecessor_v1"]["contract"]["path"])["payload"]["frozen_inputs"]
    for key in (
        "macro_archive_v0_integration_revision",
        "target_conditioned_utility_selector_design",
        "historical_matched_call_reference",
    ):
        v2_without_git_objects = _without_git_objects(frozen[key])
        assert v2_without_git_objects == _without_git_objects(v1[key])


def test_matched_curves_and_missing_reference_abstentions_are_exact() -> None:
    payload = _load(CONTRACT_PATH)["payload"]
    checkpoints = payload["prospective_design"]["matched_summary_calls"]
    assert checkpoints == [1, 5, 10, 20, 50, 100]
    assert payload["cells"] == [
        "5ht1b_0",
        "braf_1",
        "jak2_1",
        "parp1_0",
        "fa7_0",
    ]

    source = _load(ROOT / payload["frozen_inputs"]["historical_matched_call_reference"]["path"])[
        "payload"
    ]
    contract_curves = payload["historical_references"]["by_cell"]
    for cell in ("5ht1b_0", "braf_1", "jak2_1"):
        unit_id = f"{cell}_r0"
        for arm in ("dynamic_v0", "full_146"):
            source_points = {
                row["query"]: row["best_score"]
                for row in source[arm][unit_id]["curve"]
                if row["query"] in checkpoints
            }
            bound_points = {
                int(call): score
                for call, score in contract_curves[cell][arm]["best_score_by_call"].items()
            }
            assert source_points == bound_points

    for cell in ("parp1_0", "fa7_0"):
        for arm in ("dynamic_v0", "full_146"):
            assert contract_curves[cell][arm]["status"] == ("abstain_missing_historical_curve")


def test_v2_replaces_call1_rejection_with_predeclared_trajectory_gates() -> None:
    payload = _load(CONTRACT_PATH)["payload"]
    gates = payload["promotion_gates"]
    paired = gates["utility_jump_beats_structural_selection"]
    assert paired["denominator_cells"] == 5
    assert paired["maximum_cell_balanced_mean_difference_exclusive"] == 0.0
    assert paired["minimum_strict_paired_wins"] == 3
    assert "dynamic_v0_call_1" not in paired

    call1 = gates["call_1_dynamic_v0_diagnostic"]
    assert call1["mandatory"] is True
    assert call1["hard_per_cell_promotion_gate"] is False

    trajectory = gates["cell_balanced_early_trajectory"]
    assert trajectory["checkpoint_calls"] == [1, 5, 10, 20]
    assert trajectory["reference_cells"] == ["5ht1b_0", "braf_1", "jak2_1"]
    assert trajectory["maximum_cell_balanced_mean_regret_inclusive"] == 0.0
    assert "4*(d_c(1)+d_c(5))/2" in trajectory["per_cell_normalized_signed_cumulative_regret"]
    assert "/ 19" in trajectory["per_cell_normalized_signed_cumulative_regret"]
    assert trajectory["complete_reference_panel_required"] is True
    assert trajectory["tolerance"].startswith("none")

    catch_up = gates["catch_up_by_call_20"]
    assert catch_up["call"] == 20
    assert catch_up["reference_cells"] == trajectory["reference_cells"]
    assert catch_up["maximum_per_cell_difference_inclusive"] == 0.0
    assert catch_up["tolerance"].startswith("none")

    later = gates["later_hard_case_full_route_gap"]
    assert later["reference_hard_case"] == "jak2_1"
    assert later["matched_calls"] == [50, 100]
    assert later["dynamic_v0_reference_gaps"] == {"50": 1.4, "100": 1.2}
    assert later["fa7_status"].startswith("abstain_missing_historical")
    assert later["tolerance"].startswith("none")


def test_v2_is_zero_oracle_fixed_budget_and_has_no_plateau_stop() -> None:
    payload = _load(CONTRACT_PATH)["payload"]
    design = payload["prospective_design"]
    ceiling = design["qualification_ceiling_if_separately_authorized"]
    assert ceiling == {
        "structural_jump_calls_per_cell": 1,
        "utility_plus_v0_total_calls_per_cell": 100,
        "maximum_new_charged_calls": 505,
        "maximum_concurrent_single_cpu_workers": 5,
        "gpu": False,
        "automatic_retry": False,
        "confirmation_calls": 0,
        "plateau_stopping": False,
    }
    assert payload["frozen_protocol"]["plateau_stopping"] is False
    assert "No extension, plateau stop or later checkpoint" in design["continuation_rule"]
    assert payload["authorized_actions"] == {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "live_run_reads": 0,
        "frozen_artifact_mutations": 0,
    }
