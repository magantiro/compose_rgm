from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import modal_apps.run_editing_t1_uniform_successor_capacity as modal_runner
import scripts.run_editing_t1_uniform_successor_capacity as runner
from compose_v4.experiments.editing_t1_successor_runtime import EditingT1RuntimeError

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "editing_t1_uniform_successor_capacity_v1.json"


@dataclass(frozen=True)
class _Example:
    state: str
    time: float
    target_key: str
    importance_weight: float
    teacher_rate: float = 3.0
    data_lane: str = "source"


@dataclass(frozen=True)
class _Fiber:
    target_key: str
    aliases: tuple[str, ...]


def test_uniform_contract_is_self_hashed_and_non_authorizing() -> None:
    contract = runner.load_uniform_capacity_contract(CONTRACT)
    assert contract["training_authorized"] is False
    assert contract["bounded_p50_authorized"] is False
    assert contract["optimization_law"]["importance_coefficient"] == 1.0
    assert contract["panel_policy"]["teacher_mark_alias_policy"].startswith("aggregate")


def test_uniform_contract_rejects_post_freeze_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_bytes())
    payload["optimization_law"]["importance_coefficient"] = 0.5
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    payload["contract_sha256"] = runner._stable_sha256(body)
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EditingT1RuntimeError, match="identity"):
        runner.load_uniform_capacity_contract(mutated)


def test_json_normalization_matches_immutable_round_trip() -> None:
    payload = {
        "tuple_field": ("atom_insert", "cycle_attach"),
        "nested": {"values": (1, 2)},
    }
    normalized = runner._json_normalized(payload)
    assert normalized == {
        "nested": {"values": [1, 2]},
        "tuple_field": ["atom_insert", "cycle_attach"],
    }
    assert json.loads(runner._canonical_json_bytes(normalized)) == normalized


def test_uniformization_replaces_coefficients_after_unique_successor_checks(monkeypatch) -> None:
    examples = (
        _Example("state-a", 0.5, "target-a", 1e-12),
        _Example("state-b", 0.5, "target-b", 4.0),
    )
    fibers = (
        _Fiber("target-a", ("alias-a", "alias-b")),
        _Fiber("target-b", ("alias-c",)),
    )
    observed = {}

    def fake_prepare(model, rows, cached_fibers):
        observed["model"] = model
        observed["rows"] = tuple(rows)
        observed["fibers"] = tuple(cached_fibers)
        return "uniform-panel"

    monkeypatch.setattr(runner, "persistent_slot_state_sha256", lambda state: f"sha:{state}")
    monkeypatch.setattr(runner, "prepare_cached_successor_panel", fake_prepare)
    uniform, audit = runner.prepare_uniform_unique_successor_panel(
        "model",
        SimpleNamespace(examples=examples, fibers=fibers),
    )
    assert uniform == "uniform-panel"
    assert observed["fibers"] == fibers
    assert [row.importance_weight for row in observed["rows"]] == [1.0, 1.0]
    assert [row.teacher_rate for row in observed["rows"]] == [1.0, 1.0]
    assert audit["count"] == 2
    assert audit["minimum"] == 1e-12
    assert audit["maximum"] == 4.0
    assert 1.0 <= audit["coefficient_effective_sample_size"] < 2.0


def test_uniformization_rejects_repeated_prediction_problem(monkeypatch) -> None:
    examples = (
        _Example("same-state", 0.5, "target-a", 1.0),
        _Example("same-state", 0.5, "target-b", 1.0),
    )
    fibers = (
        _Fiber("target-a", ("alias-a",)),
        _Fiber("target-b", ("alias-b",)),
    )
    monkeypatch.setattr(runner, "persistent_slot_state_sha256", lambda state: f"sha:{state}")
    with pytest.raises(EditingT1RuntimeError, match="repeats an exact source/time"):
        runner.prepare_uniform_unique_successor_panel(
            "model",
            SimpleNamespace(examples=examples, fibers=fibers),
        )


def test_modal_selection_is_explicit_and_active8_bounded() -> None:
    assert modal_runner.REMOTE_PROJECT_ROOT == modal_runner.REMOTE_ROOT
    assert modal_runner._selected_families("", all_families=True) == modal_runner.FAMILIES
    assert modal_runner._selected_families(
        "cycle_attach,bond_reroute",
        all_families=False,
    ) == ("cycle_attach", "bond_reroute")
    with pytest.raises(ValueError, match="select --families"):
        modal_runner._selected_families("", all_families=False)
    with pytest.raises(ValueError, match="unknown Active8"):
        modal_runner._selected_families("ring_system_grow", all_families=False)


def test_uniform_thresholds_are_applied_without_weighted_rank_substitution() -> None:
    report = {
        "optimizer_steps_with_nonzero_gradient": 500,
        "component_gradient_update_counts": {"family_head": 500, "cycle_attach": 500},
        "required_components_without_gradient": [],
        "final": {
            "teacher_successor_top1_recall": 1.0,
            "teacher_successor_probability": 0.9,
            "canonical_successor_nll": 0.1,
            "per_example": [
                {
                    "teacher_successor_probability": 0.9,
                    "teacher_successor_nll": 0.10536051565782628,
                    "teacher_successor_rank": 1,
                },
                {
                    "teacher_successor_probability": 0.8,
                    "teacher_successor_nll": 0.2231435513142097,
                    "teacher_successor_rank": 1,
                },
            ],
        },
    }
    checks, floors = runner._capacity_checks(
        report,
        runner.EXPECTED_THRESHOLDS,
    )
    assert all(checks.values())
    assert floors == {
        "minimum_teacher_successor_probability": 0.8,
        "maximum_teacher_successor_nll": 0.2231435513142097,
        "maximum_teacher_successor_rank": 1,
        "top1_example_count": 2,
        "example_count": 2,
    }

    report["final"]["per_example"][1] = {
        "teacher_successor_probability": 0.79,
        "teacher_successor_nll": 0.23572233352106983,
        "teacher_successor_rank": 2,
    }
    report["required_components_without_gradient"] = ["cycle_attach"]
    checks, _ = runner._capacity_checks(
        report,
        runner.EXPECTED_THRESHOLDS,
    )
    assert checks["require_every_example_teacher_successor_top1"] is False
    assert checks["minimum_every_example_teacher_successor_probability"] is False
    assert checks["require_every_declared_component_nonzero_gradient"] is False


def test_partial_modal_receipt_is_explicitly_incomplete() -> None:
    receipt = modal_runner._build_receipt(
        run_label="uniform-test-v1",
        source_commit="a" * 40,
        active8_inventory="/artifacts/inventory.json",
        active8_inventory_file_sha256="b" * 64,
        requested_families=("atom_insert", "cycle_attach"),
        stage="GPU_RESULTS",
        cpu_preflight_bindings=(
            {"family": "atom_insert"},
            {"family": "cycle_attach"},
        ),
        results=(
            {
                "family": "atom_insert",
                "result_sha256": "c" * 64,
            },
        ),
        failures=(
            {
                "family": "cycle_attach",
                "error_type": "RuntimeError",
                "error_message": "failure",
            },
        ),
    )
    assert receipt["status"] == modal_runner.INCOMPLETE_RECEIPT_STATUS
    assert receipt["requested_families"] == ["atom_insert", "cycle_attach"]
    assert receipt["failures"][0]["family"] == "cycle_attach"


def test_partial_cpu_preflight_receipt_is_explicitly_incomplete() -> None:
    receipt = modal_runner._build_receipt(
        run_label="uniform-preflight-test-v1",
        source_commit="a" * 40,
        active8_inventory="/artifacts/inventory.json",
        active8_inventory_file_sha256="b" * 64,
        requested_families=("atom_insert", "cycle_attach"),
        stage="CPU_PREFLIGHT",
        cpu_preflight_bindings=({"family": "atom_insert"},),
        results=(),
        failures=(
            {
                "family": "cycle_attach",
                "error_type": "RuntimeError",
                "error_message": "cache validation failed",
            },
        ),
    )
    assert receipt["status"] == modal_runner.INCOMPLETE_RECEIPT_STATUS
    assert receipt["stage"] == "CPU_PREFLIGHT"
    assert receipt["results"] == []
    assert receipt["cpu_preflight_bindings"] == [{"family": "atom_insert"}]
    assert receipt["failures"][0]["family"] == "cycle_attach"


def test_aggregate_pass_cannot_mask_one_failed_example() -> None:
    checks, _ = runner._capacity_checks(
        {
            "optimizer_steps_with_nonzero_gradient": 500,
            "component_gradient_update_counts": {"family_head": 500, "atom_insert": 500},
            "required_components_without_gradient": [],
            "final": {
                "teacher_successor_top1_recall": 0.984375,
                "teacher_successor_probability": 0.98,
                "canonical_successor_nll": 0.05,
                "per_example": [
                    {
                        "teacher_successor_probability": 0.99,
                        "teacher_successor_nll": 0.01005033585350145,
                        "teacher_successor_rank": 1,
                    },
                    {
                        "teacher_successor_probability": 1e-6,
                        "teacher_successor_nll": 13.815510557964274,
                        "teacher_successor_rank": 12,
                    },
                ],
            },
        },
        runner.EXPECTED_THRESHOLDS,
    )
    assert checks["minimum_unique_state_teacher_successor_top1"] is True
    assert checks["minimum_unique_state_teacher_successor_probability"] is True
    assert checks["maximum_unique_state_teacher_successor_nll"] is True
    assert checks["require_every_example_teacher_successor_top1"] is False
    assert checks["minimum_every_example_teacher_successor_probability"] is False
