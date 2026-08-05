from __future__ import annotations

import pytest

from modal_apps import run_process_v2_t1_app as app


def _row(identifier: str, *, family: str = "atom_insert") -> dict[str, object]:
    return {
        "panel_entry_sha256": identifier,
        "model_family": family,
        "capability_cell_id": f"cell:{family}",
        "teacher_successor_probability": 0.75,
        "teacher_successor_nll": 0.2876820724517809,
        "teacher_successor_rank": 1,
        "teacher_successor_top1": True,
        "canonical_successor_count": 4,
    }


def _entry(identifier: str) -> dict[str, object]:
    return {
        "panel_entry_sha256": identifier,
        "raw_mark_count": 9,
        "productive_alias_count": 8,
        "virtual_alias_count": 1,
        "production_successor_alias_multiplicity": 2,
    }


def test_terminal_audit_rows_attach_prepared_census() -> None:
    rows = app._terminal_audit_rows([_row("a" * 64)], entries=[_entry("a" * 64)])

    assert rows[0]["raw_mark_count"] == 9
    assert rows[0]["productive_alias_count"] == 8
    assert rows[0]["virtual_alias_count"] == 1
    assert rows[0]["teacher_alias_multiplicity"] == 2


def test_terminal_audit_rows_refuse_another_panel() -> None:
    with pytest.raises(RuntimeError, match="another prepared panel"):
        app._terminal_audit_rows([_row("a" * 64)], entries=[_entry("b" * 64)])


def test_metric_projection_uses_frozen_family_order() -> None:
    rows = [
        _row("b" * 64, family="atom_delete"),
        _row("a" * 64, family="atom_insert"),
    ]

    projected = app._metric_result_projection(
        rows,
        family_order=("atom_insert", "atom_delete"),
    )

    assert [row["family"] for row in projected] == ["atom_insert", "atom_delete"]
    assert set(projected[0]) == {
        "panel_entry_sha256",
        "family",
        "semantic_cell_id",
        "teacher_successor_probability",
        "canonical_successor_nll",
        "teacher_successor_rank",
        "teacher_successor_top1",
    }


def _environment(device_name: str, *, torch_version: str = "2.4.0") -> dict[str, object]:
    return {
        "device_name": device_name,
        "device_capability": "8.6",
        "torch_version": torch_version,
        "environment_sha256": "a" * 64,
    }


def test_terminal_audit_accepts_only_the_observed_a10_name_alias() -> None:
    expected = _environment("NVIDIA A10")
    observed = {
        **_environment("NVIDIA A10G"),
        "environment_sha256": "b" * 64,
    }

    assert (
        app._terminal_audit_environment_disposition(observed, expected)
        == "MODAL_A10_DEVICE_NAME_ALIAS_ONLY"
    )


def test_terminal_audit_refuses_substantive_environment_drift() -> None:
    expected = _environment("NVIDIA A10")
    observed = {
        **_environment("NVIDIA A10G", torch_version="2.5.0"),
        "environment_sha256": "b" * 64,
    }

    with pytest.raises(RuntimeError, match="torch_version"):
        app._terminal_audit_environment_disposition(observed, expected)


def _family_audit_result(
    family: str,
    *,
    observed_device_name: str,
    disposition: str,
) -> dict[str, object]:
    expected = _environment("NVIDIA A10")
    observed = {
        **_environment(observed_device_name),
        "environment_sha256": (
            expected["environment_sha256"]
            if observed_device_name == "NVIDIA A10"
            else "b" * 64
        ),
    }
    return {
        "family": family,
        "selected_step": 250,
        "terminal_step": 500,
        "selected_model_state_sha256": "c" * 64,
        "terminal_model_state_sha256": "d" * 64,
        "gradient_evidence": {"checkpoint_wide": {"finite": True}},
        "checkpoint_execution_environment": expected,
        "audit_execution_environment": observed,
        "execution_environment_disposition": disposition,
    }


def test_terminal_audit_retains_mixed_exact_and_alias_worker_receipts() -> None:
    exact = _family_audit_result(
        "atom_insert",
        observed_device_name="NVIDIA A10",
        disposition="EXACT",
    )
    alias = _family_audit_result(
        "atom_delete",
        observed_device_name="NVIDIA A10G",
        disposition="MODAL_A10_DEVICE_NAME_ALIAS_ONLY",
    )

    assert app._terminal_audit_checkpoint_projection(exact) == (
        app._terminal_audit_checkpoint_projection(alias)
    )
    receipts = app._terminal_audit_worker_execution_receipts(
        {"atom_insert": exact, "atom_delete": alias},
        family_order=("atom_insert", "atom_delete"),
    )

    assert [receipt["execution_environment_disposition"] for receipt in receipts] == [
        "EXACT",
        "MODAL_A10_DEVICE_NAME_ALIAS_ONLY",
    ]


def test_terminal_audit_refuses_a_forged_worker_disposition() -> None:
    result = _family_audit_result(
        "atom_insert",
        observed_device_name="NVIDIA A10G",
        disposition="EXACT",
    )

    with pytest.raises(RuntimeError, match="disposition disagrees"):
        app._terminal_audit_worker_execution_receipts(
            {"atom_insert": result},
            family_order=("atom_insert",),
        )
