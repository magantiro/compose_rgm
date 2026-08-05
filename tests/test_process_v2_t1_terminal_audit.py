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
