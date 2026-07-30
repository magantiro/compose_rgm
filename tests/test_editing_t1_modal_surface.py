from __future__ import annotations

import pytest

pytest.importorskip("modal")

from modal_apps.run_editing_t1_successor_gate import (  # noqa: E402
    FAMILIES,
    SCOPES,
    build_task_matrix,
)


def test_t1_modal_surface_requires_explicit_family_and_defaults_to_one_arm():
    with pytest.raises(ValueError, match="families must be selected explicitly"):
        build_task_matrix("editing-t1-test-v1", families="")

    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="cycle_attach",
    )

    assert tasks == (
        (
            "editing-t1-test-v1",
            "cycle_attach",
            "unique_state",
            "heads_only",
        ),
    )


def test_t1_modal_surface_requires_explicit_all_scopes_opt_in():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="cycle_attach",
        all_scopes=True,
    )

    assert tuple(task[-1] for task in tasks) == SCOPES


def test_t1_modal_surface_all_families_does_not_imply_all_scopes():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="",
        all_families=True,
    )

    assert len(tasks) == len(FAMILIES)
    assert {task[-1] for task in tasks} == {"heads_only"}


def test_t1_modal_surface_rejects_unavailable_or_ambiguous_panels():
    with pytest.raises(
        ValueError,
        match="no empirical within-family repeated multi-successor panel",
    ):
        build_task_matrix(
            "editing-t1-test-v1",
            families="atom_delete",
            panel_kinds="within_family_repeated_state_distribution",
        )
    with pytest.raises(ValueError, match="requires exactly"):
        build_task_matrix(
            "editing-t1-test-v1",
            families="cycle_attach",
            panel_kinds="global_repeated_state_distribution",
        )


def test_t1_modal_surface_repeated_panel_is_explicit_and_bounded():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="cycle_attach,bond_reroute",
        panel_kinds="within_family_repeated_state_distribution",
        scopes="heads_only,all",
    )

    assert len(tasks) == 4
    assert {task[1] for task in tasks} == {"cycle_attach", "bond_reroute"}
    assert {task[2] for task in tasks} == {"within_family_repeated_state_distribution"}
    assert {task[3] for task in tasks} == {"heads_only", "all"}


def test_t1_modal_surface_global_repeated_law_is_one_explicit_mixed_family_arm():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="all_families",
        panel_kinds="global_repeated_state_distribution",
    )

    assert tasks == (
        (
            "editing-t1-test-v1",
            "all_families",
            "global_repeated_state_distribution",
            "heads_only",
        ),
    )

    with pytest.raises(ValueError, match="requires exactly"):
        build_task_matrix(
            "editing-t1-test-v1",
            families="all_families,cycle_attach",
            panel_kinds="global_repeated_state_distribution",
        )
