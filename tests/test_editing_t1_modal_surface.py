from __future__ import annotations

import pytest

pytest.importorskip("modal")

from modal_apps.run_editing_t1_successor_gate import (
    FAMILIES,
    LOCAL_ADAPTER_FAMILIES,
    SCOPES,
)
from modal_apps.run_editing_t1_successor_gate import (
    build_task_matrix as _build_task_matrix,
)

ACTIVE8_INVENTORY = "active8_trace_inventory_v1/ACTIVE8_TRACE_INVENTORY.json"
ACTIVE8_FILE_SHA256 = "a" * 64
RESOLVED_ACTIVE8_INVENTORY = (
    "/artifacts/active8_trace_inventory_v1/ACTIVE8_TRACE_INVENTORY.json"
)


def build_task_matrix(*args, **kwargs):
    kwargs.setdefault("active8_inventory", ACTIVE8_INVENTORY)
    kwargs.setdefault(
        "active8_inventory_file_sha256",
        ACTIVE8_FILE_SHA256,
    )
    return _build_task_matrix(*args, **kwargs)


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
            RESOLVED_ACTIVE8_INVENTORY,
            ACTIVE8_FILE_SHA256,
        ),
    )


def test_t1_modal_surface_requires_explicit_all_scopes_opt_in():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="cycle_attach",
        all_scopes=True,
    )

    assert tuple(task[3] for task in tasks) == SCOPES


def test_t1_modal_surface_skips_nonexistent_local_adapter_arms():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="",
        all_families=True,
        all_scopes=True,
    )
    assert len(tasks) == 2 * len(FAMILIES) + len(LOCAL_ADAPTER_FAMILIES)
    for task in tasks:
        if task[1] not in LOCAL_ADAPTER_FAMILIES:
            assert task[3] != "heads_plus_local_adapter"

    with pytest.raises(ValueError, match="changes no parameters"):
        build_task_matrix(
            "editing-t1-test-v1",
            families="atom_insert",
            scopes="heads_plus_local_adapter",
        )


def test_t1_modal_surface_all_families_does_not_imply_all_scopes():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="",
        all_families=True,
    )

    assert len(tasks) == len(FAMILIES)
    assert {task[3] for task in tasks} == {"heads_only"}


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
            RESOLVED_ACTIVE8_INVENTORY,
            ACTIVE8_FILE_SHA256,
        ),
    )

    with pytest.raises(ValueError, match="requires exactly"):
        build_task_matrix(
            "editing-t1-test-v1",
            families="all_families,cycle_attach",
            panel_kinds="global_repeated_state_distribution",
        )


def test_t1_modal_surface_requires_physical_active8_path_and_hash():
    with pytest.raises(ValueError, match="active8_inventory must be selected"):
        _build_task_matrix(
            "editing-t1-test-v1",
            families="cycle_attach",
            active8_inventory_file_sha256=ACTIVE8_FILE_SHA256,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _build_task_matrix(
            "editing-t1-test-v1",
            families="cycle_attach",
            active8_inventory=ACTIVE8_INVENTORY,
            active8_inventory_file_sha256="not-a-sha",
        )
