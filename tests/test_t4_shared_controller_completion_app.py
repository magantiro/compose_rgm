import hashlib
import inspect

import modal_apps.t4_shared_controller_completion_v1_app as launcher
from compose_v4.experiments.t4_shared_controller_cell_runtime import (
    QUERY_RECEIPT_SCHEMA,
    make_query_lock,
    make_query_reservation,
    mark_driver_running,
    reserve_driver_generation,
    settle_query_lock,
)
from compose_v4.experiments.t4_shared_controller_checkpoint import (
    initialize_after_settled_root,
    select_parent_and_batch,
)
from compose_v4.experiments.t4_shared_controller_scored_runtime import (
    attach_endpoint_fingerprints,
    filter_stale_braf_candidates,
)


def _hashes(count, *, include=()):
    values = set(include)
    index = 0
    while len(values) < count:
        values.add(hashlib.sha256(f"fixture-{index}".encode()).hexdigest())
        index += 1
    return sorted(values)


def test_launcher_registers_nine_private_cell_runtimes_and_is_inert_on_import():
    assert len(launcher.CELL_KEYS) == 9
    assert set(launcher.CELL_FUNCTIONS) == set(launcher.CELL_KEYS)
    assert len(set(launcher.CELL_VOLUMES.values())) == 9
    assert all(
        set(launcher.CELL_FUNCTIONS[cell_key])
        == {"volume", "proposal", "particle", "dock", "driver", "status"}
        for cell_key in launcher.CELL_KEYS
    )
    report = launcher.scored_preflight_report()
    assert report["modal_calls_created"] == 0
    if not report["ready"]:
        assert report["missing"]
    main_source = (launcher.ROOT / launcher.APP_RELATIVE_PATH).read_text()
    assert "reserve_driver_generation" in main_source
    assert "_spawn_reserved_drivers" in main_source
    assert "REMOTE_ROOT / 'modal_apps'" in main_source


def test_resume_plan_is_independent_and_never_overlaps_a_live_cell():
    cells = ("preempted", "live", "finished")
    running = {}
    for cell_key in cells:
        reserved = reserve_driver_generation(
            phase_status="running", existing_state=None
        )["state"]
        running[cell_key] = mark_driver_running(reserved, f"fc-{cell_key}")
    receipt = {
        "launch": {"cell_keys": list(cells)},
        "cells": {cell_key: {"driver_state": running[cell_key]} for cell_key in cells},
    }
    plan = launcher._plan_driver_resume(
        receipt,
        terminal_by_cell={"preempted": True, "live": False, "finished": True},
        phase_status_by_cell={
            "preempted": "queries_running",
            "live": "running",
            "finished": "complete_budget",
        },
    )
    assert plan["preempted"]["action"] == "spawn"
    assert plan["preempted"]["state"]["generation"] == 1
    assert plan["live"] == {"action": "wait", "state": running["live"]}
    assert plan["finished"]["action"] == "finish"
    assert plan["finished"]["state"]["state"] == "terminal"


def test_driver_filters_bound_stale_braf_hashes_before_pure_selection():
    driver_source = inspect.getsource(launcher._drive_cell)
    assert driver_source.index(
        "pools, exclusion_ledger = filter_stale_braf_candidates"
    ) < driver_source.index("plan = select_parent_and_batch")

    stale = hashlib.sha256(b"CCO").hexdigest()
    contract = {
        "stale_braf_v4_reconciliation": {
            "excluded_canonical_smiles_sha256": {
                "braf_0": _hashes(20, include=(stale,)),
                "braf_1": _hashes(23),
            }
        }
    }
    pools, ledger = filter_stale_braf_candidates(
        {
            "shallow": [{"smiles": "CCO"}, {"smiles": "CCN"}],
            "anchored_replacement": [],
            "route_complete_region": [],
        },
        cell={"target": "braf", "source_cell": "braf_0"},
        contract=contract,
    )
    assert [row["smiles"] for row in pools["shallow"]] == ["CCN"]
    assert ledger == {
        "source_cell": "braf_0",
        "denylist_size": 20,
        "excluded_by_expert": {
            "anchored_replacement": 0,
            "route_complete_region": 0,
            "shallow": 1,
        },
        "excluded_total": 1,
        "prior_scores_or_receipts_read": False,
    }


def test_particle_route_rows_are_fiber_gated_before_query_lock():
    cell = next(row for row in launcher.CELLS if row["cell_key"] == "braf_1_d06")
    parent = cell["source_smiles"]
    eligible_smiles = "CC(=O)NC(=O)Nc1ccc(Oc2ccnc(C(=O)NCCN3CCOCC3)c2)cc1"
    base = {
        "proposal_lane": "route_complete_region",
        "proposal_experts": ["route_complete_region"],
        "families": ["route_complete_region"],
        "program_families": ["route_complete_region"],
        "parent": parent,
        "parent_score": -8.0,
        "delta": cell["delta"],
        "regions": 1,
        "created": 1,
        "deleted": 1,
        "realized_primitives": 2,
        "realized_primitive_band": "small",
    }
    gated = launcher._fiber_gate_route_rows(
        [{**base, "smiles": "CC"}, {**base, "smiles": eligible_smiles}],
        cell=cell,
        contract={"support": "compose_valid"},
    )
    assert [row["smiles"] for row in gated] == [eligible_smiles]
    assert all(key in gated[0] for key in ("similarity", "qed", "sa"))

    root_lock = make_query_lock(
        cell_key=cell["cell_key"],
        round_index=0,
        charged_before=0,
        selected_rows=[{"smiles": parent}],
        query_deadline=1.0,
    )
    root_query = root_lock["queries"][0]
    root_receipt = {
        **make_query_reservation(root_lock, root_query["query_id"]),
        "schema_version": QUERY_RECEIPT_SCHEMA,
        "status": "complete",
        "answer": {"score": -8.0, "failure": None},
    }
    checkpoint = initialize_after_settled_root(
        cell_key=cell["cell_key"],
        source_smiles=parent,
        delta=cell["delta"],
        budget_ceiling=49,
        controller_config=launcher.PREPARATION["controller"],
        controller_seed=cell["controller_seed"],
        root_query_lock=root_lock,
        settled_root=settle_query_lock(
            root_lock, {root_query["query_id"]: root_receipt}, now=1.0
        ),
    )
    prepared = attach_endpoint_fingerprints(
        gated,
        original_seed=parent,
        delta=cell["delta"],
        support="compose_valid",
    )
    plan = select_parent_and_batch(
        checkpoint,
        {
            "shallow": [],
            "anchored_replacement": [],
            "route_complete_region": prepared,
        },
        cell_key=cell["cell_key"],
        source_smiles=parent,
        delta=cell["delta"],
        budget_ceiling=49,
        controller_config=launcher.PREPARATION["controller"],
        query_deadline=2.0,
    )
    assert [row["smiles"] for row in plan["query_lock"]["queries"]] == [eligible_smiles]
