#!/usr/bin/env python3
"""Spawn T4 cells against the DEPLOYED app, so they outlive this process.

`modal run --detach` still cancels a .map() when the client dies, and spawning
from an ephemeral `modal run` is no better -- the ephemeral app is torn down the
moment the entrypoint returns, taking its spawned calls with it. Four T4
attempts died that way. A deployed app persists independently, so a call
spawned into it keeps running with nothing local attached.

Usage:
    modal deploy modal_apps/genmol_t4_opt_app.py     # once, after any edit
    python3 tools/t4_launch.py --budget 20 --n-seeds 1 --deltas 0.4 --arms macro_prior
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--deltas", default="0.4")
    ap.add_argument("--arms", default="macro_prior")
    ap.add_argument("--session", default="compose_iclr")
    ap.add_argument(
        "--option-decision-audit",
        action="store_true",
        help="four conditional option decision checks; zero docking",
    )
    ap.add_argument(
        "--frontier-compare",
        action="store_true",
        help="one paired resumable-frontier round, at most 40 new dockings total",
    )
    ap.add_argument(
        "--target-recovery",
        action="store_true",
        help="one answer-known 32-edit controller diagnostic; zero docking",
    )
    ap.add_argument(
        "--task-search-audit",
        action="store_true",
        help="one bounded cross-option production preparation; zero docking",
    )
    ap.add_argument(
        "--ring-program-round",
        action="store_true",
        help="one parameterized-ring round from the 46-call archive, at most 20 new calls",
    )
    ap.add_argument(
        "--feedback-round",
        action="store_true",
        help="one guided round from all 33 evaluated molecules, at most 20 new calls",
    )
    ap.add_argument(
        "--partial-docking",
        action="store_true",
        help="approved saved partial-round batch only, at most 13 new dockings",
    )
    ap.add_argument(
        "--warm-continuation",
        action="store_true",
        help="two guided warm rounds from the frozen full archive, at most 40 new calls",
    )
    ap.add_argument(
        "--matched-pilot",
        action="store_true",
        help="one frozen paired cold-start round, at most 20 dockings per arm",
    )
    ap.add_argument(
        "--continuation-profile",
        action="store_true",
        help="one frozen, capped zero-docking profile; no T4 cells are launched",
    )
    ap.add_argument(
        "--fused-reference-profile",
        action="store_true",
        help="one capped frozen-model fused reference trajectory; no docking or lookahead",
    )
    ap.add_argument(
        "--lazy-reference-probe",
        action="store_true",
        help="one saved-parent task-search cost probe; zero docking",
    )
    ap.add_argument(
        "--uncapped-lookahead-probe",
        action="store_true",
        help="same parent and horizon without executor/state cutoffs; zero docking",
    )
    # V_z = 0 until a T4-specific estimator exists; the QED table was null and
    # learned QED-region effects, and an interface accepting it is not a reason
    # to use it here.  `macro_prior` names the new unlearned option-prior arm.
    ap.add_argument("--lineages", type=int, default=8)
    ap.add_argument("--per-round", type=int, default=20)
    ap.add_argument("--regions-per-lineage", type=int, default=3)
    ap.add_argument("--particles-per-region", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--max-frontier",
        type=int,
        default=8,
        help="search width per bundle (NOT the candidate pool)",
    )
    ap.add_argument(
        "--emit-per-bundle",
        type=int,
        default=3,
        help="representatives offered to the oracle per bundle",
    )
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--epsilon-option", type=float, default=0.1)
    ap.add_argument("--macro-temperature", type=float, default=2.0)
    ap.add_argument("--epsilon-macro", type=float, default=0.15)
    a = ap.parse_args()
    if (
        sum(
            (
                a.continuation_profile,
                a.fused_reference_profile,
                a.matched_pilot,
                a.warm_continuation,
                a.partial_docking,
                a.feedback_round,
                a.ring_program_round,
                a.task_search_audit,
                a.lazy_reference_probe,
                a.uncapped_lookahead_probe,
                a.target_recovery,
                a.frontier_compare,
                a.option_decision_audit,
            )
        )
        > 1
    ):
        ap.error("choose only one diagnostic")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.session):
        raise SystemExit("--session must contain only letters, numbers, '_' or '-'")

    from preflight import assert_synced

    preflight = assert_synced(strict=True)
    if preflight["dirty"]:
        raise SystemExit(
            "PREFLIGHT FAIL: Modal scientific launches require a fully clean "
            "committed tree, not only clean mounted src/configs"
        )

    if (
        a.continuation_profile
        or a.fused_reference_profile
        or a.matched_pilot
        or a.warm_continuation
        or a.partial_docking
        or a.feedback_round
        or a.ring_program_round
        or a.task_search_audit
        or a.lazy_reference_probe
        or a.uncapped_lookahead_probe
        or a.target_recovery
        or a.frontier_compare
        or a.option_decision_audit
    ):
        launch_continuation_profile(
            preflight,
            fused=a.fused_reference_profile,
            matched=a.matched_pilot,
            warm=a.warm_continuation,
            partial=a.partial_docking,
            feedback=a.feedback_round,
            rings=a.ring_program_round,
            task_search=a.task_search_audit,
            lazy_probe=a.lazy_reference_probe,
            uncapped_probe=a.uncapped_lookahead_probe,
            target_recovery=a.target_recovery,
            frontier_compare=a.frontier_compare,
            option_decision=a.option_decision_audit,
        )
        return

    fn = modal.Function.from_name("genmol-t4-opt", "t4_population_cell")
    seed_manifest = ROOT / "docs/GENMOL_T4_SEEDS.json"
    seeds = json.loads(seed_manifest.read_text())[: a.n_seeds]
    tbl = None
    p = ROOT / "diagnostics/task_value_qed.json"
    if p.exists():
        tbl = json.loads(p.read_text())

    spawned = []
    for i, s in enumerate(seeds):
        for d in [float(x) for x in a.deltas.split(",")]:
            for arm in [x.strip() for x in a.arms.split(",")]:
                t = {
                    "cell": f"{a.session}_{arm}_{s['target']}_{i}_d{d}",
                    "session": a.session,
                    "smiles": s["smiles"],
                    "target": s["target"],
                    "delta": d,
                    "budget": a.budget,
                    "arm": arm,
                    "tau": a.tau,
                    "seed_rng": 1000 + i,
                    "kappa": 1.0,
                    "epsilon_region": 0.2,
                    "epsilon": 0.1,
                    "epsilon_option": a.epsilon_option,
                    "macro_temperature": a.macro_temperature,
                    "epsilon_macro": a.epsilon_macro,
                    "code_revision": preflight["commit"],
                    "seed_manifest_sha256": _sha256(seed_manifest),
                    "lineages": a.lineages,
                    "per_round": a.per_round,
                    "regions_per_lineage": a.regions_per_lineage,
                    "particles_per_region": a.particles_per_region,
                    "workers": a.workers,
                    "max_frontier": a.max_frontier,
                    "emit_per_bundle": a.emit_per_bundle,
                }
                if arm == "Q_taskvalue":
                    if tbl is None:
                        raise SystemExit("no task_value table for the tilt arm")
                    t["value_table"] = tbl
                spawned.append((t["cell"], fn.spawn(t).object_id))

    (ROOT / "diagnostics").mkdir(exist_ok=True)
    receipt_path = ROOT / "diagnostics/t4_spawned_three_level.json"
    receipt_payload = json.dumps(
        {
            "schema_version": "t4_three_level_spawn_v1",
            "session": a.session,
            "code_revision": preflight["commit"],
            "seed_manifest_sha256": _sha256(seed_manifest),
            "budget": a.budget,
            "epsilon_option": a.epsilon_option,
            "macro_temperature": a.macro_temperature,
            "epsilon_macro": a.epsilon_macro,
            "cells": [{"cell": c, "call": o} for c, o in spawned],
        },
        indent=2,
        sort_keys=True,
    )
    temporary_path = receipt_path.with_suffix(".json.tmp")
    temporary_path.write_text(receipt_payload)
    temporary_path.replace(receipt_path)
    print(f"spawned {len(spawned)} cell(s) into the DEPLOYED app; nothing local is holding them")
    print(f"receipt: {receipt_path}")
    for c, o in spawned[:8]:
        print(f"  {c}  {o}")


def launch_continuation_profile(
    preflight: dict,
    *,
    fused: bool = False,
    matched: bool = False,
    warm: bool = False,
    partial: bool = False,
    feedback: bool = False,
    rings: bool = False,
    task_search: bool = False,
    lazy_probe: bool = False,
    uncapped_probe: bool = False,
    target_recovery: bool = False,
    frontier_compare: bool = False,
    option_decision: bool = False,
) -> None:
    """Spawn only the fixed diagnostic into the deployed app; retain its call ID."""
    import sys

    sys.path.insert(0, str(ROOT))
    from modal_apps.run_process_v2_p50_app import local_image_revision

    kind = (
        "t4_option_decision_audit"
        if option_decision
        else "t4_frontier_compare"
        if frontier_compare
        else "t4_target_recovery"
        if target_recovery
        else "t4_uncapped_lookahead_probe"
        if uncapped_probe
        else "t4_lazy_reference_probe"
        if lazy_probe
        else "t4_task_search_audit"
        if task_search
        else "t4_ring_program_round"
        if rings
        else "t4_feedback_round"
        if feedback
        else "t4_partial_docking"
        if partial
        else "t4_warm_continuation"
        if warm
        else "t4_matched_pilot"
        if matched
        else "fused_reference_profile"
        if fused
        else "continuation_profile"
    )
    contract = (
        ROOT
        / {
            "t4_option_decision_audit": "configs/t4_option_decision_audit.json",
            "t4_frontier_compare": "configs/t4_frontier_compare.json",
            "t4_target_recovery": "configs/t4_target_recovery.json",
            "t4_task_search_audit": "configs/t4_task_search_audit.json",
            "t4_lazy_reference_probe": "configs/t4_lazy_reference_probe.json",
            "t4_uncapped_lookahead_probe": "configs/t4_uncapped_lookahead_probe.json",
            "t4_ring_program_round": "configs/t4_ring_program_round.json",
            "t4_feedback_round": "configs/t4_feedback_round.json",
            "t4_partial_docking": "configs/t4_partial_docking.json",
            "t4_warm_continuation": "configs/t4_warm_continuation.json",
            "t4_matched_pilot": "configs/t4_matched_pilot.json",
            "fused_reference_profile": "configs/fused_reference_profile_contract.json",
            "continuation_profile": "configs/continuation_profile_v1.json",
        }[kind]
    )
    revision = local_image_revision(expected_commit=preflight["commit"])
    identity = {
        "contract_sha256": _sha256(contract),
        "image_revision_sha256": revision["image_revision_sha256"],
        "app_sha256": _sha256(ROOT / "modal_apps/genmol_t4_opt_app.py"),
    }
    run_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    task = {
        "run_id": run_id,
        "image_revision": revision,
        "contract_sha256": identity["contract_sha256"],
        "app_sha256": identity["app_sha256"],
    }
    function = modal.Function.from_name("genmol-t4-opt", kind)
    if option_decision:
        receipt = {
            "schema_version": "t4_option_decision_spawn_v1",
            "task": task,
            "volume": "compose-v4-artifacts",
            "oracle_calls": 0,
            "cases": [],
        }
        path = ROOT / "diagnostics/t4_option_decision_spawn.json"
        for case in range(4):
            call = function.spawn({**task, "case_index": case})
            volume_path = f"/{kind}/case_{case}/{run_id}"
            receipt["cases"].append(
                {"case_index": case, "call_id": call.object_id, "volume_path": volume_path}
            )
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
            temporary.replace(path)
            print(f"spawned case {case}: {call.object_id}; {volume_path}; zero docking")
        return
    call = function.spawn(task)
    receipt = {
        "schema_version": f"{kind}_spawn_v1",
        "task": task,
        "call_id": call.object_id,
        "oracle_calls": 0,
        "oracle_call_limit": 20
        if feedback or rings
        else 13
        if partial
        else 40
        if matched or warm or frontier_compare
        else 0,
        "volume": "compose-v4-artifacts",
        "volume_path": f"/{kind}/{run_id}",
    }
    path = ROOT / f"diagnostics/{kind}_spawn.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)
    print(f"spawned {kind}: {call.object_id}; oracle limit={receipt['oracle_call_limit']}")
    print(f"volume result: {receipt['volume_path']}/result.json")
    print(f"receipt: {path}")


if __name__ == "__main__":
    main()
