#!/usr/bin/env python3
"""Prepare and launch the conditional 90-unit route-distilled T4 benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_route_distilled_full import (
    ACTOR,
    APP,
    CONTRACT,
    EMPTY_LIBRARY,
    POLICY_SELECTION,
    PREFLIGHT,
    QUALIFICATION,
    SEED_PAIRS,
    SOURCE_CONTRACT,
    full_payload,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "diagnostics/t4_route_distilled_full"
LAUNCH = ATTEMPT / "launch.json"
VOLUME = "compose-t4-route-distilled-artifacts"
APP_NAME = "compose-t4-route-distilled"


def _material_inputs() -> tuple[str, ...]:
    return (
        SOURCE_CONTRACT,
        EMPTY_LIBRARY,
        POLICY_SELECTION,
        ACTOR,
        QUALIFICATION,
        "docs/GENMOL_T4_SEEDS.json",
        "docs/T4_ROUTE_DISTILLED_DYNAMIC.md",
        APP,
        "modal_apps/genmol_t4_opt_app.py",
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/dynamic_program_synthesis.py",
        "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        "src/compose_v4/control/route_distilled_program_policy.py",
        "src/compose_v4/control/route_distilled_dynamic_optimizer.py",
        "src/compose_v4/control/program_transfer.py",
        "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
        "src/compose_v4/experiments/t4_route_distilled_full.py",
        "tools/t4_route_distilled_full.py",
    )


def _contract(delta: float) -> dict:
    source = unseal(ROOT / SOURCE_CONTRACT)
    contract = deepcopy(source)
    contract["schema_version"] = "t4_route_distilled_full_benchmark_v1"
    contract["delta"] = delta
    contract["library_path"] = EMPTY_LIBRARY
    contract["library_programs"] = 0
    contract["plateau_stop"] = None
    contract["confirmation_call_ceiling"] = 0
    contract["total_call_ceiling"] = 45000
    contract["container_limit"] = 15
    contract["units"] = deepcopy(source["units"])
    for unit in contract["units"]:
        domain = unit["oracle_domain"]
        domain["schema_version"] = (
            "t4_strict_delta04_route_distilled_oracle_v1"
            if delta == 0.4
            else "t4_strict_delta06_route_distilled_oracle_v1"
        )
        domain["strict_filters"] = {
            "qed_gt": 0.6,
            "sa_lt": 4.0,
            "similarity_gt": delta,
        }
        unit["oracle_protocol"] = identity(domain)
    contract["inputs"] = {path: sha256_file(ROOT / path) for path in _material_inputs()}
    contract["information_regime"] = {
        **contract["information_regime"],
        "route_distilled_dynamic": (
            "one frozen target-free actor trained on all 77 public T4 routes; "
            "no executable route bank or endpoint at runtime"
        ),
        "cross_threshold": "only the strict similarity threshold changes",
    }
    contract["limitations"] = [
        *contract["limitations"],
        "This is trained-on-public-T4 performance, not held-out-route evidence.",
        "The two thresholds share a controller and paired random seeds.",
        "No plateau stopping or final confirmation docking is part of this wave.",
    ]
    contract["route_distilled_full"] = full_payload(delta)
    return contract


def prepare() -> None:
    qualification = unseal(ROOT / QUALIFICATION)
    if qualification.get("passed") is not True:
        raise ValueError("qualification failed; full T4 contracts are not authorized")
    if LAUNCH.exists() or any((ROOT / path).exists() for path in CONTRACT.values()):
        raise ValueError("full route-distilled benchmark already prepared")
    for delta, relative in CONTRACT.items():
        seal(ROOT / relative, _contract(delta))
        load_contract(ROOT, delta=delta)
    print(
        json.dumps(
            {
                "status": "prepared",
                "contracts": {
                    str(delta): {
                        "path": path,
                        "sha256": sha256_file(ROOT / path),
                    }
                    for delta, path in CONTRACT.items()
                },
                "units": 90,
                "new_call_ceiling": 90000,
                "max_concurrent_workers": 15,
            },
            indent=2,
        )
    )


def build_preflight(delta: float, *, code_revision: str | None = None) -> dict:
    contract = load_contract(ROOT, delta=delta)
    source = unseal(ROOT / SOURCE_CONTRACT)
    if [row["unit_id"] for row in contract["units"]] != [
        row["unit_id"] for row in source["units"]
    ]:
        raise ValueError("route-distilled full unit order changed")
    paired = sorted(
        {
            (row["replicate"], row["controller_seed"], row["docking_seed"])
            for row in contract["units"]
        }
    )
    expected = [(index, *pair) for index, pair in enumerate(SEED_PAIRS)]
    if paired != expected:
        raise ValueError("route-distilled full paired seeds changed")
    rows = []
    for cell, current in sorted(contract["cells"].items()):
        scorer = __import__(
            "compose_v4.experiments.t4_frozen_program_benchmark",
            fromlist=["strict_endpoint_scorer"],
        ).strict_endpoint_scorer(current["original_seed"], delta=delta)
        seed_result = scorer({"smiles": current["original_seed"]})
        if seed_result["sim"] != 1.0:
            raise ValueError(f"route-distilled source identity changed: {cell}")
        rows.append({"cell": cell, "source_similarity": seed_result["sim"]})
    return {
        "schema_version": "t4_route_distilled_full_preflight_v1",
        "passed": len(rows) == 15,
        "delta": delta,
        "contract_sha256": sha256_file(ROOT / CONTRACT[delta]),
        "policy_selection_sha256": sha256_file(ROOT / POLICY_SELECTION),
        "policy_checkpoint_sha256": sha256_file(ROOT / ACTOR),
        "qualification_sha256": sha256_file(ROOT / QUALIFICATION),
        "units": 45,
        "sources": rows,
        "plateau_stopping": False,
        "new_oracle_calls": 0,
        "code_revision": (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            if code_revision is None
            else code_revision
        ),
    }


def _task(delta: float, *, include_preflight=True):
    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    revision = local_image_revision(
        expected_commit=assert_synced(strict=True)["commit"], repo_root=ROOT
    )
    paths = [CONTRACT[delta], APP, POLICY_SELECTION, ACTOR, QUALIFICATION]
    if include_preflight:
        paths.append(PREFLIGHT[delta])
    body = {
        "delta": delta,
        "image_revision": revision,
        "files_sha256": {path: sha256_file(ROOT / path) for path in paths},
    }
    return {**body, "run_id": identity(body)}


def remote_preflight() -> None:
    import modal

    if any((ROOT / path).exists() for path in PREFLIGHT.values()):
        raise ValueError("full route-distilled preflight already exists")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("full route-distilled preflight requires clean source")
    function = modal.Function.from_name(APP_NAME, "structural_preflight_full")
    for delta in sorted(CONTRACT):
        task = _task(delta, include_preflight=False)
        result = function.remote(task)
        if result.get("passed") is not True or result.get("new_oracle_calls") != 0:
            raise ValueError(f"full route-distilled preflight failed at delta {delta}")
        publish_json(ROOT / PREFLIGHT[delta], result)
    print(
        json.dumps({"status": "preflight_passed", "deltas": sorted(CONTRACT)}, indent=2)
    )


def launch() -> None:
    import modal

    if LAUNCH.exists():
        raise ValueError("full route-distilled launch exists; monitor instead")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("full route-distilled launch requires clean source")
    worker = modal.Function.from_name(APP_NAME, "worker_full")
    calls, tasks = {}, {}
    for delta in sorted(CONTRACT):
        contract = load_contract(ROOT, delta=delta)
        task = _task(delta)
        preflight = modal.Function.from_name(APP_NAME, "preflight_full").remote(task)
        if (
            preflight.get("passed") is not True
            or preflight.get("new_oracle_calls") != 0
        ):
            raise ValueError(f"runtime preflight failed at delta {delta}")
        tasks[str(delta)] = task
        for unit in contract["units"]:
            key = f"{delta}:{unit['unit_id']}"
            calls[key] = worker.spawn({**task, "unit_id": unit["unit_id"]}).object_id
    body = {
        "schema_version": "t4_route_distilled_full_launch_v1",
        "tasks": tasks,
        "calls": calls,
        "volume": VOLUME,
        "units": 90,
        "new_call_ceiling": 90000,
        "max_concurrent_workers": 15,
        "automatic_retries": 0,
    }
    seal(LAUNCH, body)
    print(
        json.dumps(
            {key: value for key, value in body.items() if key != "tasks"}, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "remote-preflight", "launch"))
    parser.add_argument("--delta", type=float, choices=(0.4, 0.6))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "remote-preflight":
        remote_preflight()
    else:
        launch()


if __name__ == "__main__":
    main()
