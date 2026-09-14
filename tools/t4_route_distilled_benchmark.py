#!/usr/bin/env python3
"""Prepare, preflight, launch and monitor route-distilled T4 qualification."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.route_distilled_dynamic_optimizer import (
    RouteDistilledDynamicOptimizer,
    initial_route_distilled_program_batch,
)
from compose_v4.control.route_distilled_program_policy import (
    RouteDistilledProgramPolicy,
)
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_route_distilled_qualification import (
    ACTOR,
    APP,
    APP_NAME,
    CONTRACT,
    EMPTY_LIBRARY,
    KIND,
    POLICY_SELECTION,
    PREFLIGHT,
    SELECTED_UNITS,
    SOURCE_CONTRACT,
    load_contract,
    qualification_payload,
)
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "diagnostics/t4_route_distilled_qualification"
LAUNCH = ATTEMPT / "launch.json"
RESULT = ATTEMPT / "result.json"
VOLUME = "compose-t4-route-distilled-artifacts"


def _material_inputs() -> tuple[str, ...]:
    return (
        SOURCE_CONTRACT,
        EMPTY_LIBRARY,
        POLICY_SELECTION,
        ACTOR,
        "docs/GENMOL_T4_SEEDS.json",
        "docs/T4_ROUTE_DISTILLED_DYNAMIC.md",
        "modal_apps/genmol_t4_opt_app.py",
        APP,
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/dynamic_program_synthesis.py",
        "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        "src/compose_v4/control/route_distilled_program_policy.py",
        "src/compose_v4/control/route_distilled_dynamic_optimizer.py",
        "src/compose_v4/control/program_transfer.py",
        "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
        "src/compose_v4/experiments/t4_route_distilled_qualification.py",
        "tools/t4_route_distilled_benchmark.py",
    )


def _actor_checkpoint() -> dict:
    with gzip.open(ROOT / ACTOR, "rt") as handle:
        envelope = json.load(handle)
    if identity(envelope.get("payload")) != envelope.get("payload_sha256"):
        raise ValueError("route-distilled actor checkpoint changed")
    return envelope["payload"]


def prepare() -> None:
    target = ROOT / CONTRACT
    if target.exists() or (ROOT / PREFLIGHT).exists() or LAUNCH.exists():
        raise ValueError("route-distilled qualification already prepared")
    selection = json.loads((ROOT / POLICY_SELECTION).read_text())
    if selection.get("selected_policy") != "context_module_prototype":
        raise ValueError("held-source policy gate has not selected the actor")
    source = unseal(ROOT / SOURCE_CONTRACT)
    contract = deepcopy(source)
    contract["library_path"] = EMPTY_LIBRARY
    contract["library_programs"] = 0
    contract["inputs"] = {path: sha256_file(ROOT / path) for path in _material_inputs()}
    contract["information_regime"] = {
        **contract["information_regime"],
        "route_distilled_dynamic": (
            "one target-free numeric actor trained on 77 public T4 trajectories; "
            "zero stored executable routes or endpoints at runtime"
        ),
        "comparators": "offline only; no comparator result is a runtime input",
    }
    contract["limitations"] = [
        *contract["limitations"],
        "This is a trained-on-public-T4 five-cell qualification.",
        "One stochastic search and docking seed per cell limits general conclusions.",
        "The 90-unit dual-threshold wave is conditional on the frozen qualification gate.",
    ]
    contract["route_distilled_qualification"] = qualification_payload()
    seal(target, contract)
    load_contract(ROOT)
    print(
        json.dumps(
            {
                "contract": CONTRACT,
                "contract_sha256": sha256_file(target),
                "selected_units": SELECTED_UNITS,
                "new_call_ceiling": 5000,
                "plateau_stopping": False,
                "new_oracle_calls": 0,
            },
            indent=2,
        )
    )


def _replay(candidate, config, *, label):
    program = EditProgram.from_payload(candidate["program"])
    _, replay = execute_program_graph(
        decode_state(candidate["source_state"]),
        compile_program_graph(program),
        tuple(candidate["assignment"]),
        max_primitives=config.max_primitives,
        max_blocks=config.max_blocks,
    )
    if replay != candidate["trace"] or replay["endpoint"] != candidate["endpoint"]:
        raise ValueError(f"route-distilled candidate changed on replay: {label}")


def build_preflight(*, code_revision: str | None = None) -> dict:
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("route-distilled preflight requires pinned RDKit 2024.03.5")
    contract = load_contract(ROOT)
    checkpoint = _actor_checkpoint()
    policy = RouteDistilledProgramPolicy.from_checkpoint(checkpoint)
    RouteDistilledDynamicOptimizer.configure_policy(checkpoint)
    began, rows = perf_counter(), []
    for unit_id in SELECTED_UNITS:
        unit = next(row for row in contract["units"] if row["unit_id"] == unit_id)
        source = decode_state(contract["cells"][unit["cell"]]["source_state"])
        config = replace(
            benchmark.configured(contract, unit),
            seed=contract["cold_start_seed"],
            candidates_per_batch=4,
            attempts_per_batch=16,
            wall_seconds=20,
        )
        kwargs = {
            "source_group": identity(
                {
                    "target": unit["target"],
                    "source_idx": unit["source_idx"],
                    "seed": unit["original_seed"],
                }
            ),
            "oracle_protocol": unit["oracle_protocol"],
            "eligibility": benchmark.strict_endpoint_scorer(
                unit["original_seed"], delta=contract["delta"]
            ),
            "policy": policy,
        }
        first = initial_route_distilled_program_batch(source, (), config, **kwargs)
        second = initial_route_distilled_program_batch(source, (), config, **kwargs)
        if first["batch_id"] != second["batch_id"]:
            raise ValueError(
                f"route-distilled cold start is nondeterministic: {unit_id}"
            )
        if not first["candidates"]:
            raise ValueError(
                f"route-distilled cold start yielded no candidate: {unit_id}"
            )
        for candidate in first["candidates"]:
            _replay(candidate, config, label=unit_id)
        optimizer = RouteDistilledDynamicOptimizer(
            benchmark.configured(contract, unit),
            source_group=kwargs["source_group"],
            oracle_protocol=kwargs["oracle_protocol"],
            hierarchy=None,
        )
        snapshot = optimizer.snapshot(include_history=False)
        restored = RouteDistilledDynamicOptimizer.restore(snapshot, hierarchy=None)
        if restored.snapshot(include_history=False) != snapshot:
            raise ValueError(f"route-distilled resume changed state: {unit_id}")
        rows.append(
            {
                "unit_id": unit_id,
                "candidates": len(first["candidates"]),
                "attempts": len(first["attempts"]),
                "channels": sorted({row["channel"] for row in first["attempts"]}),
                "batch_id": first["batch_id"],
            }
        )
    return {
        "schema_version": "t4_route_distilled_qualification_preflight_v1",
        "passed": len(rows) == len(SELECTED_UNITS),
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "policy_selection_sha256": sha256_file(ROOT / POLICY_SELECTION),
        "policy_checkpoint_sha256": sha256_file(ROOT / ACTOR),
        "policy_training_identity": policy.training_identity,
        "development_cells": rows,
        "runtime_route_rows": 0,
        "runtime_teacher_lookup": False,
        "plateau_stopping": False,
        "new_oracle_calls": 0,
        "seconds": perf_counter() - began,
        "rdkit": rdBase.rdkitVersion,
        "code_revision": (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            if code_revision is None
            else code_revision
        ),
    }


def _task(*, include_preflight=True):
    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    revision = local_image_revision(
        expected_commit=assert_synced(strict=True)["commit"], repo_root=ROOT
    )
    paths = [CONTRACT, APP, POLICY_SELECTION, ACTOR]
    if include_preflight:
        paths.append(PREFLIGHT)
    body = {
        "image_revision": revision,
        "files_sha256": {path: sha256_file(ROOT / path) for path in paths},
    }
    return {**body, "run_id": identity(body)}


def remote_preflight() -> None:
    import modal

    output = ROOT / PREFLIGHT
    if output.exists():
        raise ValueError("route-distilled preflight exists; do not overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("route-distilled remote preflight requires clean source")
    task = _task(include_preflight=False)
    body = modal.Function.from_name(APP_NAME, "structural_preflight").remote(task)
    if body.get("passed") is not True or body.get("new_oracle_calls") != 0:
        raise ValueError("route-distilled remote preflight did not pass")
    publish_json(output, body)
    print(json.dumps(body, indent=2))


def launch() -> None:
    import modal

    if LAUNCH.exists():
        raise ValueError("route-distilled launch exists; monitor instead")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("route-distilled launch requires clean source")
    contract = load_contract(ROOT)
    task = _task()
    remote = modal.Function.from_name(APP_NAME, "preflight").remote(task)
    if remote.get("passed") is not True or remote.get("new_oracle_calls") != 0:
        raise ValueError("route-distilled runtime-input preflight failed")
    worker = modal.Function.from_name(APP_NAME, "worker")
    calls = {
        unit: worker.spawn({**task, "unit_id": unit}).object_id
        for unit in SELECTED_UNITS
    }
    saved = {
        "schema_version": "t4_route_distilled_qualification_launch_v1",
        "task": task,
        "calls": calls,
        "volume": VOLUME,
        "volume_path": f"{KIND}/{task['run_id']}",
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "preflight_sha256": sha256_file(ROOT / PREFLIGHT),
        "new_call_ceiling": contract["route_distilled_qualification"][
            "new_call_ceiling"
        ],
        "automatic_retries": 0,
    }
    seal(LAUNCH, saved)
    print(
        json.dumps(
            {key: value for key, value in saved.items() if key != "task"}, indent=2
        )
    )


def _remote_json(volume, path):
    raw = b"".join(volume.read_file(path))
    value = json.loads(raw)
    if set(value) == {"payload", "payload_sha256"}:
        if identity(value["payload"]) != value["payload_sha256"]:
            raise ValueError(f"corrupt sealed remote artifact: {path}")
        return value["payload"]
    return value


def _unit_state(volume, saved, unit):
    folder = f"{saved['volume_path']}/units/{unit}"
    try:
        names = {row.path.rsplit("/", 1)[-1] for row in volume.listdir(folder)}
    except __import__("modal").exception.NotFoundError:
        return {}
    for name in ("result.json", "failure.json", "progress.json"):
        if name in names:
            return _remote_json(volume, f"{folder}/{name}")
    if "checkpoint.json.gz" in names:
        saved_state = json.loads(
            gzip.decompress(b"".join(volume.read_file(f"{folder}/checkpoint.json.gz")))
        )
        champion = saved_state.get("champion")
        return {
            "status": "running",
            "oracle_calls": saved_state.get("query_count", 0),
            "best_score": None if champion is None else champion["score"],
        }
    return {}


def status() -> None:
    import modal

    saved = unseal(LAUNCH)
    volume = modal.Volume.from_name(saved["volume"])
    rows = []
    for unit in SELECTED_UNITS:
        current = _unit_state(volume, saved, unit)
        calls = current.get(
            "oracle_calls", current.get("queries_total", current.get("queries", 0))
        )
        best = current.get("best_score")
        if best is None and current.get("champion") is not None:
            best = current["champion"]["score"]
        rows.append(
            {
                "unit": unit,
                "status": current.get("status", "pending"),
                "calls": calls,
                "best_score": best,
            }
        )
    print(
        json.dumps(
            {"units": rows, "total_calls": sum(row["calls"] for row in rows)}, indent=2
        )
    )


def collect() -> None:
    import modal

    if RESULT.exists():
        raise ValueError("route-distilled qualification result exists")
    saved = unseal(LAUNCH)
    contract = load_contract(ROOT)
    volume = modal.Volume.from_name(saved["volume"])
    rows = []
    for unit_id in SELECTED_UNITS:
        current = _remote_json(
            volume, f"{saved['volume_path']}/units/{unit_id}/result.json"
        )
        if current.get("status") != "complete":
            raise ValueError(f"qualification unit is not complete: {unit_id}")
        unit = current["unit"]
        champion = current.get("champion")
        rows.append(
            {
                "unit": unit_id,
                "oracle_calls": current["oracle_calls"],
                "successful_oracle_calls": current["successful_oracle_calls"],
                "termination": current["termination"],
                "best_score": None if champion is None else champion["score"],
                "ivg_mean": contract["ivg_reported"][unit["cell"]]["mean"],
                "curve": current["curve"],
            }
        )
    scored = all(
        row["successful_oracle_calls"] > 0 and row["best_score"] is not None
        for row in rows
    )
    wins = sum(
        row["best_score"] <= row["ivg_mean"]
        for row in rows
        if row["best_score"] is not None
    )
    passed = scored and wins >= 4
    body = {
        "schema_version": "t4_route_distilled_qualification_result_v1",
        "decision": (
            "full_dual_threshold_authorized" if passed else "qualification_failed"
        ),
        "passed": passed,
        "cells_at_or_better_than_ivg_mean": wins,
        "required_cells_at_or_better_than_ivg_mean": 4,
        "rows": rows,
        "new_oracle_calls": sum(row["oracle_calls"] for row in rows),
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "launch_sha256": sha256_file(LAUNCH),
        "evidence": "trained-on-public-T4 route-distilled five-cell qualification",
    }
    seal(RESULT, body)
    print(
        json.dumps(
            {key: value for key, value in body.items() if key != "rows"}, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "remote-preflight", "launch", "status", "collect")
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "remote-preflight": remote_preflight,
        "launch": launch,
        "status": status,
        "collect": collect,
    }[args.action]()


if __name__ == "__main__":
    main()
