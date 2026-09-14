#!/usr/bin/env python3
"""Prepare, preflight, launch and monitor Dynamic COMPOSE v2.1."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    initial_dynamic_program_batch,
    synthesize_dynamic_program,
)
from compose_v4.control.dynamic_program_synthesis_v2 import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
    synthesize_structured_program,
)
from compose_v4.control.dynamic_program_synthesis_v21 import (
    CHANNELS,
    DynamicV21ProgramOptimizer,
    arbitrate_candidates,
    initial_allocator_state,
    initial_dynamic_program_batch_v21,
    validate_allocator_state,
)
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.t4_dynamic_v21 import (
    APP,
    APP_NAME,
    CONTRACT,
    EMPTY_LIBRARY,
    KIND,
    PREFLIGHT,
    SELECTED_UNITS,
    SOURCE_CONTRACT,
    load_contract,
    v21_contract_payload,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "diagnostics/t4_dynamic_v21"
RECEIPT = ATTEMPT / "launch.json"
RESULT = ATTEMPT / "result.json"


def _material_inputs() -> tuple[str, ...]:
    return (
        SOURCE_CONTRACT,
        EMPTY_LIBRARY,
        "docs/GENMOL_T4_SEEDS.json",
        "docs/T4_DYNAMIC_V21.md",
        "modal_apps/genmol_t4_opt_app.py",
        APP,
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/dynamic_program_synthesis.py",
        "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        "src/compose_v4/control/dynamic_program_synthesis_v2.py",
        "src/compose_v4/control/dynamic_program_synthesis_v21.py",
        "src/compose_v4/control/program_transfer.py",
        "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
        "src/compose_v4/experiments/t4_dynamic_v21.py",
        "tools/t4_dynamic_v21_benchmark.py",
    )


def prepare() -> None:
    path = ROOT / CONTRACT
    if path.exists() or (ROOT / PREFLIGHT).exists() or RECEIPT.exists():
        raise ValueError("Dynamic-v2.1 artifacts already exist; do not relock")
    source = unseal(ROOT / SOURCE_CONTRACT)
    contract = deepcopy(source)
    for field in ("dynamic_v1_development", "dynamic_v1_prequery_repairs"):
        contract.pop(field, None)
    contract["library_path"] = EMPTY_LIBRARY
    contract["library_programs"] = 0
    contract["inputs"] = {
        relative: sha256_file(ROOT / relative) for relative in _material_inputs()
    }
    contract["information_regime"] = {
        **contract["information_regime"],
        "dynamic_v21": (
            "zero initial routes; independent generic shallow and structured "
            "proposal streams; post-filter allocation from current-run outcomes only"
        ),
        "comparison": "offline only; no comparator artifact is a runtime input",
    }
    contract["limitations"] = [
        *contract["limitations"],
        "Dynamic-v2.1 was designed on five answer-known T4 development cells.",
        "One search replicate and stochastic docking limit general conclusions.",
        "The other ten T4 cells are outside this development milestone.",
    ]
    contract["dynamic_v21_development"] = v21_contract_payload()
    seal(path, contract)
    load_contract(ROOT)
    print(
        json.dumps(
            {
                "contract": CONTRACT,
                "contract_sha256": sha256_file(path),
                "selected_units": SELECTED_UNITS,
                "new_call_ceiling": 5000,
                "new_oracle_calls": 0,
            },
            indent=2,
        )
    )


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


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
        raise ValueError(f"v2.1 exact replay changed: {label}")


def _representative_channel(channel: str) -> dict:
    source = production_state_from_smiles("C1CCCCC1CCO", max_atoms=48)
    synthesizer = (
        synthesize_dynamic_program
        if channel == SHALLOW_CHANNEL
        else synthesize_structured_program
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([20260913, 221, CHANNELS.index(channel)])
    )
    for attempt in range(64):
        try:
            _, program, binding, trace, metadata = synthesizer(
                source,
                rng,
                max_modules=3,
                max_primitives=32,
                max_blocks=8,
            )
        except ValueError:
            continue
        _, replay = execute_program_graph(
            source,
            compile_program_graph(program),
            binding,
            max_primitives=32,
            max_blocks=8,
        )
        if replay != trace:
            raise ValueError(f"v2.1 representative {channel} replay changed")
        return {
            "channel": channel,
            "attempt": attempt,
            "endpoint": trace["endpoint"],
            "program_id": program.program_id,
            "metadata": metadata,
        }
    raise ValueError(f"v2.1 representative {channel} produced no legal program")


def build_preflight(*, code_revision: str | None = None) -> dict:
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("Dynamic-v2.1 structural preflight requires RDKit 2024.03.5")
    contract = load_contract(ROOT)
    if contract["library_programs"] != 0 or json.loads(
        (ROOT / EMPTY_LIBRARY).read_text()
    ):
        raise ValueError("Dynamic-v2.1 preflight found an initialized route")
    if any("comparison" in path or "result" in path for path in contract["inputs"]):
        raise ValueError("Dynamic-v2.1 runtime inputs contain comparator outcomes")
    began, rows = perf_counter(), []
    representative = {channel: _representative_channel(channel) for channel in CHANNELS}
    for unit_id in SELECTED_UNITS:
        unit = next(row for row in contract["units"] if row["unit_id"] == unit_id)
        cell = contract["cells"][unit["cell"]]
        source = decode_state(cell["source_state"])
        config = replace(
            benchmark.configured(contract, unit),
            seed=contract["cold_start_seed"],
            candidates_per_batch=4,
            attempts_per_batch=2,
            wall_seconds=3600,
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
        }
        first = initial_dynamic_program_batch_v21(source, (), config, **kwargs)
        second = initial_dynamic_program_batch_v21(source, (), config, **kwargs)
        if first["batch_id"] != second["batch_id"]:
            raise ValueError(
                f"Dynamic-v2.1 initial pool is nondeterministic: {unit_id}"
            )
        validate_allocator_state(first["allocator_state_after_preparation"])
        for candidate in first["candidates"]:
            _replay(candidate, config, label=unit_id)

        v0 = initial_dynamic_program_batch(source, (), config, **kwargs)
        shallow = [
            row
            for row in first["proposal_pool"]["candidates"]
            if row["provenance"]["planner_channel"] == SHALLOW_CHANNEL
        ]
        common = min(len(v0["candidates"]), len(shallow))
        if common and [row["endpoint"] for row in shallow[:common]] != [
            row["endpoint"] for row in v0["candidates"][:common]
        ]:
            raise ValueError(f"Dynamic-v2.1 shallow cold stream changed: {unit_id}")
        channels = {row["provenance"]["planner_channel"] for row in first["candidates"]}
        rows.append(
            {
                "unit_id": unit_id,
                "batch_id": first["batch_id"],
                "attempts": len(first["attempts"]),
                "eligible_union": len(first["proposal_pool"]["candidates"]),
                "selected": len(first["candidates"]),
                "selected_channels": sorted(channels),
                "shallow_v0_prefix_checked": common,
                "proposal_seconds": first["proposal_seconds"],
            }
        )

    left = DynamicV21ProgramOptimizer(
        ProgramSearchConfig.program_only_recipe(seed=17),
        source_group="preflight",
        oracle_protocol="zero-oracle",
        hierarchy=None,
    )
    right = DynamicV21ProgramOptimizer(
        ProgramSearchConfig.program_only_recipe(seed=17),
        source_group="preflight",
        oracle_protocol="zero-oracle",
        hierarchy=None,
    )
    right.structured_rng.random(100)
    rng_independent = (
        left.shallow_rng.bit_generator.state == right.shallow_rng.bit_generator.state
    )
    if not rng_independent:
        raise ValueError("structured stream advanced the shallow RNG")

    dummy = [
        {
            "candidate_id": f"shallow-{index}",
            "provenance": {"planner_channel": SHALLOW_CHANNEL},
        }
        for index in range(3)
    ]
    selected, fallback = arbitrate_candidates(
        dummy,
        limit=2,
        state=initial_allocator_state(score_direction="minimize"),
        rng=np.random.default_rng(9),
    )
    if len(selected) != 2 or fallback["selected_by_channel"][STRUCTURED_CHANNEL] != 0:
        raise ValueError("v2.1 empty-channel fallback changed")
    return {
        "schema_version": "t4_dynamic_v21_preflight_v1",
        "passed": len(rows) == len(SELECTED_UNITS),
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "representative_legal_execution": representative,
        "development_cells": rows,
        "rng_streams_independent": rng_independent,
        "empty_channel_fallback": fallback,
        "route_archive_rows": 0,
        "runtime_comparison_inputs": 0,
        "seconds": perf_counter() - began,
        "rdkit": rdBase.rdkitVersion,
        "new_oracle_calls": 0,
        "code_revision": _revision() if code_revision is None else code_revision,
    }


def preflight() -> None:
    output = ROOT / PREFLIGHT
    if output.exists():
        raise ValueError("Dynamic-v2.1 preflight exists; do not overwrite")
    body = build_preflight()
    publish_json(output, body)
    print(json.dumps(body, indent=2))


def refresh_prequery_contract() -> None:
    path = ROOT / CONTRACT
    if (ROOT / PREFLIGHT).exists() or RECEIPT.exists():
        raise ValueError("cannot refresh Dynamic-v2.1 after preflight or launch")
    contract = unseal(path)
    if contract.get("dynamic_v21_development") != v21_contract_payload():
        raise ValueError("Dynamic-v2.1 scientific policy changed before refresh")
    prior = sha256_file(path)
    contract["inputs"] = {
        relative: sha256_file(ROOT / relative) for relative in _material_inputs()
    }
    contract.setdefault("dynamic_v21_prequery_repairs", []).append(
        {
            "reason": "reseal focused pre-oracle implementation identities",
            "prior_contract_sha256": prior,
            "oracle_calls": 0,
            "scientific_policy_changed": False,
        }
    )
    seal(path, contract)
    load_contract(ROOT)
    print(
        json.dumps(
            {
                "status": "prequery_contract_refreshed",
                "prior_contract_sha256": prior,
                "contract_sha256": sha256_file(path),
                "oracle_calls": 0,
            },
            indent=2,
        )
    )


def _task(*, include_preflight=True):
    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    revision = local_image_revision(
        expected_commit=assert_synced(strict=True)["commit"]
    )
    paths = [CONTRACT, APP, "modal_apps/genmol_t4_opt_app.py"]
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
        raise ValueError("Dynamic-v2.1 preflight exists; do not overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("remote preflight requires a clean committed worktree")
    task = _task(include_preflight=False)
    body = modal.Function.from_name(APP_NAME, "structural_preflight").remote(task)
    if body.get("passed") is not True or body.get("new_oracle_calls") != 0:
        raise ValueError("remote Dynamic-v2.1 structural preflight did not pass")
    publish_json(output, body)
    print(json.dumps(body, indent=2))


def launch() -> None:
    import modal

    if RECEIPT.exists():
        raise ValueError("Dynamic-v2.1 launch exists; monitor rather than relaunch")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Dynamic-v2.1 launch requires a clean committed worktree")
    contract = load_contract(ROOT)
    task = _task()
    remote = modal.Function.from_name(APP_NAME, "preflight").remote(task)
    if remote.get("passed") is not True or remote.get("new_oracle_calls") != 0:
        raise ValueError("Dynamic-v2.1 remote zero-oracle preflight failed")
    worker = modal.Function.from_name(APP_NAME, "worker")
    calls = {
        unit: worker.spawn({**task, "unit_id": unit}).object_id
        for unit in SELECTED_UNITS
    }
    saved = {
        "schema_version": "t4_dynamic_v21_launch_v1",
        "task": task,
        "calls": calls,
        "volume": "compose-v4-artifacts",
        "volume_path": f"{KIND}/{task['run_id']}",
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "preflight_sha256": sha256_file(ROOT / PREFLIGHT),
        "new_call_ceiling": contract["dynamic_v21_development"]["new_call_ceiling"],
        "automatic_retries": 0,
    }
    seal(RECEIPT, saved)
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


def _remote_gzip_json(volume, path):
    return json.loads(gzip.decompress(b"".join(volume.read_file(path))))


def _unit_state(volume, saved, unit):
    folder = f"{saved['volume_path']}/units/{unit}"
    try:
        names = {row.path.rsplit("/", 1)[-1] for row in volume.listdir(folder)}
    except __import__("modal").exception.NotFoundError:
        names = set()
    for name in ("result.json", "failure.json"):
        if name in names:
            return _remote_json(volume, f"{folder}/{name}")
    if "checkpoint.json.gz" in names:
        saved_state = _remote_gzip_json(volume, f"{folder}/checkpoint.json.gz")
        champion = saved_state.get("champion")
        return {
            "status": "running",
            "oracle_calls": saved_state.get("query_count", 0),
            "best_score": None if champion is None else champion["score"],
            "curve": saved_state.get("curve", []),
        }
    if "progress.json" in names:
        return _remote_json(volume, f"{folder}/progress.json")
    return {}


def status() -> None:
    import modal

    saved = unseal(RECEIPT)
    volume = modal.Volume.from_name(saved["volume"])
    v1_path = ROOT / "diagnostics/t4_dynamic_v1/launch.json"
    v1_saved = unseal(v1_path) if v1_path.exists() else None
    rows = []
    for unit in SELECTED_UNITS:
        current = _unit_state(volume, saved, unit)
        calls = current.get(
            "oracle_calls", current.get("queries_total", current.get("queries", 0))
        )
        best = current.get("best_score")
        if best is None and current.get("champion") is not None:
            best = current["champion"]["score"]
        v1 = (
            _unit_state(volume, v1_saved, unit)
            if v1_saved and unit in v1_saved["calls"]
            else {}
        )
        v1_best = v1.get("best_score")
        if v1_best is None and v1.get("champion") is not None:
            v1_best = v1["champion"]["score"]
        rows.append(
            {
                "unit": unit,
                "dynamic_v21_status": current.get("status", "pending"),
                "dynamic_v21_calls": calls,
                "dynamic_v21_best": best,
                "dynamic_v1_status": v1.get("status") if v1 else None,
                "dynamic_v1_calls": (
                    v1.get(
                        "oracle_calls", v1.get("queries_total", v1.get("queries", 0))
                    )
                    if v1
                    else None
                ),
                "dynamic_v1_best": v1_best,
            }
        )
    print(
        json.dumps(
            {
                "units": rows,
                "dynamic_v21_total_calls": sum(
                    row["dynamic_v21_calls"] for row in rows
                ),
            },
            indent=2,
        )
    )


def collect() -> None:
    import modal

    if RESULT.exists():
        raise ValueError("Dynamic-v2.1 result exists; do not overwrite")
    saved = unseal(RECEIPT)
    volume = modal.Volume.from_name(saved["volume"])
    rows, calls = [], 0
    for unit in SELECTED_UNITS:
        current = _remote_json(
            volume, f"{saved['volume_path']}/units/{unit}/result.json"
        )
        if current.get("status") != "complete":
            raise ValueError(f"Dynamic-v2.1 unit is not complete: {unit}")
        calls += current["oracle_calls"]
        rows.append(
            {
                "unit": unit,
                "best_score": current["champion"]["score"],
                "oracle_calls": current["oracle_calls"],
                "termination": current["termination"],
                "curve": current["curve"],
            }
        )
    body = {
        "schema_version": "t4_dynamic_v21_result_v1",
        "contract": CONTRACT,
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "launch_receipt_sha256": sha256_file(RECEIPT),
        "new_oracle_calls": calls,
        "new_call_ceiling": 5000,
        "rows": rows,
        "evidence": "winner-informed five-cell Dynamic COMPOSE v2.1 development diagnostic",
        "limitations": [
            "one search replicate per cell",
            "stochastic docking",
            "not held-out benchmark evidence",
        ],
    }
    seal(RESULT, body)
    print(json.dumps({"new_oracle_calls": calls, "rows": rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "refresh-prequery-contract",
            "preflight",
            "remote-preflight",
            "launch",
            "status",
            "collect",
        ),
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "refresh-prequery-contract": refresh_prequery_contract,
        "preflight": preflight,
        "remote-preflight": remote_preflight,
        "launch": launch,
        "status": status,
        "collect": collect,
    }[args.mode]()


if __name__ == "__main__":
    main()
