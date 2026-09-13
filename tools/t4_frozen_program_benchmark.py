"""Prepare, preflight, launch and retrieve the frozen full T4 benchmark."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.control.edit_program_policy import ProgramEntry
from compose_v4.control.program_transfer import initial_program_batch
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_frozen_program_benchmark import (
    APP,
    APP_NAME,
    CONTRACT,
    KIND,
    PREFLIGHT,
    configured,
    load_contract,
    strict_endpoint_scorer,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
REGISTRY = "docs/GENMOL_T4_SEEDS.json"
CENSUS = "diagnostics/ivg_t4_census/census.json"
SEARCH_SEEDS = (20260913, 20260914, 20260915)
DOCKING_SEEDS = (1701, 1702, 1703)
UPSTREAM_REVISION = "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb"
WINNER_PATHS = "diagnostics/ivg_winner_paths/pairs"


def _exact_sources(registry):
    local_index, expected = {}, {}
    for global_index, row in enumerate(registry):
        source_idx = local_index.get(row["target"], 0)
        local_index[row["target"]] = source_idx + 1
        expected[(row["target"], source_idx)] = {
            "global_index": global_index,
            "smiles": row["smiles"],
        }
    found = {}
    for path in sorted((ROOT / WINNER_PATHS).glob("*.json.gz")):
        envelope = json.loads(gzip.decompress(path.read_bytes()))
        if identity(envelope["payload"]) != envelope["payload_sha256"]:
            raise ValueError(f"corrupt exact-state source receipt: {path}")
        state = envelope["payload"]["path"].get("source_state")
        if state is None:
            continue
        for reference in envelope["pair"]["references"]:
            key = (reference["target"], reference["source_idx"])
            if (
                key not in expected
                or key in found
                or envelope["pair"]["source"] != expected[key]["smiles"]
            ):
                continue
            found[key] = {
                **expected[key],
                "source_state": state,
                "source_state_origin": str(path.relative_to(ROOT)),
                "source_state_origin_sha256": sha256_file(path),
            }
    missing = sorted(set(expected) - set(found))
    if missing:
        raise ValueError(f"missing exact saved T4 source states: {missing}")
    return found


def _runtime_hashes():
    transfer = unseal(ROOT / "configs/t4_program_curriculum_lock.json")
    winner = json.loads((ROOT / "configs/t4_winner_refinement.json").read_text())
    result = dict(transfer["runtime_input_sha256"])
    result["/opt/dock/receptors/parp1.pdbqt"] = winner["expected_input_sha256"]["receptor"]
    if result["/opt/dock/qvina02"] != winner["expected_input_sha256"]["qvina02"]:
        raise ValueError("existing T4 locks disagree on the docking binary")
    expected = {
        "/opt/dock/qvina02",
        *{
            f"/opt/dock/receptors/{target}.pdbqt"
            for target in ("parp1", "fa7", "5ht1b", "braf", "jak2")
        },
    }
    if set(result) != expected:
        raise ValueError("existing locks do not cover every full-suite docking input")
    return result


def _ivg_reported(census):
    rows = {}
    for cell in census["cells"]:
        if cell["delta"] != 0.4:
            continue
        key = f"{cell['target']}_{cell['source_idx']}"
        scores = [float(run["reported_docking_score"]) for run in cell["runs"]]
        if len(scores) != 3:
            raise ValueError(f"official IVG cell lacks three search runs: {key}")
        rows[key] = {
            "run_bests": scores,
            "mean": mean(scores),
            "sample_std": stdev(scores),
            "released_rows": cell["row_count"],
        }
    if len(rows) != 15:
        raise ValueError("official delta=0.4 census does not contain all 15 cells")
    return rows


def prepare():
    path = ROOT / CONTRACT
    if path.exists():
        raise ValueError("full-suite contract already exists; do not relock")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("prepare the exact-state lock with RDKit 2024.03.5")
    inputs = {}
    for material in (
        REGISTRY,
        CENSUS,
        LIBRARY,
        "docs/T4_FROZEN_PROGRAM_BENCHMARK.md",
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/program_transfer.py",
        "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
        "modal_apps/genmol_t4_opt_app.py",
        "tools/t4_frozen_program_benchmark.py",
    ):
        inputs[material] = sha256_file(ROOT / material)
    registry = json.loads((ROOT / REGISTRY).read_text())
    census = json.loads((ROOT / CENSUS).read_text())
    library = json.loads((ROOT / LIBRARY).read_text())
    if len(registry) != 15 or len(library) != 146:
        raise ValueError("expected 15 T4 sources and the frozen 146-program library")
    exact_sources = _exact_sources(registry)
    cells = {}
    for (target, source_idx), saved in sorted(exact_sources.items()):
        cell = f"{target}_{source_idx}"
        state = decode_state(saved["source_state"])
        if state.n_real_atoms > 40:
            raise ValueError(f"T4 source exceeds the frozen support: {cell}")
        inputs[saved["source_state_origin"]] = saved["source_state_origin_sha256"]
        cells[cell] = {
            "cell": cell,
            "global_index": saved["global_index"],
            "target": target,
            "source_idx": source_idx,
            "original_seed": saved["smiles"],
            "source_state": saved["source_state"],
            "source_state_sha256": identity(saved["source_state"]),
            "source_state_origin": saved["source_state_origin"],
            "source_state_origin_sha256": saved["source_state_origin_sha256"],
        }
    runtime = _runtime_hashes()
    base = replace(
        ProgramSearchConfig.program_only_recipe(seed=0),
        attempts_per_batch=128,
        candidates_per_batch=16,
        wall_seconds=45.0,
        proposal_cache_entries=128,
        mutation_sampling="random",
        composition_probability=0.0,
    )
    units = []
    for replicate, (controller_seed, docking_seed) in enumerate(
        zip(SEARCH_SEEDS, DOCKING_SEEDS, strict=True)
    ):
        for cell, row in cells.items():
            domain = {
                "schema_version": "t4_strict_delta04_program_oracle_v1",
                "cell": cell,
                "target": row["target"],
                "source_idx": row["source_idx"],
                "original_seed": row["original_seed"],
                "strict_filters": {"similarity_gt": 0.4, "qed_gt": 0.6, "sa_lt": 4.0},
                "fingerprint": {"kind": "Morgan", "radius": 2, "bits": 2048},
                "qvina02_sha256": runtime["/opt/dock/qvina02"],
                "receptor_sha256": runtime[f"/opt/dock/receptors/{row['target']}.pdbqt"],
                "box_definition": "BOXES in hash-bound modal_apps/genmol_t4_opt_app.py",
                "docking": {
                    "cpu": 1,
                    "exhaustiveness": 1,
                    "num_modes": 10,
                    "ligand_preparation": "Open Babel --gen3d, unchanged production wrapper",
                },
            }
            units.append(
                {
                    "unit_id": f"{cell}_r{replicate}",
                    "cell": cell,
                    "target": row["target"],
                    "source_idx": row["source_idx"],
                    "original_seed": row["original_seed"],
                    "replicate": replicate,
                    "controller_seed": controller_seed,
                    "docking_seed": docking_seed,
                    "oracle_protocol": identity(domain),
                    "oracle_domain": domain,
                    "budget": 1000,
                }
            )
    cpu_price, memory_price = 0.0000131, 0.00000222
    worker_timeout = 4 * 3600
    worst_case = (
        45 * worker_timeout * (cpu_price + 4 * memory_price)
        + 8 * 3600 * (cpu_price + 2 * memory_price)
        + 30 * 900 * (cpu_price + 4 * memory_price)
    )
    contract = {
        "schema_version": "t4_frozen_program_benchmark_v2",
        "delta": 0.4,
        "cells": cells,
        "units": units,
        "search_replicates": 3,
        "calls_per_unit": 1000,
        "search_call_ceiling": 45000,
        "confirmation_call_ceiling": 30,
        "total_call_ceiling": 45030,
        "confirmation_seeds": [9101, 9102],
        "cold_start_seed": 20260913,
        "max_rounds": 128,
        "max_consecutive_empty_rounds": 3,
        "plateau_stop": {
            "minimum_calls": 500,
            "window_calls": 250,
            "minimum_improvement": 0.3,
            "requires_best_at_or_better_than_ivg_mean": True,
        },
        "library_path": LIBRARY,
        "library_programs": len(library),
        "controller": asdict(base),
        "runtime_input_sha256": runtime,
        "inputs": inputs,
        "ivg_reported": _ivg_reported(census),
        "upstream_protocol": {
            "repository": "https://github.com/invirtuolabs/InVirtuoGen_results",
            "revision": UPSTREAM_REVISION,
            "license": "CC-BY-NC-SA-4.0 public academic development input",
            "readme_blob": "4fa973f25aee7aebf83e7829940aa0e6ba3f83e2",
            "results_table_blob": "675598a086167acb04ed81ac4b083b2442eb35fa",
            "ppo_docking_blob": "d5ed4ace693ee36aef1a7065e3a380ef4f67add0",
            "verified_budget": "README command --max_oracle_calls 1000",
            "verified_aggregation": "best strict-eligible candidate grouped by three run seeds",
            "implementation_gap": "pinned ppo_docking.py blob contains an unrelated PMO table script",
        },
        "information_regime": {
            "development": "public T4 winner routes contributed to one shared frozen program library",
            "inference": "no cell-specific route lookup, task label, endpoint template or controller change",
            "evaluation": "prospective after one shared controller/configuration lock",
        },
        "container_limit": 30,
        "reserved_usd": 20,
        "compute": {
            "driver_containers": 1,
            "worker_max_containers": 29,
            "cpu_per_worker": 1,
            "worker_memory_gib": 4,
            "worker_timeout_seconds": worker_timeout,
            "driver_timeout_seconds": 8 * 3600,
            "confirmation_timeout_seconds": 900,
            "gpu": False,
            "automatic_retries": 0,
            "price_source": "https://modal.com/pricing",
            "cpu_core_second_usd": cpu_price,
            "gib_second_usd": memory_price,
            "worst_case_timeout_reservation_usd": worst_case,
            "expected_wall_hours": [1.5, 5.0],
        },
        "limitations": [
            "The shared controller library used public T4 winner routes during development.",
            "The upstream release states the budget and supplies result rows but its pinned ppo_docking.py is mispackaged, so exact internal accounting cannot be independently replayed.",
            "Docking is stochastic; first-score benchmark results and fresh confirmations are reported separately.",
            "COMPOSE editing support excludes stereochemistry and formal-charge changes and caps active heavy atoms at 40.",
        ],
    }
    if worst_case >= contract["reserved_usd"]:
        raise ValueError("timeout reservation exceeds the user-approved compute ceiling")
    seal(path, contract)
    print(
        json.dumps(
            {
                "contract": str(path),
                "contract_sha256": sha256_file(path),
                "cells": len(cells),
                "units": len(units),
                "call_ceiling": contract["total_call_ceiling"],
                "worst_case_timeout_reservation_usd": worst_case,
                "new_oracle_calls": 0,
            },
            indent=2,
        )
    )


def preflight():
    output = ROOT / PREFLIGHT
    if output.exists():
        raise ValueError("full-suite preflight already exists; do not overwrite")
    contract = load_contract(ROOT)
    library_rows = json.loads((ROOT / contract["library_path"]).read_text())
    entries = tuple(
        ProgramEntry(EditProgram.from_payload(row["program"]), tuple(row["source_groups"]))
        for row in library_rows
    )
    began, results = perf_counter(), []
    for cell in sorted(contract["cells"]):
        row = contract["cells"][cell]
        unit = next(
            unit for unit in contract["units"] if unit["cell"] == cell and unit["replicate"] == 0
        )
        config = replace(configured(contract, unit), candidates_per_batch=16)
        source = decode_state(row["source_state"])
        arguments = {
            "source_group": identity(
                {
                    "target": row["target"],
                    "source_idx": row["source_idx"],
                    "seed": row["original_seed"],
                }
            ),
            "oracle_protocol": unit["oracle_protocol"],
            "eligibility": strict_endpoint_scorer(row["original_seed"]),
        }
        first = initial_program_batch(source, entries, config, **arguments)
        second = initial_program_batch(source, entries, config, **arguments)
        if first["batch_id"] != second["batch_id"]:
            raise ValueError(f"cold-start preparation is nondeterministic: {cell}")
        if not first["candidates"]:
            raise ValueError(f"frozen controller has zero initial yield: {cell}")
        for candidate in first["candidates"]:
            program = EditProgram.from_payload(candidate["program"])
            _, replay = execute_program_graph(
                decode_state(candidate["source_state"]),
                compile_program_graph(program),
                tuple(candidate["assignment"]),
                max_primitives=config.max_primitives,
                max_blocks=config.max_blocks,
            )
            if replay != candidate["trace"] or replay["endpoint"] != candidate["endpoint"]:
                raise ValueError(f"cold-start candidate failed exact replay: {cell}")
            if not arguments["eligibility"]({"smiles": candidate["endpoint"]})["oracle_eligible"]:
                raise ValueError(f"cold-start candidate lost strict eligibility: {cell}")
        results.append(
            {
                "cell": cell,
                "candidates": len(first["candidates"]),
                "attempts": len(first["attempts"]),
                "proposal_seconds_first": first["proposal_seconds"],
                "proposal_seconds_repeat": second["proposal_seconds"],
                "batch_id": first["batch_id"],
            }
        )
        print(json.dumps(results[-1]), flush=True)
    contract_sha = sha256_file(ROOT / CONTRACT)
    body = {
        "schema_version": "t4_frozen_program_preflight_v1",
        "passed": len(results) == 15 and all(row["candidates"] > 0 for row in results),
        "contract_sha256": contract_sha,
        "cells": results,
        "seconds": perf_counter() - began,
        "rdkit": rdBase.rdkitVersion,
        "new_oracle_calls": 0,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    publish_json(output, {**body, "result_sha256": identity(body)})
    print(json.dumps({"passed": body["passed"], "seconds": body["seconds"]}, indent=2))


def launch(receipt):
    import modal

    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    if receipt.exists():
        raise ValueError("launch receipt already exists; monitor it instead of spawning again")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("full-suite launch requires clean committed source")
    contract = load_contract(ROOT)
    revision = local_image_revision(expected_commit=assert_synced(strict=True)["commit"])
    files = {
        path: sha256_file(ROOT / path)
        for path in (CONTRACT, PREFLIGHT, APP, "modal_apps/genmol_t4_opt_app.py")
    }
    body = {"image_revision": revision, "files_sha256": files}
    task = {**body, "run_id": identity(body)}
    call = modal.Function.from_name(APP_NAME, "run").spawn(task)
    saved = {
        "task": task,
        "call_id": call.object_id,
        "volume": "compose-v4-artifacts",
        "volume_path": f"{KIND}/{task['run_id']}",
        "call_ceiling": contract["total_call_ceiling"],
        "reserved_usd": contract["reserved_usd"],
    }
    publish_json(receipt, saved)
    print(json.dumps({key: value for key, value in saved.items() if key != "task"}, indent=2))


def retrieve(receipt, output):
    import modal

    saved = json.loads(receipt.read_text())
    result = modal.FunctionCall.from_id(saved["call_id"]).get(timeout=0)
    seal(output, result)
    print(json.dumps(result["summary"], indent=2))


def _remote_json(volume, path):
    raw = json.loads(b"".join(volume.read_file(path)))
    if set(raw) == {"payload", "payload_sha256"}:
        payload = raw["payload"]
        if identity(payload) != raw["payload_sha256"]:
            raise ValueError(f"remote sealed artifact is corrupt: {path}")
        return payload
    return raw


def status(receipt):
    """Read durable per-unit progress without starting another container."""
    import modal

    saved = json.loads(receipt.read_text())
    contract = load_contract(ROOT)
    volume = modal.Volume.from_name(saved["volume"])
    prefix = saved["volume_path"].strip("/")
    units, total_calls = [], 0
    for unit in contract["units"]:
        unit_prefix = f"{prefix}/units/{unit['unit_id']}"
        try:
            names = {entry.path.rsplit("/", 1)[-1] for entry in volume.listdir(unit_prefix)}
        except modal.exception.NotFoundError:
            names = set()
        filename = next(
            (name for name in ("result.json", "failure.json", "progress.json") if name in names),
            None,
        )
        row = {} if filename is None else _remote_json(volume, f"{unit_prefix}/{filename}")
        calls = row.get("oracle_calls", row.get("queries_total", row.get("queries", 0)))
        best = row.get("best_score")
        if best is None and row.get("champion") is not None:
            best = row["champion"]["score"]
        ivg = contract["ivg_reported"][unit["cell"]]["mean"]
        total_calls += calls
        units.append(
            {
                "unit": unit["unit_id"],
                "status": row.get("status", "running" if filename else "pending"),
                "termination": row.get("termination"),
                "calls": calls,
                "best_score": best,
                "ivg_mean": ivg,
                "margin_vs_ivg": None if best is None else best - ivg,
                "calls_to_ivg_mean": row.get("calls_to_ivg_mean"),
                "updated_at_utc": row.get("at", row.get("completed_at_utc")),
            }
        )
    summary = {
        "call_id": saved["call_id"],
        "total_calls": total_calls,
        "completed_units": sum(row["status"] == "complete" for row in units),
        "failed_units": sum(row["status"] == "failed" for row in units),
        "running_units": sum(row["status"] == "running" for row in units),
        "units_at_or_better_than_ivg": sum(
            row["margin_vs_ivg"] is not None and row["margin_vs_ivg"] <= 0 for row in units
        ),
        "units": units,
    }
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "launch", "status", "retrieve"))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "diagnostics/t4_frozen_program_benchmark/launch.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics/t4_frozen_program_benchmark/result.json",
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "launch":
        launch(args.receipt)
    elif args.mode == "status":
        status(args.receipt)
    else:
        retrieve(args.receipt, args.output)


if __name__ == "__main__":
    main()
