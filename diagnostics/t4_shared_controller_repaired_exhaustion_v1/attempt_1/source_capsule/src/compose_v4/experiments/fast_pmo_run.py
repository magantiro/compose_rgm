"""One explicitly metered program-only PMO run; no reference-model loading."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import resource
import threading
from dataclasses import asdict, replace
from time import perf_counter, process_time

from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_policy import ProgramEntry
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask, archive_top_k
from compose_v4.experiments.continuation_profile import publish_json, verify_file
from compose_v4.experiments.parent_edit_cycles import PMO_ORACLE_SHA256, pmo_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

KIND = "fast_pmo_run"
CONTRACT = f"configs/{KIND}.json"
APP = f"modal_apps/{KIND}_app.py"
APP_NAME = "compose-fast-pmo"


def configuration():
    return replace(
        ProgramSearchConfig.program_only_recipe(seed=20260921, score_direction="maximize"),
        proposal_cache_entries=128,
        candidates_per_batch=16,
        attempts_per_batch=128,
        wall_seconds=45.0,
    )


def load_contract(root):
    c = unseal(root / CONTRACT)
    expected = {
        "name": "albuterol_similarity",
        "seed": 20260921,
        "budget": 128,
        "rounds": 7,
        "queries_per_round": 16,
        "initialization_count": 16,
        "initialization_mode": "all_scored_pool",
        "initial_parent_fraction": 0.2,
        "max_seconds": 420,
    }
    if c["run"] != expected or c["configuration"] != json.loads(
        json.dumps(asdict(configuration()))
    ):
        raise ValueError("fast PMO recipe differs from the bounded approved run")
    if c["compute"] != {
        "cpu": 1,
        "memory_mib": 4096,
        "containers": 1,
        "timeout_seconds": 600,
        "gpu": False,
        "retries": 0,
        "reserved_usd": 1,
    }:
        raise ValueError("fast PMO resources differ from the bounded approval")
    for path, digest in c["inputs"].items():
        verify_file(root / path, digest)
    initialized = json.loads((root / c["initialization_path"]).read_text())
    if len(initialized["candidates"]) != c["run"]["initialization_count"]:
        raise ValueError("initialization count differs from the query reservation")
    return c


def execute(c, root, folder, *, evaluate, flush, progress):
    """Shared campaign adapter; the remote boundary qualifies the real oracle."""
    recipe = c["run"]
    pt = ProgramTask(recipe["name"], identity(c["oracle_protocol"]), "pmo")
    initialized = json.loads((root / c["initialization_path"]).read_text())
    library = tuple(
        ProgramEntry(EditProgram.from_payload(r["program"]), tuple(r["source_groups"]))
        for r in json.loads((root / c["library_path"]).read_text())
    )
    ledger = ProgramQueryLedger(
        folder / "oracle", pt, evaluate, budget=recipe["budget"], flush=flush
    )
    result = run_program_campaign(
        output=folder / "campaign",
        task=pt,
        config=ProgramSearchConfig(**c["configuration"]),
        initialization=initialized,
        library=library,
        ledger=ledger,
        rounds=recipe["rounds"],
        queries_per_round=recipe["queries_per_round"],
        hierarchy=None,
        fit_model=None,
        stagnation_rounds=None,
        bootstrap_rounds=1,
        initialization_mode=recipe["initialization_mode"],
        initial_parent_fraction=recipe["initial_parent_fraction"],
        max_seconds=recipe["max_seconds"],
        progress=progress,
    )
    history = result["history"]
    reason = (
        "query_budget"
        if not ledger.remaining
        else "empty_proposal_pool"
        if history and history[-1]["queried_candidates"] == 0
        else "round_limit"
        if len(history) == recipe["rounds"]
        else "time_limit"
    )
    curve = []
    for index, row in enumerate(ledger.rows):
        observations = [(r["endpoint"], r["utility"]) for r in ledger.rows[: index + 1]]
        curve.append(
            {
                "queries": index + 1,
                "best": max(u for _, u in observations),
                "top10": archive_top_k(observations, k=10),
            }
        )
    return {
        **result,
        "rows": ledger.rows,
        "query_curve": curve,
        "stop_reason": reason,
        "oracle_seconds": sum(r["seconds"] for r in ledger.rows),
        "proposal_seconds": sum(r["proposal_seconds"] for r in history),
        "reference_calls": 0,
        "model_fits": 0,
        "library_programs": len(library),
    }


def run_remote(task, root, artifacts, volume, validate_revision):
    validate_revision(task["image_revision"])
    for path, digest in task["files_sha256"].items():
        verify_file(root / path, digest)
    if set(task["files_sha256"]) != {CONTRACT, APP}:
        raise ValueError("launch must identify contract and app bytes")
    if task["run_id"] != identity({k: v for k, v in task.items() if k != "run_id"}):
        raise ValueError("fast PMO launch identity changed")
    c = load_contract(root)
    volume.reload()
    folder = artifacts / KIND / task["run_id"]
    if (folder / "result.json").exists():
        return unseal(folder / "result.json")
    if (folder / "started.json").exists():
        raise RuntimeError("started fast PMO run requires manual receipt inspection, not relaunch")
    seal(folder / "started.json", {"task": task, "contract": c, "at": _stamp()})
    volume.commit()
    began, cpu_began = perf_counter(), process_time()
    lock, stop = threading.RLock(), threading.Event()
    latest = {"phase": "runtime", "oracle_calls": 0}

    def flush():
        with lock:
            volume.commit()

    def report(row):
        with lock:
            latest.update(row)
            publish_json(folder / "progress.json", {**latest, "at": _stamp()})
            print(json.dumps(row), flush=True)
            volume.commit()

    def heartbeat():
        while not stop.wait(30):
            report({"elapsed_seconds": perf_counter() - began})

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        if rdBase.rdkitVersion != "2024.03.5" or importlib.metadata.version("PyTDC") != "0.3.6":
            raise ValueError("oracle chemistry/package version mismatch")
        evaluate = pmo_oracle(c["run"]["name"], {c["run"]["name"]: c["oracle_protocol"]})
        from pathlib import Path

        import tdc.chem_utils.oracle.oracle as oracle_module

        verify_file(Path(oracle_module.__file__), PMO_ORACLE_SHA256)
        report({"phase": "oracle_qualified", "oracle_module_sha256": PMO_ORACLE_SHA256})
        result = execute(c, root, folder, evaluate=evaluate, flush=flush, progress=report)
        result.update(
            schema_version="fast_pmo_result_v1",
            task=task,
            contract=c,
            seconds=perf_counter() - began,
            cpu_seconds=process_time() - cpu_began,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            software={k: importlib.metadata.version(k) for k in ("PyTDC", "numpy", "scipy")}
            | {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "machine": platform.machine(),
            },
            completed_at_utc=_stamp(),
        )
        seal(folder / "result.json", result)
        flush()
        return result
    except Exception as error:
        # Durable failure, not a substituted result or retry. Original traceback propagates.
        seal(
            folder / "failure.json",
            {"error": repr(error), "task": task, "at": _stamp(), "automatic_retry": False},
        )
        flush()
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
