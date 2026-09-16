"""Restart-safe Stage-20 runner for the frozen hybrid top-three controller."""

from __future__ import annotations

import gzip
import json
import platform
import subprocess
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    DynamicProgramOptimizer,
    initial_dynamic_program_batch,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import (
    canonical_bytes,
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "t4_hybrid_top3_v0_stage20_launch_contract_v1"
CONTRACT_PATH = "configs/t4_hybrid_top3_v0_stage20_launch_v1.json"
KIND = "t4_hybrid_top3_v0_stage20"
STAGE_CEILING = 20
FULL_FROZEN_CEILING = 100
REPORT_CALLS = (1, 5, 10, 20)


def _read_self_hashed(
    path: Path, *, expected_physical: str, expected_payload: str
) -> dict:
    if sha256_file(path) != expected_physical:
        raise ValueError(f"bound physical input changed: {path}")
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256", envelope.get("contract_sha256"))
    if (
        not isinstance(payload, dict)
        or claimed != expected_payload
        or identity(payload) != claimed
    ):
        raise ValueError(f"bound payload input changed: {path}")
    return payload


def load_contract(root: Path) -> dict:
    path = root / CONTRACT_PATH
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError("invalid Stage-20 launch contract")
    protocol = payload["protocol"]
    expected = {
        "stage_call_ceiling_per_cell": STAGE_CEILING,
        "full_frozen_run_call_ceiling_per_cell": FULL_FROZEN_CEILING,
        "stage_total_call_ceiling": 100,
        "report_calls": list(REPORT_CALLS),
        "maximum_concurrent_single_cpu_workers": 5,
        "gpu": False,
        "automatic_retry": False,
        "replacement": False,
        "backfill": False,
        "confirmation_calls": 0,
        "plateau_stopping": False,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage-20 execution scope changed")
    locks = payload["immutable_inputs"]
    for name in (
        "hybrid_contract",
        "candidate_lock",
        "admission_lock",
        "qualification_v2",
    ):
        spec = locks[name]
        _read_self_hashed(
            root / spec["path"],
            expected_physical=spec["physical_sha256"],
            expected_payload=spec["payload_sha256"],
        )
    source_spec = locks["source_protocol"]
    source = _read_self_hashed(
        root / source_spec["path"],
        expected_physical=source_spec["physical_sha256"],
        expected_payload=source_spec["payload_sha256"],
    )
    for name in (
        "dynamic_v0_implementation",
        "macro_archive_integration",
        "scoring_implementation",
    ):
        spec = locks[name]
        if sha256_file(root / spec["path"]) != spec["physical_sha256"]:
            raise ValueError(f"bound implementation changed: {name}")
    if (
        source.get("delta") != 0.4
        or source.get("controller", {}).get("proposal_mode") != "program_only"
    ):
        raise ValueError(
            "source protocol is not the frozen delta-0.4 Dynamic-v0 recipe"
        )
    selected = [row for row in source["units"] if row["unit_id"] in payload["units"]]
    if [row["unit_id"] for row in selected] != payload["units"]:
        by_id = {row["unit_id"]: row for row in selected}
        selected = [by_id[name] for name in payload["units"]]
    if len(selected) != 5 or any(row["replicate"] != 0 for row in selected):
        raise ValueError("Stage-20 source unit census changed")
    return {"launch": payload, "source": source, "units": selected}


def _locks(root: Path, contract: dict) -> tuple[dict, dict]:
    inputs = contract["launch"]["immutable_inputs"]
    candidate = _read_self_hashed(
        root / inputs["candidate_lock"]["path"],
        expected_physical=inputs["candidate_lock"]["physical_sha256"],
        expected_payload=inputs["candidate_lock"]["payload_sha256"],
    )
    admission = _read_self_hashed(
        root / inputs["admission_lock"]["path"],
        expected_physical=inputs["admission_lock"]["physical_sha256"],
        expected_payload=inputs["admission_lock"]["payload_sha256"],
    )
    return candidate, admission


def run_identity(contract: dict) -> str:
    inputs = contract["launch"]["immutable_inputs"]
    return identity(
        {
            "schema_version": "t4_hybrid_top3_v0_full_run_identity_v1",
            "cells": contract["launch"]["cells"],
            "units": contract["launch"]["units"],
            "candidate_lock": inputs["candidate_lock"],
            "admission_lock": inputs["admission_lock"],
            "qualification_v2": inputs["qualification_v2"],
            "full_frozen_call_ceiling_per_cell": FULL_FROZEN_CEILING,
        }
    )


def _validate_task(
    task: dict, root: Path, validate_revision: Callable[[dict], None]
) -> dict:
    validate_revision(task["image_revision"])
    contract = load_contract(root)
    if task.get("run_id") != run_identity(contract):
        raise ValueError("Stage-20 run identity changed")
    if task.get("stage_call_ceiling") != STAGE_CEILING:
        raise ValueError("worker stage ceiling changed")
    material = task.get("files_sha256")
    if not isinstance(material, dict) or not material:
        raise ValueError("worker lacks exact launch-file identities")
    for relative, digest in material.items():
        verify_file(root / relative, digest)
    base = {
        key: value for key, value in task.items() if key not in {"unit_id", "stage_id"}
    }
    expected_stage = identity(
        {
            "run_id": task["run_id"],
            "stage_call_ceiling": STAGE_CEILING,
            "image_revision": task["image_revision"],
            "files_sha256": material,
        }
    )
    if task.get("stage_id") != expected_stage or base.get("stage_id") is not None:
        raise ValueError("Stage-20 invocation identity changed")
    return contract


def remote_preflight(
    task: dict, root: Path, validate_revision: Callable[[dict], None]
) -> dict:
    contract = _validate_task(task, root, validate_revision)
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("Stage-20 chemistry runtime changed")
    assets = contract["launch"]["runtime_assets"]
    verify_file(Path("/opt/dock/qvina02"), assets["qvina02_sha256"])
    verified = {}
    for target, digest in assets["receptor_sha256"].items():
        path = Path("/opt/dock/receptors") / f"{target}.pdbqt"
        verified[target] = verify_file(path, digest)
    candidate, admission = _locks(root, contract)
    schedules = {row["cell"]: row for row in candidate["cells"]}
    admissions = {row["cell"]: row for row in admission["cells"]}
    if set(schedules) != set(contract["launch"]["cells"]) or set(admissions) != set(
        schedules
    ):
        raise ValueError("Stage-20 lock census changed")
    for cell in schedules:
        if (
            schedules[cell]["initial_macro_calls"]
            != admissions[cell]["initial_macro_calls"]
        ):
            raise ValueError(f"candidate/admission schedule differs: {cell}")
        for call in schedules[cell]["initial_macro_calls"]:
            source = decode_state(call["candidate"]["source_state"])
            endpoint, trace = execute_program(source, call["candidate"]["actions"])
            if (
                trace["endpoint"] != call["canonical_smiles"]
                or encode_state(endpoint) != call["candidate"]["endpoint_state"]
            ):
                raise ValueError(f"locked macro replay changed: {cell}/{call['call']}")
    return {
        "schema_version": "t4_hybrid_top3_v0_stage20_remote_preflight_v1",
        "status": "PASS_NO_DOCKING",
        "run_id": task["run_id"],
        "stage_id": task["stage_id"],
        "cells": contract["launch"]["cells"],
        "rdkit": rdBase.rdkitVersion,
        "qvina02_sha256": assets["qvina02_sha256"],
        "receptor_sha256": verified,
        "docking_calls": 0,
    }


def _write_gzip(path: Path, payload: dict) -> str:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    raw = canonical_bytes(envelope) + b"\n"
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(compressed)
    temporary.replace(path)
    return sha256_file(path)


def _read_gzip(path: Path) -> dict:
    envelope = json.loads(gzip.decompress(path.read_bytes()))
    payload = envelope["payload"]
    if envelope.get("payload_sha256") != identity(payload):
        raise ValueError(f"corrupt Stage-20 compressed artifact: {path}")
    return payload


def _macro_batch(schedule: dict, *, source_group: str, oracle_protocol: str) -> dict:
    candidates, attempts = [], []
    for locked in schedule["initial_macro_calls"]:
        raw = locked["candidate"]
        source = decode_state(raw["source_state"])
        endpoint, trace = execute_program(source, raw["actions"])
        program, assignment = extract_program(source, [trace])
        if (
            trace["endpoint"] != locked["canonical_smiles"]
            or encode_state(endpoint) != raw["endpoint_state"]
        ):
            raise ValueError("frozen macro changed during launch conversion")
        candidate = {
            "source_group": source_group,
            "oracle_protocol": oracle_protocol,
            "source_state": raw["source_state"],
            "program": program.payload(),
            "assignment": list(assignment),
            "trace": trace,
            "endpoint": trace["endpoint"],
            "provenance": {
                "channel": "frozen_hybrid_macro_bootstrap",
                "locked_call": locked["call"],
                "locked_candidate_id": locked["candidate_id"],
                "selection_role": locked["selection_role"],
                "source_case_id": locked["source_case_id"],
            },
            "score": None,
            "candidate_id": locked["candidate_id"],
        }
        candidates.append(candidate)
        attempts.append(
            {
                "attempt": locked["call"] - 1,
                "status": "eligible",
                "channel": "frozen_hybrid_macro_bootstrap",
                "endpoint": locked["canonical_smiles"],
                "provenance": candidate["provenance"],
            }
        )
    body = {
        "schema_version": "t4_hybrid_top3_bootstrap_batch_v1",
        "source_group": source_group,
        "oracle_protocol": oracle_protocol,
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "candidates": candidates,
        "attempts": attempts,
        "new_oracle_calls": 0,
    }
    return {**body, "batch_id": identity(body), "proposal_seconds": 0.0}


def _configure_refinement_rng(
    search: DynamicProgramOptimizer, words: list[int]
) -> None:
    if len(words) != 4 or any(
        type(word) is not int or not 0 <= word < 2**32 for word in words
    ):
        raise ValueError("invalid frozen Dynamic-v0 refinement RNG namespace")
    search.rng = np.random.default_rng(np.random.SeedSequence(words))


def _checkpoint(
    path: Path, *, search, next_round, rows, curve, champion, empty_rounds
) -> dict:
    if search is not None:
        search.history.clear()
    payload = {
        "schema_version": "t4_hybrid_top3_v0_resume_checkpoint_v1",
        "next_round": next_round,
        "query_count": len(rows),
        "search": None if search is None else search.snapshot(include_history=False),
        "curve": curve,
        "champion": champion,
        "consecutive_empty_rounds": empty_rounds,
        "updated_at_utc": _stamp(),
    }
    _write_gzip(path, payload)
    return payload


def _summary_points(curve: list[dict]) -> dict[str, float | None]:
    return {
        str(call): curve[call - 1]["best_score"] if len(curve) >= call else None
        for call in REPORT_CALLS
    }


def run_unit(
    task: dict,
    root: Path,
    artifacts: Path,
    volume: Any,
    validate_revision: Callable[[dict], None],
    dock: Callable[[str, str, str, int], float | None],
) -> dict:
    contract = _validate_task(task, root, validate_revision)
    unit = next(
        (row for row in contract["units"] if row["unit_id"] == task.get("unit_id")),
        None,
    )
    if unit is None:
        raise ValueError("worker unit is outside the exact five-cell census")
    candidate_lock, admission_lock = _locks(root, contract)
    schedule = next(
        row for row in candidate_lock["cells"] if row["cell"] == unit["cell"]
    )
    admission = next(
        row for row in admission_lock["cells"] if row["cell"] == unit["cell"]
    )
    if schedule["initial_macro_calls"] != admission["initial_macro_calls"]:
        raise ValueError("candidate and admission locks disagree")

    volume.reload()
    folder = artifacts / KIND / task["run_id"] / "units" / unit["unit_id"]
    stage_result_path = folder / "stages" / "q20" / "result.json"
    if stage_result_path.exists():
        return unseal(stage_result_path)
    assets = contract["launch"]["runtime_assets"]
    verify_file(Path("/opt/dock/qvina02"), assets["qvina02_sha256"])
    verify_file(
        Path("/opt/dock/receptors") / f"{unit['target']}.pdbqt",
        assets["receptor_sha256"][unit["target"]],
    )
    source_contract = contract["source"]
    source = decode_state(source_contract["cells"][unit["cell"]]["source_state"])
    config = benchmark.configured(source_contract, unit)
    endpoint_score = benchmark.strict_endpoint_scorer(unit["original_seed"], delta=0.4)
    source_group = identity(
        {
            "target": unit["target"],
            "source_idx": unit["source_idx"],
            "seed": unit["original_seed"],
        }
    )
    oracle_protocol = unit["oracle_protocol"]
    run_started = folder / "started.json"
    run_record = {
        "schema_version": "t4_hybrid_top3_v0_run_start_v1",
        "run_id": task["run_id"],
        "unit": unit,
        "candidate_lock_payload_sha256": contract["launch"]["immutable_inputs"][
            "candidate_lock"
        ]["payload_sha256"],
        "admission_lock_payload_sha256": contract["launch"]["immutable_inputs"][
            "admission_lock"
        ]["payload_sha256"],
    }
    if run_started.exists():
        prior = unseal(run_started)
        if any(prior.get(key) != value for key, value in run_record.items()):
            raise ValueError("resume changed the frozen full-run identity")
    else:
        seal(run_started, {**run_record, "started_at_utc": _stamp()})
        volume.commit()
    stage_started = folder / "stages" / "q20" / "started.json"
    if not stage_started.exists():
        seal(
            stage_started,
            {
                "schema_version": "t4_hybrid_top3_v0_stage_start_v1",
                "run_id": task["run_id"],
                "stage_id": task["stage_id"],
                "unit_id": unit["unit_id"],
                "stage_call_ceiling": STAGE_CEILING,
                "started_at_utc": _stamp(),
            },
        )
        volume.commit()

    rows = benchmark._query_rows(folder)
    checkpoint_path = folder / "checkpoint.json.gz"
    if checkpoint_path.exists():
        saved = _read_gzip(checkpoint_path)
        if saved["query_count"] > len(rows) or saved["query_count"] != len(
            saved["curve"]
        ):
            raise ValueError("resume checkpoint disagrees with durable ledger")
        search = (
            None
            if saved["search"] is None
            else DynamicProgramOptimizer.restore(saved["search"], hierarchy=None)
        )
        next_round = saved["next_round"]
        curve = saved["curve"]
        champion = saved["champion"]
        empty_rounds = saved["consecutive_empty_rounds"]
    else:
        search, next_round, curve, champion, empty_rounds = None, 0, [], None, 0
    checkpoint_query_count = len(curve)
    if checkpoint_query_count > len(rows) or len(rows) > STAGE_CEILING:
        raise ValueError("durable Stage-20 state exceeds its bound")
    began = perf_counter()
    heartbeat_lock, stop = threading.RLock(), threading.Event()

    def heartbeat() -> None:
        while not stop.wait(30):
            with heartbeat_lock:
                publish_json(
                    folder / "progress.json",
                    {
                        "unit_id": unit["unit_id"],
                        "stage": 20,
                        "queries": len(rows),
                        "round": next_round,
                        "best_score": None if champion is None else champion["score"],
                        "summary_points": _summary_points(curve),
                        "seconds_this_invocation": perf_counter() - began,
                        "at": _stamp(),
                    },
                )
                volume.commit()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    termination = None
    try:
        while (
            len(rows) < STAGE_CEILING
            and next_round < source_contract["max_rounds"]
            and empty_rounds < source_contract["max_consecutive_empty_rounds"]
        ):
            remaining = STAGE_CEILING - len(rows)
            round_folder = folder / "rounds" / f"round_{next_round:04d}"
            batch_path = round_folder / "batch.json.gz"
            bootstrap = search is None or not search.entries
            if batch_path.exists():
                locked = _read_gzip(batch_path)
                if locked["round"] != next_round or locked["bootstrap"] != bootstrap:
                    raise ValueError("resumed round lock changed")
                batch = locked["batch"]
                if not bootstrap:
                    benchmark._restore_pending(search, batch)
            else:
                if bootstrap:
                    if schedule["initial_macro_calls"]:
                        batch = _macro_batch(
                            schedule,
                            source_group=source_group,
                            oracle_protocol=oracle_protocol,
                        )
                    else:
                        batch = initial_dynamic_program_batch(
                            source,
                            (),
                            replace(
                                config,
                                seed=source_contract["cold_start_seed"],
                                candidates_per_batch=min(
                                    config.candidates_per_batch, remaining
                                ),
                            ),
                            source_group=source_group,
                            oracle_protocol=oracle_protocol,
                            eligibility=endpoint_score,
                        )
                else:
                    batch = search.propose_batch(endpoint_score)
                    if len(batch["candidates"]) > remaining:
                        ids = [
                            row["candidate_id"]
                            for row in batch["candidates"][:remaining]
                        ]
                        batch = search.lock_query_subset(
                            batch["batch_id"],
                            ids,
                            {
                                "policy": "stage20_query_budget_prefix_v1",
                                "selected_ids": ids,
                                "available": len(batch["candidates"]),
                                "remaining_stage_budget": remaining,
                            },
                        )
                _write_gzip(
                    batch_path,
                    {"round": next_round, "bootstrap": bootstrap, "batch": batch},
                )
                if not bootstrap:
                    _checkpoint(
                        checkpoint_path,
                        search=search,
                        next_round=next_round,
                        rows=rows,
                        curve=curve,
                        champion=champion,
                        empty_rounds=empty_rounds,
                    )
                volume.commit()
            outcomes = []
            for candidate_index, candidate in enumerate(batch["candidates"]):
                query_index = checkpoint_query_count + candidate_index
                if query_index > len(rows):
                    raise ValueError("query ledger has a gap inside a locked round")
                result = benchmark._dock_candidate(
                    folder=folder,
                    candidate=candidate,
                    query_index=query_index,
                    round_index=next_round,
                    candidate_index=candidate_index,
                    unit=unit,
                    run_id=task["run_id"],
                    volume=volume,
                    dock=dock,
                )
                if query_index == len(rows):
                    rows.append(result)
                elif rows[query_index] != result:
                    raise ValueError("resumed query result differs from durable ledger")
                if result["score"] is not None and (
                    champion is None
                    or (result["score"], result["endpoint"])
                    < (champion["score"], champion["endpoint"])
                ):
                    champion = {
                        "score": result["score"],
                        "endpoint": result["endpoint"],
                        "receipt_id": result["receipt_id"],
                        "candidate": candidate,
                        "query_index": result["query_index"],
                    }
                if query_index == len(curve):
                    curve.append(
                        {
                            "query": query_index + 1,
                            "best_score": (
                                None if champion is None else champion["score"]
                            ),
                        }
                    )
                outcomes.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "receipt_id": result["receipt_id"],
                        "score": result["score"],
                        "failure": result["failure"],
                        "oracle_protocol": oracle_protocol,
                    }
                )
                if query_index + 1 in REPORT_CALLS:
                    print(
                        json.dumps(
                            {
                                "unit": unit["unit_id"],
                                "call": query_index + 1,
                                "best_score": curve[-1]["best_score"],
                                "status": result["status"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if bootstrap:
                if search is None:
                    search = DynamicProgramOptimizer(
                        config,
                        source_group=source_group,
                        oracle_protocol=oracle_protocol,
                        hierarchy=None,
                    )
                    _configure_refinement_rng(
                        search, admission["rng_namespaces"]["dynamic_v0_refinement"]
                    )
                for candidate, outcome in zip(
                    batch["candidates"], outcomes, strict=True
                ):
                    if outcome["score"] is not None:
                        search.add_measured_program(
                            candidate,
                            receipt_id=outcome["receipt_id"],
                            score=outcome["score"],
                            static_score=outcome["score"],
                        )
            else:
                search.observe_batch(batch["batch_id"], outcomes)
            empty_rounds = empty_rounds + 1 if not batch["candidates"] else 0
            summary = {
                "round": next_round,
                "bootstrap": bootstrap,
                "queries_this_round": len(batch["candidates"]),
                "queries_total": len(rows),
                "best_score": None if champion is None else champion["score"],
                "summary_points": _summary_points(curve),
                "proposal_seconds": batch.get("proposal_seconds", 0.0),
                "attempts": len(batch["attempts"]),
                "attempt_status": dict(
                    Counter(row["status"] for row in batch["attempts"])
                ),
                "completed_at_utc": _stamp(),
            }
            publish_json(round_folder / "complete.json", summary)
            next_round += 1
            checkpoint_query_count = len(rows)
            _checkpoint(
                checkpoint_path,
                search=search,
                next_round=next_round,
                rows=rows,
                curve=curve,
                champion=champion,
                empty_rounds=empty_rounds,
            )
            publish_json(
                folder / "progress.json",
                {"unit_id": unit["unit_id"], "stage": 20, **summary, "at": _stamp()},
            )
            volume.commit()
        termination = (
            "stage_call_ceiling_reached"
            if len(rows) >= STAGE_CEILING
            else (
                "proposal_yield_exhausted"
                if empty_rounds >= source_contract["max_consecutive_empty_rounds"]
                else "round_limit_reached"
            )
        )
        result = {
            "schema_version": "t4_hybrid_top3_v0_stage20_unit_result_v1",
            "run_id": task["run_id"],
            "stage_id": task["stage_id"],
            "unit": unit,
            "status": "complete" if len(rows) == STAGE_CEILING else "incomplete",
            "termination": termination,
            "oracle_calls": len(rows),
            "successful_oracle_calls": sum(row["score"] is not None for row in rows),
            "failed_oracle_calls": sum(row["score"] is None for row in rows),
            "champion": champion,
            "curve": curve,
            "summary_points": _summary_points(curve),
            "resume_checkpoint_path": str(checkpoint_path.relative_to(artifacts)),
            "resume_checkpoint_sha256": sha256_file(checkpoint_path),
            "terminal_full_run_result_written": False,
            "later_extension_authorized": False,
            "seconds_this_invocation": perf_counter() - began,
            "completed_at_utc": _stamp(),
            "software": {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "openbabel": subprocess.check_output(
                    ["obabel", "-V"], text=True
                ).strip(),
            },
        }
        seal(stage_result_path, result)
        volume.commit()
        return result
    except (
        ValueError,
        RuntimeError,
        OSError,
        ImportError,
        KeyError,
        TypeError,
    ) as error:
        failure = {
            "schema_version": "t4_hybrid_top3_v0_stage20_unit_failure_v1",
            "run_id": task["run_id"],
            "stage_id": task["stage_id"],
            "unit": unit,
            "status": "failed",
            "error": repr(error),
            "oracle_calls": len(rows),
            "curve": curve,
            "summary_points": _summary_points(curve),
            "automatic_retry": False,
            "at": _stamp(),
        }
        seal(folder / "stages" / "q20" / "failure.json", failure)
        volume.commit()
        return failure
    finally:
        stop.set()
        thread.join(timeout=2)


__all__ = [
    "CONTRACT_PATH",
    "FULL_FROZEN_CEILING",
    "KIND",
    "REPORT_CALLS",
    "SCHEMA",
    "STAGE_CEILING",
    "load_contract",
    "remote_preflight",
    "run_identity",
    "run_unit",
]
