"""Frozen full-suite T4 program-controller benchmark.

The controller is a fast optimization proposal over exact executable programs.
It is not the frozen reference law. Every paid candidate is locked before the
docking call, and each search replicate owns an independent query ledger.
"""

from __future__ import annotations

import gzip
import json
import platform
import shutil
import subprocess
import threading
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer, ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_policy import ProgramEntry
from compose_v4.control.program_transfer import initial_program_batch
from compose_v4.experiments.continuation_profile import (
    canonical_bytes,
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

KIND = "t4_frozen_program_benchmark"
CONTRACT = "configs/t4_frozen_program_benchmark_v2.json"
PREFLIGHT = "diagnostics/t4_frozen_program_benchmark/preflight_v2.json"
APP = "modal_apps/t4_frozen_program_benchmark_app.py"
APP_NAME = "compose-t4-frozen-program-benchmark"


def strict_endpoint_scorer(seed_smiles: str, *, delta: float = 0.4):
    """Official IVG table filters, without COMPOSE's optional med-chem screen."""
    seed = Chem.MolFromSmiles(seed_smiles)
    if seed is None:
        raise ValueError("invalid T4 source SMILES")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(seed)

    def score(row):
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            return {
                **row,
                "qed": None,
                "sa": None,
                "sim": None,
                "oracle_eligible": False,
                "endpoint_exclusion_reasons": ["invalid_smiles"],
            }
        qed = float(QED.qed(molecule))
        sa = float(sascorer.calculateScore(molecule))
        similarity = float(
            DataStructs.TanimotoSimilarity(seed_fp, generator.GetFingerprint(molecule))
        )
        failures = []
        if not similarity > delta:
            failures.append("similarity_not_strictly_above_delta")
        if not qed > 0.6:
            failures.append("qed_not_strictly_above_0.6")
        if not sa < 4.0:
            failures.append("sa_not_strictly_below_4")
        return {
            **row,
            "qed": qed,
            "sa": sa,
            "sim": similarity,
            "oracle_eligible": not failures,
            "endpoint_exclusion_reasons": failures,
        }

    return score


def configured(contract, unit):
    payload = dict(contract["controller"])
    payload["channel_probabilities"] = tuple(payload["channel_probabilities"])
    payload["seed"] = unit["controller_seed"]
    config = ProgramSearchConfig(**payload)
    expected = ProgramSearchConfig.program_only_recipe(seed=unit["controller_seed"])
    protected = {
        "proposal_mode": "program_only",
        "parent_allocation": "score_blind",
        "continuation_root": "exact_current_state",
        "decompose_programs": True,
        "require_broad_runtime": False,
        "composition_probability": 0.0,
        "mutation_sampling": "random",
    }
    for field, value in protected.items():
        if getattr(config, field) != value or getattr(expected, field) != value:
            raise ValueError(f"frozen controller changed protected field {field}")
    return config


def load_contract(root):
    contract = unseal(root / CONTRACT)
    if (
        contract["schema_version"] != "t4_frozen_program_benchmark_v2"
        or contract["delta"] != 0.4
        or contract["search_replicates"] != 3
        or contract["calls_per_unit"] != 1000
        or contract["search_call_ceiling"] != 45000
        or contract["confirmation_call_ceiling"] != 30
        or contract["total_call_ceiling"] != 45030
        or contract["container_limit"] != 30
        or contract["reserved_usd"] != 20
        or contract["cold_start_seed"] != 20260913
        or contract["plateau_stop"]
        != {
            "minimum_calls": 500,
            "window_calls": 250,
            "minimum_improvement": 0.3,
            "requires_best_at_or_better_than_ivg_mean": True,
        }
    ):
        raise ValueError("full T4 contract differs from the authorized ceiling")
    if len(contract["cells"]) != 15 or len(contract["units"]) != 45:
        raise ValueError("full T4 contract requires 15 cells and 45 search units")
    if len({row["unit_id"] for row in contract["units"]}) != 45:
        raise ValueError("T4 unit identifiers are not unique")
    if sum(row["budget"] for row in contract["units"]) != 45000:
        raise ValueError("T4 per-unit budgets do not sum to the frozen ceiling")
    for path, digest in contract["inputs"].items():
        verify_file(root / path, digest)
    return contract


def competitive_plateau(curve, ivg_mean, policy):
    """Return a predeclared early-stop diagnostic for lower-is-better scores."""
    minimum_calls = policy["minimum_calls"]
    window_calls = policy["window_calls"]
    threshold = policy["minimum_improvement"]
    if len(curve) < minimum_calls or len(curve) <= window_calls:
        return {
            "stop": False,
            "reason": "insufficient_calls",
            "calls": len(curve),
        }
    current = curve[-1]["best_score"]
    previous = curve[-window_calls - 1]["best_score"]
    if current is None or previous is None:
        return {
            "stop": False,
            "reason": "missing_successful_score",
            "calls": len(curve),
        }
    competitive = current <= ivg_mean
    improvement = previous - current
    stop = competitive and improvement < threshold
    return {
        "stop": stop,
        "reason": (
            "competitive_plateau"
            if stop
            else "still_improving"
            if competitive
            else "not_yet_competitive"
        ),
        "calls": len(curve),
        "window_calls": window_calls,
        "window_start_best": previous,
        "current_best": current,
        "improvement": improvement,
        "minimum_improvement": threshold,
        "ivg_mean": ivg_mean,
        "competitive": competitive,
    }


def first_crossing(curve, threshold):
    """First query whose best-so-far docking score reaches a lower-is-better threshold."""
    return next(
        (
            row["query"]
            for row in curve
            if row["best_score"] is not None and row["best_score"] <= threshold
        ),
        None,
    )


def load_library(root, contract):
    rows = json.loads((root / contract["library_path"]).read_text())
    entries = tuple(
        ProgramEntry(EditProgram.from_payload(row["program"]), tuple(row["source_groups"]))
        for row in rows
    )
    if len(entries) != contract["library_programs"]:
        raise ValueError("shared program library size changed")
    return entries


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
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"corrupt compressed benchmark artifact: {path}")
    return payload


def _query_rows(folder: Path):
    rows = []
    for started_path in sorted((folder / "queries").glob("query_*/started.json")):
        started = unseal(started_path)
        result_path = started_path.with_name("result.json")
        if not result_path.exists():
            raise RuntimeError(f"ambiguous charged query; no automatic retry: {started_path}")
        result = unseal(result_path)
        if any(result.get(key) != value for key, value in started.items()):
            raise ValueError(f"query result changed its reservation: {result_path}")
        if result["query_index"] != len(rows):
            raise ValueError(f"noncontiguous T4 query ledger: {result_path}")
        rows.append(result)
    return rows


def _dock_candidate(
    *,
    folder,
    candidate,
    query_index,
    round_index,
    candidate_index,
    unit,
    run_id,
    volume,
    dock,
):
    output = folder / "queries" / f"query_{query_index:04d}"
    started = {
        "query_index": query_index,
        "round_index": round_index,
        "candidate_index": candidate_index,
        "candidate_id": candidate["candidate_id"],
        "endpoint": candidate["endpoint"],
        "target": unit["target"],
        "docking_seed": unit["docking_seed"],
        "started_at_utc": _stamp(),
    }
    if (output / "result.json").exists():
        result = unseal(output / "result.json")
        if any(
            result.get(key) != value for key, value in started.items() if key != "started_at_utc"
        ):
            raise ValueError("existing result differs from its frozen candidate")
        return result
    if (output / "started.json").exists():
        raise RuntimeError(f"ambiguous charged query; no automatic retry: {output}")
    seal(output / "started.json", started)
    volume.commit()
    began = perf_counter()
    tag = f"t4full_{run_id[:10]}_{unit['unit_id']}_{query_index:04d}"
    score = dock(candidate["endpoint"], unit["target"], tag, unit["docking_seed"])
    hashes = {}
    for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
        path = Path("/tmp") / tag / name
        if path.exists():
            hashes[name] = sha256_file(path)
    body = {
        **started,
        "status": "complete" if score is not None else "failed",
        "score": None if score is None else float(score),
        "failure": None if score is not None else "oracle_failed",
        "seconds": perf_counter() - began,
        "pose_sha256": hashes,
        "completed_at_utc": _stamp(),
    }
    result = {**body, "receipt_id": identity(body)}
    seal(output / "result.json", result)
    volume.commit()
    return result


def _checkpoint(path, *, search, next_round, rows, curve, champion, empty_rounds):
    if search is not None:
        search.history.clear()
    payload = {
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


def _restore_pending(search, batch):
    if search.pending is not None:
        if search.pending["batch_id"] != batch["batch_id"]:
            raise ValueError("checkpoint pending batch differs from round lock")
        return
    pending = {
        key: value
        for key, value in batch.items()
        if key not in ("proposal_seconds", "work_cache", "new_oracle_calls")
    }
    if (
        identity({key: value for key, value in pending.items() if key != "batch_id"})
        != pending["batch_id"]
    ):
        raise ValueError("cannot reconstruct optimizer pending state from round lock")
    search.pending = pending


def validate_launch(task, root, validate_revision):
    validate_revision(task["image_revision"])
    base = {key: value for key, value in task.items() if key not in ("run_id", "unit_id", "index")}
    if identity(base) != task["run_id"]:
        raise ValueError("frozen T4 launch identity changed")
    for path, digest in task["files_sha256"].items():
        verify_file(root / path, digest)
    preflight = json.loads((root / PREFLIGHT).read_text())
    if (
        preflight.get("passed") is not True
        or preflight.get("contract_sha256") != task["files_sha256"][CONTRACT]
        or preflight.get("new_oracle_calls") != 0
    ):
        raise ValueError("full-suite zero-oracle preflight is absent or mismatched")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("frozen T4 chemistry runtime changed")
    return load_contract(root)


def run_unit(task, root, artifacts, volume, validate_revision, dock):
    """Run or resume one independent 1,000-call cell/replicate unit."""
    contract = validate_launch(task, root, validate_revision)
    unit = next((row for row in contract["units"] if row["unit_id"] == task["unit_id"]), None)
    if unit is None:
        raise ValueError("search unit is outside the frozen census")
    volume.reload()
    folder = artifacts / KIND / task["run_id"] / "units" / unit["unit_id"]
    if (folder / "result.json").exists():
        return unseal(folder / "result.json")
    for path in ("/opt/dock/qvina02", f"/opt/dock/receptors/{unit['target']}.pdbqt"):
        verify_file(Path(path), contract["runtime_input_sha256"][path])
    source = decode_state(contract["cells"][unit["cell"]]["source_state"])
    entries = load_library(root, contract)
    config = configured(contract, unit)
    endpoint_score = strict_endpoint_scorer(unit["original_seed"], delta=contract["delta"])
    source_group = identity(
        {"target": unit["target"], "source_idx": unit["source_idx"], "seed": unit["original_seed"]}
    )
    oracle_protocol = unit["oracle_protocol"]
    started = {"task": task, "unit": unit, "started_at_utc": _stamp()}
    if (folder / "started.json").exists():
        prior = unseal(folder / "started.json")
        if prior["task"] != task or prior["unit"] != unit:
            raise ValueError("unit restart changed its frozen identity")
    else:
        seal(folder / "started.json", started)
        volume.commit()
    rows = _query_rows(folder)
    checkpoint_path = folder / "checkpoint.json.gz"
    if checkpoint_path.exists():
        saved = _read_gzip(checkpoint_path)
        if saved["query_count"] > len(rows):
            raise ValueError("checkpoint is ahead of the durable query ledger")
        if saved["query_count"] != len(saved["curve"]):
            raise ValueError("checkpoint query count differs from its score curve")
        search = (
            None
            if saved["search"] is None
            else ProgramOptimizer.restore(saved["search"], hierarchy=None)
        )
        next_round = saved["next_round"]
        curve = saved["curve"]
        champion = saved["champion"]
        empty_rounds = saved["consecutive_empty_rounds"]
    else:
        search, next_round, curve, champion, empty_rounds = None, 0, [], None, 0
    checkpoint_query_count = len(curve)
    if checkpoint_query_count > len(rows):
        raise ValueError("saved score curve is ahead of the query ledger")
    began = perf_counter()
    heartbeat_lock, stop = threading.RLock(), threading.Event()

    def heartbeat():
        while not stop.wait(30):
            with heartbeat_lock:
                publish_json(
                    folder / "progress.json",
                    {
                        "unit_id": unit["unit_id"],
                        "queries": len(rows),
                        "round": next_round,
                        "best_score": None if champion is None else champion["score"],
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
            len(rows) < unit["budget"]
            and next_round < contract["max_rounds"]
            and empty_rounds < contract["max_consecutive_empty_rounds"]
        ):
            remaining = unit["budget"] - len(rows)
            round_folder = folder / "rounds" / f"round_{next_round:04d}"
            batch_path = round_folder / "batch.json.gz"
            bootstrap = search is None or not search.entries
            if batch_path.exists():
                locked = _read_gzip(batch_path)
                if locked["round"] != next_round or locked["bootstrap"] != bootstrap:
                    raise ValueError("resumed round lock differs from checkpoint state")
                batch = locked["batch"]
                if not bootstrap:
                    _restore_pending(search, batch)
            else:
                if bootstrap:
                    batch = initial_program_batch(
                        source,
                        entries,
                        replace(
                            config,
                            seed=contract["cold_start_seed"],
                            candidates_per_batch=min(config.candidates_per_batch, remaining),
                        ),
                        source_group=source_group,
                        oracle_protocol=oracle_protocol,
                        eligibility=endpoint_score,
                    )
                else:
                    batch = search.propose_batch(endpoint_score)
                    if len(batch["candidates"]) > remaining:
                        ids = [row["candidate_id"] for row in batch["candidates"][:remaining]]
                        selection = {
                            "policy": "query_budget_prefix_v1",
                            "selected_ids": ids,
                            "available": len(batch["candidates"]),
                            "remaining_budget": remaining,
                        }
                        batch = search.lock_query_subset(batch["batch_id"], ids, selection)
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
                result = _dock_candidate(
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
                    raise ValueError("resumed query result differs from the durable ledger")
                if result["score"] is not None and (
                    champion is None
                    or (result["score"], result["endpoint"])
                    < (
                        champion["score"],
                        champion["endpoint"],
                    )
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
                            "best_score": None if champion is None else champion["score"],
                        }
                    )
                elif query_index >= len(curve):
                    raise ValueError("score curve has a gap inside a locked round")
                outcomes.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "receipt_id": result["receipt_id"],
                        "score": result["score"],
                        "failure": result["failure"],
                        "oracle_protocol": oracle_protocol,
                    }
                )
            if bootstrap:
                if search is None:
                    search = ProgramOptimizer(
                        config,
                        source_group=source_group,
                        oracle_protocol=oracle_protocol,
                        hierarchy=None,
                    )
                for candidate, outcome in zip(batch["candidates"], outcomes, strict=True):
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
                "proposal_seconds": batch.get("proposal_seconds", 0.0),
                "attempts": len(batch["attempts"]),
                "attempt_status": dict(Counter(row["status"] for row in batch["attempts"])),
                "eligible_channels": dict(
                    Counter(row["provenance"]["channel"] for row in batch["candidates"])
                ),
                "work_cache": batch.get("work_cache"),
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
                {
                    "unit_id": unit["unit_id"],
                    **summary,
                    "calls_to_ivg_mean": first_crossing(
                        curve, contract["ivg_reported"][unit["cell"]]["mean"]
                    ),
                    "at": _stamp(),
                },
            )
            volume.commit()
            print(json.dumps({"unit": unit["unit_id"], **summary}), flush=True)
            plateau = competitive_plateau(
                curve,
                contract["ivg_reported"][unit["cell"]]["mean"],
                contract["plateau_stop"],
            )
            if plateau["stop"]:
                termination = "competitive_plateau"
                break
        if termination is None:
            termination = (
                "query_budget_exhausted"
                if len(rows) >= unit["budget"]
                else "proposal_yield_exhausted"
                if empty_rounds >= contract["max_consecutive_empty_rounds"]
                else "round_limit_reached"
            )
        ivg_mean = contract["ivg_reported"][unit["cell"]]["mean"]
        result = {
            "schema_version": "t4_frozen_program_unit_v1",
            "unit": unit,
            "status": "complete",
            "termination": termination,
            "oracle_calls": len(rows),
            "successful_oracle_calls": sum(row["score"] is not None for row in rows),
            "failed_oracle_calls": sum(row["score"] is None for row in rows),
            "rounds": next_round,
            "champion": champion,
            "curve": curve,
            "ivg_mean": ivg_mean,
            "calls_to_ivg_mean": first_crossing(curve, ivg_mean),
            "plateau_diagnostic": competitive_plateau(curve, ivg_mean, contract["plateau_stop"]),
            "seconds_this_invocation": perf_counter() - began,
            "completed_at_utc": _stamp(),
            "software": {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "openbabel": subprocess.check_output(["obabel", "-V"], text=True).strip(),
            },
        }
        seal(folder / "result.json", result)
        volume.commit()
        return result
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as error:
        failure = {
            "schema_version": "t4_frozen_program_unit_failure_v1",
            "unit": unit,
            "status": "failed",
            "error": repr(error),
            "oracle_calls": len(rows),
            "champion": champion,
            "curve": curve,
            "at": _stamp(),
            "automatic_retry": False,
        }
        seal(folder / "failure.json", failure)
        volume.commit()
        return failure
    finally:
        stop.set()
        thread.join(timeout=2)


def confirmation_lock(contract, unit_results):
    by_cell = defaultdict(list)
    for result in unit_results:
        if result.get("status") == "complete" and result.get("champion") is not None:
            by_cell[result["unit"]["cell"]].append(result)
    locked, missing = [], []
    for cell in sorted(contract["cells"]):
        choices = by_cell[cell]
        if not choices:
            missing.append(cell)
            continue
        selected = min(
            choices,
            key=lambda row: (
                row["champion"]["score"],
                row["champion"]["endpoint"],
                row["unit"]["replicate"],
            ),
        )
        for repeat, docking_seed in enumerate(contract["confirmation_seeds"]):
            body = {
                "cell": cell,
                "target": selected["unit"]["target"],
                "source_idx": selected["unit"]["source_idx"],
                "endpoint": selected["champion"]["endpoint"],
                "first_score": selected["champion"]["score"],
                "first_receipt_id": selected["champion"]["receipt_id"],
                "source_unit": selected["unit"]["unit_id"],
                "repeat": repeat,
                "docking_seed": docking_seed,
                "oracle_protocol": selected["unit"]["oracle_protocol"],
            }
            locked.append({**body, "query_id": identity(body)})
    if len(locked) > contract["confirmation_call_ceiling"]:
        raise ValueError("confirmation lock exceeds its call ceiling")
    return {"queries": locked, "missing_cells": missing}


def run_confirmation(task, root, artifacts, volume, validate_revision, dock):
    contract = validate_launch(task, root, validate_revision)
    volume.reload()
    folder = artifacts / KIND / task["run_id"]
    lock = unseal(folder / "confirmation_lock.json")
    index = task["index"]
    if type(index) is not int or not 0 <= index < len(lock["queries"]):
        raise ValueError("confirmation index outside frozen lock")
    row = lock["queries"][index]
    output = folder / "confirmations" / row["query_id"]
    if (output / "result.json").exists():
        return unseal(output / "result.json")
    if (output / "started.json").exists():
        raise RuntimeError("ambiguous confirmation is not retried")
    cell = contract["cells"][row["cell"]]
    if not strict_endpoint_scorer(cell["original_seed"], delta=contract["delta"])(
        {"smiles": row["endpoint"]}
    )["oracle_eligible"]:
        raise ValueError("confirmation endpoint lost official eligibility")
    for path in ("/opt/dock/qvina02", f"/opt/dock/receptors/{row['target']}.pdbqt"):
        verify_file(Path(path), contract["runtime_input_sha256"][path])
    seal(output / "started.json", {**row, "started_at_utc": _stamp()})
    volume.commit()
    began = perf_counter()
    tag = f"t4full_confirm_{task['run_id'][:10]}_{index:02d}"
    score = dock(row["endpoint"], row["target"], tag, row["docking_seed"])
    pose_hashes = {}
    for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
        source = Path("/tmp") / tag / name
        if source.exists():
            output.mkdir(parents=True, exist_ok=True)
            destination = output / name
            shutil.copyfile(source, destination)
            pose_hashes[name] = sha256_file(destination)
    result = {
        **row,
        "score": None if score is None else float(score),
        "status": "complete" if score is not None else "failed",
        "failure": None if score is not None else "oracle_failed",
        "seconds": perf_counter() - began,
        "pose_sha256": pose_hashes,
        "completed_at_utc": _stamp(),
        "oracle_call_charged": True,
    }
    seal(output / "result.json", result)
    volume.commit()
    return result


def aggregate(contract, results, confirmations):
    by_cell = defaultdict(list)
    for result in results:
        if result.get("status") == "complete":
            by_cell[result["unit"]["cell"]].append(result)
    cells = []
    for cell in sorted(contract["cells"]):
        runs = sorted(by_cell[cell], key=lambda row: row["unit"]["replicate"])
        bests = [row["champion"]["score"] for row in runs if row["champion"] is not None]
        reported = contract["ivg_reported"][cell]
        crossings = [row.get("calls_to_ivg_mean") for row in runs]
        cells.append(
            {
                "cell": cell,
                "completed_replicates": len(runs),
                "run_bests": bests,
                "compose_mean": mean(bests) if bests else None,
                "compose_sample_std": stdev(bests) if len(bests) > 1 else None,
                "ivg_run_bests": reported["run_bests"],
                "ivg_mean": reported["mean"],
                "ivg_sample_std": reported["sample_std"],
                "mean_better_than_ivg": len(bests) == 3 and mean(bests) < reported["mean"],
                "calls_to_ivg_mean_by_run": crossings,
                "runs_reaching_ivg_mean": sum(value is not None for value in crossings),
                "mean_calls_to_ivg_mean_when_reached": (
                    mean(value for value in crossings if value is not None)
                    if any(value is not None for value in crossings)
                    else None
                ),
                "search_calls": sum(row["oracle_calls"] for row in runs),
                "champion_confirmations": [row for row in confirmations if row.get("cell") == cell],
            }
        )
    complete = [row for row in cells if row["completed_replicates"] == 3]
    return {
        "cells": cells,
        "complete_cells": len(complete),
        "cells_better_than_ivg_mean": sum(row["mean_better_than_ivg"] for row in complete),
        "compose_sum_of_cell_means": sum(row["compose_mean"] for row in complete),
        "ivg_sum_on_complete_cells": sum(row["ivg_mean"] for row in complete),
    }


def run_all(task, root, artifacts, volume, validate_revision, parallel, confirm):
    contract = validate_launch(task, root, validate_revision)
    volume.reload()
    folder = artifacts / KIND / task["run_id"]
    if (folder / "result.json").exists():
        return unseal(folder / "result.json")
    if not (folder / "started.json").exists():
        seal(folder / "started.json", task)
        volume.commit()
    unit_ids = [row["unit_id"] for row in contract["units"]]
    results = []
    for unit_id, row in zip(
        unit_ids,
        parallel([{**task, "unit_id": unit_id} for unit_id in unit_ids]),
        strict=True,
    ):
        if isinstance(row, BaseException):
            volume.reload()
            unit_folder = folder / "units" / unit_id
            started_calls = len(list((unit_folder / "queries").glob("query_*/started.json")))
            completed_calls = len(list((unit_folder / "queries").glob("query_*/result.json")))
            row = {
                "status": "failed",
                "unit": next(unit for unit in contract["units"] if unit["unit_id"] == unit_id),
                "error": repr(row),
                "oracle_calls": started_calls,
                "completed_oracle_calls": completed_calls,
                "ambiguous_oracle_calls": started_calls - completed_calls,
                "automatic_retry": False,
                "at": _stamp(),
            }
        if row["unit"]["unit_id"] != unit_id:
            raise ValueError("parallel unit/result order changed")
        results.append(row)
        publish_json(
            folder / "progress.json",
            {
                "completed_units": len(results),
                "total_units": len(unit_ids),
                "search_calls": sum(result.get("oracle_calls", 0) for result in results),
                "latest": unit_id,
                "at": _stamp(),
            },
        )
        volume.commit()
    first = {
        "units": results,
        "search_calls": sum(row.get("oracle_calls", 0) for row in results),
        "completed_at_utc": _stamp(),
    }
    seal(folder / "first_result.json", first)
    locked = confirmation_lock(contract, results)
    seal(folder / "confirmation_lock.json", locked)
    volume.commit()
    confirmed = list(confirm([{**task, "index": index} for index in range(len(locked["queries"]))]))
    normalized_confirmations = []
    for query, row in zip(locked["queries"], confirmed, strict=True):
        if isinstance(row, BaseException):
            volume.reload()
            started = (folder / "confirmations" / query["query_id"] / "started.json").exists()
            row = {
                **query,
                "status": "failed",
                "score": None,
                "failure": repr(row),
                "oracle_call_charged": started,
                "automatic_retry": False,
                "at": _stamp(),
            }
        normalized_confirmations.append(row)
    confirmed = normalized_confirmations
    summary = aggregate(contract, results, confirmed)
    result = {
        "schema_version": "t4_frozen_program_benchmark_result_v1",
        "task": task,
        "search": first,
        "confirmations": confirmed,
        "summary": summary,
        "search_calls": first["search_calls"],
        "confirmation_calls": sum(row.get("oracle_call_charged", False) for row in confirmed),
        "total_calls": first["search_calls"]
        + sum(row.get("oracle_call_charged", False) for row in confirmed),
        "completed_at_utc": _stamp(),
        "limitations": contract["limitations"],
    }
    if result["total_calls"] > contract["total_call_ceiling"]:
        raise RuntimeError("completed run exceeded frozen call ceiling")
    seal(folder / "result.json", result)
    volume.commit()
    return result
