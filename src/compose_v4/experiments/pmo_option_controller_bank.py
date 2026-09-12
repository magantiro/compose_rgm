"""Prepare and collect a zero-oracle balanced option-continuation bank.

This experiment exercises the production WHERE/WHAT/HOW hierarchy.  It does
not fit a value model and it never calls the PMO oracle.  Exact persistent
source states are inherited from the locked complete-plan assay rather than
reconstructed from canonical SMILES.
"""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import (
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.t4_matched_pilot import _stamp
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_option_controller_bank"
APP_NAME = "compose-pmo-option-controller-bank"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
PROTOCOL = "docs/PMO_OPTION_CONTROLLER_BANK.md"
SCORED_REPORT = "diagnostics/pmo_plan_pool_prevalence/report.json"
ONLINE_CONTRACT = "configs/pmo_online_policy.json"
SEED = 20260922
PARENTS = 8
STREAMS_PER_PARENT = 2
BOUNDARIES = 3
PRIMITIVE_BUDGET = 96


def _verify_self_hash(payload: dict, field: str = "content_sha256") -> None:
    expected = payload.get(field)
    if (
        not isinstance(expected, str)
        or identity({k: v for k, v in payload.items() if k != field}) != expected
    ):
        raise ValueError(f"input has an invalid {field}")


def _source_rows(report: dict) -> list[dict]:
    rows = report.get("scored")
    if not isinstance(rows, list) or not rows:
        raise ValueError("complete-plan report has no scored rows")
    if len(rows) != report.get("new_oracle_calls") or len(
        {row.get("candidate_id") for row in rows}
    ) != len(rows):
        raise ValueError("complete-plan scored census or candidate identities disagree")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        source = row.get("source")
        score = row.get("parent_score")
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(score, (int, float))
        ):
            raise ValueError("complete-plan row lacks a source or parent score")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("complete-plan parent score is outside [0, 1]")
        grouped.setdefault(source, []).append(row)

    sources = []
    for source, members in grouped.items():
        scores = {float(row["parent_score"]) for row in members}
        if len(scores) != 1:
            raise ValueError(f"source has inconsistent parent scores: {source}")
        exact: dict[str, dict] = {}
        traces = []
        for row in members:
            path = Path(row["path"])
            verify_file(path, row["sha256"])
            trace = json.loads(path.read_text())
            states = trace.get("states")
            if not isinstance(states, list) or not states:
                raise ValueError(f"compiled trace lacks an exact source state: {path}")
            state = states[0]
            graph = decode_state(state)
            if canonical_state_key(graph) != source:
                raise ValueError(
                    f"exact trace source disagrees with canonical identity: {path}"
                )
            exact[identity(state)] = state
            traces.append(
                {
                    "candidate_id": row["candidate_id"],
                    "path": str(path),
                    "sha256": row["sha256"],
                }
            )
        if len(exact) != 1:
            raise ValueError(f"source has divergent persistent layouts: {source}")
        state_id, state = next(iter(exact.items()))
        sources.append(
            {
                "source_id": identity(
                    {
                        "canonical_smiles": source,
                        "exact_state_id": state_id,
                        "parent_score": next(iter(scores)),
                    }
                ),
                "canonical_smiles": source,
                "parent_score": next(iter(scores)),
                "exact_state": state,
                "exact_state_id": state_id,
                "trace_census": len(traces),
                "trace_evidence": min(traces, key=lambda row: row["candidate_id"]),
            }
        )
    return sorted(
        sources, key=lambda row: (-row["parent_score"], row["canonical_smiles"])
    )


def prepare(root: Path, report_path: Path | None = None) -> dict:
    """Freeze exact top-parent inputs and a launch-denied experiment contract."""

    root = Path(root)
    report_path = root / SCORED_REPORT if report_path is None else Path(report_path)
    report = json.loads(report_path.read_text())
    _verify_self_hash(report)
    if (
        report.get("schema_version") != "pmo_plan_pool_prevalence_result_v1"
        or report.get("task") != "perindopril_mpo"
        or report.get("new_oracle_calls") != 95
    ):
        raise ValueError(
            "complete-plan report is not the locked 95-call Perindopril assay"
        )
    sources = _source_rows(report)
    if len(sources) < PARENTS:
        raise ValueError(
            f"option bank requires at least {PARENTS} distinct exact parents"
        )
    selected = sources[:PARENTS]
    try:
        report_reference = report_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        report_reference = str(report_path)
    prepared = {
        "schema_version": "pmo_option_controller_bank_prepared_v1",
        "task": "perindopril_mpo",
        "selection": "top_parent_score_then_canonical_identity",
        "parents": selected,
        "source_census": len(sources),
        "input_report": {
            "path": report_reference,
            "sha256": sha256_file(report_path),
            "content_sha256": report["content_sha256"],
            "analysis_commit": report["analysis_commit"],
        },
        "oracle_calls_during_preparation": 0,
    }
    prepared["content_sha256"] = identity(prepared)
    publish_json(root / PREPARED, prepared)

    online = json.loads((root / ONLINE_CONTRACT).read_text())
    if identity(
        {k: v for k, v in online.items() if k != "contract_sha256"}
    ) != online.get("contract_sha256"):
        raise ValueError("production inference contract hash mismatch")
    contract = {
        "schema_version": "pmo_option_controller_bank_contract_v1",
        "authorization": "2026-09-12 local zero-oracle preparation only",
        "proposal_generation_authorized": False,
        "oracle_authorized": False,
        "reference_training_authorized": False,
        "value_training_authorized": False,
        "task": "perindopril_mpo",
        "seed": SEED,
        "parents": PARENTS,
        "streams_per_parent": STREAMS_PER_PARENT,
        "streams": PARENTS * STREAMS_PER_PARENT,
        "option_boundaries": BOUNDARIES,
        "primitive_budget": PRIMITIVE_BUDGET,
        "where_policy": "qualified_production_region_reference",
        "what_policy": "applicability_balanced_reference_with_generic",
        "how_policy": "production_option_reference",
        "resampling": False,
        "include_carbonyl_options": True,
        "include_region_replacement": True,
        "enable_build_ring_system": False,
        "runtime_contract_sha256": online["contract_sha256"],
        "expected_input_sha256": online["expected_input_sha256"],
        "law_caches": [],
        "prepared": {"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        "protocol": {"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        "compute": {
            "max_workers": 16,
            "worker_tasks": 16,
            "driver_containers": 1,
            "worker_timeout_seconds": 900,
            "driver_timeout_seconds": 1200,
            "hard_wall_minutes": 20,
            "cpu_per_container": 1,
            "memory_mib": 8192,
            "gpu": False,
            "retries": 0,
        },
    }
    contract["contract_sha256"] = identity(contract)
    publish_json(root / CONTRACT, contract)
    return contract


def load_contract(root: Path, *, require_proposal_authority: bool = False) -> dict:
    root = Path(root)
    contract = json.loads((root / CONTRACT).read_text())
    if identity(
        {k: v for k, v in contract.items() if k != "contract_sha256"}
    ) != contract.get("contract_sha256"):
        raise ValueError("option-controller bank contract hash mismatch")
    expected = (
        contract.get("task"),
        contract.get("seed"),
        contract.get("parents"),
        contract.get("streams_per_parent"),
        contract.get("streams"),
        contract.get("option_boundaries"),
        contract.get("primitive_budget"),
        contract.get("resampling"),
        contract.get("enable_build_ring_system"),
    )
    if expected != (
        "perindopril_mpo",
        SEED,
        PARENTS,
        STREAMS_PER_PARENT,
        PARENTS * STREAMS_PER_PARENT,
        BOUNDARIES,
        PRIMITIVE_BUDGET,
        False,
        False,
    ):
        raise ValueError("option-controller bank contract exceeds the frozen recipe")
    for key in (
        "oracle_authorized",
        "reference_training_authorized",
        "value_training_authorized",
    ):
        if contract.get(key) is not False:
            raise ValueError(f"option-controller bank must keep {key}=false")
    if (
        require_proposal_authority
        and contract.get("proposal_generation_authorized") is not True
    ):
        raise PermissionError(
            "remote option-bank proposal generation is not authorized"
        )
    for key in ("prepared", "protocol"):
        verify_file(root / contract[key]["path"], contract[key]["sha256"])
    prepared = json.loads((root / contract["prepared"]["path"]).read_text())
    _verify_self_hash(prepared)
    if len(prepared["parents"]) != PARENTS:
        raise ValueError("prepared parent census changed")
    return contract


def stream_identity(contract: dict, parent: dict, stream_index: int) -> str:
    return identity(
        {
            "contract_sha256": contract["contract_sha256"],
            "source_id": parent["source_id"],
            "stream_index": int(stream_index),
        }
    )


@contextmanager
def session(task, root, artifact_root, volume, validate_revision):
    validate_revision(task["image_revision"])
    contract = load_contract(root, require_proposal_authority=True)
    verify_file(root / APP, task["app_sha256"])
    body = {k: task[k] for k in ("contract_sha256", "image_revision", "app_sha256")}
    if (
        identity(body) != task["run_id"]
        or task["contract_sha256"] != contract["contract_sha256"]
    ):
        raise ValueError("option-bank deployment identity mismatch")
    output = artifact_root / KIND / task["run_id"]
    if "stream_id" in task:
        prepared = json.loads((root / PREPARED).read_text())
        parent_index, stream_index = task["parent_index"], task["stream_index"]
        if (
            type(parent_index) is not int
            or type(stream_index) is not int
            or not 0 <= parent_index < PARENTS
            or not 0 <= stream_index < STREAMS_PER_PARENT
            or task["stream_id"]
            != stream_identity(
                contract, prepared["parents"][parent_index], stream_index
            )
        ):
            raise ValueError("worker lies outside the frozen option-bank stream census")
        output = output / "workers" / task["stream_id"]
    volume.reload()
    lock, stop = threading.RLock(), threading.Event()

    def commit():
        with lock:
            volume.commit()

    store, started = Store(output, commit), perf_counter()
    progress = {"phase": "initialization", "oracle_calls": 0}
    store.save("identity", {k: v for k, v in task.items() if k != "image_revision"})

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {**progress, "at": _stamp(), "seconds": perf_counter() - started},
            )
            store.flush(force=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield contract, store, progress
    except Exception as error:
        store.save(
            "failure", {"error": repr(error), "progress": progress, "at": _stamp()}
        )
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        store.flush(force=True)


def _zero_controller():
    import torch

    from compose_v4.control.improvement_value import (
        MonotoneImprovementModel,
        improvement_parameter_id,
    )
    from compose_v4.control.option_controller import OptionBoundaryController
    from compose_v4.control.option_features import (
        REGION_CONTEXT_NAMES,
        option_feature_names,
    )
    from compose_v4.control.option_policy import (
        AdvantageWeightedOptionActor,
        option_actor_parameter_id,
    )
    from compose_v4.control.trajectory_value import molecule_features

    value_dim = len(molecule_features("CC")) + 3
    actor = AdvantageWeightedOptionActor(
        value_dim + len(REGION_CONTEXT_NAMES), len(option_feature_names()), hidden=16
    )
    value = MonotoneImprovementModel(value_dim, BOUNDARIES, 3, hidden=16)
    for model in (actor, value):
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
    return OptionBoundaryController(
        actor,
        value,
        actor_snapshot=option_actor_parameter_id(actor),
        value_snapshot=improvement_parameter_id(value),
        horizons=tuple(range(1, BOUNDARIES + 1)),
        thresholds=(0.005, 0.01, 0.02),
        base_floor=0.1,
        max_kl=1.0,
        beta=1.0,
    )


def worker_remote(task, root, artifact_root, volume, validate_revision):
    """Advance one exact source stream through at most three complete options."""

    from compose_v4.chem.molecular_graph import is_element
    from compose_v4.control.molecular_task_search import (
        MolecularHierarchy,
        MolecularSearchState,
    )
    from compose_v4.control.option_continuation import (
        EXECUTABLE_PRODUCT_GATE,
        OptionContinuationKernel,
    )
    from compose_v4.control.option_controller_runtime import (
        HowControl,
        OptionControllerRuntime,
    )
    from compose_v4.control.region_rewrite import Lineage
    from compose_v4.experiments.continuation_profile import ExecutorMeter
    from compose_v4.experiments.pmo_online_policy import runtime as production_runtime
    from compose_v4.experiments.saved_marked_law import SavedMarkedLaw

    with session(task, root, artifact_root, volume, validate_revision) as (
        contract,
        store,
        progress,
    ):
        previous = store.read("complete")
        if previous is not None:
            return previous
        prepared = json.loads((root / PREPARED).read_text())
        parent = prepared["parents"][task["parent_index"]]
        graph = decode_state(parent["exact_state"])
        if (
            identity(parent["exact_state"]) != parent["exact_state_id"]
            or canonical_state_key(graph) != parent["canonical_smiles"]
        ):
            raise ValueError("prepared exact parent state failed admission")
        started = perf_counter()
        production = production_runtime(
            root, artifact_root, contract["runtime_contract_sha256"]
        )
        for key, field in (
            ("r_theta_checkpoint", "model_checkpoint"),
            ("r_theta_run_paths", "run_paths"),
        ):
            verify_file(Path(production[field]), contract["expected_input_sha256"][key])
        store.save("runtime_validation", production["validation"])
        primed = production["primed_law"]
        store.save(f"laws/{identity(primed['source'])}", primed)
        law = SavedMarkedLaw(
            production["model"],
            store.output,
            store.save,
            store.read,
            repo_root=root,
            artifact_root=artifact_root,
            contract=contract,
            progress=progress,
        )
        meter = ExecutorMeter(None)
        with meter.instrument():
            kernel = OptionContinuationKernel(
                law,
                production["system"],
                max_executor_applications=None,
                product_gate=EXECUTABLE_PRODUCT_GATE,
            )
            hierarchy = MolecularHierarchy(
                kernel,
                lazy_applicability=True,
                include_carbonyl_options=contract["include_carbonyl_options"],
                include_region_replacement=contract["include_region_replacement"],
            )
            constant_utility = float(parent["parent_score"])
            engine = OptionControllerRuntime(
                hierarchy,
                _zero_controller(),
                total_options=contract["option_boundaries"],
                reference_snapshot=production["validation"]["manifest_sha256"],
                feature_snapshot="morgan512_topology5_option_context_v1",
                utility_snapshot="constant_parent_score_no_guidance_v1",
                how=HowControl("reference", "production_option_reference_v1", 0, 0),
                utility=lambda _node: constant_utility,
                how_terminal_weight=lambda _node: 1.0,
                how_fallback_weight=lambda _node: 1.0,
                enable_build_ring_system=contract["enable_build_ring_system"],
            )
            node = MolecularSearchState(
                graph,
                Lineage.initial(np.flatnonzero(is_element(graph.atom_types))),
                contract["primitive_budget"],
                parent["source_id"],
            )
            population = engine.start_population(
                (node,),
                incumbent=constant_utility,
                seed=SEED + task["parent_index"] * 2 + task["stream_index"],
            )
            boundaries, candidates = [], []
            for boundary in range(1, contract["option_boundaries"] + 1):
                progress.update(phase="option_boundary", boundary=boundary)
                population, receipt = engine.advance(
                    population, incumbent=constant_utility, resample=False
                )
                for transition in receipt["transitions"]:
                    if not math.isclose(
                        transition.get("option_reference_probability", 0.0),
                        transition.get("option_proposal_probability", -1.0),
                        rel_tol=0,
                        abs_tol=1e-12,
                    ) or not math.isclose(
                        transition.get("option_kl", math.inf), 0.0, abs_tol=1e-12
                    ):
                        raise RuntimeError(
                            "zero actor changed the balanced WHAT reference law"
                        )
                locked = list(engine.locked_candidates(population))
                for row in locked:
                    row.update(
                        source_id=parent["source_id"],
                        source_smiles=parent["canonical_smiles"],
                        parent_score=constant_utility,
                        stream_index=task["stream_index"],
                        stream_id=task["stream_id"],
                    )
                boundaries.append(receipt)
                candidates.extend(locked)
                store.save(
                    f"boundaries/{boundary:02d}",
                    {"receipt": receipt, "candidates": locked},
                )
                if receipt["particle_update"]["status"] == "extinct":
                    break
        result = {
            "schema_version": "pmo_option_controller_bank_worker_v1",
            "status": "complete",
            "stream_id": task["stream_id"],
            "source_id": parent["source_id"],
            "parent_index": task["parent_index"],
            "stream_index": task["stream_index"],
            "runtime_snapshot": engine.snapshot,
            "boundaries": boundaries,
            "candidates": candidates,
            "executor_applications": meter.calls,
            "executor_seconds": meter.seconds,
            "law_counts": law.counts,
            "seconds": perf_counter() - started,
            "new_oracle_calls": 0,
        }
        store.save("complete", result)
        return result


def candidate_lock(worker_results: list[dict]) -> dict:
    """Deduplicate for later scoring while preserving every path origin."""

    results = sorted(worker_results, key=lambda row: row["stream_id"])
    origins: dict[str, list[dict]] = {}
    exact: dict[str, dict[str, dict]] = {}
    for result in results:
        if result.get("status") != "complete" or result.get("new_oracle_calls") != 0:
            raise ValueError("candidate lock requires complete zero-oracle workers")
        for candidate in result["candidates"]:
            smiles = candidate["canonical_smiles"]
            origins.setdefault(smiles, []).append(
                {
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "source_id",
                        "source_smiles",
                        "parent_score",
                        "stream_index",
                        "stream_id",
                        "option_boundary",
                        "particle_id",
                        "exact_state_id",
                        "history",
                    )
                }
            )
            exact.setdefault(smiles, {})[candidate["exact_state_id"]] = candidate[
                "exact_state"
            ]
    locked = []
    for smiles in sorted(origins):
        variants = exact[smiles]
        locked.append(
            {
                "candidate_id": identity(
                    {
                        "canonical_smiles": smiles,
                        "exact_state_ids": sorted(variants),
                        "origins": sorted(
                            row["candidate_id"] for row in origins[smiles]
                        ),
                    }
                ),
                "canonical_smiles": smiles,
                "exact_state_variants": [variants[key] for key in sorted(variants)],
                "exact_state_ids": sorted(variants),
                "origins": sorted(
                    origins[smiles],
                    key=lambda row: (row["stream_id"], row["option_boundary"]),
                ),
                "locked": True,
                "oracle_status": "unrequested",
            }
        )
    body = {
        "schema_version": "pmo_option_controller_candidate_lock_v1",
        "locked": True,
        "oracle_status": "unrequested",
        "new_oracle_calls": 0,
        "worker_streams": len(results),
        "trajectory_boundary_candidates": sum(
            len(row["candidates"]) for row in results
        ),
        "unique_canonical_candidates": len(locked),
        "candidates": locked,
    }
    return {**body, "content_sha256": identity(body)}


def driver_remote(task, root, artifact_root, volume, validate_revision, parallel):
    with session(task, root, artifact_root, volume, validate_revision) as (
        contract,
        store,
        progress,
    ):
        previous = store.read("complete")
        if previous is not None:
            return previous
        prepared = json.loads((root / PREPARED).read_text())
        tasks = []
        for parent_index, parent in enumerate(prepared["parents"]):
            for stream_index in range(STREAMS_PER_PARENT):
                stream_id = stream_identity(contract, parent, stream_index)
                tasks.append(
                    {
                        **task,
                        "parent_index": parent_index,
                        "stream_index": stream_index,
                        "stream_id": stream_id,
                    }
                )
        if len(tasks) != contract["streams"]:
            raise RuntimeError("driver stream census changed")
        started, results = perf_counter(), []
        progress.update(
            phase="proposal_workers", completed_streams=0, total_streams=len(tasks)
        )
        for result in parallel(tasks):
            results.append(result)
            progress["completed_streams"] = len(results)
            store.save(f"worker_receipts/{result['stream_id']}", result)
        if {row["stream_id"] for row in results} != {row["stream_id"] for row in tasks}:
            raise RuntimeError("proposal result census is incomplete")
        lock = candidate_lock(results)
        store.save("candidate_lock", lock)
        option_counts = Counter()
        completion_counts = Counter()
        for result in results:
            for receipt in result["boundaries"]:
                for transition in receipt["transitions"]:
                    option = transition.get("option")
                    if option is not None:
                        option_counts[option] += 1
                        completion_counts[(option, transition["status"])] += 1
        summary = {
            "schema_version": "pmo_option_controller_bank_result_v1",
            "status": "complete_candidate_lock_no_oracle",
            "task": contract["task"],
            "run_id": task["run_id"],
            "contract_sha256": contract["contract_sha256"],
            "streams": len(results),
            "option_selections": dict(sorted(option_counts.items())),
            "option_outcomes": {
                f"{option}|{status}": count
                for (option, status), count in sorted(completion_counts.items())
            },
            "trajectory_boundary_candidates": lock["trajectory_boundary_candidates"],
            "unique_canonical_candidates": lock["unique_canonical_candidates"],
            "candidate_lock_sha256": lock["content_sha256"],
            "proposal_seconds_sum": sum(row["seconds"] for row in results),
            "executor_applications": sum(
                row["executor_applications"] for row in results
            ),
            "new_oracle_calls": 0,
            "seconds": perf_counter() - started,
            "completed_at": _stamp(),
        }
        summary["content_sha256"] = identity(summary)
        store.save("complete", summary)
        return summary
