"""One bounded production-reference path, not an exhaustive continuation tree.

The one binding work ceiling is total public executor applications. A platform
timeout is incomplete evidence. No docking, training, or automatic retry lives
in this module. The existing option kernel owns every probability row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.fused_option import (
    BUILD_FUSED_RING_OPTION,
    FUSED_HORIZON,
    FusedProgress,
    eligible_fusion_edges,
)
from compose_v4.control.graph_geometry import structural_displacement
from compose_v4.control.option_continuation import OptionContinuationKernel, OptionState
from compose_v4.control.option_selector import bundle_identity
from compose_v4.control.region import enumerate_regions
from compose_v4.control.region_rewrite import Lineage, context_from_region
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    canonical_bytes,
    encode_action,
    publish_json,
    state_payload,
    verify_file,
)
from compose_v4.experiments.ring_construction_probe import topology_witness
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

CONTRACT_PATH = "configs/fused_reference_profile_contract.json"


def load_contract(path: Path) -> dict:
    value = json.loads(path.read_text())
    body = {k: v for k, v in value.items() if k != "contract_sha256"}
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != value.get("contract_sha256"):
        raise ValueError(f"contract self-hash mismatch: {path}")
    expected = {
        "schema_version": "fused_reference_profile_contract_v1",
        "option": BUILD_FUSED_RING_OPTION,
        "horizon": FUSED_HORIZON,
        "estimator": "reference",
        "max_expansions": 0,
        "max_terminal_evaluations": 0,
        "max_executor_applications": 2000,
        "kappa": 1,
        "macro_temperature": 2,
        "macro_exploration": 0.15,
        "oracle_calls": 0,
        "training_authorized": False,
        "source_index": 0,
        "persistent_slots": 48,
        "max_active_atoms": 40,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(f"{path}: unsupported {field}={value.get(field)!r}")
    return value


def initial_state(contract: dict, seed_manifest: list) -> tuple[OptionState, dict]:
    """Predeclared applicability-only selection, explicitly NOT a Q(M) sample."""
    source = seed_manifest[contract["source_index"]]
    if source["target"] != "parp1":
        raise ValueError("the frozen profile source must be PARP1 seed0")
    smiles = source["smiles"]
    # This is fresh source initialization, never reconstruction of an intermediate.
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), contract["persistent_slots"])
    regions = enumerate_regions(smiles)
    applicable = [r for r in regions if eligible_fusion_edges(graph, r.atoms)]
    if not applicable:
        raise ValueError(
            "no enumerated region has a necessary eligible fused edge; no replacement source"
        )
    region = min(applicable, key=lambda r: (r.size, tuple(sorted(r.atoms)), r.boundary))
    context = context_from_region(region)
    node = OptionState(
        graph,
        graph,
        context,
        Lineage.initial(np.flatnonzero(is_element(graph.atom_types))),
        BUILD_FUSED_RING_OPTION,
        0,
        FUSED_HORIZON,
        bundle_identity(smiles, 0, region.key(), BUILD_FUSED_RING_OPTION),
        FusedProgress(),
    )
    return node, {
        "selection_rule": contract["region_selection"],
        "is_qm_sample": False,
        "enumerated_regions": len(regions),
        "necessary_applicable_regions": len(applicable),
        "atoms": sorted(region.atoms),
        "boundary": region.boundary,
        "interface": region.interface,
        "intended_r_release": region.released_fraction,
        "eligible_oriented_edges": eligible_fusion_edges(graph, region.atoms),
        "source_role": contract["source_role"],
    }


def run_reference_profile(
    initial: OptionState,
    enumerate_law: Callable,
    system,
    contract: dict,
    output: Path,
    *,
    snapshot_id: str,
    commit_volume: Callable[[], None],
    progress: dict,
) -> dict:
    """Persist complete law/row units and every selected step with its RNG state."""
    started = perf_counter()
    meter = ExecutorMeter(contract["max_executor_applications"])
    rows, trace, law_receipts = [], [], []
    timings = {"enumeration": 0.0, "persistence": 0.0}
    candidate_keys = set()

    def persist(relative, payload):
        start = perf_counter()
        digest = publish_json(output / relative, payload)
        commit_volume()
        timings["persistence"] += perf_counter() - start
        return digest

    def timed_law(graph):
        progress.update(phase="law_enumeration", law_index=len(law_receipts))
        start, meter.phase = perf_counter(), "law_enumeration"
        try:
            families, actions, probabilities = enumerate_law(graph)
        finally:
            timings["enumeration"] += perf_counter() - start
            meter.phase = "option_validation"
        p = np.asarray(probabilities, dtype=float)
        if (
            p.ndim != 1
            or len(families) != len(actions)
            or len(actions) != len(p)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or (len(p) and not np.isclose(p.sum(), 1.0, atol=1e-8, rtol=1e-8))
        ):
            raise ValueError("production marked law is malformed; not publishing it")
        payload = {
            "schema_version": "fused_reference_law_v1",
            "snapshot_id": snapshot_id,
            "state": encode_state(graph),
            "marks": [encode_action(f, a) for f, a in zip(families, actions)],
            "probabilities": p.tolist(),
        }
        identity = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        path = f"laws/{identity}.json"
        law_receipts.append({"path": path, "sha256": persist(path, payload), "marks": len(p)})
        return families, actions, probabilities

    kernel = OptionContinuationKernel(timed_law, system, max_executor_applications=meter.limit)
    rng = np.random.default_rng(contract["seed"])
    node, status, log_probability = initial, "complete", 0.0
    with meter.instrument():
        while node.remaining:
            progress.update(phase="reference_row", step=node.step)
            attempts_start = len(meter.attempts)
            try:
                with meter.boundary():
                    row = kernel.row(node)
            except ContinuationBudgetExceeded:
                status = "executor_budget_exhausted"
                break
            finally:
                progress.update(
                    total_public_executor_calls=meter.calls, kernel_work=asdict(kernel.work)
                )
                if len(meter.attempts) > attempts_start:
                    persist(
                        f"attempts/{attempts_start:06d}.json",
                        {
                            "source": state_payload(node),
                            "attempts": meter.attempts[attempts_start:],
                            "partial_row": node.key() not in kernel._rows,
                        },
                    )
            work_before = asdict(kernel.work)
            cached = kernel.row(node)
            if (
                row.probabilities != cached.probabilities
                or tuple(s.key() for s in row.successors)
                != tuple(s.key() for s in cached.successors)
                or kernel.work.executor_applications != work_before["executor_applications"]
                or kernel.work.law_enumerations != work_before["law_enumerations"]
            ):
                raise RuntimeError("exact row-cache parity failed")
            keys = [canonical_state_key(s.graph) for s in row.successors]
            candidate_keys.update(keys)
            payload = {
                "schema_version": "fused_reference_row_v1",
                "snapshot_id": snapshot_id,
                "source": state_payload(node),
                "successors": [state_payload(s) for s in row.successors],
                "probabilities": row.probabilities,
                "marks": [encode_action(f, a) for f, a in kernel.marks(node)],
                "support_funnel": kernel.fused_support(node),
                "unique_canonical_products": len(set(keys)),
            }
            path = f"rows/{hashlib.sha256(canonical_bytes(payload)).hexdigest()}.json"
            rows.append(
                {
                    "path": path,
                    "sha256": persist(path, payload),
                    "step": node.step,
                    "augmented_successors": len(row.successors),
                }
            )
            progress["completed_rows"] = len(rows)
            if not row.successors:
                status = "no_admissible_action"
                break
            chosen = int(rng.choice(len(row.successors), p=row.probabilities))
            successor = row.successors[chosen]
            probability = row.probabilities[chosen]
            log_probability += float(np.log(probability))
            trace.append(
                {
                    "step": successor.step,
                    "source": state_payload(node),
                    "successor": state_payload(successor),
                    "row_path": path,
                    "selected_index": chosen,
                    "reference_probability": probability,
                    "probability": probability,
                    "kl": 0.0,
                    "mark": payload["marks"][chosen],
                    "realized_change": structural_displacement(
                        initial.graph, successor.graph, initial.lineage, successor.lineage
                    ),
                }
            )
            node = successor
            persist(
                "sampled_path.json",
                {
                    "schema_version": "fused_reference_path_v1",
                    "snapshot_id": snapshot_id,
                    "seed": contract["seed"],
                    "rng_state": rng.bit_generator.state,
                    "current_state": state_payload(node),
                    "trace": trace,
                    "conditional_path_log_probability": log_probability,
                    "sampling_complete": node.remaining == 0,
                },
            )
    if sorted(r["call_index"] for r in meter.attempts) != list(range(meter.calls)):
        raise RuntimeError("executor receipts do not cover every entered call")
    witness = topology_witness(initial.graph, node.graph, "fused") if node.remaining == 0 else None
    if witness is not None and not witness["passed"]:
        status = "independent_topology_mismatch"
    return {
        "schema_version": "fused_reference_result_v1",
        "status": status,
        "initial_state": state_payload(initial),
        "final_state": state_payload(node),
        "endpoint": canonical_state_key(node.graph) if status == "complete" else None,
        "independent_topology_witness": witness,
        "trace": trace,
        "rows": rows,
        "laws": law_receipts,
        "conditional_augmented_path_log_probability": log_probability,
        "kernel_work": asdict(kernel.work),
        "total_public_executor_calls": meter.calls,
        "proposal_seconds": perf_counter() - started,
        "timings": timings,
        "unique_canonical_candidate_products": len(candidate_keys),
        "candidate_role": "counterfactual valid primitive products, not extra Q(M) draws or oracle candidates",
        "terminal_objective_calls": 0,
        "oracle_calls": 0,
        "comparison_authorized": False,
    }


def run_remote_task(task, *, repo_root, artifact_root, volume, runtime_factory, validate_revision):
    """Thin authenticated remote boundary, with durable heartbeat and failure receipts."""
    import platform
    import resource
    import threading
    import traceback
    from datetime import datetime, timezone

    import torch
    from rdkit import rdBase

    from compose_v4.experiments.production_successor_kernel import enumerate_factorized_marked_law
    from compose_v4.rewrite.typed_ring_catalog import ring_catalog_fingerprint

    validate_revision(task["image_revision"])
    verify_file(repo_root / "modal_apps/genmol_t4_opt_app.py", task["app_sha256"])
    contract_file = repo_root / CONTRACT_PATH
    verify_file(contract_file, task["contract_sha256"])
    contract = load_contract(contract_file)
    identity = {
        "contract_sha256": task["contract_sha256"],
        "image_revision_sha256": task["image_revision"]["image_revision_sha256"],
        "app_sha256": task["app_sha256"],
    }
    if hashlib.sha256(canonical_bytes(identity)).hexdigest() != task["run_id"]:
        raise ValueError("fused profile launch identity mismatch")
    output = artifact_root / "fused_reference_profile" / task["run_id"]
    if (output / "result.json").exists():
        return json.loads((output / "result.json").read_text())
    if (output / "launch.json").exists():
        raise RuntimeError("partial fused profile exists; preserve and audit before any retry")
    started = perf_counter()
    progress = {
        "run_id": task["run_id"],
        "code_revision": task["image_revision"]["commit"],
        "phase": "inputs",
        "complete": False,
        "oracle_calls": 0,
    }
    stop, lock = threading.Event(), threading.RLock()

    def commit():
        with lock:
            volume.commit()

    def heartbeat():
        while not stop.wait(contract["compute"]["heartbeat_seconds"]):
            publish_json(
                output / "heartbeat.json",
                {
                    **progress,
                    "elapsed_seconds": perf_counter() - started,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            commit()

    publish_json(output / "launch.json", task)
    publish_json(output / "progress.json", progress)
    commit()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        inputs = {
            name: verify_file(Path(contract[name]), contract[f"{name}_sha256"])
            for name in ("run_paths", "checkpoint")
        }
        inputs["source_manifest"] = verify_file(
            repo_root / contract["source_manifest"], contract["source_manifest_sha256"]
        )
        node, selection = initial_state(
            contract, json.loads((repo_root / contract["source_manifest"]).read_text())
        )
        publish_json(output / "selection.json", selection)
        progress["phase"] = "frozen_runtime_initialization"
        start = perf_counter()
        runtime = runtime_factory()
        runtime_seconds = perf_counter() - start
        if (
            runtime["model_checkpoint"] != contract["checkpoint"]
            or runtime["run_paths"] != contract["run_paths"]
        ):
            raise ValueError("runtime did not consume the bound model paths")
        gate = {
            "status": "existing_frozen_runtime_gates_passed",
            "input_sha256": inputs,
            "catalog_fingerprint": ring_catalog_fingerprint(runtime["model"].ring_catalog),
            "runtime_seconds": runtime_seconds,
        }
        publish_json(output / "runtime_gate.json", gate)
        commit()

        def enumerate_law(graph):
            law = enumerate_factorized_marked_law(runtime["model"], graph, contract["time_point"])
            return (
                tuple(m.executor_rule_name for m in law.marks),
                tuple(m.action for m in law.marks),
                tuple(m.probability for m in law.marks),
            )

        result = run_reference_profile(
            node,
            enumerate_law,
            runtime["system"],
            contract,
            output,
            snapshot_id=task["run_id"],
            commit_volume=commit,
            progress=progress,
        )
        result.update(
            {
                "code_revision": task["image_revision"]["commit"],
                "configuration": contract,
                "contract_sha256": task["contract_sha256"],
                "input_sha256": inputs,
                "selection": selection,
                "runtime_gate": gate,
                "run_id": task["run_id"],
                "elapsed_seconds": perf_counter() - started,
                "software": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "rdkit": rdBase.rdkitVersion,
                },
                "hardware": {
                    "machine": platform.machine(),
                    "cpu": platform.processor(),
                    "threads": torch.get_num_threads(),
                    "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                },
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        publish_json(output / "result.json", result)
        progress.update(phase="complete", complete=True, status=result["status"])
        return result
    except Exception as error:
        progress.update(phase="failed", error_type=type(error).__name__, error=str(error))
        publish_json(output / "failure.json", {**progress, "traceback": traceback.format_exc()})
        raise
    finally:
        stop.set()
        thread.join(timeout=30)
        publish_json(
            output / "progress.json", {**progress, "elapsed_seconds": perf_counter() - started}
        )
        commit()
