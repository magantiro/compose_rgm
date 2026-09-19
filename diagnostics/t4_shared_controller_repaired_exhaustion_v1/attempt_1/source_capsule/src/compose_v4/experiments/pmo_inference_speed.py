"""Zero-oracle export and paired acceleration check on exact archived states."""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import threading
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import (
    encode_action,
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.inference_package import (
    CACHE_SOURCES,
    dependency_sources,
    load_package,
    software,
    write_package,
)
from compose_v4.experiments.production_successor_kernel import (
    _default_rewrite_system,
    enumerate_factorized_marked_law,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.rewrite.trace_shard import decode_state, encode_state

CONTRACT = "configs/pmo_inference_speed.json"
PREPARED = "diagnostics/pmo_inference_speed/prepared.json"
APP = "modal_apps/pmo_inference_speed_app.py"


def read_contract(root):
    contract = json.loads((root / CONTRACT).read_text())
    if (
        identity({k: v for k, v in contract.items() if k != "contract_sha256"})
        != contract["contract_sha256"]
    ):
        raise ValueError("inference speed contract hash mismatch")
    verify_file(root / PREPARED, contract["prepared_sha256"])
    verify_file(root / "docs/PMO_INFERENCE_SPEED.md", contract["protocol_sha256"])
    if (
        contract["oracle_calls"] != 0
        or contract["repeats"] != 3
        or set(contract["cache_sources"]) != set(CACHE_SOURCES)
    ):
        raise ValueError("inference speed scope changed")
    return contract


@contextmanager
def session(task, root, artifact_root, volume, validate):
    validate(task["image_revision"])
    verify_file(root / APP, task["app_sha256"])
    contract = read_contract(root)
    if (
        task["contract_sha256"] != contract["contract_sha256"]
        or identity({k: v for k, v in task.items() if k != "run_id"}) != task["run_id"]
    ):
        raise ValueError("inference speed launch identity mismatch")
    output = artifact_root / "pmo_inference_speed" / task["run_id"]
    volume.reload()
    progress, stop = {"phase": "initialization", "oracle_calls": 0}, threading.Event()
    started = perf_counter()

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {**progress, "seconds": perf_counter() - started, "at": _stamp()},
            )
            volume.commit()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield contract, output, progress
    except Exception as error:
        seal(output / "failure.json", {"error": repr(error), "progress": progress, "at": _stamp()})
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        volume.commit()


def law_values(model, graph):
    law = enumerate_factorized_marked_law(model, graph, 0.5)
    return law, [(encode_action(m.executor_rule_name, m.action), m.probability) for m in law.marks]


def export_remote(task, root, artifact_root, volume, validate, runtime_factory):
    from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
    from compose_v4.rewrite.editing_v2_process_identity import require_editing_process_v2_identity

    with session(task, root, artifact_root, volume, validate) as (contract, output, progress):
        if (output / "complete.json").exists():
            return unseal(output / "complete.json")
        require_editing_process_v2_identity(contract["frozen_process_sha256"])
        inputs = contract["reference_inputs"]
        for value in inputs.values():
            verify_file(Path(value["path"]), value["sha256"])
        sources = dependency_sources(task["image_revision"]["serialized_sources"])
        package_key = identity(
            {
                "inputs": inputs,
                "sources": sources,
                "software": software(),
                "exporter_sha256": sha256_file(
                    root / "src/compose_v4/experiments/inference_package.py"
                ),
            }
        )
        directory = artifact_root / "inference_packages" / package_key
        start = perf_counter()
        if (directory / "qualified_export.json").exists():
            result = unseal(directory / "qualified_export.json")
            load_package(directory, manifest_sha256=result["manifest_sha256"], repo_root=root)
            seal(output / "complete.json", result)
            return result
        progress.update(phase="original_runtime_load")
        runtime = runtime_factory()
        startup_seconds = perf_counter() - start
        model = runtime["model"]
        checkpoint = torch.load(
            inputs["checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        tensor_hash = state_dict_semantic_sha256(model.state_dict())
        if tensor_hash != state_dict_semantic_sha256(checkpoint["selected_model_state"]):
            raise ValueError("export does not contain the original selected checkpoint tensors")
        progress.update(phase="export_reload")
        if not (directory / "manifest.json").exists():
            write_package(
                directory,
                model,
                provenance={
                    "dependency_sources": sources,
                    "reference_inputs": inputs,
                    "frozen_process_sha256": contract["frozen_process_sha256"],
                    "producer_revision": task["image_revision"]["commit"],
                },
            )
        manifest_hash = sha256_file(directory / "manifest.json")
        start = perf_counter()
        loaded, manifest = load_package(directory, manifest_sha256=manifest_hash, repo_root=root)
        if (
            manifest["tensor_sha256"] != tensor_hash
            or manifest["provenance"]["reference_inputs"] != inputs
        ):
            raise ValueError("partial export does not match authenticated reference inputs")
        load_seconds = perf_counter() - start
        graph = decode_state(json.loads((root / PREPARED).read_text())["states"][0]["graph"])
        progress.update(phase="export_law_parity")
        _, original = law_values(model, graph)
        _, restored = law_values(loaded, graph)
        if original != restored:
            raise ValueError("inference export/reload changed the production marked law")
        result = {
            "schema_version": "inference_export_result_v1",
            "status": "verified_export",
            "package_path": str(directory),
            "manifest_sha256": manifest_hash,
            "tensor_sha256": manifest["tensor_sha256"],
            "original_runtime_seconds": startup_seconds,
            "package_load_seconds": load_seconds,
            "roundtrip_law_sha256": identity(original),
            "marks": len(original),
            "oracle_calls": 0,
            "run_id": task["run_id"],
            "contract_sha256": contract["contract_sha256"],
            "at": _stamp(),
        }
        seal(directory / "qualified_export.json", result)
        seal(output / "complete.json", result)
        return result


def measure_row(model, state, baseline_source, repeats, progress):
    from compose_v4.chem import molecular_graph
    from compose_v4.rewrite.kernel import canonical_state_key

    definition = next(
        n
        for n in ast.parse(baseline_source).body
        if isinstance(n, ast.FunctionDef) and n.name == "molecular_graph_to_smiles"
    )
    namespace = dict(vars(molecular_graph))
    # Trusted, physically pinned committed source, diagnostic process only.
    exec(  # noqa: S102
        compile(ast.Module(body=[definition], type_ignores=[]), "committed_baseline", "exec"),
        namespace,
    )
    function = molecular_graph.molecular_graph_to_smiles
    baseline_code, candidate_code = namespace[function.__name__].__code__, function.__code__
    times, outputs, products = {"baseline": [], "cached": []}, {}, {}
    graph, system = decode_state(state["graph"]), _default_rewrite_system(model)
    try:
        for repeat in range(repeats):
            for arm in ("baseline", "cached") if repeat % 2 == 0 else ("cached", "baseline"):
                function.__code__ = baseline_code if arm == "baseline" else candidate_code
                progress.update(phase="enumeration", state=state["id"], repeat=repeat, arm=arm)
                start = perf_counter()
                law, values = law_values(model, graph)
                times[arm].append(perf_counter() - start)
                if arm in outputs and values != outputs[arm]:
                    raise ValueError("repeated marked probabilities changed")
                outputs[arm] = values
                if repeat == 0:
                    progress.update(phase="exact_product_comparison")
                    executed = [
                        system.apply(graph, m.executor_rule_name, m.action) for m in law.marks
                    ]
                    if not all(molecular_graph.is_rdkit_valid(g) for g in executed):
                        raise ValueError("enumeration exposed an invalid product")
                    products[arm] = [(encode_state(g), canonical_state_key(g)) for g in executed]
        if outputs["baseline"] != outputs["cached"] or products["baseline"] != products["cached"]:
            raise ValueError("cache changed marks, probabilities or exact/canonical products")
    finally:
        function.__code__ = candidate_code
    return {
        "state_id": state["id"],
        "source_sha256": identity(state["graph"]),
        "marks": len(outputs["cached"]),
        "law_sha256": identity(outputs["cached"]),
        "products_sha256": identity(products["cached"]),
        "parity": True,
        "seconds": times,
        "speedup": median(times["baseline"]) / median(times["cached"]),
    }


def probe_remote(task, root, artifact_root, volume, validate):
    with session(task, root, artifact_root, volume, validate) as (contract, output, progress):
        if (output / "complete.json").exists():
            return unseal(output / "complete.json")
        export = task["export"]
        if (
            export["status"] != "verified_export"
            or export["contract_sha256"] != contract["contract_sha256"]
        ):
            raise ValueError("unqualified inference export")
        directory = Path(export["package_path"])
        if directory.parent != artifact_root / "inference_packages":
            raise ValueError("inference package outside authorized volume prefix")
        if unseal(directory / "qualified_export.json") != export:
            raise ValueError("export receipt differs from immutable source")
        progress.update(phase="package_load")
        start = perf_counter()
        model, manifest = load_package(
            directory,
            manifest_sha256=export["manifest_sha256"],
            repo_root=root,
            probe_source_changes=contract["cache_sources"],
        )
        load_seconds = perf_counter() - start
        data = json.loads((root / PREPARED).read_text())
        if (
            hashlib.sha256(data["baseline_source"].encode()).hexdigest()
            != contract["baseline_serializer_sha256"]
        ):
            raise ValueError("baseline serializer source changed")
        rows = []
        for i, state in enumerate(data["states"]):
            path = output / f"rows/{i}.json"
            row = (
                unseal(path)
                if path.exists()
                else measure_row(
                    model, state, data["baseline_source"], contract["repeats"], progress
                )
            )
            if row["source_sha256"] != identity(state["graph"]) or not row["parity"]:
                raise ValueError("cached probe row belongs to another exact state")
            seal(path, row)
            volume.commit()
            rows.append(row)
        result = {
            "schema_version": "inference_speed_result_v1",
            "status": "pass" if all(r["speedup"] > 1 for r in rows) else "no_speed_gain",
            "oracle_calls": 0,
            "rows": rows,
            "package_load_seconds": load_seconds,
            "export": export,
            "package_tensor_sha256": manifest["tensor_sha256"],
            "cache_sources": contract["cache_sources"],
            "software": software(),
            "hardware": {
                "platform": platform.platform(),
                "device": "cpu",
                "threads": 1,
                "precision": "float32",
            },
            "code_revision": task["image_revision"]["commit"],
            "contract_sha256": contract["contract_sha256"],
            "run_id": task["run_id"],
            "at": _stamp(),
            "scope": "Three exact archived states, frozen selected model; no search-quality or universal equivalence claim",
        }
        seal(output / "complete.json", result)
        return result
