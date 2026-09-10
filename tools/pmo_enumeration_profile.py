"""Bounded old/new serialization and production-law parity, with no oracle calls."""

import argparse
import ast
import hashlib
import inspect
import json
import os
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.chem import molecular_graph
from compose_v4.experiments.continuation_profile import encode_action, publish_json, sha256_file
from compose_v4.experiments.pmo_archive_pilot import load_contract
from compose_v4.experiments.production_successor_kernel import enumerate_factorized_marked_law
from compose_v4.experiments.t4_macro_beam import exact_archive_graph
from compose_v4.model.factorized_tracelet_rate_model import (
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import encode_state
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/compose_v4/chem/molecular_graph.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="8ef5eb2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    contract = load_contract(ROOT)
    if rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("requires contract-pinned RDKit")
    baseline_revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline], cwd=ROOT, text=True
    ).strip()
    baseline_source = subprocess.check_output(
        ["git", "show", f"{baseline_revision}:{SOURCE}"], cwd=ROOT, text=True
    )
    old_definition = next(
        n
        for n in ast.parse(baseline_source).body
        if isinstance(n, ast.FunctionDef) and n.name == "molecular_graph_to_smiles"
    )
    namespace = dict(vars(molecular_graph))
    # Trusted local committed production function only; no downloaded code.
    exec(compile(ast.Module(body=[old_definition], type_ignores=[]), "baseline", "exec"), namespace)  # noqa: S102
    function = molecular_graph.molecular_graph_to_smiles
    old_code, new_code = namespace[function.__name__].__code__, function.__code__
    new_source = inspect.getsource(function)
    # All imported references point to this same function object. Swapping only
    # its code in this offline process compares real call sites without copying
    # executor/model logic or modifying any running job.
    states = [exact_archive_graph(r) for r in contract["roots"][:2]]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_heteroatom_scan=True,
        enable_cycle_ops=True,
        editing_process_semantics=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=molecular_graph.ORGANIC_VOCABULARY,
    ).eval()
    system = editing_v2_semantic_rewrite_system()
    rows = []
    try:
        for index, state in enumerate(states):
            outputs, times, products, validity = {}, {"before": [], "after": []}, {}, {}
            for repeat in range(args.repeats):
                order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
                for arm in order:
                    function.__code__ = old_code if arm == "before" else new_code
                    started = perf_counter()
                    law = enumerate_factorized_marked_law(model, state, 0.5)
                    elapsed = perf_counter() - started
                    times[arm].append(elapsed)
                    row = [
                        (encode_action(m.executor_rule_name, m.action), m.probability)
                        for m in law.marks
                    ]
                    if arm in outputs and row != outputs[arm]:
                        raise ValueError("repeated marked law changed")
                    outputs[arm] = row
                    if repeat == 0:
                        executed = [
                            system.apply(state, m.executor_rule_name, m.action) for m in law.marks
                        ]
                        products[arm] = [
                            (encode_state(g), canonical_state_key(g)) for g in executed
                        ]
                        validity[arm] = [molecular_graph.is_rdkit_valid(g) for g in executed]
                    print(
                        f"root={index} {arm} repeat={repeat} seconds={elapsed:.3f} "
                        f"marks={len(row)}",
                        flush=True,
                    )
            if outputs["before"] != outputs["after"] or products["before"] != products["after"]:
                raise ValueError(
                    "changed ordered marks, probabilities, exact products or canonical identities"
                )
            if validity["before"] != validity["after"] or not all(validity["after"]):
                raise ValueError("changed or invalid successor validity")
            rows.append(
                {
                    "root_index": index,
                    "source": encode_state(state),
                    "source_sha256": digest(encode_state(state)),
                    "marks": len(outputs["after"]),
                    "ordered_mark_probability_sha256": digest(outputs["after"]),
                    "exact_and_canonical_products_sha256": digest(products["after"]),
                    "parity": True,
                    "seconds": times,
                    "speedup": median(times["before"]) / median(times["after"]),
                }
            )
    finally:
        function.__code__ = new_code
    result = {
        "schema_version": "pmo_enumeration_profile_v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "baseline_revision": baseline_revision,
        "baseline_source_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
        "candidate_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "candidate_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "source_path": SOURCE,
        "source_sha256": sha256_file(ROOT / SOURCE),
        "production_kernel_sha256": sha256_file(
            ROOT / "src/compose_v4/experiments/production_successor_kernel.py"
        ),
        "new_function_sha256": hashlib.sha256(new_source.encode()).hexdigest(),
        "script_sha256": sha256_file(Path(__file__)),
        "contract_sha256": contract["contract_sha256"],
        "configuration": {
            "roots": [0, 1],
            "repeats": args.repeats,
            "seed": 0,
            "time": 0.5,
            "hidden_dim": 12,
            "message_passing_steps": 1,
            "ring_catalog": "empty",
            "precision": "float32",
            "threads": 1,
            "device": "cpu",
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "rows": rows,
        "oracle_calls": 0,
        "exclusions": [],
        "interpretation": "Two exact development roots, initialized small model; mechanical law parity, not learned-model quality or Modal end-to-end speed. Unprofiled alternating paired wall times.",
    }
    publish_json(args.output, result)
    print(json.dumps({"parity": True, "speedups": [r["speedup"] for r in rows]}), flush=True)


if __name__ == "__main__":
    main()
