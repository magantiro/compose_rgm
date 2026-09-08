"""Save small model-free expansion witnesses and all negative requests."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.option_continuation import (
    OptionContinuationKernel,
    OptionState,
    sample_option_trajectory,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.control.ring_expansion import (
    EXPAND_RING_OPTION,
    ExpansionProgress,
    completed_expansion,
)
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    publish_json,
    sha256_file,
    state_payload,
)
from compose_v4.gates.med_chem_gate import validity_reasons
from compose_v4.rewrite.factorized_fiber import _factorized_candidates
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import CycleOpenEdge
from compose_v4.rewrite.trace_shard import decode_state

CONFIG = {
    "requests": [
        "isolated_5_to_6",
        "isolated_6_to_7",
        "isolated_7_to_8",
        "isolated_8_to_9",
        "fused_7_to_8",
        "fused_8_to_9",
    ],
    "source_smiles": ["C1CCCC1", "C1CCCCC1", "C1CCCCCC1", "c1ccc2c(c1)CCCCC2"],
    "reference": "uniform neutral primitive descriptors with semantic cycle opening; not R_theta",
    "seed": 0,
    "seed_derivation": "same explicit seed for each request; no search over seeds",
    "max_public_executor_calls_per_request": 128,
    "safety_seconds_per_request": 30,
    "persistent_slots": 48,
    "max_active_atoms": 40,
    "workers": 1,
    "precision": "float64 probabilities; integer exact graph states",
    "split": "synthetic engineering fixtures only; no training or evaluation panel",
    "winner_inputs": [],
    "docking_calls": 0,
    "exclusions": [],
}


def fixture_law(graph):
    """Declared engineering support, not a target-conditioned or learned law."""
    marks = [
        ("cycle_open", CycleOpenEdge(a.a, a.b)) if rule == "bond_delete" else (rule, a)
        for rule, a in _factorized_candidates(
            graph, allow_bond_reroute=True, vocabulary=ORGANIC_VOCABULARY
        )
    ]
    return tuple(r for r, _ in marks), tuple(a for _, a in marks), (1 / len(marks),) * len(marks)


def initial(smiles="C1CCCCCC1", graph=None):
    if graph is None:
        graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), CONFIG["persistent_slots"])
    slots = frozenset(int(i) for i in np.flatnonzero(is_element(graph.atom_types)))
    return OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), slots, (), "multi", 0),
        Lineage.initial(slots),
        EXPAND_RING_OPTION,
        0,
        3,
        "synthetic-expansion",
        expansion_progress=ExpansionProgress(),
    )


def describe(graph):
    smiles = canonical_state_key(graph)
    mol = Chem.MolFromSmiles(smiles)
    return {
        "smiles": smiles,
        "heavy_atoms": mol.GetNumAtoms(),
        "cycle_rank": mol.GetNumBonds() - mol.GetNumAtoms() + len(Chem.GetMolFrags(mol)),
        "ring_sizes": sorted(len(r) for r in mol.GetRingInfo().AtomRings()),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "gate_reasons": validity_reasons(smiles),
    }


def run_request(request, node):
    if node is None:
        return {"request": request, "status": "dependency_blocked", "executor_calls": 0}, None
    started = perf_counter()
    enumerated = []

    def law(graph):
        if perf_counter() - started > CONFIG["safety_seconds_per_request"]:
            raise TimeoutError("expansion audit deadline exceeded")
        row = fixture_law(graph)
        enumerated.append(len(row[0]))
        return row

    process = OptionContinuationKernel(
        law, editing_v2_rewrite_system(), max_executor_applications=128
    )
    meter = ExecutorMeter(CONFIG["max_public_executor_calls_per_request"])
    trajectory = None
    try:
        with meter.instrument():
            trajectory = sample_option_trajectory(
                node,
                process,
                lambda _: 1.0,
                lambda _: 1.0,
                snapshot_id="expansion-fixture-v1",
                seed=CONFIG["seed"],
                max_expansions=0,
                max_terminal_evaluations=0,
                estimator="reference",
            )
        status = trajectory["status"]
    except (ContinuationBudgetExceeded, TimeoutError) as error:
        status = type(error).__name__
    if sorted(r["call_index"] for r in meter.attempts) != list(range(meter.calls)):
        raise ValueError("expansion executor ledger is incomplete")
    product = None
    if status == "complete":
        product = decode_state(trajectory["trace"][-1]["exact_state"])
        progress = ExpansionProgress.from_payload(trajectory["final_expansion_progress"])
        if not completed_expansion(node.graph, product, progress):
            raise ValueError("completed trajectory lacks the exact edge-subdivision witness")
    rejected_gate_products = [
        describe(decode_state(r["product"]))
        for r in meter.attempts
        if r["status"] == "executed"
        and validity_reasons(canonical_state_key(decode_state(r["product"])))
    ]
    return {
        "request": request,
        "status": status,
        "initial_state": state_payload(node),
        "before": describe(node.graph),
        "after": describe(product) if product else None,
        "trajectory": trajectory,
        "executor_calls": meter.calls,
        "executor_receipts": meter.attempts,
        "enumerated_marks_per_row": enumerated,
        "rejected_gate_products": rejected_gate_products,
        "wall_seconds": perf_counter() - started,
    }, product


def main():
    root = Path(__file__).resolve().parents[1]
    results = []
    for request, smiles in zip(CONFIG["requests"][:3], CONFIG["source_smiles"][:3]):
        row, product = run_request(request, initial(smiles))
        results.append(row)
    row, _ = run_request("isolated_8_to_9", initial(graph=product) if product is not None else None)
    results.append(row)
    row, product = run_request("fused_7_to_8", initial(CONFIG["source_smiles"][3]))
    results.append(row)
    row, _ = run_request("fused_8_to_9", initial(graph=product) if product is not None else None)
    results.append(row)
    inputs = subprocess.check_output(["git", "ls-files", "src"], cwd=root, text=True).splitlines()
    inputs += [
        "docs/RING_EXPANSION_OPTION.md",
        "tools/ring_expansion_audit.py",
        "tests/test_ring_expansion.py",
    ]
    report = {
        "schema_version": "ring_expansion_audit_v1",
        "configuration": CONFIG,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "input_sha256": {p: sha256_file(root / p) for p in sorted(set(inputs))},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": platform.machine(),
        "results": results,
        "completed": sum(r["status"] == "complete" for r in results),
        "requested": len(results),
        "attempted": sum(r["status"] != "dependency_blocked" for r in results),
        "population_coverage": None,
        "limitations": [
            "Synthetic model-free witnesses, not production discovery or docking evidence.",
            "The isolated macrocycle gate remains frozen, including its failed request.",
            "No T4 launcher enablement or ring-option weight change.",
        ],
    }
    path = root / "diagnostics/ring_expansion/result.json"
    digest = publish_json(path, report)
    print(
        json.dumps(
            {
                "output": str(path),
                "sha256": digest,
                "results": [
                    {k: r[k] for k in ("request", "status", "executor_calls")} for r in results
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
