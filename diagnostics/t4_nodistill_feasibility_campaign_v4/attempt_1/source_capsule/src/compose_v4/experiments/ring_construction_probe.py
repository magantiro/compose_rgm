"""Bounded executor witnesses, not a learned-controller performance benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    ORGANIC_VOCABULARY,
    MolecularGraph,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_valid_state, pad_molecular_graph
from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.macro_engine import (
    build_fused_ring_exact,
    build_ring_system_exact,
    ring_systems,
)
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.control.region_rewrite import graph_connected
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    canonical_bytes,
    encode_action,
    publish_json,
    sha256_file,
)
from compose_v4.gates.med_chem_gate import is_executable
from compose_v4.rewrite.factorized_fiber import _factorized_candidates
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.trace_shard import encode_state

CONFIG = {
    "source_smiles": "c1ccccc1",
    "topologies": ["pendant", "fused"],
    "size": 6,
    "ring_state": "aromatic",
    "composition": "carbon_rich",
    "persistent_slots": 48,
    "max_active_atoms": 40,
    "max_executor_applications": 256,
    "safety_seconds": 30.0,
    "max_aromatise": 24,
    "anchor_rank": 0,
    "refine": 0,
    "reference": "uniform neutral primitive candidates, ORGANIC_VOCABULARY",
    "seed": None,
    "seed_reason": "deterministic enumeration and first-site descriptor ranking",
    "workers": 1,
    "precision": "float64 uniform candidate weights; integer graph coordinates",
    "split": "synthetic engineering fixture only",
    "winner_inputs": [],
    "oracle_calls": 0,
    "exclusions": [],
}


def topology_witness(source: MolecularGraph, product: MolecularGraph, topology: str) -> dict:
    """Audit this six-carbon construction using exact persistent atom slots.

    This is a witness predicate for the fixed monocyclic fixture, not a general
    SSSR-based classifier or an inference of atom lineage from endpoint SMILES.
    """
    if topology not in CONFIG["topologies"]:
        raise ValueError(f"unknown probe topology: {topology!r}")
    old = set(np.flatnonzero(is_element(source.atom_types)).tolist())
    real = set(np.flatnonzero(is_element(product.atom_types)).tolist())
    new = real - old
    old_slots = sorted(old)
    retained = old <= real and np.array_equal(
        source.atom_types[old_slots], product.atom_types[old_slots]
    )
    retained &= np.array_equal(
        source.bonds[np.ix_(old_slots, old_slots)] != 0,
        product.bonds[np.ix_(old_slots, old_slots)] != 0,
    )
    attachments = sorted((a, b) for a in old for b in new if product.bonds[a, b])
    shared = {a for a, _ in attachments}
    cycle = sorted(new if topology == "pendant" else new | shared)
    adjacency = product.bonds[np.ix_(cycle, cycle)] != 0
    cycle_ok = len(cycle) == CONFIG["size"] and bool(np.all(adjacency.sum(axis=0) == 2))
    if cycle:
        reached, pending = {0}, [0]
        while pending:
            for v in np.flatnonzero(adjacency[pending.pop()]):
                if int(v) not in reached:
                    reached.add(int(v))
                    pending.append(int(v))
        cycle_ok &= len(reached) == len(cycle)
    if topology == "pendant":
        attachment_ok = len(new) == 6 and len(attachments) == 1
    else:
        attachment_ok = len(new) == 4 and len(attachments) == 2 and len(shared) == 2
        if attachment_ok:
            u, v = sorted(shared)
            attachment_ok &= bool(source.bonds[u, v])
    before, after = (Chem.MolFromSmiles(canonical_state_key(g)) for g in (source, product))
    if before is None or after is None:
        raise ValueError("executor state could not be inspected as a molecule")

    def rank(m):
        return m.GetNumBonds() - m.GetNumAtoms() + len(Chem.GetMolFrags(m))

    deltas = {
        "heavy_atoms": after.GetNumHeavyAtoms() - before.GetNumHeavyAtoms(),
        "cycle_rank": rank(after) - rank(before),
        "ring_systems": len(ring_systems(after)) - len(ring_systems(before)),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(after)
        - rdMolDescriptors.CalcNumAromaticRings(before),
    }
    carbon = ELEMENT_TO_IDX["C"]
    checks = {
        "initial_atoms_and_connectivity_retained": bool(retained),
        "exact_new_six_cycle": bool(cycle_ok),
        "requested_attachment": bool(attachment_ok),
        "six_carbon_cycle": bool(cycle) and bool(np.all(product.atom_types[cycle] == carbon)),
        "cycle_rank_gain_one": deltas["cycle_rank"] == 1,
        "ring_system_delta": deltas["ring_systems"] == int(topology == "pendant"),
        "aromatic_ring_gain_one": deltas["aromatic_rings"] == 1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": deltas,
        "new_slots": sorted(new),
        "shared_slots": sorted(shared) if topology == "fused" else [],
        "cycle_slots": cycle,
        "attachment_edges": attachments,
    }


class ProbeRuntime:
    """Existing enumerator/executor with exact-state caching and durable receipts."""

    def __init__(self, source: MolecularGraph, limit: int, safety_seconds: float):
        if not math.isfinite(safety_seconds) or safety_seconds < 0:
            raise ValueError("safety_seconds must be finite and nonnegative")
        self.source = source
        self.system = editing_v2_rewrite_system()
        self.meter = ExecutorMeter(limit)
        self.deadline = perf_counter() + safety_seconds
        self.cache: dict[tuple, tuple] = {}
        self.graphs = {exact_graph_key(source): source}
        self.parents: dict[tuple, tuple] = {}
        self.enumerated_counts: list[int] = []
        self.cache_hits = 0

    def check_deadline(self) -> None:
        if perf_counter() >= self.deadline:
            raise TimeoutError("ring probe safety deadline reached; comparison incomplete")

    def enumerate(self, graph: MolecularGraph) -> tuple:
        self.check_deadline()
        key = exact_graph_key(graph)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        candidates = tuple(
            _factorized_candidates(graph, allow_bond_reroute=True, vocabulary=ORGANIC_VOCABULARY)
        )
        self.check_deadline()
        families = tuple(rule for rule, _ in candidates)
        actions = tuple(action for _, action in candidates)
        weights = tuple(1.0 / len(candidates) for _ in candidates)
        self.cache[key] = families, actions, weights
        self.enumerated_counts.append(len(candidates))
        return self.cache[key]

    def apply(self, graph: MolecularGraph, index: int) -> MolecularGraph | None:
        self.check_deadline()
        families, actions, _ = self.enumerate(graph)
        try:
            product = self.system.apply(graph, families[index], actions[index])
        except InvalidRewrite:
            return None
        if not (
            0 < product.n_real_atoms <= CONFIG["max_active_atoms"]
            and is_valid_state(product)
            and graph_connected(product)
            and charge_policy_preserved(graph, product)
        ):
            raise RuntimeError("executed candidate violates frozen molecular support")
        key = exact_graph_key(product)
        self.graphs[key] = product
        if key != exact_graph_key(self.source):
            self.parents.setdefault(
                key, (exact_graph_key(graph), encode_action(families[index], actions[index]))
            )
        return product

    def witness_path(self, endpoint: str) -> tuple[MolecularGraph, list[dict]]:
        matches = [
            key for key, graph in self.graphs.items() if canonical_state_key(graph) == endpoint
        ]
        if not matches:
            raise RuntimeError("constructor endpoint has no executed state witness")
        key = matches[0]
        product = self.graphs[key]
        path, seen = [], set()
        while key != exact_graph_key(self.source):
            if key in seen:
                raise RuntimeError("execution witness contains a cyclic parent chain")
            seen.add(key)
            parent, mark = self.parents[key]
            path.append(
                {
                    "mark": mark,
                    "source": encode_state(self.graphs[parent]),
                    "product": encode_state(self.graphs[key]),
                }
            )
            key = parent
        return product, list(reversed(path))


def run_probe(topology: str, *, limit: int = 256, safety_seconds: float = 30.0) -> dict:
    if topology not in CONFIG["topologies"]:
        raise ValueError(f"unknown probe topology: {topology!r}")
    source = pad_molecular_graph(smiles_to_molecular_graph(CONFIG["source_smiles"]), 48)
    runtime = ProbeRuntime(source, limit, safety_seconds)
    started = perf_counter()
    constructor = build_ring_system_exact if topology == "pendant" else build_fused_ring_exact
    result: dict = {}
    witness: dict | None = None
    path: list[dict] = []
    status = "constructor_unsat"
    try:
        with runtime.meter.instrument():
            result = constructor(
                runtime.enumerate,
                runtime.apply,
                canonical_state_key,
                is_executable,
                source,
                size=CONFIG["size"],
                composition=CONFIG["composition"],
                state=CONFIG["ring_state"],
                anchor_rank=CONFIG["anchor_rank"],
                max_aromatise=CONFIG["max_aromatise"],
                refine=CONFIG["refine"],
            )
        if result.get("smiles"):
            product, path = runtime.witness_path(result["smiles"])
            witness = topology_witness(source, product, topology)
        if result.get("status") == "OK":
            status = "verified_witness" if witness and witness["passed"] else "topology_mismatch"
    except ContinuationBudgetExceeded as error:
        status, result = "budget_abstention", {"reason": str(error)}
    except TimeoutError as error:
        status, result = "timeout_incomplete", {"reason": str(error)}
    if sorted(r["call_index"] for r in runtime.meter.attempts) != list(range(runtime.meter.calls)):
        raise RuntimeError("executor receipts do not cover every entered application")
    return {
        "topology": topology,
        "status": status,
        "constructor": result,
        "topology_witness": witness,
        "execution_path": path,
        "initial_state": encode_state(source),
        "work": {
            "executor_applications": runtime.meter.calls,
            "executor_limit": limit,
            "safety_seconds": safety_seconds,
            "enumerated_candidates_per_state": runtime.enumerated_counts,
            "row_cache_hits": runtime.cache_hits,
            "wall_seconds": perf_counter() - started,
            "attempt_status_counts": dict(
                sorted(Counter(r["status"] for r in runtime.meter.attempts).items())
            ),
        },
        "executor_receipts": runtime.meter.attempts,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    source_names = subprocess.check_output(
        ["git", "ls-files", "src", "configs"], cwd=root, text=True
    ).splitlines()
    inputs = sorted(
        set(source_names)
        | {
            "docs/RING_CONSTRUCTION_PROBE.md",
            "src/compose_v4/experiments/ring_construction_probe.py",
            "tests/test_ring_construction_probe.py",
            "tools/ring_construction_probe.py",
        }
    )
    results = [run_probe(topology) for topology in CONFIG["topologies"]]
    report = {
        "schema_version": "ring_construction_probe_v1",
        "evidence_role": "synthetic executor witnesses, not integrated-controller qualification",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "input_sha256": {name: sha256_file(root / name) for name in inputs},
        "configuration": CONFIG,
        "configuration_sha256": hashlib.sha256(canonical_bytes(CONFIG)).hexdigest(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": platform.machine(),
        "results": results,
        "accepted_requests": sum(r["status"] == "verified_witness" for r in results),
        "attempted_requests": len(results),
        "population_coverage": None,
        "new_docking_calls": 0,
        "limitations": [
            "Uniform primitive weights, not learned R_theta or a KL-controlled policy.",
            "Existing descriptor constructors are not registered compound options in the region controller.",
            "One synthetic source cannot estimate discovery coverage, diversity, or efficiency.",
            "The primitive enumeration lane omits correlated RingCore tracelets.",
        ],
    }
    output = root / "diagnostics/ring_construction_probe/result.json"
    digest = publish_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "results": [
                    {"topology": r["topology"], "status": r["status"], "work": r["work"]}
                    for r in results
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
