"""Local engineering gate, explicitly NOT a learned-generator chemistry benchmark.

The hand-declared two-branch fixture isolates delayed terminal credit. Every
edge is executed by the current executor. No benchmark source, winner, model
checkpoint, docking label, or learned preprocessing is used.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.continuation import (
    FiniteHorizonContinuation,
    ReferenceRow,
    continuation_decision,
    exact_doob_distribution,
)
from compose_v4.control.option_continuation import (
    OptionContinuationKernel,
    OptionState,
    sample_option_trajectory,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.rewrite.kernel import editing_v2_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, BondInsert


def engineering_fixture() -> tuple[OptionState, OptionContinuationKernel]:
    """A two-step growth/closure plateau under an explicitly synthetic reference."""
    graph = pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 6)
    source = OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), frozenset(range(4)), (), "pendant", 0),
        Lineage.initial(range(4)),
        "generic",
        0,
        2,
        "engineering-ring-plateau-v1",
    )

    def reference(graph):
        if graph.n_real_atoms == 4:
            return (
                ("atom_insert", "atom_insert"),
                (
                    AtomInsert(4, ELEMENT_TO_IDX["C"], 0, 3, ((3, 1),)),
                    AtomInsert(4, ELEMENT_TO_IDX["F"], 0, 0, ((3, 1),)),
                ),
                (0.5, 0.5),
            )
        if graph.atom_types[4] == ELEMENT_TO_IDX["C"]:
            return ("bond_insert",), (BondInsert(0, 4, 1),), (1.0,)
        return (
            ("atom_insert",),
            (AtomInsert(5, ELEMENT_TO_IDX["F"], 0, 0, ((0, 1),)),),
            (1.0,),
        )

    kernel = OptionContinuationKernel(
        reference, editing_v2_rewrite_system(), max_executor_applications=64
    )
    return source, kernel


def ring_terminal_weight(node: OptionState) -> float:
    """A topology predicate, not identity/similarity to any target molecule."""
    if node.remaining:
        return 0.0
    edges = int(np.count_nonzero(np.triu(node.graph.bonds, 1)))
    return float(edges - node.graph.n_real_atoms + 1 >= 1)


def evaluate_engineering_gate() -> dict:
    source, kernel = engineering_fixture()
    continuation = FiniteHorizonContinuation(
        kernel.row,
        ring_terminal_weight,
        OptionState.key,
        snapshot_id="synthetic-reference-ring-goal-v1",
        max_expansions=32,
        max_terminal_evaluations=32,
    )
    row = kernel.row(source)
    decision = continuation_decision(row, 2, continuation, fallback_values=(1, 1))
    values = np.asarray(decision.successor_values)
    before_cache = asdict(kernel.work)
    cached_decision = continuation_decision(row, 2, continuation, fallback_values=(1, 1))
    if decision != cached_decision or asdict(kernel.work) != before_cache:
        raise RuntimeError("cached decision changed its law or performed new executor work")
    trace = sample_option_trajectory(
        source,
        kernel,
        ring_terminal_weight,
        lambda node: 1,
        snapshot_id="synthetic-reference-ring-goal-v1",
        seed=0,
        max_expansions=32,
        max_terminal_evaluations=32,
    )
    cache_stats = {}
    matrix = ((0.3, 0.7), (0.4, 0.6))
    for memoize in (False, True):
        value = FiniteHorizonContinuation(
            lambda s: ReferenceRow((0, 1), matrix[s]),
            lambda s: (0.2, 1.0)[s],
            lambda s: s,
            snapshot_id="two-state-cache-fixture-v1",
            max_expansions=100,
            max_terminal_evaluations=100,
            memoize=memoize,
        )
        cache_stats[str(memoize)] = {"value": value.value(0, 5), "work": asdict(value.work)}
    checks = {
        "future_values_match_declared_paths": np.allclose(values, (1, 0), atol=1e-12),
        "future_guidance_crosses_plateau": decision.probabilities[0] > row.probabilities[0],
        "kl_bound": decision.kl <= 1 + 1e-10,
        "exact_doob_normalized": np.allclose(exact_doob_distribution(row, values), (1, 0)),
        "cached_uncached_value_parity": abs(
            cache_stats["False"]["value"] - cache_stats["True"]["value"]
        )
        < 1e-12,
        "complete_executable_sample": trace["status"] == "complete",
    }
    return {
        "checks": {name: bool(result) for name, result in checks.items()},
        "passed": all(checks.values()),
        "reference_success_probability": float(np.dot(row.probabilities, values)),
        "guided_success_probability": float(np.dot(decision.probabilities, values)),
        "decision": asdict(decision),
        "kernel_work_for_decision": before_cache,
        "cache_fixture": cache_stats,
        "sampled_trace": trace,
        "exact_source_state": {
            field: getattr(source.graph, field).tolist()
            for field in ("atom_types", "bonds", "formal_charges", "implicit_h_counts")
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    inputs = (
        "docs/CONTINUATION_CONTROLLER_IMPLEMENTATION.md",
        "src/compose_v4/control/continuation.py",
        "src/compose_v4/control/option_continuation.py",
        "src/compose_v4/control/option_selector.py",
        "src/compose_v4/control/macro_engine.py",
        "src/compose_v4/control/region_rewrite.py",
        "src/compose_v4/chem/molecular_graph.py",
        "src/compose_v4/chem/state.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/operators.py",
        "src/compose_v4/data/charge_policy.py",
        "src/compose_v4/gates/med_chem_gate.py",
        "src/compose_v4/experiments/continuation_gate.py",
        "tests/test_continuation.py",
        "tests/test_option_continuation.py",
    )
    started = perf_counter()
    result = evaluate_engineering_gate()
    report = {
        "schema_version": "continuation_engineering_gate_v1",
        "evidence_role": "computed engineering fixture, not a learned-model benchmark",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "input_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in inputs
        },
        "configuration": {
            "reference_law": "hand-declared two-branch fixture, not R_theta",
            "objective": "cycle rank >= 1 at complete two-step endpoint",
            "horizon": 2,
            "persistent_slots": 6,
            "kappa": 1.0,
            "exploration": 0.1,
            "max_executor_applications": 64,
            "max_expansions": 32,
            "max_terminal_evaluations": 32,
            "seed": 0,
            "seed_derivation": "literal fixed fixture seed",
            "device": "cpu",
            "precision": "float64 probability arrays",
            "workers": 1,
            "split": "synthetic engineering-only, not part of train/calibration/final panels",
            "source_count": 1,
            "exclusions": [],
            "external_oracle_calls": 0,
            "winner_inputs": [],
            "trained_models": [],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": platform.machine(),
        "result": result,
        "seconds": perf_counter() - started,
        "limitations": [
            "The fixture deliberately isolates delayed credit and does not measure discovery coverage.",
            "No frozen learned-model runtime was loaded; this timing cannot price T4.",
            "The exact bounded solver is a reference, not a scalable MCTS implementation.",
            "No docking-specific continuation estimator has been qualified or deployed.",
        ],
    }
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if json.dumps(json.loads(payload), sort_keys=True, indent=2, allow_nan=False) + "\n" != payload:
        raise RuntimeError("engineering receipt failed JSON round-trip")
    output = root / "diagnostics/continuation_controller/engineering_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(payload)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": result["checks"],
                "seconds": report["seconds"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
