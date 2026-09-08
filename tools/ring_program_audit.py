"""One small, saved capability audit; no checkpoint, docking, or winner inputs."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.macro_engine import slot_ring_systems
from compose_v4.control.option_continuation import (
    OptionContinuationKernel,
    OptionState,
    sample_option_trajectory,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.control.ring_program import (
    RingProgress,
    RingSpec,
    completed_construction,
    default_ring_options,
    real_slots,
    ring_spec,
)
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, state_payload
from compose_v4.experiments.ring_construction_probe import ProbeRuntime
from compose_v4.gates.med_chem_gate import is_valid
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = default_ring_options() + (RingSpec("pendant", 6, (5, 1, 0), "nonaromatic", 1).option,)
CONFIG = {
    "source_smiles": "c1ccccc1",
    "options": list(OPTIONS),
    "persistent_slots": 48,
    "max_active_atoms": 40,
    "seed": 0,
    "seed_derivation": "same explicit seed per request; no retries",
    "reference": "uniform neutral primitive law, broad-organic vocabulary; NOT R_theta",
    "max_public_executor_calls_per_request": 256,
    "deadline_seconds_per_request": 30,
    "workers": 1,
    "precision": "float64 probabilities; integer molecular states",
    "split": "synthetic engineering fixture; no training/calibration/final-test panel",
    "winner_inputs": [],
    "oracle_calls": 0,
    "exclusions": [],
}


def run_request(option):
    spec = ring_spec(option)
    graph = pad_molecular_graph(smiles_to_molecular_graph(CONFIG["source_smiles"]), 48)
    slots = real_slots(graph)
    node = OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), slots, (), "multi", 0),
        Lineage.initial(slots),
        option,
        0,
        spec.horizon,
        option,
        ring_progress=RingProgress(),
    )
    runtime = ProbeRuntime(graph, 256, 30)
    kernel = OptionContinuationKernel(
        runtime.enumerate, runtime.system, max_executor_applications=256
    )
    started = perf_counter()
    try:
        with runtime.meter.instrument():
            trajectory = sample_option_trajectory(
                node,
                kernel,
                lambda _: 1,
                lambda _: 1,
                snapshot_id="parameterized-ring-fixture-v1",
                seed=0,
                estimator="reference",
                max_expansions=0,
                max_terminal_evaluations=0,
            )
    except (ContinuationBudgetExceeded, TimeoutError) as error:
        trajectory = {"status": type(error).__name__, "reason": str(error)}
    witness = None
    if trajectory["status"] == "complete":
        progress = RingProgress.from_payload(trajectory["final_ring_progress"])
        endpoint = decode_state(trajectory["trace"][-1]["exact_state"])
        at_closure = decode_state(trajectory["trace"][spec.growth]["exact_state"])
        cycle = progress.cycle(spec)
        atoms = endpoint.atom_types[list(cycle)]
        mol = Chem.MolFromSmiles(trajectory["endpoint"])
        checks = {
            "exact_construction_witness": completed_construction(graph, at_closure, progress, spec),
            "all_committed_states_valid": all(
                is_valid(step["after"]) for step in trajectory["trace"]
            ),
            "expected_ring_sizes": sorted(map(len, mol.GetRingInfo().AtomRings()))
            == sorted((6, spec.size)),
            "expected_ring_system_delta": len(slot_ring_systems(endpoint))
            - len(slot_ring_systems(graph))
            == int(spec.topology == "pendant"),
            "no_bridgeheads_or_spiro": rdMolDescriptors.CalcNumSpiroAtoms(mol)
            == rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            == 0,
        }
        if not all(checks.values()):
            raise ValueError(f"{option}: completed output failed independent checks: {checks}")
        witness = {
            "checks": checks,
            "cycle_slots": list(cycle),
            "realized_ring_element_codes": [int(a) for a in atoms],
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "ring_sizes": sorted(map(len, mol.GetRingInfo().AtomRings())),
        }
    return {
        "option": option,
        "initial": state_payload(node),
        "trajectory": trajectory,
        "witness": witness,
        "executor_calls": runtime.meter.calls,
        "executor_receipts": runtime.meter.attempts,
        "seconds": perf_counter() - started,
    }


def main():
    out = ROOT / "diagnostics/ring_program_audit/result.json"
    if out.exists():
        raise ValueError(f"preserve previous audit: {out}")
    paths = [
        "tools/ring_program_audit.py",
        "docs/RING_PROGRAM_INTEGRATION.md",
        "src/compose_v4/control/ring_program.py",
        "src/compose_v4/control/option_continuation.py",
        "src/compose_v4/control/option_selector.py",
        "src/compose_v4/control/macro_engine.py",
        "src/compose_v4/control/region_rewrite.py",
        "src/compose_v4/experiments/ring_construction_probe.py",
        "src/compose_v4/experiments/continuation_profile.py",
    ]
    paths += subprocess.check_output(
        [
            "git",
            "ls-files",
            "src/compose_v4/chem",
            "src/compose_v4/rewrite",
            "src/compose_v4/gates",
            "src/compose_v4/data/charge_policy.py",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    report = {
        "schema_version": "ring_program_capability_audit_v1",
        "configuration": CONFIG,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": {p: sha256_file(ROOT / p) for p in sorted(set(paths))},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {"machine": platform.machine(), "device": "CPU"},
        "requests": [run_request(option) for option in OPTIONS],
    }
    report["summary"] = {
        "requests": len(OPTIONS),
        "completed": sum(r["trajectory"]["status"] == "complete" for r in report["requests"]),
        "executor_calls": sum(r["executor_calls"] for r in report["requests"]),
        "seconds": sum(r["seconds"] for r in report["requests"]),
        "oracle_calls": 0,
    }
    publish_json(out, report)
    print(json.dumps(report["summary"], indent=2))
    for row in report["requests"]:
        print(
            row["option"],
            row["trajectory"]["status"],
            row["executor_calls"],
            round(row["seconds"], 3),
        )


if __name__ == "__main__":
    main()
