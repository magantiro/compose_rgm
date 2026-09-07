"""One bounded, model-free integration audit of the stateful fused reference.

This is NOT the frozen R_theta production audit or a discovery benchmark.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.fused_option import BUILD_FUSED_RING_OPTION, FusedProgress
from compose_v4.control.option_continuation import (
    OptionContinuationKernel,
    OptionState,
    sample_option_trajectory,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.experiments.continuation_profile import (
    canonical_bytes,
    publish_json,
    sha256_file,
    state_payload,
)
from compose_v4.experiments.ring_construction_probe import ProbeRuntime, topology_witness

CONFIG = {
    "source_smiles": "c1ccccc1",
    "persistent_slots": 48,
    "max_active_atoms": 40,
    "option": BUILD_FUSED_RING_OPTION,
    "horizon": 5,
    "seed": 0,
    "seed_derivation": "default_rng(seed), no planner",
    "estimator": "reference",
    "max_executor_applications": 256,
    "safety_seconds": 30.0,
    "macro_temperature": 2.0,
    "macro_exploration": 0.15,
    "inherited_support": {"global_cap": 300, "family_floor": 20},
    "kappa": 1.0,
    "lookahead_expansions": 0,
    "terminal_evaluations": 0,
    "reference": "uniform neutral primitive candidates, ORGANIC_VOCABULARY; no RingCore tracelets",
    "outer_draws": "one fixed synthetic whole-region fused bundle; no Q(M) or Q(o) comparison",
    "workers": 1,
    "precision": "float64 probabilities; exact integer state coordinates",
    "split": "synthetic engineering fixture only",
    "attempts": 1,
    "exclusions": [],
    "winner_inputs": [],
    "oracle_calls": 0,
}


class RecordedKernel(OptionContinuationKernel):
    def __init__(self, runtime: ProbeRuntime):
        super().__init__(
            runtime.enumerate, runtime.system, max_executor_applications=runtime.meter.limit
        )
        self.runtime = runtime
        self.states: dict[str, dict] = {}
        self.rows: dict[str, dict] = {}
        self.nodes: dict[str, OptionState] = {}

    def record(self, node):
        payload = state_payload(node)
        identity = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        self.states[identity] = payload
        self.nodes[identity] = node
        return identity

    def row(self, node):
        identity = self.record(node)
        with self.runtime.meter.boundary():
            row = super().row(node)
        self.rows[identity] = {
            "successors": [self.record(s) for s in row.successors],
            "probabilities": list(row.probabilities),
            "support_funnel": self.fused_support(node),
        }
        return row


def run_audit() -> dict:
    graph = pad_molecular_graph(
        smiles_to_molecular_graph(CONFIG["source_smiles"]), CONFIG["persistent_slots"]
    )
    real = frozenset(int(i) for i in np.flatnonzero(is_element(graph.atom_types)))
    initial = OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), real, (), "multi", 0),
        Lineage.initial(real),
        BUILD_FUSED_RING_OPTION,
        0,
        5,
        "synthetic-fused-reference-seed0",
        FusedProgress(),
    )
    runtime = ProbeRuntime(graph, CONFIG["max_executor_applications"], CONFIG["safety_seconds"])
    kernel = RecordedKernel(runtime)

    def forbidden(_):
        raise RuntimeError("reference-only audit must not call objective or lookahead scoring")

    started = perf_counter()
    try:
        with runtime.meter.instrument():
            trajectory = sample_option_trajectory(
                initial,
                kernel,
                forbidden,
                forbidden,
                snapshot_id="synthetic-fused-reference-v1",
                seed=CONFIG["seed"],
                max_expansions=0,
                max_terminal_evaluations=0,
                estimator="reference",
            )
    except TimeoutError as error:
        trajectory = {"status": "timeout_incomplete", "reason": str(error)}
    witness = None
    if trajectory["status"] == "complete":
        progress = FusedProgress.from_payload(trajectory["final_fused_progress"])
        endpoints = [
            n
            for n in kernel.nodes.values()
            if n.remaining == 0
            and n.fused_progress == progress
            and state_payload(n)["graph"] == trajectory["trace"][-1]["exact_state"]
        ]
        if len(endpoints) != 1:
            raise RuntimeError(
                "sampled endpoint does not identify exactly one recorded exact state"
            )
        witness = topology_witness(graph, endpoints[0].graph, "fused")
    if sorted(r["call_index"] for r in runtime.meter.attempts) != list(range(runtime.meter.calls)):
        raise RuntimeError("executor receipts do not cover every entered application")
    return {
        "trajectory": trajectory,
        "independent_topology_witness": witness,
        "initial_state": state_payload(initial),
        "states": dict(sorted(kernel.states.items())),
        "rows": dict(sorted(kernel.rows.items())),
        "executor_receipts": runtime.meter.attempts,
        "work": {
            "public_executor_applications": runtime.meter.calls,
            "enumerated_candidates_per_state": runtime.enumerated_counts,
            "wall_seconds": perf_counter() - started,
            "raw_law_cache_hits": runtime.cache_hits,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output = root / "diagnostics/fused_option_audit/result.json"
    if output.exists():
        raise SystemExit(f"preserve existing audit; refusing to overwrite {output}")
    drift = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "configs"], cwd=root, text=True
    )
    if drift.strip():
        raise SystemExit("commit source/configuration before producing an authoritative audit")
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "configs"], cwd=root, text=True
    ).splitlines()
    inputs = sorted(
        set(tracked)
        | {
            "docs/FUSED_OPTION_INTEGRATION.md",
            "tests/test_fused_option.py",
            "tests/test_fused_option_audit.py",
            "tools/fused_option_audit.py",
        }
    )
    report = {
        "schema_version": "fused_option_audit_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "one model-free stateful option reference trajectory; no discovery claim",
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
        "hardware": {"machine": platform.machine(), "processor": platform.processor(), "gpu": None},
        "result": run_audit(),
        "population_coverage": None,
        "new_docking_calls": 0,
    }
    digest = publish_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "status": report["result"]["trajectory"]["status"],
                "work": report["result"]["work"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
