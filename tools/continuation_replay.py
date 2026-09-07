"""Run the frozen zero-compute recorded-process continuation diagnostic."""

import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from compose_v4.experiments.continuation_profile import canonical_bytes, publish_json, sha256_file
from compose_v4.experiments.continuation_replay import RecordedProfile, replay

CONFIG = {
    "seed": 1000,
    "samples_per_successor": 32,
    "alpha": 0.05,
    "kappa": 1.0,
    "exploration": 0.1,
    "max_expansions": 10000,
    "max_terminal_evaluations": 10000,
    "max_rollouts": 10000,
    "seed_derivation": "literal fixed seed from the recorded production profile",
    "terminal": "completed program with increased cycle rank and ring-system count",
    "role": "incomplete recorded development graph, not a discovery comparison",
    "workers": 1,
    "precision": "float64 probability arrays",
    "exclusions": [],
}


def main():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    inventory_path = root / "diagnostics/sampled_continuation/prior_profile_audit.json"
    inventory = json.loads(inventory_path.read_text())
    # The inventory is a frozen recorded dataset, not a live remote lookup.
    expected = "32c58e7cd79cda79b08bc743d297ee4793a2e01de9ee76990d1723751ec6e20c"
    if sha256_file(inventory_path) != expected:
        raise ValueError("recorded-profile inventory differs from the frozen diagnostic input")
    run_id = inventory["metadata"]["launch"]["run_id"]
    directory = root / "diagnostics/continuation_profile/attempt_1" / run_id
    profile = RecordedProfile(directory, inventory)
    result = replay(profile, CONFIG)
    baseline_retained = np.allclose(
        result["decision"]["decision"]["probabilities"],
        result["reference_root"],
        atol=1e-12,
        rtol=1e-12,
    )
    if result["missing_rows"] and not baseline_retained:
        raise RuntimeError("incomplete replay changed the declared baseline")
    source_files = (
        "docs/CONTINUATION_REPLAY_GATE.md",
        "src/compose_v4/experiments/continuation_replay.py",
        "src/compose_v4/experiments/continuation_profile.py",
        "src/compose_v4/control/sampled_continuation.py",
        "src/compose_v4/control/continuation.py",
        "src/compose_v4/control/region_rewrite.py",
        "src/compose_v4/control/graph_geometry.py",
        "src/compose_v4/rewrite/trace_shard.py",
        "src/compose_v4/chem/molecular_graph.py",
        "tests/test_continuation_replay.py",
        "tools/continuation_replay.py",
    )
    report = {
        "schema_version": "recorded_continuation_replay_v1",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "configuration": CONFIG,
        "configuration_sha256": hashlib.sha256(canonical_bytes(CONFIG)).hexdigest(),
        "input_inventory": str(inventory_path.relative_to(root)),
        "input_inventory_sha256": expected,
        "implementation_sha256": {name: sha256_file(root / name) for name in sorted(source_files)},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "hardware": platform.machine(),
        "source_count": 1,
        "complete_recorded_rows": len(profile.rows),
        "evidence_role": "computed replay of a partial recorded production process",
        "result": result,
        "baseline_retained": bool(baseline_retained),
        "limitations": [
            "No new molecule or docking score was generated.",
            "Missing data is not chemical unreachability.",
            "This is not a live evaluator-equivalence certificate or a speedup estimate.",
            "The inspected PARP1 bundle is development data, not an independent test panel.",
        ],
    }
    output = root / "diagnostics/sampled_continuation/recorded_replay.json"
    digest = publish_json(output, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "baseline_retained": bool(baseline_retained),
                "work": result["work"],
                "missing_rows": result["missing_rows"],
                "output": str(output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
