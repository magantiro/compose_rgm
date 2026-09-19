"""Reproducible zero-oracle engineering comparison, not a discovery benchmark."""

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

from compose_v4.control.continuation import (
    FiniteHorizonContinuation,
    ReferenceRow,
    continuation_decision,
)
from compose_v4.control.option_continuation import OptionState
from compose_v4.control.sampled_continuation import (
    SampledContinuation,
    sampled_continuation_decision,
)
from compose_v4.experiments.continuation_gate import engineering_fixture, ring_terminal_weight
from compose_v4.experiments.continuation_profile import publish_json, sha256_file

CONFIG = {
    "samples_per_successor": 32,
    "alpha": 0.05,
    "seed": 17,
    "max_expansions": 4096,
    "max_terminal_evaluations": 4096,
    "max_rollouts": 4096,
    "kappa": 1.0,
    "exploration": 0.1,
    "tree_branching": 4,
    "tree_remaining": 6,
    "binding_resource": "none: finite engineering problem, not a matched-compute discovery comparison",
    "device": "cpu",
    "workers": 1,
    "precision": "float64 probabilities",
    "split": "synthetic engineering only; no train, calibration or final-test source",
    "seed_derivation": "literal fixed seed independently for each engineering case",
    "external_oracle_calls": 0,
    "winner_inputs": [],
    "trained_models": [],
    "exclusions": [],
}


def compare(reference, terminal, key, root, remaining, snapshot_id):
    results = {}
    for arm in ("exact", "sampled"):
        common = {
            "snapshot_id": snapshot_id,
            "max_expansions": CONFIG["max_expansions"],
            "max_terminal_evaluations": CONFIG["max_terminal_evaluations"],
        }
        started = perf_counter()
        if arm == "exact":
            estimator = FiniteHorizonContinuation(reference, terminal, key, **common)
            decision = continuation_decision(
                root, remaining, estimator, fallback_values=[1] * len(root.successors)
            )
            estimate = None
        else:
            estimator = SampledContinuation(
                reference,
                terminal,
                key,
                **common,
                samples_per_successor=CONFIG["samples_per_successor"],
                alpha=CONFIG["alpha"],
                seed=CONFIG["seed"],
                max_rollouts=CONFIG["max_rollouts"],
            )
            sampled = sampled_continuation_decision(
                root, remaining, estimator, fallback_values=[1] * len(root.successors)
            )
            decision, estimate = sampled.decision, sampled.estimate
        results[arm] = {
            "decision": asdict(decision),
            "estimate": asdict(estimate) if estimate else None,
            "work": asdict(estimator.work),
            "seconds": perf_counter() - started,
        }
    return results


def evaluate():
    branching = CONFIG["tree_branching"]

    # Unique tuple prefixes: this fixture deliberately has no transpositions.
    def reference(prefix):
        return ReferenceRow(
            tuple(prefix + (i,) for i in range(branching)), (1 / branching,) * branching
        )

    tree = compare(
        reference,
        lambda prefix: float(prefix[0]),
        lambda prefix: prefix,
        ReferenceRow(((0,), (1,)), (0.5, 0.5)),
        CONFIG["tree_remaining"],
        "hand-declared-prefix-tree-v1",
    )
    chemistry = {}
    # Separate kernels: neither arm inherits the other's molecular work cache.
    for arm in ("exact", "sampled"):
        source, kernel = engineering_fixture()
        root = kernel.row(source)
        common = {
            "snapshot_id": "hand-declared-ring-plateau-v1",
            "max_expansions": 64,
            "max_terminal_evaluations": 64,
        }
        if arm == "exact":
            estimator = FiniteHorizonContinuation(
                kernel.row, ring_terminal_weight, OptionState.key, **common
            )
            decision = continuation_decision(root, 2, estimator, fallback_values=(1, 1))
        else:
            estimator = SampledContinuation(
                kernel.row,
                ring_terminal_weight,
                OptionState.key,
                **common,
                samples_per_successor=CONFIG["samples_per_successor"],
                alpha=CONFIG["alpha"],
                seed=CONFIG["seed"],
                max_rollouts=64,
            )
            decision = sampled_continuation_decision(
                root, 2, estimator, fallback_values=(1, 1)
            ).decision
        chemistry[arm] = {
            "decision": asdict(decision),
            "work": asdict(estimator.work),
            "kernel_work": asdict(kernel.work),
        }
    sampled, exact = tree["sampled"], tree["exact"]
    checks = {
        "analytic_tree_means": sampled["estimate"]["means"] == (0.0, 1.0),
        "same_tree_decision": np.allclose(
            sampled["decision"]["probabilities"], exact["decision"]["probabilities"], atol=1e-10
        ),
        "bounded_row_visits": sampled["work"]["row_visits"]
        <= 2 * CONFIG["samples_per_successor"] * (CONFIG["tree_remaining"] - 1),
        "fewer_tree_expansions": sampled["work"]["expansions"] < exact["work"]["expansions"],
        "chemistry_same_decision": np.allclose(
            chemistry["sampled"]["decision"]["probabilities"],
            chemistry["exact"]["decision"]["probabilities"],
            atol=1e-10,
        ),
        "same_chemistry_executor_work": chemistry["sampled"]["kernel_work"]["executor_applications"]
        == chemistry["exact"]["kernel_work"]["executor_applications"]
        == 4,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "tree": tree,
        "chemistry": chemistry,
    }


def main():
    root = Path(__file__).resolve().parents[3]
    # Hash the full tracked scientific source closure conservatively for this
    # tiny local receipt. This is provenance, NOT a deterministic cache key.
    tracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src/compose_v4",
            "configs",
            "tests/test*continuation*",
            "docs/CONTINUATION_CONTROLLER_IMPLEMENTATION.md",
            "tools/sampled_continuation_gate.py",
        ],
        cwd=root,
        text=True,
    ).splitlines()
    new = [
        "src/compose_v4/control/sampled_continuation.py",
        "src/compose_v4/experiments/sampled_continuation_gate.py",
        "tests/test_sampled_continuation.py",
        "tools/sampled_continuation_gate.py",
    ]
    inputs = {
        name: sha256_file(root / name)
        for name in sorted(set(tracked + new))
        if (root / name).is_file()
    }
    manifest_path = root / "diagnostics/sampled_continuation/inputs.json"
    manifest_hash = publish_json(manifest_path, inputs)
    result = evaluate()
    report = {
        "schema_version": "sampled_continuation_engineering_v1",
        "evidence_role": "computed synthetic engineering, not learned-model performance",
        "configuration": CONFIG,
        "configuration_sha256": hashlib.sha256(
            json.dumps(CONFIG, sort_keys=True).encode()
        ).hexdigest(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "input_manifest": str(manifest_path.relative_to(root)),
        "input_manifest_sha256": manifest_hash,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": platform.machine(),
        "source_count": 2,
        "result": result,
        "limitations": [
            "The algebraic tree has deterministic terminal classes and is not a representative molecular branching benchmark.",
            "The molecular reference is hand-declared; R_theta was not loaded.",
            "The tiny chemistry fixture shows no executor-call saving; sampling adds statistical overhead there.",
            "Monte Carlo confidence is not calibration of docking predictions, nor proof of better T4 search.",
            "No live T4 controller was replaced and no new oracle calls were spent.",
        ],
    }
    path = root / "diagnostics/sampled_continuation/engineering_gate.json"
    digest = publish_json(path, report)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output": str(path),
                "sha256": digest,
                "checks": result["checks"],
            },
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
