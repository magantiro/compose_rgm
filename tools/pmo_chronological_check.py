"""Run the fixed offline PMO prediction check, with no oracle/network imports."""

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_chronological import (
    RECIPE,
    evaluate_stream,
    prepare_stream,
    state_features,
)
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("chronological check requires recorded RDKit 2024.03.5")
    start = perf_counter()
    hashes, arms, feature_hashes = {}, [], {}
    for arm in ("balanced", "adaptive"):
        path = ROOT / f"diagnostics/pmo_archive_pilot_100/{arm}.json"
        result = json.loads(path.read_text())
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
        roots = []
        for i in range(4):
            path = ROOT / f"diagnostics/pmo_chronological/roots/{arm}/{i:04}.json"
            digest = sha256_file(path)
            if digest != result["oracle_ledger"][i]:
                raise ValueError(
                    f"root receipt differs from original charged oracle ledger: {path}"
                )
            hashes[str(path.relative_to(ROOT))] = digest
            roots.append(unseal(path))
        stream = prepare_stream(result, roots)
        kernel, counts = state_features(stream["smiles"])
        feature_hashes[arm] = identity(
            {"smiles": stream["smiles"], "kernel": kernel.tolist(), "counts": counts.tolist()}
        )
        assessed = evaluate_stream(stream, kernel, counts)
        assessed["charged_query_count"] = len(stream["scores"])
        assessed["root_receipts_match_ledger"] = True
        assessed["stream_and_locked_curve_match"] = True
        arms.append(assessed)
        print(arm, json.dumps(assessed["metrics"]), flush=True)
    candidates = []
    for model in ("context_delta", "endpoint_score", "edit_delta"):
        passes = True
        for arm in arms:
            m = arm["metrics"]
            passes &= m[model]["mae"] < min(m[b]["mae"] for b in ("parent_copy", "option_delta"))
            passes &= m[model]["concordance"] > max(
                0.5, *(m[b]["concordance"] for b in ("parent_copy", "option_delta"))
            )
        if passes:
            candidates.append(model)
    dependencies = [
        Path(__file__),
        ROOT / "docs/PMO_CHRONOLOGICAL_CHECK.md",
        ROOT / "src/compose_v4/experiments/pmo_chronological.py",
        ROOT / "src/compose_v4/control/docking_value.py",
        ROOT / "src/compose_v4/control/molecular_search_codec.py",
        ROOT / "src/compose_v4/chem/molecular_graph.py",
        ROOT / "src/compose_v4/rewrite/trace_shard.py",
    ]
    report = {
        "schema_version": "pmo_chronological_check_v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "recipe": RECIPE,
        "recipe_sha256": identity(RECIPE),
        "input_hashes": hashes,
        "feature_hashes": feature_hashes,
        "implementation_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in dependencies},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "torch": importlib.metadata.version("torch"),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu",
            "precision": "float64",
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "thread_environment": {
                k: os.getenv(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "random_sampling": False,
        "split": "chronological_within_each_inspected_development_arm",
        "seconds": perf_counter() - start,
        "new_oracle_calls": 0,
        "arms": arms,
        "offline_candidates": candidates,
        "decision": "nominate_for_prospective_comparison_only"
        if candidates
        else "no_predictor_qualified",
        "interpretation": "Retrospective fixed-recipe prediction on logged trajectories; no off-policy return estimate, no calibrated uncertainty, no new queries or deployed guidance. Candidate-structure models require completed proposals.",
    }
    publish_json(args.output, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("decision", "offline_candidates", "seconds", "new_oracle_calls")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
