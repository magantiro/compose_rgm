"""Bounded exact-output benchmark against the recorded pre-repair graph kernel."""

import argparse
import ast
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import graph_kernel, molecular_features
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = "src/compose_v4/control/docking_value.py"
    revision = subprocess.check_output(["git", "rev-parse", "cc15546"], cwd=root, text=True).strip()
    old_source = subprocess.check_output(["git", "show", f"{revision}:{source}"], cwd=root)
    parsed = ast.parse(old_source)
    definition = next(
        n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "graph_kernel"
    )
    namespace = {"np": np}
    exec(  # noqa: S102 - only the recorded repository function, not external code
        compile(ast.Module(body=[definition], type_ignores=[]), "recorded_kernel", "exec"),
        namespace,
    )
    rows = unseal(a.input)["rows"][:256]
    features = [molecular_features(r["smiles"])[1] for r in rows]
    before = perf_counter()
    expected = namespace["graph_kernel"](features, features)
    old_seconds = perf_counter() - before
    before = perf_counter()
    actual = graph_kernel(features, features)
    new_seconds = perf_counter() - before
    np.testing.assert_array_equal(actual, expected)
    result = {
        "schema_version": "molecular_kernel_speed_v1",
        "input": {"path": str(a.input), "sha256": sha256_file(a.input)},
        "selection": "first 256 recorded scored neighbor rows, unchanged ordering",
        "old_commit": revision,
        "old_implementation_sha256": hashlib.sha256(old_source).hexdigest(),
        "implementation_sha256": {
            s: sha256_file(root / s) for s in (source, "tools/pmo_kernel_profile.py")
        },
        "rows": len(rows),
        "shape": list(actual.shape),
        "matrix_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "max_absolute_error": float(np.abs(actual - expected).max()),
        "old_seconds": old_seconds,
        "new_seconds": new_seconds,
        "speed_ratio": old_seconds / new_seconds,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "hardware": {
            "machine": platform.machine(),
            "device": "cpu",
            "numeric_threads": 1,
            "precision": "float64",
        },
        "new_oracle_calls": 0,
        "new_neural_calls": 0,
    }
    publish_json(a.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
