"""Verify the exact official evaluator before resuming checkpointed generation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from fetch_official_fragment_evaluator import verify_only
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prompt_path = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    inputs = {str(p.resolve()): physical_sha256(p) for p in (prompt_path, Path(__file__))}
    rows = []
    for prompt in load_genmol_prompts(prompt_path):
        if prompt.task not in (FragmentTask.MOTIF_EXTENSION, FragmentTask.SCAFFOLD_DECORATION):
            continue
        path = (
            args.baseline_dir / "shards" / f"{prompt.task.value}__{prompt.drug_name}__baseline.json"
        )
        inputs[str(path.resolve())] = physical_sha256(path)
        old = json.loads(path.read_text())
        samples = [r["committed_smiles"] or "" for r in old["attempt_records"]]
        current = official_prompt_metrics(samples, expected_samples=20)
        differences = {k: abs(current[k] - old["metrics"][k]) for k in current}
        rows.append(
            {
                "task": prompt.task.value,
                "drug": prompt.drug_name,
                "recorded": old["metrics"],
                "recomputed": current,
                "absolute_differences": differences,
            }
        )
    passed = len(rows) == 20 and all(
        all(v < 1e-10 for v in row["absolute_differences"].values()) for row in rows
    )
    result = {
        "schema": "fragment_evaluator_environment_repair_v1",
        "reason": "workers stopped after first prompt: missing pandas import; generated traces preserved",
        "inputs": inputs,
        "official_blobs": verify_only(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "pandas",
                "numpy",
                "rdkit",
                "torch",
                "python-dateutil",
                "pytz",
                "tzdata",
                "six",
                "tqdm",
            )
        },
        "scope": "post-generation metric import; no sampler, checkpoint or official evaluator change",
        "baseline_prompts": len(rows),
        "baseline_attempts": 20 * len(rows),
        "maximum_absolute_difference": max(
            v for r in rows for v in r["absolute_differences"].values()
        ),
        "passed": passed,
        "rows": rows,
    }
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {k: result[k] for k in ("baseline_prompts", "maximum_absolute_difference", "passed")}
        )
    )
    if not passed:
        raise RuntimeError("official evaluator differs from frozen baseline; do not resume")


if __name__ == "__main__":
    main()
