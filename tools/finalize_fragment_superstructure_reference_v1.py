"""Finalize the completed superstructure reference comparison by prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

LEARNED_SHA256 = "8406f57dc5ff6736110593d309f207d7a5d0e878eeb1b7a2f2c423fe2d9b3652"
UNIFORM_SHA256 = "007f59645ffda58f1722eb846444d4f604307468016d4f813740b3ab9c9b0b5d"
METRICS = ("quality", "uniqueness", "diversity", "validity")
SEED = 20260925
DRAWS = 20000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, expected_hash: str) -> dict:
    if sha256(path) != expected_hash:
        raise ValueError(f"superstructure result hash mismatch: {path}")
    result = json.loads(path.read_text())
    rows = result["rows"]
    if (
        len(rows) != 30
        or result["attempts"] != 3000
        or result["committed"] != 3000
        or result["prompt_compliant"] != 3000
        or len({row["drug"] for row in rows}) != 10
        or {row["seed"] for row in rows} != {0, 1, 2}
        or any(
            row["attempts"] != 100 or row["committed"] != 100 or row["prompt_compliant"] != 100
            for row in rows
        )
    ):
        raise ValueError(f"incomplete superstructure result: {path}")
    if len({(row["drug"], row["seed"]) for row in rows}) != 30:
        raise ValueError(f"duplicate prompt-seed row: {path}")
    for metric in METRICS:
        measured = float(np.mean([row["official"][metric] for row in rows]))
        if not np.isclose(measured, result["official_mean"][metric], atol=1e-10):
            raise ValueError(f"official mean differs from rows for {metric}: {path}")
    return result


def analyze(learned_path: Path, uniform_path: Path) -> dict:
    learned = load(learned_path, LEARNED_SHA256)
    uniform = load(uniform_path, UNIFORM_SHA256)
    by_arm = {
        arm: {(row["drug"], row["seed"]): row for row in result["rows"]}
        for arm, result in (("learned", learned), ("uniform", uniform))
    }
    if set(by_arm["learned"]) != set(by_arm["uniform"]):
        raise ValueError("the completed arms have different prompt-seed panels")
    per_seed = []
    for drug, seed in sorted(by_arm["learned"]):
        rows = {arm: by_arm[arm][(drug, seed)] for arm in by_arm}
        per_seed.append(
            {
                "drug": drug,
                "seed": seed,
                "arms": {arm: row["official"] for arm, row in rows.items()},
                "attempts_per_arm": 100,
                "prompt_compliant_per_arm": {
                    arm: row["prompt_compliant"] for arm, row in rows.items()
                },
            }
        )
    per_prompt = []
    for drug in sorted({drug for drug, _ in by_arm["learned"]}):
        cohort = [row for row in per_seed if row["drug"] == drug]
        arms = {
            arm: {
                metric: float(np.mean([row["arms"][arm][metric] for row in cohort]))
                for metric in METRICS
            }
            for arm in by_arm
        }
        per_prompt.append(
            {
                "drug": drug,
                "arms": arms,
                "learned_minus_uniform": {
                    metric: arms["learned"][metric] - arms["uniform"][metric] for metric in METRICS
                },
            }
        )
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, 10, size=(DRAWS, 10))
    estimates = {}
    for metric in METRICS:
        differences = np.asarray(
            [row["learned_minus_uniform"][metric] for row in per_prompt],
            dtype=np.float64,
        )
        resampled = differences[samples].mean(axis=1)
        estimates[metric] = {
            "learned_mean": float(learned["official_mean"][metric]),
            "uniform_mean": float(uniform["official_mean"][metric]),
            "paired_prompt_mean_difference": float(differences.mean()),
            "prompt_bootstrap_percentile_95": [
                float(np.quantile(resampled, 0.025)),
                float(np.quantile(resampled, 0.975)),
            ],
            "prompts_learned_higher": int(np.count_nonzero(differences > 0)),
            "prompts_equal": int(np.count_nonzero(differences == 0)),
            "prompts_uniform_higher": int(np.count_nonzero(differences < 0)),
        }
    return {
        "schema": "fragment_superstructure_reference_final_v1",
        "interpretation": "learned versus uniform family and within-family native-mark probabilities; learned hazard retained in both arms",
        "randomization_note": "matching seed labels do not pair molecular trajectories because the arms consume random draws differently",
        "structural_unit": "ten prompts; three seeds averaged within each prompt",
        "bootstrap": {"draws": DRAWS, "seed": SEED, "interval": "percentile 95%"},
        "arm_inputs": {
            "learned": {"path": str(learned_path.resolve()), "sha256": LEARNED_SHA256},
            "uniform": {"path": str(uniform_path.resolve()), "sha256": UNIFORM_SHA256},
        },
        "input_contract_payload_sha256": {
            "learned": learned["contract_payload_sha256"],
            "uniform": uniform["contract_payload_sha256"],
        },
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "analysis_sha256": sha256(Path(__file__)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "attempts_per_arm": 3000,
        "prompt_compliant_per_arm": {"learned": 3000, "uniform": 3000},
        "per_seed": per_seed,
        "per_prompt": per_prompt,
        "estimates": estimates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned", type=Path, required=True)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.learned, args.uniform)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
