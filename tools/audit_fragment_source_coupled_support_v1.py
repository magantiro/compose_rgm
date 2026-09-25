"""Measure training-source coupling support without scoring benchmark molecules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler
from compose_v4.benchmark.source_coupled_pendant_policy import SourceCoupledPendantSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def _seed(drug: str) -> int:
    digest = hashlib.sha256(f"source-coupled-support-v1:{drug}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _input_attempts(path: Path) -> tuple[list[dict], dict[str, str]]:
    if not path.is_dir():
        raise ValueError(f"decoration attempt directory is absent: {path}")
    files = sorted(path.glob("*.json"))
    if not files:
        raise ValueError(f"decoration attempt directory is empty: {path}")
    rows = []
    digests = {}
    for file in files:
        rows.append(json.loads(file.read_text()))
        digests[file.name] = physical_sha256(file)
    return rows, digests


def audit(catalog_path: Path, mass_path: Path, attempts_path: Path, draws: int) -> dict:
    if draws < 1:
        raise ValueError("draws per prompt must be positive")
    catalog = json.loads(catalog_path.read_text())
    mass = json.loads(mass_path.read_text())
    baseline = JointMassPendantSampler(catalog, mass)
    coupled = SourceCoupledPendantSampler(catalog, mass)
    attempts, attempt_hashes = _input_attempts(attempts_path)
    by_drug = {}
    for attempt in attempts:
        drug = attempt["drug"]
        by_drug.setdefault(drug, []).append(attempt)
    reports = []
    for drug, rows in sorted(by_drug.items()):
        supported = 0
        shared = 0
        initial = None
        for row in rows:
            for offer in row["panel"]["offered"]:
                if offer.get("status") != "model_supported":
                    continue
                plan = offer["provenance"]["pendant_plan"]
                keys = tuple(
                    (draw["context"], draw["heavy_atoms"], draw["rings"]) for draw in plan["draws"]
                )
                if initial is None:
                    initial = (
                        tuple(draw["context"] for draw in plan["draws"]),
                        40 - plan["core_heavy_atoms"],
                    )
                source_rows = set(coupled.group_rows[keys[0]])
                for key in keys[1:]:
                    source_rows.intersection_update(coupled.group_rows[key])
                supported += 1
                shared += bool(source_rows)
        if initial is None:
            raise ValueError(f"no model-supported development offer for {drug}")
        contexts, capacity = initial
        baseline_rng = np.random.default_rng(_seed(drug))
        coupled_rng = np.random.default_rng(_seed(drug))
        baseline_plans = set()
        coupled_plans = set()
        used = Counter()
        for _ in range(draws):
            prior_entries, _ = baseline.sample(contexts, capacity, baseline_rng)
            new_entries, receipt = coupled.sample(contexts, capacity, coupled_rng)
            baseline_plans.add(tuple(entry["rooted_smiles"] for entry in prior_entries))
            coupled_plans.add(tuple(entry["rooted_smiles"] for entry in new_entries))
            used["coupled" if receipt["source_coupling_used"] else "independent"] += 1
            used["no_shared_row"] += receipt["shared_training_rows"] == 0
        reports.append(
            {
                "drug": drug,
                "development_attempts": len(rows),
                "model_supported_offers": supported,
                "offers_with_shared_source_row": shared,
                "interfaces": len(contexts),
                "plan_draws_per_arm": draws,
                "distinct_baseline_content_plans": len(baseline_plans),
                "distinct_coupled_content_plans": len(coupled_plans),
                "coupled_draws": used["coupled"],
                "independent_draws": used["independent"],
                "no_shared_source_draws": used["no_shared_row"],
            }
        )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "schema": "fragment_source_coupled_support_v1",
        "role": "zero_quality_support_only",
        "code_revision": revision,
        "input_paths": {
            "catalog": str(catalog_path.resolve()),
            "mass_prior": str(mass_path.resolve()),
            "development_attempts": str(attempts_path.resolve()),
        },
        "input_sha256": {
            "catalog": physical_sha256(catalog_path),
            "mass_prior": physical_sha256(mass_path),
            "attempt_files": attempt_hashes,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "seed_derivation": "first 64 bits of SHA-256(source-coupled-support-v1:drug)",
        "precision": "CPU float64",
        "coupling_probability": 0.5,
        "quality_labels_used": False,
        "benchmark_prompts_used_to_fit": False,
        "observed_decoration_bundle_claim": False,
        "prompts": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--mass-prior", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws-per-prompt", type=int, default=200)
    args = parser.parse_args()
    result = audit(args.catalog, args.mass_prior, args.attempts, args.draws_per_prompt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite support result: {args.output}")
    temporary = args.output.with_name(f".{args.output.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"unfinished support result requires review: {temporary}")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "prompts": len(result["prompts"])}))


if __name__ == "__main__":
    main()
