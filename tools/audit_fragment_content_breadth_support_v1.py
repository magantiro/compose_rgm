"""Compare train-only pendant-content plan breadth without molecular scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _seed(drug: str) -> int:
    digest = hashlib.sha256(f"content-breadth-v1:{drug}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _effective_plans(counts: Counter[tuple[str, ...]]) -> float:
    total = sum(counts.values())
    return 1.0 / sum((count / total) ** 2 for count in counts.values())


def audit(catalog_path: Path, prior_path: Path, attempts_path: Path, draws: int) -> dict:
    if draws < 1:
        raise ValueError("draws per prompt must be positive")
    if not attempts_path.is_dir():
        raise ValueError(f"development attempts directory is absent: {attempts_path}")
    catalog = json.loads(catalog_path.read_text())
    prior = json.loads(prior_path.read_text())
    samplers = {
        "frozen": JointMassPendantSampler(catalog, prior),
        "uniform_within_cell": JointMassPendantSampler(
            catalog, prior, content_allocation="uniform_within_cell"
        ),
    }
    by_drug: dict[str, list[dict]] = defaultdict(list)
    attempt_hashes = {}
    for path in sorted(attempts_path.glob("*.json")):
        row = json.loads(path.read_text())
        by_drug[row["drug"]].append(row)
        attempt_hashes[path.name] = physical_sha256(path)
    if len(by_drug) != 10 or sum(map(len, by_drug.values())) != 200:
        raise ValueError("support audit requires exactly 200 attempts across ten prompts")
    reports = []
    for drug, attempts in sorted(by_drug.items()):
        if len(attempts) != 20:
            raise ValueError(f"support audit requires 20 locked attempts for {drug}")
        plans = [
            offer["provenance"]["pendant_plan"]
            for attempt in attempts
            for offer in attempt["panel"]["offered"]
            if offer.get("status") == "model_supported"
        ]
        if not plans:
            raise ValueError(f"no model-supported offer for {drug}")
        contexts = tuple(draw["context"] for draw in plans[0]["draws"])
        capacity = 40 - plans[0]["core_heavy_atoms"]
        if any(
            tuple(draw["context"] for draw in plan["draws"]) != contexts
            or 40 - plan["core_heavy_atoms"] != capacity
            for plan in plans
        ):
            raise ValueError(f"development prompt contexts changed within {drug}")
        arms = {}
        for name, sampler in samplers.items():
            rng = np.random.default_rng(_seed(drug))
            counts: Counter[tuple[str, ...]] = Counter()
            content = set()
            masses = Counter()
            for _ in range(draws):
                entries, receipt = sampler.sample(contexts, capacity, rng)
                key = tuple(entry["rooted_smiles"] for entry in entries)
                counts[key] += 1
                content.update(key)
                masses[receipt["planned_decoration_mass"]] += 1
            arms[name] = {
                "distinct_content_plans": len(counts),
                "effective_content_plans": _effective_plans(counts),
                "distinct_pendant_contents": len(content),
                "decoration_mass_draws": dict(sorted(masses.items())),
            }
        reports.append(
            {
                "drug": drug,
                "interfaces": len(contexts),
                "model_supported_development_offers": len(plans),
                "draws_per_arm": draws,
                "arms": arms,
            }
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "schema": "fragment_content_breadth_support_v1",
        "role": "zero_quality_training_content_plan_support",
        "code_revision": revision,
        "input_paths": {
            "catalog": str(catalog_path.resolve()),
            "mass_prior": str(prior_path.resolve()),
            "development_attempts": str(attempts_path.resolve()),
        },
        "input_sha256": {
            "catalog": physical_sha256(catalog_path),
            "mass_prior": physical_sha256(prior_path),
            "attempt_files": attempt_hashes,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "training_split": catalog["split"],
        "device": "cpu",
        "precision": "float64",
        "workers": 1,
        "seed_derivation": "first 64 bits of SHA-256(content-breadth-v1:drug)",
        "draws_per_prompt_arm": draws,
        "quality_labels_used": False,
        "benchmark_prompts_used_to_fit": False,
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
    if any(
        subprocess.run(command, cwd=ROOT, check=False).returncode
        for command in (("git", "diff", "--quiet"), ("git", "diff", "--cached", "--quiet"))
    ):
        raise ValueError("support audit requires clean tracked source")
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
