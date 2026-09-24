"""Rescore frozen fragment shards on the comparator's chemical-output population.

Historical v1 shards persisted all chemically committed endpoint SMILES but
passed only prompt-compliant emissions to the official evaluator. This audit
reconstructs the 100-attempt chemical population from those saved endpoints
and no-output placeholders, leaving every original shard immutable. It first
reproduces each saved task-filtered official row before publishing a corrected
headline row and a separate prompt-fidelity diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import rdkit
from fetch_official_fragment_evaluator import OFFICIAL_BLOBS, verify_only

from compose_v4.benchmark.fragment_official_metrics import (
    official_distance,
    official_prompt_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/fragment_attachment_control_v1"
OUTPUT_DIR = ROOT / "diagnostics/fragment_attachment_headline_rescore_v1"
OUTPUT = OUTPUT_DIR / "result.json"
ARMS = ("baseline", "attachment")
TASKS = ("motif_extension", "scaffold_decoration", "superstructure_generation")
METRICS = ("validity", "uniqueness", "quality", "diversity", "distance")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chemical_attempt_samples(row: dict) -> list[str]:
    """Recover the official 100-attempt population without task censoring."""
    n = row["attempts"]
    committed = row["committed_endpoint_smiles"]
    if not isinstance(committed, list) or len(committed) != row["committed_endpoints"]:
        raise ValueError("committed molecule list/count mismatch")
    if n != 100 or len(committed) > n:
        raise ValueError(f"unexpected attempt denominator or commit count: {n}/{len(committed)}")
    if any(not smiles for smiles in committed):
        raise ValueError("committed endpoint is empty")
    # The old shards did not retain attempt-aligned committed SMILES, but the
    # official set-based metric is invariant to the placement of empty entries.
    return list(committed) + [""] * (n - len(committed))


def main() -> None:
    verified = verify_only()
    evaluator_hash = OFFICIAL_BLOBS["in_virtuo_gen/train_utils/metrics.py"][1]
    property_hash = OFFICIAL_BLOBS["in_virtuo_gen/utils/mol.py"][1]
    if evaluator_hash not in verified.values() or property_hash not in verified.values():
        raise RuntimeError("official evaluator/property hashes not verified")
    table = SOURCE / "before_after_table.json"
    identities = json.loads(table.read_text())
    if identities["partial"] is not False or identities["instances_compared"] != 60:
        raise RuntimeError("the historical matched attachment comparison is incomplete")
    input_hashes = {str(table.relative_to(ROOT)): sha256(table)}
    rows = []
    for arm in ARMS:
        for task in TASKS:
            for path in sorted((SOURCE / "shards" / arm).glob(f"{task}__*__seed*.json")):
                input_hashes[str(path.relative_to(ROOT))] = sha256(path)
                payload = json.loads(path.read_text())
                if payload["schema"] != "compose_fragment_official_suite_v1":
                    raise RuntimeError(f"unexpected source schema: {path}")
                if payload["protocol"]["samples_per_prompt"] != 100:
                    raise RuntimeError(f"nonofficial attempt count: {path}")
                expected_arm = (
                    "attachment_control" if arm == "attachment" else "frozen_sampler_baseline"
                )
                if payload["attachment_control"]["arm"] != expected_arm:
                    raise RuntimeError(f"attachment arm mismatch: {path}")
                per_drug = payload["results"][task]["per_drug"]
                if len(per_drug) != 1:
                    raise RuntimeError(f"shard is not one prompt: {path}")
                [(drug, [row])] = per_drug.items()
                seed = row["seed"]
                if seed not in (0, 1, 2):
                    raise RuntimeError(f"unexpected seed: {path}")
                strict_samples = row["emitted_samples"]
                if len(strict_samples) != 100:
                    raise RuntimeError(f"task-filtered sample census mismatch: {path}")
                strict = official_prompt_metrics(strict_samples, expected_samples=100)
                for metric in ("validity", "uniqueness", "quality", "diversity"):
                    if abs(strict[metric] - row["official"][metric]) > 1e-9:
                        raise RuntimeError(
                            f"frozen task-filtered metric did not replay: {path}/{metric}"
                        )
                chemical_samples = chemical_attempt_samples(row)
                chemical = official_prompt_metrics(chemical_samples, expected_samples=100)
                chemical["distance"] = official_distance(
                    chemical_samples, row["distance_reference_prompt"]
                )
                if chemical["validity"] != float(row["committed_chemically_valid"]):
                    raise RuntimeError(f"chemical validity/commit count mismatch: {path}")
                if sum(bool(smiles) for smiles in strict_samples) != row["emitted_nonempty"]:
                    raise RuntimeError(f"task-fidelity count mismatch: {path}")
                if row["emitted_nonempty"] > row["committed_endpoints"]:
                    raise RuntimeError(f"task success exceeds commits: {path}")
                rows.append(
                    {
                        "arm": arm,
                        "task": task,
                        "drug": drug,
                        "seed": seed,
                        "attempts": 100,
                        "committed": row["committed_endpoints"],
                        "chemically_valid_committed": row["committed_chemically_valid"],
                        "fragment_containing_committed": row["committed_fragment_preserving"],
                        "prompt_compliant": row["emitted_nonempty"],
                        "no_output": 100 - row["committed_endpoints"],
                        "official_chemical": chemical,
                        "task_filtered_diagnostic": strict,
                        "source": str(path.relative_to(ROOT)),
                    }
                )
    expected = {
        (arm, task, drug, seed)
        for arm in ARMS
        for task in TASKS
        for drug in {row["drug"] for row in rows if row["task"] == task}
        for seed in (0, 1, 2)
    }
    seen = {(row["arm"], row["task"], row["drug"], row["seed"]) for row in rows}
    if len(rows) != 180 or len(seen) != 180 or seen != expected:
        raise RuntimeError(
            f"shard census is not 2 arms x 3 tasks x 10 prompts x 3 seeds: {len(rows)}"
        )
    if any(row["committed"] != row["chemically_valid_committed"] for row in rows):
        raise RuntimeError("at least one committed endpoint is chemically invalid")

    summary = []
    for arm in ARMS:
        for task in TASKS:
            selected = [row for row in rows if row["arm"] == arm and row["task"] == task]
            by_seed = defaultdict(list)
            for row in selected:
                by_seed[row["seed"]].append(row)
            if set(by_seed) != {0, 1, 2} or any(len(cohort) != 10 for cohort in by_seed.values()):
                raise RuntimeError(f"incomplete seed/prompt grid: {arm}/{task}")
            seed_metrics = [
                {
                    metric: float(
                        np.mean([row["official_chemical"][metric] for row in by_seed[seed]])
                    )
                    for metric in METRICS
                }
                for seed in (0, 1, 2)
            ]
            summary.append(
                {
                    "arm": arm,
                    "task": task,
                    "attempts": 3000,
                    "committed": sum(row["committed"] for row in selected),
                    "prompt_compliant": sum(row["prompt_compliant"] for row in selected),
                    "fragment_containing_committed": sum(
                        row["fragment_containing_committed"] for row in selected
                    ),
                    "per_seed": seed_metrics,
                    "official_chemical_mean": {
                        metric: float(np.mean([seed[metric] for seed in seed_metrics]))
                        for metric in METRICS
                    },
                    "official_chemical_std": {
                        metric: float(np.std([seed[metric] for seed in seed_metrics]))
                        for metric in METRICS
                    },
                    "prompt_fidelity_over_attempts_pct": 100.0
                    * sum(row["prompt_compliant"] for row in selected)
                    / 3000,
                    "prompt_fidelity_over_commits_pct": 100.0
                    * sum(row["prompt_compliant"] for row in selected)
                    / sum(row["committed"] for row in selected),
                }
            )
    result = {
        "schema": "fragment_attachment_headline_rescore_v1",
        "evidence_role": "Retrospective exact rescore of frozen runs; no new generation or model selection",
        "input_sha256": dict(sorted(input_hashes.items())),
        "official_evaluator_sha256": evaluator_hash,
        "official_property_sha256": property_hash,
        "auditor_sha256": sha256(Path(__file__)),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "software": {"python": sys.version.split()[0], "rdkit": rdkit.__version__},
        "rows": rows,
        "summary": summary,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=OUTPUT_DIR, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, OUTPUT)
    for row in summary:
        print(
            row["arm"],
            row["task"],
            row["official_chemical_mean"],
            "prompt_fidelity",
            row["prompt_fidelity_over_attempts_pct"],
        )


if __name__ == "__main__":
    main()
