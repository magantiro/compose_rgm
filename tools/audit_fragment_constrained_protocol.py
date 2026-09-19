#!/usr/bin/env python3
"""Write a deterministic zero-oracle audit of the frozen fragment prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from rdkit import rdBase

from compose_v4.benchmark.fragment_constrained import (
    SCHEMA_VERSION,
    check_fragment_constraint,
    load_genmol_prompts,
)

AUDIT_SCHEMA = "fragment_constrained_protocol_audit_v1"
EXPECTED_INPUT_SHA256 = (
    "a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9"
)
ADAPTER_PATH = Path("src/compose_v4/benchmark/fragment_constrained.py")

# Reported means and standard deviations from InVirtuoGen Table 4.  This table
# uses GenMol's one-step linker result, not its rejection-filtered two-step row.
PAPER_COMPARATORS = {
    "motif_extension": {
        "SAFE-GPT": [[96.10, 1.90], [66.80, 1.20], [18.60, 2.10], [0.562, 0.003]],
        "GenMol": [[82.90, 0.10], [77.50, 0.10], [30.10, 0.40], [0.617, 0.002]],
        "InVirtuoGen": [[68.97, 0.759], [96.83, 0.290], [39.27, 1.078], [0.620, 0.005]],
    },
    "linker_design": {
        "SAFE-GPT": [[76.60, 5.10], [82.50, 1.90], [21.70, 1.10], [0.545, 0.007]],
        "GenMol_one_step": [[16.70, 0.20], [97.80, 0.50], [4.30, 0.40], [0.530, 0.002]],
        "InVirtuoGen": [[60.37, 0.573], [84.76, 1.620], [22.33, 1.250], [0.520, 0.004]],
    },
    "scaffold_morphing": {
        "SAFE-GPT": [[58.90, 6.80], [70.40, 5.70], [16.70, 2.30], [0.514, 0.011]],
        "GenMol_one_step": [[16.70, 0.20], [97.80, 0.50], [4.30, 0.40], [0.530, 0.002]],
        "InVirtuoGen": [[60.37, 0.573], [84.76, 1.620], [22.33, 1.250], [0.520, 0.004]],
    },
    "superstructure_generation": {
        "SAFE-GPT": [[95.70, 2.00], [83.00, 5.90], [14.30, 3.70], [0.573, 0.028]],
        "GenMol": [[97.50, 0.90], [83.60, 1.00], [34.80, 1.00], [0.599, 0.009]],
        "InVirtuoGen": [[75.70, 0.898], [99.41, 0.157], [27.43, 0.953], [0.730, 0.001]],
    },
    "scaffold_decoration": {
        "SAFE-GPT": [[97.70, 0.30], [74.70, 2.50], [10.00, 1.40], [0.575, 0.008]],
        "GenMol": [[96.60, 0.80], [82.70, 1.80], [31.80, 0.50], [0.591, 0.001]],
        "InVirtuoGen": [[90.70, 0.616], [88.58, 1.130], [36.37, 1.096], [0.560, 0.003]],
    },
    "average": {
        "SAFE-GPT": [[85.00, 1.788], [75.48, 1.773], [16.26, 1.031], [0.550, 0.006]],
        "GenMol": [[62.08, 0.242], [87.88, 0.436], [21.06, 0.263], [0.570, 0.002]],
        "InVirtuoGen": [[71.22, 0.399], [90.87, 0.445], [29.55, 0.813], [0.590, 0.001]],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_audit(input_path: Path) -> dict:
    input_hash = _sha256(input_path)
    if input_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(
            f"fragment input hash mismatch for {input_path}: {input_hash}; "
            f"expected {EXPECTED_INPUT_SHA256}"
        )
    prompts = load_genmol_prompts(input_path)
    reference_satisfied = sum(
        check_fragment_constraint(prompt, prompt.original_smiles).satisfied
        for prompt in prompts
    )
    task_counts = Counter(prompt.task.value for prompt in prompts)
    fragment_count_histogram = Counter(len(prompt.fragments) for prompt in prompts)
    attachment_count_histogram = Counter(
        sum(fragment.count("*") for fragment in prompt.fragments) for prompt in prompts
    )
    return {
        "schema": AUDIT_SCHEMA,
        "adapter_schema": SCHEMA_VERSION,
        "status": "zero_oracle_adapter_ready",
        "scientific_scope": "SAFE-drug fragment-constrained conditional generation",
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "upstream_repository": "https://github.com/NVIDIA-BioNeMo/genmol",
            "upstream_commit": "add09fc83b7255bd09c797e527c0f4b51f5fb7c1",
            "upstream_path": "data/fragments.csv",
            "dataset_license": "CC-BY-4.0",
        },
        "counts": {
            "drugs": len({prompt.drug_name for prompt in prompts}),
            "tasks": len(task_counts),
            "explicit_prompts": len(prompts),
            "attempts_per_prompt": 100,
            "attempts_per_task_per_run": 1000,
            "reported_runs": 3,
            "reference_drug_constraint_checks": len(prompts),
            "reference_drug_constraint_satisfied": reference_satisfied,
            "task_counts": dict(sorted(task_counts.items())),
            "fragment_count_histogram": {
                str(key): value
                for key, value in sorted(fragment_count_histogram.items())
            },
            "attachment_count_histogram": {
                str(key): value
                for key, value in sorted(attachment_count_histogram.items())
            },
        },
        "metrics": {
            "validity": "valid outputs / 100 attempts",
            "uniqueness": "unique valid outputs / valid outputs",
            "diversity": "mean pairwise Tanimoto distance; Morgan radius=2 bits=2048",
            "quality": "unique valid QED>=0.6 and SA<=4 outputs / 100 attempts",
            "central_distance": "mean distance to original; Morgan radius=2 bits=1024",
            "compose_addition": "explicit fragment-constraint validity",
        },
        "reported_comparators": {
            "evidence_class": "reported",
            "source": "InVirtuoGen arXiv:2509.26405v2 Table 4",
            "field_order": [
                "validity_percent",
                "uniqueness_percent",
                "quality_percent",
                "diversity",
            ],
            "value_encoding": "[mean, standard_deviation]",
            "table": PAPER_COMPARATORS,
        },
        "sources": {
            "genmol_paper": "https://arxiv.org/abs/2501.06158",
            "invirtuogen_paper": "https://arxiv.org/abs/2509.26405",
            "safe_paper": "https://arxiv.org/abs/2310.10773",
            "invirtuogen_commit": "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb",
            "safe_commit": "d162d23e3e8c8a77b54c559e88a35d08b3b1d1b5",
        },
        "limitations": [
            "no COMPOSE candidates generated or scored",
            "InVirtuoGen superstructure attachment prompts are randomized before seed initialization",
            "connected-or-null COMPOSE runtime needs a protected two-fragment linker proposal contract",
        ],
        "provenance": {
            "code_revision": _revision(),
            "adapter_path": str(ADAPTER_PATH),
            "adapter_sha256": _sha256(ADAPTER_PATH),
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "deterministic": True,
            "oracle_calls": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/fragment_constrained_protocol_v1/audit.json"),
    )
    args = parser.parse_args()
    payload = build_audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
