"""Write the five-prompt, zero-oracle fragment proposal support smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import rdBase

from compose_v4.benchmark.fragment_constrained import load_genmol_prompts
from compose_v4.benchmark.fragment_constrained_runner import (
    ProposalLimits,
    run_smoke_panel,
)

DEFAULT_ASSET = Path(
    "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
)
DEFAULT_OUTPUT = Path("diagnostics/fragment_constrained_proposal_smoke_v1/result.json")
MATERIAL_CODE = (
    Path("src/compose_v4/benchmark/fragment_constrained.py"),
    Path("src/compose_v4/benchmark/fragment_constrained_runner.py"),
    Path("src/compose_v4/control/edit_program.py"),
    Path("src/compose_v4/control/edit_program_graph.py"),
    Path("src/compose_v4/experiments/whole_ring_plan.py"),
    Path("tools/fragment_constrained_proposal_smoke.py"),
)


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


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)


def build_artifact(
    *,
    asset: Path,
    variant_index: int,
    limits: ProposalLimits,
) -> dict:
    prompts = load_genmol_prompts(asset)
    result = run_smoke_panel(prompts, variant_index=variant_index, limits=limits)
    return {
        **result,
        "evidence_class": "computed_zero_oracle_support_smoke",
        "claim_boundary": (
            "five deterministic executor-validated proposals; not a 100-sample "
            "benchmark run, optimization result, or comparator comparison"
        ),
        "configuration": {
            "variant_index": variant_index,
            "limits": {
                "max_active_atoms": limits.max_active_atoms,
                "max_primitives": limits.max_primitives,
                "max_blocks": limits.max_blocks,
            },
        },
        "inputs": {
            str(asset): _sha256(asset),
            "manifest_role": "all 50 prompts loaded; first prompt per task frozen before execution",
        },
        "provenance": {
            "code_revision": _revision(),
            "code_revision_role": (
                "base revision; material file hashes bind the requested uncommitted changes"
            ),
            "material_code_committed": False,
            "material_code_sha256": {
                str(path): _sha256(path) for path in MATERIAL_CODE
            },
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
            "deterministic": True,
            "cpu_only": True,
            "oracle_calls": 0,
            "scoring_calls": 0,
            "random_seed": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant-index", type=int, default=0)
    parser.add_argument("--max-active-atoms", type=int, default=40)
    parser.add_argument("--max-primitives", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=8)
    args = parser.parse_args()
    limits = ProposalLimits(
        max_active_atoms=args.max_active_atoms,
        max_primitives=args.max_primitives,
        max_blocks=args.max_blocks,
    )
    artifact = build_artifact(
        asset=args.asset,
        variant_index=args.variant_index,
        limits=limits,
    )
    _atomic_json(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "attempted": artifact["attempted_prompts"],
                "completed": artifact["completed"],
                "abstained": artifact["abstained"],
                "oracle_calls": artifact["oracle_calls"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
