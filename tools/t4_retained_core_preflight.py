"""Seal the zero-oracle BRAF retained-core proposal preflight."""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from pathlib import Path

from rdkit import rdBase

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.retained_core_pruning import enumerate_retained_core_prunes
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_fiber_campaign import Fiber
from compose_v4.experiments.t4_matched_pilot import seal, unseal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="diagnostics/t4_integrated_route_fiber_braf_v2/retained_core_preflight.json",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_path = root / "configs/t4_integrated_route_fiber_braf_v1.json"
    source = unseal(source_path)
    configuration = {
        "maximum_fragment_atoms": 16,
        "maximum_stages": 2,
        "maximum_primitives": 32,
        "maximum_prefixes": 4096,
        "delta": 0.6,
        "support": "compose_valid",
    }
    rows = []
    for cell in source["cells"][:2]:
        started = time.perf_counter()
        graph = pad_molecular_graph(smiles_to_molecular_graph(cell["smiles"]), 48)
        proposals = enumerate_retained_core_prunes(
            graph,
            maximum_fragment_atoms=configuration["maximum_fragment_atoms"],
            maximum_stages=configuration["maximum_stages"],
            maximum_primitives=configuration["maximum_primitives"],
            maximum_prefixes=configuration["maximum_prefixes"],
        )
        fiber = Fiber(cell["smiles"], configuration["delta"], support=configuration["support"])
        eligible = []
        for proposal in proposals:
            gate = fiber.check(molecular_graph_to_smiles(proposal.product))
            if gate is None:
                continue
            eligible.append(
                {
                    **gate,
                    "primitive_edits": len(proposal.actions),
                    "stages": list(proposal.stages),
                }
            )
        rows.append(
            {
                "cell": cell["cell"],
                "exact_programs": len(proposals),
                "eligible_unique": len(eligible),
                "eligible": sorted(eligible, key=lambda row: row["smiles"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    payload = {
        "schema_version": "t4_retained_core_preflight_v1",
        "evidence_status": "computed zero-oracle exact-execution preflight",
        "configuration": configuration,
        "cells": rows,
        "promotion_passed": all(row["eligible_unique"] > 0 for row in rows),
        "new_oracle_calls": 0,
        "inputs": {
            "source_contract": str(source_path.relative_to(root)),
            "source_contract_sha256": sha256_file(source_path),
            "retained_core_code_sha256": sha256_file(
                root / "src/compose_v4/control/retained_core_pruning.py"
            ),
        },
        "runtime": {
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "claim_boundary": (
            "Free feasibility support only; no docking utility or benchmark-performance claim"
        ),
    }
    destination = root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    seal(destination, payload)
    print(destination)


if __name__ == "__main__":
    main()
