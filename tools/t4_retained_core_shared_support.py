"""Seal shared retained-core support on the frozen nine-cell completion matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from compose_v4.experiments.t4_fiber_campaign import Fiber


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _seal(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained-core artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": _identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def _scientific_projection(payload: dict) -> dict:
    projected = json.loads(json.dumps(payload))
    projected.pop("runtime", None)
    for cell in projected["cells"]:
        cell.pop("wall_seconds", None)
    projected["schema_version"] = "t4_retained_core_shared_support_scientific_v1"
    return projected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-contract",
        default="configs/t4_shared_controller_completion_v1.json",
    )
    parser.add_argument(
        "--output",
        default="diagnostics/t4_retained_core_shared_support_v1/result.json",
    )
    parser.add_argument("--compare-scientific")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_path = root / arguments.source_contract
    source = json.loads(source_path.read_text())
    if source.get("status") != "PREPARED_PENDING_ZERO_ORACLE_GATE":
        raise ValueError("unexpected shared-completion source contract")
    cells = source.get("cells")
    if not isinstance(cells, list) or len(cells) != 9:
        raise ValueError(
            "shared retained-core support requires the exact nine-cell matrix"
        )
    configuration = {
        "maximum_fragment_atoms": 16,
        "maximum_stages": 2,
        "maximum_primitives": 32,
        "maximum_prefixes": 4096,
        "support": "compose_valid",
        "proposal_expert": "route_complete_region",
        "program_family": "retained_core_prune",
        "target_conditioning": False,
    }
    rows = []
    for cell in cells:
        started = time.perf_counter()
        graph = pad_molecular_graph(
            smiles_to_molecular_graph(cell["source_smiles"]), 48
        )
        proposals = enumerate_retained_core_prunes(
            graph,
            maximum_fragment_atoms=configuration["maximum_fragment_atoms"],
            maximum_stages=configuration["maximum_stages"],
            maximum_primitives=configuration["maximum_primitives"],
            maximum_prefixes=configuration["maximum_prefixes"],
        )
        fiber = Fiber(
            cell["source_smiles"],
            float(cell["delta"]),
            support=configuration["support"],
        )
        eligible = {}
        for proposal in proposals:
            endpoint = fiber.check(molecular_graph_to_smiles(proposal.product))
            if endpoint is None or endpoint["smiles"] == cell["source_smiles"]:
                continue
            eligible[endpoint["smiles"]] = {
                **endpoint,
                "primitive_edits": len(proposal.actions),
                "regions": len(proposal.stages),
                "deleted": sum(
                    int(stage["deleted_atoms"]) for stage in proposal.stages
                ),
            }
        rows.append(
            {
                "cell_key": cell["cell_key"],
                "delta": cell["delta"],
                "exact_programs": len(proposals),
                "eligible_unique": len(eligible),
                "eligible": [eligible[key] for key in sorted(eligible)],
                "wall_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"[{cell['cell_key']}] exact={len(proposals)} eligible={len(eligible)}",
            flush=True,
        )
    payload = {
        "schema_version": "t4_retained_core_shared_support_v1",
        "evidence_status": "computed_zero_oracle_exact_execution_support",
        "configuration": configuration,
        "cells": rows,
        "gate": {
            "complete_nine_cell_census": len(rows) == 9,
            "every_cell_has_nonzero_eligible_support": all(
                row["eligible_unique"] > 0 for row in rows
            ),
        },
        "inputs": {
            "source_contract": arguments.source_contract,
            "source_contract_sha256": _sha256(source_path),
            "retained_core_code_sha256": _sha256(
                root / "src/compose_v4/control/retained_core_pruning.py"
            ),
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "runtime": {
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "claim_boundary": (
            "Free exact proposal support only; no docking utility or benchmark claim"
        ),
    }
    scientific = _scientific_projection(payload)
    if arguments.compare_scientific:
        previous = json.loads((root / arguments.compare_scientific).read_text())
        if previous.get("payload") != scientific or previous.get(
            "payload_sha256"
        ) != _identity(scientific):
            raise RuntimeError("retained-core scientific rerun is not byte-identical")
    output = root / arguments.output
    _seal(output, payload)
    _seal(output.with_name("scientific_result.json"), scientific)
    print(output)


if __name__ == "__main__":
    main()
