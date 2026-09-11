"""Inspect the recovered exact routes without scoring any additional molecules."""

from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter
from itertools import pairwise
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from compose_v4.experiments.winner_paths import graph_stats, replay
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import implementation_closure, publish, sha

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "diagnostics/pmo_public_winner_recovery"


def audit():
    result_path = DIRECTORY / "result.json"
    result = json.loads(result_path.read_text())
    inputs = {str(result_path.relative_to(ROOT)): sha(result_path)}
    rows = []
    for item in result["paths"]:
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"saved path hash mismatch: {path}")
        inputs[item["path"]] = sha(path)
        route = json.loads(path.read_text())["result"]
        if route["status"] != "witness_found":
            rows.append({"source_id": item["source_id"], "status": route["status"]})
            continue
        states = replay(route["source_state"], route["actions"], route["target_2d"])
        if states != route["states"]:
            raise ValueError(f"state-by-state replay mismatch: {path}")
        molecules = []
        for depth, payload in enumerate(states):
            graph = decode_state(payload)
            smiles = canonical_state_key(graph)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"unparseable replay at {path}:{depth}")
            molecules.append(
                {
                    "depth": depth,
                    "smiles": smiles,
                    **graph_stats(graph),
                    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                }
            )
        families = [decode_action(action)[0] for action in route["actions"]]
        rows.append(
            {
                "source_id": item["source_id"],
                "status": route["status"],
                "steps": len(families),
                "family_counts": dict(sorted(Counter(families).items())),
                "min_heavy_atoms": min(m["heavy_atoms"] for m in molecules),
                "max_heavy_atoms": max(m["heavy_atoms"] for m in molecules),
                "initial": molecules[0],
                "terminal": molecules[-1],
                "ring_events": [
                    {
                        "step": index,
                        "family": families[index - 1],
                        "before": previous,
                        "after": current,
                    }
                    for index, (previous, current) in enumerate(pairwise(molecules), 1)
                    if (previous["cycle_rank"], previous["aromatic_rings"])
                    != (current["cycle_rank"], current["aromatic_rings"])
                ],
                "states": molecules,
                "learned_law_probability": "not evaluated",
                "macro_program_count": "not compiled or certified; primitive count is not option-decision count",
            }
        )
    closure = implementation_closure(Path(__file__).resolve())
    closure["tools/ivg_winner_paths.py"] = sha(ROOT / "tools/ivg_winner_paths.py")
    output = {
        "schema_version": "pmo_public_target_route_audit_v1",
        "input_sha256": inputs,
        "implementation_sha256": closure,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_worktree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "configuration": {
            "all_locked_routes": True,
            "replay": "exact saved-slot states and production executor",
            "seed": None,
            "deterministic": True,
            "exclusions": [],
        },
        "software": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "device": "cpu",
        },
        "precision": "integer executor states",
        "new_oracle_calls": 0,
        "split": "answer-known development",
        "routes": rows,
        "witness_count": sum(r["status"] == "witness_found" for r in rows),
        "route_count": len(rows),
        "search_seconds": sum(r["seconds"] for r in result["paths"]),
        "interpretation": "executor witness coverage and exact endpoint identity only; not blind optimization, learned-law support, or shortest paths",
    }
    publish(DIRECTORY / "route_audit.json", output)
    print(
        json.dumps(
            [
                {
                    k: row[k]
                    for k in (
                        "source_id",
                        "steps",
                        "family_counts",
                        "min_heavy_atoms",
                        "max_heavy_atoms",
                    )
                }
                for row in rows
            ],
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    audit()
