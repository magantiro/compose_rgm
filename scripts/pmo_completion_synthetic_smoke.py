#!/usr/bin/env python
"""End-to-end SYNTHETIC-scorer smoke of the scored A/B chain. ZERO oracle calls.

Runs the real ``execute_task`` -- the real controller, campaign, ledger, budget
termination and snapshot path -- against a deterministic synthetic scorer, and
then proves from the campaign's OWN artifacts that the completion law reached
the proposals.  A declared flag that no proposal carries is an inert repair, and
a signature check cannot see that.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BUDGET = int(os.environ.get("SMOKE_BUDGET", "40"))


def synthetic(smiles: str) -> float:
    """Deterministic, bounded, structure-dependent, and not an oracle."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    rings = mol.GetRingInfo().NumRings()
    return min(1.0, 0.02 * mol.GetNumAtoms() + 0.05 * rings)


def completion_census(folder: Path) -> dict:
    """Count proposals whose modules carry a repaired completion."""

    census = collections.Counter()
    sizes = []
    for path in folder.rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if "completion" in node and "requested_size" in node:
                    census[str(node["completion"])] += 1
                    census["total"] += 1
                    sizes.append(int(node.get("inserted_atoms") or 0))
                    if int(node.get("requested_size") or 0) > 8:
                        census["requested_size_gt_8"] += 1
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return {
        "counts": dict(census),
        "max_inserted_atoms": max(sizes) if sizes else 0,
        "inserted_atoms_ge_9": sum(1 for s in sizes if s >= 9),
    }


def main() -> None:
    from pmo_completion_scored_ab import load_local_contract  # noqa: E402

    from compose_v4.experiments.pmo_population_v1 import execute_task

    contract = load_local_contract()
    out = {}
    with tempfile.TemporaryDirectory() as workspace:
        for arm, spec in contract["arms"].items():
            folder = Path(workspace) / arm
            folder.mkdir(parents=True)
            result = execute_task(
                contract, ROOT, folder, "celecoxib_rediscovery",
                evaluate=synthetic,
                charged_calls_per_task=BUDGET,
                enable_online_memory=bool(spec["enable_online_memory"]),
                completion_law=spec["completion_law"],
            )
            out[arm] = {
                "charged": result["charged_oracle_calls"],
                "best": result["best_score"],
                "arm": result["arm"],
                "completion_census": completion_census(folder),
            }
            print(f"{arm}: charged={out[arm]['charged']} best={out[arm]['best']} "
                  f"completions={out[arm]['completion_census']['counts']}",
                  flush=True)

    control = out["A_deployed_b"]["completion_census"]["counts"]
    treatment = out["B_completion"]["completion_census"]["counts"]
    problems = []
    for arm in out.values():
        if arm["charged"] != BUDGET:
            problems.append(f"budget not enforced: charged {arm['charged']} of {BUDGET}")
    if control.get("total"):
        problems.append("the CONTROL arm carries repaired-completion provenance")
    if not treatment.get("component"):
        problems.append(
            "the TREATMENT arm carries NO component completion: the law is INERT "
            "on the production campaign path"
        )
    # NOT a pass criterion: the predeclared validity check is component OR
    # size>8, and a 40-call smoke sees few completions. Recorded so the scored
    # run can be read against it.
    notes = []
    if not treatment.get("requested_size_gt_8"):
        notes.append(
            "no completion in this short smoke requested a size past the v1 "
            "ceiling; the downstream work limit (max_primitives 32 / max_blocks "
            "8) is the candidate explanation and is measured separately"
        )
    verdict = "CHAIN_READY" if not problems else "CHAIN_NOT_READY"
    report = {
        "schema_version": "pmo_completion_synthetic_smoke_v1",
        "new_oracle_calls": 0,
        "scorer": "synthetic, deterministic, structure-dependent",
        "budget_per_arm": BUDGET,
        "arms": out,
        "problems": problems,
        "notes": notes,
        "verdict": verdict,
    }
    target = ROOT / "diagnostics/pmo_completion_repair_v1/synthetic_smoke_v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, sort_keys=True, indent=1))
    print("\nVERDICT:", verdict)
    for problem in problems:
        print("  -", problem)
    for note in notes:
        print("  note:", note)
    sys.exit(0 if verdict == "CHAIN_READY" else 1)


main()
