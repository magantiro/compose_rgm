"""Per-task prescreen initialization banks in the production initialization format.

SELECTION (deterministic, documented): from the frozen top-40 prescreen seed bank take
the 8 highest-scoring molecules, then 8 more by greedy Morgan-fingerprint diversity
against those already chosen.  The fingerprint is used ONLY to spread the population
across distinct structural niches -- it is never compared against a benchmark target.
Population size stays at the frozen production `count`, so no controller hyperparameter
moves; only WHICH molecules initialize the run changes.

The emitted candidate rows carry `endpoint`, `source_id` and `state` and deliberately
carry NO score and NO task field, because the production loader refuses initialization
containing task information.  The prescreen scores stay in the frozen table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.rewrite.trace_shard import encode_state

RDLogger.DisableLog("rdApp.*")
SEEDS = Path("diagnostics/pmo_prescreen_v1/seeds")
OUT = Path("diagnostics/pmo_prescreen_v1/initialization")
SLOTS = 48
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def select(smiles: list[str], count: int) -> list[str]:
    top = smiles[: count // 2]
    chosen = list(top)
    fps = {s: GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles}
    while len(chosen) < count and len(chosen) < len(smiles):
        best, best_sim = None, 2.0
        for cand in smiles:
            if cand in chosen:
                continue
            sim = max(
                DataStructs.TanimotoSimilarity(fps[cand], fps[c]) for c in chosen
            )
            if sim < best_sim:
                best, best_sim = cand, sim
        if best is None:
            break
        chosen.append(best)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--source-sha256", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(SEEDS.glob("*.json")):
        task = path.stem
        bank = json.loads(path.read_text())
        picked = select(bank["smiles"], args.count)
        candidates = []
        for rank, smiles in enumerate(picked):
            graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)
            candidates.append({
                "endpoint": smiles,
                "source_id": f"{path.as_posix()}:{rank}",
                "state": encode_state(graph),
            })
        body = {
            "accounting": "all initialization scores count against each run's oracle budget",
            "available_unique": len(bank["smiles"]),
            "candidates": candidates,
            "count": len(candidates),
            "new_oracle_calls": len(candidates),
            "schema_version": "pmo_prescreen_initialization_v1",
            "seed": 0,
            "source_sha256": args.source_sha256,
            "task_independent": False,
            "selection_rule": (
                "top-8 by this task's official prescreen oracle score, then 8 by greedy "
                "Morgan diversity over the frozen top-40; no target structure used"
            ),
        }
        body["lock_sha256"] = identity(body)
        dest = OUT / f"{task}.json"
        dest.write_text(json.dumps(body, indent=1) + "\n")
        print(f"  {task:26s} {len(candidates)} candidates  lock={body['lock_sha256'][:16]}")
    print(f"\nwrote {len(list(OUT.glob('*.json')))} prescreen initialization banks to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
