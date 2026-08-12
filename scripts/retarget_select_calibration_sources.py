"""Freeze the held-in calibration cohort BEFORE the calibration runs.

Selection is outcome-independent and mechanical:

  * held-in only -- these sources are training molecules, so nothing here can
    contaminate the held-out evaluation panel;
  * representable and inside the frozen heavy-atom band;
  * does not already satisfy the developability region at step zero, because a
    source that starts inside it cannot show anything about reaching it.

No property is used to rank or prefer a source, and no rollout is run. Writing
the cohort to a committed file rather than choosing it inside the run entrypoint
means the set is auditable after the fact and identical across reruns.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

MIN_HEAVY, MAX_HEAVY = 18, 38
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)
CANONICAL_SLOTS = 48
SEED = 20260812


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", type=Path,
                        default=Path("diagnostics/editing_v2_matched_validation_reserve_ids.json.gz"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, QED

    RDLogger.DisableLog("rdApp.*")
    reserve = json.load(gzip.open(args.reserve, "rt"))
    held_in = sorted(reserve["training_source_keys"])
    random.Random(SEED).shuffle(held_in)

    chosen, scanned, rejected = [], 0, {"unparseable": 0, "size_band": 0,
                                        "already_developable": 0}
    for smiles in held_in:
        if len(chosen) >= args.count:
            break
        scanned += 1
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rejected["unparseable"] += 1
            continue
        if not (MIN_HEAVY <= mol.GetNumHeavyAtoms() <= MAX_HEAVY):
            rejected["size_band"] += 1
            continue
        try:
            qed, logp = QED.qed(mol), Crippen.MolLogP(mol)
        except Exception:  # noqa: BLE001
            rejected["unparseable"] += 1
            continue
        if qed >= QED_FLOOR and LOGP_BOX[0] <= logp <= LOGP_BOX[1]:
            rejected["already_developable"] += 1
            continue
        chosen.append({
            "index": len(chosen), "source": smiles, "slots": CANONICAL_SLOTS,
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "qed": round(float(qed), 4), "clogp": round(float(logp), 4),
        })

    digest = hashlib.sha256(
        json.dumps([c["source"] for c in chosen], sort_keys=True).encode()
    ).hexdigest()
    print(f"scanned {scanned:,} held-in molecules -> {len(chosen)} sources")
    for reason, count in rejected.items():
        print(f"  rejected {reason}: {count:,}")
    print(f"cohort sha256 {digest[:16]}")

    args.out.write_text(json.dumps({
        "schema": "compose.retarget.calibration_cohort",
        "status": "FROZEN_BEFORE_CALIBRATION_OUTCOME_INDEPENDENT",
        "pool": "held-in training sources only",
        "criteria": {"heavy_atoms": [MIN_HEAVY, MAX_HEAVY],
                     "qed_floor": QED_FLOOR, "logp_box": list(LOGP_BOX),
                     "rule": "exclude sources already inside the developability region"},
        "seed": SEED, "scanned": scanned, "rejected": rejected,
        "cohort_sha256": digest, "sources": chosen,
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
