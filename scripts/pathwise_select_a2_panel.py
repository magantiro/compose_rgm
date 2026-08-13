#!/usr/bin/env python3
"""Freeze the Stage A2 panel: NEW held-in sources for the cLogP corridor.

OUTCOME-INDEPENDENT. Reads the source molecule only. No successor enumeration,
no trajectory, no arm outcome.

DISJOINTNESS, enforced and asserted:
  * the 6 stage-A pathwise sources        -- A2 must be new sources, not more
                                             rollouts on the same molecules
  * the 30 retargeting calibration sources
  * `reserve_source_keys` (held-out)      -- asserted empty intersection

ONE ELIGIBILITY CRITERION IS NEW, AND IT MATTERS
------------------------------------------------
A2 requires `cLogP(x_0)` to be INSIDE the frozen corridor. A source that starts
outside it can never "leave and return", so it cannot produce the event this
census counts. In stage A this was not required and it cost half the sample:
21 of 42 trajectories -- and 3 of 6 sources -- started outside the corridor and
were excluded from the denominator after the fact.

This is not a threshold change. The corridor is unchanged at the frozen held-in
interquartile range. It is a statement about which sources the constraint is
even applicable to, it reads x_0 only, and it makes every A2 source contribute.

Status of the emitted artifact: DESIGN_ONLY until the run is authorised.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import QED, Crippen  # noqa: E402

from compose_v4.drd2_oracle import load_default_oracle  # noqa: E402
from compose_v4.experiments.pathwise_constraints import HEAVY_ATOM_BAND  # noqa: E402
from compose_v4.experiments.pathwise_reversible_families import (  # noqa: E402
    clogp_corridor,
)

RDLogger.DisableLog("rdApp.*")

POTENCY_THRESHOLD = 0.5
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)
SLOTS = 48

#: Distinct from stage A (20260813) and the retargeting cohort (20260812).
SEED = 20260814


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reserve", type=Path,
        default=REPO / "diagnostics/editing_v2_matched_validation_reserve_ids.json.gz")
    parser.add_argument(
        "--normalizers", type=Path,
        default=REPO / "diagnostics/retarget_goal_language_normalizers.json")
    parser.add_argument(
        "--oracle-manifest", type=Path,
        default=REPO / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json")
    parser.add_argument(
        "--retarget-cohort", type=Path,
        default=REPO / "diagnostics/retarget_calibration_cohort.json")
    parser.add_argument(
        "--stage-a-panel", type=Path,
        default=REPO / "diagnostics/pathwise_constraints_smoke_panel.json")
    parser.add_argument("--sources", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out", type=Path,
        default=REPO / "diagnostics/pathwise_a2_panel.json")
    args = parser.parse_args()

    reserve = json.load(gzip.open(args.reserve, "rt"))
    held_in = sorted(reserve["training_source_keys"])
    held_out = set(reserve["reserve_source_keys"])
    random.Random(args.seed).shuffle(held_in)

    excluded: set[str] = set()
    if args.retarget_cohort.exists():
        excluded |= {r["source"] for r in
                     json.loads(args.retarget_cohort.read_text())["sources"]}
    stage_a: set[str] = set()
    if args.stage_a_panel.exists():
        stage_a = {r["source"] for r in
                   json.loads(args.stage_a_panel.read_text())["sources"]}
    excluded |= stage_a

    norms = json.loads(args.normalizers.read_text())["normalizers"]
    s_drd2 = norms["drd2_logodds"]["iqr"]
    s_qed = norms["qed"]["iqr"]
    s_logp = norms["clogp"]["iqr"]
    oracle = load_default_oracle(str(args.oracle_manifest))
    low, high = clogp_corridor()

    import numpy as np

    potency_cut = float(np.log(POTENCY_THRESHOLD / (1 - POTENCY_THRESHOLD)))

    chosen: list[dict] = []
    scanned = 0
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for smiles in held_in:
        if len(chosen) >= args.sources:
            break
        scanned += 1
        if smiles in held_out:
            raise SystemExit("held-out leak: a reserve key appeared in the held-in pool")
        if smiles in excluded:
            reject("already_used_source")
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            reject("unparseable")
            continue
        heavy = mol.GetNumHeavyAtoms()
        if not HEAVY_ATOM_BAND[0] <= heavy <= HEAVY_ATOM_BAND[1]:
            reject("size_band")
            continue
        clogp = float(Crippen.MolLogP(mol))
        if not low <= clogp <= high:
            reject("source_outside_corridor")
            continue

        drd2 = float(oracle.margin_many([smiles])[0])
        try:
            qed = float(QED.qed(mol))
        except Exception:  # noqa: BLE001
            qed = float("nan")
        worst = min((drd2 - potency_cut) / s_drd2,
                    (qed - QED_FLOOR) / s_qed,
                    min(clogp - LOGP_BOX[0], LOGP_BOX[1] - clogp) / s_logp)
        if worst >= 0.0:
            reject("already_satisfies_goal")
            continue

        chosen.append({
            "index": len(chosen),
            "source": smiles,
            "slots": SLOTS,
            "heavy_atoms": heavy,
            "clogp": round(clogp, 4),
            "clogp_headroom_low": round(clogp - low, 4),
            "clogp_headroom_high": round(high - clogp, 4),
            "qed": round(qed, 4),
            "drd2_logodds": round(drd2, 4),
            "b_worst_margin_at_source": round(worst, 4),
        })

    if len(chosen) < args.sources:
        raise SystemExit(f"only {len(chosen)} eligible sources found")

    assert not ({r["source"] for r in chosen} & stage_a), "stage-A overlap"
    assert not ({r["source"] for r in chosen} & held_out), "held-out overlap"

    payload = {
        "schema": "compose.pathwise.a2_panel",
        "status": "DESIGN_ONLY",
        "held_out_opened": False,
        "pool": "held-in training_source_keys only",
        "seed": args.seed,
        "scanned": scanned,
        "rejected": rejected,
        "constraint": {
            "family": "B_physchem_corridor",
            "clogp_corridor": [low, high],
            "provenance": ("frozen held-in interquartile range, read at runtime "
                           "from retarget_goal_language_normalizers.json; "
                           "UNCHANGED from the three-family census"),
        },
        "eligibility": {
            "heavy_atom_band": list(HEAVY_ATOM_BAND),
            "source_clogp_inside_corridor": True,
            "excludes_sources_already_satisfying_B": True,
            "excludes_stage_a_sources": sorted(stage_a),
            "excludes_retarget_cohort": True,
            "note": ("requiring x_0 inside the corridor is applicability, not a "
                     "threshold change: a source that starts outside can never "
                     "leave and return"),
        },
        "selection_rule": (
            "first N eligible sources in shuffled scan order; no ranking, no "
            "preference, no successor enumeration, no arm outcome"),
        "sources": chosen,
    }
    payload["panel_sha256"] = hashlib.sha256(
        json.dumps(payload["sources"], sort_keys=True).encode()).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"corridor [{low:.4f}, {high:.4f}]  scanned {scanned}")
    print(f"rejected: {rejected}")
    print(f"selected {len(chosen)} sources; panel_sha256 {payload['panel_sha256'][:16]}")
    for row in chosen:
        print(f"  [{row['index']:2d}] heavy={row['heavy_atoms']:2d} "
              f"clogp={row['clogp']:6.3f} "
              f"(room -{row['clogp_headroom_low']:.2f}/+{row['clogp_headroom_high']:.2f}) "
              f"{row['source'][:46]}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
