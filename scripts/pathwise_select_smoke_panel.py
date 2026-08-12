#!/usr/bin/env python3
"""Freeze the held-in pathwise smoke panel. LOCAL ONLY, OUTCOME-INDEPENDENT.

Selection reads the SOURCE molecule and nothing else. It never enumerates a
successor, never runs a controller, and never consults an arm outcome, so
admitting a source cannot depend on whether the constraint turns out to bite.

The one criterion deliberately NOT applied is "the frozen kernel offers both
motif-preserving and motif-destroying legal successors". That property is the
numerator of the vacuity gate; selecting on it would guarantee a non-vacuous
mask by construction and turn the headline measurement into a definition.

Sources are accepted in shuffled scan order under a fixed seed. Nothing ranks
or prefers a source: the first N eligible entries win.

Status of the emitted artifact: DESIGN_ONLY until the smoke runs against it.
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
from compose_v4.experiments.pathwise_constraints import (  # noqa: E402
    HEAVY_ATOM_BAND,
    MIN_FREE_ATOMS,
    MIN_MOTIF_ATOMS,
    MOTIF_FRACTION_BAND,
    MOTIF_RULE_VERSION,
    derive_protected_motif,
    eligibility,
)

RDLogger.DisableLog("rdApp.*")

POTENCY_THRESHOLD = 0.5
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)
SEED = 20260813
SLOTS = 48  # frozen padding width, identical to every other COMPOSE panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reserve",
        type=Path,
        default=REPO / "diagnostics/editing_v2_matched_validation_reserve_ids.json.gz",
    )
    parser.add_argument(
        "--normalizers",
        type=Path,
        default=REPO / "diagnostics/retarget_goal_language_normalizers.json",
    )
    parser.add_argument(
        "--oracle-manifest",
        type=Path,
        default=REPO / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json",
    )
    parser.add_argument(
        "--retarget-cohort",
        type=Path,
        default=REPO / "diagnostics/retarget_calibration_cohort.json",
    )
    parser.add_argument("--sources", type=int, default=6, help="4-8 for the smoke")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "diagnostics/pathwise_constraints_smoke_panel.json",
    )
    args = parser.parse_args()

    reserve = json.load(gzip.open(args.reserve, "rt"))
    held_in = sorted(reserve["training_source_keys"])
    held_out = set(reserve["reserve_source_keys"])
    random.Random(args.seed).shuffle(held_in)

    excluded = set()
    if args.retarget_cohort.exists():
        excluded = {
            row["source"]
            for row in json.loads(args.retarget_cohort.read_text())["sources"]
        }

    normalizers = json.loads(args.normalizers.read_text())["normalizers"]
    s_drd2 = normalizers["drd2_logodds"]["iqr"]
    s_qed = normalizers["qed"]["iqr"]
    s_logp = normalizers["clogp"]["iqr"]
    oracle = load_default_oracle(str(args.oracle_manifest))

    import numpy as np

    potency_cut = float(np.log(POTENCY_THRESHOLD / (1 - POTENCY_THRESHOLD)))

    chosen: list[dict] = []
    scanned = 0
    rejected: dict[str, int] = {}

    for smiles in held_in:
        if len(chosen) >= args.sources:
            break
        scanned += 1
        if smiles in held_out:
            raise SystemExit("held-out leak: a reserve key appeared in the held-in pool")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rejected["unparseable"] = rejected.get("unparseable", 0) + 1
            continue
        motif = derive_protected_motif(smiles)

        drd2 = float(oracle.margin_many([smiles])[0])
        try:
            qed = float(QED.qed(mol))
        except Exception:  # noqa: BLE001
            qed = float("nan")
        logp = float(Crippen.MolLogP(mol))
        potency_margin = (drd2 - potency_cut) / s_drd2
        qed_margin = (qed - QED_FLOOR) / s_qed
        logp_margin = min(logp - LOGP_BOX[0], LOGP_BOX[1] - logp) / s_logp
        worst = min(potency_margin, qed_margin, logp_margin)

        verdict = eligibility(smiles, motif, already_satisfies_goal=worst >= 0.0)
        if smiles in excluded:
            verdict["eligible"] = False
            verdict["reasons"].append("in_retarget_cohort")
        if not verdict["eligible"]:
            for reason in verdict["reasons"]:
                rejected[reason] = rejected.get(reason, 0) + 1
            continue

        chosen.append(
            {
                "index": len(chosen),
                "source": smiles,
                "slots": SLOTS,
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "motif_smarts": motif.smarts,
                "motif_atoms": motif.atom_count,
                "motif_bonds": motif.bond_count,
                "motif_rings": motif.ring_count,
                "motif_fraction": round(motif.fraction, 4),
                "free_atoms": motif.free_atoms,
                "qed": round(qed, 4),
                "clogp": round(logp, 4),
                "drd2_logodds": round(drd2, 4),
                "b_worst_margin_at_source": round(worst, 4),
                "binding_term_at_source": min(
                    (
                        ("potency", potency_margin),
                        ("qed", qed_margin),
                        ("logp", logp_margin),
                    ),
                    key=lambda pair: pair[1],
                )[0],
            }
        )

    if len(chosen) < args.sources:
        raise SystemExit(f"only {len(chosen)} eligible sources found")

    payload = {
        "schema": "compose.pathwise.smoke_panel",
        "status": "DESIGN_ONLY",
        "held_out_opened": False,
        "pool": "held-in training_source_keys only",
        "motif_rule": MOTIF_RULE_VERSION,
        "seed": args.seed,
        "scanned": scanned,
        "rejected": rejected,
        "selection_rule": (
            "first N eligible sources in shuffled scan order; no ranking, no "
            "preference, no successor enumeration, no arm outcome"
        ),
        "eligibility_constants": {
            "heavy_atom_band": list(HEAVY_ATOM_BAND),
            "min_motif_atoms": MIN_MOTIF_ATOMS,
            "motif_fraction_band": list(MOTIF_FRACTION_BAND),
            "min_free_atoms": MIN_FREE_ATOMS,
            "excludes_sources_already_satisfying_B": True,
        },
        "goal": {
            "definition": "B = P AND D",
            "potency_threshold": POTENCY_THRESHOLD,
            "qed_floor": QED_FLOOR,
            "logp_box": list(LOGP_BOX),
        },
        "disjoint_from": [
            "diagnostics/retarget_calibration_cohort.json (30 sources, excluded by SMILES)",
            "reserve_source_keys (held-out; asserted empty intersection)",
        ],
        "sources": chosen,
    }
    payload["panel_sha256"] = hashlib.sha256(
        json.dumps(payload["sources"], sort_keys=True).encode()
    ).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
