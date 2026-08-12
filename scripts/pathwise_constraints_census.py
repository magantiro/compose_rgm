#!/usr/bin/env python3
"""Held-in viability census for the pathwise-constraint workstream. LOCAL ONLY.

WHAT THIS CAN AND CANNOT ANSWER
-------------------------------
Everything here reads SOURCE MOLECULES only: motif derivation, motif geometry,
eligibility and the terminal objective all run on RDKit plus the frozen DRD2
SVM, with no successor kernel and no Modal.

That bounds what the census can settle. It answers:

    does a protected motif exist, how large is it, how much of the molecule
    does it lock, how many sources are eligible, and does the objective have
    headroom at step zero?

It CANNOT answer the two questions the vacuity gate actually turns on:

    how often does unconstrained control violate the motif mid-path, and what
    fraction of the legal successor support does the mask remove?

Both require enumerating successors, which requires the frozen R_theta on
Modal. They are resolved by the smoke run, not here. Anything in this file
that gestures at them is labelled PROXY_NOT_MEASUREMENT and must not be quoted
as the gate.

Status of the emitted artifact: SMOKE_HELD_IN (census component).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
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

# Frozen goal language, verbatim from modal_apps/retarget_intervention_app.py.
# Not re-derived and not retuned: this workstream borrows the goal, it does not
# invent one.
POTENCY_THRESHOLD = 0.5
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)

#: Distinct from the retargeting cohort seed (20260812) so the pathwise panel
#: does not simply re-draw the first 57 shuffled held-in entries.
SEED = 20260813


def quantiles(values: list[float], points=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for point in points:
        index = min(len(ordered) - 1, int(point * (len(ordered) - 1) + 0.5))
        out[f"p{int(point * 100):02d}"] = round(float(ordered[index]), 4)
    return out


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
        help="sources to exclude so the pathwise panel is disjoint from it",
    )
    parser.add_argument("--scan", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out", type=Path, default=REPO / "diagnostics/pathwise_constraints_census.json"
    )
    args = parser.parse_args()

    reserve = json.load(gzip.open(args.reserve, "rt"))
    held_in = sorted(reserve["training_source_keys"])
    held_out = set(reserve["reserve_source_keys"])
    random.Random(args.seed).shuffle(held_in)
    scanned = held_in[: args.scan]

    excluded = set()
    if args.retarget_cohort.exists():
        cohort = json.loads(args.retarget_cohort.read_text())
        excluded = {row["source"] for row in cohort["sources"]}

    # HELD-OUT BOUNDARY ASSERTION. Cheap, and it is the one mistake that would
    # invalidate every downstream artifact silently.
    leak = held_out.intersection(scanned)
    if leak:
        raise SystemExit(f"held-out leak: {len(leak)} scanned sources are in the reserve")

    normalizers = json.loads(args.normalizers.read_text())["normalizers"]
    s_drd2 = normalizers["drd2_logodds"]["iqr"]
    s_qed = normalizers["qed"]["iqr"]
    s_logp = normalizers["clogp"]["iqr"]
    oracle = load_default_oracle(str(args.oracle_manifest))

    import numpy as np

    potency_cut = float(np.log(POTENCY_THRESHOLD / (1 - POTENCY_THRESHOLD)))

    rows: list[dict] = []
    reasons: Counter = Counter()
    parsed, no_ring, writer_errors = 0, 0, 0

    for smiles in scanned:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            reasons["unparseable"] += 1
            continue
        parsed += 1
        try:
            motif = derive_protected_motif(smiles)
        except Exception as error:  # noqa: BLE001
            writer_errors += 1
            print(f"MOTIF DERIVATION FAILED {smiles}: {error}", file=sys.stderr)
            continue
        if motif is None:
            no_ring += 1
        rows.append(
            {
                "smiles": smiles,
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "bonds": mol.GetNumBonds(),
                "motif": motif,
                "in_retarget_cohort": smiles in excluded,
            }
        )

    # Terminal objective at step zero, batched.
    smis = [row["smiles"] for row in rows]
    margins = oracle.margin_many(smis)
    for row, drd2 in zip(rows, margins, strict=True):
        mol = Chem.MolFromSmiles(row["smiles"])
        try:
            qed = float(QED.qed(mol))
        except Exception:  # noqa: BLE001
            qed = float("nan")
        logp = float(Crippen.MolLogP(mol))
        row["potency_margin"] = (float(drd2) - potency_cut) / s_drd2
        row["qed_margin"] = (qed - QED_FLOOR) / s_qed
        row["logp_margin"] = min(logp - LOGP_BOX[0], LOGP_BOX[1] - logp) / s_logp
        row["b_worst_margin"] = min(
            row["potency_margin"], row["qed_margin"], row["logp_margin"]
        )
        row["satisfies_b"] = row["b_worst_margin"] >= 0.0
        row["satisfies_d"] = min(row["qed_margin"], row["logp_margin"]) >= 0.0
        row["satisfies_p"] = row["potency_margin"] >= 0.0

    eligible: list[dict] = []
    for row in rows:
        verdict = eligibility(
            row["smiles"], row["motif"], already_satisfies_goal=row["satisfies_b"]
        )
        row["verdict"] = verdict
        if row["in_retarget_cohort"]:
            verdict["reasons"].append("in_retarget_cohort")
            verdict["eligible"] = False
        for reason in verdict["reasons"]:
            reasons[reason] += 1
        if verdict["eligible"]:
            eligible.append(row)

    with_motif = [row for row in rows if row["motif"] is not None]
    motif_atoms = [row["motif"].atom_count for row in with_motif]
    motif_fraction = [row["motif"].fraction for row in with_motif]
    free_atoms = [row["motif"].free_atoms for row in with_motif]
    motif_rings = [row["motif"].ring_count for row in with_motif]

    # PROXY_NOT_MEASUREMENT. The share of the molecular graph the motif locks
    # bounds nothing rigorously; it is reported because the operators act on
    # atoms and bonds, so a motif covering half the graph plausibly intersects
    # many legal edits. The real number comes from the kernel in the smoke run.
    bond_share = []
    for row in with_motif:
        if row["bonds"]:
            bond_share.append(row["motif"].bond_count / row["bonds"])

    payload = {
        "schema": "compose.pathwise.heldin_census",
        "status": "SMOKE_HELD_IN",
        "result_status_note": (
            "source-only census; the vacuity gate (mid-path violation rate) and the "
            "support-removal rate are NOT resolved here and require the Modal kernel"
        ),
        "motif_rule": MOTIF_RULE_VERSION,
        "pool": "held-in training_source_keys only",
        "seed": args.seed,
        "scanned": len(scanned),
        "parsed": parsed,
        "writer_errors": writer_errors,
        "goal": {
            "definition": "B = P AND D, frozen from retarget_intervention_app.py",
            "potency_threshold": POTENCY_THRESHOLD,
            "qed_floor": QED_FLOOR,
            "logp_box": list(LOGP_BOX),
            "normalizers": {"drd2_logodds": s_drd2, "qed": s_qed, "clogp": s_logp},
        },
        "eligibility_constants": {
            "heavy_atom_band": list(HEAVY_ATOM_BAND),
            "min_motif_atoms": MIN_MOTIF_ATOMS,
            "motif_fraction_band": list(MOTIF_FRACTION_BAND),
            "min_free_atoms": MIN_FREE_ATOMS,
        },
        "motif_existence": {
            "sources": len(rows),
            "with_ring_system": len(with_motif),
            "acyclic": no_ring,
            "fraction_with_motif": round(len(with_motif) / len(rows), 4) if rows else 0.0,
        },
        "motif_geometry": {
            "atom_count": quantiles([float(v) for v in motif_atoms]),
            "fraction_of_molecule": quantiles(motif_fraction),
            "free_atoms": quantiles([float(v) for v in free_atoms]),
            "ring_count": quantiles([float(v) for v in motif_rings]),
            "PROXY_NOT_MEASUREMENT_bond_share": quantiles(bond_share),
        },
        "objective_headroom": {
            "satisfies_B_at_source": sum(1 for r in rows if r["satisfies_b"]),
            "satisfies_D_at_source": sum(1 for r in rows if r["satisfies_d"]),
            "satisfies_P_at_source": sum(1 for r in rows if r["satisfies_p"]),
            "denominator": len(rows),
            "note": (
                "sources already satisfying B are excluded by eligibility, so the "
                "panel has headroom on the binding term by construction"
            ),
        },
        "eligibility": {
            "eligible": len(eligible),
            "denominator": len(rows),
            "yield": round(len(eligible) / len(rows), 4) if rows else 0.0,
            "rejection_reasons": dict(reasons.most_common()),
            "note": (
                "reasons are not mutually exclusive; a source may fail several"
            ),
        },
        "not_an_eligibility_criterion": (
            "'the kernel offers motif-destroying successors' is deliberately NOT "
            "used to select sources: it is the numerator of the vacuity gate, and "
            "filtering on it would make a non-vacuous mask true by construction"
        ),
    }
    payload["census_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "smiles"}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
