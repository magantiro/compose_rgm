#!/usr/bin/env python3
"""Freeze the Stage B panel: 24 NEW held-in sources for the corridor experiment.

OUTCOME-INDEPENDENT AND EXCURSION-BLIND. Eligibility reads the source molecule
only:

    parses - heavy atoms in the frozen band - cLogP(x_0) inside the frozen
    corridor - potency headroom at step zero

**Nothing about whether a molecule previously produced an excursion enters
here.** Selecting on excursion propensity would make the primary pathwise
estimand -- the rate at which endpoint-only trajectories pass through forbidden
states -- true by construction, which is the failure mode this lane has spent
three stages avoiding. `test_stage_b_eligibility_is_excursion_blind` pins the
signature so it cannot be reintroduced quietly.

DISJOINTNESS, enforced and asserted: the 6 stage-A sources, the 12 A2 sources,
the 30 retargeting calibration sources, and `reserve_source_keys` (held-out).

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
from rdkit.Chem import Crippen  # noqa: E402

from compose_v4.drd2_oracle import load_default_oracle  # noqa: E402
from compose_v4.experiments.pathwise_constraints import HEAVY_ATOM_BAND  # noqa: E402
from compose_v4.experiments.pathwise_reversible_families import (  # noqa: E402
    clogp_corridor,
)

RDLogger.DisableLog("rdApp.*")

#: Frozen from the retargeting lane. The terminal objective is potency ALONE.
POTENCY_THRESHOLD = 0.5
SLOTS = 48

#: Distinct from stage A (20260813) and A2 (20260814).
SEED = 20260815


def eligibility(smiles: str, *, clogp: float, heavy: int, potency_margin: float,
                low: float, high: float) -> list[str]:
    """Source-only reasons to reject. No trajectory, no excursion history."""
    reasons: list[str] = []
    if not HEAVY_ATOM_BAND[0] <= heavy <= HEAVY_ATOM_BAND[1]:
        reasons.append("size_band")
    if not low <= clogp <= high:
        reasons.append("source_outside_corridor")
    if potency_margin >= 0.0:
        reasons.append("already_potent")
    return reasons


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
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out", type=Path, default=REPO / "diagnostics/pathwise_stage_b_panel.json")
    args = parser.parse_args()

    reserve = json.load(gzip.open(args.reserve, "rt"))
    held_in = sorted(reserve["training_source_keys"])
    held_out = set(reserve["reserve_source_keys"])
    random.Random(args.seed).shuffle(held_in)

    used: set[str] = set()
    used_from: dict[str, int] = {}
    for name, path in (
        ("stage_a", REPO / "diagnostics/pathwise_constraints_smoke_panel.json"),
        ("a2", REPO / "diagnostics/pathwise_a2_panel.json"),
        ("retarget", REPO / "diagnostics/retarget_calibration_cohort.json"),
    ):
        if path.exists():
            keys = {r["source"] for r in json.loads(path.read_text())["sources"]}
            used |= keys
            used_from[name] = len(keys)

    norms = json.loads(args.normalizers.read_text())["normalizers"]
    s_drd2 = norms["drd2_logodds"]["iqr"]
    oracle = load_default_oracle(str(args.oracle_manifest))
    low, high = clogp_corridor()

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
        if smiles in used:
            rejected["already_used_source"] = rejected.get("already_used_source", 0) + 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rejected["unparseable"] = rejected.get("unparseable", 0) + 1
            continue
        heavy = mol.GetNumHeavyAtoms()
        clogp = float(Crippen.MolLogP(mol))
        drd2 = float(oracle.margin_many([smiles])[0])
        potency_margin = (drd2 - potency_cut) / s_drd2

        reasons = eligibility(smiles, clogp=clogp, heavy=heavy,
                              potency_margin=potency_margin, low=low, high=high)
        if reasons:
            for reason in reasons:
                rejected[reason] = rejected.get(reason, 0) + 1
            continue

        chosen.append({
            "index": len(chosen),
            "source": smiles,
            "slots": SLOTS,
            "heavy_atoms": heavy,
            "clogp": round(clogp, 4),
            "clogp_headroom_low": round(clogp - low, 4),
            "clogp_headroom_high": round(high - clogp, 4),
            "drd2_logodds": round(drd2, 4),
            "potency_margin_at_source": round(potency_margin, 4),
        })

    if len(chosen) < args.sources:
        raise SystemExit(f"only {len(chosen)} eligible sources found")

    keys = {r["source"] for r in chosen}
    assert len(keys) == len(chosen), "duplicate sources"
    assert not (keys & used), "overlap with an earlier panel"
    assert not (keys & held_out), "held-out overlap"

    payload = {
        "schema": "compose.pathwise.stage_b_panel",
        "status": "DESIGN_ONLY",
        "held_out_opened": False,
        "pool": "held-in training_source_keys only",
        "seed": args.seed,
        "scanned": scanned,
        "rejected": rejected,
        "excluded_prior_panels": used_from,
        "objective": {
            "name": "DRD2 potency (goal P)",
            "provenance": ("frozen in the retargeting lane; chosen because it is "
                           "independently motivated and already frozen, NOT because "
                           "it maximises the pathwise effect"),
            "threshold": POTENCY_THRESHOLD,
        },
        "constraint": {
            "clogp_corridor": [low, high],
            "provenance": ("frozen held-in interquartile range, read at runtime; "
                           "UNCHANGED since the three-family census"),
        },
        "eligibility": {
            "heavy_atom_band": list(HEAVY_ATOM_BAND),
            "source_clogp_inside_corridor": True,
            "excludes_sources_already_potent": True,
            "excursion_history_used": False,
            "note": ("eligibility is EXCURSION-BLIND: selecting on excursion "
                     "propensity would make the primary estimand true by "
                     "construction"),
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
    print(f"excluded prior panels: {used_from} ({len(used)} keys)")
    print(f"selected {len(chosen)}; panel_sha256 {payload['panel_sha256'][:16]}")
    heads = [r["clogp_headroom_low"] for r in chosen]
    print(f"clogp headroom below corridor top: min {min(heads):.3f} "
          f"max {max(heads):.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
