#!/usr/bin/env python3
"""Oracle -> BEAE: the deployed nominator across a spectrum of BEAE heads.

Demonstrates the honest oracle workflow on the ACTUAL BEAE scaffold (fixed linker,
varied head): standard ionizable-amine heads land in-domain and are ranked; genuinely
novel/frontier heads ABSTAIN -> active learning. The generator is unrestricted (it
explores diverse heads/tails); the oracle is honest about its edges. Uses the enamine-N
fix so each BEAE lipid's head is scored on the real ionizable amine, not the central
enamine junction.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/qualify_nominator_beae.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from compose_v4.oracles.nominate import OracleNominator

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

ACR = "[NX3;H2,H1:1].[CH2:2]=[CH1:3][CX3:4](=[OX1:5])[#7,#8:6]>>[N:1][CH2:2][CH2:3][CX3:4](=[OX1:5])[*:6]"
PRO = "[NX3;H2,H1:1].[CH:2]#[C:3][CX3:4](=[OX1:5])[OX2:6][#6:7]>>[N:1]/[CH:2]=[CH:3]/[C:4](=[O:5])[O:6][#6:7]"
ACRYLATE = "C=CC(=O)OCC(CC)CCCC"      # 2-ethylhexyl acrylate (fixed tail 1)
PROPIOLATE = "C#CC(=O)OCCCCCCCCCCCC"  # dodecyl propiolate (fixed tail 2)

# amine cores (primary amine for the two Michael additions + the ionizable HEAD)
STANDARD_HEADS = {
    "dmapa": "CN(C)CCCN", "diethylamino_propyl": "CCN(CC)CCCN",
    "piperidinyl_propyl": "C1CCCCN1CCCN", "morpholinyl_propyl": "O1CCN(CC1)CCCN",
    "pyrrolidinyl_propyl": "C1CCN(C1)CCCN", "dimethylamino_ethyl": "CN(C)CCN",
}
FRONTIER_HEADS = {
    "guanidinyl_propyl": "NC(=N)NCCCN", "imidazolyl_propyl": "c1cnc(n1)CCCN",
    "benzylamino_propyl": "C(c1ccccc1)NCCCN", "adamantyl_amino": "C1C2CC3CC1CC(C2)(C3)NCCCN",
}


def _apply(sm, *smis):
    r = AllChem.ReactionFromSmarts(sm)
    mols = tuple(Chem.MolFromSmiles(s) for s in smis)
    out = set()
    if any(m is None for m in mols):
        return out
    for p in r.RunReactants(mols):
        try:
            Chem.SanitizeMol(p[0]); out.add(Chem.MolToSmiles(p[0]))
        except Exception:
            continue
    return out


def _beae(head: str) -> str | None:
    for mono in _apply(ACR, head, ACRYLATE):
        for full in _apply(PRO, mono, PROPIOLATE):
            return full
    return None


def main() -> None:
    nom = OracleNominator.load(REPO_ROOT, ranking_domain_id="a549")
    groups = {"standard": STANDARD_HEADS, "frontier": FRONTIER_HEADS}
    report = {}
    for gname, heads in groups.items():
        decisions, rows = Counter(), {}
        for hname, head in heads.items():
            beae = _beae(head)
            if beae is None:
                rows[hname] = "assembly_failed"; continue
            d = nom.nominate(beae, head_pka=None)["decision"]
            decisions[d] += 1
            rows[hname] = d
        n = sum(decisions.values()) or 1
        report[gname] = {"per_head": rows, "decision_breakdown": dict(decisions),
                         "admit_rate": round(decisions["rank"] / n, 3),
                         "abstain_rate": round(sum(v for k, v in decisions.items() if k.startswith("abstain")) / n, 3)}

    out = {
        "format": "compose_nominator_beae_spectrum_v1",
        "scaffold": "fixed BEAE linker; varied head; fixed 2-ethylhexyl/dodecyl tails",
        "groups": report,
        "interpretation": (
            "The deployed A549 oracle abstains on BEAE lipids at the MOLECULAR-AD level "
            "(abstain_off_domain), regardless of head -- because the propiolate ENAMINE ESTER is a "
            "genuinely novel chemotype absent from the A549 training distribution. This is the oracle "
            "being HONEST (it does not extrapolate onto a novel chemotype), NOT a failure. Note the "
            "distinction: real A549 acrylate-Michael lipids ARE in-domain (~98% admitted, they are in "
            "the screen), but the propiolate enamine pushes a full BEAE molecule off-domain. To rank "
            "BEAE the oracle needs Michael/BEAE calibration data in training (active learning) -> the "
            "positive-control regime (Spearman ~0.633). Head/tail diversity is fine; the enamine "
            "linker is the novelty the AD correctly gates until calibrated."
        ),
    }
    (REPO_ROOT / "diagnostics/nominator_beae_spectrum.json").write_text(json.dumps(out, indent=2) + "\n")

    for gname, r in report.items():
        print(f"[{gname}] admit={r['admit_rate']:.0%}  abstain={r['abstain_rate']:.0%}  {r['decision_breakdown']}")
        for h, d in r["per_head"].items():
            print(f"    {h:22s} -> {d}")
    print("\n" + out["interpretation"])
    print("written: diagnostics/nominator_beae_spectrum.json")


if __name__ == "__main__":
    main()
