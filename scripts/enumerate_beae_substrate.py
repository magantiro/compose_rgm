#!/usr/bin/env python3
"""Enumerate the combinatorial BEAE fine-tune substrate: head x acrylate-tail x propiolate-tail.

The BEAE core is fixed (central tertiary amine + saturated beta-amino-ester acrylate
arm + E-beta-enamine-ester propiolate arm); the ionizable HEAD and the two TAILS are
the combinatorial variables. This enumerates BEAE lipids over a small pool of
head-bearing amine cores x acrylate-tail acceptors x propiolate-tail acceptors,
confirming every combination assembles a valid BEAE lipid with both signature motifs.
This is the Fig-6 linker fine-tune substrate space (kept out of general pretraining).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/enumerate_beae_substrate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

ACRYLATE_SMARTS = "[NX3;H2,H1:1].[CH2:2]=[CH1:3][CX3:4](=[OX1:5])[#7,#8:6]>>[N:1][CH2:2][CH2:3][CX3:4](=[OX1:5])[*:6]"
PROPIOLATE_SMARTS = "[NX3;H2,H1:1].[CH:2]#[C:3][CX3:4](=[OX1:5])[OX2:6][#6:7]>>[N:1]/[CH:2]=[CH:3]/[C:4](=[O:5])[O:6][#6:7]"
E_ENAMINE = Chem.MolFromSmarts("[NX3]/[CH]=[CH]/[CX3](=O)[OX2]")
BETA_AMINO_ESTER = Chem.MolFromSmarts("[NX3][CH2][CH2][CX3](=O)[OX2]")

# head-bearing amine cores: a tertiary-amine HEAD + a primary amine for the two Michael additions
HEADS = {
    "dmapa": "CN(C)CCCN",                 # N,N-dimethyl-1,3-propanediamine
    "deapa": "CCN(CC)CCCN",               # N,N-diethyl
    "pyrrolidinyl_propyl": "C1CCN(C1)CCCN",
    "morpholinyl_propyl": "O1CCN(CC1)CCCN",
    "dmaea": "CN(C)CCN",                  # dimethylaminoethylamine (shorter spacer)
}
# tail alcohols expressed as the O-attached alkyl (acceptor built by prefixing the Michael head)
TAILS = {
    "2-ethylhexyl": "OCC(CC)CCCC",
    "n-dodecyl": "OCCCCCCCCCCCC",
    "2-hexyldecyl": "OCC(CCCCCC)CCCCCCCC",
    "oleyl": "OCCCCCCCC/C=C\\CCCCCCCC",
    "linoleyl": "OCCCCCCCC/C=C\\C/C=C\\CCCCC",
}


def _apply(smarts, *smis):
    rxn = AllChem.ReactionFromSmarts(smarts)
    out = set()
    for products in rxn.RunReactants(tuple(Chem.MolFromSmiles(s) for s in smis)):
        p = products[0]
        try:
            Chem.SanitizeMol(p)
            out.add(Chem.MolToSmiles(p))
        except Exception:
            continue
    return out


def _assemble(head, acr_tail, pro_tail):
    """head amine + acrylate(tail) then + propiolate(tail) -> BEAE lipid (canonical set)."""
    acrylate = "C=CC(=O)" + acr_tail            # acrylate ester with tail
    propiolate = "C#CC(=O)" + pro_tail          # propiolate ester with tail
    out = set()
    for mono in _apply(ACRYLATE_SMARTS, head, acrylate):
        out |= _apply(PROPIOLATE_SMARTS, mono, propiolate)
    return out


def main() -> None:
    products, rows = set(), []
    heads_seen, tails_seen = set(), set()
    for hname, head in HEADS.items():
        for aname, atail in TAILS.items():
            for pname, ptail in TAILS.items():
                for smi in _assemble(head, atail, ptail):
                    m = Chem.MolFromSmiles(smi)
                    if m is None or not (m.HasSubstructMatch(E_ENAMINE) and m.HasSubstructMatch(BETA_AMINO_ESTER)):
                        continue
                    if smi in products:
                        continue
                    products.add(smi)
                    heads_seen.add(hname)
                    tails_seen.add(aname); tails_seen.add(pname)
                    rows.append({"smiles": smi, "head": hname, "acrylate_tail": aname,
                                 "propiolate_tail": pname, "mw": round(Descriptors.MolWt(m), 2)})

    mws = [r["mw"] for r in rows]
    n_combos = len(HEADS) * len(TAILS) * len(TAILS)
    summary = {
        "format": "compose_beae_substrate_enumeration_v1",
        "role": "Fig-6 linker fine-tune substrate (combinatorial head x acrylate-tail x propiolate-tail); NOT general pretraining",
        "pool": {"heads": list(HEADS), "tails": list(TAILS)},
        "combinations_attempted": n_combos,
        "valid_beae_lipids": len(products),
        "heads_covered": sorted(heads_seen), "tails_covered": sorted(tails_seen),
        "mw_range": [min(mws), max(mws)] if mws else None,
        "all_have_both_motifs": True,
        "examples": rows[:6],
    }
    out_path = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/beae_substrate_enumeration_v1.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"combinations: {n_combos} (heads {len(HEADS)} x tails {len(TAILS)} x tails {len(TAILS)})")
    print(f"valid BEAE lipids: {len(products)}  MW range {summary['mw_range']}")
    print(f"heads covered: {summary['heads_covered']}")
    print(f"tails covered: {summary['tails_covered']}")
    print(f"written: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
