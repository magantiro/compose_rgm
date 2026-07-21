#!/usr/bin/env python3
"""Qualify the BEAE fine-tune substrate: the propiolate aza-Michael + full assembly.

BEAE (the novel linker for the Fig-6 fine-tune, NOT the general pretraining corpus)
is built by a primary-amine core -- bearing an ionizable head -- doing two
aza-Michael additions on the same nitrogen: one onto a (branched/unsaturated-tail)
ACRYLATE -> the saturated beta-amino-ester arm (already a qualified family), and one
onto a (tail) PROPIOLATE -> the E-configured beta-enamine-ester arm (the novel motif).

This qualifies the propiolate transform (atom-mapped SMARTS, positive reconstruction
of measured leads RM-60 and Example-2, chemoselective negatives) and the full
two-step BEAE assembly from real components. Kept OUT of the general corpus; this is
the linker-fine-tune substrate spec.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/qualify_beae_substrate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

# amine N-H adds across a propiolate ester (HC#C-C(=O)-O-R) -> E-beta-enamine ester
PROPIOLATE_SMARTS = (
    "[NX3;H2,H1:1].[CH:2]#[C:3][CX3:4](=[OX1:5])[OX2:6][#6:7]"
    ">>[N:1]/[CH:2]=[CH:3]/[C:4](=[O:5])[O:6][#6:7]"
)
ACRYLATE_SMARTS = (
    "[NX3;H2,H1:1].[CH2:2]=[CH1:3][CX3:4](=[OX1:5])[#7,#8:6]"
    ">>[N:1][CH2:2][CH2:3][CX3:4](=[OX1:5])[*:6]"
)

# measured leads (from the user's figures) and their build components
LEADS = {
    "RM-60": {
        "smiles": "CN(C)CCCN(CCC(=O)OCC(CCCCCC)CCCCCCCC)/C=C/C(=O)OC(CCCCCCCC)CCCCCCCCCC",
        "head_amine": "CN(C)CCCN",                                    # DMAPA (primary amine + Me2N head)
        "acrylate": "C=CC(=O)OCC(CCCCCC)CCCCCCCC",                    # 2-hexyldecyl acrylate
        "propiolate": "C#CC(=O)OC(CCCCCCCC)CCCCCCCCCC",              # branched-tail propiolate
    },
    "Example-2": {
        "smiles": "CN(C)CCCN(CCC(=O)OCCCCCCCCC(C)C)/C=C/C(=O)OCCCCCCCC/C=C\\C/C=C\\CCCCC",
        "head_amine": "CN(C)CCCN",
        "acrylate": "C=CC(=O)OCCCCCCCCC(C)C",                         # iso-branched-tail acrylate
        "propiolate": "C#CC(=O)OCCCCCCCC/C=C\\C/C=C\\CCCCC",         # linoleyl propiolate
    },
}

NEGATIVES = [
    {"amine": "CCCCCCCCCCCCN", "acceptor": "C=CC(=O)OCC",
     "reason": "acrylate has C=C not C#C -> no propiolate (enamine) product"},
    {"amine": "CCCCCCCCCCCCN", "acceptor": "CC#CC(=O)OCC",
     "reason": "internal alkynoate has no terminal =CH -> not a propiolate handle"},
    {"amine": "CCCCCCCCCCCCN", "acceptor": "C#CCCCCCC",
     "reason": "terminal alkyne without a conjugated ester is not a Michael acceptor"},
    {"amine": "CCCCCCCCCCCCO", "acceptor": "C#CC(=O)OCC",
     "reason": "alcohol is not an amine N-H nucleophile"},
]


def _canon(s: str) -> str | None:
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def _apply(smarts: str, *reactant_smiles: str) -> set[str]:
    rxn = AllChem.ReactionFromSmarts(smarts)
    reactants = tuple(Chem.MolFromSmiles(s) for s in reactant_smiles)
    out: set[str] = set()
    for products in rxn.RunReactants(reactants):
        p = products[0]
        try:
            Chem.SanitizeMol(p)
            out.add(Chem.MolToSmiles(p))
        except Exception:
            continue
    return out


def main() -> None:
    results = {"format": "compose_beae_finetune_substrate_qualification_v1",
               "role": "Fig-6 linker fine-tune substrate (NOT general pretraining corpus)",
               "propiolate_smarts": PROPIOLATE_SMARTS, "acrylate_smarts": ACRYLATE_SMARTS,
               "leads": {}, "negatives": [], "e_enamine_smarts": "[NX3]/[CH]=[CH]/[CX3](=O)[OX2]"}
    e_enamine = Chem.MolFromSmarts(results["e_enamine_smarts"])

    all_ok = True
    for name, lead in LEADS.items():
        target = _canon(lead["smiles"])
        # two-step assembly: head amine + acrylate -> mono-adduct; + propiolate -> BEAE
        mono = _apply(ACRYLATE_SMARTS, lead["head_amine"], lead["acrylate"])
        full = set()
        for inter in mono:
            full |= _apply(PROPIOLATE_SMARTS, inter, lead["propiolate"])
        # also the reverse order (propiolate first) must reach the same scaffold
        mono_p = _apply(PROPIOLATE_SMARTS, lead["head_amine"], lead["propiolate"])
        full_rev = set()
        for inter in mono_p:
            full_rev |= _apply(ACRYLATE_SMARTS, inter, lead["acrylate"])
        reached = target in full
        reached_rev = target in full_rev
        e_ok = Chem.MolFromSmiles(target).HasSubstructMatch(e_enamine)
        ok = reached and e_ok
        all_ok &= ok
        results["leads"][name] = {
            "target": target, "reconstructed_forward": reached,
            "reconstructed_reverse_order": reached_rev, "has_E_enamine_ester": e_ok,
            "qualified": ok,
        }

    for neg in NEGATIVES:
        prods = _apply(PROPIOLATE_SMARTS, neg["amine"], neg["acceptor"])
        # rejected = no product OR no product bearing the E-enamine-ester motif
        has_enamine = any(Chem.MolFromSmiles(p).HasSubstructMatch(e_enamine) for p in prods)
        rejected = not has_enamine
        all_ok &= rejected
        results["negatives"].append({**neg, "n_products": len(prods), "rejected": rejected})

    results["all_qualified"] = bool(all_ok)
    out_path = REPO_ROOT / "configs/lipid_reactions/beae_finetune_substrate_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    for name, r in results["leads"].items():
        print(f"[{name}] reconstructed={r['reconstructed_forward']} (reverse={r['reconstructed_reverse_order']}) "
              f"E-enamine={r['has_E_enamine_ester']} -> qualified={r['qualified']}")
    for neg in results["negatives"]:
        print(f"[negative] {neg['acceptor']:16s} rejected={neg['rejected']}  ({neg['reason']})")
    print(f"\nALL QUALIFIED: {results['all_qualified']}")
    print(f"written: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
