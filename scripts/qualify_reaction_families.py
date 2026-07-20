#!/usr/bin/env python3
"""Qualify a complementary batch of lipid reaction families.

Extends the corpus beyond Ugi-3CR with five architecturally distinct families so
the structural corpus spans multiple reaction chemistries (not one scaffold).
Each transform is a frozen atom-mapped reaction SMARTS verified by exact
reconstruction of a mechanism-determined product plus chemoselectivity-negative
rejection.

Validation levels are labeled honestly:
- ``experimental_library_reconstruction`` (Ugi-3CR only, in the separate
  qualified_reactions_v1.json): reproduces 1,200 released AGILE products.
- ``mechanism_verified`` (this batch): the atom-mapped transform reproduces the
  mechanistically-determined product of a documented reaction class and rejects
  incompatible substrates. Exact primary-source product examples must be attached
  at submission-time literature audit (recorded as a TODO per family).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/qualify_reaction_families.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]


def canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m else None


def run_products(smarts: str, reactants: tuple[str, ...]) -> set[str]:
    rxn = AllChem.ReactionFromSmarts(smarts)
    mols = [Chem.MolFromSmiles(s) for s in reactants]
    if any(m is None for m in mols):
        return set()
    out: set[str] = set()
    for product_set in rxn.RunReactants(tuple(mols)):
        for product in product_set:
            try:
                Chem.SanitizeMol(product)
            except Exception:
                continue
            if len(Chem.GetMolFrags(product)) == 1:
                c = canonical(Chem.MolToSmiles(product))
                if c:
                    out.add(c)
    return out


# Each family: SMARTS (reactant-template order == roles order), roles with
# handles + mapped atoms, mechanism-verified positive example, negatives, source.
FAMILIES = {
    "aza_michael_amine_acrylate": {
        "architecture": "degradable amino-ester lipidoid (aza-Michael)",
        "smarts": "[NX3;H2,H1:1].[CH2:2]=[CH1:3][CX3:4](=[OX1:5])[#7,#8:6]>>[N:1][CH2:2][CH2:3][CX3:4](=[OX1:5])[*:6]",
        "roles": [
            {"name": "amine_head", "handle": "[NX3;H2,H1]", "mapped": [1]},
            {"name": "alkyl_acrylate_or_acrylamide_tail", "handle": "[CH2]=[CH1][CX3](=[OX1])[#7,#8]", "mapped": [2, 3, 4]},
        ],
        "selectivity": "Amine N-H conjugate-adds across the acrylate/acrylamide beta-carbon giving a beta-amino ester/amide. Only alpha,beta-unsaturated carbonyls (terminal CH2=CH-C(=O)[O/N]) are acceptors; isolated alkenes and saturated esters are inert.",
        "positive": {"reactants": ["CCCCCCCCCCCCN", "C=CC(=O)OC"], "expected": "CCCCCCCCCCCCNCCC(=O)OC"},
        "negatives": [{"reactants": ["CCCCCCCCCCCCN", "CCCCC=CCCCC"], "reason": "isolated alkene is not a Michael acceptor"}],
        "source": {"kind": "doi", "identifier": "10.1002/anie.201203263", "locator": "Whitehead/Anderson degradable amino-ester lipidoid libraries (aza-Michael of amines + acrylates)"},
        "primary_example_todo": "attach an exact reported amine+acrylate substrate/product pair at submission-time literature audit",
    },
    "epoxide_opening_amine": {
        "architecture": "amino-alcohol / epoxide-derived lipidoid",
        "smarts": "[NX3;H2,H1:1].[CH2:2]1[CH1:3][OX2:4]1>>[N:1][CH2:2][CH1:3][OH1:4]",
        "roles": [
            {"name": "amine_head", "handle": "[NX3;H2,H1]", "mapped": [1]},
            {"name": "alkyl_epoxide_tail", "handle": "[CH2]1[CH1][OX2]1", "mapped": [2, 3, 4]},
        ],
        "selectivity": "Amine opens a terminal alkyl epoxide at the less-hindered CH2, giving a beta-amino alcohol. Reactive-site multiplicity of the amine controls tail count; alkanes and ethers are inert.",
        "positive": {"reactants": ["CCCCCCCCCCCCN", "CCCCCCCCC1CO1"], "expected": "CCCCCCCCCCCCNCC(O)CCCCCCCC"},
        "negatives": [{"reactants": ["CCCCCCCCCCCCN", "CCCCCCCCCCCC"], "reason": "alkane has no epoxide handle"}],
        "source": {"kind": "doi", "identifier": "10.1073/pnas.1322937111", "locator": "epoxide-derived ionizable lipidoid libraries (amine + alkyl epoxide ring-opening)"},
        "primary_example_todo": "attach an exact reported amine+epoxide substrate/product pair at submission-time literature audit",
    },
    "passerini_3cr": {
        "architecture": "asymmetric degradable alpha-acyloxy amide (Passerini)",
        "smarts": "[CX3:1](=[OX1:2])[OX2H1:3].[CX3H1:4]=[OX1:5].[C;-1,+0;X1:6]#[N;+1,+0;X2:7]>>[C:1](=[O:2])[O:3][CH1:4][C+0:6](=[O:5])[NH1+0:7]",
        "roles": [
            {"name": "carboxylic_acid_tail", "handle": "[CX3](=[OX1])[OX2H1]", "mapped": [1, 2, 3]},
            {"name": "aldehyde_body", "handle": "[CX3H1]=[OX1]", "mapped": [4, 5]},
            {"name": "isocyanide_tail", "handle": "[C;-1,+0;X1]#[N;+1,+0;X2]", "mapped": [6, 7]},
        ],
        "selectivity": "Carboxylic acid + aldehyde + terminal isocyanide condense to an alpha-acyloxy amide. Requires a free carboxylic acid (alcohols/esters inert), an aldehyde (ketones inert), and a terminal isocyanide carbon (nitriles inert).",
        "positive": {"reactants": ["CC(=O)O", "CCCCCCCC=O", "CCCCCCCCCCCC[N+]#[C-]"], "expected": "CCCCCCCC(OC(C)=O)C(=O)NCCCCCCCCCCCC"},
        "negatives": [{"reactants": ["CCO", "CCCCCCCC=O", "CCCCCCCCCCCC[N+]#[C-]"], "reason": "alcohol is not a carboxylic acid"}],
        "source": {"kind": "doi", "identifier": "10.1021/jacs.5b09084", "locator": "Passerini multicomponent biodegradable lipid libraries"},
        "primary_example_todo": "attach an exact reported acid+aldehyde+isocyanide substrate/product triple at submission-time literature audit",
    },
    "reductive_amination_amine_aldehyde": {
        "architecture": "reductive-amination amino lipid",
        "smarts": "[NX3;H2,H1:1].[CX3H1:2]=[OX1:3]>>[N:1][CH2:2]",
        "roles": [
            {"name": "amine_head", "handle": "[NX3;H2,H1]", "mapped": [1]},
            {"name": "aldehyde_tail", "handle": "[CX3H1]=[OX1]", "mapped": [2]},
        ],
        "selectivity": "Amine condenses with an aldehyde and is reduced to a new C-N bond (secondary/tertiary amine). Aldehydes only (ketones give the ketone-reductive-amination variant, excluded here); the carbonyl oxygen leaves as water.",
        "positive": {"reactants": ["CCCCCCCCCCCCN", "CCCCCCCCCCCC=O"], "expected": "CCCCCCCCCCCCNCCCCCCCCCCCC"},
        "negatives": [{"reactants": ["CCCCCCCCCCCCN", "CCCCC(=O)CCCC"], "reason": "ketone is not an aldehyde handle here"}],
        "source": {"kind": "doi", "identifier": "10.1038/nbt.3423", "locator": "reductive-amination lipidoid libraries"},
        "primary_example_todo": "attach an exact reported amine+aldehyde substrate/product pair at submission-time literature audit",
    },
    "thiol_michael_thioether": {
        "architecture": "redox-responsive thioether-linked lipid (thiol-Michael)",
        "smarts": "[SX2H1:1].[CH2:2]=[CH1:3][CX3:4](=[OX1:5])[#7,#8:6]>>[S:1][CH2:2][CH2:3][CX3:4](=[OX1:5])[*:6]",
        "roles": [
            {"name": "thiol_tail", "handle": "[SX2H1]", "mapped": [1]},
            {"name": "alkyl_acrylate_or_acrylamide_tail", "handle": "[CH2]=[CH1][CX3](=[OX1])[#7,#8]", "mapped": [2, 3, 4]},
        ],
        "selectivity": "A thiol conjugate-adds across an acrylate to give a thioether-linked ester (a redox/degradable motif). Only alpha,beta-unsaturated carbonyls are acceptors.",
        "positive": {"reactants": ["CCCCCCCCCCCCS", "C=CC(=O)OCCCCCC"], "expected": "CCCCCCCCCCCCSCCC(=O)OCCCCCC"},
        "negatives": [{"reactants": ["CCCCCCCCCCCCS", "CCCCC=CCCCC"], "reason": "isolated alkene is not a Michael acceptor"}],
        "source": {"kind": "doi", "identifier": "10.1016/j.bbamem.2011.05.014", "locator": "thioether/disulfide degradable linkers in cationic lipids (Whitehead review refs 61-63)"},
        "primary_example_todo": "attach an exact reported thiol+acrylate substrate/product pair at submission-time literature audit",
    },
    "disulfide_coupling": {
        "architecture": "bioreducible disulfide-linked lipid",
        "smarts": "[SX2H1:1].[SX2H1:2]>>[S:1][S:2]",
        "roles": [
            {"name": "thiol_tail", "handle": "[SX2H1]", "mapped": [1]},
            {"name": "thiol_head_or_tail", "handle": "[SX2H1]", "mapped": [2]},
        ],
        "selectivity": "Oxidative coupling of two thiols gives a bioreducible disulfide linker (cleaved in the reducing cytosol). Enumerates symmetric and asymmetric disulfides; a redox-labile degradable motif.",
        "positive": {"reactants": ["CCCCCCCCCCCCS", "CCCCCCS"], "expected": "CCCCCCSSCCCCCCCCCCCC"},
        "negatives": [{"reactants": ["CCCCCCCCCCCCS", "CCCCCCCC"], "reason": "alkane has no thiol handle"}],
        "source": {"kind": "doi", "identifier": "10.1016/j.bbamem.2011.05.014", "locator": "disulfide-spacer multivalent cationic lipids for gene delivery (Whitehead review refs 61-63)"},
        "primary_example_todo": "attach an exact reported thiol/thiol disulfide substrate/product pair at submission-time literature audit",
    },
    "carbamate_amine_chloroformate": {
        "architecture": "carbamate-linked ionizable lipid",
        "smarts": "[NX3;H2,H1:1].[Cl][CX3:2](=[OX1:3])[OX2:4]>>[N:1][C:2](=[O:3])[O:4]",
        "roles": [
            {"name": "amine_head", "handle": "[NX3;H2,H1]", "mapped": [1]},
            {"name": "chloroformate_tail", "handle": "[Cl][CX3](=[OX1])[OX2]", "mapped": [2, 3, 4]},
        ],
        "selectivity": "An N-H amine reacts with an alkyl chloroformate to give a carbamate linker (hydrolytically degradable). Requires an N-H amine and a chloroformate; tertiary amines and simple esters are inert.",
        "positive": {"reactants": ["CCCCCCCCCCCCN", "O=C(Cl)OCCCCCC"], "expected": "CCCCCCCCCCCCNC(=O)OCCCCCC"},
        "negatives": [{"reactants": ["CCN(CC)CC", "O=C(Cl)OCCCCCC"], "reason": "tertiary amine has no N-H handle"}],
        "source": {"kind": "doi", "identifier": "10.1038/s44222-026-00401-1", "locator": "carbamate among preferred degradable linkers (Whitehead & Arral 2026 review)"},
        "primary_example_todo": "attach an exact reported amine+chloroformate substrate/product pair at submission-time literature audit",
    },
    "urea_amine_isocyanate": {
        "architecture": "urea-linked ionizable lipid",
        "smarts": "[NX3;H2,H1:1].[NX2:2]=[CX2:3]=[OX1:4]>>[N:1][C:3](=[O:4])[N:2]",
        "roles": [
            {"name": "amine_head", "handle": "[NX3;H2,H1]", "mapped": [1]},
            {"name": "isocyanate_tail", "handle": "[NX2]=[CX2]=[OX1]", "mapped": [2, 3, 4]},
        ],
        "selectivity": "An N-H amine adds to an isocyanate to give a urea linker (a degradable/H-bonding motif in the Whitehead review linker palette). Requires an N-H amine and an isocyanate.",
        "positive": {"reactants": ["CCCCCCCCCCCCN", "O=C=NCCCCCCCC"], "expected": "CCCCCCCCCCCCNC(=O)NCCCCCCCC"},
        "negatives": [{"reactants": ["CCN(CC)CC", "O=C=NCCCCCCCC"], "reason": "tertiary amine has no N-H handle"}],
        "source": {"kind": "doi", "identifier": "10.1038/s44222-026-00401-1", "locator": "urea among degradable linkers (Whitehead & Arral 2026)"},
        "primary_example_todo": "attach an exact reported amine+isocyanate substrate/product pair at submission-time literature audit",
    },
    "acetal_aldehyde_diol": {
        "architecture": "acid-degradable acetal/ketal-linked lipid",
        "smarts": "[CX3H1:1]=[OX1:2].[OX2H1:3][CX4:4][CX4:5][OX2H1:6]>>[CH1:1]1[O:3][C:4][C:5][O:6]1",
        "roles": [
            {"name": "aldehyde_tail", "handle": "[CX3H1]=[OX1]", "mapped": [1, 2]},
            {"name": "diol_linker", "handle": "[OX2H1][CX4][CX4][OX2H1]", "mapped": [3, 4, 5, 6]},
        ],
        "selectivity": "An aldehyde condenses with a 1,2-diol to a 1,3-dioxolane (acid-labile acetal linker, a pH-degradable motif). Requires an aldehyde and a 1,2-diol.",
        "positive": {"reactants": ["CCCCCCCCCCCC=O", "OCC(O)CCCCCCCC"], "expected": "CCCCCCCCCCCC1OCC(CCCCCCCC)O1"},
        "negatives": [{"reactants": ["CCCCCCCCCCCC=O", "CCCCCCCCCCCCO"], "reason": "a mono-ol is not a 1,2-diol"}],
        "source": {"kind": "doi", "identifier": "10.1038/s44222-026-00401-1", "locator": "ketal/acetal among degradable linkers (Whitehead & Arral 2026)"},
        "primary_example_todo": "attach an exact reported aldehyde+diol acetal substrate/product pair at submission-time literature audit",
    },
    "iphos_amine_dioxaphospholane": {
        "architecture": "ionizable phospholipid (iPhos; adds phosphate/P)",
        "smarts": "[NX3;H2,H1:1].[CH2:2]1[CH2:3][OX2:4][PX4:5](=[OX1:6])[OX2:7]1>>[N:1][CH2:2][CH2:3][O:4][P:5](=[O:6])[O:7]",
        "roles": [
            {"name": "amine_head", "handle": "[NX3;H2,H1]", "mapped": [1]},
            {"name": "dioxaphospholane_tail", "handle": "[CH2]1[CH2][OX2][PX4](=[OX1])[OX2]1", "mapped": [2, 3, 4, 5, 6, 7]},
        ],
        "selectivity": "An amine opens a 2-alkoxy-1,3,2-dioxaphospholane-2-oxide to an ionizable alkyl-phosphate phospholipid (iPhos chemistry). Introduces the phosphate linker and the P element.",
        "positive": {"reactants": ["CN(C)CCN", "CCCCCCCCCCOP1(=O)OCCO1"], "expected": "CCCCCCCCCCOP(=O)(O)OCCNCCN(C)C"},
        "negatives": [{"reactants": ["CN(C)CCN", "CCCCCCCCCCCC"], "reason": "alkane has no dioxaphospholane handle"}],
        "source": {"kind": "doi", "identifier": "10.1038/s41563-020-00886-0", "locator": "iPhos ionizable phospholipids (Liu et al., Nat Mater 2021)"},
        "primary_example_todo": "attach an exact reported amine+dioxaphospholane substrate/product pair at submission-time literature audit",
    },
    "amide_coupling_acid_amine": {
        "architecture": "clinical-like amide-linked ionizable lipid",
        "smarts": "[CX3:1](=[OX1:2])[OX2H1:3].[NX3;H2,H1:4]>>[C:1](=[O:2])[N:4]",
        "roles": [
            {"name": "carboxylic_acid_tail", "handle": "[CX3](=[OX1])[OX2H1]", "mapped": [1, 2, 3]},
            {"name": "amine_head", "handle": "[NX3;H2,H1]", "mapped": [4]},
        ],
        "selectivity": "Carboxylic acid + amine condense to an amide (water lost). Requires a free acid and an N-H amine; esters and tertiary amines are inert.",
        "positive": {"reactants": ["CCCCCCCCCCCC(=O)O", "CCCCCCCCCCCCN"], "expected": "CCCCCCCCCCCCNC(=O)CCCCCCCCCCC"},
        "negatives": [{"reactants": ["CCCCCCCCCCCC", "CCCCCCCCCCCCN"], "reason": "no carboxylic acid handle"}],
        "source": {"kind": "doi", "identifier": "10.1038/s41467-024-55072-6", "locator": "amide/ester-linked rationally designed ionizable lipids"},
        "primary_example_todo": "attach an exact reported acid+amine substrate/product pair at submission-time literature audit",
    },
}


def verify_family(name: str, spec: dict) -> dict:
    smarts = spec["smarts"]
    pos = spec["positive"]
    prods = run_products(smarts, tuple(pos["reactants"]))
    target = canonical(pos["expected"])
    pos_ok = target in prods
    neg_results = []
    for neg in spec["negatives"]:
        np_ = run_products(smarts, tuple(neg["reactants"]))
        neg_results.append({**neg, "rejected": len(np_) == 0})
    return {
        "reaction_id": name,
        "architecture": spec["architecture"],
        "atom_mapped_reaction_smarts": smarts,
        "validation_level": "mechanism_verified",
        "positive_reconstruction": {"reactants": pos["reactants"], "expected": target,
                                    "reproduced_exactly": pos_ok, "enumerated_products": len(prods)},
        "negative_rejection": {"cases": neg_results, "all_rejected": all(n["rejected"] for n in neg_results)},
        "primary_example_todo": spec["primary_example_todo"],
    }


def build_registry_entry(name: str, spec: dict, audit_sha: str) -> dict:
    return {
        "reaction_id": name,
        "reaction_version": 1,
        "status": "qualified_for_enumeration",
        "architecture": spec["architecture"],
        "sources": [{"kind": spec["source"]["kind"], "identifier": spec["source"]["identifier"],
                     "locator": spec["source"]["locator"],
                     "notes": "Reaction-class primary source; mechanism-verified atom-mapped transform. Exact primary-source product example pending submission-time literature audit."}],
        "reactant_roles": [
            {"name": r["name"], "count": 1, "required_handle_smarts": r["handle"],
             "mapped_reactive_atoms": r["mapped"], "forbidden_smarts": [], "allowed_site_multiplicity": [1, 2]}
            for r in spec["roles"]
        ],
        "atom_mapped_reaction_smarts": spec["smarts"],
        "selectivity_policy": spec["selectivity"],
        "stereochemistry_policy": "Preserve substrate stereochemistry; new stereocenters left unspecified.",
        "protonation_and_salt_policy": "Register neutral full-molecule products under the frozen compose_lipid canonicalization contract.",
        "conditions": {"solvent": [], "temperature_c": None, "time_h": None, "catalyst_or_reagent": [], "reported_yield_range": None},
        "known_positive_examples": [{"reactants": spec["positive"]["reactants"],
                                     "expected": canonical(spec["positive"]["expected"]),
                                     "reason": "mechanism-verified atom-mapped reconstruction"}],
        "known_negative_examples": [{"reactants": n["reactants"], "expected": "", "reason": n["reason"]}
                                    for n in spec["negatives"]],
        "implementation": {"enumerator": "compose_v4.lipids.reaction_enumeration:ReactionEnumerator",
                           "test_manifest": "tests/test_reaction_families.py",
                           "artifact_hash": audit_sha},
    }


def write_json(path: Path, payload) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
                        default=REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/reaction_families_qualification.json")
    parser.add_argument("--registry", type=Path,
                        default=REPO_ROOT / "configs/lipid_reactions/qualified_reaction_families_v1.json")
    args = parser.parse_args()

    audits = [verify_family(name, spec) for name, spec in FAMILIES.items()]
    all_pass = all(a["positive_reconstruction"]["reproduced_exactly"] and a["negative_rejection"]["all_rejected"]
                   for a in audits)
    audit_doc = {"format": "compose_lipid_reaction_families_qualification_v1",
                 "families": audits, "all_qualified": all_pass}
    audit_sha = write_json(args.audit, audit_doc)

    registry = {"registry_version": "qualified_reaction_families_v1",
                "reactions": [build_registry_entry(name, spec, audit_sha) for name, spec in FAMILIES.items()]}
    write_json(args.registry, registry)

    for a in audits:
        pr = a["positive_reconstruction"]; nr = a["negative_rejection"]
        print(f"  {a['reaction_id']:36s} pos={'PASS' if pr['reproduced_exactly'] else 'FAIL'} "
              f"neg={'PASS' if nr['all_rejected'] else 'FAIL'}")
    print(f"\naudit: {args.audit.relative_to(REPO_ROOT)} (sha {audit_sha[:12]})")
    print(f"registry: {args.registry.relative_to(REPO_ROOT)}")
    print(f"ALL QUALIFIED: {all_pass}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
