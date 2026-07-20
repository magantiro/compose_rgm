#!/usr/bin/env python3
"""Build a multi-source, role-annotated building-block pool for the lipid corpus.

Diversity multiplier for the general (linker-agnostic) structural corpus. Blocks
come from three provenance classes:
  - curated_literature: documented ionizable amine heads (polyamines, amino-
    alcohols, cyclic amines) used in lipidoid libraries;
  - programmatic_rational: fatty reactive substrates (acid / aldehyde / epoxide /
    acrylate / isocyanide) generated over chain length x unsaturation x branching
    -- the rational-design feasibility envelope;
  - agile_measured: the real A/B/C components from the AGILE library.

Each block is validated to parse, to carry exactly one intended reactive handle,
and is mapped to the qualified reaction roles it can fill. Blocks are NOT the
head/linker/tail fragments of finished lipids; they are reactive substrates.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/build_building_block_pool.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- reactive-handle definitions (SMARTS) mapped to the roles they can fill ----
HANDLES = {
    "amine": ("[NX3;H1,H2;!$(NC=O);!$(N=*)]", [
        "amine_head",  # Ugi, aza-Michael, epoxide, reductive amination, amide
    ]),
    "carboxylic_acid": ("[CX3](=[OX1])[OX2H1]", ["carboxylic_acid_tail"]),
    "aldehyde": ("[CX3H1]=[OX1]", ["aldehyde_body", "aldehyde_tail", "oxoester_aldehyde_body_tail"]),
    "epoxide": ("[CH2]1[CH1][OX2]1", ["alkyl_epoxide_tail"]),
    "acrylate": ("[CH2]=[CH1][CX3](=[OX1])[#7,#8]", ["alkyl_acrylate_or_acrylamide_tail"]),
    "isocyanide": ("[C;-1,+0;X1]#[N;+1,+0;X2]", ["isocyanide_tail", "isocyanide_tail"]),
    # degradable-linker chemistry (Whitehead & Arral 2026 design principles): adds S
    "thiol": ("[SX2H1]", ["thiol_tail", "thiol_head_or_tail"]),
    "chloroformate": ("[Cl][CX3](=[OX1])[OX2]", ["chloroformate_tail"]),
    # full linker palette: urea, acetal, phosphate (adds P)
    "isocyanate": ("[NX2]=[CX2]=[OX1]", ["isocyanate_tail"]),
    "diol": ("[OX2H1][CX4][CX4][OX2H1]", ["diol_linker"]),
    "dioxaphospholane": ("[CH2]1[CH2][OX2][PX4](=[OX1])[OX2]1", ["dioxaphospholane_tail"]),
}

# ---- curated ionizable amine heads (documented lipidoid chemistry) ----
CURATED_AMINES = {
    # simple alkyl amines
    "methylamine": "CN", "ethylamine": "CCN", "propylamine": "CCCN", "butylamine": "CCCCN",
    "hexylamine": "CCCCCCN", "octylamine": "CCCCCCCCN", "dodecylamine": "CCCCCCCCCCCCN",
    "dimethylamine": "CNC", "diethylamine": "CCNCC", "N-methylbutylamine": "CCCCNC",
    # amino-alcohols
    "ethanolamine": "OCCN", "diethanolamine": "OCCNCCO", "dmae": "OCCN(C)C",
    "dmap_ol": "OCCCN(C)C", "n-methyldiethanolamine": "OCCN(C)CCO", "aminopropanediol": "OCC(O)CN",
    # di/poly-amines
    "ethylenediamine": "NCCN", "13-diaminopropane": "NCCCN", "dmapa": "CN(C)CCN",
    "nn-dimethylethylenediamine": "CN(C)CCN", "deta": "NCCNCCN", "teta": "NCCNCCNCCN",
    "spermidine": "NCCCNCCCCN", "spermine": "NCCCNCCCCNCCCN", "peha": "NCCNCCNCCNCCNCCN",
    "tren": "NCCN(CCN)CCN", "bis-hexamethylenetriamine": "NCCCCCCNCCCCCCN",
    # cyclic amines
    "piperazine": "C1CNCCN1", "n-methylpiperazine": "CN1CCNCC1", "morpholine": "C1COCCN1",
    "piperidine": "C1CCNCC1", "4-aminopiperidine": "NC1CCNCC1", "pyrrolidine": "C1CCNC1",
    "1-2-aminoethylpiperazine": "NCCN1CCNCC1", "homopiperazine": "C1CNCCNC1",
    "2-piperidin-1-ylethanamine": "NCCN1CCCCC1", "4-2-aminoethylmorpholine": "NCCN1CCOCC1",
    "1-2-hydroxyethylpiperazine": "OCCN1CCNCC1", "2-morpholinoethanamine": "NCCN1CCOCC1",
    # branched / hindered
    "isopropylamine": "CC(C)N", "tert-butylamine": "CC(C)(C)N", "2-ethylhexylamine": "CCCCC(CC)CN",
    "neopentylamine": "CC(C)(C)CN", "cyclohexylamine": "NC1CCCCC1",
    # functionalized
    "histamine": "NCCc1cnc[nH]1", "3-morpholinopropylamine": "NCCCN1CCOCC1",
    "n-3-aminopropylmorpholine": "NCCCN1CCOCC1", "furfurylamine": "NCc1ccco1",
    "benzylamine": "NCc1ccccc1", "phenethylamine": "NCCc1ccccc1",
    # extended polyamine / lipidoid-core set (documented amine cores)
    "14-diaminobutane": "NCCCCN", "15-diaminopentane": "NCCCCCN",
    "16-diaminohexane": "NCCCCCCN", "3-3-diaminodipropylamine": "NCCCNCCCN",
    "bis-3-aminopropylethylenediamine": "NCCCNCCNCCCN", "n-methyl-13-propanediamine": "CNCCCN",
    "2-methyl-12-propanediamine": "CC(N)CN", "12-diaminopropane": "CC(N)CN",
    "1-4-aminobutylpiperazine": "NCCCCN1CCNCC1", "1-3-aminopropylpiperazine": "NCCCN1CCNCC1",
    "4-3-aminopropylmorpholine": "NCCCN1CCOCC1", "trans-14-diaminocyclohexane": "NC1CCC(N)CC1",
    "2-2-aminoethylaminoethanol": "OCCNCCN", "3-amino-1-propanol": "NCCCO",
    "2-amino-2-methylpropanol": "CC(C)(N)CO", "diglycolamine": "NCCOCCO",
    "n-n-dimethyldipropylenetriamine": "CN(C)CCCNCCCN", "bishexamethylenetriamine": "NCCCCCCNCCCCCCN",
    "1-2-aminoethylpyrrolidine": "NCCN1CCCC1", "2-piperazin-1-ylethanamine": "NCCN1CCNCC1",
    "n-boc-ethylenediamine-free": "NCCNC", "aminoethylethanolamine": "NCCNCCO",
    "34-diaminobenzene-free": "Nc1ccccc1N", "13-diaminopropan-2-ol": "NCC(O)CN",
    "tris-2-aminoethylamine": "NCCN(CCN)CCN", "n1-2-aminoethyl-13-propanediamine": "NCCNCCCN",
    "1-aminomethylcyclohexylamine": "NCC1(N)CCCCC1", "2-2-aminoethoxyethanamine": "NCCOCCN",
    "dodecane-112-diamine": "NCCCCCCCCCCCCN", "octane-18-diamine": "NCCCCCCCCN",
    "n-oleyl-13-propanediamine": "CCCCCCCC/C=C\\CCCCCCCCNCCCN", "n-dodecyl-13-propanediamine": "CCCCCCCCCCCCNCCCN",
    # disulfide/thioether-bearing amine cores -> bioreducible degradable heads (adds S)
    "cystamine": "NCCSSCCN", "aminoethyl-disulfide-ethanol": "NCCSSCCO",
    "2-aminoethanethiol": "NCCS", "3-aminopropane-1-thiol": "NCCCS",
    "thiodiethylamine": "NCCSCCN", "bis-2-aminoethyl-disulfide": "NCCSSCCN",
}


def canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m else None


def alkyl_chain(length: int, unsaturation: int, n_branches: int) -> str:
    """Return an alkyl-chain SMILES of `length` carbons with a controlled number
    of methyl branches OR cis double bonds (varied separately to keep valid,
    interpretable structures across the branchedness and unsaturation axes)."""
    # branched saturated chain: insert n methyl branches at spread interior sites
    if n_branches >= 1 and unsaturation == 0 and length >= 6:
        toks = ["C"] * length
        step = max(2, length // (n_branches + 1))
        for i in range(1, n_branches + 1):
            pos = min(length - 2, i * step)
            toks[pos] = "C(C)"  # methyl branch
        return "".join(toks)
    # unsaturated linear chain (methylene-interrupted cis double bonds mid-chain)
    if unsaturation >= 2 and length >= 14:
        half = length // 2
        return "C" * (half - 3) + "/C=C\\C/C=C\\" + "C" * (length - half - 3)
    if unsaturation >= 1 and length >= 10:
        half = length // 2
        return "C" * (half - 1) + "/C=C\\" + "C" * (length - half - 1)
    return "C" * length


def make_substrate(form: str, chain: str) -> str | None:
    """Attach a reactive group `form` to an alkyl `chain`."""
    if form == "carboxylic_acid":
        s = chain + "C(=O)O"
    elif form == "aldehyde":
        s = chain + "C=O"
    elif form == "epoxide":
        s = chain + "C1CO1"
    elif form == "acrylate":
        s = "C=CC(=O)OC" + chain  # alkyl acrylate ester
    elif form == "isocyanide":
        s = chain + "[N+]#[C-]"
    elif form == "thiol":
        s = chain + "S"
    elif form == "chloroformate":
        s = "ClC(=O)O" + chain
    elif form == "isocyanate":
        s = chain + "N=C=O"
    elif form == "diol":
        s = "OCC(O)" + chain          # 3-alkyl-1,2-propanediol (glycerol-like 1,2-diol)
    elif form == "dioxaphospholane":
        s = chain + "OP1(=O)OCCO1"    # 2-alkoxy-1,3,2-dioxaphospholane-2-oxide
    else:
        return None
    return canonical(s)


def load_agile_components() -> dict[str, list[tuple[str, str]]]:
    manifest = json.loads(
        (REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/source_manifest.json").read_text())
    path = Path(next(s["local_artifact_path"] for s in manifest["sources"]
                     if s["source_id"] == "agile_measured1200"))
    rows = list(csv.DictReader(path.open()))
    out = {"A": set(), "B": set(), "C": set()}
    for r in rows:
        for col, key in [("A_smiles", "A"), ("B_smiles", "B"), ("C_smiles", "C")]:
            c = canonical(r[col])
            if c:
                out[key].add(c)
    return {k: sorted(v) for k, v in out.items()}


def classify_handles(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    hits = []
    for name, (smarts, _roles) in HANDLES.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            hits.append(name)
    return hits


def roles_for(handle: str) -> list[str]:
    return sorted(set(HANDLES[handle][1]))


def build_pool() -> dict:
    blocks: list[dict] = []
    seen: set[str] = set()

    def add(smiles, handle_intended, provenance, descriptors):
        c = canonical(smiles)
        if c is None or c in seen:
            return
        present = classify_handles(c)
        if handle_intended not in present:
            return  # must carry the intended handle
        seen.add(c)
        blocks.append({
            "block_id": f"{handle_intended}:{len([b for b in blocks if b['handle']==handle_intended]):04d}",
            "canonical_smiles": c, "handle": handle_intended,
            "reaction_roles": roles_for(handle_intended),
            "provenance": provenance, "descriptors": descriptors,
            "heavy_atoms": Chem.MolFromSmiles(c).GetNumHeavyAtoms(),
        })

    # curated amine heads
    for name, smi in CURATED_AMINES.items():
        add(smi, "amine", {"source": "curated_literature", "name": name}, {})

    # programmatic fatty substrates
    lengths = [6, 8, 10, 12, 14, 16, 18, 20, 22]
    for form in ["carboxylic_acid", "aldehyde", "epoxide", "acrylate", "isocyanide", "thiol",
                 "chloroformate", "isocyanate", "diol", "dioxaphospholane"]:
        for L in lengths:
            for unsat in [0, 1, 2]:
                for n_branches in [0, 1, 2]:
                    if unsat > 0 and n_branches > 0:
                        continue  # varied separately (alkyl_chain doesn't combine them)
                    if unsat > 0 and L < 10:
                        continue
                    if unsat == 2 and L < 14:
                        continue
                    if n_branches > 0 and L < 6:
                        continue
                    chain = alkyl_chain(L, unsat, n_branches)
                    sub = make_substrate(form, chain)
                    if sub:
                        add(sub, form, {"source": "programmatic_rational"},
                            {"chain_length": L, "unsaturation": unsat, "n_branches": n_branches})

    # AGILE real components
    agile = load_agile_components()
    for key, comps in agile.items():
        for smi in comps:
            for h in classify_handles(smi):
                add(smi, h, {"source": "agile_measured", "agile_component": key}, {})

    by_handle: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for b in blocks:
        by_handle[b["handle"]] = by_handle.get(b["handle"], 0) + 1
        by_source[b["provenance"]["source"]] = by_source.get(b["provenance"]["source"], 0) + 1

    return {
        "format": "compose_lipid_building_block_pool_v1",
        "note": "Reactive substrates (not finished-lipid fragments) for the general linker-agnostic corpus.",
        "total_blocks": len(blocks),
        "blocks_by_handle": dict(sorted(by_handle.items())),
        "blocks_by_source": dict(sorted(by_source.items())),
        "reaction_role_coverage": {
            role: sum(1 for b in blocks if role in b["reaction_roles"])
            for role in sorted({r for b in blocks for r in b["reaction_roles"]})
        },
        "blocks": sorted(blocks, key=lambda b: (b["handle"], b["canonical_smiles"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "configs/lipid_reactions/building_block_pool_v1.json")
    args = parser.parse_args()
    pool = build_pool()
    text = json.dumps(pool, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    print(f"total building blocks: {pool['total_blocks']}")
    print(f"by handle: {pool['blocks_by_handle']}")
    print(f"by source: {pool['blocks_by_source']}")
    print(f"role coverage: {pool['reaction_role_coverage']}")
    print(f"written: {args.output.relative_to(REPO_ROOT)} (sha {sha[:12]})")


if __name__ == "__main__":
    main()
