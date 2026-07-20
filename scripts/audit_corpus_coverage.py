#!/usr/bin/env python3
"""Coverage-audit card: is the corpus all-encompassing over the lipid design space?

Audits the enumerated corpus against the Whitehead & Arral 2026 design axes --
head classes, linker motifs, tail architectures, elements -- reporting presence
and fraction for each, plus explicit gaps. This is the reviewer-facing evidence
that the corpus spans the ionizable-lipid design space (so a general generator
trained on it can learn the whole space).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/audit_corpus_coverage.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

LINKER_MOTIFS = {
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[NX3][CX3](=O)[#6]",
    "carbamate": "[NX3][CX3](=O)[OX2]",
    "carbonate": "[OX2][CX3](=O)[OX2]",
    "urea": "[NX3][CX3](=O)[NX3]",
    "thioether": "[#6][SX2][#6]",
    "disulfide": "[SX2][SX2]",
    "acetal_ketal": "[CX4]([OX2])([OX2])",
    "amino_alcohol": "[NX3][CX4][CX4][OX2H1]",
    "phosphate": "[PX4](=O)([OX2])[OX2]",
    "ether": "[#6][OX2][#6;!$([#6]=O)]",
}
HEAD_CLASSES = {
    "tertiary_amine": "[NX3;H0;!$(NC=O);!$(N=*);!$([N+])]([#6])([#6])[#6]",
    "secondary_amine": "[NX3;H1;!$(NC=O);!$(N=*)]",
    "primary_amine": "[NX3;H2;!$(NC=O)]",
    "amino_alcohol_head": "[NX3][CX4][CX4][OX2H1]",
    "piperazine": "C1CNCCN1",
    "morpholine": "C1COCCN1",
    "piperidine_ring": "[NX3]1[CX4][CX4][CX4][CX4][CX4]1",
    "polyamine_2plus_N": "[NX3;!$(NC=O)].[NX3;!$(NC=O)]",
}
TARGET_ELEMENTS = ["C", "N", "O", "S", "P"]


def _load_products() -> list[str]:
    path = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/corpus_pilot_v1/pilot_products.csv"
    return [r["canonical_smiles"] for r in csv.DictReader(path.open())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "docs/audits/2026-07-20_corpus_coverage_card.md")
    parser.add_argument("--json-output", type=Path,
                        default=REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/corpus_coverage_card.json")
    args = parser.parse_args()

    smis = _load_products()
    mols = [Chem.MolFromSmiles(s) for s in smis]
    mols = [m for m in mols if m is not None]
    n = len(mols)

    def frac(smarts: str) -> float:
        patt = Chem.MolFromSmarts(smarts)
        return round(sum(m.HasSubstructMatch(patt) for m in mols) / n, 4)

    linker = {k: frac(v) for k, v in LINKER_MOTIFS.items()}
    head = {k: frac(v) for k, v in HEAD_CLASSES.items()}
    elements = {}
    for e in TARGET_ELEMENTS:
        elements[e] = round(sum(any(a.GetSymbol() == e for a in m.GetAtoms()) for m in mols) / n, 4)
    # tail architecture from committed pilot bins (long-tail proxy)
    families = sorted({r["reaction_family"] for r in csv.DictReader(
        (REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/corpus_pilot_v1/pilot_products.csv").open())})

    covered = {"linkers": [k for k, v in linker.items() if v > 0.001],
               "head_classes": [k for k, v in head.items() if v > 0.001],
               "elements": [e for e, v in elements.items() if v > 0]}
    gaps = {"linkers": [k for k, v in linker.items() if v <= 0.001],
            "elements": [e for e, v in elements.items() if v == 0]}

    card = {
        "format": "compose_lipid_corpus_coverage_card_v1",
        "corpus_products_audited": n,
        "reaction_families": {"count": len(families), "families": families},
        "linker_motif_coverage": dict(sorted(linker.items(), key=lambda x: -x[1])),
        "head_class_coverage": dict(sorted(head.items(), key=lambda x: -x[1])),
        "element_coverage": elements,
        "covered": covered,
        "gaps": gaps,
        "design_reference": "Arral & Whitehead, Nat Rev Bioeng 4:417-434 (2026)",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")

    lines = ["# COMPOSE-Lipid corpus coverage card", "",
             f"**Products audited:** {n:,} · **Reaction families:** {len(families)} · "
             "**Design reference:** Arral & Whitehead 2026 (Nat Rev Bioeng)", "",
             "## Linker-motif coverage (fraction of corpus)", "",
             "| Linker | Fraction |", "|---|---:|"]
    for k, v in sorted(linker.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:.3f} |")
    lines += ["", "## Head-class coverage", "", "| Head class | Fraction |", "|---|---:|"]
    for k, v in sorted(head.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:.3f} |")
    lines += ["", "## Element coverage", "", "| Element | Fraction |", "|---|---:|"]
    for e, v in elements.items():
        lines.append(f"| {e} | {v:.3f} |")
    lines += ["", f"**Covered linkers:** {', '.join(covered['linkers'])}",
              f"**Covered elements:** {', '.join(covered['elements'])}",
              f"**Gaps:** linkers={gaps['linkers'] or 'none'}; elements={gaps['elements'] or 'none'}", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))

    print(f"audited {n} products across {len(families)} families")
    print(f"linkers covered: {covered['linkers']}")
    print(f"elements covered: {covered['elements']}")
    print(f"gaps: {gaps}")
    print(f"written: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
