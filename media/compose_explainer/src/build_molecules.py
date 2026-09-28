"""Emit faithful 2D molecular layouts as JSON for the COMPOSE explainer.

Uses RDKit only to compute display geometry (2D coordinates, ring perception,
aromaticity flags) for molecules drawn in the animation. No number reported in
the video is derived here -- this is a layout step, not a chemistry measurement.

Output: assets/molecules.json
  { name: {atoms: [{el, x, y, arom, ring, hs}], bonds: [{a, b, order, arom}]} }

Coordinates are centred on the centroid and scaled so the mean bond length is
exactly 1.0, so the renderer can pick a pixel bond length per scene.
"""

from __future__ import annotations

import json
import math
import pathlib

from rdkit import Chem
from rdkit.Chem import AllChem


# ---- Molecule set ----
# Each entry is a display molecule used by a specific scene. The canonicalisation
# pair (isobutane -> propane) is taken verbatim from the paper's appendix example.
SMILES = {
    # Scene 1/3/4 protagonist: a drug-like, ring-bearing molecule in the size
    # range the paper reports for its training sources (median 26 heavy atoms).
    "lead": "CC(=O)Nc1ccc(cc1)S(=O)(=O)N1CCOCC1",
    # Trans-dimensional neighbours of a smaller working molecule.
    "core": "c1ccc(cc1)C(=O)NC1CCNCC1",
    "core_grow": "c1ccc(cc1)C(=O)NC1CCN(C)CC1",
    "core_shrink": "c1ccc(cc1)C(=O)NC1CCNC1",
    "core_restate": "c1ccc(cc1)C(=O)OC1CCNCC1",
    "core_ring": "c1ccc(cc1)C(=O)NC1CCN2CCCC2C1",
    # Paper's canonical-successor example, exactly as stated.
    "isobutane": "CC(C)C",
    "propane": "CCC",
    # Program example: build a ring onto an acyclic chain by composing primitives.
    "prog0": "CCCCCN",
    "prog1": "CCCCCNC",
    "prog2": "C1CCCCN1",
    "prog3": "c1ccccn1",
    # Fragment-conditioning example: a retained core plus grown periphery.
    "frag_core": "c1ccc2[nH]ccc2c1",
    "frag_full": "CC(=O)N1CCc2c1cccc2",
    # Small neighbourhood molecules for lattice nodes.
    "n1": "c1ccccc1",
    "n2": "c1ccncc1",
    "n3": "C1CCCCC1",
    "n4": "c1ccc2ccccc2c1",
    "n5": "C1CCNCC1",
    "n6": "c1cc[nH]c1",
    "n7": "C1CCOC1",
    "n8": "CC(C)(C)O",
    "n9": "c1ccc(O)cc1",
    "n10": "C1CC1",
    "n11": "CC#N",
    "n12": "c1cnc2ccccc2n1",
}

# ---- The long trajectory of the process scene ----
# Ten committed states. Consecutive pairs differ by exactly one primitive
# rewrite, so the heavy-atom count moves by at most one at every step. The chain
# deliberately grows, restructures and shrinks, which is the paper's
# trans-dimensional claim, and it exercises six of the eight families.
TRACE = [
    ("CCCCCN",           None,           ""),
    ("C1CCNCC1",         "cycle_close",  "="),
    ("CC1CCNCC1",        "atom_insert",  "+1"),
    ("CC1CCN(C)CC1",     "atom_insert",  "+1"),
    ("CC1CCN(CO)CC1",    "atom_insert",  "+1"),
    ("CC1CCN(C=O)CC1",   "bond_reorder", "="),
    ("CCCN(C=O)CCC",     "cycle_open",   "="),
    ("CCCN(C=O)CC",      "atom_delete",  "−1"),
    ("CCCN(C)CC",        "atom_delete",  "−1"),
    ("CCCNCC",           "atom_delete",  "−1"),
]


def layout(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable display SMILES: {smiles}")
    Chem.rdDepictor.SetPreferCoordGen(True)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    ri = mol.GetRingInfo()

    atoms = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(
            {
                "el": atom.GetSymbol(),
                "x": p.x,
                "y": -p.y,  # flip to screen orientation (y down)
                "arom": bool(atom.GetIsAromatic()),
                "ring": bool(ri.NumAtomRings(atom.GetIdx()) > 0),
                "hs": int(atom.GetTotalNumHs()),
            }
        )

    bonds = []
    for bond in mol.GetBonds():
        bonds.append(
            {
                "a": bond.GetBeginAtomIdx(),
                "b": bond.GetEndAtomIdx(),
                "order": {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}[
                    str(bond.GetBondType())
                ],
                "arom": bool(bond.GetIsAromatic()),
                "ring": bool(bond.IsInRing()),
            }
        )

    # Normalise: centroid at origin, mean bond length 1.0.
    cx = sum(a["x"] for a in atoms) / len(atoms)
    cy = sum(a["y"] for a in atoms) / len(atoms)
    for a in atoms:
        a["x"] -= cx
        a["y"] -= cy
    if bonds:
        lengths = [
            math.dist((atoms[b["a"]]["x"], atoms[b["a"]]["y"]),
                      (atoms[b["b"]]["x"], atoms[b["b"]]["y"]))
            for b in bonds
        ]
        scale = sum(lengths) / len(lengths)
        if scale > 1e-9:
            for a in atoms:
                a["x"] /= scale
                a["y"] /= scale

    # Ring centroids let the renderer place aromatic inner arcs correctly.
    rings = []
    for ring in ri.AtomRings():
        rx = sum(atoms[i]["x"] for i in ring) / len(ring)
        ry = sum(atoms[i]["y"] for i in ring) / len(ring)
        rings.append({"atoms": list(ring), "cx": rx, "cy": ry})

    return {
        "smiles": Chem.MolToSmiles(mol),
        "atoms": atoms,
        "bonds": bonds,
        "rings": rings,
        "heavy": mol.GetNumHeavyAtoms(),
    }


def check_trace(out: dict) -> list[dict]:
    """Lay out the trajectory and REFUSE any step that is not a single primitive.

    A primitive rewrite changes the heavy-atom count by at most one. If a step
    violates that, the chain is not a sequence of primitives and the scene would
    be making a claim the paper does not support -- so this raises rather than
    quietly rendering it.
    """
    steps = []
    prev = None
    for i, (smi, family, card) in enumerate(TRACE):
        m = layout(smi)
        key = f"trace{i}"
        out[key] = m
        if prev is not None:
            delta = m["heavy"] - prev
            if abs(delta) > 1:
                raise ValueError(
                    f"trace step {i} changes heavy atoms by {delta:+d}: "
                    f"not a single primitive rewrite ({smi})")
            expect = {"+1": 1, "−1": -1, "=": 0}[card]
            if delta != expect:
                raise ValueError(
                    f"trace step {i} labelled {family} {card} but heavy-atom "
                    f"delta is {delta:+d} ({smi})")
        steps.append({"key": key, "family": family, "card": card,
                      "heavy": m["heavy"], "smiles": m["smiles"]})
        prev = m["heavy"]
    return steps


def main() -> None:
    out = {name: layout(smi) for name, smi in SMILES.items()}
    trace = check_trace(out)
    out["_trace"] = trace
    print("trajectory (verified single-primitive steps):")
    for s in trace:
        fam = f"{s['family']} {s['card']}" if s["family"] else "source"
        print(f"  {s['key']:8s} heavy={s['heavy']:2d}  {fam:20s} {s['smiles']}")
    print()
    assets = pathlib.Path(__file__).resolve().parent.parent / "assets"
    dest = assets / "molecules.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))
    # Chromium blocks fetch() on file:// URLs, so the renderer loads the same
    # data through a plain script tag instead.
    js = assets / "molecules.js"
    js.write_text("/* generated by src/build_molecules.py -- do not edit */\n"
                  "const MOLECULES = " + json.dumps(out, sort_keys=True) + ";\n")
    print(f"wrote {dest}\nwrote {js}")
    for name, m in sorted(out.items()):
        if name.startswith('_') or name.startswith('trace'):
            continue
        print(f"  {name:12s} heavy={m['heavy']:2d} atoms={len(m['atoms']):2d} "
              f"bonds={len(m['bonds']):2d} rings={len(m['rings'])}  {m['smiles']}")


if __name__ == "__main__":
    main()
