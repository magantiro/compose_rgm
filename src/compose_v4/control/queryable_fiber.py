"""The set of molecules an expensive oracle call may be spent on.

COMPOSE can decide feasibility exactly and cheaply, so docking should never be asked
whether a candidate is usable -- only how well it binds. That makes the admissible set an
explicit object:

    Q(G) = { T(G,O) : O legal, and the result is connected, size-bounded,
                      QED-valid, similarity-valid, and chemically stable }

The stability predicate is not cosmetic and was added because of a measured failure. In
the first 32-call JAK2 experiment the best-docking candidate was
`CC(=O)ONOC(=O)CC1Nc2ccccc2-...`, an acyl-nitroso / mixed anhydride. It reproduced at
-10.6, -10.5 and -10.4 across three seeds, so the score was real and the molecule was
not: QED 0.62 and a clean similarity did nothing to exclude a species that would not
survive contact with water. The clean runner-up scored -9.80.

QED, SA and similarity are drug-likeness, accessibility and novelty filters. None of them
is a stability filter, and an extreme-value objective will find that gap every time,
because a reactive electrophile is exactly the kind of thing a docking function likes --
the acyl-nitroso above scores SA 3.85 and passes the accessibility gate cleanly.

The SA gate itself was omitted from the first JAK2 experiment, which is worse. Only 11 of
its 28 docked candidates pass `SA < 4`, and NONE of its six control candidates do, so the
controller-against-control comparison could not be made on benchmark-eligible molecules
at all. Every gate the benchmark applies belongs here, applied before a call is spent.

The patterns below are deliberately conservative: each one is a motif no medicinal chemist
would carry forward, not a general reactivity model. Being conservative is the right error
here, since a rejected candidate costs nothing and an accepted one costs an oracle call
and can anchor a whole search.
"""

from __future__ import annotations

import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig

RDLogger.DisableLog("rdApp.*")

# The real synthetic-accessibility scorer ships with RDKit's contrib tree but is not on
# the import path by default. It is the gate the T4 benchmark applies, and an earlier
# version of this module substituted a hand-rolled ring-strain proxy for it -- which
# rejected ordinary bridged azabicyclics while admitting the rest. Use the real one.
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

SCHEMA_VERSION = "queryable_fiber_v1"

# (name, SMARTS). Motifs that are unstable, indiscriminately electrophilic, or otherwise
# not carryable. Named so a rejection can be explained rather than just counted.
UNSTABLE_MOTIFS = (
    ("acyl_nitroso_or_N_O_acyl", "[CX3](=O)[OX2][NX3]"),
    ("N_O_N_or_peroxide_like", "[OX2][OX2,NX3][OX2]"),
    ("peroxide", "[OX2][OX2]"),
    ("anhydride", "[CX3](=O)[OX2][CX3]=O"),
    ("acyl_halide", "[CX3](=O)[F,Cl,Br,I]"),
    ("azide", "[NX1]~[NX2]~[NX2,NX1]"),
    ("diazo_or_diazonium", "[#6]=[NX2+]=[NX1-,NX1]"),
    ("nitroso", "[NX2]=O"),
    ("isocyanate_isothiocyanate", "[NX2]=[CX2]=[OX1,SX1]"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("michael_acceptor_ketone", "[CX3]=[CX3][CX3]=[OX1]"),
    ("epoxide_aziridine", "[OX2,NX3]1[CX4][CX4]1"),
    ("geminal_diol_or_hemiketal", "[CX4]([OX2H])([OX2H,OX2])"),
    ("thiol", "[SX2H]"),
    ("hydrazine", "[NX3][NX3]"),
    ("hydroxylamine", "[NX3][OX2H,OX2]"),
    ("radical_or_open_valence", "[#6&v0,#6&v1,#6&v2,#7&v0,#7&v1,#8&v0]"),
)

_COMPILED = tuple((name, Chem.MolFromSmarts(pattern)) for name, pattern in UNSTABLE_MOTIFS)


def instability(smiles: str) -> list[str]:
    """Every unstable motif this molecule carries, by name. Empty means clean."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ["unparseable"]
    return [
        name
        for name, pattern in _COMPILED
        if pattern is not None and mol.HasSubstructMatch(pattern)
    ]


def admit(
    smiles: str,
    *,
    reference_fingerprint,
    generator,
    similarity_floor: float = 0.4,
    qed_floor: float = 0.6,
    sa_ceiling: float = 4.0,
    heavy_range: tuple[int, int] = (18, 40),
) -> dict:
    """Is this molecule worth an oracle call, and if not, exactly why?

    Returns the verdict with every gate it failed rather than a bare boolean, so the
    funnel can be reported and a filter can never silently swallow the candidate pool.
    """
    from rdkit.Chem import DataStructs

    verdict = {"smiles": smiles, "admitted": False, "failed": []}
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        verdict["failed"].append("unparseable")
        return verdict
    canonical = Chem.MolToSmiles(mol)
    verdict["smiles"] = canonical

    if "." in canonical:
        verdict["failed"].append("disconnected")
    heavy = mol.GetNumHeavyAtoms()
    verdict["heavy_atoms"] = heavy
    if not heavy_range[0] <= heavy <= heavy_range[1]:
        verdict["failed"].append("size")

    quality = QED.qed(mol)
    verdict["qed"] = round(quality, 3)
    if quality <= qed_floor:
        verdict["failed"].append("qed")

    similarity = DataStructs.TanimotoSimilarity(
        reference_fingerprint, generator.GetFingerprint(mol)
    )
    verdict["similarity"] = round(similarity, 3)
    if similarity <= similarity_floor:
        verdict["failed"].append("similarity")

    motifs = instability(canonical)
    verdict["unstable_motifs"] = motifs
    if motifs:
        verdict["failed"].append("stability")

    accessibility = sascorer.calculateScore(mol)
    verdict["sa"] = round(accessibility, 2)
    if accessibility >= sa_ceiling:
        verdict["failed"].append("sa")

    verdict["admitted"] = not verdict["failed"]
    return verdict
