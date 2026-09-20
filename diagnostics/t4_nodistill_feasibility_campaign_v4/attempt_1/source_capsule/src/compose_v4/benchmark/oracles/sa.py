"""Synthetic accessibility, with the fragment table frozen out of a pickle.

The algorithm is Ertl & Schuffenhauer's, as implemented in RDKit's
`Contrib/SA_Score/sascorer.py` -- the same code TDC copied into its own oracle
module, which is what MOLLEO calls.  It is vendored here rather than imported so
that the benchmark does not depend on an RDKit contrib path that moves between
releases, and the 705,292-entry fragment table is loaded from a checksummed npz
instead of a pickle.

The arithmetic is reproduced line for line, including the parts that differ from
the paper (the macrocycle penalty and the symmetry correction) and the 8.0
smoothing branch.  Deviating from any of them would produce a plausible SA-like
number that is not the benchmark's SA.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

#: Raw SA is bounded to [1, 10] by the transform at the end of `calculate_score`.
SA_MIN = 1.0
SA_MAX = 10.0
#: The score of a fragment the table has never seen.
UNKNOWN_FRAGMENT_SCORE = -4.0


class FragmentScores:
    """The frozen Ertl fragment-contribution table."""

    def __init__(self, parameters_path: Path):
        data = np.load(Path(parameters_path))
        ids = np.asarray(data["fragment_id"], dtype=np.int64)
        scores = np.asarray(data["score"], dtype=np.float64)
        # A dict beats a searchsorted lookup here: SA is called once per
        # molecule with ~30 fragment ids, so per-lookup constant factor is what
        # matters, not vectorisation.
        self.table: dict[int, float] = dict(zip(ids.tolist(), scores.tolist()))

    def __len__(self) -> int:
        return len(self.table)

    def get(self, fragment_id: int) -> float:
        return self.table.get(fragment_id, UNKNOWN_FRAGMENT_SCORE)


def calculate_score(mol, fragments: FragmentScores) -> float:
    """`sascorer.calculateScore`, reproduced exactly, on a frozen table."""

    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    # radius 2, counts, unfolded -- the identifiers ARE the table's keys, so
    # this fingerprint may not be folded or bit-vectorised.
    counts = rdMolDescriptors.GetMorganFingerprint(mol, 2).GetNonzeroElements()

    score1 = 0.0
    n_fragments = 0
    for fragment_id, count in counts.items():
        n_fragments += count
        score1 += fragments.get(fragment_id) * count
    score1 /= n_fragments

    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ring_info = mol.GetRingInfo()
    n_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_macrocycles = sum(1 for ring in ring_info.AtomRings() if len(ring) > 8)

    size_penalty = n_atoms ** 1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    # Deliberately NOT log10(n_macrocycles + 1): the upstream implementation
    # departs from the paper here and every published SA number follows the
    # implementation.
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

    score2 = (0.0 - size_penalty - stereo_penalty - spiro_penalty
              - bridge_penalty - macrocycle_penalty)

    # Fingerprint-density correction: symmetrical molecules are easier to make.
    score3 = 0.0
    if n_atoms > len(counts):
        score3 = math.log(float(n_atoms) / len(counts)) * 0.5

    sascore = score1 + score2 + score3

    # Transform the raw value onto [1, 10].
    low, high = -4.0, 2.5
    sascore = 11.0 - (sascore - low + 1) / (high - low) * 9.0
    if sascore > 8.0:                      # smooth the 10-end
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0
    return sascore


def sa_from_smiles(smiles: str, fragments: FragmentScores) -> float:
    """Raw SA for a SMILES; 10.0 (the worst) for anything unparseable.

    TDC returns 100 for an unparseable molecule, which is off the [1, 10] scale
    the normalisation assumes.  Clamping to the worst in-range value keeps the
    normalised objective inside [0, 1] without ever rewarding a non-molecule --
    the two agree wherever the benchmark actually operates.
    """

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return SA_MAX
    return calculate_score(mol, fragments)
