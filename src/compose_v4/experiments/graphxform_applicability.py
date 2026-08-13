"""GraphXForm's declared applicability domain — FROZEN 2026-08-13.

Frozen **before any COMPOSE-versus-GraphXForm number exists**, which is the only
thing that makes it legitimate. GraphXForm cannot represent certain molecules at
all, and that is a native constraint of the method, not a judgement about which
molecules it would score badly on. Applied transparently and up front it is
fair; applied after seeing results it would be silent cherry-picking.

**A molecule outside this domain is reported as `INAPPLICABLE` for GraphXForm.
It is never silently dropped from the comparison.** A reader must be able to see
how many molecules each method could not attempt.

Every rule below is verified against the upstream source, not inferred:

* **Element set** — `config.py` lines 20–49 declare 23 vocabulary entries
  covering nine elements. A molecule containing anything else fails with an
  **uncaught `KeyError`** at `molecule_design.py:649`
  (`atom_idx = atomic_num_to_atom_idx[k]`), not a graceful error.
* **Formal charges and chirality** are supported for the elements that declare
  them, and the key is built as ``f"{Z}_{charge}"`` / ``f"{Z}@{tag}"``, so an
  unsupported charge state on a supported element also raises `KeyError`.
* **Explicit hydrogens** raise `KeyError: 1` — H is not in the vocabulary.
* **Salts and counterions** raise `KeyError` on the counterion's element (e.g.
  `KeyError: '11_1'` for Na+). Multi-fragment inputs are reduced to the largest
  fragment before the check.
* **Heavy-atom ceiling** — `config.max_num_atoms = 50` becomes
  `upper_limit_atoms`, and exceeding it raises
  ``AssertionError: Trying to take action 1 on level 0, but it is set to
  infeasible``.
* **Headroom** — GraphXForm is strictly constructive: `AddAtom`, `AddBond`,
  `DontChange`, with removal deferred to future work. A source at the ceiling
  can therefore take no productive action at all, so the domain reserves
  headroom below the ceiling rather than admitting molecules the method cannot
  extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The nine elements GraphXForm's shipped vocabulary can represent, by atomic
#: number. From config.py lines 20-49: C, N, O, F, P, S, Cl, Br, I.
ALLOWED_ATOMIC_NUMBERS = frozenset({6, 7, 8, 9, 15, 16, 17, 35, 53})

#: (atomic number, formal charge) pairs the vocabulary declares. A charge state
#: outside this set raises KeyError even on an allowed element.
ALLOWED_CHARGE_STATES = frozenset(
    {
        (6, 0), (6, -1), (6, 1),
        (7, 0), (7, -1), (7, 1),
        (8, 0), (8, -1), (8, 1),
        (9, 0),
        (15, 0), (15, -1), (15, 1),
        (16, 0), (16, -1), (16, 1),
        (17, 0), (35, 0), (53, 0),
    }
)

#: config.max_num_atoms as shipped.
MAX_NUM_ATOMS = 50

#: CHOSEN SAFETY MARGIN, NOT A STRUCTURAL DERIVATION. Audited 2026-08-13 and
#: downgraded after that audit.
#:
#: A structural derivation would need a native bound on maximum growth over a
#: registered optimization horizon. GraphXForm has NO such bound: the search
#: runs until the policy emits TERMINATE or a wall-clock limit expires
#: (`wall_clock_limit`, 8 h in the paper), and `num_epochs` bounds fine-tuning
#: epochs rather than atoms added. So maximum growth is not derivable from the
#: action semantics, and any headroom number is a judgement call.
#:
#: 8 was chosen as a round margin allowing a small substituent. It is honest to
#: record that it also comfortably admits the COMPOSE cohort (max 34 heavy
#: atoms), which is exactly the reasoning that would be illegitimate if it were
#: the *justification* rather than an observation about the consequence. It is
#: not presented as structural, and the ceiling must not be retuned after any
#: comparative outcome exists.
REQUIRED_HEADROOM = 8
HEADROOM_BASIS = "CHOSEN_SAFETY_MARGIN"
HEADROOM_AUDIT = (
    "Not structurally derived. GraphXForm bounds its search by TERMINATE or "
    "wall clock, not by a maximum atom count over a registered horizon, so no "
    "native maximum-growth bound exists to derive 8 from. Recorded as a chosen "
    "margin. Frozen 2026-08-13 before any COMPOSE-versus-GraphXForm number."
)

#: Molecules with at most this many heavy atoms are eligible.
HEAVY_ATOM_CEILING = MAX_NUM_ATOMS - REQUIRED_HEADROOM

#: Multi-fragment inputs are reduced to the largest fragment first, matching the
#: usual convention for stripping counterions. If the discarded fragments are
#: not counterions this is recorded, because it changes the molecule.
STRIP_TO_LARGEST_FRAGMENT = True

UPSTREAM_EVIDENCE = {
    "repository": "https://github.com/grimmlab/graphxform",
    "commit": "867bdcf9d9a8d4a6bfe29a93106b62b383b1a232",
    "element_set": "config.py:20-49 (atom_vocabulary, 23 entries)",
    "unsupported_element_failure": (
        "molecule_design.py:649 `atom_idx = atomic_num_to_atom_idx[k]` -- uncaught KeyError"
    ),
    "atom_ceiling": "config.py:18 `self.max_num_atoms = 50` -> upper_limit_atoms",
    "atom_ceiling_failure": (
        "AssertionError: Trying to take action 1 on level 0, but it is set to infeasible"
    ),
    "constructive_only": (
        "action space is AddAtom / AddBond / DontChange; arXiv:2411.01667 section 3.3.3 "
        "defers removal to future work"
    ),
}


@dataclass(frozen=True)
class ApplicabilityVerdict:
    smiles: str
    applicable: bool
    reason: str
    heavy_atoms: int | None = None
    offending: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "applicable": self.applicable,
            "reason": self.reason,
            "heavy_atoms": self.heavy_atoms,
            "offending": list(self.offending),
        }


def check_applicability(smiles: str) -> ApplicabilityVerdict:
    """Decide whether GraphXForm can represent this molecule at all.

    Returns a verdict rather than raising, so an ineligible molecule can be
    REPORTED as inapplicable instead of vanishing from the comparison.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return ApplicabilityVerdict(smiles, False, "unparseable")

    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    stripped = False
    if len(fragments) > 1:
        if not STRIP_TO_LARGEST_FRAGMENT:
            return ApplicabilityVerdict(smiles, False, "multi_fragment")
        molecule = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
        stripped = True

    offending: list[str] = []
    for atom in molecule.GetAtoms():
        number = atom.GetAtomicNum()
        if number not in ALLOWED_ATOMIC_NUMBERS:
            offending.append(f"Z={number}")
            continue
        if (number, int(atom.GetFormalCharge())) not in ALLOWED_CHARGE_STATES:
            offending.append(f"Z={number}{atom.GetFormalCharge():+d}")
        if atom.GetNumExplicitHs() and atom.GetAtomicNum() == 1:
            offending.append("explicit_H")

    heavy = molecule.GetNumHeavyAtoms()
    if offending:
        return ApplicabilityVerdict(
            smiles,
            False,
            "unsupported_element_or_charge_state",
            heavy,
            tuple(sorted(set(offending))),
        )
    if heavy > HEAVY_ATOM_CEILING:
        return ApplicabilityVerdict(
            smiles,
            False,
            f"exceeds_heavy_atom_ceiling_{HEAVY_ATOM_CEILING}",
            heavy,
        )
    return ApplicabilityVerdict(
        smiles,
        True,
        "applicable_after_largest_fragment" if stripped else "applicable",
        heavy,
    )


def partition_panel(smiles: list[str]) -> dict[str, Any]:
    """Split a task panel into applicable and inapplicable, and report both.

    The inapplicable list is the point: it is what lets a reader see how many
    molecules GraphXForm could not attempt.
    """
    verdicts = [check_applicability(smi) for smi in smiles]
    applicable = [v for v in verdicts if v.applicable]
    inapplicable = [v for v in verdicts if not v.applicable]
    reasons: dict[str, int] = {}
    for verdict in inapplicable:
        reasons[verdict.reason] = reasons.get(verdict.reason, 0) + 1
    return {
        "n_total": len(smiles),
        "n_applicable": len(applicable),
        "n_inapplicable": len(inapplicable),
        "inapplicable_reasons": reasons,
        "applicable_smiles": [v.smiles for v in applicable],
        "inapplicable": [v.as_dict() for v in inapplicable],
        "domain": {
            "allowed_atomic_numbers": sorted(ALLOWED_ATOMIC_NUMBERS),
            "heavy_atom_ceiling": HEAVY_ATOM_CEILING,
            "max_num_atoms": MAX_NUM_ATOMS,
            "required_headroom": REQUIRED_HEADROOM,
            "headroom_basis": HEADROOM_BASIS,
            "headroom_audit": HEADROOM_AUDIT,
            "strip_to_largest_fragment": STRIP_TO_LARGEST_FRAGMENT,
        },
        "frozen": "2026-08-13, before any COMPOSE-versus-GraphXForm number existed",
        "upstream_evidence": UPSTREAM_EVIDENCE,
    }
