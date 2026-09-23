"""Contextual macro value: which transformation is worth trying FROM THIS MOLECULE.

WHY THE MACRO DESCRIPTORS ARE NOT ENOUGH.  A model reading only edit counts, ring-operation
counts, retention and channel cannot separate two macros that differ in WHICH chemistry changed:
if `phi(G, o1) == phi(G, o2)` the predictor must assign them the same value even when their
endpoints score differently.  That is a representation limit, not a credit-assignment problem, and
no reward reshaping fixes it.  Measured on the beam's own transitions, siblings from one parent
span a median 0.3434 of the objective, so there is a great deal the coarse descriptors cannot see.

FEATURE BLOCKS ARE SEPARABLE ON PURPOSE.  "Molecular context helps" is a claim to measure, not to
assume, so each block can be enabled independently and the marginal value of each is reported
rather than asserted.

INFORMATION BOUNDARY.  Every feature is computed from molecules the optimizer itself produced and
from scores it paid for.  The candidate's structure is available BEFORE its oracle call -- COMPOSE
has already executed the program -- so representing it leaks nothing.  No target fingerprint, no
hidden answer structure, no teacher route, and nothing a normal PMO optimizer could not compute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")

#: Folded fingerprint width. Deliberately narrow: the development set is thousands of rows, not
#: millions, and a 2048-bit design over 9,500 examples is a variance problem wearing a
#: representation's clothes.
FINGERPRINT_BITS = 128

MACRO_FAMILIES: tuple[str, ...] = (
    "region_replace",
    "segment_grow",
    "segment_shrink",
    "segment_replace",
    "substituent_delete",
    "append_ring",
    "fuse_ring",
    "functionalize",
    "carbonyl_insert",
    "heteroatom_substitute",
    "bond_reroute",
    "cycle_open",
    "cycle_close",
    "ring_system_restate",
    # Anchored rebuild options reachable only through a region replacement.  Without
    # these the model cannot prefer "excise and rebuild a ring" over "excise and regrow a
    # chain" -- it would see both as the single label ``region_replace``.
    "regrow",
    "ring_then_grow",
)

FEATURE_BLOCKS: tuple[str, ...] = (
    "macro_intent",
    "parent_descriptor",
    "parent_fingerprint",
    "macro_realized",
    "endpoint_descriptor",
    "change",
    "endpoint_fingerprint",
)

#: Available BEFORE the macro is executed, so legal for deciding what to propose.  A family and a
#: requested module count are INTENTS -- the question `Q_pre` answers is "if I attempt this family
#: from this parent, what is it worth", which is exactly the allocation decision.
PRE_EXECUTION_BLOCKS: tuple[str, ...] = (
    "macro_intent",
    "parent_descriptor",
    "parent_fingerprint",
)

#: Everything, including the realized endpoint.  Legal only AFTER COMPOSE has built the molecule,
#: i.e. for deciding which realized candidates deserve an oracle call.
POST_EXECUTION_BLOCKS: tuple[str, ...] = FEATURE_BLOCKS


def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def _descriptors(mol) -> np.ndarray:
    """Cheap, interpretable bulk properties. None is task-specific."""
    if mol is None:
        return np.zeros(8)
    return np.asarray(
        [
            mol.GetNumHeavyAtoms() / 40.0,
            Descriptors.MolWt(mol) / 500.0,
            Crippen.MolLogP(mol) / 5.0,
            Descriptors.TPSA(mol) / 140.0,
            mol.GetRingInfo().NumRings() / 6.0,
            Descriptors.NumHDonors(mol) / 5.0,
            Descriptors.NumHAcceptors(mol) / 10.0,
            Descriptors.NumRotatableBonds(mol) / 10.0,
        ],
        dtype=float,
    )


def _folded(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(FINGERPRINT_BITS)
    bits = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_BITS)
    array = np.zeros(FINGERPRINT_BITS, dtype=float)
    DataStructs.ConvertToNumpyArray(bits, array)
    return array


def macro_families(candidate) -> tuple[str, ...]:
    """The MACRO families a candidate realized, read from the synthesis metadata.

    THE POINT OF THIS FUNCTION IS WHAT IT REFUSES TO READ.  A candidate also carries a
    histogram of EXECUTOR RULE NAMES (``atom_insert``, ``cycle_close``, ``bond_reroute``
    ...), and four of those names are spelled identically to macro families.  Keying the
    value model on that histogram would therefore half-work: some family one-hots would
    fire, for the wrong reason, from primitive counts.  That is precisely the
    semi-Markov violation the macro abstraction exists to prevent -- the coherent program
    is the action, not the primitives inside it.

    A region replacement is reported at BOTH resolutions: the compound
    ``region_replace:<rebuild>`` and the bare rebuild family, so the model can learn "a
    ring rebuild is good here" while still sharing strength with ring builds reached any
    other way.
    """
    modules = ((candidate.get("provenance") or {}).get("metadata") or {}).get("modules")
    if modules is None:
        modules = (candidate.get("metadata") or {}).get("modules") or ()
    found: list[str] = []
    for module in modules:
        family = str(module.get("family", ""))
        if not family:
            continue
        found.append(family)
        rebuild = (module.get("parameters") or {}).get("rebuild_option")
        if rebuild:
            found.extend([f"{family}:{rebuild}", str(rebuild)])
    return tuple(dict.fromkeys(found))


def macro_scale(candidate) -> dict:
    """Realized SCALE of a macro: how much structure it actually moved.

    Reported separately from primitive count because they are different quantities -- a
    coherent eleven-primitive ring operation and an arbitrary eleven-primitive program
    have the same length and different scale, and the whole reason to represent macros is
    that the second number is the informative one.
    """
    modules = ((candidate.get("provenance") or {}).get("metadata") or {}).get("modules")
    if modules is None:
        modules = (candidate.get("metadata") or {}).get("modules") or ()
    excised = rebuilt = largest = 0
    for module in modules:
        parameters = module.get("parameters") or {}
        excised = max(excised, int(parameters.get("excised_atoms", 0) or 0))
        rebuilt = max(rebuilt, int(parameters.get("rebuild_capacity", 0) or 0))
        largest = max(largest, int(module.get("primitive_edits", 0) or 0))
    changes = (candidate.get("provenance") or {}).get("actual_changes") or {}
    return {
        "excised_atoms": excised,
        "rebuild_capacity": rebuilt,
        "largest_module_primitives": largest,
        "d_heavy": float(changes.get("heavy_atom_delta", 0) or 0),
        "d_rings": float(changes.get("ring_delta", 0) or 0),
    }


def _macro_intent_block(edge) -> np.ndarray:
    """Only what a caller knows BEFORE executing: the family it intends to attempt, the module
    count it requested, the parent's own score, and where in the search it stands.

    `primitives` and `module_count` are REALIZED and are deliberately excluded here -- a program's
    true length is known only once it compiles, so using it to decide what to propose would be
    reading the answer.
    """
    families = set(edge.get("families") or ())
    return np.asarray(
        [
            float(edge.get("parent_score", 0.0)),
            float(edge.get("requested_modules", 0)) / 4.0,
            float(edge.get("depth", 0)) / 16.0,
            float(edge.get("generation", 0)) / 16.0,
            float(bool(edge.get("capacity_aware"))),
            *[float(name in families) for name in MACRO_FAMILIES],
        ],
        dtype=float,
    )


def edge_features(edge, *, blocks=FEATURE_BLOCKS) -> np.ndarray:
    """Feature vector for one executed (parent, macro, endpoint) transition."""
    unknown = set(blocks) - set(FEATURE_BLOCKS)
    if unknown:
        raise ValueError(f"unknown feature blocks: {sorted(unknown)}")
    parent = _mol(edge["parent"])
    child = _mol(edge["endpoint"]) if _needs_endpoint(blocks) else None
    parts: list[np.ndarray] = []
    if "macro_intent" in blocks:
        parts.append(_macro_intent_block(edge))
    if "parent_descriptor" in blocks:
        parts.append(_descriptors(parent))
    if "parent_fingerprint" in blocks:
        parts.append(_folded(parent))
    if "macro_realized" in blocks:
        parts.append(
            np.asarray(
                [
                    float(edge.get("primitives", 0)) / 32.0,
                    float(edge.get("module_count", 0)) / 4.0,
                    # SCALE, not length.  An eleven-primitive coherent ring operation and
                    # an arbitrary eleven-primitive program agree on the two rows above and
                    # differ here, which is the distinction the macro abstraction is for.
                    float(edge.get("excised_atoms", 0)) / 16.0,
                    float(edge.get("rebuild_capacity", 0)) / 16.0,
                    float(edge.get("largest_module_primitives", 0)) / 16.0,
                    float(edge.get("d_heavy", 0)) / 16.0,
                    float(edge.get("d_rings", 0)) / 4.0,
                    # The family that ACTUALLY executed, which need not be the one intended:
                    # a region replacement rebuilds with whichever option first executes.
                    *[
                        float(name in set(edge.get("realized_families") or ()))
                        for name in MACRO_FAMILIES
                    ],
                ],
                dtype=float,
            )
        )
    if "endpoint_descriptor" in blocks:
        parts.append(_descriptors(child))
    if "change" in blocks:
        similarity = 0.0
        if parent is not None and child is not None:
            similarity = DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprint(parent, 2), AllChem.GetMorganFingerprint(child, 2)
            )
        parts.append(np.concatenate([[similarity], _descriptors(child) - _descriptors(parent)]))
    if "endpoint_fingerprint" in blocks:
        parts.append(_folded(child))
    parts.append(np.ones(1))
    return np.concatenate(parts)


def _needs_endpoint(blocks) -> bool:
    return bool({"macro_realized", "endpoint_descriptor", "change", "endpoint_fingerprint"} & set(blocks))


@dataclass
class ContextualMacroValue:
    """Ridge over the enabled blocks, predicting the ENDPOINT PMO score.

    The endpoint score is the primary target because it is DENSE: every scored candidate teaches
    something, including the ones that add nothing to the archive. Archive gain is computed FROM
    the prediction at decision time rather than regressed directly, which would discard every
    sub-threshold observation.
    """

    penalty: float = 1.0
    blocks: tuple[str, ...] = FEATURE_BLOCKS
    weights: np.ndarray | None = None
    _mean: np.ndarray | None = None
    _scale: np.ndarray | None = None

    def design(self, edges) -> np.ndarray:
        return np.vstack([edge_features(e, blocks=self.blocks) for e in edges])

    def fit(self, edges, targets) -> None:
        x = self.design(edges)
        y = np.asarray(targets, dtype=float)
        mean, scale = x.mean(axis=0), x.std(axis=0)
        constant = scale < 1e-12
        mean = np.where(constant, 0.0, mean)
        scale = np.where(constant, 1.0, scale)
        z = (x - mean) / scale
        ridge = self.penalty * np.eye(z.shape[1])
        ridge[-1, -1] = 0.0
        self.weights = np.linalg.solve(z.T @ z + ridge * len(y), z.T @ y)
        self._mean, self._scale = mean, scale

    def predict(self, edges) -> np.ndarray:
        if self.weights is None:
            raise ValueError("model is unfitted")
        return ((self.design(edges) - self._mean) / self._scale) @ self.weights
