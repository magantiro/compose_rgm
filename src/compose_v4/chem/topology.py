"""Shared molecular topology descriptors, independent of any controller."""

from __future__ import annotations

from compose_v4.chem.molecular_graph import MolecularGraph


def cycle_rank(state: MolecularGraph | None, key: str | None = None) -> int:
    """First Betti number, bonds minus atoms plus components; zero for null.

    Keep the registered canonicalization and optional-key interface used by the
    exact-graph reports. Ring-perception counts are not a substitute for this
    graph invariant, particularly for bridged molecules.
    """
    # The executor owns canonical molecular identity. Import lazily so the
    # descriptor introduces no chemistry/executor import cycle.
    from rdkit import Chem

    from compose_v4.rewrite.kernel import canonical_state_key

    resolved = key if key is not None else canonical_state_key(state)
    if resolved == "<NULL>":
        return 0
    molecule = Chem.MolFromSmiles(resolved, sanitize=False)
    if molecule is None:
        return 0
    components = len(Chem.GetMolFrags(molecule))
    return int(molecule.GetNumBonds() - molecule.GetNumAtoms() + components)
