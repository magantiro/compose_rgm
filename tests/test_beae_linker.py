"""The BEAE linker: an invariant 12-atom core, the frozen scaffold for Fig-6.

Locks the definition: the frozen linker core is identical across BEAE lipids
(independent of head/tail), the decomposition partitions every atom into
{linker, head, tail1, tail2}, the ionizable amine is in the head (not the frozen
linker), and non-BEAE lipids are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

from compose_v4.lipids.beae_linker import (
    beae_attachment_points,
    beae_decompose,
    beae_linker_atoms,
    is_beae,
)
from compose_v4.oracles.head_domain import basic_nitrogens

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/lipid_reactions/beae_linker_v1.json"

BEAE = {
    "RM-60": "CN(C)CCCN(CCC(=O)OCC(CCCCCC)CCCCCCCC)/C=C/C(=O)OC(CCCCCCCC)CCCCCCCCCC",
    "Ex-2": "CN(C)CCCN(CCC(=O)OCCCCCCCCC(C)C)/C=C/C(=O)OCCCCCCCC/C=C\\C/C=C\\CCCCC",
    "H3-head": "C1CCCCN1CCCN(CCC(=O)OCC(CC)CCCC)/C=C/C(=O)OCCCCCCCCCCCC",
    "H4-head": "CN(C)CC(C)(C)CN(CCC(=O)OCC(CC)CCCC)/C=C/C(=O)OCCCCCCCCCCCC",
}
NON_BEAE = "CCCCCCCCCCCCOC(=O)CCN(C)CCC(=O)OCCCCCCCC"  # di-ester amine, no propiolate enamine


def test_frozen_linker_core_is_invariant() -> None:
    sizes = set()
    for smi in BEAE.values():
        m = Chem.MolFromSmiles(smi)
        assert is_beae(m)
        sizes.add(len(beae_linker_atoms(m)))
    assert sizes == {11}  # exactly the 11 frozen core atoms, every time


def test_decomposition_partitions_all_atoms_head_has_amine() -> None:
    for name, smi in BEAE.items():
        m = Chem.MolFromSmiles(smi)
        d = beae_decompose(m)
        parts = [d["linker"], d["head"], d["tail1"], d["tail2"]]
        union = set().union(*parts)
        assert len(union) == m.GetNumHeavyAtoms(), name          # covers everything
        assert sum(len(p) for p in parts) == len(union), name     # disjoint (no overlap)
        assert any(i in d["head"] for i in basic_nitrogens(m)), name  # ionizable amine in the head
        # attachment points are outside the frozen linker
        for anchor in beae_attachment_points(m).values():
            assert anchor not in d["linker"], name


def test_non_beae_is_rejected() -> None:
    m = Chem.MolFromSmiles(NON_BEAE)
    assert not is_beae(m)
    assert beae_linker_atoms(m) == frozenset()
    assert beae_decompose(m) is None


def test_config_matches_module() -> None:
    c = json.loads(CONFIG.read_text())
    assert c["frozen_atom_maps"] == list(range(1, 12))
    assert c["attachment_points"] == {"head": 14, "tail1": 12, "tail2": 13}
    assert c["provenance"]["not_in_general_pretraining"] is True
