"""Guards for the mined attachment-compatibility table.

The fixture is the MEASURED defect: an unfiltered macro bank put `S(=O)(=O)S(=O)(=O)`
disulfones in 4 of 8 macro seats. Every one is valence-legal and executor-constructible --
validity closure is a VALENCE guarantee and says nothing about whether a bond environment
occurs in real chemistry.
"""
from __future__ import annotations

from rdkit import Chem

from compose_v4.control.attachment_compatibility import (
    admissible,
    bond_environment_key,
    build_attachment_table,
    new_bond_indices,
)

# Small but real: sulfonamides and biaryls occur, sulfonyl-sulfonyl does not.
CORPUS = [
    "Cc1ccc(S(=O)(=O)N2CCOCC2)cc1", "Cc1ccc(-c2ccccc2)cc1",
    "CC(C)c1ccc(S(N)(=O)=O)cc1", "O=S(=O)(Nc1ccccc1)c1ccccc1",
    "Cc1ccc(S(=O)(=O)c2ccccc2)cc1", "CCOc1ccc(C(=O)NC)cc1",
    "Cc1ccc(-n2cccn2)cc1", "CSc1ccccc1", "CCS(=O)(=O)c1ccccc1",
    # Supplies all three environments the celecoxib sulfamoylphenyl swap introduces:
    # aryl-aryl, aryl-sulfonyl and sulfonyl-N. Chosen by computing the swap's novel keys
    # and finding a molecule that carries them, not by guessing.
    "NS(=O)(=O)c1ccc(-c2ccn[nH]2)cc1",
]


def _table():
    return build_attachment_table(CORPUS)


def test_table_is_built_from_structure_only():
    table = _table()
    assert table.molecules == len(CORPUS)
    assert table.counts, "no bond environments mined from a real corpus"


def test_the_measured_disulfone_defect_is_refused():
    table = _table()
    source = "Cc1ccc(S(=O)(=O)c2nnn(-c3ccc(C(C)C)cc3)c2N)cc1"
    disulfone = "Cc1ncc(-c2ccc(S(=O)(=O)S(=O)(=O)c3nnn(-c4ccc(C(C)C)cc4)c3N)cc2)o1"
    assert Chem.MolFromSmiles(disulfone) is not None, "fixture must be RDKit-valid"
    assert not admissible(source, disulfone, table)


def test_an_ordinary_biaryl_attachment_is_admitted():
    table = _table()
    assert admissible("Cc1ccc(-n2nc(C(F)(F)F)cc2-c2cccs2)cc1",
                      "Cc1ccc(-n2nc(C(F)(F)F)cc2-c2ccc(S(N)(=O)=O)cc2)cc1", table)


def test_a_bond_environment_key_ignores_the_endpoint_order():
    mol = Chem.MolFromSmiles("Cc1ccc(S(=O)(=O)N2CCOCC2)cc1")
    bond = next(b for b in mol.GetBonds()
                if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE)
    assert (bond_environment_key(mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            == bond_environment_key(mol, bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))


def test_only_bonds_the_macro_INTRODUCED_are_judged():
    # A molecule compared with itself introduces nothing, so nothing can be refused --
    # otherwise the filter would reject the source's own pre-existing chemistry.
    source = "Cc1ccc(S(=O)(=O)c2ccccc2)cc1"
    assert new_bond_indices(source, source) == []
    assert admissible(source, source, build_attachment_table(["CCO"]))


def test_strictness_scales_with_the_corpus_and_an_undersized_table_refuses_broadly():
    """The operating characteristic, pinned so nobody deploys an undersized table.

    MEASURED: a 60,000-molecule table admitted the celecoxib swap that worked; a table
    built from a single unrelated molecule refuses it. The filter is only as permissive
    as the chemistry it has seen, so a small corpus is not a conservative choice -- it is
    a filter that rejects real chemistry.
    """
    source = "Cc1ccc(-n2nc(C(F)(F)F)cc2-c2cccs2)cc1"
    endpoint = "Cc1ccc(-n2nc(C(F)(F)F)cc2-c2ccc(S(N)(=O)=O)cc2)cc1"
    assert not admissible(source, endpoint, build_attachment_table(["CCO"]))
    assert admissible(source, endpoint, _table())


def test_min_support_actually_binds():
    # A guard is only tested where it binds: an environment seen once must pass at
    # support 1 and fail at support 2.
    table = build_attachment_table(["Cc1ccc(S(=O)(=O)N2CCOCC2)cc1"])
    key = next(iter(table.counts))
    assert table.occurs(key, min_support=1)
    assert not table.occurs(key, min_support=table.support(key) + 1)


def test_table_never_reads_a_task_or_an_oracle():
    import ast
    import pathlib

    tree = ast.parse(
        pathlib.Path("src/compose_v4/control/attachment_compatibility.py").read_text())
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            seen.add((node.module or "").split(".")[0])
            seen.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            seen.update(alias.name.split(".")[0] for alias in node.names)
    assert not (seen & {"Oracle", "tdc", "oracle_score", "TanimotoSimilarity"})
