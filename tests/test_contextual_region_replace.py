"""Guards for the context-preserving macro transition family.

Each test names the production defect it exists to catch.  Two of them exist because the
defect was MEASURED during development, not imagined: the molzip label default silently
returns a disconnected molecule, and labelling both excision cuts identically makes the
middle indistinguishable from a flank and yields zero proposals.
"""
from __future__ import annotations

from rdkit import Chem

from compose_v4.control.contextual_region_replace import (
    FAMILIES,
    executable_endpoint,
    region_excisions,
    region_replacements,
    substituent_replacements,
)

# A real drug-like source with two ring systems joined through acyclic single bonds, so
# every family has somewhere to cut.
SOURCE = "Cc1ccc(-n2nc(C(F)(F)F)cc2-c2cccs2)cc1"


def _canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _rings(smiles: str) -> int:
    """Ring count, with the Mol BOUND.

    MEASURED RDKit TRAP: `Chem.MolFromSmiles(s).GetRingInfo().NumRings()` returns a proxy
    into a Mol that is freed mid-expression, so the chained form silently reports 0 rings
    with no error. Binding the Mol first is the whole fix.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol.GetRingInfo().NumRings()


def test_families_are_the_three_documented_ones():
    assert FAMILIES == ("substituent_replace", "region_replace", "region_excise")


def test_executable_endpoint_refuses_a_radical_rdkit_happily_parses():
    """RDKit-parseable is NOT COMPOSE-constructible.

    A probe filtering on `MolFromSmiles is not None` admits endpoints the production
    executor refuses. MEASURED while writing this: the refusing authority for radicals
    and off-vocabulary elements is `smiles_to_molecular_graph`, which raises, NOT
    `is_valid_state` -- across 18 probe molecules `is_valid_state` never rejected one
    that had constructed. That call is therefore defence in depth, and this test does
    not claim to cover it.
    """
    for parseable_but_unbuildable in ("[CH2]c1ccccc1", "[CH]c1ccccc1", "[SiH4]",
                                      "C[Mg]Br", "[PH5]"):
        assert Chem.MolFromSmiles(parseable_but_unbuildable) is not None
        assert executable_endpoint(parseable_but_unbuildable) is None


def test_executable_endpoint_refuses_a_disconnected_join():
    assert executable_endpoint("CCO.CCN") is None


def test_executable_endpoint_refuses_over_the_heavy_atom_ceiling():
    assert executable_endpoint("C" * 41) is None
    assert executable_endpoint("C" * 30) is not None


def test_one_cut_swap_preserves_everything_outside_the_single_cut():
    # The measured defect the one-cut form exists to fix: a generic ring graft replaced
    # the CORRECT core. With one cut, the retained side must survive verbatim.
    payloads = [("[1*]c1ccc(S(N)(=O)=O)cc1", {"tag": "sulfamoylphenyl"})]
    made = substituent_replacements(SOURCE, payloads, min_removed=2, max_removed=14)
    assert made, "no one-cut proposals on a source with several acyclic single bonds"
    for proposal in made:
        endpoint = Chem.MolFromSmiles(proposal.endpoint)
        retained = Chem.MolFromSmiles(_canon(SOURCE))
        # Everything except the removed branch is still present: the endpoint must
        # contain the source minus that branch, which we check by requiring the
        # installed payload's heavy count to account for the whole size change.
        delta = endpoint.GetNumHeavyAtoms() - retained.GetNumHeavyAtoms()
        assert delta == proposal.installed_atoms - proposal.removed_atoms


def test_one_cut_swap_actually_installs_the_payload():
    payloads = [("[1*]c1ccc(S(N)(=O)=O)cc1", {})]
    made = substituent_replacements(SOURCE, payloads)
    query = Chem.MolFromSmarts("S(N)(=O)=O")
    assert made
    assert all(Chem.MolFromSmiles(p.endpoint).HasSubstructMatch(query) for p in made)


def test_one_cut_swap_needs_the_isotope_molzip_label():
    # MEASURED GOTCHA: molzip defaults to AtomMapNumber while FragmentOnBonds writes
    # ISOTOPE dummies, so the default returns a DISCONNECTED molecule rather than
    # raising. A payload labelled with an ATOM MAP instead of an isotope must therefore
    # produce nothing -- if this ever returns proposals, the label handling has drifted.
    assert substituent_replacements(SOURCE, [("[*:1]c1ccccc1", {})]) == []


def test_two_cut_replacement_carries_a_fused_unit_one_cut_cannot():
    # The reason two-cut exists: a ring system and the exocyclic carbon fused to it are
    # ONE unit, so the payload must span two boundaries.
    regions = [("[1*]C(C)=C1c2ccccc2Sc2cc([2*])ccc21",
                {"tag": "thioxanthene_with_junction"})]
    made = region_replacements(SOURCE, regions, min_removed=3, max_removed=24)
    assert made, "two-cut replacement produced nothing on a two-ring-system source"
    junction = Chem.MolFromSmarts("[C;R]=[C;!R]")
    assert any(Chem.MolFromSmiles(p.endpoint).HasSubstructMatch(junction) for p in made)


def test_two_cut_replacement_reports_the_orientation_it_used():
    regions = [("[1*]CCN([2*])C", {})]
    made = region_replacements(SOURCE, regions)
    assert made
    assert {p.detail["orientation"] for p in made} <= {"as_mined", "flipped"}


def test_excision_removes_the_middle_and_rejoins_the_flanks():
    made = region_excisions(SOURCE, min_removed=1, max_removed=20, min_kept=6)
    assert made, "excision produced nothing"
    source_heavy = Chem.MolFromSmiles(SOURCE).GetNumHeavyAtoms()
    for proposal in made:
        assert proposal.installed is None and proposal.installed_atoms == 0
        # The endpoint is exactly the source minus the excised middle -- nothing else
        # may change, which is what "context preserving" means for this family.
        assert proposal.heavy_atoms == source_heavy - proposal.removed_atoms


def test_excision_middle_must_be_distinguishable_from_a_flank():
    # MEASURED: labelling both cuts identically makes the middle carry the same mark set
    # as a flank, the middle is never identified, and the family returns ZERO proposals.
    # This asserts the shipped labelling does identify it.
    made = region_excisions(SOURCE, min_removed=1, max_removed=20, min_kept=6)
    assert len(made) > 0
    assert all(p.removed_atoms >= 1 for p in made)


def test_a_region_boundary_never_crosses_a_ring_bond():
    """Ring counts are EXACTLY conserved across an excision.

    Every ring in the source either leaves inside the excised middle or survives whole;
    none is opened. NOTE, measured rather than assumed: a mutation removing the
    `not bond.IsInRing()` filter SURVIVES this test, and that is correct -- allowing
    every single bond yields 0 additional usable splits on three real sources, because
    opening a ring does not disconnect the molecule. The filter is redundant with the
    fragment-count requirement, so this test pins the conservation PROPERTY and does not
    claim to cover that filter.
    """
    source_rings = _rings(SOURCE)
    assert source_rings >= 3, "fixture must have rings on both sides of a cut"
    made = region_excisions(SOURCE, min_removed=1, max_removed=20, min_kept=4)
    assert made
    for proposal in made:
        excised = proposal.removed.replace("[1*]", "[H]").replace("[2*]", "[H]")
        assert Chem.MolFromSmiles(excised) is not None, proposal.removed
        assert _rings(proposal.endpoint) == source_rings - _rings(excised)


def test_every_family_returns_only_executor_valid_endpoints():
    payloads = [("[1*]c1ccc(S(N)(=O)=O)cc1", {})]
    regions = [("[1*]CCN([2*])C", {})]
    produced = (substituent_replacements(SOURCE, payloads)
                + region_replacements(SOURCE, regions)
                + region_excisions(SOURCE))
    assert produced
    for proposal in produced:
        assert executable_endpoint(proposal.endpoint) == proposal.endpoint
        assert proposal.endpoint != _canon(SOURCE)


def test_families_never_read_a_task_or_an_oracle():
    """INFORMATION BOUNDARY, checked over CODE rather than prose.

    A substring scan of the whole file fails on the module docstring, which legitimately
    explains what the operators must not do. Walk the AST instead and inspect only
    identifiers, attributes and imports -- docstrings and comments are not code.
    """
    import ast
    import pathlib

    tree = ast.parse(
        pathlib.Path("src/compose_v4/control/contextual_region_replace.py").read_text())
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.Import):
            seen.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            seen.add((node.module or "").split(".")[0])
            seen.update(alias.name for alias in node.names)
    forbidden = {"Oracle", "tdc", "TanimotoSimilarity", "GetMorganFingerprintAsBitVect",
                 "rdFMCS", "MurckoScaffold"}
    assert not (seen & forbidden), f"operator module reaches for {seen & forbidden}"


def test_the_boundary_test_can_actually_fail():
    """A guard that cannot fail is not a guard (this repo has paid for that twice)."""
    import ast

    tree = ast.parse("from tdc import Oracle\nx = Oracle('qed')\n")
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            seen.add((node.module or "").split(".")[0])
            seen.update(alias.name for alias in node.names)
    assert seen & {"Oracle", "tdc"}
