"""Invariants for the connected mutable-region representation."""

import json
from pathlib import Path

import pytest

from compose_v4.control.region import Region, enumerate_regions, scope_summary

SEEDS = [
    "Cc1c[nH]nc1-c1ccc(-c2cn(C)nc2-c2ccncc2)cc1",   # braf dev0
    "CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3",             # parp1 s0
    "c1ccccc1",                                      # benzene: ring only
    "CCO",                                           # tiny acyclic
]


def _mol(smi):
    from rdkit import Chem
    return Chem.MolFromSmiles(smi)


@pytest.mark.parametrize("smi", SEEDS)
def test_regions_are_connected(smi):
    mol = _mol(smi)
    for r in enumerate_regions(smi):
        atoms = set(r.atoms)
        start = next(iter(atoms))
        seen, stack = set(), [start]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            for nb in mol.GetAtomWithIdx(v).GetNeighbors():
                if nb.GetIdx() in atoms:
                    stack.append(nb.GetIdx())
        assert seen == atoms, f"disconnected region {sorted(atoms)} in {smi}"


@pytest.mark.parametrize("smi", SEEDS)
def test_boundary_bonds_actually_cross(smi):
    mol = _mol(smi)
    for r in enumerate_regions(smi):
        for inside, outside, _order in r.boundary:
            assert inside in r.atoms
            assert outside not in r.atoms
            assert mol.GetBondBetweenAtoms(int(inside), int(outside)) is not None


@pytest.mark.parametrize("smi", SEEDS)
def test_kind_matches_arity(smi):
    expect = {0: "whole", 1: "substituent", 2: "linker"}
    for r in enumerate_regions(smi):
        assert r.kind == expect.get(r.arity, "multi")


@pytest.mark.parametrize("smi", SEEDS)
def test_enumeration_is_deterministic(smi):
    """Required: a scope proposal over this set must be likelihood-evaluable."""
    a = [r.key() for r in enumerate_regions(smi)]
    b = [r.key() for r in enumerate_regions(smi)]
    assert a == b
    assert len(a) == len(set(a)), "duplicate regions leak multiplicity into Q_scope"


@pytest.mark.parametrize("smi", SEEDS[:2])
def test_scale_axis_spans_local_to_global(smi):
    """The whole point: not another mostly-local editor."""
    regs = enumerate_regions(smi)
    fr = [r.released_fraction for r in regs]
    assert min(fr) <= 0.10, "no genuinely local region"
    assert max(fr) >= 0.60, "no genuinely large-scope region"
    assert sum(1 for f in fr if f > 0.5) >= 5, "large-scope regions are too rare"


@pytest.mark.parametrize("smi", SEEDS[:2])
def test_constructive_cases_present(smi):
    """One-cut and two-cut are the first two compilers we will build."""
    z = scope_summary(enumerate_regions(smi))
    assert z["n_substituent"] >= 1
    assert z["n_linker"] >= 1


def test_region_never_covers_whole_molecule():
    for smi in SEEDS:
        mol = _mol(smi)
        for r in enumerate_regions(smi):
            assert r.size < mol.GetNumAtoms()


def test_unparseable_smiles_returns_empty():
    assert enumerate_regions("not_a_molecule") == []


@pytest.mark.parametrize("smi", SEEDS)
def test_interface_agrees_with_context_connectivity(smi):
    """Deletion safety is arity AND context connectivity, not arity alone."""
    for r in enumerate_regions(smi):
        if r.arity == 1:
            assert r.interface == "pendant"
        elif r.arity == 2:
            assert r.interface == ("segment" if r.n_context_components <= 1
                                   else "bridge")
        elif r.arity >= 3:
            assert r.interface == "multi"


def test_bridge_regions_really_would_disconnect():
    """A 'bridge' must be one whose removal splits the preserved context."""
    smi = SEEDS[0]
    mol = _mol(smi)
    n = mol.GetNumAtoms()
    seen_bridge = False
    for r in enumerate_regions(smi):
        if r.interface != "bridge":
            continue
        seen_bridge = True
        context = set(range(n)) - set(r.atoms)
        start = next(iter(context))
        reach, stack = set(), [start]
        while stack:
            v = stack.pop()
            if v in reach:
                continue
            reach.add(v)
            for nb in mol.GetAtomWithIdx(v).GetNeighbors():
                if nb.GetIdx() in context:
                    stack.append(nb.GetIdx())
        assert reach != context, "labelled bridge but context stays connected"
    assert seen_bridge, "no bridge regions found to check"


def test_changed_fraction_is_zero_for_identity():
    from compose_v4.control.region import actual_changed_fraction
    assert actual_changed_fraction(SEEDS[0], SEEDS[0]) == pytest.approx(0.0, abs=1e-9)
