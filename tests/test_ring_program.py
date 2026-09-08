"""Small behavior checks for the parameterized option, using the real executor."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.option_continuation import (
    OptionContinuationKernel,
    OptionState,
    sample_option_trajectory,
)
from compose_v4.control.option_selector import OPTIONS, applicable_options, balanced_option_prior
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.control.ring_program import (
    RingProgress,
    RingSpec,
    construction_branches,
    real_slots,
    ring_spec,
)
from compose_v4.gates.med_chem_gate import is_valid
from compose_v4.rewrite.factorized_fiber import _factorized_candidates
from compose_v4.rewrite.kernel import editing_v2_rewrite_system


def fixture(spec):
    graph = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 48)
    slots = real_slots(graph)
    return OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), slots, (), "multi", 0),
        Lineage.initial(slots),
        spec.option,
        0,
        spec.horizon,
        "ring-fixture",
        ring_progress=RingProgress(),
    )


def uniform_law(graph):
    marks = tuple(
        _factorized_candidates(graph, allow_bond_reroute=True, vocabulary=ORGANIC_VOCABULARY)
    )
    return tuple(f for f, _ in marks), tuple(a for _, a in marks), (1 / len(marks),) * len(marks)


@pytest.mark.parametrize(
    "spec",
    [
        RingSpec("pendant", 5, (4, 1, 0), "aromatic"),
        RingSpec("pendant", 6, (5, 0, 1), "nonaromatic"),
        RingSpec("fused", 5, (4, 1, 0), "aromatic"),
        RingSpec("fused", 6, (5, 1, 0), "aromatic"),
    ],
)
def test_real_primitive_paths_build_requested_ring(spec):
    process = OptionContinuationKernel(
        uniform_law, editing_v2_rewrite_system(), max_executor_applications=256
    )
    node = fixture(spec)
    result = sample_option_trajectory(
        node,
        process,
        lambda _: 1,
        lambda _: 1,
        snapshot_id="uniform-engineering",
        seed=0,
        estimator="reference",
        max_expansions=0,
        max_terminal_evaluations=0,
    )
    assert result["status"] == "complete", result
    assert len(result["trace"]) == spec.horizon
    assert all(is_valid(step["after"]) for step in result["trace"])
    mol = Chem.MolFromSmiles(result["endpoint"])
    assert mol.GetNumAtoms() == 6 + spec.growth
    assert sorted(map(len, mol.GetRingInfo().AtomRings())) == sorted((6, spec.size))
    assert (
        rdMolDescriptors.CalcNumSpiroAtoms(mol) == rdMolDescriptors.CalcNumBridgeheadAtoms(mol) == 0
    )
    assert sum(a.GetAtomicNum() == 7 for a in mol.GetAtoms()) == spec.counts[1]
    assert sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) == spec.counts[2]


def test_defaults_prior_floor_and_region_applicability_are_preserved():
    option = RingSpec("fused", 5, (4, 1, 0), "aromatic").option
    assert option not in OPTIONS
    assert option not in applicable_options(["atom_insert"], [0])
    assert option in applicable_options(
        ["atom_insert"], [0], n_free_slots=3, ring_options=(option,)
    )
    assert option not in applicable_options(
        ["atom_insert"], [0], n_free_slots=2, ring_options=(option,)
    )
    names = ("generic", "grow", "cyclize", option)
    assert balanced_option_prior(names, exploration=0) == pytest.approx(
        [1 / 3, 1 / 3, 1 / 6, 1 / 6]
    )
    assert np.min(balanced_option_prior(names)) >= 0.1 / len(names)
    node = fixture(ring_spec(option))
    assert construction_branches(node.graph, {0}, ring_spec(option)) == ()


def test_progress_roundtrip_and_quota_identity():
    a = RingSpec("pendant", 5, (4, 1, 0), "aromatic")
    b = RingSpec("pendant", 5, (4, 0, 1), "aromatic")
    assert ring_spec(a.option) == a
    assert fixture(a).key() != fixture(b).key()
    assert RingProgress.from_payload(RingProgress().payload()) == RingProgress()
    with pytest.raises(ValueError, match="initial ring program"):
        replace(fixture(a), ring_progress=RingProgress((0,), (1, 2, 1, 2, 1), ()))
    with pytest.raises(ValueError, match="summing"):
        RingSpec("pendant", 5, (5, 1, 0), "aromatic")


def test_parameter_variants_share_only_physical_work():
    first = fixture(RingSpec("pendant", 5, (4, 1, 0), "nonaromatic"))
    second = fixture(RingSpec("pendant", 6, (5, 1, 0), "nonaromatic"))
    process = OptionContinuationKernel(
        uniform_law, editing_v2_rewrite_system(), max_executor_applications=256
    )
    row_a = process.row(first)
    calls = process.work.executor_applications
    row_b = process.row(second)
    assert row_a.successors and row_b.successors
    assert process.work.executor_applications == calls
    assert process.work.product_cache_hits > 0
    assert {node.key() for node in row_a.successors}.isdisjoint(
        node.key() for node in row_b.successors
    )
