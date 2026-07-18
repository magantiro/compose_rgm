import numpy as np
import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, is_element, smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.tracelet_rate_model import TraceletRateModel
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.ring_junctions import describe_ring_transaction
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
from compose_v4.rewrite.tracelet_fiber import enumerate_tracelet_cnof_fiber
from compose_v4.rewrite.tracelets import CycleAttach
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import (
    RING_SYSTEM_ELECTRONIC_ALIAS_VERSION,
    RingSystemElectronicAliasVersionError,
    TypedRingCatalog,
    build_typed_ring_catalog,
    build_typed_ring_catalog_from_paths,
    require_ring_system_electronic_aliases,
)


@pytest.mark.parametrize(
    "smiles,expected_families",
    [
        ("C1CCCCC1", ("cycle_insert",)),
        ("c1ncccc1", ("cycle_insert",)),
        ("c1ccc2ccccc2c1", ("cycle_insert", "ring_ear_insert")),
        ("C1CC2CCC1C2", ("cycle_insert", "ring_ear_insert")),
        ("C1CCC2(CC1)CCCC2", ("cycle_insert", "ring_ear_insert")),
        ("c1ccccc1-c2ccccc2", ("cycle_insert", "cycle_attach")),
    ],
)
def test_typed_ring_tracelets_reconstruct_ring_topologies(
    smiles: str,
    expected_families: tuple[str, ...],
) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)
    trace = compile_null_to_target_tracelets(target, typed_ring_payloads=True)
    assert trace.metadata["compiler"] == "typed_block_ear_tracelets_v1"
    assert tuple(step.rule_name for step in trace.steps) == expected_families
    assert canonical_state_key(TraceProgressCTMC(trace).states[-1]) == canonical_state_key(
        target
    )


def test_typed_training_catalog_contains_each_teacher_successor() -> None:
    targets = tuple(
        pad_molecular_graph(smiles_to_molecular_graph(text), 16)
        for text in (
            "C1CCCCC1",
            "c1ccccc1",
            "c1ncccc1",
            "c1ccc2ccccc2c1",
            "c1ccccc1-c2ccccc2",
        )
    )
    proposal_traces = tuple(
        compile_null_to_target_tracelets(target, typed_ring_payloads=True)
        for target in targets
    )
    catalog = build_typed_ring_catalog(
        proposal_traces,
        max_cycle_templates=32,
        max_ear_templates=32,
    )
    for target in targets:
        trace = compile_null_to_target_tracelets(
            target,
            typed_ring_payloads=True,
            ring_catalog=catalog,
        )
        path = TraceProgressCTMC(trace)
        for progress in range(path.path_length):
            fiber = enumerate_tracelet_cnof_fiber(
                path.states[progress],
                ring_catalog=catalog,
            )
            teacher_key = canonical_state_key(path.states[progress + 1])
            assert teacher_key in {
                transition.successor_key for transition in fiber.transitions
            }


def test_direct_typed_tracelet_materialization_matches_verified_runtime() -> None:
    targets = tuple(
        pad_molecular_graph(smiles_to_molecular_graph(text), 16)
        for text in (
            "c1ccc2ccccc2c1",
            "C1CCC2(CC1)CCCC2",
            "c1ccccc1-c2ccccc2",
        )
    )
    traces = tuple(
        compile_null_to_target_tracelets(target, typed_ring_payloads=True)
        for target in targets
    )
    catalog = build_typed_ring_catalog(
        traces,
        max_cycle_templates=16,
        max_attach_templates=16,
        max_ear_templates=16,
    )
    runtime = de_novo_rewrite_system()
    checked = set()
    for trace in traces:
        for state in TraceProgressCTMC(trace).states:
            fiber = enumerate_tracelet_cnof_fiber(state, ring_catalog=catalog)
            for family in ("cycle_insert", "cycle_attach", "ring_ear_insert"):
                for transition in fiber.by_family[family][:2]:
                    expected = runtime.apply(state, family, transition.action)
                    assert np.array_equal(
                        transition.successor.atom_types,
                        expected.atom_types,
                    )
                    assert np.array_equal(transition.successor.bonds, expected.bonds)
                    assert transition.successor_key == canonical_state_key(expected)
                    checked.add(family)
    assert checked == {"cycle_insert", "cycle_attach", "ring_ear_insert"}


def test_linked_ring_enters_closed_in_one_attachment_event() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccccc1CCc2ccccc2"),
        16,
    )
    trace = compile_null_to_target_tracelets(target, typed_ring_payloads=True)
    path = TraceProgressCTMC(trace)
    attachment_index = next(
        index for index, step in enumerate(trace.steps) if isinstance(step.action, CycleAttach)
    )
    before = path.states[attachment_index]
    after = path.states[attachment_index + 1]
    action = trace.steps[attachment_index].action
    new_slots = {int(atom.slot) for atom in action.atoms}
    assert all(not bool(is_element(before.atom_types[v])) for v in new_slots)
    assert all(bool(is_element(after.atom_types[v])) for v in new_slots)
    assert len(new_slots) >= 3
    for atom in action.atoms:
        v = int(atom.slot)
        cycle_neighbors = sum(
            int(after.bonds[v, int(other.slot)] != 0)
            for other in action.atoms
            if int(other.slot) != v
        )
        assert cycle_neighbors == 2


def test_acyclic_prefix_cannot_be_reinterpreted_by_scalar_or_ear_closure() -> None:
    proposal_targets = tuple(
        pad_molecular_graph(smiles_to_molecular_graph(text), 16)
        for text in (
            "c1ccccc1",
            "c1ccc2ccccc2c1",
            "c1ccccc1-c2ccccc2",
        )
    )
    proposal_traces = tuple(
        compile_null_to_target_tracelets(target, typed_ring_payloads=True)
        for target in proposal_targets
    )
    catalog = build_typed_ring_catalog(
        proposal_traces,
        max_cycle_templates=32,
        max_attach_templates=32,
        max_ear_templates=32,
    )
    chain = pad_molecular_graph(smiles_to_molecular_graph("CCC"), 16)
    fiber = enumerate_tracelet_cnof_fiber(chain, ring_catalog=catalog)
    assert not fiber.by_family["bond_insert"]
    assert not fiber.by_family["ring_ear_insert"]
    assert fiber.by_family["cycle_attach"]


@pytest.mark.parametrize(
    "smiles,expected_kind",
    [
        ("c1ccc2ccccc2c1", "fused"),
        ("C1CC2CCC1C2", "bridged"),
        ("C1CCC2(CC1)CCCC2", "spiro"),
        ("c1ccccc1-c2ccccc2", "attached"),
    ],
)
def test_ring_transactions_expose_chemical_junction_type(
    smiles: str,
    expected_kind: str,
) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)
    trace = compile_null_to_target_tracelets(target, typed_ring_payloads=True)
    path = TraceProgressCTMC(trace)
    kinds = {
        describe_ring_transaction(path.states[index], step.action).kind
        for index, step in enumerate(trace.steps)
        if step.rule_name in {"cycle_insert", "cycle_attach", "ring_ear_insert"}
    }
    assert expected_kind in kinds


def test_superposed_rate_model_adds_enabled_family_hazards() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 16)
    trace = compile_null_to_target_tracelets(target, typed_ring_payloads=True)
    catalog = build_typed_ring_catalog(
        (trace,),
        max_cycle_templates=8,
        max_ear_templates=8,
    )
    state = TraceProgressCTMC(trace).states[0]
    fiber = enumerate_tracelet_cnof_fiber(state, ring_catalog=catalog)
    model = TraceletRateModel(
        hidden_dim=16,
        message_passing_steps=1,
        rate_factorization="superposed",
        ring_catalog=catalog,
    ).eval()
    prediction = model.predict_tracelet_fiber(state, 0.25, fiber=fiber)
    assert np.isfinite(float(prediction.total_hazard))
    assert float(prediction.total_hazard) > 0.0
    assert bool((prediction.marked_rates >= 0).all())
    assert float(prediction.total_hazard) == pytest.approx(
        float(prediction.marked_rates.sum())
    )


def _tree_ring_path(smiles: str, seed: int) -> TraceProgressCTMC:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 20)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(seed),
        n_slots=20,
    )
    return TraceProgressCTMC(
        compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )
    )


def test_path_catalog_retains_paired_heteroaromatic_electronics() -> None:
    paths = (
        _tree_ring_path("c1ncccc1", 1201),
        _tree_ring_path("c1ccc2[nH]ccc2c1", 1202),
    )
    catalog = build_typed_ring_catalog_from_paths(
        paths,
        max_ring_system_templates=64,
    )
    items = require_ring_system_electronic_aliases(catalog)
    ring_observations = sum(
        step.rule_name == "ring_system_grow"
        for path in paths
        for step in path.trace.steps
    )
    nitrogen = int(ELEMENT_TO_IDX["N"])

    assert catalog.ring_system_electronic_alias_version == (
        RING_SYSTEM_ELECTRONIC_ALIAS_VERSION
    )
    assert sum(count for _, count in items) == ring_observations
    assert any(
        any(atom_type == nitrogen for atom_type, _, _ in alias.target_atoms)
        for alias, _ in items
    )
    assert any(
        any(atom_type == nitrogen and hydrogens == 1 for atom_type, _, hydrogens in alias.target_atoms)
        for alias, _ in items
    )
    assert all(len(alias.target_atoms) == alias.pattern.span for alias, _ in items)


def test_repeated_ring_electronic_alias_observations_merge() -> None:
    path = _tree_ring_path("c1ncccc1", 1211)
    single = build_typed_ring_catalog_from_paths(
        (path,),
        max_ring_system_templates=32,
    )
    repeated = build_typed_ring_catalog_from_paths(
        (path, path),
        max_ring_system_templates=32,
    )
    single_items = require_ring_system_electronic_aliases(single)
    repeated_items = require_ring_system_electronic_aliases(repeated)

    assert len(single_items) == len(repeated_items) == 1
    assert repeated_items[0][0] == single_items[0][0]
    assert repeated_items[0][1] == 2 * single_items[0][1]
    assert sum(repeated.ring_system_template_counts) == (
        2 * sum(single.ring_system_template_counts)
    )


def test_legacy_ring_catalog_fails_loudly_in_exact_electronic_mode() -> None:
    legacy = TypedRingCatalog((), (), ())
    with pytest.raises(RingSystemElectronicAliasVersionError, match="rebuild"):
        require_ring_system_electronic_aliases(legacy)

    rebuilt_empty = build_typed_ring_catalog_from_paths(())
    assert require_ring_system_electronic_aliases(rebuilt_empty) == ()
