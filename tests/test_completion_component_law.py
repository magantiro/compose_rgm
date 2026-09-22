"""Guards for the PMO completion scale/content repair.

The repair is inert unless a production caller passes it, so these tests check
CONSUMPTION on the production path, not the existence of a keyword, and every
guard below is mutation-proven in
``scripts/pmo_completion_mutation_battery.py``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    ALLOWED_VALENCES,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_valid_state, pad_molecular_graph
from compose_v4.control.completion_component_law import (
    OCTAVE_BANDS,
    ComponentBank,
    CompletionLaw,
    ScaleBalancedRegionLaw,
    ScaleLaw,
    extract_components,
    install_component_actions,
    octave_bands,
)
from compose_v4.control.completion_law_contract import (
    CompletionLawContractError,
    assert_completion_law_is_consumed,
    completion_law_identity,
    load_bank,
    resolve_completion_law,
)
from compose_v4.control.dynamic_program_synthesis import (
    MAX_SEGMENT_LENGTH,
    compile_generic_module,
    synthesize_dynamic_program,
)
from compose_v4.control.dynamic_program_synthesis_v2 import (
    synthesize_structured_program,
)
from compose_v4.rewrite.trace_shard import decode_state

REPO = Path(__file__).resolve().parents[1]
INIT = REPO / "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
SLOTS = 48


def _parents():
    lock = json.loads(INIT.read_text())
    return [decode_state(row["state"]) for row in lock["candidates"]]


def _padded(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)


# ---- Scale ---------------------------------------------------------------


def test_v1_scale_never_exceeds_eight_and_expanded_scale_does():
    rng = np.random.default_rng(0)
    bounded = ScaleLaw(maximum=MAX_SEGMENT_LENGTH)
    expanded = ScaleLaw(maximum=None)
    bounded_draws = [bounded.draw(rng, 20) for _ in range(400)]
    expanded_draws = [expanded.draw(rng, 20) for _ in range(400)]
    assert max(bounded_draws) <= MAX_SEGMENT_LENGTH
    # The whole point of the repair: mass above v1's ceiling, and at the scale
    # 58.3% of productive transitions need.
    assert max(expanded_draws) > MAX_SEGMENT_LENGTH
    assert sum(1 for v in expanded_draws if v >= 13) / len(expanded_draws) > 0.05


def test_octave_bands_are_capacity_bounded_and_equal_mass():
    assert octave_bands(0) == ()
    assert octave_bands(1) == ((1, 1),)
    assert all(high <= 7 for _low, high in octave_bands(7))
    rng = np.random.default_rng(1)
    draws = [ScaleLaw(maximum=None).draw(rng, 32) for _ in range(6000)]
    assert max(draws) <= 32
    per_band = [
        sum(1 for v in draws if low <= v <= high) / len(draws)
        for low, high in octave_bands(32)
    ]
    # Equal probability per octave, so no band may collapse or dominate.
    assert min(per_band) > 0.5 / len(per_band)
    assert max(per_band) < 2.0 / len(per_band)


def test_region_law_preserves_support_and_reaches_beyond_the_v1_cap():
    graph = _parents()[0]
    law = ScaleBalancedRegionLaw()
    regions = law.regions(graph)
    weights = law.weights(graph, regions)
    assert len(regions) == len(weights) and len(regions) > 0
    # A re-ranking, never a filter: every drawable region keeps positive mass.
    assert float(weights.min()) > 0.0
    assert max(region.size for region in regions) > MAX_SEGMENT_LENGTH


# ---- Component installation ---------------------------------------------


def test_component_round_trips_through_the_production_executor():
    donor = _padded("Cc1ccc(N2CCOCC2)cc1")
    components = extract_components(donor)
    assert components
    host = _padded("CCCC")
    anchor = int(np.flatnonzero(is_element(host.atom_types))[0])
    installed = 0
    for spec in components:
        if spec.size < 4:
            continue
        try:
            actions, product = install_component_actions(host, anchor, spec)
        except Exception:  # noqa: BLE001 - the declared refusal scope
            continue
        installed += 1
        assert is_valid_state(product)
        assert product.n_real_atoms == host.n_real_atoms + spec.size
        assert len(actions) >= spec.size
        assert molecular_graph_to_smiles(product) is not None
    assert installed > 0


def test_installed_component_carries_its_ring_and_branching():
    bank = load_bank()
    host = _padded("CCCC")
    anchor = int(np.flatnonzero(is_element(host.atom_types))[0])
    rings = 0
    for spec in bank.components:
        if not (spec.has_ring and spec.branch_points and 6 <= spec.size <= 12):
            continue
        actions, product = install_component_actions(host, anchor, spec)
        # A ring needs at least one non-tree edge, which is a cycle_close.
        closures = [a for a in actions if a.get("executor_rule") == "cycle_close"]
        assert closures, "a ring-bearing component installed with no ring closure"
        smiles = molecular_graph_to_smiles(product)
        assert smiles is not None and "1" in smiles
        rings += 1
        if rings >= 5:
            break
    assert rings >= 5


def test_reserved_hydrogen_keeps_every_intermediate_valence_allowed():
    """The install schedule's invariant: valence is constant, so each state is real."""

    bank = load_bank()
    host = _padded("CCCC")
    anchor = int(np.flatnonzero(is_element(host.atom_types))[0])
    for spec in bank.components[:40]:
        actions, _product = install_component_actions(host, anchor, spec)
        current = host
        from compose_v4.experiments.whole_ring_plan import execute_program

        for record in actions:
            current, _ = execute_program(current, [record])
            assert is_valid_state(current)
        for slot in np.flatnonzero(is_element(current.atom_types)):
            slot = int(slot)
            element = ALLOWED_VALENCES[
                __import__(
                    "compose_v4.chem.molecular_graph", fromlist=["IDX_TO_ELEMENT"]
                ).IDX_TO_ELEMENT[int(current.atom_types[slot])]
            ]
            valence = int(current.implicit_h_counts[slot]) + int(
                current.bonds[slot].sum()
            )
            assert valence in element


# ---- The bank ------------------------------------------------------------


def test_bank_is_objective_blind_and_richer_than_a_linear_chain():
    bank = load_bank()
    assert len(bank.components) > 200
    provenance = bank.provenance
    assert provenance["oracle_calls"] == 0
    assert provenance["task_identities_read"] == 0
    assert "PMO_INIT_BANK" in provenance["donor_path"]
    ring = sum(1 for spec in bank.components if spec.has_ring)
    branched = sum(1 for spec in bank.components if spec.branch_points)
    large = sum(1 for spec in bank.components if spec.size >= 13)
    # The census requirement is median 11, 80.6% ring-bearing, 36.1% >= 13.
    assert ring / len(bank.components) > 0.6
    assert branched / len(bank.components) > 0.5
    assert large / len(bank.components) > 0.25
    # Charge is preserved: a charged component would move the net formal charge.
    assert all(not any(spec.charges) for spec in bank.components)


def test_bank_identity_is_content_addressed():
    bank = load_bank()
    payload = json.loads((REPO / "configs/pmo_completion_component_bank_v1.json").read_text())
    assert bank.identity_sha256 == payload["bank_identity_sha256"]
    mutated = ComponentBank(
        components=bank.components[:-1], provenance=bank.provenance
    )
    assert mutated.identity_sha256 != bank.identity_sha256


def test_bank_order_respects_capacity_and_prefers_the_target_size():
    bank = load_bank()
    rng = np.random.default_rng(3)
    ordered = bank.order_for(rng, target=11, capacity=12)
    assert ordered
    assert all(spec.size <= 12 for spec in ordered)
    assert ordered[0].size == min(
        (s for s in bank.sizes if s <= 12), key=lambda s: (abs(s - 11), s)
    )
    assert bank.order_for(rng, target=11, capacity=0) == []


# ---- The contract --------------------------------------------------------


def test_absent_field_is_off_and_an_unregistered_name_is_refused():
    assert resolve_completion_law(None) is None
    assert completion_law_identity(None)["state"] == "v1_verbatim"
    with pytest.raises(CompletionLawContractError):
        resolve_completion_law("uniform_v1")
    with pytest.raises(CompletionLawContractError):
        resolve_completion_law(8)


@pytest.mark.parametrize(
    "arm,maximum,banked,excision",
    [
        ("scale_only_v1", None, False, True),
        ("content_only_v1", MAX_SEGMENT_LENGTH, True, False),
        ("scale_content_v1", None, True, True),
    ],
)
def test_each_registered_arm_resolves_to_its_declared_axes(arm, maximum, banked, excision):
    law = resolve_completion_law(arm)
    assert law.scale.maximum == maximum
    assert (law.bank is not None) is banked
    assert law.expand_excision is excision
    assert (law.region_law is not None) is excision


def test_off_does_not_touch_the_v1_excision_bound():
    """v1 draws its excision uniformly over fragments of at most eight atoms."""

    import compose_v4.control.dynamic_program_synthesis as module

    seen = {}
    original = module._delete_pendant_fragment

    def spy(source, rng, *, maximum=MAX_SEGMENT_LENGTH, law=None):
        seen["maximum"], seen["law"] = maximum, law
        return original(source, rng, maximum=maximum, law=law)

    module._delete_pendant_fragment = spy
    try:
        compile_generic_module(
            _parents()[0], np.random.default_rng(5), "segment_replace"
        )
        assert seen == {"maximum": MAX_SEGMENT_LENGTH, "law": None}
        seen.clear()
        compile_generic_module(
            _parents()[0],
            np.random.default_rng(5),
            "segment_replace",
            completion_law=resolve_completion_law("scale_content_v1"),
        )
        assert seen["law"] is not None
        assert seen["maximum"] > MAX_SEGMENT_LENGTH
    finally:
        module._delete_pendant_fragment = original


# ---- Consumption on the production path ----------------------------------


def _shallow_draw(law, rng):
    return synthesize_dynamic_program(
        _parents()[0],
        rng,
        max_modules=3,
        max_primitives=32,
        max_blocks=8,
        completion_law=law,
    )


def _structured_draw(law, rng):
    return synthesize_structured_program(
        _parents()[0],
        rng,
        max_modules=3,
        max_primitives=32,
        max_blocks=8,
        completion_law=law,
    )


def test_both_production_lanes_consult_the_completion_law():
    for draw in (_shallow_draw, _structured_draw):
        receipt = assert_completion_law_is_consumed(draw)
        assert receipt["consumed"] is True


def test_a_dropped_keyword_is_refused_by_the_consumption_check():
    def dropping(law, rng):
        return synthesize_dynamic_program(
            _parents()[0], rng, max_modules=3, max_primitives=32, max_blocks=8
        )

    with pytest.raises(CompletionLawContractError):
        assert_completion_law_is_consumed(dropping)


def _forwarding_call_sites(path: Path, callee: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callee
    ]


@pytest.mark.parametrize(
    "module_path,callee,method",
    [
        ("src/compose_v4/control/dynamic_program_synthesis.py",
         "synthesize_dynamic_program", "_mutate"),
        ("src/compose_v4/control/dynamic_program_synthesis_v2.py",
         "synthesize_structured_program", "_mutate"),
        ("src/compose_v4/control/dynamic_program_synthesis_v21.py",
         "synthesize_structured_program", "_channel_proposal"),
    ],
)
def test_optimizer_call_sites_forward_the_law(module_path, callee, method):
    """Derived from the CALL SITE, so it cannot agree with a stale constant."""

    tree = ast.parse((REPO / module_path).read_text())
    target = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == method
    ]
    assert target, f"{method} not found in {module_path}"
    calls = [
        node
        for function in target
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callee
    ]
    assert calls, f"{method} does not call {callee}"
    for call in calls:
        assert any(
            keyword.arg == "completion_law" for keyword in call.keywords
        ), f"{module_path}:{method} calls {callee} without forwarding completion_law"


def test_the_pmo_controller_accepts_the_arm_on_both_construction_and_restore():
    """A flag the constructor takes and ``restore`` drops resumes arm B as arm A."""

    import inspect

    from compose_v4.control.pmo_population_controller import PmoPopulationController

    constructor = inspect.signature(PmoPopulationController.__init__).parameters
    restore = inspect.signature(PmoPopulationController.restore).parameters
    for name in ("enable_online_memory", "completion_law"):
        assert name in constructor and name in restore


@pytest.mark.parametrize("family", ["segment_grow", "segment_replace"])
def test_each_completion_family_consults_the_law_on_its_own(family):
    """Two hops share one sink, so one consultation test is not enough.

    Dropping the law from ``segment_replace`` alone leaves a whole-program
    consultation check GREEN because ``segment_grow`` still threads it. This is
    the near-miss the region-law wiring hit, so each family is checked alone.
    """

    def draw(law, rng):
        return compile_generic_module(
            _parents()[0], rng, family, completion_law=law
        )

    receipt = assert_completion_law_is_consumed(draw)
    assert receipt["consumed"] is True


def test_the_completion_law_actually_installs_a_ring_through_segment_replace():
    """A behavioural guard: consultation is not the same as taking effect."""

    law = resolve_completion_law("scale_content_v1")
    parents = _parents()
    installed_ring = 0
    for index, graph in enumerate(parents):
        for draw in range(6):
            rng = np.random.default_rng(991 * index + draw)
            try:
                _product, stage = compile_generic_module(
                    graph, rng, "segment_replace", completion_law=law
                )
            except ValueError:
                continue
            if stage["parameters"].get("has_ring"):
                installed_ring += 1
    assert installed_ring > 0, "no segment_replace completion installed a ring"


def test_the_completion_law_excises_past_the_v1_cap():
    law = resolve_completion_law("scale_content_v1")
    largest = 0
    for index, graph in enumerate(_parents()):
        for draw in range(8):
            rng = np.random.default_rng(3571 * index + draw)
            try:
                _product, stage = compile_generic_module(
                    graph, rng, "segment_replace", completion_law=law
                )
            except ValueError:
                continue
            largest = max(largest, int(stage["parameters"]["deleted_atoms"]))
    assert largest > MAX_SEGMENT_LENGTH, (
        "the expanded arm never excised past v1's eight-atom ceiling"
    )
