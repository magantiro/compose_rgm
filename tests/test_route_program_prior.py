"""Guards for the offline teacher-route structural prior.

Every guard here is mutation-checked: the comment on each one names the
production change that turns it red. A test whose expectation is recomputed
from the code under test cannot fail, and this repository has been bitten by
exactly that twice -- a dense mask compared against itself, and a file-inventory
constant compared against a list derived from itself -- so none of these
rebuild their expectation from the object they are checking.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import bridge_separated_regions
from compose_v4.control.dynamic_program_synthesis import (
    GENERIC_MODULES,
    synthesize_dynamic_program,
)
from compose_v4.control.region_law_contract import (
    RegionLawNotConsumed,
    assert_region_law_is_consumed,
)
from compose_v4.control.route_program_prior import (
    REGION_FEATURE_NAMES,
    RULE_VOCABULARY,
    ProgramShape,
    RegionChoice,
    RouteProgramPrior,
    RouteProgramPriorError,
    assert_payload_is_address_free,
    fit_route_program_prior,
    normalize_emission_profile,
    parent_context,
    parent_features,
    region_role_features,
)

_LEAD = "FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)NCCN1CCOCC1)c2)cc3)ccc4Cl"
_SLOTS = 48


def _state(smiles: str = _LEAD):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


def _emissions() -> dict:
    return {
        "substituent_delete": {"atom_delete": 10},
        "segment_grow": {"atom_insert": 10},
        "append_ring": {"atom_insert": 6, "cycle_close": 1},
        "heteroatom_substitute": {"atom_restate_semantic": 10},
        "cycle_close": {"cycle_close": 10},
    }


def _choice(graph, chosen: int, source_id: str = "a" * 64) -> RegionChoice:
    support = bridge_separated_regions(graph, maximum=None)
    features = np.stack([region_role_features(graph, r) for r in support])
    return RegionChoice(
        source_id=source_id,
        context=parent_context(graph),
        features=features[chosen],
        support=features,
        chosen=chosen,
    )


def _shape(source_id: str = "a" * 64) -> ProgramShape:
    return ProgramShape(
        source_id=source_id,
        context=("large", "open"),
        module_count=2,
        rule_counts={"atom_delete": 7, "atom_insert": 3},
        primitive_count=10,
    )


def _prior(choices=None, shapes=None) -> RouteProgramPrior:
    graph = _state()
    return fit_route_program_prior(
        choices if choices is not None else [_choice(graph, 0)],
        shapes if shapes is not None else [_shape()],
        training_identity="test",
        emission_profile=_emissions(),
    )


# ---- The law is a re-ranking, never a filter -----------------------------


def test_order_returns_the_whole_support_not_a_top_k():
    """MUTATION: truncating `order` to a top-k turns this red.

    Support equality is what makes the learned and uniform arms comparable; a
    law that deleted part of the support would be measured against a different
    question.
    """

    graph = _state()
    prior = _prior()
    support = bridge_separated_regions(graph, maximum=None)
    ordered = prior.order(graph, np.random.default_rng(0))
    assert len(ordered) == len(support)
    assert {(r.fragment, r.anchor) for r in ordered} == {
        (r.fragment, r.anchor) for r in support
    }


def test_the_support_floor_binds_and_is_what_preserves_the_support():
    """MUTATION: `return np.exp(scores)` -- dropping `np.maximum(self.floor, ...)`
    -- turns this red.

    An earlier version of this guard asserted only `min(weight) > 0`, which the
    un-floored law satisfies too, so it could not fail. The property that
    matters is that the floor BINDS: under a tilt sharp enough to drive some
    region's unfloored weight below it, that region is held AT the floor rather
    than being driven toward zero. Checked with a deliberately extreme
    coefficient vector so the condition is reached by construction rather than
    by luck of the fitted values.
    """

    graph = _state()
    fitted = _prior()
    sharp = RouteProgramPrior(
        region_coefficients=np.full(len(REGION_FEATURE_NAMES), -40.0),
        region_feature_names=REGION_FEATURE_NAMES,
        module_count=fitted.module_count,
        rule_table=fitted.rule_table,
        emission_profile=fitted.emission_profile,
        training_sources=fitted.training_sources,
        training_identity="sharp",
    )
    regions = bridge_separated_regions(graph, maximum=None)
    weights = sharp.region_weights(graph, regions)
    assert len(weights) == len(regions)
    assert float(weights.min()) == pytest.approx(sharp.floor)
    assert float(np.count_nonzero(weights <= sharp.floor)) >= 1.0
    # and the whole support still gets drawn, which is the point of flooring
    ordered = sharp.order(graph, np.random.default_rng(1))
    assert len(ordered) == len(regions)


def test_every_region_keeps_strictly_positive_weight_under_the_fitted_law():
    graph = _state()
    regions = bridge_separated_regions(graph, maximum=None)
    weights = _prior().region_weights(graph, regions)
    assert len(weights) == len(regions)
    assert float(weights.min()) > 0.0


def test_support_is_uncapped_and_exceeds_the_v1_eight_atom_cap():
    """The v1 law cannot draw a coherent 9-15 atom excision at all."""

    graph = _state()
    uncapped = bridge_separated_regions(graph, maximum=None)
    capped = bridge_separated_regions(graph, maximum=8)
    assert len(uncapped) > len(capped)
    assert max(len(r.fragment) for r in uncapped) > 8


# ---- The production draw site actually consults it -----------------------


def test_the_production_draw_site_consumes_this_prior_as_a_law():
    """MUTATION: renaming `RouteProgramPrior.order` turns this red.

    The check RUNS the production synthesis path rather than inspecting a
    signature, because a keyword can exist and be dropped one hop later.
    """

    graph = _state()

    def draw(law, seed):
        return synthesize_dynamic_program(
            graph, np.random.default_rng(seed), max_modules=1, region_law=law
        )

    assert assert_region_law_is_consumed(draw, attempts=16) >= 1


def test_consumption_check_fails_when_the_law_is_not_threaded():
    """NEGATIVE CONTROL: a draw that ignores the law must be caught."""

    graph = _state()

    def unwired(law, seed):
        return synthesize_dynamic_program(
            graph, np.random.default_rng(seed), max_modules=1
        )

    with pytest.raises(RegionLawNotConsumed):
        assert_region_law_is_consumed(unwired, attempts=4)


# ---- Roles, not source fingerprints --------------------------------------


def test_region_features_are_computed_on_the_parent_being_proposed_for():
    """The same structural role on two different molecules scores the same.

    This is the property whose absence made the v1 PMO plan representation
    untransferable: 97.7% of its refusals were at depth 0 because no atom of the
    production parent carried a descriptor copied from the teacher's molecule.
    """

    small = _state("CCCCc1ccccc1")
    features = region_role_features(small, bridge_separated_regions(small)[0])
    assert len(features) == len(REGION_FEATURE_NAMES)
    assert np.isfinite(features).all()
    # relative_size is a fraction of THIS parent, so it is scale-free
    assert 0.0 < features[0] <= 1.0


def test_payload_refuses_a_smiles_or_an_atom_index():
    """MUTATION: deleting a branch of `assert_payload_is_address_free` turns
    one of these red. A prior that can store a teacher molecule will eventually
    store one."""

    payload = _prior().to_payload()
    assert_payload_is_address_free(payload)

    leaked = json.loads(json.dumps(payload))
    leaked["training_sources"] = [_LEAD]
    with pytest.raises(RouteProgramPriorError):
        assert_payload_is_address_free(leaked)

    leaked = json.loads(json.dumps(payload))
    leaked["rule_table"]["large/open"] = {"atom_at_slot_17": 1.0}
    with pytest.raises(RouteProgramPriorError):
        assert_payload_is_address_free(leaked)

    leaked = json.loads(json.dumps(payload))
    leaked["region_feature_names"] = [*REGION_FEATURE_NAMES, "teacher_neighbourhood"]
    with pytest.raises(RouteProgramPriorError):
        assert_payload_is_address_free(leaked)

    leaked = json.loads(json.dumps(payload))
    leaked["module_count"][_LEAD] = [1.0]
    with pytest.raises(RouteProgramPriorError):
        assert_payload_is_address_free(leaked)


def test_payload_round_trip_preserves_the_law():
    graph = _state()
    prior = _prior()
    restored = RouteProgramPrior.from_payload(prior.to_payload())
    regions = bridge_separated_regions(graph, maximum=None)
    np.testing.assert_allclose(
        prior.region_weights(graph, regions), restored.region_weights(graph, regions)
    )
    assert restored.rule_probabilities(graph) == prior.rule_probabilities(graph)


def test_a_coefficient_vector_cannot_be_reinterpreted_against_another_order():
    """MUTATION: dropping the feature-order check lets a stale vector load."""

    payload = _prior().to_payload()
    payload["region_feature_names"] = list(reversed(REGION_FEATURE_NAMES))
    with pytest.raises(RouteProgramPriorError):
        RouteProgramPrior.from_payload(payload)


# ---- The fit learns something, and the right something -------------------


def test_the_conditional_logit_recovers_a_planted_preference():
    """A teacher that always takes the LARGEST region must give log_size > 0.

    The expectation is planted by construction and is not recomputed from the
    fitted object, so a fit that learned nothing fails.
    """

    graph = _state()
    support = bridge_separated_regions(graph, maximum=None)
    largest = int(np.argmax([len(r.fragment) for r in support]))
    choices = [_choice(graph, largest) for _ in range(40)]
    prior = fit_route_program_prior(
        choices, [_shape()], training_identity="planted", emission_profile=_emissions()
    )
    index = REGION_FEATURE_NAMES.index("log_size")
    assert prior.region_coefficients[index] > 0.0
    weights = prior.region_weights(graph, support)
    assert int(np.argmax(weights)) == largest


def test_family_weights_follow_the_measured_emission_matrix():
    """A teacher vocabulary of pure `cycle_close` must prefer ring closure.

    The expectation comes from the emission matrix supplied to the test, not
    from the projection code, so a projection that ignored the rule table fails.
    """

    graph = _state()
    prior = _prior(
        shapes=[
            ProgramShape(
                source_id="b" * 64,
                context=parent_context(graph),
                module_count=1,
                rule_counts={"cycle_close": 500},
                primitive_count=500,
            )
        ]
    )
    weights = prior.family_weights(graph)
    assert weights["cycle_close"] > weights["heteroatom_substitute"]
    assert weights["cycle_close"] > weights["segment_grow"]
    assert min(weights.values()) > 0.0, "a projection must never delete a family"


def test_sampled_families_are_production_families_of_the_declared_length():
    graph = _state()
    prior = _prior()
    families = prior.sample_family_sequence(
        graph, np.random.default_rng(3), module_count=3
    )
    assert len(families) == 3
    assert set(families) <= set(GENERIC_MODULES)


def test_emission_profile_refuses_an_unknown_rule_or_an_empty_row():
    with pytest.raises(RouteProgramPriorError):
        normalize_emission_profile({"segment_grow": {"not_a_rule": 1}})
    with pytest.raises(RouteProgramPriorError):
        normalize_emission_profile({"segment_grow": {}})
    normalized = normalize_emission_profile({"segment_grow": {"atom_insert": 4}})
    assert set(normalized["segment_grow"]) == set(RULE_VOCABULARY)
    assert normalized["segment_grow"]["atom_insert"] == pytest.approx(1.0)


# ---- Parent conditioning -------------------------------------------------


def test_parent_context_separates_capacity_and_size():
    big = _state()
    small = _state("CCCCc1ccccc1")
    assert parent_context(small)[0] == "small"
    assert parent_context(big) != parent_context(small)
    assert parent_features(big)["heavy_atoms"] > parent_features(small)["heavy_atoms"]
    assert parent_features(small)["capacity_headroom"] > 0.0


def test_module_count_and_rule_tables_always_have_a_pooled_fallback():
    """MUTATION: removing the '*' bucket turns this red -- an unseen parent
    class must back off, never raise inside a proposal loop."""

    prior = _prior()
    exotic = _state("C1CCCCC1")
    assert len(prior.module_count_probabilities(exotic)) > 0
    assert sum(prior.rule_probabilities(exotic).values()) == pytest.approx(1.0)
