"""Guards for parent-first structural program construction.

The property under test is NOT "a program was produced".  It is that the program
produced is the one the intent DEMANDED -- same excision size, really consuming parent
atoms -- because the failure mode this module exists to detect is a proposer that
quietly shrinks a transformation until it binds and then reports a 1.000 retained
fraction as a success.

Every test drives the LIVE functions over real drug-like molecules.  Nothing compares
the module against a transcription of itself: the excision endpoint is cross-checked
against ``bridge_region_law.excise_region``'s INDEPENDENT closed form, and the
realized programs are replayed by the production executor.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import (
    BridgeRegionLaw,
    RegionRealizationError,
    bridge_separated_regions,
    excise_region,
)
from compose_v4.control.parent_first_program import (
    ABSTAIN_NO_REGION_IN_BAND,
    ABSTENTION_REASONS,
    PLAN_SCHEMA_VERSION,
    ProgramIntent,
    SizeBandRegionLaw,
    _deletion_schedule,
    construct_parent_first_program,
    excision_accounting,
    intent_of_plan,
    plan_of_execution,
    roles_of_execution,
)
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.kernel import canonical_state_key

# Production PMO parents carry n_atoms == 48 (measured, unanimous over all 226 parents
# of the completed 3x250 run).  A TIGHT graph has no free slot and therefore no
# atom_insert support at all, which alone makes every construction unrealizable --
# so the slot count is part of the contract, not an incidental fixture detail.
PRODUCTION_SLOTS = 48

# Real, bridge-rich drug-like molecules.
LEADS = (
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C(=O)N(C)c2ncn(C)c2C1=O",
    "O=C(Nc1ccc(cc1)S(=O)(=O)N)c1ccccc1",
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
)


def _parent(smiles: str = LEADS[0], slots: int = PRODUCTION_SLOTS):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)


def _intent(excise: int, insert: int = 2) -> ProgramIntent:
    return ProgramIntent(
        excise_atoms=excise,
        insert_atoms=insert,
        elements=("C", "N", "O"),
        source_plan_id="fixture",
        primitive_count=excise + insert,
    )


# ---- the no-shrinking invariant ----


def test_a_realized_region_always_lies_inside_the_demanded_band():
    """The whole point: the program realized is the one the intent asked for."""

    realized = 0
    for smiles in LEADS:
        source = _parent(smiles)
        for excise in (2, 3, 4, 5, 6):
            intent = _intent(excise)
            result = construct_parent_first_program(
                source, intent, np.random.default_rng(11)
            )
            if not result["realized"]:
                assert result["reason"] in ABSTENTION_REASONS
                continue
            realized += 1
            assert result["region_size"] == excise, (
                "a realized region outside the demanded band means the constructor "
                "shrank the transformation to make it bind"
            )
            assert result["demanded_band"] == [excise, excise]
    assert realized >= 5, "fixture must realize for this guard to discriminate"


def test_a_parent_with_no_region_in_the_band_abstains_rather_than_shrinking():
    source = _parent(LEADS[1])
    biggest = max((r.size for r in bridge_separated_regions(source)), default=0)
    intent = _intent(biggest + 5)
    result = construct_parent_first_program(source, intent, np.random.default_rng(3))
    assert not result["realized"]
    assert result["reason"] == ABSTAIN_NO_REGION_IN_BAND
    assert "plan" not in result and "actions" not in result


def test_the_size_band_law_refuses_regions_below_its_floor():
    source = _parent()
    law = SizeBandRegionLaw(minimum=4, maximum=6)
    sizes = {region.size for region in law.regions(source)}
    assert sizes, "fixture must carry regions in this band"
    assert min(sizes) >= 4 and max(sizes) <= 6
    # And the floor is what a plain BridgeRegionLaw does NOT impose.
    unbounded = {region.size for region in BridgeRegionLaw(maximum=6).regions(source)}
    assert min(unbounded) < 4


def test_an_empty_size_band_is_refused_at_construction():
    with pytest.raises(ValueError):
        SizeBandRegionLaw(minimum=6, maximum=3)


# ---- excision is real, and the two retention measures are separated ----


def test_a_realized_excision_actually_deletes_parent_atoms():
    source = _parent(LEADS[2])
    found = []
    for excise in (3, 4, 5, 6):
        result = construct_parent_first_program(
            source, _intent(excise, insert=1), np.random.default_rng(5)
        )
        if result["realized"]:
            found.append(result)
    assert found, "fixture must realize for this guard to discriminate"
    for result in found:
        assert len(result["deleted_parent_slots"]) == result["region_size"]
        assert result["true_retained_fraction"] < 1.0, (
            "an excision-demanding program that retains every parent atom did not "
            "perform the excision it declared"
        )


def test_excision_accounting_separates_kept_from_deleted_and_refilled():
    """The controller's slot rule cannot make this distinction; this one must."""

    source = _parent()
    # A construction large enough to refill the slots the excision freed is exactly
    # the case where the two measures must diverge.
    disagreeing = None
    for excise in (2, 3, 4):
        result = construct_parent_first_program(
            source, _intent(excise, insert=excise + 2), np.random.default_rng(17)
        )
        if result["realized"] and result["retention_measures_disagree"]:
            disagreeing = result
            break
    assert disagreeing is not None, "fixture must produce a refilled slot"
    assert disagreeing["retained_fraction"] > disagreeing["true_retained_fraction"], (
        "the controller's measure must read HIGHER when slots were refilled"
    )
    assert disagreeing["refilled_slots"]
    assert set(disagreeing["refilled_slots"]) <= set(disagreeing["deleted_parent_slots"])


def test_excision_accounting_is_computed_from_the_actions_not_from_the_endpoint():
    """A deleted-then-refilled slot is still DELETED; the endpoint alone cannot say."""

    source = _parent()
    result = None
    for excise in (2, 3, 4):
        candidate = construct_parent_first_program(
            source, _intent(excise, insert=excise + 2), np.random.default_rng(17)
        )
        if candidate["realized"] and candidate["retention_measures_disagree"]:
            result = candidate
            break
    assert result is not None
    endpoint, _ = execute_program(source, result["actions"])
    accounting = excision_accounting(source, endpoint, result["actions"])
    # Recomputing with an EMPTY action list is the discriminating control: the
    # endpoint is unchanged, so an endpoint-only rule returns the same answer while
    # the action-derived one correctly reports no deletion.
    blind = excision_accounting(source, endpoint, [])
    assert accounting.true_retained_fraction < blind.true_retained_fraction
    assert blind.deleted_parent_slots == ()


# ---- the execution is the production executor's, not a closed form ----


def test_the_deletion_schedule_endpoint_matches_the_independent_closed_form():
    """Executor replay vs ``excise_region``: two derivations, one answer."""

    checked = 0
    for smiles in LEADS:
        source = _parent(smiles)
        for region in bridge_separated_regions(source, maximum=6)[:8]:
            try:
                closed = excise_region(source, region)
            except RegionRealizationError:
                continue
            try:
                actions, executed = _deletion_schedule(
                    source, region, np.random.default_rng(2)
                )
            except ValueError:
                # A region whose atoms admit no leaf deletion order (a fused ring, say)
                # is refused by the executor, which is the constructor's own signal to
                # try the next region.  The invariant under test is that a schedule
                # which SUCCEEDS agrees with the closed form, not that all regions
                # delete.
                continue
            assert canonical_state_key(executed) == canonical_state_key(closed)
            replayed, _ = execute_program(source, actions)
            assert canonical_state_key(replayed) == canonical_state_key(executed)
            checked += 1
    assert checked >= 10, "fixture must exercise several regions"


def test_a_realized_program_replays_through_the_production_executor():
    source = _parent(LEADS[3])
    result = None
    for excise in (3, 4, 5):
        candidate = construct_parent_first_program(
            source, _intent(excise), np.random.default_rng(23)
        )
        if candidate["realized"]:
            result = candidate
            break
    assert result is not None
    endpoint, receipt = execute_program(source, result["actions"])
    assert canonical_state_key(endpoint) == result["endpoint_key"]
    assert list(receipt["states"]) == result["states"]
    assert len(result["states"]) == len(result["actions"]) + 1
    assert result["primitive_count"] == len(result["actions"])


# ---- the emitted object is a v1-shaped program, not a look-alike ----


def test_the_emitted_plan_is_the_v1_representation_with_its_own_declared_conditions():
    source = _parent()
    result = construct_parent_first_program(
        source, _intent(4, insert=2), np.random.default_rng(31)
    )
    assert result["realized"]
    plan = result["plan"]
    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert plan["primitive_count"] == len(plan["roles"]) == len(result["actions"])
    assert plan["exact_replay"] and plan["complete_representation_supported"]
    # Roles are the PINNED supervision function's output over the realized execution.
    assert roles_of_execution(source, result["actions"]) == plan["roles"]
    # And the plan is content-addressed by its roles, as the v1 library's are.
    assert plan_of_execution(source, result["actions"])["plan_id"] == plan["plan_id"]


def test_an_intent_carries_no_parent_address_or_neighbourhood_fingerprint():
    """The intent is what makes the program a SCHEMA rather than a template."""

    roles = [
        {
            "executor_rule": "atom_delete",
            "model_family": "atom_delete",
            "parameters": {},
            "operands": [
                {
                    "role": "atom",
                    "descriptor": {
                        "origin": "preexisting",
                        "atom_type": 0,
                        "formal_charge": 0,
                        "neighbor_element_histogram": {"C": 3},
                        "created_ordinal": None,
                        "creation_lag": None,
                    },
                }
            ],
            "created_handle_dependencies": [],
        },
        {
            "executor_rule": "atom_insert",
            "model_family": "atom_insert",
            "parameters": {"atom_type": 0, "formal_charge": 0, "implicit_hydrogens": 3},
            "operands": [],
            "created_handle_dependencies": [],
        },
    ]
    intent = intent_of_plan(
        {"plan_id": "p", "roles": roles, "primitive_count": 2, "component_count": 1}
    )
    assert intent.excise_atoms == 1 and intent.insert_atoms == 1
    body = repr(intent)
    for leaked in ("neighbor_element_histogram", "created_ordinal", "creation_lag", "descriptor"):
        assert leaked not in body


# ---- the state-semantics contract ----


def test_a_tight_graph_has_no_insertion_capacity_and_the_constructor_says_so():
    """The 48-slot contract, stated as a behaviour rather than an assertion."""

    tight = smiles_to_molecular_graph(LEADS[0])
    assert int(tight.n_atoms) == int(is_element(tight.atom_types).sum())
    padded = _parent(LEADS[0])
    assert int(padded.n_atoms) == PRODUCTION_SLOTS
    assert int(padded.n_atoms) > int(is_element(padded.atom_types).sum())

    intent = _intent(3, insert=3)
    tight_result = construct_parent_first_program(tight, intent, np.random.default_rng(1))
    padded_result = construct_parent_first_program(padded, intent, np.random.default_rng(1))
    # The tight graph frees slots by deleting, so it is not categorically barred; what
    # it cannot do is build MORE than it removed.
    grow = _intent(2, insert=8)
    assert not construct_parent_first_program(tight, grow, np.random.default_rng(1))["realized"]
    assert tight_result["realized"] or padded_result["realized"]
