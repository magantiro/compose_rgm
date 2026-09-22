"""Bounded protection of a declared macro-option bridge: clock, floor, reservation.

Zero oracle calls and no controller: these exercise the library's invariants directly
so a failure names the invariant rather than a campaign.
"""

import numpy as np
import pytest

from compose_v4.control.pmo_macro_option import (
    BRIDGE_CHARGED,
    DECLARED,
    EXPIRED,
    PROTECTION_ROUNDS_CEILING,
    REACHED,
    MacroOption,
    MacroOptionRegistry,
    MacroOptionStage,
    option_identity,
    protected_parent_weights,
    reserve_continuation_slots,
)


def _option(*, protection_rounds=1, stage_primitives=(14, 14), ceiling=23, origin="CCO"):
    chain = ["CCN", "CCCN", "CCCCN"][: len(stage_primitives)]
    stages, parent = [], origin
    for index, (endpoint, primitives) in enumerate(zip(chain, stage_primitives, strict=True)):
        stages.append(MacroOptionStage(index, parent, endpoint, primitives))
        parent = endpoint
    return MacroOption(
        option_id=option_identity(origin, chain),
        origin_endpoint=origin,
        stages=tuple(stages),
        protection_rounds=protection_rounds,
        declared_at_round=1,
        realization_ceiling=ceiling,
    )


def _registry(option):
    registry = MacroOptionRegistry()
    registry.declare(option, {"stages": [{} for _ in option.stages]})
    return registry


# ---- Declaration ----


def test_declaration_refuses_a_macro_one_round_could_realize():
    # A macro inside the realization ceiling needs no staging and no protection;
    # declaring it would manufacture a success the mechanism did not earn.
    with pytest.raises(ValueError, match="realization ceiling"):
        _option(stage_primitives=(4, 5), ceiling=23)


def test_declaration_refuses_an_unbounded_or_absent_window():
    for rounds in (0, -1, PROTECTION_ROUNDS_CEILING + 1, 1.0, None):
        with pytest.raises(ValueError, match="bounded explicit round count"):
            _option(protection_rounds=rounds)


def test_declaration_refuses_a_broken_stage_chain():
    with pytest.raises(ValueError, match="contiguous chain"):
        MacroOption(
            option_id="x",
            origin_endpoint="CCO",
            stages=(
                MacroOptionStage(0, "CCO", "CCN", 14),
                MacroOptionStage(1, "SOMETHING_ELSE", "CCCN", 14),
            ),
            protection_rounds=1,
            declared_at_round=1,
            realization_ceiling=23,
        )


def test_declaration_requires_one_executed_record_per_stage():
    registry = MacroOptionRegistry()
    with pytest.raises(ValueError, match="executed record per stage"):
        registry.declare(_option(), {"stages": [{}]})


# ---- Clock ----


def test_a_one_round_window_covers_exactly_the_round_after_the_bridge_is_charged():
    option = _option(protection_rounds=1)
    registry = _registry(option)
    registry.advance_round(1)
    assert registry.status(option.option_id) == DECLARED
    assert not registry.protected_bridges()
    registry.note_charged(option.bridge_endpoint, 1)
    assert registry.status(option.option_id) == BRIDGE_CHARGED
    registry.advance_round(2)
    assert registry.protected_bridges() == {option.bridge_endpoint: option.option_id}
    registry.advance_round(3)
    assert not registry.protected_bridges()
    assert registry.status(option.option_id) == EXPIRED


def test_a_two_round_window_covers_exactly_two_rounds():
    option = _option(protection_rounds=2)
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged(option.bridge_endpoint, 1)
    for round_index in (2, 3):
        registry.advance_round(round_index)
        assert registry.protected_bridges(), round_index
    registry.advance_round(4)
    assert not registry.protected_bridges()


def test_reaching_the_destination_releases_protection_immediately():
    option = _option()
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged(option.bridge_endpoint, 1)
    registry.advance_round(2)
    registry.note_charged(option.destination_endpoint, 2)
    assert registry.status(option.option_id) == REACHED
    assert not registry.protected_bridges()


def test_expiry_is_terminal_and_a_later_charge_cannot_revive_it():
    option = _option()
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged(option.bridge_endpoint, 1)
    registry.advance_round(3)
    assert registry.status(option.option_id) == EXPIRED
    registry.note_charged(option.bridge_endpoint, 4)
    assert registry.status(option.option_id) == EXPIRED
    assert not registry.protected_bridges()


def test_the_clock_never_runs_backwards():
    registry = _registry(_option())
    registry.advance_round(4)
    with pytest.raises(ValueError, match="backwards"):
        registry.advance_round(3)


def test_a_repeated_advance_inside_one_round_does_not_shorten_the_window():
    option = _option(protection_rounds=1)
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged(option.bridge_endpoint, 1)
    for _ in range(5):
        registry.advance_round(2)
    assert registry.protected_bridges()


# ---- Durable state ----


def test_restore_carries_an_open_window_so_a_charged_bridge_is_not_stranded():
    option = _option(protection_rounds=2)
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged(option.bridge_endpoint, 1)
    registry.advance_round(2)
    resumed = MacroOptionRegistry.restore(registry.payload())
    assert resumed.protected_bridges() == {option.bridge_endpoint: option.option_id}
    assert resumed.construction(option.option_id)["stages"]
    resumed.advance_round(3)
    assert resumed.protected_bridges()
    resumed.advance_round(4)
    assert not resumed.protected_bridges()


def test_restore_refuses_a_window_lengthened_in_the_payload():
    option = _option(protection_rounds=1)
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged(option.bridge_endpoint, 1)
    payload = registry.payload()
    payload["records"][0]["protected_through_round"] += 1
    with pytest.raises(ValueError, match="exceeds its declaration"):
        MacroOptionRegistry.restore(payload)


def test_restore_refuses_a_window_lengthened_on_a_partly_walked_chain():
    # The horizon moves with the frontier, so the guard must validate against the round
    # the CURRENT window opened -- not against the first crossing.
    option = _chain(["B1", "B2", "DEST"])
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged("B1", 1)
    registry.advance_round(2)
    registry.note_charged("B2", 2)
    payload = registry.payload()
    assert MacroOptionRegistry.restore(payload).next_stage_index(option.option_id) == 2
    payload["records"][0]["protected_through_round"] += 1
    with pytest.raises(ValueError, match="exceeds its declaration"):
        MacroOptionRegistry.restore(payload)


def test_restore_refuses_a_foreign_schema():
    registry = _registry(_option())
    payload = registry.payload()
    payload["schema_version"] = "something_else_v9"
    with pytest.raises(ValueError, match="foreign schema"):
        MacroOptionRegistry.restore(payload)


def test_an_absent_payload_restores_an_empty_registry():
    assert MacroOptionRegistry.restore(None).report()["declared"] == 0


# ---- Parent-mass floor ----


def _weights(values):
    values = np.asarray(values, dtype=float)
    return values / values.sum()


def test_the_floor_lifts_a_starved_bridge_to_exactly_the_declared_share():
    keys = ["a", "b", "c", "d"]
    weights = _weights([0.9, 0.05, 0.04, 0.01])
    endpoint_of = {"a": "A", "b": "B", "c": "C", "d": "BRIDGE"}
    lifted, detail = protected_parent_weights(
        keys, weights, endpoint_of, {"BRIDGE": "opt"}, floor=0.25
    )
    assert detail["applied"] is True
    assert detail["mass_before"] == pytest.approx(0.01)
    assert lifted[3] == pytest.approx(0.25)
    assert lifted.sum() == pytest.approx(1.0)


def test_the_floor_is_a_reranking_and_never_a_filter():
    # Every entry the ordinary policy could draw keeps positive mass, and their
    # relative order among themselves is untouched.
    keys = ["a", "b", "c", "d"]
    weights = _weights([0.6, 0.3, 0.09, 0.01])
    endpoint_of = {"a": "A", "b": "B", "c": "C", "d": "BRIDGE"}
    lifted, _ = protected_parent_weights(keys, weights, endpoint_of, {"BRIDGE"}, floor=0.25)
    assert (lifted > 0).all()
    assert lifted[0] / lifted[1] == pytest.approx(weights[0] / weights[1])


def test_a_bridge_already_above_the_floor_is_left_untouched():
    keys = ["a", "b"]
    weights = _weights([0.4, 0.6])
    lifted, detail = protected_parent_weights(
        keys, weights, {"a": "A", "b": "BRIDGE"}, {"BRIDGE"}, floor=0.25
    )
    assert detail["applied"] is False
    assert lifted.tolist() == weights.tolist()


def test_no_protected_entry_leaves_the_distribution_identical():
    keys = ["a", "b"]
    weights = _weights([0.4, 0.6])
    lifted, detail = protected_parent_weights(keys, weights, {"a": "A", "b": "B"}, set(), floor=0.25)
    assert detail["protected_entries"] == 0
    assert lifted.tolist() == weights.tolist()


def test_the_floor_must_be_a_proper_fraction():
    for floor in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="strictly inside"):
            protected_parent_weights(["a"], np.asarray([1.0]), {"a": "A"}, {"A"}, floor=floor)


# ---- Allocation reservation ----


def _row(candidate_id):
    return {"candidate_id": candidate_id}


def test_a_reserved_continuation_displaces_from_the_tail_not_the_head():
    chosen = [_row(f"c{i}") for i in range(4)]
    pool = chosen + [_row("reserved")]
    merged, detail = reserve_continuation_slots(chosen, pool, {"reserved"}, limit=4)
    assert [row["candidate_id"] for row in merged] == ["reserved", "c0", "c1", "c2"]
    assert detail["reserved_added"] == 1 and detail["displaced"] == 1
    assert len(merged) == 4


def test_a_reservation_never_exceeds_the_authorized_batch():
    chosen = [_row("c0")]
    pool = chosen + [_row(f"r{i}") for i in range(6)]
    merged, detail = reserve_continuation_slots(
        chosen, pool, {f"r{i}" for i in range(6)}, limit=3
    )
    assert len(merged) == 3 and detail["reserved_added"] == 3


def test_a_continuation_the_ordinary_policy_already_chose_is_not_double_counted():
    chosen = [_row("reserved"), _row("c1")]
    merged, detail = reserve_continuation_slots(chosen, chosen, {"reserved"}, limit=2)
    assert merged == chosen
    assert detail["reserved_added"] == 0 and detail["reserved_already_chosen"] == 1


def test_no_reservation_leaves_the_ordinary_choice_identical():
    chosen = [_row("c0"), _row("c1")]
    merged, detail = reserve_continuation_slots(chosen, chosen, set(), limit=2)
    assert merged == chosen and detail["reserved_added"] == 0


# ---- Multi-leg chains ----


def _chain(endpoints, *, protection_rounds=1, primitives=12, ceiling=23, origin="CCO"):
    stages, parent = [], origin
    for index, endpoint in enumerate(endpoints):
        stages.append(MacroOptionStage(index, parent, endpoint, primitives))
        parent = endpoint
    return MacroOption(
        option_id=option_identity(origin, list(endpoints)),
        origin_endpoint=origin,
        stages=tuple(stages),
        protection_rounds=protection_rounds,
        declared_at_round=1,
        realization_ceiling=ceiling,
    )


def test_a_chain_costs_one_bounded_window_per_crossing():
    option = _chain(["B1", "B2", "DEST"])
    assert option.bridge_endpoints == ("B1", "B2")
    assert option.total_protected_rounds_budget == 2
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged("B1", 1)
    registry.advance_round(2)
    assert registry.protected_bridges() == {"B1": option.option_id}
    registry.note_charged("B2", 2)
    registry.advance_round(3)
    # The window moved with the frontier; exactly one state is protected at a time.
    assert registry.protected_bridges() == {"B2": option.option_id}
    registry.note_charged("DEST", 3)
    assert registry.status(option.option_id) == REACHED
    assert not registry.protected_bridges()


def test_the_frontier_only_moves_forward():
    option = _chain(["B1", "B2", "DEST"])
    registry = _registry(option)
    registry.advance_round(1)
    # Charging a later stage before its predecessor must not skip ahead.
    registry.note_charged("DEST", 1)
    assert registry.status(option.option_id) == DECLARED
    registry.note_charged("B1", 1)
    registry.note_charged("B1", 1)
    assert registry.next_stage_index(option.option_id) == 1


def test_a_chain_longer_than_the_declared_maximum_is_refused():
    with pytest.raises(ValueError, match="2 to 4 stages"):
        _chain([f"S{i}" for i in range(5)])


def test_a_chain_that_expires_mid_way_is_terminal():
    option = _chain(["B1", "B2", "DEST"])
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged("B1", 1)
    registry.advance_round(3)
    assert registry.status(option.option_id) == EXPIRED
    registry.note_charged("B2", 3)
    assert registry.status(option.option_id) == EXPIRED
    assert not registry.protected_bridges()


def test_restore_preserves_the_frontier_of_a_partly_walked_chain():
    option = _chain(["B1", "B2", "DEST"])
    registry = _registry(option)
    registry.advance_round(1)
    registry.note_charged("B1", 1)
    registry.advance_round(2)
    registry.note_charged("B2", 2)
    resumed = MacroOptionRegistry.restore(registry.payload())
    assert resumed.next_stage_index(option.option_id) == 2
    resumed.advance_round(3)
    assert resumed.protected_bridges() == {"B2": option.option_id}
