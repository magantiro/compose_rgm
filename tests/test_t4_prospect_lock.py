"""Guards for the prospected candidate lock.

A lock exists to be fixed before scores are seen. The failures that matter are a lock
that can be edited afterwards, one that charges more than it is allowed, one that
spends a call on a molecule already docked, and a selection that lets a score in.
"""

from __future__ import annotations

import pytest

from compose_v4.experiments.t4_prospect_assessment import endpoint_sha256
from compose_v4.experiments.t4_prospect_lock import SCHEMA_VERSION, build_lock, select, verify_lock


def _row(endpoint, primitives, families=("append_ring",)):
    return {
        "endpoint": endpoint,
        "primitives": primitives,
        "families": list(families),
        "changed_originals": 6,
        "created": 14,
        "properties": {"oracle_eligible": True},
    }


def _lock(selections=None, **overrides):
    kwargs = {
        "call_limit": 20,
        "authorization_ceiling": 180,
        "docking_seed": 1701,
        "required_rdkit": "2024.03.5",
        "oracle_protocols": {"parp1_0": {"delta": 0.4}},
        "input_sha256": {"pool": "a" * 64},
    }
    kwargs.update(overrides)
    if selections is None:
        selections = {"parp1_0": select([_row("CCO", 20)], set(), take=1)}
    return build_lock(selections, **kwargs)


# ---- Selection ----


def test_an_already_docked_endpoint_is_not_charged_again():
    pool = [_row("CCO", 22), _row("CCN", 20)]
    chosen = select(pool, {endpoint_sha256("CCO")}, take=5)
    assert chosen["rediscovered_excluded"] == 1
    assert [row["endpoint"] for row in chosen["take"]] == ["CCN"]


def test_selection_is_ordered_by_size_and_is_deterministic():
    pool = [_row("CCN", 12), _row("CCO", 22), _row("CCC", 18)]
    first = select(pool, set(), take=3)
    second = select(list(reversed(pool)), set(), take=3)
    assert [row["primitives"] for row in first["take"]] == [22, 18, 12]
    assert first["take"] == second["take"]


def test_the_diversity_cap_stops_one_construction_shape_taking_the_budget():
    pool = [_row(f"C{'C' * i}O", 20) for i in range(10)]
    chosen = select(pool, set(), take=10, per_signature=3)
    assert len(chosen["take"]) == 3
    assert chosen["shortfall"] == 7


def test_distinct_shapes_are_each_allowed_their_cap():
    pool = [_row("CCO", 20, ("append_ring",)), _row("CCN", 20, ("fuse_ring",))]
    assert len(select(pool, set(), take=10, per_signature=1)["take"]) == 2


def test_a_selection_needs_a_positive_budget():
    for bad in ({"take": 0}, {"per_signature": 0}):
        with pytest.raises(ValueError, match="positive take"):
            select(
                [_row("CCO", 20)],
                set(),
                take=bad.get("take", 1),
                per_signature=bad.get("per_signature", 1),
            )


def test_selection_never_consults_a_score():
    """No pool row carries a score, and the rule must not acquire one."""
    chosen = select([_row("CCO", 20)], set(), take=1)
    assert all("score" not in key for row in chosen["take"] for key in row)
    assert "no docking score" in chosen["selection_rule"]


# ---- Lock ----


def test_a_sealed_lock_refuses_later_edits():
    lock = _lock()
    verify_lock(lock)
    lock["selections"]["parp1_0"]["take"][0]["endpoint"] = "CCN"
    with pytest.raises(ValueError, match="modified after it was sealed"):
        verify_lock(lock)


def test_a_lock_cannot_exceed_its_call_limit():
    selections = {
        "parp1_0": select(
            [_row(f"C{'C' * i}O", 20) for i in range(5)], set(), take=5, per_signature=5
        )
    }
    with pytest.raises(ValueError, match="against a limit"):
        _lock(selections, call_limit=2)


def test_a_call_limit_cannot_exceed_the_authorization_ceiling():
    with pytest.raises(ValueError, match="authorization ceiling"):
        _lock(call_limit=200, authorization_ceiling=180)


def test_every_charged_cell_must_have_a_bound_oracle_protocol():
    selections = {"jak2_1": select([_row("CCO", 20)], set(), take=1)}
    with pytest.raises(ValueError, match="no oracle protocol bound"):
        _lock(selections)


def test_a_lock_that_charges_one_molecule_twice_is_refused():
    lock = _lock()
    lock["selections"]["parp1_0"]["take"].append(dict(lock["selections"]["parp1_0"]["take"][0]))
    lock["charged_calls"] = 2
    lock["lock_id"] = __import__(
        "compose_v4.control.docking_value", fromlist=["identity"]
    ).identity({k: v for k, v in lock.items() if k != "lock_id"})
    with pytest.raises(ValueError, match="same endpoint twice"):
        verify_lock(lock)


def test_the_lock_records_that_retries_and_replacement_are_off():
    lock = _lock()
    assert lock["schema_version"] == SCHEMA_VERSION
    assert lock["automatic_retries"] == 0
    assert lock["replacement_after_scoring"] is False
    assert "not an IVG comparison" in lock["interpretation"]
