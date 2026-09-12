import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_plan_pool_lock import (
    Pool,
    role_queues,
    round_robin_pools,
    validate_lock,
)


def pool(source, score, phase, slot, suffix=""):
    products = tuple(f"{source}_{suffix}_{i}" for i in range(10))
    probabilities = tuple((i + 1) / 55 for i in range(10))
    releases = tuple(i / 10 for i in range(10))
    return Pool(
        worker_id=f"{source}_{phase}_{slot}_{suffix}",
        source=source,
        parent_score=score,
        phase=phase,
        slot=slot,
        products=products,
        probabilities=probabilities,
        releases=releases,
        plans={
            smiles: {"release": release} for smiles, release in zip(products, releases)
        },
        parent={},
    )


def test_round_robin_balances_strong_parent_structures():
    rows = [
        pool("a", 0.7, 1, 0, "x"),
        pool("a", 0.7, 2, 0, "y"),
        pool("b", 0.69, 1, 1, "x"),
        pool("weak", 0.4, 1, 2, "x"),
    ]
    assert [row.source for row in round_robin_pools(rows, 3)] == ["a", "b", "a"]


def test_role_queues_are_deterministic_disjoint_and_exclude_known():
    row = pool("a", 0.7, 1, 0)
    known = {row.products[0]}
    first_used, second_used = set(), set()
    first = role_queues(row, known, first_used)
    second = role_queues(row, known, second_used)
    assert first == second
    assert first_used == second_used
    assert row.products[0] not in first_used
    assert len(first_used) == 9
    assert set(first) == {"actor_top", "uniform_hash", "largest_release"}
    assert all(len(queue) == 3 for queue in first.values())
    assert first["actor_top"][0]["smiles"] == row.products[-1]
    flattened = [entry["smiles"] for queue in first.values() for entry in queue]
    assert len(flattened) == len(set(flattened))


def test_queue_lock_identity_fails_closed():
    body = {
        "schema_version": "pmo_plan_pool_queue_lock_v1",
        "roles": ["actor_top", "uniform_hash", "largest_release"],
        "queue_depth": 3,
        "new_oracle_calls": 0,
    }
    lock = {**body, "lock_sha256": identity(body), "analysis_commit": "abc"}
    validate_lock(lock)
    lock["queue_depth"] = 4
    with pytest.raises(ValueError, match="invalid or altered"):
        validate_lock(lock)
