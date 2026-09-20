"""Archive continuity across physically distinct bootstrap pools.

The failure this guards against is concrete: a recursive PMO campaign creates a new
physical cold-start pool when it revisits an initialization parent, and the Dynamic-v2.1
invariant refused the resulting measured program with "bootstrap candidates came from
different pools".  Both the live 250-call pilot and a free-oracle reproduction died at
round six that way, with the joint-credit allocation active AND inert, so the refusal is
a property of the pool bookkeeping rather than of the allocator.
"""

from __future__ import annotations

import inspect

import pytest

from compose_v4.control.bootstrap_pool_continuity import (
    SCHEMA_VERSION,
    BootstrapPoolContinuity,
)

POOL_A = "a" * 64
POOL_B = "b" * 64
POOL_C = "c" * 64


def _record(pool_id, *, shallow=1, structured=2):
    return {
        "endpoint": "CCO",
        "provenance": {
            "dynamic_v21_bootstrap": {
                "pool_id": pool_id,
                "shallow_rng": {"state": shallow},
                "structured_rng": {"state": structured},
            }
        },
    }


def test_a_second_physical_pool_is_absorbed_rather_than_refused():
    continuity = BootstrapPoolContinuity()
    first, new_first = continuity.adapt(_record(POOL_A))
    second, new_second = continuity.adapt(_record(POOL_B))

    assert new_first is True and new_second is True
    assert continuity.continuity_pool_id == POOL_A
    # The core invariant sees one identity...
    assert first["provenance"]["dynamic_v21_bootstrap"]["pool_id"] == POOL_A
    assert second["provenance"]["dynamic_v21_bootstrap"]["pool_id"] == POOL_A
    # ...while the physical identity survives as round provenance.
    assert second["provenance"]["dynamic_v21_bootstrap"]["generation_pool_id"] == POOL_B
    assert second["provenance"]["pmo_bootstrap_continuity"] == {
        "schema_version": SCHEMA_VERSION,
        "generation_pool_id": POOL_B,
        "continuity_pool_id": POOL_A,
        "boundary": "campaign_admission",
    }
    assert continuity.generation_pool_ids == [POOL_A, POOL_B]


def test_the_callers_record_is_never_mutated():
    continuity = BootstrapPoolContinuity()
    continuity.adapt(_record(POOL_A))
    original = _record(POOL_B)
    adapted, _ = continuity.adapt(original)
    assert original["provenance"]["dynamic_v21_bootstrap"]["pool_id"] == POOL_B
    assert "pmo_bootstrap_continuity" not in original["provenance"]
    assert adapted is not original


def test_only_the_first_sighting_of_a_pool_asks_for_its_sampler_state():
    continuity = BootstrapPoolContinuity()
    assert continuity.adapt(_record(POOL_A))[1] is True
    assert continuity.adapt(_record(POOL_A))[1] is False
    assert continuity.adapt(_record(POOL_B))[1] is True
    assert continuity.adapt(_record(POOL_B))[1] is False
    assert continuity.generation_pool_ids == [POOL_A, POOL_B]


def test_the_continuity_identity_is_the_first_pool_and_never_moves():
    continuity = BootstrapPoolContinuity()
    for pool in (POOL_A, POOL_B, POOL_C, POOL_B):
        continuity.adapt(_record(pool))
    assert continuity.continuity_pool_id == POOL_A
    assert continuity.payload()["physical_pools_absorbed"] == 3


def test_a_record_without_bootstrap_provenance_passes_through():
    continuity = BootstrapPoolContinuity()
    record = {"endpoint": "CCO", "provenance": {}}
    adapted, is_new = continuity.adapt(record)
    assert adapted is record and is_new is False
    assert continuity.continuity_pool_id is None


@pytest.mark.parametrize("pool_id", [None, "", "short", 12345, "z" * 63])
def test_a_missing_or_malformed_pool_identity_is_refused(pool_id):
    continuity = BootstrapPoolContinuity()
    with pytest.raises(ValueError, match="physical pool identity"):
        continuity.adapt(_record(pool_id))


def test_semantics_stay_bound_to_the_frozen_recovery_adapter():
    """The recovery adapter is pinned by a sealed contract and cannot be refactored.

    This reads its source rather than its behaviour, so it cannot drift silently: if the
    promoted implementation is renamed or its provenance keys change, the two copies stop
    agreeing and this fails.
    """
    from compose_v4.experiments import pmo_dynamic_v21_recovery as recovery

    source = inspect.getsource(recovery.PMODynamicV21ContinuityAdapter._adapt_bootstrap)
    for token in (SCHEMA_VERSION, "generation_pool_id", "continuity_pool_id",
                  "campaign_admission", "pmo_bootstrap_continuity"):
        assert token in source, f"recovery adapter no longer agrees on {token!r}"
