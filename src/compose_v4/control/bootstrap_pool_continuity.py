"""Stable archive-continuity identity across physically distinct bootstrap pools.

The Dynamic-v2.1 core treats ``dynamic_v21_bootstrap.pool_id`` as a single
archive-continuity identity and refuses measured programs that disagree about it.  That
invariant is incompatible with a recursive population optimizer: ``pool_id`` is the
identity of one round's merged candidate pool, so it changes every round, and a campaign
whose scored children become parents creates a NEW physical cold-start pool whenever it
revisits an initialization parent.  Surviving descendants are the point of the optimizer,
so the refusal fires on correct behaviour.

The resolution is not to drop the check.  Each physical pool identity is retained as
round provenance under ``generation_pool_id`` while the first pool's identity is presented
to the unchanged core invariant as ``pool_id``.  ``PMODynamicV21ContinuityAdapter`` in
``experiments/pmo_dynamic_v21_recovery.py`` established these semantics for a one-time
recovery of a failed campaign; that module is pinned by a sealed contract and must stay
byte-stable, so this is the live implementation and ``tests/test_bootstrap_pool_continuity.py``
asserts the two agree rather than letting a second copy drift.

Invariants maintained:
  * the continuity identity is the FIRST physical pool seen and never changes afterwards;
  * every physical pool identity is recorded once, in first-seen order;
  * a record is rewritten only when its pool differs from the continuity identity, and the
    rewrite is a copy -- the caller's record is never mutated;
  * a rewritten record carries the physical identity it came from, so the round provenance
    survives the adaptation.
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA_VERSION = "pmo_dynamic_v21_bootstrap_continuity_v1"
_BOOTSTRAP_KEY = "dynamic_v21_bootstrap"


class BootstrapPoolContinuity:
    """Present one stable pool identity while retaining every physical pool it saw."""

    def __init__(self) -> None:
        self.continuity_pool_id: str | None = None
        self.generation_pool_ids: list[str] = []

    def adapt(self, record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Return ``(record, is_new_generation_pool)``.

        ``is_new_generation_pool`` is True the first time a physical pool is seen, which
        is when the caller must adopt that pool's sampler state: a fresh cold-start pool
        carries its own RNG, and continuing with the previous pool's state would silently
        desynchronise the proposal streams from the candidates they produced.
        """
        bootstrap = (record.get("provenance") or {}).get(_BOOTSTRAP_KEY)
        if bootstrap is None:
            return record, False
        generation_pool_id = bootstrap.get("generation_pool_id", bootstrap.get("pool_id"))
        if not isinstance(generation_pool_id, str) or len(generation_pool_id) != 64:
            raise ValueError("PMO bootstrap lacks a physical pool identity")

        is_new = generation_pool_id not in self.generation_pool_ids
        if is_new:
            self.generation_pool_ids.append(generation_pool_id)
        if self.continuity_pool_id is None:
            self.continuity_pool_id = generation_pool_id
        if generation_pool_id == self.continuity_pool_id:
            return record, is_new

        adapted = json.loads(json.dumps(record))
        adapted_bootstrap = adapted["provenance"][_BOOTSTRAP_KEY]
        adapted_bootstrap["generation_pool_id"] = generation_pool_id
        adapted_bootstrap["pool_id"] = self.continuity_pool_id
        adapted["provenance"]["pmo_bootstrap_continuity"] = {
            "schema_version": SCHEMA_VERSION,
            "generation_pool_id": generation_pool_id,
            "continuity_pool_id": self.continuity_pool_id,
            "boundary": "campaign_admission",
        }
        return adapted, is_new

    def payload(self) -> dict[str, Any]:
        """Auditable record of which physical pools the archive absorbed."""
        return {
            "schema_version": SCHEMA_VERSION,
            "continuity_pool_id": self.continuity_pool_id,
            "generation_pool_ids": list(self.generation_pool_ids),
            "physical_pools_absorbed": len(self.generation_pool_ids),
        }

    @classmethod
    def restore(cls, payload: dict[str, Any] | None) -> BootstrapPoolContinuity:
        """Rebuild from `payload`; a missing payload restores an empty adapter.

        A resumed campaign that forgot which pools it had already absorbed would adopt
        whichever pool arrived first as its continuity identity, which need not be the
        one the archive was actually built on.
        """
        restored = cls()
        if not payload:
            return restored
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("bootstrap pool continuity payload has a foreign schema")
        restored.continuity_pool_id = payload.get("continuity_pool_id")
        restored.generation_pool_ids = list(payload.get("generation_pool_ids") or ())
        if restored.continuity_pool_id is not None and (
            not restored.generation_pool_ids
            or restored.generation_pool_ids[0] != restored.continuity_pool_id
        ):
            raise ValueError("continuity identity is not the first physical pool recorded")
        return restored
