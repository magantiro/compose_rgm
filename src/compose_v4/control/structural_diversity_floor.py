"""A bounded structural-diversity floor over the PMO archive.

WHAT THIS FIXES, AND WHERE IT WAS MEASURED
------------------------------------------
The PMO population controller draws one parent per proposal attempt from a
score-tilted distribution, and each proposal lane stops the moment it holds
``CHANNEL_CANDIDATE_LIMIT`` eligible candidates.  Those two facts compose badly.
Measured on the committed celecoxib 250-call archive, round zero:

    68 attempts        shallow 17 | structured 19 | jump 32
    the shallow and structured lanes each hit the 16-candidate cap, so the
    round-zero eligible pool is 32 and positions 18..128 of both schedules
    were never reached at all
    14 of 16 initialization parents received an attempt; 10 produced an
    endpoint; the top parent took 21 of 68 attempts and contributed 10 of the
    32 eligible endpoints

So whichever parent happens to be drawn early AND happens to yield executable
endpoints fills the cap, and parents sitting later in the schedule are never
tried.  The defect is CONTROL at the generation step, not absent support: the
same run's proposals change ring count on 42.8% of executable endpoints and
span a heavy-atom delta of -20..+12.

THE MECHANISM
-------------
Partition the archive into structurally distinct clusters, then RESERVE a
bounded share of schedule slots and of selected-batch slots for those clusters.

    partition      -> which archive molecules are structurally alike
    attempt_plan   -> which schedule positions are reserved, and for whom
    pool_plan      -> which selected-batch slots are reserved, and for whom

It is a floor, never a filter.  A reserved slot changes WHO is asked, not WHAT
is admitted: inside a reserved cluster the controller's own selection law still
decides, and every unreserved slot draws from the unmodified distribution.  No
candidate is ever removed from the eligible pool, and no gate is relaxed.

The partition is deliberately SCORE-FREE.  :func:`exploration_niches` already
implements the max-min partition this needs -- it is the same function the
production ``niche_score`` parent allocation calls -- and its only use of the
utility vector is to pick the first center and to order members inside a niche.
Passing a constant utility makes the first center the lexicographically largest
endpoint and leaves every later center chosen by structural distance alone, so
the clusters describe chemistry and nothing else.  Reusing it also means there
is exactly ONE clustering implementation in the PMO path rather than a parallel
allocator with its own notion of "distinct".

WHY THE SAME PARTITION AT BOTH HOPS
-----------------------------------
A selected-batch slot is reserved by its candidate's PARENT cluster, not by a
fresh clustering of the candidates.  That is not a shortcut: the quantity the
floor exists to protect is coverage of the ARCHIVE's structural variety, and
re-clustering the pool would measure the variety the generation step already
produced rather than the variety it failed to reach.  It also means one
partition per batch, so the floor cannot disagree with itself.

PARAMETERS ARE SEARCH SETTINGS, NOT ESTABLISHED OPTIMA
------------------------------------------------------
``max_clusters``, ``attempt_share`` and ``pool_slots`` were fixed before the
support test ran, from the measured cap arithmetic above: with 128 schedule
slots and ``attempt_share=0.5`` the reserved positions sit at stride two, so the
~18 attempts a lane reaches before its cap fills contain ~9 reserved positions
-- enough to visit eight clusters once.  They were not tuned against the
outcome.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from compose_v4.control.edit_learning_data import exploration_niches

SCHEMA_VERSION = "structural_diversity_floor_v1"

#: The one registered floor.  A name is added here only when it has been
#: measured, so a contract cannot request a policy nobody has run.
ARCHIVE_CLUSTER_FLOOR_V1 = "archive_cluster_floor_v1"


class DiversityFloorError(ValueError):
    """The floor cannot honour the guarantee it was configured to make."""


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise DiversityFloorError(f"{name} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True)
class StructuralDiversityFloor:
    """Reserve proposal attempts and selected-batch slots for archive clusters.

    ``max_clusters``   upper bound on the partition; the realized cluster count
                       is ``min(max_clusters, distinct archive endpoints)``.
    ``attempt_share``  fraction of a lane's schedule positions that are
                       reserved.  The rest draw from the unmodified parent law.
    ``pool_slots``     selected-batch slots reserved across distinct clusters.
    """

    max_clusters: int = 8
    attempt_share: float = 0.5
    pool_slots: int = 4

    def __post_init__(self) -> None:
        _positive_int("max_clusters", self.max_clusters)
        _positive_int("pool_slots", self.pool_slots)
        share = self.attempt_share
        if type(share) is not float or not 0.0 < share <= 1.0:
            raise DiversityFloorError(
                f"attempt_share must be a float in (0, 1], got {share!r}"
            )

    # ---- Partition -------------------------------------------------------

    def partition(self, endpoints: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        """Structurally distinct clusters of the archive, score-free.

        Delegates to :func:`exploration_niches` with a CONSTANT utility vector,
        so the partition is decided by the fingerprint kernel alone.  Endpoints
        are sorted first, which makes the result a pure function of the archive
        contents rather than of insertion order.
        """

        unique = sorted(set(endpoints))
        if not unique:
            raise DiversityFloorError("the diversity floor needs a non-empty archive")
        niches = exploration_niches(
            unique,
            [0.0] * len(unique),
            max_niches=min(self.max_clusters, len(unique)),
            per_niche=1,
        )
        groups = tuple(tuple(group) for group in niches["members"] if group)
        if not groups:
            raise DiversityFloorError("the archive partition came back empty")
        return groups

    @staticmethod
    def cluster_of_endpoint(
        partition: Sequence[Sequence[str]],
    ) -> dict[str, int]:
        """Endpoint -> cluster index, for attributing a parent to its cluster."""

        return {
            endpoint: index
            for index, group in enumerate(partition)
            for endpoint in group
        }

    # ---- Hop one: proposal attempts --------------------------------------

    def reserved_slot_count(self, slots: int) -> int:
        """How many of ``slots`` schedule positions the floor reserves."""

        _positive_int("slots", slots)
        return min(slots, max(1, math.ceil(self.attempt_share * slots)))

    def attempt_plan(
        self, *, clusters: int, slots: int, rotation: int = 0
    ) -> tuple[int | None, ...]:
        """One entry per schedule position: a cluster index, or ``None``.

        ``None`` means "draw from the unmodified parent law".  The reserved
        positions are stride-interleaved rather than packed at the front,
        because a lane stops at its candidate cap after roughly a fifth of its
        schedule: a block of reserved slots at the head would displace the base
        law entirely inside the window that matters, and a block at the tail
        would never be reached.

        Raises when the reservation is too small to give every cluster a slot.
        Silently delivering fewer attempts than the declared floor is the
        failure this surface exists to prevent, so it fails closed.
        """

        _positive_int("clusters", clusters)
        _positive_int("slots", slots)
        if type(rotation) is not int or rotation < 0:
            raise DiversityFloorError(f"rotation must be a non-negative int, got {rotation!r}")
        reserved = self.reserved_slot_count(slots)
        if reserved < clusters:
            raise DiversityFloorError(
                f"attempt_share={self.attempt_share} reserves {reserved} of {slots} "
                f"schedule slots, which cannot give each of {clusters} clusters an "
                "attempt; raise attempt_share or lower max_clusters"
            )
        positions = [(index * slots) // reserved for index in range(reserved)]
        if len(set(positions)) != reserved:
            raise DiversityFloorError("stride interleaving produced repeated positions")
        plan: list[int | None] = [None] * slots
        for order, position in enumerate(positions):
            plan[position] = (rotation + order) % clusters
        return tuple(plan)

    # ---- Hop two: selected-batch slots -----------------------------------

    def pool_plan(
        self,
        *,
        ranked_by_cluster: Mapping[int, Sequence[int]],
        clusters: int,
        slots: int,
        rotation: int = 0,
    ) -> tuple[int, ...]:
        """Reserve up to ``slots`` picks, round-robin over distinct clusters.

        ``ranked_by_cluster`` maps a cluster index to that cluster's candidate
        positions, best first, as the caller's own value model ordered them --
        so the floor decides WHICH CLUSTER gets a slot and the existing ranking
        decides which candidate inside it.  Clusters with nothing left are
        skipped; a second pick from a cluster happens only once every other
        represented cluster has had one.
        """

        _positive_int("clusters", clusters)
        if type(slots) is not int or slots < 0:
            raise DiversityFloorError(f"slots must be a non-negative int, got {slots!r}")
        if type(rotation) is not int or rotation < 0:
            raise DiversityFloorError(f"rotation must be a non-negative int, got {rotation!r}")
        queues = {
            int(cluster): list(positions)
            for cluster, positions in ranked_by_cluster.items()
            if positions
        }
        if any(not 0 <= cluster < clusters for cluster in queues):
            raise DiversityFloorError("a ranked cluster index is outside the partition")
        picks: list[int] = []
        while len(picks) < slots and queues:
            drawn = False
            for step in range(clusters):
                cluster = (rotation + step) % clusters
                queue = queues.get(cluster)
                if not queue:
                    continue
                picks.append(int(queue.pop(0)))
                drawn = True
                if not queue:
                    queues.pop(cluster, None)
                if len(picks) >= slots:
                    break
            if not drawn:
                break
        return tuple(picks)

    # ---- Telemetry -------------------------------------------------------

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "law": ARCHIVE_CLUSTER_FLOOR_V1,
            "max_clusters": self.max_clusters,
            "attempt_share": self.attempt_share,
            "pool_slots": self.pool_slots,
        }
