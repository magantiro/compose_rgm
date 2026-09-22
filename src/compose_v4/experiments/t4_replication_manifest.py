"""The T4 replication manifest: clone the historical method, change only the run seed.

WHY THIS EXISTS
---------------
Both published comparators report a MULTI-RUN MEAN per cell.
``docs/genmol_t4_all_methods.json`` states its metric as "mean docking score of the
most optimized lead over 3 runs"; ``configs/t4_published_invirtuogen_baseline.json``
states "value is InVirtuoGen mean, stderr in parentheses in the source" and carries
per-cell stderrs of 0.1-0.9 kcal/mol.  The frozen COMPOSE table reports a SINGLE-RUN
best.  That is not like-for-like, and docking here is measurably not reproducible
across runs -- one molecule scored -7.5, -8.30 and -8.8 in three separate runs -- so
the honest repair is to report a distribution, not to argue about a single row.

WHAT A REPLICATE IS, AND WHAT IT IS NOT
---------------------------------------
A replicate CLONES the historical per-cell method and changes exactly one thing: the
independent stochastic run seed.  Held fixed: starting ligand, delta, proposal law,
controller, support operator, constraints, oracle ceiling and stopping rule.

It is NOT the canonical shared controller and NOT the frozen generic escalation
ladder.  The question is "is the method that produced our published T4 table
reproducible across independent runs", so substituting a better controller would
answer a different question and silently invalidate the comparison.

THE PANEL IS NOT ONE CONFIGURATION, AND THAT IS THE HAZARD
-----------------------------------------------------------
Measured across the ten panel contracts, two executable families exist:

    PARP1, BRAF      docking_seed 20260918   route expert beam 32 / expansion 24 /
                                             realization 32, no scale_balanced
    FA7, 5HT1B, JAK2 docking_seed 20260919   route expert beam 48 / expansion 48 /
                                             realization 64, scale_balanced true

and the PARP1/BRAF contracts carry no ``route_scale_floor_rounds``,
``phase_poll_seconds``, ``proposal_wait_seconds``, ``query_wait_seconds`` or
``prior_operational_waste`` at all.  Six cells additionally ran under a rescue arm,
and two DIFFERENT support operators are involved: a bridge-separated region law
(FA7/BRAF) and a protonation-aware proposal expert (5HT1B seed 3).  Rebuilding any
row from today's defaults, or replacing one operator with the other, would test a
method the table never used.  ``audit_manifest`` fails closed on exactly that.

SEEDS
-----
``controller_seed`` is the single stochastic entry point: the cell driver seeds its
own generator from it, and every proposal seed is derived from it by

    controller_seed + 1_000_003*round + 10_007*parent + 101*expert

with further declared offsets for the rescue lanes.  A replicate therefore only has
to move ``controller_seed``.  ``derive_replicate_seed`` moves it by a stride chosen
to exceed the widest span any single cell's derived seeds can occupy, and
``seed_stream_collisions`` PROVES disjointness by enumerating the streams rather than
asserting it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = "t4_replication_manifest_v1"

#: Replicate 1 is the historical run already on the volumes. It is never re-run.
HISTORICAL_REPLICATE = 1
#: The replicates this manifest authorizes.
NEW_REPLICATES = (2, 3)

#: Offset between one replicate's ``controller_seed`` and the next.
#:
#: It must exceed the widest span a single cell's derived seeds can occupy, or two
#: replicates of the same cell could draw the same proposal stream at different
#: rounds. The widest observed derivation is the support-expansion ladder's
#: ``700_000_003*(attempt+1) + 13_000_003*replicate + 1_000_003*round
#: + 10_007*parent + 101*expert``, which at attempt 2, replicate 16, round 32 reaches
#: about 2.34e9. Eight billion clears that with room for a longer ladder, and
#: ``seed_stream_collisions`` re-proves it rather than trusting this comment.
REPLICATE_SEED_STRIDE = 8_000_000_029

#: Bounds used when ENUMERATING a cell's derived seed stream for the collision proof.
#: They are deliberate over-estimates of what a 250-call budget at batch 8 can reach.
MAX_ROUNDS = 64
MAX_PARENTS = 8
MAX_EXPERTS = 4
#: The rescue/expansion lane offsets, as the apps derive them.
FALLBACK_OFFSET = 900_007
LADDER_ATTEMPT_STRIDE = 700_000_003
LADDER_REPLICATE_STRIDE = 13_000_003
MAX_LADDER_ATTEMPTS = 4
MAX_LADDER_REPLICATES = 20

#: Support operators. These are DIFFERENT MECHANISMS and the manifest refuses to
#: let one stand in for another, because the frozen table labels both
#: `support_expansion` and that label conflates them.
SUPPORT_OPERATORS = ("none", "region_repair", "protonation", "support_expansion")

#: Cells whose historical method is NOT the vanilla panel arm. Keyed
#: (target, seed, delta) with seed 1-based exactly as the published table prints it.
#: Each entry names the operator that cell's historical run actually used.
REQUIRED_SUPPORT_OPERATORS: dict[tuple[str, int, float], str] = {
    ("fa7", 3, 0.4): "region_repair",
    ("5ht1b", 3, 0.4): "protonation",
    ("fa7", 3, 0.6): "region_repair",
    ("5ht1b", 3, 0.6): "protonation",
    ("braf", 1, 0.6): "region_repair",
    ("braf", 2, 0.6): "region_repair",
}

#: The one cell the frozen table leaves blank. It is carried in the manifest so the
#: replication is explicit about it rather than silently 29 rows.
BLANK_CELL = ("fa7", 1, 0.6)


class ManifestAuditError(AssertionError):
    """The manifest cannot be honoured as written. Nothing docks."""


@dataclass(frozen=True)
class ReplicationRow:
    """One (target, seed, delta) cell and everything needed to clone its method."""

    target: str
    seed: int              # 1-based, as the published table prints it
    cell: str              # 0-based cell id, e.g. "braf_0"
    delta: float           # READ FROM THE CONTRACT, never from a name
    source_label: str      # the frozen table's own `source` string
    arm: str               # registered arm name
    contract_path: str
    contract_payload_sha256: str
    contract_file_sha256: str
    code_revision: str
    app_module: str
    controller_family: str
    support_operator: str
    charged_calls_per_cell: int
    total_charged_call_ceiling: int
    historical_charged_calls: int | None
    historical_best: float | None
    controller_seed: int
    replicate_seeds: dict[int, int] = field(default_factory=dict)
    completed_runs_today: int | None = None
    selection_verdict: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, int, float]:
        return (self.target, int(self.seed), float(self.delta))

    def as_record(self) -> dict:
        return {
            "target": self.target,
            "seed": int(self.seed),
            "cell": self.cell,
            "delta": float(self.delta),
            "source_label": self.source_label,
            "arm": self.arm,
            "contract_path": self.contract_path,
            "contract_payload_sha256": self.contract_payload_sha256,
            "contract_file_sha256": self.contract_file_sha256,
            "code_revision": self.code_revision,
            "app_module": self.app_module,
            "controller_family": self.controller_family,
            "support_operator": self.support_operator,
            "charged_calls_per_cell": int(self.charged_calls_per_cell),
            "total_charged_call_ceiling": int(self.total_charged_call_ceiling),
            "historical_charged_calls": self.historical_charged_calls,
            "historical_best": self.historical_best,
            "controller_seed": int(self.controller_seed),
            "replicate_seeds": {str(k): int(v) for k, v in sorted(self.replicate_seeds.items())},
            "completed_runs_today": self.completed_runs_today,
            "selection_verdict": self.selection_verdict,
            "notes": list(self.notes),
        }


def derive_replicate_seed(controller_seed: int, replicate: int) -> int:
    """The ``controller_seed`` replicate ``r`` runs under.

    Replicate 1 IS the historical seed, unchanged, because replicate 1 is the run
    already on the volume and is never re-run. Returning it unchanged is what makes
    "we changed only the run seed" checkable rather than asserted.
    """

    if replicate < 1:
        raise ValueError("replicate indices are 1-based")
    return int(controller_seed) + REPLICATE_SEED_STRIDE * (int(replicate) - 1)


def derived_seed_offsets() -> np.ndarray:
    """Every OFFSET a cell's derived proposal seeds can sit at, relative to its base.

    The offset set is identical for every cell, because every app derives its seeds
    from ``controller_seed`` by adding one of these. Factoring the base out is what
    makes the collision proof cheap enough to run exhaustively instead of sampled:
    two streams collide iff their base difference lies in this set's difference set,
    which is tested directly in ``seed_stream_collisions``.
    """

    base_offsets = []
    for round_index in range(MAX_ROUNDS + 1):
        for parent in range(MAX_PARENTS):
            for expert in range(MAX_EXPERTS):
                base_offsets.append(1_000_003 * round_index + 10_007 * parent + 101 * expert)
    base_offsets_array = np.asarray(sorted(set(base_offsets)), dtype=np.int64)

    lanes = [np.zeros(1, dtype=np.int64), np.asarray([FALLBACK_OFFSET], dtype=np.int64)]
    ladder = [
        LADDER_ATTEMPT_STRIDE * (attempt + 1) + LADDER_REPLICATE_STRIDE * replicate
        for attempt in range(MAX_LADDER_ATTEMPTS)
        for replicate in range(MAX_LADDER_REPLICATES)
    ]
    lanes.append(np.asarray(sorted(set(ladder)), dtype=np.int64))
    stacked = np.concatenate(
        [(lane[:, None] + base_offsets_array[None, :]).ravel() for lane in lanes]
    )
    return np.unique(stacked)


def derived_seed_stream(controller_seed: int) -> set[int]:
    """Every proposal seed a cell can derive from one ``controller_seed``.

    Enumerated over the declared bounds, so the collision check is a PROOF rather
    than an appeal to a stride being "big enough".
    """

    return {int(controller_seed) + int(offset) for offset in derived_seed_offsets()}


def seed_stream_collisions(rows: Sequence[ReplicationRow]) -> list[dict]:
    """Replicates OF THE SAME CELL that share a derived proposal seed.

    WHAT THIS CHECKS, AND WHY IT IS SCOPED THIS WAY. The property a replication needs
    is that replicate 2 of a cell is not a relabelled repeat of replicate 1 of the
    SAME cell. Two DIFFERENT cells sharing a seed value is not a defect: the stream
    drives proposals on a different starting molecule under a different fiber, so the
    draws differ regardless, and reusing a seed across independent tasks is ordinary.

    A first version of this function compared every stream against every other and
    reported 96 "collisions", of which every one was either the same cell at its two
    delta thresholds -- which share a ``controller_seed`` in the HISTORICAL contracts
    and must keep sharing it, because a replicate clones the historical method -- or
    two unrelated proteins whose base seeds happen to differ by a representable
    ladder offset. Failing the audit on those would have blocked a correct manifest
    and, worse, read as evidence that the seed scheme was broken.

    Exact, not sampled: streams ``a`` and ``b`` share a seed iff
    ``offsets & (offsets + (base_b - base_a))`` is non-empty, so the question reduces
    to one intersection per distinct base difference against a single array.
    """

    offsets = derived_seed_offsets()
    by_cell: dict[tuple[str, int, float], dict[int, int]] = {}
    for row in rows:
        by_cell.setdefault(row.key, {}).update(
            {int(replicate): int(seed) for replicate, seed in row.replicate_seeds.items()}
        )

    collisions = []
    for key, seeds in sorted(by_cell.items()):
        replicates = sorted(seeds)
        for index, left in enumerate(replicates):
            for right in replicates[index + 1 :]:
                difference = seeds[right] - seeds[left]
                if not np.intersect1d(
                    offsets, offsets + difference, assume_unique=True
                ).size:
                    continue
                shared = np.intersect1d(
                    offsets + seeds[left], offsets + seeds[right], assume_unique=True
                )
                collisions.append(
                    {
                        "cell": list(key),
                        "replicates": [left, right],
                        "shared_count": int(shared.size),
                        "example": int(shared.min()),
                    }
                )
    return collisions


def inherited_cross_cell_seed_sharing(rows: Sequence[ReplicationRow]) -> list[dict]:
    """Cells that share a derived seed with ANOTHER cell, reported not refused.

    This is a property of the HISTORICAL contracts, not of the replication: the two
    delta arms of one cell carry the same ``controller_seed``, and some unrelated
    base seeds differ by a representable offset. It is recorded so the manifest
    states it rather than a reader discovering it, and so a future change that
    INTRODUCES sharing is visible against this baseline.
    """

    offsets = derived_seed_offsets()
    bases: dict[tuple[str, int, float, int], int] = {}
    for row in rows:
        for replicate, seed in row.replicate_seeds.items():
            bases[(row.target, row.seed, row.delta, int(replicate))] = int(seed)
    keys = sorted(bases)
    cached: dict[int, bool] = {}
    shared_pairs = []
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            if (left[0], left[1]) == (right[0], right[1]) and left[3] == right[3]:
                reason = "same_cell_both_deltas"
            elif (left[0], left[1], left[2]) == (right[0], right[1], right[2]):
                continue  # within-cell replicates: handled by seed_stream_collisions
            else:
                reason = "unrelated_cells"
            difference = bases[right] - bases[left]
            if difference not in cached:
                cached[difference] = bool(
                    np.intersect1d(offsets, offsets + difference, assume_unique=True).size
                )
            if cached[difference]:
                shared_pairs.append({"left": list(left), "right": list(right), "reason": reason})
    return shared_pairs


def audit_manifest(
    rows: Sequence[ReplicationRow],
    *,
    contract_delta: dict[str, float],
    contract_ceiling: dict[str, tuple[int, int]],
    contracts_rebuilt: Iterable[str] = (),
    resume_enabled_rows: Iterable[tuple[str, int, float]] = (),
) -> dict:
    """Fail closed unless every replicate provably clones its historical method.

    ``contract_delta`` and ``contract_ceiling`` are read from the CONTRACTS ON DISK by
    the caller and passed in, so this function checks the manifest against the
    contracts rather than against itself -- a comparison whose expectation is
    recomputed from the thing under test cannot fail.

    Raises ``ManifestAuditError`` naming every violation. Returns the passing report.
    """

    violations: list[str] = []
    seen: set[tuple[str, int, float]] = set()

    for row in rows:
        if row.key in seen:
            violations.append(f"duplicate manifest row for {row.key}")
        seen.add(row.key)

        # (5) delta must come from the contract. A shipped contract has carried 0.4
        # while its name said 0.6, so a name-derived delta is not evidence.
        declared = contract_delta.get(row.contract_path)
        if declared is None:
            violations.append(
                f"{row.key}: no contract delta supplied for {row.contract_path}; "
                "delta must be read from the contract, never from a name"
            )
        elif abs(float(declared) - float(row.delta)) > 1e-12:
            violations.append(
                f"{row.key}: manifest delta {row.delta} does not match the contract's "
                f"{declared} in {row.contract_path}"
            )

        # (6) the oracle ceiling must be the historical one.
        ceiling = contract_ceiling.get(row.contract_path)
        if ceiling is None:
            violations.append(f"{row.key}: no contract ceiling supplied for {row.contract_path}")
        else:
            per_cell, total = ceiling
            if int(per_cell) != int(row.charged_calls_per_cell):
                violations.append(
                    f"{row.key}: charged_calls_per_cell {row.charged_calls_per_cell} does not "
                    f"match the historical contract's {per_cell}"
                )
            if int(total) != int(row.total_charged_call_ceiling):
                violations.append(
                    f"{row.key}: total_charged_call_ceiling {row.total_charged_call_ceiling} "
                    f"does not match the historical contract's {total}"
                )

        # (1)(2)(3) the support operator must be the one that cell historically ran.
        required = REQUIRED_SUPPORT_OPERATORS.get(row.key, "none")
        if row.support_operator not in SUPPORT_OPERATORS:
            violations.append(f"{row.key}: unknown support operator {row.support_operator!r}")
        elif required == "none":
            if row.support_operator != "none":
                violations.append(
                    f"{row.key}: historical run used the vanilla panel arm but the manifest "
                    f"assigns support operator {row.support_operator!r}"
                )
        elif row.support_operator != required:
            violations.append(
                f"{row.key}: historical support operator is {required!r} but the manifest "
                f"assigns {row.support_operator!r}; the protonation and region operators are "
                "different mechanisms and neither may stand in for the other"
            )

        # (4) no row may be rebuilt from today's defaults.
        if row.contract_path in set(contracts_rebuilt):
            violations.append(
                f"{row.key}: {row.contract_path} was rebuilt rather than cloned from the "
                "historical sealed contract"
            )

        # replicate coverage
        missing = [r for r in NEW_REPLICATES if r not in row.replicate_seeds]
        if missing:
            violations.append(f"{row.key}: no seed declared for replicate(s) {missing}")
        if row.replicate_seeds.get(HISTORICAL_REPLICATE) not in (None, row.controller_seed):
            violations.append(
                f"{row.key}: replicate 1 must carry the historical controller seed unchanged"
            )
        for replicate in NEW_REPLICATES:
            expected = derive_replicate_seed(row.controller_seed, replicate)
            actual = row.replicate_seeds.get(replicate)
            if actual is not None and int(actual) != expected:
                violations.append(
                    f"{row.key}: replicate {replicate} seed {actual} is not the declared "
                    f"derivation {expected}"
                )

    # (7) no replicate may inherit state from replicate 1.
    for key in resume_enabled_rows:
        violations.append(
            f"{tuple(key)}: a replicate is configured to resume; every replicate must start "
            "from scratch or it is not an independent run"
        )

    collisions = seed_stream_collisions(rows)
    for collision in collisions:
        violations.append(
            f"{tuple(collision['cell'])}: replicates {collision['replicates']} share "
            f"{collision['shared_count']} derived seeds (e.g. {collision['example']}), so one "
            "replicate would repeat the other's draws rather than be an independent run"
        )

    if violations:
        raise ManifestAuditError(
            "the replication manifest is not safe to run; nothing docks.\n  - "
            + "\n  - ".join(violations)
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "rows_audited": len(rows),
        "within_cell_seed_stream_collisions": 0,
        "inherited_cross_cell_seed_sharing": len(inherited_cross_cell_seed_sharing(rows)),
        "support_operators_pinned": {
            "/".join(str(part) for part in key): value
            for key, value in sorted(REQUIRED_SUPPORT_OPERATORS.items())
        },
        "blank_cell": list(BLANK_CELL),
        "verdict": "AUDIT_GREEN_REPLICATES_MAY_BE_LAUNCHED",
    }
