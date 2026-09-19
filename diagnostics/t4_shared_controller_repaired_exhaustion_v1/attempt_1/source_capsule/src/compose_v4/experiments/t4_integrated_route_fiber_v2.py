"""Four-expert FiberControl policy including retained-core feasibility repair."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from compose_v4.control.fiber_control import (
    ProgramValue,
    SearchState,
    acquisition,
    program_features,
)

EXPERTS = (
    "shallow",
    "anchored_replacement",
    "route_complete_region",
    "retained_core_prune",
)


def durable_phase_action(*, lock_exists: bool, receipt_statuses: Iterable[str]) -> str:
    """Decide whether a scored phase may start, recover, or must fail closed.

    A published query lock is the irreversible boundary.  Once it exists, the
    controller may only reconstruct a phase from complete durable receipts.  It
    must never resubmit a missing or unresolved query after an executor eviction.
    """

    statuses = tuple(str(value) for value in receipt_statuses)
    if not lock_exists:
        if statuses:
            raise ValueError("query receipts exist without an immutable phase lock")
        return "start"
    if statuses and all(value == "complete" for value in statuses):
        return "recover"
    return "fail_closed"


def proposal_collection_action(
    *, receipt_statuses: Iterable[str], now: float, deadline: float
) -> str:
    """Return the nonblocking action for independently persisted expert jobs."""

    statuses = tuple(str(value) for value in receipt_statuses)
    terminal = {"complete", "failed"}
    if statuses and all(value in terminal for value in statuses):
        return "collect"
    if float(now) < float(deadline):
        return "wait"
    return "collect_with_abstentions"


def query_collection_action(
    *,
    lock_exists: bool,
    receipt_statuses: Iterable[str],
    now: float,
    deadline: float | None,
) -> str:
    """Choose a no-retry action for one immutable scored-query lock.

    Missing or reserved receipts are never resubmitted.  They are allowed time to
    finish, then conservatively consume their already locked budget as unresolved
    failures so the next checkpointed round can continue.
    """

    statuses = tuple(str(value) for value in receipt_statuses)
    if not lock_exists:
        if statuses:
            raise ValueError("query receipts exist without an immutable phase lock")
        return "start"
    if statuses and all(value == "complete" for value in statuses):
        return "recover"
    if deadline is None:
        raise ValueError("locked query phase has no frozen receipt deadline")
    if float(now) < float(deadline):
        return "wait"
    return "recover_with_unresolved"


def migrate_v1_checkpoint(
    checkpoint: dict,
    *,
    cell: str,
    original_seed: str,
    new_contract_payload_sha256: str,
    charged_call_ceiling: int,
    legacy_run_id: str,
    legacy_checkpoint_sha256: str,
    legacy_checkpoint_payload_sha256: str,
    legacy_round_lock_sha256: str,
    legacy_round_lock_payload_sha256: str,
    unresolved_round_lock: dict | None = None,
    legacy_unresolved_round_lock_sha256: str | None = None,
    legacy_unresolved_round_lock_payload_sha256: str | None = None,
    parents: int | None = None,
    parent_explore: float | None = None,
    value_penalty: float | None = None,
    batch: int | None = None,
    exploration: int | None = None,
    expert_floor_rounds: int | None = None,
    route_scale_floor_rounds: int | None = None,
) -> dict:
    """Convert one completed v1 round checkpoint into the restartable v2 schema.

    Only information already sealed in the v1 checkpoint is carried forward.  The
    root result is reconstructed from the source entry in the sealed archive, not
    from console logs or a new oracle call.  The legacy checkpoint and round lock
    identities remain attached to the migrated state.
    """

    if checkpoint.get("schema_version") != "t4_integrated_route_fiber_checkpoint_v1":
        raise ValueError("legacy checkpoint has the wrong schema")
    if checkpoint.get("status") != "running" or checkpoint.get("cell") != cell:
        raise ValueError("legacy checkpoint status or cell mismatch")
    archive = {str(key): float(value) for key, value in checkpoint["archive"].items()}
    if original_seed not in archive:
        raise ValueError("legacy checkpoint does not contain the declared source")
    rounds = list(checkpoint["rounds"])
    history = list(checkpoint["history"])
    if not rounds or len(rounds) != len(history):
        raise ValueError("legacy checkpoint must contain aligned completed rounds")
    expected_rounds = list(range(1, len(rounds) + 1))
    if [int(row["round"]) for row in rounds] != expected_rounds:
        raise ValueError("legacy checkpoint rounds are not contiguous from one")
    if [int(row["round"]) for row in history] != expected_rounds:
        raise ValueError("legacy checkpoint history is not contiguous from one")
    charged_calls = int(checkpoint["charged_calls"])
    if charged_calls != 1 + sum(int(row["charged_this_round"]) for row in rounds):
        raise ValueError(
            "legacy charged-call count is inconsistent with completed rounds"
        )
    if int(checkpoint["budget_remaining"]) != charged_call_ceiling - charged_calls:
        raise ValueError(
            "legacy remaining budget is inconsistent with the frozen ceiling"
        )
    if (
        rounds[-1].get("candidate_lock_payload_sha256")
        != legacy_round_lock_payload_sha256
    ):
        raise ValueError(
            "legacy checkpoint does not bind the declared final round lock"
        )
    if len(checkpoint["features"]) != len(checkpoint["improvements"]):
        raise ValueError("legacy value-training rows are misaligned")
    if "rng_state" not in checkpoint:
        raise ValueError("legacy checkpoint has no resumable RNG state")

    migrated_rounds = rounds
    migrated_history = history
    migrated_rng_state = checkpoint["rng_state"]
    unresolved_charged_calls = 0
    if unresolved_round_lock is not None:
        if unresolved_round_lock.get("schema_version") != "t4_integrated_round_lock_v1":
            raise ValueError("legacy unresolved lock has the wrong schema")
        unresolved_round = len(rounds) + 1
        if (
            unresolved_round_lock.get("cell") != cell
            or int(unresolved_round_lock.get("round", -1)) != unresolved_round
            or int(unresolved_round_lock.get("charged_before", -1)) != charged_calls
        ):
            raise ValueError("legacy unresolved lock does not follow the checkpoint")
        queries = list(unresolved_round_lock.get("queries") or [])
        if not queries:
            raise ValueError("legacy unresolved lock has no selected queries")
        if (
            not legacy_unresolved_round_lock_sha256
            or not legacy_unresolved_round_lock_payload_sha256
        ):
            raise ValueError("legacy unresolved lock identities are required")
        selected = list(unresolved_round_lock.get("selected_rows") or [])
        migrated_rng_state = unresolved_round_lock.get("rng_state_after_selection")
        if not selected or migrated_rng_state is None:
            required = {
                "parents": parents,
                "parent_explore": parent_explore,
                "value_penalty": value_penalty,
                "batch": batch,
                "exploration": exploration,
                "expert_floor_rounds": expert_floor_rounds,
                "route_scale_floor_rounds": route_scale_floor_rounds,
            }
            missing = sorted(key for key, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "legacy unresolved selection replay is missing settings: "
                    + ", ".join(missing)
                )
            from compose_v4.experiments.t4_integrated_route_fiber import (
                select_batch as select_v1_batch,
            )

            replay_state = SearchState(
                archive=dict(archive),
                budget=charged_call_ceiling - charged_calls,
                rounds=len(rounds),
                history=list(history),
            )
            replay_value = ProgramValue(penalty=float(value_penalty))
            replay_value.fit(checkpoint["features"], checkpoint["improvements"])
            replay_rng = np.random.default_rng()
            replay_rng.bit_generator.state = checkpoint["rng_state"]
            replay_state.parents(
                limit=int(parents),
                rng=replay_rng,
                explore=float(parent_explore),
            )
            candidates = [
                {**row, "fingerprint": set(row.get("fingerprint") or [])}
                for row in unresolved_round_lock.get("candidate_pool") or []
            ]
            selected = select_v1_batch(
                candidates,
                replay_value,
                replay_state,
                replay_rng,
                round_index=unresolved_round,
                batch=min(int(batch), len(queries)),
                exploration=min(int(exploration), len(queries)),
                expert_floor_rounds=int(expert_floor_rounds),
                route_scale_floor_rounds=int(route_scale_floor_rounds),
            )
            selected = [
                {
                    **row,
                    "fingerprint": sorted(row.get("fingerprint") or []),
                }
                for row in selected
            ]
            migrated_rng_state = replay_rng.bit_generator.state
        if len(queries) != len(selected):
            raise ValueError("legacy unresolved lock has misaligned selected queries")
        query_fields = (
            "smiles",
            "selection_kind",
            "proposal_experts",
            "parent",
            "parent_score",
        )
        if any(
            any(row.get(field) != query.get(field) for field in query_fields)
            for row, query in zip(selected, queries, strict=True)
        ):
            raise ValueError(
                "legacy unresolved selection replay does not match its lock"
            )
        unresolved_charged_calls = len(queries)
        if charged_calls + unresolved_charged_calls > charged_call_ceiling:
            raise ValueError("legacy unresolved lock exceeds the charged-call ceiling")
        observed = []
        for row, query in zip(selected, queries, strict=True):
            if row.get("smiles") != query.get("smiles"):
                raise ValueError(
                    "legacy unresolved lock query does not match selected row"
                )
            observed.append(
                {
                    **row,
                    "query_id": query["query_id"],
                    "score": None,
                    "failure": "legacy_unresolved_locked_query_after_preemption",
                    "elapsed_seconds": None,
                }
            )
        migrated_rounds = [
            *rounds,
            {
                "round": unresolved_round,
                "charged_calls": charged_calls + unresolved_charged_calls,
                "charged_this_round": unresolved_charged_calls,
                "candidate_lock_payload_sha256": legacy_unresolved_round_lock_payload_sha256,
                "pool_census": unresolved_round_lock.get("pool_census", {}),
                "selected_census": unresolved_round_lock.get("selected_census", {}),
                "selection_kind_census": {
                    kind: sum(row.get("selection_kind") == kind for row in selected)
                    for kind in (
                        "route_scale_floor",
                        "expert_floor",
                        "model",
                        "exploration",
                    )
                },
                "docked": observed,
                "round_best": None,
                "best_so_far": min(archive.values()),
                "improved": False,
                "value_training_rows": len(checkpoint["improvements"]),
                "operational_recovery": "unresolved v1 locked calls charged without retry",
            },
        ]
        migrated_history = [*history, {"round": unresolved_round, "improved": False}]

    migrated_charged_calls = charged_calls + unresolved_charged_calls
    return {
        "schema_version": "t4_integrated_route_fiber_checkpoint_v2",
        "status": "running",
        "cell": cell,
        "contract_payload_sha256": new_contract_payload_sha256,
        "charged_calls": migrated_charged_calls,
        "budget_remaining": charged_call_ceiling - migrated_charged_calls,
        "archive": dict(sorted(archive.items())),
        "features": list(checkpoint["features"]),
        "improvements": list(checkpoint["improvements"]),
        "history": migrated_history,
        "rounds": migrated_rounds,
        "rounds_completed": len(migrated_rounds),
        "root_result": {
            "query_id": f"{legacy_run_id}_{cell}_root",
            "smiles": original_seed,
            "score": archive[original_seed],
            "failure": None,
            "recovered_from": "sealed_v1_archive_source_entry",
        },
        "rng_state": migrated_rng_state,
        "migration_provenance": {
            "legacy_run_id": legacy_run_id,
            "legacy_contract_payload_sha256": checkpoint["contract_payload_sha256"],
            "legacy_checkpoint_sha256": legacy_checkpoint_sha256,
            "legacy_checkpoint_payload_sha256": legacy_checkpoint_payload_sha256,
            "legacy_round_lock_sha256": legacy_round_lock_sha256,
            "legacy_round_lock_payload_sha256": legacy_round_lock_payload_sha256,
            "legacy_unresolved_round_lock_sha256": legacy_unresolved_round_lock_sha256,
            "legacy_unresolved_round_lock_payload_sha256": (
                legacy_unresolved_round_lock_payload_sha256
            ),
            "legacy_completed_charged_calls": charged_calls,
            "legacy_unresolved_charged_calls": unresolved_charged_calls,
            "legacy_charged_calls": migrated_charged_calls,
        },
    }


def _experts(record: dict) -> tuple[str, ...]:
    values = record.get("proposal_experts") or [record.get("proposal_lane")]
    result = tuple(sorted({str(value) for value in values if value}))
    unknown = set(result).difference(EXPERTS)
    if unknown:
        raise ValueError(f"unknown proposal experts: {sorted(unknown)}")
    return result


def merge_expert_pools(pools: dict[str, Iterable[dict]]) -> list[dict]:
    merged: dict[str, dict] = {}
    origins: dict[str, list[dict]] = {}
    expert_sets: dict[str, set[str]] = {}
    family_sets: dict[str, set[str]] = {}
    for expert in EXPERTS:
        for position, source in enumerate(pools.get(expert, ())):
            row = dict(source)
            if row.get("proposal_lane") not in (None, expert):
                raise ValueError(
                    f"pool {expert!r} contains lane {row.get('proposal_lane')!r}"
                )
            row["proposal_lane"] = expert
            row["proposal_experts"] = sorted(set(_experts(row)) | {expert})
            smiles = str(row.get("smiles") or "")
            if not smiles:
                raise ValueError(f"proposal from {expert!r} has no canonical smiles")
            origins.setdefault(smiles, []).append(
                {
                    "expert": expert,
                    "position": position,
                    "parent": row.get("parent"),
                    "parent_score": row.get("parent_score"),
                    "program_families": sorted(set(row.get("program_families") or [])),
                    "route_program_id": row.get("route_program_id"),
                }
            )
            expert_sets.setdefault(smiles, set()).update(row["proposal_experts"])
            family_sets.setdefault(smiles, set()).update(row.get("families") or [])
            incumbent = merged.get(smiles)
            key = (
                float(row.get("parent_score", float("inf"))),
                EXPERTS.index(expert),
                position,
            )
            incumbent_key = None
            if incumbent is not None:
                incumbent_key = (
                    float(incumbent.get("parent_score", float("inf"))),
                    EXPERTS.index(str(incumbent["proposal_lane"])),
                    int(incumbent["_pool_position"]),
                )
            if incumbent is None or key < incumbent_key:
                merged[smiles] = {**row, "_pool_position": position}
    result = []
    for smiles in sorted(merged):
        row = dict(merged[smiles])
        row.pop("_pool_position", None)
        row["proposal_experts"] = sorted(expert_sets[smiles])
        row["families"] = sorted(family_sets[smiles])
        row["proposal_origins"] = sorted(
            origins[smiles],
            key=lambda value: (
                value["expert"],
                str(value.get("parent") or ""),
                int(value["position"]),
            ),
        )
        result.append(row)
    return result


def integrated_features(record: dict, state: SearchState) -> np.ndarray:
    base = program_features(record, state)
    experts = set(_experts(record))
    return np.concatenate(
        [
            base[:-1],
            np.asarray([float(name in experts) for name in EXPERTS], dtype=float),
            np.asarray([float(len(experts) > 1), 1.0], dtype=float),
        ]
    )


def attach_features(records: Iterable[dict], state: SearchState, fiber) -> list[dict]:
    from rdkit import Chem

    prepared = []
    for source in records:
        row = dict(source)
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            continue
        row["features"] = integrated_features(row, state)
        row["fingerprint"] = set(fiber.generator.GetFingerprint(molecule).GetOnBits())
        prepared.append(row)
    return prepared


def select_batch(
    candidates: list[dict],
    value: ProgramValue,
    state: SearchState,
    rng,
    *,
    round_index: int,
    batch: int,
    exploration: int,
    expert_floor_rounds: int,
) -> list[dict]:
    if batch < 1 or exploration < 0 or exploration > batch:
        raise ValueError("invalid batch or exploration quota")
    available = [row for row in candidates if row["smiles"] not in state.archive]
    selected: list[dict] = []
    used: set[str] = set()
    if round_index <= expert_floor_rounds:
        for expert in EXPERTS:
            choices = [
                row
                for row in available
                if expert in _experts(row) and row["smiles"] not in used
            ]
            if not choices or len(selected) >= batch:
                continue
            chosen = choices[int(rng.integers(len(choices)))]
            selected.append({**chosen, "selection_kind": "expert_floor"})
            used.add(chosen["smiles"])
    remaining = [row for row in available if row["smiles"] not in used]
    room = min(batch - len(selected), len(remaining))
    if room:
        random_quota = min(exploration, room)
        indices = acquisition(
            remaining, value, state, rng, batch=room, exploration=random_quota
        )
        model_count = room - random_quota if value.weights is not None else 0
        for position, index in enumerate(indices):
            kind = "model" if position < model_count else "exploration"
            selected.append({**remaining[index], "selection_kind": kind})
    return selected


def expert_census(records: Iterable[dict]) -> dict[str, int]:
    counts = {expert: 0 for expert in EXPERTS}
    counts.update({"multi_expert": 0, "unique_endpoints": 0})
    for record in records:
        experts = _experts(record)
        for expert in experts:
            counts[expert] += 1
        counts["multi_expert"] += int(len(experts) > 1)
        counts["unique_endpoints"] += 1
    return counts


__all__ = [
    "EXPERTS",
    "attach_features",
    "durable_phase_action",
    "expert_census",
    "integrated_features",
    "merge_expert_pools",
    "migrate_v1_checkpoint",
    "proposal_collection_action",
    "query_collection_action",
    "select_batch",
]
