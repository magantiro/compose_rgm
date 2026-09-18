"""Exact historical assessment and score-blind lock for anchored replacements.

This module deliberately uses only exact canonical endpoint identity.  Structural
similarity to a known high-scoring molecule is not a utility label and must not affect
prospective selection.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_prospect_assessment import endpoint_sha256
from compose_v4.experiments.t4_route_guided_support import jak2_motif_flags

SCHEMA_VERSION = "t4_anchored_replacement_candidate_lock_v1"


def unique_lane_candidates(
    attempts: Iterable[dict[str, Any]], *, lane: str
) -> list[dict[str, Any]]:
    """Return one deterministic record for every canonical endpoint in ``lane``."""

    by_endpoint: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        if attempt.get("lane") != lane:
            continue
        if attempt.get("error"):
            raise ValueError(f"proposal attempt failed: {attempt['error']}")
        for candidate in attempt.get("candidates", ()):
            smiles = str(candidate["smiles"])
            prior = by_endpoint.get(smiles)
            if prior is not None:
                chemical_fields = ("similarity", "qed", "sa", "heavy")
                if any(prior[name] != candidate[name] for name in chemical_fields):
                    raise ValueError(f"inconsistent duplicate candidate record: {smiles}")
                candidate = min(
                    (prior, candidate), key=lambda row: json.dumps(row, sort_keys=True)
                )
            by_endpoint[smiles] = candidate
    return [by_endpoint[key] for key in sorted(by_endpoint)]


def build_exact_assessment_and_lock(
    candidates: Iterable[dict[str, Any]],
    historical_records: Iterable[dict[str, Any]],
    *,
    cell: str,
    delta: float,
    root_smiles: str,
    docking_seed: int,
    input_sha256: dict[str, str],
) -> dict[str, Any]:
    """Join basin candidates to exact historical labels and lock every novel one.

    The lock contains all and only eligible amide-plus-diamine candidates.  Exact
    rediscoveries are excluded because their historical charged score is already known.
    No score, structural distance, or model ranking orders the novel set.
    """

    basin = []
    for candidate in candidates:
        smiles = str(candidate["smiles"])
        if jak2_motif_flags(smiles)["basin"]:
            basin.append({**candidate, "endpoint_sha256": endpoint_sha256(smiles)})
    basin.sort(key=lambda row: row["endpoint_sha256"])
    if len({row["endpoint_sha256"] for row in basin}) != len(basin):
        raise ValueError("basin candidates are not unique by endpoint identity")

    historical: dict[str, list[dict[str, Any]]] = {}
    historical_rows = 0
    for record in historical_records:
        historical_rows += 1
        if record.get("cell") == cell:
            historical.setdefault(str(record["endpoint_sha256"]), []).append(record)

    hits, novel = [], []
    for row in basin:
        prior = historical.get(row["endpoint_sha256"], [])
        if prior:
            best = min(prior, key=lambda record: float(record["score_mean"]))
            hits.append(
                {
                    "endpoint": row["smiles"],
                    "endpoint_sha256": row["endpoint_sha256"],
                    "historical_score": float(best["score_mean"]),
                    "record_id": best.get("record_id"),
                    "receipt_ids": list(best.get("receipt_ids", ())),
                    "protocol": best.get("protocol"),
                    "arms": list(best.get("arms", ())),
                    "observation_count": int(best.get("observation_count", 1)),
                }
            )
        else:
            novel.append(
                {
                    "endpoint": row["smiles"],
                    "endpoint_sha256": row["endpoint_sha256"],
                    "properties": {
                        name: row[name] for name in ("similarity", "qed", "sa", "heavy")
                    },
                    "program_families": list(row.get("program_families", ())),
                    "created": int(row.get("created", 0)),
                    "deleted": int(row.get("deleted", 0)),
                    "regions": int(row.get("regions", 0)),
                }
            )

    lock_body = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "target": cell.rsplit("_", 1)[0],
        "delta": delta,
        "root_smiles": root_smiles,
        "docking_seed": docking_seed,
        "support": "compose_valid",
        "selection": novel,
        "charged_calls": len(novel),
        "new_oracle_call_limit": len(novel),
        "automatic_retries": 0,
        "replacement_after_scoring": False,
        "selection_rule": (
            "all exact-historical-novel eligible amide-plus-diamine endpoints from "
            "the prospectively frozen anchored-replacement support pool, ordered by "
            "endpoint digest; no docking score or structural-distance surrogate is used"
        ),
        "input_sha256": dict(sorted(input_sha256.items())),
    }
    lock = {**lock_body, "lock_id": identity(lock_body)}
    return {
        "schema_version": "t4_anchored_replacement_assessment_v1",
        "cell": cell,
        "historical_rows_total": historical_rows,
        "historical_unique_endpoints_in_cell": len(historical),
        "eligible_anchored_basin_endpoints": len(basin),
        "exact_historical_rediscoveries": len(hits),
        "historical_hits": sorted(hits, key=lambda row: row["historical_score"]),
        "prospective_novel_endpoints": len(novel),
        "lock": lock,
        "new_oracle_calls": 0,
        "interpretation": (
            "exact rediscovery is free evidence that the repaired proposal law reaches "
            "historically useful chemistry; it does not label novel locked endpoints"
        ),
    }


def verify_lock(lock: dict[str, Any]) -> None:
    """Reject a modified or internally inconsistent candidate lock."""

    body = {key: value for key, value in lock.items() if key != "lock_id"}
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a {SCHEMA_VERSION} lock")
    if identity(body) != lock.get("lock_id"):
        raise ValueError("candidate lock was modified after sealing")
    if len(lock["selection"]) != lock["charged_calls"]:
        raise ValueError("candidate lock disagrees with its charged-call count")
    if lock["charged_calls"] > lock["new_oracle_call_limit"]:
        raise ValueError("candidate lock exceeds its call limit")
    digests = [row["endpoint_sha256"] for row in lock["selection"]]
    if len(digests) != len(set(digests)):
        raise ValueError("candidate lock charges a duplicate endpoint")
