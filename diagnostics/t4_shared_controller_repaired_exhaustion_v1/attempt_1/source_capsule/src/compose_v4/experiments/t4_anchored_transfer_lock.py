"""Score-blind lock for a transferred anchored-replacement candidate pool."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_prospect_assessment import endpoint_sha256

SCHEMA_VERSION = "t4_anchored_transfer_candidate_lock_v1"


def _diverse_selection(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deterministic max-min fingerprint selection with digest-only tie breaking."""

    if limit < 1:
        raise ValueError("selection limit must be positive")
    if len(rows) <= limit:
        return sorted(rows, key=lambda row: row["endpoint_sha256"])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    for row in rows:
        molecule = Chem.MolFromSmiles(row["endpoint"])
        if molecule is None:
            raise ValueError(f"locked endpoint does not parse: {row['endpoint']!r}")
        fingerprints[row["endpoint_sha256"]] = generator.GetFingerprint(molecule)

    remaining = {row["endpoint_sha256"]: row for row in rows}
    first = min(remaining)
    chosen = [remaining.pop(first)]
    while remaining and len(chosen) < limit:
        best_digest = None
        best_distance = -1.0
        for digest in sorted(remaining):
            closest = max(
                DataStructs.TanimotoSimilarity(
                    fingerprints[digest], fingerprints[row["endpoint_sha256"]]
                )
                for row in chosen
            )
            distance = 1.0 - closest
            if distance > best_distance:
                best_digest = digest
                best_distance = distance
        assert best_digest is not None
        chosen.append(remaining.pop(best_digest))
    return chosen


def build_transfer_lock(
    candidates: Iterable[dict[str, Any]],
    historical_records: Iterable[dict[str, Any]],
    *,
    cell: str,
    delta: float,
    root_smiles: str,
    docking_seed: int,
    selection_limit: int,
    input_sha256: dict[str, str],
) -> dict[str, Any]:
    """Exclude exact historical endpoints, then diversity-lock novel candidates.

    Structural proximity to a known winner and all docking values are excluded from
    selection. Historical scores are reported only for exact endpoint rediscoveries.
    """

    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        endpoint = str(candidate["smiles"])
        digest = endpoint_sha256(endpoint)
        if digest in unique and unique[digest]["endpoint"] != endpoint:
            raise ValueError("endpoint digest collision")
        unique[digest] = {
            "endpoint": endpoint,
            "endpoint_sha256": digest,
            "properties": {
                name: candidate[name] for name in ("similarity", "qed", "sa", "heavy")
            },
            "program_families": list(candidate.get("program_families", ())),
            "interventions": list(candidate.get("families", ())),
            "created": int(candidate.get("created", 0)),
            "deleted": int(candidate.get("deleted", 0)),
            "regions": int(candidate.get("regions", 0)),
        }

    historical: dict[str, list[dict[str, Any]]] = {}
    total_rows = 0
    for record in historical_records:
        total_rows += 1
        if record.get("cell") == cell:
            historical.setdefault(str(record["endpoint_sha256"]), []).append(record)

    hits, novel = [], []
    for digest, row in sorted(unique.items()):
        prior = historical.get(digest, ())
        if not prior:
            novel.append(row)
            continue
        best = min(prior, key=lambda record: float(record["score_mean"]))
        hits.append(
            {
                "endpoint": row["endpoint"],
                "endpoint_sha256": digest,
                "historical_score": float(best["score_mean"]),
                "record_id": best.get("record_id"),
                "observation_count": int(best.get("observation_count", 1)),
            }
        )

    selected = _diverse_selection(novel, selection_limit)
    lock_body = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "target": cell.rsplit("_", 1)[0],
        "delta": delta,
        "root_smiles": root_smiles,
        "docking_seed": docking_seed,
        "support": "compose_valid",
        "selection": selected,
        "charged_calls": len(selected),
        "new_oracle_call_limit": len(selected),
        "automatic_retries": 0,
        "replacement_after_scoring": False,
        "selection_rule": (
            "exact-historical-novel anchored endpoints selected by deterministic "
            "Morgan-radius-2 max-min diversity; the first and all ties are resolved "
            "by endpoint digest; no docking score or winner proximity is used"
        ),
        "input_sha256": dict(sorted(input_sha256.items())),
    }
    lock = {**lock_body, "lock_id": identity(lock_body)}
    return {
        "schema_version": "t4_anchored_transfer_assessment_v1",
        "cell": cell,
        "candidate_endpoints": len(unique),
        "historical_rows_total": total_rows,
        "historical_unique_endpoints_in_cell": len(historical),
        "exact_historical_rediscoveries": len(hits),
        "historical_hits": sorted(hits, key=lambda row: row["historical_score"]),
        "best_historical_rediscovery": (
            min((row["historical_score"] for row in hits), default=None)
        ),
        "prospective_novel_endpoints": len(novel),
        "lock": lock,
        "new_oracle_calls": 0,
    }


def verify_transfer_lock(lock: dict[str, Any]) -> None:
    """Reject a changed or internally inconsistent transfer lock."""

    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a {SCHEMA_VERSION} lock")
    body = {key: value for key, value in lock.items() if key != "lock_id"}
    if identity(body) != lock.get("lock_id"):
        raise ValueError("transfer candidate lock was modified after sealing")
    selection = lock.get("selection", ())
    if len(selection) != lock.get("charged_calls"):
        raise ValueError("transfer lock disagrees with its charged-call count")
    digests = [row["endpoint_sha256"] for row in selection]
    if len(digests) != len(set(digests)):
        raise ValueError("transfer lock contains a duplicate endpoint")
