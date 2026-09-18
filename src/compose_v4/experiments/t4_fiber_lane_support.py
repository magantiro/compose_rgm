"""Matched zero-oracle summaries for shallow and progressive FiberControl lanes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from compose_v4.experiments.t4_route_guided_support import jak2_motif_flags

SCHEMA_VERSION = "t4_fiber_lane_support_v1"


def summarize_attempts(results: Iterable[dict[str, Any]], *, expected_attempts: int) -> dict:
    """Validate and summarize independently seeded proposal attempts for one lane."""

    rows = list(results)
    if len(rows) != expected_attempts:
        raise ValueError(f"expected {expected_attempts} attempts, received {len(rows)}")
    lanes = {str(row.get("lane")) for row in rows}
    if len(lanes) != 1 or None in lanes:
        raise ValueError(f"attempts must contain exactly one proposal lane: {sorted(lanes)}")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != expected_attempts:
        raise ValueError("proposal-attempt seeds must be unique")

    endpoint_records = [candidate for row in rows for candidate in row.get("candidates", ())]
    unique = {}
    for record in endpoint_records:
        smiles = str(record["smiles"])
        unique.setdefault(smiles, record)

    motifs = Counter()
    families = Counter()
    for smiles, record in unique.items():
        for name, present in jak2_motif_flags(smiles).items():
            motifs[name] += int(present)
        families.update(map(str, record.get("program_families", ())))

    return {
        "lane": lanes.pop(),
        "attempts": expected_attempts,
        "attempts_with_eligible_endpoint": sum(bool(row.get("candidates")) for row in rows),
        "eligible_records": len(endpoint_records),
        "unique_eligible_endpoints": len(unique),
        "eligible_endpoints_per_attempt": len(unique) / expected_attempts,
        "motif_counts_over_unique_eligible": dict(sorted(motifs.items())),
        "program_families_over_unique_eligible": dict(sorted(families.items())),
        "wall_seconds_sum": sum(float(row.get("elapsed_seconds", 0.0)) for row in rows),
        "worker_failures": sum(bool(row.get("error")) for row in rows),
        "new_oracle_calls": 0,
        "candidate_smiles": sorted(unique),
    }


def compare_lanes(shallow: dict, structured: dict) -> dict:
    """Predeclared support comparison; no docking or structural distance surrogate."""

    if shallow.get("lane") != "shallow" or structured.get("lane") != "structured":
        raise ValueError("comparison requires shallow then structured summaries")
    shallow_motifs = shallow["motif_counts_over_unique_eligible"]
    structured_motifs = structured["motif_counts_over_unique_eligible"]
    basin_gain = int(structured_motifs.get("basin", 0)) - int(shallow_motifs.get("basin", 0))
    return {
        "unique_eligible_gain": (
            int(structured["unique_eligible_endpoints"])
            - int(shallow["unique_eligible_endpoints"])
        ),
        "amide_gain": int(structured_motifs.get("amide", 0)) - int(shallow_motifs.get("amide", 0)),
        "diamine_ring_gain": (
            int(structured_motifs.get("diamine_ring", 0))
            - int(shallow_motifs.get("diamine_ring", 0))
        ),
        "basin_gain": basin_gain,
        "scored_pilot_support_gate": "PASS" if basin_gain > 0 else "FAIL",
        "gate_definition": (
            "structured must produce strictly more eligible amide-plus-diamine basin "
            "endpoints than shallow at the frozen equal attempt budget"
        ),
        "new_oracle_calls": 0,
    }
