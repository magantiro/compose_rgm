#!/usr/bin/env python3
"""Reversibility feasibility census over PREDECLARED constraint families.

Reads the committed stage-A shards; needs no Modal access and no new compute.

WHAT THIS DECIDES
-----------------
The ring-system family failed because violation was ABSORBING: 19 of 42
trajectories broke the motif and 0 came back. This census asks whether any
predeclared family is REVERSIBLE instead -- whether a trajectory can leave the
feasible set and be back inside it at the endpoint. That event is what makes
endpoint-only filtering insufficient, and therefore what makes the whole
pathwise/endpoint distinction real.

The families, their thresholds and the pass criteria were committed BEFORE this
script was written (see PROTOCOL.md addendum and DECISION_LOG.md). This script
only reads them.

WHAT IT CANNOT DECIDE
---------------------
Criterion V4 -- mask support retention and mask-empty frequency -- needs the
enumerated successor set at each state, which the shards do not store. A family
clearing V1-V3 is reported as CONDITIONAL_PASS_PENDING_V4, never as a full pass.

Status of the emitted artifact: SMOKE_HELD_IN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rdkit import RDLogger  # noqa: E402

from compose_v4.experiments.pathwise_reversible_families import (  # noqa: E402
    FAMILIES,
    FAMILY_ORDER,
    FAMILY_PRECEDENCE_RULE,
    HEAVY_ATOM_CORRIDOR,
    UNDESIRED_GROUPS,
    audit_trajectory,
    clogp_corridor,
    clogp_of,
    heavy_atoms_of,
)

RDLogger.DisableLog("rdApp.*")

#: Fixed in PROTOCOL.md before the census ran.
V2_RETURN_RATE_FLOOR = 0.10
V3_MIN_EVENTS = 20

#: Arms whose trajectories explored the UNCONSTRAINED support. The masked arms
#: are excluded: they were shaped by the ring-system mask, so they are not a
#: free sample of what the reference process does.
UNCONSTRAINED_ARMS = ("unconstrained_greedy", "unconstrained_verified",
                      "endpoint_only")


def collect_trajectories(shard_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(shard_dir.glob("*.json")):
        shard = json.loads(path.read_text())
        if shard.get("status") != "SMOKE_HELD_IN":
            continue
        for arm, body in shard["arms"].items():
            if arm not in UNCONSTRAINED_ARMS:
                continue
            pool = body.get("rollout_audits") or [{"trajectory": body["trajectory"]}]
            for index, roll in enumerate(pool):
                rows.append({
                    "source_index": shard["index"],
                    "arm": arm,
                    "rollout": index,
                    "trajectory": roll["trajectory"],
                })
    return rows


#: How far outside the corridor a violating state sits. Added AFTER seeing
#: family B's high return rate, as a guard against the obvious artifact: if
#: violations were tiny excursions a hair past the boundary, "reversible" would
#: be measurement noise rather than chemistry. It changes no threshold and no
#: verdict -- it only qualifies a number that already existed.
BOUNDARY_NOISE_LOGP = 0.10


def _corridor_bounds(family: str) -> tuple[float, float] | None:
    if family == "B_physchem_corridor":
        return clogp_corridor()
    if family == "C_size_corridor":
        return float(HEAVY_ATOM_CORRIDOR[0]), float(HEAVY_ATOM_CORRIDOR[1])
    return None


def _corridor_value(family: str, smiles: str) -> float:
    if family == "B_physchem_corridor":
        return clogp_of(smiles)
    return float(heavy_atoms_of(smiles))


def excursion_depths(family: str, rows: list[dict]) -> dict:
    """Depth of each violating excursion, for corridor families only."""
    bounds = _corridor_bounds(family)
    if bounds is None:
        return {"applicable": False,
                "note": "severity is categorical for a motif family"}
    low, high = bounds
    depths: list[float] = []
    for row in rows:
        values = [_corridor_value(family, key) for key in row["trajectory"]]
        if not values or not (low <= values[0] <= high):
            continue  # source already outside; it never left the set
        outside = [v for v in values[1:] if not (low <= v <= high)]
        if outside:
            depths.append(max(max(v - high, low - v, 0.0) for v in outside))
    if not depths:
        return {"applicable": True, "n": 0}
    depths.sort()
    width = high - low
    return {
        "applicable": True,
        "n": len(depths),
        "min": round(min(depths), 4),
        "median": round(depths[len(depths) // 2], 4),
        "max": round(max(depths), 4),
        "median_as_fraction_of_corridor_width": round(
            depths[len(depths) // 2] / width, 4) if width else None,
        "boundary_noise_excursions": sum(
            1 for d in depths if d < BOUNDARY_NOISE_LOGP),
        "boundary_noise_threshold": BOUNDARY_NOISE_LOGP,
        "interpretation": (
            "excursions well inside the corridor width are real departures; a "
            "high count of boundary-noise excursions would mean the return "
            "statistic is an artifact of where the bound happens to fall"
        ),
    }


def census_family(family: str, rows: list[dict]) -> dict:
    audits = []
    for row in rows:
        audit = audit_trajectory(family, row["trajectory"])
        audit["source_index"] = row["source_index"]
        audits.append(audit)

    # A trajectory whose SOURCE already violates cannot "leave" the feasible
    # set; it was never in it. Those are reported separately rather than
    # counted as violations, which would inflate criterion V1.
    startable = [a for a in audits if a["source_feasible"]]
    unstartable = len(audits) - len(startable)

    violators = [a for a in startable if a["any_violation"]]
    returned = [a for a in violators if a["returned"]]
    events = len(returned)  # endpoint-valid AND path-invalid

    denom = len(startable) or 1
    violation_rate = len(violators) / denom
    return_rate = len(returned) / len(violators) if violators else 0.0

    v1 = 0 < len(violators) < len(startable)
    v2 = return_rate >= V2_RETURN_RATE_FLOOR
    v3 = events >= V3_MIN_EVENTS
    verdict = ("CONDITIONAL_PASS_PENDING_V4" if (v1 and v2 and v3) else "FAIL")

    # what would it take to reach V3 at the observed rate?
    per_traj = events / denom if denom else 0.0
    projection = (
        int(-(-V3_MIN_EVENTS // per_traj)) if per_traj > 0 else None
    )

    return {
        "family": family,
        "description": FAMILIES[family].description,
        "threshold_provenance": FAMILIES[family].threshold_provenance,
        "trajectories_considered": len(audits),
        "source_already_infeasible": unstartable,
        "denominator_source_feasible": len(startable),
        "violating_trajectories": len(violators),
        "violation_rate": round(violation_rate, 4),
        "returned_to_feasible_endpoint": len(returned),
        "RETURN_RATE_among_violators": round(return_rate, 4),
        "endpoint_valid_path_invalid_events": events,
        "violation_count_per_violating_path": (
            round(statistics.fmean([a["violation_count"] for a in violators]), 3)
            if violators else None
        ),
        "criteria": {
            "V1_non_vacuous": v1,
            "V2_reversible_at_10pct": v2,
            "V3_at_least_20_events": v3,
            "V4_mask_leaves_room": "NOT_EVALUABLE_FROM_SHARDS",
        },
        "verdict": verdict,
        "failure_mode": (
            None if verdict != "FAIL"
            else "VACUOUS_no_violations" if not violators
            else "ABSORBING_violations_never_return" if not returned
            else "UNDERPOWERED_reversible_but_too_few_events"
        ),
        "trajectories_needed_for_V3_at_observed_rate": projection,
        "excursion_depth": excursion_depths(family, rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", type=Path,
        default=REPO / "diagnostics/pathwise_constraints_smoke_stageA_shards")
    parser.add_argument(
        "--out", type=Path,
        default=REPO / "diagnostics/pathwise_reversibility_census.json")
    args = parser.parse_args()

    rows = collect_trajectories(args.shards)
    if not rows:
        raise SystemExit(f"no usable trajectories under {args.shards}")

    families = [census_family(name, rows) for name in FAMILY_ORDER]
    passing = [f for f in families if f["verdict"] != "FAIL"]
    # PRECEDENCE: first in the declared order, never the most favourable.
    advance = passing[0]["family"] if passing else None

    payload = {
        "schema": "compose.pathwise.reversibility_census",
        "status": "SMOKE_HELD_IN",
        "held_out_opened": False,
        "new_compute_used": False,
        "provenance": (
            "computed from committed stage-A shards; families, thresholds, rank "
            "order and pass criteria were sealed in commit bf14d53 BEFORE this "
            "script was written"
        ),
        "trajectory_pool": {
            "trajectories": len(rows),
            "states": sum(len(r["trajectory"]) for r in rows),
            "arms_used": list(UNCONSTRAINED_ARMS),
            "note": ("masked arms excluded: their trajectories were shaped by "
                     "the ring-system mask and are not a free sample"),
        },
        "declared_thresholds": {
            "A_undesired_motif": sorted(UNDESIRED_GROUPS),
            "B_clogp_corridor": list(clogp_corridor()),
            "C_heavy_atom_corridor": list(HEAVY_ATOM_CORRIDOR),
        },
        "criteria": {
            "V2_return_rate_floor": V2_RETURN_RATE_FLOOR,
            "V3_min_events": V3_MIN_EVENTS,
            "V4": "requires enumerated successor sets; not stored in shards",
        },
        "precedence_rule": FAMILY_PRECEDENCE_RULE,
        "families": families,
        "families_passing_V1_V3": [f["family"] for f in passing],
        "advance": advance,
        "outcome": (
            f"advance {advance} (first passing in declared order)" if advance
            else "NO FAMILY PASSES -- pathwise constraints leave the main paper; "
                 "the ring-system negative goes to the appendix; no fourth "
                 "predicate is searched"
        ),
    }
    payload["census_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"pool: {len(rows)} unconstrained trajectories, "
          f"{payload['trajectory_pool']['states']} states\n")
    header = (f"{'family':22s} {'viol':>10s} {'returned':>9s} "
              f"{'ret.rate':>9s} {'events':>7s}  verdict")
    print(header)
    print("-" * len(header))
    for f in families:
        print(f"{f['family']:22s} "
              f"{f['violating_trajectories']:>4d}/{f['denominator_source_feasible']:<5d} "
              f"{f['returned_to_feasible_endpoint']:>9d} "
              f"{f['RETURN_RATE_among_violators']:>9.3f} "
              f"{f['endpoint_valid_path_invalid_events']:>7d}  {f['verdict']}")
    print(f"\n{payload['outcome']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
