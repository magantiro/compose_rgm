"""Endpoint-only T4 allocation, separate from legal trajectory support.

The production property evaluator computes v from the frozen QED, SA and
original-seed similarity constraints. This module does not introduce a ring
catalog, quality reward, or additional medicinal-chemistry threshold.
"""

from __future__ import annotations

import math
from numbers import Real

LEGACY_RANK_ALL = "legacy_rank_all_v1"
T4_FEASIBLE_ONLY = "t4_feasible_only_v1"


def calculate_properties(molecule, *, seed_fp, generator, sa_scorer, delta, qed_min, sa_max):
    """The existing T4 property calculation, shared by generation and saved docking."""
    from rdkit import DataStructs
    from rdkit.Chem import QED

    if molecule is None:
        return None
    q = float(QED.qed(molecule))
    sa = float(sa_scorer(molecule))
    sim = float(DataStructs.TanimotoSimilarity(seed_fp, generator.GetFingerprint(molecule)))
    v = max(
        max(0.0, qed_min - q) / qed_min,
        max(0.0, sa - sa_max) / sa_max,
        max(0.0, delta - sim) / delta,
    )
    return {"qed": q, "sa": sa, "sim": sim, "v": v}


def validate_policy(policy: str) -> None:
    if policy not in (LEGACY_RANK_ALL, T4_FEASIBLE_ONLY):
        raise ValueError(f"unknown T4 endpoint selection policy: {policy!r}")


def feasible_endpoint(candidate: dict) -> bool:
    for field in ("qed", "sa", "sim", "v"):
        value = candidate.get(field)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"endpoint has missing or nonfinite property {field}: {value!r}")
    if candidate["v"] < 0:
        raise ValueError("endpoint violation v must be nonnegative")
    return candidate["v"] == 0


def annotate_endpoints(pool: list[dict], policy: str) -> list[dict]:
    """Keep every record; mark eligibility only under the explicit new policy."""
    validate_policy(policy)
    if policy == LEGACY_RANK_ALL:
        return pool
    annotated = []
    for candidate in pool:
        eligible = feasible_endpoint(candidate)
        annotated.append(
            {
                **candidate,
                "oracle_eligible": eligible,
                "endpoint_exclusion_reasons": []
                if eligible
                else ["existing_t4_constraint_violation"],
            }
        )
    return annotated
