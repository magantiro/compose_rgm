"""Zero-oracle endpoint funnel audit for frozen T4 proposal streams.

The audit observes the existing production expansion routine without changing its
random-number stream or endpoint-admission behavior.  It records every distinct
executed endpoint before the task fiber removes it, which separates synthesis,
execution, structural validity, benchmark constraints, and final admission.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from rdkit import Chem
from rdkit.Chem import QED, DataStructs

from compose_v4.experiments import t4_fiber_campaign as campaign

SCHEMA_VERSION = "t4_proposal_funnel_audit_v1"


def assess_endpoint(
    seed_smiles: str,
    endpoint_smiles: str,
    *,
    delta: float,
    support: str,
) -> dict[str, Any]:
    """Measure every endpoint gate without short-circuiting later properties."""

    seed = Chem.MolFromSmiles(seed_smiles)
    if seed is None:
        raise ValueError("proposal-funnel seed SMILES is invalid")
    molecule = Chem.MolFromSmiles(endpoint_smiles) if endpoint_smiles else None
    connected = molecule is not None and "." not in endpoint_smiles
    if molecule is None:
        return {
            "smiles": endpoint_smiles,
            "parseable": False,
            "connected": False,
            "within_representation_cap": False,
            "structurally_valid": False,
            "similarity": None,
            "qed": None,
            "sa": None,
            "heavy": None,
            "heavy_delta": None,
            "similarity_margin": None,
            "qed_margin": None,
            "sa_margin": None,
            "similarity_pass": False,
            "qed_pass": False,
            "sa_pass": False,
            "compose_valid_pass": False,
            "normalized_max_deficit": None,
            "normalized_total_deficit": None,
        }

    canonical = Chem.MolToSmiles(molecule)
    generator = campaign.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fingerprint = generator.GetFingerprint(seed)
    similarity = float(
        DataStructs.TanimotoSimilarity(seed_fingerprint, generator.GetFingerprint(molecule))
    )
    qed = float(QED.qed(molecule))
    sa = float(campaign.sascorer.calculateScore(molecule))
    heavy = int(molecule.GetNumHeavyAtoms())
    seed_heavy = int(seed.GetNumHeavyAtoms())
    within_cap = heavy <= campaign.REPRESENTABLE_HEAVY_ATOMS
    structural = bool(connected and campaign.structurally_valid(canonical))
    similarity_margin = similarity - float(delta)
    qed_margin = qed - campaign.QED_MIN
    sa_margin = campaign.SA_MAX - sa
    similarity_pass = similarity_margin >= 0.0
    qed_pass = qed_margin >= 0.0
    sa_pass = sa_margin >= 0.0
    compose_pass = bool(
        connected
        and within_cap
        and similarity_pass
        and qed_pass
        and sa_pass
        and (support == campaign.BENCHMARK_ONLY or structural)
        and (support != campaign.LEGACY_SCREENED or not campaign.instability(canonical))
    )
    normalized = (
        max(0.0, -similarity_margin) / float(delta),
        max(0.0, -qed_margin) / campaign.QED_MIN,
        max(0.0, -sa_margin) / campaign.SA_MAX,
    )
    return {
        "smiles": canonical,
        "parseable": True,
        "connected": bool(connected),
        "within_representation_cap": within_cap,
        "structurally_valid": structural,
        "similarity": similarity,
        "qed": qed,
        "sa": sa,
        "heavy": heavy,
        "heavy_delta": heavy - seed_heavy,
        "similarity_margin": similarity_margin,
        "qed_margin": qed_margin,
        "sa_margin": sa_margin,
        "similarity_pass": similarity_pass,
        "qed_pass": qed_pass,
        "sa_pass": sa_pass,
        "compose_valid_pass": compose_pass,
        "normalized_max_deficit": max(normalized),
        "normalized_total_deficit": sum(normalized),
    }


class _AuditedFiber:
    def __init__(self, seed_smiles: str, delta: float, support: str):
        self.base = campaign.Fiber(seed_smiles, delta, support=support)
        self.seed_smiles = seed_smiles
        self.delta = self.base.delta
        self.support = support
        self.generator = self.base.generator
        self.check_calls = 0
        self.endpoint_counts: Counter[str] = Counter()
        self.assessments: dict[str, dict[str, Any]] = {}

    def check(self, smiles: str) -> dict[str, Any] | None:
        self.check_calls += 1
        assessment = assess_endpoint(
            self.seed_smiles, smiles, delta=self.delta, support=self.support
        )
        key = assessment["smiles"] if assessment["parseable"] else str(smiles)
        self.endpoint_counts[key] += 1
        self.assessments.setdefault(key, assessment)
        result = self.base.check(smiles)
        observed = result is not None
        if observed != bool(assessment["compose_valid_pass"]):
            raise RuntimeError("endpoint assessment disagrees with production Fiber.check")
        return result


@contextmanager
def _observe_production_calls(counters: Counter[str]) -> Iterator[None]:
    """Wrap module globals used by ``expand`` and restore them after one audit."""

    originals = {
        "synthesize_dynamic_program": campaign.synthesize_dynamic_program,
        "synthesize_anchored_replacement_program": (
            campaign.synthesize_anchored_replacement_program
        ),
        "extract_structural_goal": campaign.extract_structural_goal,
        "instantiate_goal": campaign.instantiate_goal,
        "molecular_graph_to_smiles": campaign.molecular_graph_to_smiles,
    }

    def wrap(name: str):
        original = originals[name]

        def observed(*args, **kwargs):
            counters[f"{name}_attempted"] += 1
            try:
                result = original(*args, **kwargs)
            except Exception as error:
                counters[f"{name}_failed"] += 1
                counters[f"{name}_failure:{type(error).__name__}"] += 1
                raise
            counters[f"{name}_completed"] += 1
            return result

        return observed

    try:
        for name in originals:
            setattr(campaign, name, wrap(name))
        yield
    finally:
        for name, original in originals.items():
            setattr(campaign, name, original)


def _summarize_assessments(
    assessments: list[dict[str, Any]], *, admitted_smiles: set[str]
) -> dict[str, Any]:
    parsed = [row for row in assessments if row["parseable"] and row["connected"]]
    within_cap = [row for row in parsed if row["within_representation_cap"]]
    structural = [row for row in within_cap if row["structurally_valid"]]
    after_similarity = [row for row in structural if row["similarity_pass"]]
    after_qed = [row for row in after_similarity if row["qed_pass"]]
    after_sa = [row for row in after_qed if row["sa_pass"]]
    eligible = [row for row in assessments if row["compose_valid_pass"]]
    closest = sorted(
        structural,
        key=lambda row: (
            row["normalized_max_deficit"],
            row["normalized_total_deficit"],
            row["smiles"],
        ),
    )[:10]
    independent = {
        "parseable_connected": len(parsed),
        "within_representation_cap": sum(
            row["within_representation_cap"] for row in assessments
        ),
        "structurally_valid": sum(row["structurally_valid"] for row in assessments),
        "similarity_pass": sum(row["similarity_pass"] for row in assessments),
        "qed_pass": sum(row["qed_pass"] for row in assessments),
        "sa_pass": sum(row["sa_pass"] for row in assessments),
        "all_compose_valid": len(eligible),
    }
    cumulative = {
        "distinct_executed_endpoints": len(assessments),
        "parseable_connected": len(parsed),
        "within_representation_cap": len(within_cap),
        "structurally_valid": len(structural),
        "then_similarity_pass": len(after_similarity),
        "then_qed_pass": len(after_qed),
        "then_sa_pass": len(after_sa),
        "all_compose_valid": len(eligible),
        "admitted_to_candidate_pool": len(admitted_smiles),
    }
    failure_combinations = Counter()
    for row in assessments:
        failures = []
        if not row["parseable"] or not row["connected"]:
            failures.append("parse_or_connectivity")
        if not row["within_representation_cap"]:
            failures.append("representation_cap")
        if not row["structurally_valid"]:
            failures.append("structural_validity")
        if not row["similarity_pass"]:
            failures.append("similarity")
        if not row["qed_pass"]:
            failures.append("qed")
        if not row["sa_pass"]:
            failures.append("sa")
        failure_combinations["+".join(failures) if failures else "eligible"] += 1
    return {
        "independent_gate_counts": independent,
        "cumulative_funnel": cumulative,
        "failure_combinations": dict(sorted(failure_combinations.items())),
        "heavy_delta_counts": {
            "shrink_7plus": sum((row["heavy_delta"] or 0) <= -7 for row in parsed),
            "shrink_1_to_6": sum(-7 < (row["heavy_delta"] or 0) <= -1 for row in parsed),
            "unchanged": sum(row["heavy_delta"] == 0 for row in parsed),
            "growth": sum((row["heavy_delta"] or 0) > 0 for row in parsed),
        },
        "closest_to_feasible": closest,
    }


def audit_expansion_lane(
    *,
    cell: str,
    seed_smiles: str,
    delta: float,
    support: str,
    lane: str,
    proposal_seed: int,
    draws: int,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay one exact production lane and expose its pre-admission funnel."""

    import numpy as np

    if lane not in {"shallow", "anchored_replacement"}:
        raise ValueError("expansion audit supports shallow or anchored_replacement")
    audited_fiber = _AuditedFiber(seed_smiles, delta, support)
    counters: Counter[str] = Counter()
    started = perf_counter()
    with _observe_production_calls(counters):
        admitted = campaign.expand(
            seed_smiles,
            0.0,
            audited_fiber,
            np.random.default_rng(proposal_seed),
            draws=draws,
            multi_region=True,
            horizon=horizon,
            proposal_lane=lane,
        )
    assessments = sorted(audited_fiber.assessments.values(), key=lambda row: row["smiles"])
    admitted_smiles = {row["smiles"] for row in admitted}
    eligible_smiles = {
        row["smiles"]
        for row in assessments
        if row["compose_valid_pass"] and row["smiles"] != seed_smiles
    }
    if admitted_smiles != eligible_smiles:
        raise RuntimeError("production candidate pool differs from audited eligible endpoints")
    synthesis_name = (
        "synthesize_dynamic_program"
        if lane == "shallow"
        else "synthesize_anchored_replacement_program"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "lane": lane,
        "proposal_seed": proposal_seed,
        "draws": draws,
        "horizon": horizon,
        "programs_attempted": counters[f"{synthesis_name}_attempted"],
        "programs_synthesized": counters[f"{synthesis_name}_completed"],
        "program_synthesis_failures": counters[f"{synthesis_name}_failed"],
        "structural_goals_extracted": counters["extract_structural_goal_completed"],
        "goal_extraction_failures": counters["extract_structural_goal_failed"],
        "instantiations_completed": counters["instantiate_goal_completed"],
        "instantiation_failures": counters["instantiate_goal_failed"],
        "endpoint_check_calls": audited_fiber.check_calls,
        "duplicate_endpoint_check_calls": audited_fiber.check_calls - len(assessments),
        "production_candidates": len(admitted),
        "elapsed_seconds": perf_counter() - started,
        "observer_counters": dict(sorted(counters.items())),
        **_summarize_assessments(assessments, admitted_smiles=admitted_smiles),
        "new_oracle_calls": 0,
    }
    return summary, assessments


def audit_route_candidates(
    *,
    cell: str,
    seed_smiles: str,
    delta: float,
    support: str,
    proposed_pool: int,
    realization_limit: int,
    telemetry: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Assess the sealed route pool without regenerating or refitting it."""

    by_smiles: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        assessment = assess_endpoint(
            seed_smiles, candidate["smiles"], delta=delta, support=support
        )
        assessment.update(
            {
                "created": int(candidate["created"]),
                "deleted": int(candidate["deleted"]),
                "regions": int(candidate["regions"]),
                "route_proposal_rank": int(candidate["route_proposal_rank"]),
                "route_template_ids": list(candidate["route_template_ids"]),
                "realized_primitives": int(candidate["realized_primitives"]),
            }
        )
        by_smiles.setdefault(assessment["smiles"], assessment)
    assessments = sorted(by_smiles.values(), key=lambda row: row["smiles"])
    admitted = {row["smiles"] for row in assessments if row["compose_valid_pass"]}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "lane": "route_complete_region",
        "programs_attempted": proposed_pool,
        "programs_selected_for_realization": realization_limit,
        "programs_synthesized": int(telemetry["complete_programs_committed"]),
        "program_synthesis_failures": realization_limit
        - int(telemetry["complete_programs_committed"]),
        "realization_status_counts": dict(telemetry["realization_status_counts"]),
        "endpoint_check_calls": len(candidates),
        "duplicate_endpoint_check_calls": len(candidates) - len(assessments),
        "production_candidates": len(admitted),
        **_summarize_assessments(assessments, admitted_smiles=admitted),
        "new_oracle_calls": 0,
    }
    return summary, assessments


__all__ = [
    "SCHEMA_VERSION",
    "assess_endpoint",
    "audit_expansion_lane",
    "audit_route_candidates",
]
