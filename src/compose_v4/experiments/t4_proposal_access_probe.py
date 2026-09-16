"""Actual-sampler support probe for the T4 reset proposal path.

The strategy report measured a concrete autonomous-support loss on JAK2: the v1
ring panel that had to supply the productive construction could only ever be one of
four deterministically seeded panels, so a compiler that reached the endpoint when
the target was supplied could not reach it through the runtime sampler. That is a
support question, and it has to be asked of the production path rather than of a
teacher-forced compiler call.

This module asks it for the reset revision, over two lanes:

* the **bootstrap** lane starts from the cell's own benchmark root with no archive
  and no teacher input, and is the autonomous measurement;
* the **archive** lane injects one historical champion for diagnostics only and
  measures conditional support around a known-good measured endpoint.

Three things are decided, all zero-oracle: a supplied known-good program still
realizes exactly, repeated proposal keeps exploring rather than repeating, and no
declared generic family is attempted repeatedly yet never realized anywhere.
Failing to sample an exact historical endpoint inside this budget is not a failure
and is never a proof of zero support.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v1 import GENERIC_MODULES
from compose_v4.control.objective_program_search import (
    ObjectiveProgramSearch,
    ObjectiveSearchConfig,
    v0_search_config,
)
from compose_v4.control.progressive_bootstrap import ProgressiveBootstrap
from compose_v4.experiments.t4_frozen_program_benchmark import strict_endpoint_scorer
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA_VERSION = "t4_proposal_access_probe_v1"

# A probe budget is an attempt count, so the operational wall cap is lifted. Elapsed
# seconds are still recorded; only the attempt count is allowed to bind.
PROBE_WALL_SECONDS = 86400.0


def declared_families() -> tuple[str, ...]:
    """The authoritative generic-module registry, read rather than retyped."""
    return tuple(GENERIC_MODULES)


# ---- Family census ----


def family_census(attempts) -> dict:
    """Which declared families were realized, and which were tried and refused.

    Both proposal lanes report `modules[].family` for what they built and
    `module_failure_counts` keyed `"<family>:<reason>"` for what they tried and could
    not build. The second is the stronger signal: a family that is attempted often
    and realized never is an actual runtime support exclusion, not simply an absence
    from a finite sample.
    """
    realized, attempted, reasons = Counter(), Counter(), Counter()
    for attempt in attempts:
        metadata = attempt.get("metadata") or {}
        for payload in _metadata_payloads(metadata):
            for module in payload.get("modules") or []:
                family = module.get("family")
                if family:
                    realized[family] += 1
                    attempted[family] += 1
            for key, count in (payload.get("module_failure_counts") or {}).items():
                family = str(key).split(":", 1)[0]
                attempted[family] += count
                reasons[key] += count
    return {
        "realized": dict(sorted(realized.items())),
        "attempted": dict(sorted(attempted.items())),
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])[:40]),
    }


def _metadata_payloads(metadata) -> list[dict]:
    """Attempt metadata is either a lane payload or a one-key wrapper around one."""
    if not isinstance(metadata, dict):
        return []
    if "modules" in metadata or "module_failure_counts" in metadata:
        return [metadata]
    return [value for value in metadata.values() if isinstance(value, dict)]


def status_census(attempts) -> dict:
    return {
        "attempts": len(attempts),
        "by_status": dict(sorted(Counter(a.get("status", "missing") for a in attempts).items())),
        "by_lane": dict(
            sorted(Counter(a.get("planner_channel", "unassigned") for a in attempts).items())
        ),
    }


# ---- Lanes ----


def attempt_signature(attempt) -> str:
    """Identity of one proposal, not of the molecule it happens to reach.

    The historical defect was a bootstrap that re-emitted the *same* deterministic
    proposals after an empty round. Two different proposals can legitimately land on
    the same endpoint when the legal space is small, so endpoint identity conflates
    that chemistry with the defect. This signature covers the lane, the chosen
    modules and their parameters as well as the endpoint.
    """
    return identity(
        {
            "planner_channel": attempt.get("planner_channel"),
            "endpoint": attempt.get("endpoint"),
            "status": attempt.get("status"),
            "reason": attempt.get("reason"),
            "modules": [
                {"family": module.get("family"), "parameters": module.get("parameters")}
                for payload in _metadata_payloads(attempt.get("metadata") or {})
                for module in payload.get("modules") or []
            ],
        }
    )


def bootstrap_lane(source, config, *, source_group, oracle_protocol, eligibility, rounds) -> dict:
    """Teacher-free cold start; the repetition repair is measured across rounds."""
    bootstrap = ProgressiveBootstrap(
        source, config, source_group=source_group, oracle_protocol=oracle_protocol
    )
    began = perf_counter()
    rows, attempts, seen, signatures = [], [], set(), set()
    for index in range(rounds):
        batch = bootstrap.next_batch(eligibility)
        endpoints = [a["endpoint"] for a in batch["attempts"] if a.get("endpoint")]
        drawn = {attempt_signature(a) for a in batch["attempts"]}
        fresh_endpoints = set(endpoints) - seen
        fresh_signatures = drawn - signatures
        seen.update(endpoints)
        signatures.update(drawn)
        attempts.extend(batch["attempts"])
        rows.append(
            {
                "round": index,
                "attempts": len(batch["attempts"]),
                "eligible": len(batch["candidates"]),
                "endpoints": len(endpoints),
                "new_endpoints": len(fresh_endpoints),
                "new_proposals": len(fresh_signatures),
                "cursor_after": batch["cursor_after"],
                "proposal_seconds": batch["proposal_seconds"],
            }
        )
    total = sum(row["attempts"] for row in rows)
    return {
        "lane": "bootstrap",
        "teacher_free": True,
        "rounds": rows,
        "unique_endpoints": len(seen),
        "distinct_proposals": len(signatures),
        "distinct_proposal_fraction": len(signatures) / total if total else 0.0,
        "eligible": sum(row["eligible"] for row in rows),
        "elapsed_seconds": perf_counter() - began,
        "explores_after_first_round": all(row["new_proposals"] > 0 for row in rows[1:]),
        "new_endpoints_after_first_round": all(row["new_endpoints"] > 0 for row in rows[1:]),
        **status_census(attempts),
        "families": family_census(attempts),
        "endpoints": sorted(seen),
        "new_oracle_calls": 0,
    }


def archive_lane(champion, config, policy, *, eligibility, receipt_id, score) -> dict:
    """Answer-known conditional support around one historical measured endpoint."""
    search = ObjectiveProgramSearch(
        config,
        policy,
        source_group=champion["source_group"],
        oracle_protocol=champion["oracle_protocol"],
    )
    began = perf_counter()
    # Admission replays the supplied program and asserts endpoint identity, so this
    # is the exact-realization control rather than a separate reimplementation.
    search.add_measured_program(champion, receipt_id=receipt_id, score=score)
    batch = search.propose_batch(eligibility, limit=1)
    pool = batch["proposal_pool"]["candidates"]
    attempts = batch["attempts"]
    return {
        "lane": "archive",
        "teacher_free": False,
        "injected_for_diagnostics_only": True,
        "archive_exact_admission": True,
        "pool": len(pool),
        "unique_pool_endpoints": len({candidate["endpoint"] for candidate in pool}),
        "elapsed_seconds": perf_counter() - began,
        **status_census(attempts),
        "families": family_census(attempts),
        "endpoints": sorted({candidate["endpoint"] for candidate in pool}),
        "new_oracle_calls": 0,
    }


# ---- Context ----


def probe_config(seed: int, *, attempts: int, candidates: int):
    return replace(
        v0_search_config(seed=seed),
        attempts_per_batch=attempts,
        candidates_per_batch=candidates,
        wall_seconds=PROBE_WALL_SECONDS,
    )


def probe_context(result, unit, contract, *, index: int, similarity=None) -> dict:
    """One probe context: teacher-free bootstrap plus answer-known archive support.

    `unit` is the revision's own frozen benchmark start graph. The historical result
    carries a champion and an evaluator protocol but no root state, so the two are
    cross-checked rather than one being trusted to describe the other.
    """
    lanes = contract["lanes"]
    historical_unit = result["unit"]
    for field in ("cell", "original_seed", "oracle_protocol"):
        if unit[field] != historical_unit[field]:
            raise ValueError(
                f"probe context disagrees with its historical result on {field}: "
                f"{unit[field]!r} != {historical_unit[field]!r}"
            )
    champion = result["champion"]["candidate"]
    seed = int(np.random.SeedSequence([contract["seeds"]["base"], index]).generate_state(1)[0])
    eligibility = strict_endpoint_scorer(unit["original_seed"], delta=contract["delta"])

    bootstrap_config = probe_config(
        seed,
        attempts=lanes["bootstrap"]["attempts_per_round"],
        candidates=lanes["bootstrap"]["attempts_per_round"],
    )
    boot = bootstrap_lane(
        decode_state(unit["source_state"]),
        bootstrap_config,
        source_group=unit["source_group"],
        oracle_protocol=unit["oracle_protocol"],
        eligibility=eligibility,
        rounds=lanes["bootstrap"]["rounds"],
    )

    archive_config = probe_config(
        seed,
        attempts=lanes["archive"]["shallow_attempts"],
        candidates=lanes["archive"]["shallow_attempts"],
    )
    policy = ObjectiveSearchConfig(
        structured_attempts=lanes["archive"]["structured_attempts"],
        structured_candidates=lanes["archive"]["structured_attempts"],
        structured_wall_seconds=PROBE_WALL_SECONDS,
        query_batch_size=1,
    )
    arc = archive_lane(
        champion,
        archive_config,
        policy,
        eligibility=eligibility,
        receipt_id=result["champion"]["receipt_id"],
        score=result["champion"]["score"],
    )

    row = {
        "cell": unit["cell"],
        "seed": seed,
        "champion_score": result["champion"]["score"],
        "configuration": {"bootstrap": asdict(bootstrap_config), "archive": asdict(archive_config)},
        "policy": asdict(policy),
        "bootstrap": boot,
        "archive": arc,
        "new_oracle_calls": 0,
    }
    if similarity is not None:
        row["historical_proximity"] = {
            lane: similarity(row[lane]["endpoints"]) for lane in ("bootstrap", "archive")
        }
    for lane in ("bootstrap", "archive"):
        row[lane] = {k: v for k, v in row[lane].items() if k != "endpoints"}
    return row


# ---- Gate ----


def support_decision(rows, contract) -> dict:
    """PASS only if realization, exploration and family support all hold."""
    gate = contract["gate"]
    minimum = gate["minimum_attempts_for_exclusion"]
    families = declared_families()
    pooled_realized, pooled_attempted = Counter(), Counter()
    for row in rows:
        for lane in ("bootstrap", "archive"):
            pooled_realized.update(row[lane]["families"]["realized"])
            pooled_attempted.update(row[lane]["families"]["attempted"])
    failures = []
    if not all(row["archive"]["archive_exact_admission"] for row in rows):
        failures.append("a supplied historical program failed exact archive admission")
    for row in rows:
        if not row["bootstrap"]["explores_after_first_round"]:
            failures.append(
                f"{row['cell']}: a bootstrap round after the first drew no new proposal"
            )
    excluded = sorted(
        family
        for family in families
        if pooled_attempted[family] >= minimum and pooled_realized[family] == 0
    )
    for family in excluded:
        failures.append(
            f"family {family} was attempted {pooled_attempted[family]} times and never realized"
        )
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "declared_families": list(families),
        "pooled_realized": dict(sorted(pooled_realized.items())),
        "pooled_attempted": dict(sorted(pooled_attempted.items())),
        "never_realized_anywhere": sorted(
            family for family in families if pooled_realized[family] == 0
        ),
        "support_excluded": excluded,
        "per_context_absent": {
            row["cell"]: sorted(
                family
                for family in families
                if not any(
                    row[lane]["families"]["realized"].get(family)
                    for lane in ("bootstrap", "archive")
                )
            )
            for row in rows
        },
    }


# ---- Historical proximity ----


def similarity_reporter(historical):
    """Nearest historically scored endpoint for each generated one.

    The historical score stays attached to the historical molecule. Nothing here
    assigns a docking value to a generated molecule.
    """
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    rdBase.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return None if mol is None else generator.GetFingerprint(mol)

    reference = [
        (fingerprint(smiles), smiles, score) for smiles, score in sorted(historical.items())
    ]
    reference = [entry for entry in reference if entry[0] is not None]

    def report(endpoints):
        if not reference:
            return {"historical_endpoints": 0, "abstained": True}
        # Each generated molecule contributes its nearest historical molecule. Ties in
        # similarity are broken toward the BETTER (lower) historical score, because the
        # question is which scored neighbourhoods the sampler reaches; breaking toward
        # the worse score would report the least informative member of a tie.
        nearest, exact_scores = [], []
        for smiles in endpoints:
            if smiles in historical:
                exact_scores.append(historical[smiles])
            query = fingerprint(smiles)
            if query is None:
                continue
            scores = DataStructs.BulkTanimotoSimilarity(query, [r[0] for r in reference])
            top = max(scores)
            best_score = min(
                entry[2] for value, entry in zip(scores, reference, strict=True) if value == top
            )
            nearest.append((float(top), float(best_score)))
        if not nearest:
            return {"historical_endpoints": len(reference), "generated_scored": 0}
        similarities = sorted(value for value, _ in nearest)
        return {
            "historical_endpoints": len(reference),
            "generated_scored": len(nearest),
            "exact_historical_endpoint_hits": len(exact_scores),
            "best_historical_score_among_exact_hits": min(exact_scores) if exact_scores else None,
            "median_nearest_similarity": float(np.median(similarities)),
            "maximum_nearest_similarity": similarities[-1],
            "best_historical_score_among_nearest_molecules": min(score for _, score in nearest),
            "best_historical_score_available_in_this_cell": min(historical.values()),
            "interpretation": (
                "every score here belongs to a historical molecule; no generated molecule "
                "is given a docking value, and section 7 of the strategy report measured "
                "that structural proximity is not a dependable utility label"
            ),
        }

    return report


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
