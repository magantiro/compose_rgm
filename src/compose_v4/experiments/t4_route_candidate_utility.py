"""Score-blind candidate locks for the T4 route-policy utility diagnostic."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_program_policy import (
    RouteDistilledProgramPolicy,
)
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_frozen_program_benchmark import (
    strict_endpoint_scorer,
)
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import (
    ComparisonConfig,
    ContrastiveRanker,
    MarginalPolicy,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_route_policy_comparison import (
    _actor_candidates,
    _candidate_vector,
    generate_marginal_candidates,
)


def _envelope(path: Path, *, compressed: bool = False) -> dict:
    raw = (
        gzip.decompress(path.read_bytes()).decode() if compressed else path.read_text()
    )
    value = json.loads(raw)
    if set(value) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid sealed artifact: {path}")
    if identity(value["payload"]) != value["payload_sha256"]:
        raise ValueError(f"sealed artifact changed: {path}")
    return value["payload"]


def _marginal(checkpoint: dict) -> MarginalPolicy:
    return MarginalPolicy(
        tuple(checkpoint["family_probabilities"]),
        tuple(checkpoint["module_count_probabilities"]),
        checkpoint["exploration_floor"],
        checkpoint["training_identity"],
    )


def _ranker(checkpoint: dict) -> ContrastiveRanker:
    return ContrastiveRanker(
        tuple(checkpoint["mean"]),
        tuple(checkpoint["scale"]),
        tuple(checkpoint["coefficients"]),
        checkpoint["exploration_floor"],
        checkpoint["training_identity"],
    )


def _candidate_row(candidate, source, ranker, scorer, config) -> dict:
    endpoint = decode_state(candidate.endpoint_state)
    smiles = canonical_state_key(endpoint)
    properties = scorer({"smiles": smiles})
    reasons = list(properties["endpoint_exclusion_reasons"])
    if endpoint.n_real_atoms > 40:
        reasons.append("heavy_atom_support_exceeded")
    return {
        "attempt_id": candidate.attempt_id,
        "generation_index": candidate.generation_index,
        "smiles": smiles,
        "endpoint_state": candidate.endpoint_state,
        "actions": list(candidate.actions),
        "families": list(candidate.families),
        "blocks": candidate.blocks,
        "heavy_atoms": endpoint.n_real_atoms,
        "rank_score": ranker.score(_candidate_vector(source, candidate, config)),
        "qed": properties["qed"],
        "sa": properties["sa"],
        "sim": properties["sim"],
        "oracle_eligible": not reasons,
        "endpoint_exclusion_reasons": reasons,
    }


def lock_source_candidates(
    root: Path,
    *,
    result_path: Path,
    model_path: Path,
    source_index: int,
    delta: float = 0.4,
) -> dict:
    """Regenerate one sealed source shard and lock its top eligible endpoint per policy."""

    result = _envelope(result_path)
    models = _envelope(model_path, compressed=True)
    if result.get("costs") != {"oracle_calls": 0, "docking_calls": 0}:
        raise ValueError("candidate utility requires a zero-oracle source shard")
    reports = result.get("fold_reports", ())
    fold_models = models.get("fold_models", ())
    if len(reports) != 1 or len(fold_models) != 1:
        raise ValueError("candidate utility requires exactly one held-source report")
    report, saved = reports[0], fold_models[0]
    source_census = report.get("source_census", ())
    if len(source_census) != 1:
        raise ValueError("candidate utility source census changed")
    cell = source_census[0]["cell"]
    source_id = source_census[0]["source_id"]
    fold = report["fold"]
    if saved["fold"] != fold or source_index not in range(5):
        raise ValueError("candidate utility fold/source index changed")

    contract = unseal(root / "configs/t4_frozen_program_benchmark_v2.json")
    seeds = json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())
    metadata = source_group_map(contract, seeds)
    if metadata.get(source_id, {}).get("cell") != cell:
        raise ValueError("candidate utility source identity changed")
    source = decode_state(contract["cells"][cell]["source_state"])
    config = ComparisonConfig(**result["configuration"])
    marginal = _marginal(saved["marginal"])
    actor = RouteDistilledProgramPolicy.from_checkpoint(saved["actor"])
    ranker = _ranker(saved["hybrid"])

    generic, _ = generate_marginal_candidates(
        source,
        marginal,
        config,
        seed=config.seed + 100_000 * fold + source_index,
        prefix=f"generic-{fold}-{source_index}",
    )
    structured, _ = _actor_candidates(
        source,
        actor,
        config,
        seed=config.seed + 200_000 * fold + source_index,
        prefix=f"actor-{fold}-{source_index}",
    )
    expected = {
        "generic_marginal": result["aggregate_results"]["autonomous_generation"][
            "generic_marginal"
        ],
        "context_module_prototype": result["aggregate_results"][
            "autonomous_generation"
        ]["context_module_prototype"],
    }
    pools = {
        "generic_marginal": generic,
        "context_module_prototype": structured,
    }
    scorer = strict_endpoint_scorer(
        contract["cells"][cell]["original_seed"], delta=delta
    )
    selected, census = [], {}
    for policy, candidates in pools.items():
        complete = [row for row in candidates if row.status == "complete"]
        unique = {
            canonical_state_key(decode_state(row.endpoint_state)) for row in complete
        }
        if len(candidates) != expected[policy]["attempts"]:
            raise ValueError(f"{policy} attempt count changed during regeneration")
        if len(complete) != expected[policy]["complete"]:
            raise ValueError(f"{policy} completion count changed during regeneration")
        if len(unique) != expected[policy]["unique_complete"]:
            raise ValueError(
                f"{policy} unique endpoint count changed during regeneration"
            )
        rows = [_candidate_row(row, source, ranker, scorer, config) for row in complete]
        eligible = [row for row in rows if row["oracle_eligible"]]
        eligible.sort(
            key=lambda row: (-row["rank_score"], row["generation_index"], row["smiles"])
        )
        if eligible:
            selected.append({**eligible[0], "policies": [policy]})
        census[policy] = {
            "attempts": len(candidates),
            "complete": len(complete),
            "unique_complete": len(unique),
            "eligible_unique": len({row["smiles"] for row in eligible}),
        }

    deduplicated = {}
    for row in selected:
        prior = deduplicated.get(row["smiles"])
        if prior is None:
            deduplicated[row["smiles"]] = row
        else:
            prior["policies"] = sorted({*prior["policies"], *row["policies"]})
    locked = sorted(
        deduplicated.values(), key=lambda row: (row["smiles"], row["policies"])
    )
    unit = next(
        row
        for row in contract["units"]
        if row["cell"] == cell and row["replicate"] == 0
    )
    body = {
        "schema_version": "t4_route_candidate_utility_source_lock_v1",
        "cell": cell,
        "source_id": source_id,
        "source_index": source_index,
        "fold": fold,
        "delta": delta,
        "target": contract["cells"][cell]["target"],
        "original_seed": contract["cells"][cell]["original_seed"],
        "docking_seed": unit["docking_seed"],
        "oracle_protocol": unit["oracle_protocol"],
        "selection_rule": "top fold-trained contrastive-ranker score among strict-eligible unique endpoints, independently per generated policy",
        "candidate_census": census,
        "locked_candidates": locked,
        "inputs": {
            str(result_path): sha256_file(result_path),
            str(model_path): sha256_file(model_path),
            "configs/t4_frozen_program_benchmark_v2.json": sha256_file(
                root / "configs/t4_frozen_program_benchmark_v2.json"
            ),
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0},
        "evidence": "post-hoc score-blind candidate-utility diagnostic",
    }
    body["lock_sha256"] = identity(body)
    return body
