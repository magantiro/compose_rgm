#!/usr/bin/env python3
"""Run the predeclared grouped, zero-oracle T4 route-policy comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.option_demonstrations import (
    DemonstrationFitConfig,
    descriptor_menu,
    fit_demonstration_actor,
)
from compose_v4.control.option_policy import option_actor_parameter_id
from compose_v4.control.route_distilled_program_policy import (
    CHECKPOINT_SCHEMA,
    RouteDistilledProgramPolicy,
    default_reference,
    normalized_module_count_probabilities,
    synthesize_route_distilled_program,
)
from compose_v4.experiments.route_proposal_quality import evaluate
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import (
    POLICY_ACTOR,
    POLICY_GENERIC,
    POLICY_HYBRID,
    SCHEMA,
    Candidate,
    ComparisonConfig,
    _recognized_families,
    candidate_features,
    complete_attempt_rows,
    family_feature_weights,
    fit_contrastive_ranker,
    fit_marginal_policy,
    generate_marginal_candidates,
    predeclared_source_folds,
    require_nonself_endpoint,
    teacher_candidate,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_route_distillation import (
    _binding_prototypes,
    _rows,
    _teacher_traces,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
LIBRARY = (
    ROOT / "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
)
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
OUTPUT = ROOT / "diagnostics/t4_route_policy_comparison/attempt_1"
ACTOR_CONFIG = DemonstrationFitConfig(
    hidden=64,
    updates=600,
    batch_size=64,
    learning_rate=0.003,
    base_floor=0.10,
    kl_penalty=0.01,
    seed=20260914,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite route-policy artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = _json_ready(payload)
    envelope = {"payload": ready, "payload_sha256": identity(ready)}
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    if compressed:
        temporary.write_bytes(gzip.compress(raw.encode(), mtime=0))
    else:
        temporary.write_text(raw)
    temporary.replace(path)


def _actor_policy(train: list[dict], *, fold: int):
    rows, route_summaries = _rows(train)
    options = descriptor_menu()
    reference = np.asarray(default_reference(), dtype=float)
    actor, history = fit_demonstration_actor(
        [row["features"] for row in rows],
        [options.index(row["option"]) for row in rows],
        [row["source_id"] for row in rows],
        options,
        reference,
        config=ACTOR_CONFIG,
    )
    counts = Counter(min(3, row["high_level_decisions"]) for row in route_summaries)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "state_dim": actor.state_dim,
        "option_dim": actor.option_dim,
        "hidden": ACTOR_CONFIG.hidden,
        "parameters": {
            name: tensor.detach().tolist()
            for name, tensor in sorted(actor.state_dict().items())
        },
        "parameter_identity": option_actor_parameter_id(actor),
        "options": list(options),
        "reference": reference.tolist(),
        "exploration_floor": ACTOR_CONFIG.base_floor,
        "module_count_probabilities": list(
            normalized_module_count_probabilities(sorted(counts.items()))
        ),
        "binding_prototypes": _binding_prototypes(rows),
        "training_identity": identity(
            {
                "schema_version": "grouped_context_actor_training_v1",
                "fold": fold,
                "sources": len({row["source_id"] for row in rows}),
                "decisions": len(rows),
                "configuration": asdict(ACTOR_CONFIG),
                "parameter_identity": option_actor_parameter_id(actor),
            }
        ),
    }
    return RouteDistilledProgramPolicy.from_checkpoint(checkpoint), checkpoint, history


def _actor_candidates(source, policy, config, *, seed, prefix):
    began = perf_counter()
    rng, result, panel_cache = np.random.default_rng(seed), [], {}
    for index in range(config.attempts_per_source):
        try:
            _, program, _, trace, metadata = synthesize_route_distilled_program(
                source,
                rng,
                policy,
                max_modules=config.max_modules,
                max_primitives=config.max_primitives,
                max_blocks=config.max_blocks,
                panel_cache=panel_cache,
            )
            endpoint = decode_state(trace["states"][-1])
            require_nonself_endpoint(source, endpoint)
            result.append(
                Candidate(
                    f"{prefix}-{index}",
                    "complete",
                    encode_state(endpoint),
                    tuple(trace["actions"]),
                    tuple(row["family"] for row in metadata["modules"]),
                    len(program.blocks),
                    index,
                )
            )
        except ValueError as error:
            result.append(
                Candidate(
                    f"{prefix}-{index}",
                    "rejected",
                    None,
                    (),
                    (),
                    0,
                    index,
                    str(error),
                )
            )
    return result, perf_counter() - began


def _candidate_vector(source, candidate: Candidate, config: ComparisonConfig):
    if candidate.endpoint_state is None:
        raise ValueError("rejected candidates have no ranking feature")
    endpoint = decode_state(candidate.endpoint_state)
    return candidate_features(
        source,
        endpoint,
        candidate.actions,
        family_feature_weights(candidate),
        module_count=len(candidate.families),
        blocks=candidate.blocks,
        max_primitives=config.max_primitives,
        max_blocks=config.max_blocks,
    )


def _teacher_vector(teacher, config):
    candidate = teacher_candidate(teacher, attempt_id="offline-teacher", index=0)
    source = teacher["source"]
    # Preserve fractional option-to-family mappings rather than interpreting
    # their family names as repeated independent modules.
    family_weights, decisions = _recognized_families(teacher["trace"])
    return candidate_features(
        source,
        decode_state(candidate.endpoint_state),
        candidate.actions,
        family_weights,
        module_count=min(3, decisions),
        blocks=1,
        max_primitives=config.max_primitives,
        max_blocks=config.max_blocks,
    )


def _rank_scores(candidates, scorer):
    return {
        candidate.attempt_id: scorer(candidate)
        for candidate in candidates
        if candidate.status == "complete"
    }


def _case(
    source_id,
    source,
    teachers,
    policy_pools,
):
    return {
        "source_id": source_id,
        "source_state": encode_state(source),
        "teachers": [
            {
                "teacher_id": f"teacher-{index}",
                "endpoint_state": teacher["trace"]["states"][-1],
            }
            for index, teacher in enumerate(teachers)
        ],
        "policy_pools": policy_pools,
    }


def _runtime_payload_audit(payloads: list[dict], forbidden: set[str]) -> dict:
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, child in value.items():
                yield str(key)
                yield from strings(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from strings(child)

    observed = {value for payload in payloads for value in strings(payload)}
    forbidden_keys = {
        "target",
        "protein",
        "route_id",
        "endpoint",
        "source_atom",
        "absolute_atom",
        "program",
    }
    return {
        "passed": not (observed & forbidden) and not (observed & forbidden_keys),
        "forbidden_value_hits": sorted(observed & forbidden),
        "forbidden_key_hits": sorted(observed & forbidden_keys),
    }


def _combined_metrics(fold_reports: list[dict], lane: str) -> dict:
    """Combine equal-source fold metrics without treating routes as replicates."""

    names = sorted({name for fold in fold_reports for name in fold[lane]["policies"]})
    result = {}
    for name in names:
        rows = [fold[lane]["policies"][name]["aggregate"] for fold in fold_reports]
        source_total = sum(row["sources"] for row in rows)

        def balanced(field, values=rows, total=source_total):
            return sum(row["sources"] * row[field] for row in values) / total

        attempts = sum(row["attempts"] for row in rows)
        complete = sum(row["complete"] for row in rows)
        unique = sum(row["unique_complete"] for row in rows)
        seconds = sum(row["proposal_seconds"] for row in rows)
        cutoffs = {}
        for cutoff in rows[0]["cutoffs"]:
            cutoffs[cutoff] = {
                metric: sum(
                    row["sources"] * row["cutoffs"][cutoff][metric] for row in rows
                )
                / source_total
                for metric in rows[0]["cutoffs"][cutoff]
            }
        result[name] = {
            "sources": source_total,
            "source_balanced_exact_mrr": balanced("source_balanced_exact_mrr"),
            "source_balanced_transformation_mrr": balanced(
                "source_balanced_transformation_mrr"
            ),
            "execution_precision": complete / max(1, attempts),
            "unique_endpoint_yield": unique / max(1, attempts),
            "attempts": attempts,
            "complete": complete,
            "unique_complete": unique,
            "proposal_seconds": seconds,
            "attempts_per_second": attempts / max(seconds, 1e-12),
            "unique_endpoints_per_second": unique / max(seconds, 1e-12),
            "cutoffs": cutoffs,
        }
    return result


def _select_folds(all_folds, fold_ids: tuple[int, ...] | None):
    requested = (
        tuple(row["fold"] for row in all_folds)
        if fold_ids is None
        else tuple(sorted(set(fold_ids)))
    )
    known = {row["fold"] for row in all_folds}
    if not requested or not set(requested) <= known:
        raise ValueError(
            f"invalid route-policy fold selection: {requested}; known={sorted(known)}"
        )
    return requested, tuple(row for row in all_folds if row["fold"] in requested)


def run(
    output: Path = OUTPUT,
    *,
    config: ComparisonConfig | None = None,
    fold_ids: tuple[int, ...] | None = None,
    code_revision: str | None = None,
    working_tree_dirty: bool | None = None,
):
    """Fit and evaluate three grouped policies without a task oracle."""

    if config is None:
        config = ComparisonConfig()
    if output.exists():
        raise ValueError(f"preserve prior route-policy comparison: {output}")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("route-policy comparison requires pinned RDKit 2024.03.5")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    contract = unseal(CONTRACT)
    seed_registry = json.loads(SEEDS.read_text())
    metadata = source_group_map(contract, seed_registry)
    library = json.loads(LIBRARY.read_text())
    traces, exclusions = _teacher_traces(library)
    if exclusions or len(traces) != 77:
        raise RuntimeError("the signed teacher census is incomplete")
    all_folds = predeclared_source_folds(metadata)
    requested_folds, folds = _select_folds(all_folds, fold_ids)
    by_source = {
        source: [row for row in traces if row["source_group"] == source]
        for source in metadata
    }
    reports, runtime_payloads, fold_models = [], [], []
    for split in folds:
        fold = split["fold"]
        train = [row for row in traces if row["source_group"] in split["train_sources"]]
        test = [row for row in traces if row["source_group"] in split["test_sources"]]
        marginal = fit_marginal_policy(
            train, exploration_floor=config.exploration_floor
        )
        actor, actor_checkpoint, actor_history = _actor_policy(train, fold=fold)

        negatives, training_generation = {}, {}
        for source_index, source_id in enumerate(split["train_sources"]):
            source = by_source[source_id][0]["source"]
            training_config = ComparisonConfig(
                **{
                    **asdict(config),
                    "attempts_per_source": config.training_negative_attempts_per_source,
                }
            )
            candidates, seconds = generate_marginal_candidates(
                source,
                marginal,
                training_config,
                seed=config.seed + 10_000 * fold + source_index,
                prefix=f"train-{fold}-{source_index}",
            )
            negatives[source_id] = [
                _candidate_vector(source, candidate, config)
                for candidate in candidates
                if candidate.status == "complete"
            ]
            training_generation[source_id] = {
                "attempts": len(candidates),
                "complete": len(negatives[source_id]),
                "seconds": seconds,
            }
        positives = [
            (
                teacher["source_group"],
                f"route-{index}",
                _teacher_vector(teacher, config),
            )
            for index, teacher in enumerate(train)
        ]
        ranker, ranker_fit = fit_contrastive_ranker(positives, negatives, config)
        marginal_checkpoint = marginal.checkpoint()
        ranker_checkpoint = ranker.checkpoint()
        runtime_payloads.extend(
            [marginal_checkpoint, actor_checkpoint, ranker_checkpoint]
        )

        shared_cases, autonomous_cases, census = [], [], []
        for source_index, source_id in enumerate(split["test_sources"]):
            source_teachers = by_source[source_id]
            source = source_teachers[0]["source"]
            marginal_candidates, marginal_seconds = generate_marginal_candidates(
                source,
                marginal,
                config,
                seed=config.seed + 100_000 * fold + source_index,
                prefix=f"generic-{fold}-{source_index}",
            )
            actor_candidates, actor_seconds = _actor_candidates(
                source,
                actor,
                config,
                seed=config.seed + 200_000 * fold + source_index,
                prefix=f"actor-{fold}-{source_index}",
            )
            generic_scores = _rank_scores(
                marginal_candidates,
                lambda candidate, policy=marginal: policy.score(candidate.families),
            )
            hybrid_began = perf_counter()
            hybrid_scores = _rank_scores(
                marginal_candidates,
                lambda candidate, model=ranker, graph=source: model.score(
                    _candidate_vector(graph, candidate, config)
                ),
            )
            hybrid_seconds = marginal_seconds + perf_counter() - hybrid_began
            autonomous_cases.append(
                _case(
                    source_id,
                    source,
                    source_teachers,
                    [
                        {
                            "policy_id": POLICY_GENERIC,
                            "proposal_seconds": marginal_seconds,
                            "attempts": complete_attempt_rows(
                                marginal_candidates, generic_scores
                            ),
                        },
                        {
                            "policy_id": POLICY_ACTOR,
                            "proposal_seconds": actor_seconds,
                            "attempts": complete_attempt_rows(actor_candidates),
                        },
                        {
                            "policy_id": POLICY_HYBRID,
                            "proposal_seconds": hybrid_seconds,
                            "attempts": complete_attempt_rows(
                                marginal_candidates, hybrid_scores
                            ),
                        },
                    ],
                )
            )

            injected = [
                teacher_candidate(
                    teacher,
                    attempt_id=f"shared-teacher-{fold}-{source_index}-{index}",
                    index=config.attempts_per_source + index,
                )
                for index, teacher in enumerate(source_teachers)
            ]
            shared = [*marginal_candidates, *injected]
            shared_generic_scores = dict(generic_scores)
            shared_hybrid_scores = dict(hybrid_scores)
            for candidate, teacher in zip(injected, source_teachers, strict=True):
                # The marginal model has no support beyond three decisions. Its
                # diagnostic score is therefore minus infinity for long routes.
                _, decisions = _recognized_families(teacher["trace"])
                shared_generic_scores[candidate.attempt_id] = (
                    marginal.score(candidate.families)
                    if decisions <= config.max_modules
                    else float("-inf")
                )
                shared_hybrid_scores[candidate.attempt_id] = ranker.score(
                    _teacher_vector(teacher, config)
                )
            shared_cases.append(
                _case(
                    source_id,
                    source,
                    source_teachers,
                    [
                        {
                            "policy_id": POLICY_GENERIC,
                            "proposal_seconds": marginal_seconds,
                            "attempts": complete_attempt_rows(
                                shared, shared_generic_scores
                            ),
                        },
                        {
                            "policy_id": POLICY_HYBRID,
                            "proposal_seconds": hybrid_seconds,
                            "attempts": complete_attempt_rows(
                                shared, shared_hybrid_scores
                            ),
                        },
                    ],
                )
            )
            census.append(
                {
                    "source_id": source_id,
                    "cell": metadata[source_id]["cell"],
                    "teacher_routes": len(source_teachers),
                    "matched_attempt_budget": config.attempts_per_source,
                    "generic_complete": sum(
                        row.status == "complete" for row in marginal_candidates
                    ),
                    "actor_complete": sum(
                        row.status == "complete" for row in actor_candidates
                    ),
                }
            )
        split_payload = {
            "evaluation_role": "test",
            "train_sources": split["train_sources"],
            "calibration_sources": split["calibration_sources"],
            "test_sources": split["test_sources"],
        }
        autonomous = evaluate(
            {
                "schema_version": "route_proposal_quality_input_v1",
                "oracle_calls": 0,
                "split": split_payload,
                "cases": autonomous_cases,
            }
        )
        shared = evaluate(
            {
                "schema_version": "route_proposal_quality_input_v1",
                "oracle_calls": 0,
                "split": split_payload,
                "cases": shared_cases,
            }
        )
        reports.append(
            {
                "fold": fold,
                "split": split_payload,
                "training_routes": len(train),
                "test_routes": len(test),
                "training_negative_generation": training_generation,
                "ranker_fit": ranker_fit,
                "actor_fit": {"history": actor_history},
                "source_census": census,
                "autonomous_generation": autonomous,
                "shared_panel_reranking": shared,
                "shared_panel_actor_abstention": {
                    "policy_id": POLICY_ACTOR,
                    "abstained": True,
                    "reason": (
                        "the current actor exposes option probabilities and aggregate "
                        "prototype ranks, not a normalized complete-candidate score; "
                        "most teacher routes also exceed its three-module horizon"
                    ),
                },
            }
        )
        fold_models.append(
            {
                "fold": fold,
                "marginal": marginal_checkpoint,
                "actor": actor_checkpoint,
                "hybrid": ranker_checkpoint,
            }
        )

    forbidden = {
        *metadata,
        *(row["original_seed"] for row in contract["cells"].values()),
        *(row["target"] for row in contract["cells"].values()),
        *(row["program_id"] for row in traces),
        *(row["trace"]["endpoint"] for row in traces),
    }
    runtime_audit = _runtime_payload_audit(runtime_payloads, forbidden)
    if not runtime_audit["passed"]:
        raise RuntimeError(f"runtime checkpoint provenance failed: {runtime_audit}")
    revision = code_revision
    if revision is None:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    dirty = working_tree_dirty
    if dirty is None:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    report = {
        "schema_version": SCHEMA,
        "decision": "zero_oracle_grouped_policy_comparison_complete",
        "evidence": "answer-known grouped T4 route-proposal development",
        "configuration": asdict(config),
        "predeclared_split": {
            "rule": "three source-group folds; test source_idx equals fold",
            "folds": list(all_folds),
            "evaluated_fold_ids": list(requested_folds),
            "source_balance": "each source equal; hybrid positives equal by source then route",
            "lineage_rule": "all routes from a source molecule remain in one fold",
        },
        "policies": {
            POLICY_GENERIC: "source-balanced family/depth marginal",
            POLICY_ACTOR: "current context option actor plus aggregate binding prototypes",
            POLICY_HYBRID: "linear graph-conditioned contrastive complete-candidate ranker",
        },
        "evaluations": {
            "autonomous_generation": (
                "no teacher endpoint or route is injected into any test proposal pool"
            ),
            "shared_panel_reranking": (
                "teacher endpoints are injected only as labeled offline diagnostic candidates; "
                "success is ranking evidence, not generation evidence"
            ),
        },
        "aggregate_results": {
            "autonomous_generation": _combined_metrics(
                reports, "autonomous_generation"
            ),
            "shared_panel_reranking": _combined_metrics(
                reports, "shared_panel_reranking"
            ),
            "shared_panel_actor_abstention": True,
        },
        "fold_reports": reports,
        "runtime_provenance": runtime_audit,
        "costs": {"oracle_calls": 0, "docking_calls": 0},
        "inputs": {
            str(CONTRACT.relative_to(ROOT)): sha256(CONTRACT),
            str(LIBRARY.relative_to(ROOT)): sha256(LIBRARY),
            str(SEEDS.relative_to(ROOT)): sha256(SEEDS),
            "teacher_receipts": {
                row["receipt"]: row["receipt_sha256"] for row in traces
            },
        },
        "implementation": {
            "revision": revision,
            "working_tree_dirty": dirty,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "rdkit": rdBase.rdkitVersion,
            "device": "CPU",
            "threads": 1,
        },
        "limitations": [
            "All teacher routes are answer-known T4 development evidence.",
            "Shared-panel recovery is reranking evidence and cannot establish autonomous generation.",
            "The current actor abstains from shared complete-candidate scoring because it exposes no faithful complete-route density.",
            "The hybrid is a bounded candidate ranker, not a full autoregressive program decoder.",
            "Stereochemistry is outside the exact graph-equivalence representation.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    publish(output / "models.json.gz", {"fold_models": fold_models}, compressed=True)
    report["outputs"] = {"models.json.gz": sha256(output / "models.json.gz")}
    publish(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--attempts-per-source", type=int, default=128)
    parser.add_argument("--training-negative-attempts-per-source", type=int, default=32)
    args = parser.parse_args()
    config = ComparisonConfig(
        attempts_per_source=args.attempts_per_source,
        training_negative_attempts_per_source=args.training_negative_attempts_per_source,
    )
    result = run(args.output, config=config)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "folds": len(result["fold_reports"]),
                "runtime_provenance": result["runtime_provenance"],
                "costs": result["costs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
