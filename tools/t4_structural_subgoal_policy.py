#!/usr/bin/env python3
"""Fit, generate and evaluate the grouped T4 structural-subgoal policy gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal import StructuralGoal, instantiate_goal
from compose_v4.control.structural_subgoal_policy import (
    ContextSubgoalRanker,
    MarginalSubgoalPolicy,
    StructuralDeltaTemplate,
    fit_context_subgoal_ranker,
    fit_marginal_subgoal_policy,
    materialize_template,
    minimize_subgoal,
    proposal_features,
    propose_structural_goals,
)
from compose_v4.control.structural_subgoal_realizer import (
    RealizerConfig,
    realize_structural_goal,
)
from compose_v4.experiments.route_proposal_quality import (
    SCHEMA as QUALITY_SCHEMA,
)
from compose_v4.experiments.route_proposal_quality import (
    evaluate as evaluate_quality,
)
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import predeclared_source_folds
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_structural_subgoal_audit import LIBRARY, teacher_traces

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_structural_subgoal_policy_v1.json"
T4_CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
REPRESENTATION_AUDIT = ROOT / "diagnostics/t4_structural_subgoal/attempt_2/summary.json"

RUNTIME_SCHEMA = "t4_structural_subgoal_policy_runtime_v1"
SOURCE_SCHEMA = "t4_structural_subgoal_policy_source_manifest_v1"
EVALUATION_SCHEMA = "t4_structural_subgoal_policy_evaluation_manifest_v1"
LOCK_SCHEMA = "t4_structural_subgoal_policy_candidate_lock_v1"
REPORT_SCHEMA = "t4_structural_subgoal_policy_report_v1"
POLICY_MARGINAL = "source_balanced_marginal_subgoal"
POLICY_CONTEXT = "graph_conditioned_subgoal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed(path: Path) -> dict:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    envelope = json.loads(raw)
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256", envelope.get("contract_sha256"))
    if not isinstance(payload, dict) or claimed != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    return payload


def publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite structural-subgoal artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(encoded, mtime=0) if compressed else encoded)
    temporary.replace(path)


def git_revision() -> tuple[str, bool]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    return revision, dirty


def _contract() -> dict:
    contract = load_sealed(CONTRACT)
    if contract["oracle"]["calls_authorized"] != 0:
        raise ValueError("structural-subgoal policy contract authorizes oracle calls")
    for name, path in (
        ("teacher_library", LIBRARY),
        ("source_registry", SEEDS),
        ("structural_subgoal_representation_audit", REPRESENTATION_AUDIT),
    ):
        if sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"structural-subgoal input hash changed: {path}")
    return contract


def _rows() -> tuple[list[dict], dict[str, dict]]:
    traces = teacher_traces()
    t4_contract = unseal(T4_CONTRACT)
    metadata = source_group_map(t4_contract, json.loads(SEEDS.read_text()))
    rows = []
    for teacher in traces:
        from compose_v4.control.structural_subgoal import extract_structural_goal

        goal, teacher_bindings, _ = extract_structural_goal(
            tuple(teacher["trace"]["states"]), tuple(teacher["trace"]["actions"])
        )
        templates, minimized_bindings = [], []
        for subgoal, binding in zip(goal.subgoals, teacher_bindings, strict=True):
            template, retained = minimize_subgoal(subgoal)
            templates.append(template)
            minimized_bindings.append(tuple(binding[index] for index in retained))
        minimized_goal = StructuralGoal(
            tuple(
                materialize_template(template, teacher["source"], binding)
                for template, binding in zip(templates, minimized_bindings, strict=True)
            )
        )
        endpoint = decode_state(teacher["trace"]["states"][-1])
        reconstructed, _ = instantiate_goal(
            teacher["source"], minimized_goal, tuple(minimized_bindings)
        )
        if canonical_state_key(reconstructed) != canonical_state_key(endpoint):
            raise RuntimeError("minimal structural delta changed a teacher endpoint")
        rows.append(
            {
                "source_group": teacher["source_group"],
                "source": teacher["source"],
                "endpoint": endpoint,
                "templates": tuple(templates),
                "goal": minimized_goal,
                "bindings": tuple(minimized_bindings),
            }
        )
    return rows, metadata


def _split(rows: list[dict], metadata: dict[str, dict], fold: int):
    split = next(
        (row for row in predeclared_source_folds(metadata) if row["fold"] == fold),
        None,
    )
    if split is None:
        raise ValueError(f"unknown structural-subgoal fold: {fold}")
    train = [row for row in rows if row["source_group"] in split["train_sources"]]
    test = [row for row in rows if row["source_group"] in split["test_sources"]]
    if not train or not test:
        raise RuntimeError("structural-subgoal grouped split is empty")
    return split, train, test


def _template_vocabulary(rows: list[dict]) -> tuple[StructuralDeltaTemplate, ...]:
    templates = {template.template_id: template for row in rows for template in row["templates"]}
    return tuple(templates[name] for name in sorted(templates))


def fit_fold(fold: int, output: Path) -> None:
    contract = _contract()
    rows, metadata = _rows()
    split, train, test = _split(rows, metadata, fold)
    proposal = contract["proposal"]
    training = contract["training"]
    began = perf_counter()
    marginal = fit_marginal_subgoal_policy(
        train, exploration_floor=float(training["exploration_floor"])
    )
    templates = _template_vocabulary(train)
    positives = [
        (
            row["source_group"],
            proposal_features(row["source"], row["endpoint"], row["templates"]),
        )
        for row in train
    ]
    teachers_by_source: dict[str, set[str]] = defaultdict(set)
    source_graphs = {}
    for row in train:
        teachers_by_source[row["source_group"]].add(canonical_state_key(row["endpoint"]))
        source_graphs[row["source_group"]] = row["source"]
    negatives = {}
    negative_telemetry = {}
    for source_group, source in sorted(source_graphs.items()):
        candidates, telemetry = propose_structural_goals(
            source,
            templates,
            marginal,
            ranker=None,
            pool_size=int(training["negatives_per_source"]),
            beam_width=int(training["negative_beam_width"]),
            expansion_width=int(training["negative_expansion_width"]),
            max_bindings_per_template=int(proposal["max_bindings_per_template"]),
        )
        negatives[source_group] = [
            proposal_features(source, row.endpoint, row.templates)
            for row in candidates
            if canonical_state_key(row.endpoint) not in teachers_by_source[source_group]
        ][: int(training["negatives_per_source"])]
        negative_telemetry[source_group] = telemetry
    ranker, ranker_fit = fit_context_subgoal_ranker(
        positives,
        negatives,
        updates=int(training["updates"]),
        learning_rate=float(training["learning_rate"]),
        l2=float(training["l2"]),
        seed=int(training["seed"]) + fold,
    )
    revision, dirty = git_revision()
    runtime = {
        "schema_version": RUNTIME_SCHEMA,
        "fold": fold,
        "templates": [row.payload() for row in templates],
        "marginal": marginal.checkpoint(),
        "context_ranker": ranker.checkpoint(),
        "proposal": proposal,
        "training_identity": identity(
            {
                "schema_version": "t4_structural_subgoal_fold_fit_v1",
                "fold": fold,
                "contract": identity(contract),
                "marginal": marginal.training_identity,
                "ranker": ranker.training_identity,
            }
        ),
        "new_oracle_calls": 0,
    }
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in test:
        by_source[row["source_group"]].append(row)
    sources, evaluation = [], []
    for source_group, teachers in sorted(by_source.items()):
        source_state = encode_state(teachers[0]["source"])
        source_case_id = identity(
            {
                "schema_version": "t4_structural_subgoal_source_case_v1",
                "fold": fold,
                "source_state": source_state,
            }
        )
        sources.append(
            {"source_case_id": source_case_id, "fold": fold, "source_state": source_state}
        )
        evaluation.append(
            {
                "source_case_id": source_case_id,
                "source_group": source_group,
                "source_state": source_state,
                "teachers": [
                    {
                        "teacher_id": f"teacher-{index}",
                        "endpoint_state": encode_state(row["endpoint"]),
                        "delta_template_ids": [
                            template.template_id for template in row["templates"]
                        ],
                    }
                    for index, row in enumerate(teachers)
                ],
            }
        )
    source_manifest = {
        "schema_version": SOURCE_SCHEMA,
        "fold": fold,
        "source_cases": sources,
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
    }
    evaluation_manifest = {
        "schema_version": EVALUATION_SCHEMA,
        "fold": fold,
        "train_sources": list(split["train_sources"]),
        "test_sources": list(split["test_sources"]),
        "cases": evaluation,
        "new_oracle_calls": 0,
    }
    report = {
        "schema_version": "t4_structural_subgoal_policy_fit_report_v1",
        "fold": fold,
        "training_sources": len(split["train_sources"]),
        "held_sources": len(split["test_sources"]),
        "training_routes": len(train),
        "held_routes": len(test),
        "training_templates": len(templates),
        "minimal_delta_reconstruction": {
            "covered": len(rows),
            "denominator": 77,
            "precision": len(rows) / 77,
        },
        "negative_telemetry": negative_telemetry,
        "negative_counts": {name: len(values) for name, values in negatives.items()},
        "ranker_fit": ranker_fit,
        "elapsed_seconds": perf_counter() - began,
        "implementation": {"revision": revision, "working_tree_dirty": dirty},
        "new_oracle_calls": 0,
    }
    publish(output / "runtime_checkpoint.json.gz", runtime, compressed=True)
    publish(output / "source_manifest.json", source_manifest)
    publish(output / "evaluation_manifest.json.gz", evaluation_manifest, compressed=True)
    publish(output / "fit_report.json", report)


def _realize_candidates(
    source,
    candidates,
    *,
    policy_id: str,
    progress_context: dict,
) -> dict:
    attempts = []
    began = perf_counter()
    telemetry = Counter()
    for rank, candidate in enumerate(candidates, 1):
        result = realize_structural_goal(
            source,
            candidate.goal,
            candidate.bindings,
            config=RealizerConfig(),
        )
        telemetry["compiler_expansions"] += int(result["expanded"])
        telemetry["compiler_attempts"] += int(result["attempted"])
        if result["status"] == "realized" and result["endpoint_matches_bound_target"]:
            endpoint = decode_state(result["states"][-1])
            telemetry["complete"] += 1
            attempts.append(
                {
                    "attempt_id": identity(
                        {
                            "schema_version": "generated_structural_goal_attempt_v1",
                            "policy_id": policy_id,
                            "rank": rank,
                            "goal": candidate.goal.payload(),
                            "bindings": candidate.bindings,
                        }
                    ),
                    "rank": rank,
                    "status": "complete",
                    "endpoint_state": encode_state(endpoint),
                    "delta_template_ids": [
                        minimize_subgoal(row)[0].template_id for row in candidate.goal.subgoals
                    ],
                    "component_count": len(candidate.goal.subgoals),
                    "primitive_count": len(result["actions"]),
                    "compiler_strategy": result["compiler_strategy"],
                }
            )
        else:
            telemetry["realizer_abstentions"] += 1
            attempts.append(
                {
                    "attempt_id": identity(
                        {
                            "schema_version": "generated_structural_goal_abstention_v1",
                            "policy_id": policy_id,
                            "rank": rank,
                            "goal": candidate.goal.payload(),
                            "bindings": candidate.bindings,
                        }
                    ),
                    "rank": rank,
                    "status": result["status"],
                    "endpoint_state": None,
                    "delta_template_ids": [],
                    "component_count": len(candidate.goal.subgoals),
                    "primitive_count": 0,
                    "compiler_strategy": result["compiler_strategy"],
                }
            )
        if rank % 16 == 0 or rank == len(candidates):
            elapsed = perf_counter() - began
            print(
                json.dumps(
                    {
                        "event": "structural_subgoal_realizer_progress",
                        **progress_context,
                        "policy": policy_id,
                        "completed": rank,
                        "total": len(candidates),
                        "realized": telemetry["complete"],
                        "abstained": telemetry["realizer_abstentions"],
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": (elapsed / rank * (len(candidates) - rank)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    telemetry["proposal_seconds"] = perf_counter() - began
    return {
        "policy_id": policy_id,
        "attempts": attempts,
        "realizer_telemetry": dict(sorted(telemetry.items())),
    }


def generate_fold(
    checkpoint_path: Path,
    source_path: Path,
    output: Path,
    *,
    source_case_index: int | None = None,
) -> None:
    runtime = load_sealed(checkpoint_path)
    sources = load_sealed(source_path)
    if runtime.get("schema_version") != RUNTIME_SCHEMA:
        raise ValueError("structural-subgoal runtime schema mismatch")
    if sources.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("structural-subgoal source schema mismatch")
    if runtime["fold"] != sources["fold"]:
        raise ValueError("structural-subgoal runtime/source fold mismatch")
    if sources.get("teacher_fields_present") is not False:
        raise ValueError("structural-subgoal generation received teacher fields")
    templates = tuple(StructuralDeltaTemplate.from_payload(row) for row in runtime["templates"])
    marginal = MarginalSubgoalPolicy.from_checkpoint(runtime["marginal"])
    ranker = ContextSubgoalRanker.from_checkpoint(runtime["context_ranker"])
    proposal = runtime["proposal"]
    source_cases = list(sources["source_cases"])
    if source_case_index is not None:
        if not 1 <= source_case_index <= len(source_cases):
            raise ValueError("source-case index is outside the frozen manifest")
        source_cases = [source_cases[source_case_index - 1]]
    cases = []
    for local_index, row in enumerate(source_cases, 1):
        case_index = source_case_index if source_case_index is not None else local_index
        source = decode_state(row["source_state"])
        policies = []
        for policy_id, model in (
            (POLICY_MARGINAL, None),
            (POLICY_CONTEXT, ranker),
        ):
            began = perf_counter()
            candidates, generation = propose_structural_goals(
                source,
                templates,
                marginal,
                ranker=model,
                pool_size=int(proposal["pool_size"]),
                beam_width=int(proposal["beam_width"]),
                expansion_width=int(proposal["expansion_width"]),
                max_bindings_per_template=int(proposal["max_bindings_per_template"]),
            )
            generated_seconds = perf_counter() - began
            result = _realize_candidates(
                source,
                candidates,
                policy_id=policy_id,
                progress_context={
                    "fold": runtime["fold"],
                    "source": case_index,
                    "sources": len(sources["source_cases"]),
                },
            )
            result["generation_telemetry"] = generation
            result["generation_seconds"] = generated_seconds
            policies.append(result)
            print(
                json.dumps(
                    {
                        "event": "structural_subgoal_policy_progress",
                        "fold": runtime["fold"],
                        "source": case_index,
                        "sources": len(sources["source_cases"]),
                        "policy": policy_id,
                        "candidates": len(candidates),
                        "complete": result["realizer_telemetry"].get("complete", 0),
                        "elapsed_seconds": generated_seconds
                        + result["realizer_telemetry"]["proposal_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        cases.append(
            {
                "source_case_id": row["source_case_id"],
                "fold": row["fold"],
                "source_state": row["source_state"],
                "policies": policies,
            }
        )
    lock = {
        "schema_version": LOCK_SCHEMA,
        "fold": runtime["fold"],
        "training_identity": runtime["training_identity"],
        "source_cases": cases,
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
    }
    publish(output, lock, compressed=output.suffix == ".gz")


def reduce_locks(inputs: list[Path], output: Path) -> None:
    if not inputs:
        raise ValueError("structural-subgoal reduction requires source locks")
    payloads = [load_sealed(path) for path in inputs]
    if any(row.get("schema_version") != LOCK_SCHEMA for row in payloads):
        raise ValueError("structural-subgoal reduction received a wrong schema")
    folds = {row["fold"] for row in payloads}
    training = {row["training_identity"] for row in payloads}
    if len(folds) != 1 or len(training) != 1:
        raise ValueError("structural-subgoal source locks disagree on fit identity")
    cases = [case for row in payloads for case in row["source_cases"]]
    identities = [row["source_case_id"] for row in cases]
    if len(cases) != 5 or len(set(identities)) != 5:
        raise ValueError("structural-subgoal fold reduction requires five unique sources")
    cases.sort(key=lambda row: row["source_case_id"])
    publish(
        output,
        {
            "schema_version": LOCK_SCHEMA,
            "fold": next(iter(folds)),
            "training_identity": next(iter(training)),
            "source_cases": cases,
            "teacher_fields_present": False,
            "task_identity_present": False,
            "new_oracle_calls": 0,
            "source_shards": len(payloads),
        },
        compressed=output.suffix == ".gz",
    )


def _subgoal_metrics(lock: dict, evaluation: dict, *, cutoffs=(8, 32, 128)) -> dict:
    locked = {row["source_case_id"]: row for row in lock["source_cases"]}
    result = {}
    for policy_id in (POLICY_MARGINAL, POLICY_CONTEXT):
        sources = []
        for reference in evaluation["cases"]:
            generated = locked[reference["source_case_id"]]
            pool = next(row for row in generated["policies"] if row["policy_id"] == policy_id)
            teachers = {
                value
                for teacher in reference["teachers"]
                for value in teacher["delta_template_ids"]
            }
            rows = []
            for attempt in pool["attempts"]:
                if attempt["status"] != "complete":
                    continue
                rows.append((attempt["rank"], set(attempt["delta_template_ids"])))
            cutoff_rows = {}
            for cutoff in cutoffs:
                emitted = [values for rank, values in rows if rank <= cutoff]
                recovered = teachers & set().union(*emitted) if emitted else set()
                matching = sum(bool(values & teachers) for values in emitted)
                cutoff_rows[str(cutoff)] = {
                    "exact_subgoal_recall": len(recovered) / max(1, len(teachers)),
                    "exact_subgoal_any": float(bool(recovered)),
                    "candidate_precision_emitted": matching / max(1, len(emitted)),
                    "unique_yield_fixed_k": len(emitted) / cutoff,
                }
            sources.append(
                {
                    "source_id": reference["source_group"],
                    "teacher_subgoals": len(teachers),
                    "cutoffs": cutoff_rows,
                }
            )
        result[policy_id] = {
            "aggregate": {
                "sources": len(sources),
                "cutoffs": {
                    str(cutoff): {
                        metric: sum(row["cutoffs"][str(cutoff)][metric] for row in sources)
                        / len(sources)
                        for metric in sources[0]["cutoffs"][str(cutoff)]
                    }
                    for cutoff in cutoffs
                },
            },
            "per_source": sources,
        }
    return result


def evaluate_fold(lock_path: Path, evaluation_path: Path, output: Path) -> None:
    lock = load_sealed(lock_path)
    evaluation = load_sealed(evaluation_path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("structural-subgoal candidate-lock schema mismatch")
    if evaluation.get("schema_version") != EVALUATION_SCHEMA:
        raise ValueError("structural-subgoal evaluation schema mismatch")
    locked = {row["source_case_id"]: row for row in lock["source_cases"]}
    cases = []
    for reference in evaluation["cases"]:
        generated = locked.get(reference["source_case_id"])
        if generated is None:
            raise ValueError("candidate lock is missing a held source")
        cases.append(
            {
                "source_id": reference["source_group"],
                "source_state": reference["source_state"],
                "teachers": [
                    {
                        "teacher_id": row["teacher_id"],
                        "endpoint_state": row["endpoint_state"],
                    }
                    for row in reference["teachers"]
                ],
                "policy_pools": [
                    {
                        "policy_id": row["policy_id"],
                        "proposal_seconds": row["generation_seconds"]
                        + row["realizer_telemetry"]["proposal_seconds"],
                        "attempts": [
                            {
                                "attempt_id": attempt["attempt_id"],
                                "rank": attempt["rank"],
                                "status": (
                                    "complete" if attempt["status"] == "complete" else "rejected"
                                ),
                                "endpoint_state": attempt["endpoint_state"],
                            }
                            for attempt in row["attempts"]
                        ],
                    }
                    for row in generated["policies"]
                ],
            }
        )
    quality = evaluate_quality(
        {
            "schema_version": QUALITY_SCHEMA,
            "oracle_calls": 0,
            "split": {
                "evaluation_role": "test",
                "train_sources": evaluation["train_sources"],
                "calibration_sources": [],
                "test_sources": evaluation["test_sources"],
            },
            "cases": cases,
        },
        cutoffs=(8, 32, 128),
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "fold": lock["fold"],
        "endpoint_and_transformation_quality": quality,
        "subgoal_quality": _subgoal_metrics(lock, evaluation),
        "telemetry": {
            row["source_case_id"]: {
                policy["policy_id"]: {
                    "generation": policy["generation_telemetry"],
                    "realizer": policy["realizer_telemetry"],
                }
                for policy in row["policies"]
            }
            for row in lock["source_cases"]
        },
        "new_oracle_calls": 0,
        "interpretation_limit": (
            "Held-source zero-oracle proposal support only; no docking or IVG claim."
        ),
    }
    publish(output, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    fit.add_argument("--output", type=Path, required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--sources", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--source-case-index", type=int)
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--lock", type=Path, required=True)
    evaluator.add_argument("--evaluation", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit_fold(args.fold, args.output)
    elif args.command == "generate":
        generate_fold(
            args.checkpoint,
            args.sources,
            args.output,
            source_case_index=args.source_case_index,
        )
    elif args.command == "reduce":
        reduce_locks(args.inputs, args.output)
    else:
        evaluate_fold(args.lock, args.evaluation, args.output)


if __name__ == "__main__":
    main()
