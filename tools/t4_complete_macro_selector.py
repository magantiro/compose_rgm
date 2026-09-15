#!/usr/bin/env python3
"""Fit, lock and evaluate the frozen zero-oracle T4 complete-macro selector."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import rdBase

from compose_v4.control.complete_macro_selector import (
    CompleteMacroSelector,
    compiler_log_work,
    complete_macro_features,
    complete_macro_signature,
    fit_complete_macro_selector,
    rank_complete_macros,
)
from compose_v4.control.compositional_structural_subgoal_generator import (
    POLICY_LEARNED,
    POLICY_UNIFORM,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal_components import (
    GRANULAR_COMPONENT_FAMILIES,
    ExactComponentVocabulary,
)
from compose_v4.experiments.route_proposal_quality import (
    SCHEMA as QUALITY_SCHEMA,
)
from compose_v4.experiments.route_proposal_quality import evaluate as evaluate_quality
from compose_v4.experiments.route_proposal_quality import transformation_equivalent
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_compositional_structural_subgoal_generator import (
    EVALUATION_SCHEMA,
    _component_from_payload,
    _component_payload,
    _coverage,
    _generated_records,
)
from tools.t4_compositional_structural_subgoal_generator import (
    LOCK_SCHEMA as BASELINE_LOCK_SCHEMA,
)
from tools.t4_structural_subgoal_component_novelty import load_sealed

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_complete_macro_selector_v1.json"
OUTPUT = ROOT / "diagnostics/t4_complete_macro_selector/attempt_1"
CHECKPOINT_SCHEMA = "t4_complete_macro_selector_fold_checkpoint_v1"
FIT_REPORT_SCHEMA = "t4_complete_macro_selector_fit_report_v1"
FIT_SUMMARY_SCHEMA = "t4_complete_macro_selector_fit_summary_v1"
LOCK_SCHEMA = "t4_complete_macro_selector_candidate_lock_v1"
RESULT_SCHEMA = "t4_complete_macro_selector_result_v1"
POLICY_SELECTOR = "complete_macro_selector"
POLICY_RAW_LEARNED = "balanced_joint_autoregressive_raw_order"
POLICY_RAW_UNIFORM = "uniform_joint_grammar_raw_order"
FAMILIES = ("whole_patch", *GRANULAR_COMPONENT_FAMILIES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite complete-macro selector artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    raw = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(raw, mtime=0) if compressed else raw)
    temporary.replace(path)


def _contract() -> dict:
    envelope = json.loads(CONTRACT.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("contract_sha256") != identity(payload):
        raise ValueError("complete-macro selector contract is not self-hashed")
    if any(
        payload["costs"][name] != 0
        for name in (
            "oracle_calls_authorized",
            "docking_calls_authorized",
            "modal_launches_authorized",
            "candidate_generation_calls_authorized",
        )
    ):
        raise ValueError("complete-macro selector contract authorizes external work")
    for row in payload["inputs"].values():
        path = ROOT / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"complete-macro selector input hash changed: {path}")
        if "payload_sha256" in row and identity(load_sealed(path)) != row["payload_sha256"]:
            raise ValueError(f"complete-macro selector input payload changed: {path}")
    return payload


def _implementation_revision() -> str:
    if (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode
    ):
        raise RuntimeError("refusing scientific artifact generation from a dirty worktree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _paths(contract: dict) -> tuple[Path, tuple[Path, ...]]:
    lock = ROOT / contract["inputs"]["candidate_lock"]["path"]
    evaluations = tuple(
        ROOT / contract["inputs"][f"fold_{fold}_evaluation"]["path"]
        for fold in contract["split"]["folds"]
    )
    return lock, evaluations


def _load_catalog(contract: dict) -> tuple[dict, dict[int, dict], dict[str, dict]]:
    lock_path, evaluation_paths = _paths(contract)
    lock = load_sealed(lock_path)
    if (
        lock.get("schema_version") != BASELINE_LOCK_SCHEMA
        or lock.get("teacher_fields_present") is not False
        or lock.get("new_oracle_calls") != 0
    ):
        raise ValueError("baseline compositional candidate lock invariant failed")
    evaluations = {}
    cases_by_id = {
        case["source_case_id"]: {**case, "baseline_fold": int(fold["fold"])}
        for fold in lock["folds"]
        for case in fold["cases"]
    }
    catalog = {}
    for path in evaluation_paths:
        evaluation = load_sealed(path)
        fold = int(evaluation.get("fold", -1))
        if evaluation.get("schema_version") != EVALUATION_SCHEMA or fold in evaluations:
            raise ValueError("baseline evaluation manifest invariant failed")
        evaluations[fold] = evaluation
        for reference in evaluation["cases"]:
            case_id = reference["source_case_id"]
            locked = cases_by_id.get(case_id)
            if locked is None or reference["source_state"] != locked["source_state"]:
                raise ValueError("baseline candidate/evaluation source mapping changed")
            source_group = reference["source_group"]
            if source_group in catalog:
                raise ValueError("a T4 source appears in multiple held folds")
            policies = {row["policy_id"]: row for row in locked["policies"]}
            if set(policies) != {POLICY_UNIFORM, POLICY_LEARNED}:
                raise ValueError("baseline candidate policy set changed")
            if any(len(row["candidates"]) != 128 for row in policies.values()):
                raise ValueError("baseline broad candidate pool size changed")
            catalog[source_group] = {
                "source_group": source_group,
                "source_case_id": case_id,
                "source_state": reference["source_state"],
                "teachers": reference["teachers"],
                "train_whole_patch_vocabulary": evaluation["train_whole_patch_vocabulary"],
                "baseline_fold": locked["baseline_fold"],
                "policies": policies,
            }
    if set(evaluations) != {0, 1, 2} or len(catalog) != 15:
        raise ValueError("the frozen three-fold 15-source census changed")
    if sum(len(row["teachers"]) for row in catalog.values()) != 77:
        raise ValueError("the frozen 77-route teacher census changed")
    return lock, evaluations, catalog


def _wrong_hard_negatives(
    source,
    teachers: list[dict],
    candidates: list[dict],
    *,
    limit: int,
) -> tuple[list[dict], dict]:
    teacher_graphs = [decode_state(row["endpoint_state"]) for row in teachers]
    teacher_keys = {canonical_state_key(row) for row in teacher_graphs}
    selected = []
    exclusions = defaultdict(int)
    for original_rank, candidate in enumerate(candidates, 1):
        endpoint = decode_state(candidate["endpoint_state"])
        if canonical_state_key(endpoint) in teacher_keys:
            exclusions["exact_teacher_endpoint"] += 1
            continue
        if any(transformation_equivalent(source, endpoint, row) for row in teacher_graphs):
            exclusions["radius2_teacher_transformation"] += 1
            continue
        selected.append({"candidate": candidate, "original_rank": original_rank})
        if len(selected) == limit:
            break
    return selected, {
        "requested": limit,
        "selected": len(selected),
        "scanned": selected[-1]["original_rank"] if selected else len(candidates),
        "exclusions": dict(sorted(exclusions.items())),
    }


def fit_all(output: Path) -> None:
    contract = _contract()
    _, evaluations, catalog = _load_catalog(contract)
    model_config = contract["model"]
    negative_limit = int(contract["data"]["hard_negatives_per_training_source"])
    reports = []
    for fold in contract["split"]["folds"]:
        fold = int(fold)
        split = evaluations[fold]
        train_sources = tuple(split["train_sources"])
        held_sources = tuple(split["held_sources"])
        if (
            set(train_sources) & set(held_sources)
            or len(train_sources) != 10
            or len(held_sources) != 5
        ):
            raise ValueError("frozen selector split invariant failed")
        positives = []
        negatives = {}
        source_reports = {}
        for source_group in train_sources:
            row = catalog[source_group]
            if source_group in held_sources:
                raise RuntimeError("held source entered selector fitting")
            source = decode_state(row["source_state"])
            for teacher in row["teachers"]:
                positives.append(
                    (
                        source_group,
                        complete_macro_features(source, teacher["endpoint_state"], teacher["goal"]),
                    )
                )
            learned = row["policies"][POLICY_LEARNED]["candidates"]
            selected, report = _wrong_hard_negatives(
                source, row["teachers"], learned, limit=negative_limit
            )
            source_reports[source_group] = {
                **report,
                "positive_routes": len(row["teachers"]),
                "negative_generator_fold": row["baseline_fold"],
                "source_was_held_from_negative_generator": True,
            }
            if len(selected) == negative_limit:
                negatives[source_group] = [
                    (
                        complete_macro_features(
                            source,
                            item["candidate"]["endpoint_state"],
                            item["candidate"]["goal"],
                        ),
                        compiler_log_work(item["candidate"]),
                    )
                    for item in selected
                ]
        fold_root = output / f"fold_{fold}"
        if len(negatives) != len(train_sources) or not positives:
            publish(
                fold_root / "fit_report.json",
                {
                    "schema_version": FIT_REPORT_SCHEMA,
                    "fold": fold,
                    "status": "abstained",
                    "reason": "missing positive or 32 eligible same-source hard negatives",
                    "sources": source_reports,
                    "new_oracle_calls": 0,
                },
            )
            reports.append({"fold": fold, "status": "abstained"})
            continue
        selector, fit_report = fit_complete_macro_selector(
            positives,
            negatives,
            updates=int(model_config["updates"]),
            learning_rate=float(model_config["learning_rate"]),
            l2=float(model_config["l2"]),
            seed=int(model_config["seed"]) + fold,
            compiler_cost_weight=float(
                model_config["compiler_cost_auxiliary"]["fixed_penalty_weight"]
            ),
        )
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA,
            "fold": fold,
            "selector": selector.checkpoint(),
            "split_identity": identity(
                {
                    "schema_version": "complete_macro_selector_fold_identity_v1",
                    "fold": fold,
                    "training_source_count": len(train_sources),
                    "held_source_count": len(held_sources),
                }
            ),
            "contract_payload_sha256": identity(contract),
            "new_oracle_calls": 0,
        }
        serialized = json.dumps(checkpoint, sort_keys=True).lower()
        for forbidden in (
            "source_group",
            "source_state",
            "endpoint_state",
            "route_id",
            "task_id",
            "protein",
            "smiles",
            "goal_payload",
            "patch_ids",
            "actions",
        ):
            if forbidden in serialized:
                raise RuntimeError(f"selector checkpoint contains forbidden field: {forbidden}")
        publish(fold_root / "runtime_checkpoint.json.gz", checkpoint, compressed=True)
        publish(
            fold_root / "fit_report.json",
            {
                "schema_version": FIT_REPORT_SCHEMA,
                "fold": fold,
                "status": "fit",
                "training_sources": len(train_sources),
                "held_sources": len(held_sources),
                "positive_routes": len(positives),
                "hard_negatives": sum(map(len, negatives.values())),
                "source_balance": "equal source, route and within-route negative mass",
                "sources": source_reports,
                "fit": fit_report,
                "training_identity": selector.training_identity,
                "new_oracle_calls": 0,
            },
        )
        reports.append(
            {"fold": fold, "status": "fit", "training_identity": selector.training_identity}
        )
    publish(
        output / "fit_summary.json",
        {
            "schema_version": FIT_SUMMARY_SCHEMA,
            "folds": reports,
            "fit_folds": sum(row["status"] == "fit" for row in reports),
            "abstained_folds": sum(row["status"] == "abstained" for row in reports),
            "new_oracle_calls": 0,
        },
    )


def rank_all(input_root: Path, output: Path) -> None:
    contract = _contract()
    lock, _, _ = _load_catalog(contract)
    folds = []
    for locked_fold in lock["folds"]:
        fold = int(locked_fold["fold"])
        checkpoint_path = input_root / f"fold_{fold}/runtime_checkpoint.json.gz"
        if not checkpoint_path.exists():
            folds.append({"fold": fold, "status": "abstained", "cases": []})
            continue
        checkpoint = load_sealed(checkpoint_path)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or checkpoint.get("fold") != fold:
            raise ValueError("selector checkpoint fold/schema mismatch")
        selector = CompleteMacroSelector.from_checkpoint(checkpoint["selector"])
        cases = []
        for case in locked_fold["cases"]:
            policies = {row["policy_id"]: row for row in case["policies"]}
            learned = policies[POLICY_LEARNED]["candidates"]
            ranked = rank_complete_macros(decode_state(case["source_state"]), learned, selector)
            if {row["candidate_identity"] for row in ranked} != {identity(row) for row in learned}:
                raise RuntimeError("selector changed immutable candidate membership")
            cases.append(
                {
                    "source_case_id": case["source_case_id"],
                    "candidate_count": len(ranked),
                    "ranked_candidates": [
                        {
                            "candidate_identity": row["candidate_identity"],
                            "original_rank": row["original_rank"],
                            "selector_rank": rank,
                            "selector_score": row["selector_score"],
                            "complete_macro_signature": list(row["complete_macro_signature"]),
                        }
                        for rank, row in enumerate(ranked, 1)
                    ],
                }
            )
        folds.append(
            {
                "fold": fold,
                "status": "ranked",
                "training_identity": selector.training_identity,
                "cases": cases,
            }
        )
    publish(
        output,
        {
            "schema_version": LOCK_SCHEMA,
            "baseline_candidate_lock_payload_sha256": contract["inputs"]["candidate_lock"][
                "payload_sha256"
            ],
            "folds": folds,
            "teacher_fields_present": False,
            "task_identity_present": False,
            "candidate_generation_calls": 0,
            "new_oracle_calls": 0,
        },
        compressed=True,
    )


def _decorate(source_group: str, source_state: dict, candidate: dict) -> dict:
    records = _generated_records(source_group, source_state, candidate)
    return {
        **candidate,
        "candidate_identity": identity(candidate),
        "endpoint_key": canonical_state_key(decode_state(candidate["endpoint_state"])),
        "complete_macro_signature": complete_macro_signature(candidate),
        "component_records": [
            {
                family: [_component_payload(value) for value in row["components"][family]]
                for family in FAMILIES
            }
            for row in records
        ],
    }


def _coverage_and_precision(
    generated: list[dict], teachers: list[dict], train_vocab, cutoff: int
) -> dict:
    result = _coverage(generated, teachers, train_vocab, cutoff)
    selected = generated[:cutoff]
    generated_components = defaultdict(list)
    for candidate in selected:
        for record in candidate["component_records"]:
            for family, rows in record.items():
                generated_components[family].extend(_component_from_payload(row) for row in rows)
    for family in FAMILIES:
        target = [
            _component_from_payload(value)
            for teacher in teachers
            for component in teacher["components"]
            for value in component[family]
        ]
        target_vocab = ExactComponentVocabulary(target)
        generated_rows = generated_components[family]
        generated_unique = ExactComponentVocabulary(generated_rows)
        matched = sum(target_vocab.contains(row) for row in generated_rows)
        matched_unique = sum(target_vocab.contains(row) for row in generated_unique.representatives)
        result["families"][family].update(
            {
                "generated_instances": len(generated_rows),
                "generated_matching_instances": matched,
                "generated_precision": matched / len(generated_rows) if generated_rows else None,
                "generated_unique": generated_unique.size,
                "generated_matching_unique": matched_unique,
                "generated_unique_precision": (
                    matched_unique / generated_unique.size if generated_unique.size else None
                ),
            }
        )
    result["unique_complete_macro_yield"] = len(
        {row["complete_macro_signature"] for row in selected}
    )
    work = [compiler_log_work(row) for row in selected]
    result["compiler_log_work"] = {
        "mean": float(np.mean(work)) if work else None,
        "maximum": max(work) if work else None,
    }
    return result


def _source_balanced_component_summary(source_rows: list[dict], policy: str, cutoff: int) -> dict:
    rows = [row["policies"][policy]["cutoffs"][str(cutoff)] for row in source_rows]
    families = {}
    for family in FAMILIES:
        family_rows = [row["families"][family] for row in rows]
        families[family] = {
            "held_coverage": float(np.mean([row["held_coverage"] for row in family_rows])),
            "generated_precision": float(
                np.mean([row["generated_precision"] for row in family_rows])
            ),
            "held_recovered": sum(row["held_recovered"] for row in family_rows),
            "held_instances": sum(row["held_instances"] for row in family_rows),
            "generated_matching_instances": sum(
                row["generated_matching_instances"] for row in family_rows
            ),
            "generated_instances": sum(row["generated_instances"] for row in family_rows),
        }
    return {
        "exact_patch_recall": families["whole_patch"]["held_coverage"],
        "exact_patch_precision": families["whole_patch"]["generated_precision"],
        "granular_component_coverage": float(
            np.mean([families[name]["held_coverage"] for name in GRANULAR_COMPONENT_FAMILIES])
        ),
        "granular_component_precision": float(
            np.mean([families[name]["generated_precision"] for name in GRANULAR_COMPONENT_FAMILIES])
        ),
        "unique_endpoint_yield": float(np.mean([row["unique_endpoint_yield"] for row in rows])),
        "unique_complete_macro_yield": float(
            np.mean([row["unique_complete_macro_yield"] for row in rows])
        ),
        "unique_patch_yield": float(np.mean([row["unique_patch_yield"] for row in rows])),
        "novel_whole_patch_yield": float(np.mean([row["novel_whole_patch_yield"] for row in rows])),
        "compiler_log_work_mean": float(
            np.mean([row["compiler_log_work"]["mean"] for row in rows])
        ),
        "families": families,
    }


def _offline_ordering_quality(payload: dict, *, cutoffs: tuple[int, ...]) -> dict:
    """Evaluate recovery while refusing degenerate zero-time throughput claims."""

    result = evaluate_quality(payload, cutoffs=cutoffs)
    body = {key: value for key, value in result.items() if key != "report_sha256"}
    for policy in body["policies"].values():
        policy["aggregate"]["attempts_per_second"] = None
        policy["aggregate"]["unique_endpoints_per_second"] = None
    body["limitations"].append(
        "Proposal throughput is unsupported because this benchmark reorders an immutable lock "
        "and performs no candidate generation."
    )
    return {**body, "report_sha256": identity(body)}


def evaluate_all(input_root: Path, selector_lock_path: Path, output: Path) -> None:
    contract = _contract()
    baseline_lock, evaluations, _ = _load_catalog(contract)
    selector_lock = load_sealed(selector_lock_path)
    if (
        selector_lock.get("schema_version") != LOCK_SCHEMA
        or selector_lock.get("teacher_fields_present") is not False
        or selector_lock.get("candidate_generation_calls") != 0
    ):
        raise ValueError("selector candidate lock invariant failed")
    selector_folds = {int(row["fold"]): row for row in selector_lock["folds"]}
    fold_reports = []
    all_quality_cases = []
    all_source_rows = []
    for locked_fold in baseline_lock["folds"]:
        fold = int(locked_fold["fold"])
        selection_fold = selector_folds[fold]
        if selection_fold["status"] != "ranked":
            fold_reports.append({"fold": fold, "status": "abstained"})
            continue
        evaluation = evaluations[fold]
        references = {row["source_case_id"]: row for row in evaluation["cases"]}
        selections = {row["source_case_id"]: row for row in selection_fold["cases"]}
        quality_cases = []
        source_rows = []
        for locked_case in locked_fold["cases"]:
            case_id = locked_case["source_case_id"]
            reference = references[case_id]
            selection = selections[case_id]
            policies = {row["policy_id"]: row for row in locked_case["policies"]}
            learned = policies[POLICY_LEARNED]["candidates"]
            by_identity = {identity(row): row for row in learned}
            selector_candidates = [
                by_identity[row["candidate_identity"]] for row in selection["ranked_candidates"]
            ]
            if set(map(identity, selector_candidates)) != set(map(identity, learned)):
                raise RuntimeError("selector evaluation changed candidate support")
            ordered = {
                POLICY_RAW_UNIFORM: policies[POLICY_UNIFORM]["candidates"],
                POLICY_RAW_LEARNED: learned,
                POLICY_SELECTOR: selector_candidates,
            }
            decorated_cache = {}
            metric_policies = {}
            quality_policies = []
            train_vocab = ExactComponentVocabulary(
                _component_from_payload(row) for row in evaluation["train_whole_patch_vocabulary"]
            )
            for policy_id, candidates in ordered.items():
                generated = []
                for candidate in candidates:
                    candidate_identity = identity(candidate)
                    if candidate_identity not in decorated_cache:
                        decorated_cache[candidate_identity] = _decorate(
                            reference["source_group"], reference["source_state"], candidate
                        )
                    generated.append(decorated_cache[candidate_identity])
                metric_policies[policy_id] = {
                    "cutoffs": {
                        str(cutoff): _coverage_and_precision(
                            generated, reference["teachers"], train_vocab, cutoff
                        )
                        for cutoff in contract["selection"]["cutoffs"]
                    },
                    "telemetry": (
                        policies[POLICY_UNIFORM]["telemetry"]
                        if policy_id == POLICY_RAW_UNIFORM
                        else policies[POLICY_LEARNED]["telemetry"]
                    ),
                }
                quality_policies.append(
                    {
                        "policy_id": policy_id,
                        "proposal_seconds": 0.0,
                        "attempts": [
                            {
                                "attempt_id": identity(candidate),
                                "rank": rank,
                                "status": "complete",
                                "endpoint_state": candidate["endpoint_state"],
                            }
                            for rank, candidate in enumerate(candidates, 1)
                        ],
                    }
                )
            source_rows.append(
                {
                    "source_id": reference["source_group"],
                    "teachers": len(reference["teachers"]),
                    "policies": metric_policies,
                }
            )
            quality_cases.append(
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
                    "policy_pools": quality_policies,
                }
            )
        quality = _offline_ordering_quality(
            {
                "schema_version": QUALITY_SCHEMA,
                "oracle_calls": 0,
                "split": {
                    "evaluation_role": "test",
                    "train_sources": evaluation["train_sources"],
                    "calibration_sources": [],
                    "test_sources": evaluation["held_sources"],
                },
                "cases": quality_cases,
            },
            cutoffs=tuple(contract["selection"]["cutoffs"]),
        )
        fold_reports.append(
            {
                "fold": fold,
                "status": "evaluated",
                "sources": source_rows,
                "endpoint_and_transformation_quality": quality,
            }
        )
        all_quality_cases.extend(quality_cases)
        all_source_rows.extend(source_rows)
    policies = (POLICY_RAW_UNIFORM, POLICY_RAW_LEARNED, POLICY_SELECTOR)
    overall_quality = _offline_ordering_quality(
        {
            "schema_version": QUALITY_SCHEMA,
            "oracle_calls": 0,
            "split": {
                "evaluation_role": "test",
                "train_sources": [],
                "calibration_sources": [],
                "test_sources": sorted(row["source_id"] for row in all_quality_cases),
            },
            "cases": all_quality_cases,
        },
        cutoffs=tuple(contract["selection"]["cutoffs"]),
    )
    summaries = {
        policy: {
            "cutoffs": {
                str(cutoff): _source_balanced_component_summary(all_source_rows, policy, cutoff)
                for cutoff in contract["selection"]["cutoffs"]
            },
            "endpoint_and_transformation_quality": overall_quality["policies"][policy],
        }
        for policy in policies
    }
    selector_precision = overall_quality["policies"][POLICY_SELECTOR]["aggregate"][
        "execution_precision"
    ]
    raw_precision = overall_quality["policies"][POLICY_RAW_LEARNED]["aggregate"][
        "execution_precision"
    ]
    diversity_floor = all(
        summaries[POLICY_SELECTOR]["cutoffs"][str(cutoff)][name]
        >= summaries[POLICY_RAW_LEARNED]["cutoffs"][str(cutoff)][name]
        for cutoff in contract["selection"]["cutoffs"]
        for name in ("unique_endpoint_yield", "unique_complete_macro_yield")
    )
    improving_cutoffs = []
    for cutoff in contract["selection"]["cutoffs"]:
        name = str(cutoff)
        selector_quality = overall_quality["policies"][POLICY_SELECTOR]["aggregate"]["cutoffs"][
            name
        ]
        raw_quality = overall_quality["policies"][POLICY_RAW_LEARNED]["aggregate"]["cutoffs"][name]
        selector_components = summaries[POLICY_SELECTOR]["cutoffs"][name]
        raw_components = summaries[POLICY_RAW_LEARNED]["cutoffs"][name]
        if any(
            left > right
            for left, right in (
                (selector_quality["exact_recall"], raw_quality["exact_recall"]),
                (
                    selector_quality["transformation_recall"],
                    raw_quality["transformation_recall"],
                ),
                (selector_components["exact_patch_recall"], raw_components["exact_patch_recall"]),
                (
                    selector_components["granular_component_coverage"],
                    raw_components["granular_component_coverage"],
                ),
            )
        ):
            improving_cutoffs.append(cutoff)
    gates = {
        "all_folds_evaluated": len(fold_reports) == 3
        and all(row["status"] == "evaluated" for row in fold_reports),
        "selector_changes_order_only": all(
            len(row["policy_pools"]) == 3 for row in all_quality_cases
        ),
        "all_selected_candidates_exactly_realized": selector_precision == 1.0,
        "diversity_floor": diversity_floor,
        "scientific_improvement_gate": bool(improving_cutoffs)
        and selector_precision >= raw_precision
        and diversity_floor,
        "improving_cutoffs": improving_cutoffs,
        "exact_teacher_recovery_required": False,
    }
    gates["passed"] = all(
        value
        for key, value in gates.items()
        if key not in {"improving_cutoffs", "exact_teacher_recovery_required"}
    )
    revision = _implementation_revision()
    publish(
        output,
        {
            "schema_version": RESULT_SCHEMA,
            "contract": {
                "path": str(CONTRACT.relative_to(ROOT)),
                "sha256": sha256(CONTRACT),
                "payload_sha256": identity(contract),
            },
            "implementation_revision": revision,
            "inputs": {
                name: {"path": row["path"], "sha256": row["sha256"]}
                for name, row in contract["inputs"].items()
            },
            "folds": fold_reports,
            "overall": {
                "summaries": summaries,
                "endpoint_and_transformation_quality": overall_quality,
                "gates": gates,
                "unsupported": {
                    "realizability_classifier": (
                        "abstained: rejected compiler attempts are not rows in the immutable lock"
                    )
                },
            },
            "costs": {
                "candidate_generation_calls": 0,
                "new_oracle_calls": 0,
                "new_docking_calls": 0,
                "modal_launches": 0,
                "device": "cpu",
                "workers": 1,
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "networkx": nx.__version__,
                "rdkit": rdBase.rdkitVersion,
            },
            "interpretation": {
                "evidence": "offline zero-oracle reranking of an immutable generated lock",
                "exact_teacher_recovery_is_diagnostic": True,
                "molecular_utility_tested": False,
                "ivg_comparison_tested": False,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "rank", "evaluate", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--selector-lock", type=Path)
    arguments = parser.parse_args()
    root = arguments.input_root or arguments.output
    selector_lock = arguments.selector_lock or root / "candidate_lock.json.gz"
    if arguments.command in {"fit", "run"}:
        fit_all(arguments.output)
    if arguments.command in {"rank", "run"}:
        rank_all(root, arguments.output / "candidate_lock.json.gz")
    if arguments.command in {"evaluate", "run"}:
        evaluate_all(root, selector_lock, arguments.output / "result.json")


if __name__ == "__main__":
    main()
