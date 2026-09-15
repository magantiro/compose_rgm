#!/usr/bin/env python3
"""Fit, lock and evaluate the split-first compositional T4 patch generator."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
from rdkit import rdBase

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.compositional_structural_subgoal_generator import (
    POLICY_LEARNED,
    POLICY_UNIFORM,
    CompositionalPatchGenerator,
    PatchTrainingEvent,
    _mutable_after,
    fit_compositional_patch_generator,
    generate_compositional_patches,
)
from compose_v4.control.dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import (
    EditProgram,
    ProgramExecutionError,
    execute_bound_program,
)
from compose_v4.control.edit_program import attachment_bindings as program_bindings
from compose_v4.control.generic_legal_action_policy import (
    RULES,
    LegalSuccessor,
    _record_parts,
)
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    extract_structural_goal,
)
from compose_v4.control.structural_subgoal_components import (
    GRANULAR_COMPONENT_FAMILIES,
    AttributedComponent,
    ExactComponentVocabulary,
)
from compose_v4.control.structural_subgoal_policy import (
    materialize_template,
    minimize_subgoal,
)
from compose_v4.experiments.route_proposal_quality import SCHEMA as QUALITY_SCHEMA
from compose_v4.experiments.route_proposal_quality import evaluate as evaluate_quality
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import predeclared_source_folds
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_structural_subgoal_audit import LIBRARY, teacher_traces
from tools.t4_structural_subgoal_component_novelty import (
    _records_from_trace,
    load_sealed,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_compositional_structural_subgoal_generator_v1.json"
T4_CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
AUDIT = ROOT / "diagnostics/t4_structural_subgoal_component_novelty/attempt_1/result.json"
REALIZER = ROOT / "src/compose_v4/control/structural_subgoal_realizer.py"
EXTRACTOR = ROOT / "src/compose_v4/control/structural_subgoal.py"

RUNTIME_SCHEMA = "t4_compositional_structural_subgoal_runtime_v1"
SOURCE_SCHEMA = "t4_compositional_structural_subgoal_sources_v1"
EVALUATION_SCHEMA = "t4_compositional_structural_subgoal_evaluation_v1"
LOCK_SCHEMA = "t4_compositional_structural_subgoal_candidate_lock_v1"
REPORT_SCHEMA = "t4_compositional_structural_subgoal_fold_report_v1"
RESULT_SCHEMA = "t4_compositional_structural_subgoal_result_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite compositional artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(encoded, mtime=0) if compressed else encoded)
    temporary.replace(path)


def _contract() -> dict:
    envelope = json.loads(CONTRACT.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("contract_sha256") != identity(payload):
        raise ValueError("compositional generator contract is not self-hashed")
    if any(
        payload["costs"][name] != 0
        for name in (
            "oracle_calls_authorized",
            "docking_calls_authorized",
            "modal_launches_authorized",
        )
    ):
        raise ValueError("compositional generator contract authorizes external evaluation")
    paths = {
        "component_novelty_result": AUDIT,
        "teacher_library": LIBRARY,
        "source_registry": SEEDS,
        "t4_benchmark_contract": T4_CONTRACT,
        "sealed_realizer_summary": ROOT
        / "diagnostics/t4_structural_subgoal/attempt_2/summary.json",
        "sealed_realizer_source": REALIZER,
        "sealed_structural_extractor_source": EXTRACTOR,
    }
    for name, path in paths.items():
        if sha256(path) != payload["inputs"][name]["sha256"]:
            raise ValueError(f"compositional generator input hash changed: {path}")
    audit = load_sealed(AUDIT)
    expected_payload = payload["inputs"]["component_novelty_result"]["payload_sha256"]
    if identity(audit) != expected_payload:
        raise ValueError("component-novelty payload identity changed")
    return payload


def _component_payload(component: AttributedComponent) -> dict:
    return asdict(component)


def _component_from_payload(payload: dict) -> AttributedComponent:
    return AttributedComponent(
        family=str(payload["family"]),
        node_labels=tuple(payload["node_labels"]),
        edges=tuple(tuple(row) for row in payload["edges"]),
        directed=bool(payload["directed"]),
    )


def _trace_regions(states: tuple[dict, ...], actions: tuple[dict, ...]) -> dict:
    return dependency_region_program(
        states,
        actions,
        DependencyRegionConfig(
            runtime_maximum_primitives=32,
            runtime_maximum_components=4,
            join_lifetime_neighbors=False,
        ),
    )


def _t4_data() -> tuple[list[dict], list[dict], dict[str, dict]]:
    raw = teacher_traces()
    metadata = source_group_map(unseal(T4_CONTRACT), json.loads(SEEDS.read_text()))
    traces, records = [], []
    for teacher in raw:
        states = tuple(teacher["trace"]["states"])
        actions = tuple(teacher["trace"]["actions"])
        rows, _ = _records_from_trace(
            source_group=teacher["source_group"],
            route_id=teacher["program_id"],
            states=states,
            actions=actions,
            metadata={"origin": "t4_complete_route"},
        )
        records.extend(rows)
        traces.append(
            {
                "domain": "t4_complete",
                "lineage": teacher["source_group"],
                "source_group": teacher["source_group"],
                "route_id": teacher["program_id"],
                "states": states,
                "actions": actions,
                "regions": _trace_regions(states, actions),
            }
        )
    if len(traces) != 77 or len(records) != 147 or len(metadata) != 15:
        raise RuntimeError("T4 compositional training census changed")
    return traces, records, metadata


def _full146_data(metadata: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    library = json.loads(LIBRARY.read_text())
    remaining = [
        row
        for row in library
        if [block["label"] for block in row["program"]["blocks"]]
        != ["compiled_complete_transformation"]
    ]
    source_graphs = {
        source_group: decode_state(unseal(T4_CONTRACT)["cells"][row["cell"]]["source_state"])
        for source_group, row in metadata.items()
    }
    traces, records = [], []
    failures = Counter()
    applications = 0
    for library_row in remaining:
        program = EditProgram.from_payload(library_row["program"])
        for source_group in library_row["source_groups"]:
            applications += 1
            source = source_graphs[source_group]
            census = program_bindings(
                program, source, max_bindings=64, max_visits=4096, contextual=True
            )
            assignments = [
                assignment
                for assignment, distance in zip(
                    census.assignments, census.context_distances, strict=True
                )
                if distance == 0
            ]
            if not assignments:
                failures["no_exact_origin_context_binding"] += 1
                continue
            receipt = None
            for assignment in assignments:
                try:
                    _, receipt = execute_bound_program(
                        source, program, assignment, max_primitives=32, max_blocks=8
                    )
                except (ProgramExecutionError, ValueError):
                    continue
                break
            if receipt is None:
                failures["no_exact_executable_origin_binding"] += 1
                continue
            states = tuple(receipt["states"])
            actions = tuple(receipt["actions"])
            try:
                rows, _ = _records_from_trace(
                    source_group=source_group,
                    route_id=program.program_id,
                    states=states,
                    actions=actions,
                    metadata={"origin": "full146_non_complete_route"},
                )
            except (RuntimeError, TypeError, ValueError) as error:
                failures[f"structural_conversion:{type(error).__name__}"] += 1
                continue
            records.extend(rows)
            traces.append(
                {
                    "domain": "full146_auxiliary",
                    "lineage": source_group,
                    "source_group": source_group,
                    "route_id": program.program_id,
                    "states": states,
                    "actions": actions,
                    "regions": _trace_regions(states, actions),
                }
            )
    summary = {
        "programs": len(remaining),
        "applications": applications,
        "converted_routes": len(traces),
        "subgoals": len(records),
        "failures": dict(sorted(failures.items())),
    }
    if summary["converted_routes"] != 55 or summary["subgoals"] != 72:
        raise RuntimeError(f"Full-146 converted census changed: {summary}")
    return traces, records, summary


def _pmo_data(path: Path) -> tuple[list[dict], list[dict], dict]:
    corpus = load_sealed(path)
    if corpus.get("schema_version") != "pmo_dependency_region_training_corpus_v2":
        raise ValueError("PMO dependency-region corpus schema changed")
    traces, records = [], []
    failures = Counter()
    for route in corpus["routes"]:
        states = tuple(route["states"])
        actions = tuple(route["actions"])
        try:
            rows, _ = _records_from_trace(
                source_group=f"pmo:{route['trace_identity']}",
                route_id=route["trace_identity"],
                states=states,
                actions=actions,
                regions_override=route["dependency_region_program"],
                metadata={
                    "origin": "pmo_dependency_region",
                    "task_family": route["task_family"],
                    "test_fold": route["test_fold"],
                },
            )
        except (RuntimeError, TypeError, ValueError) as error:
            failures[f"structural_conversion:{type(error).__name__}"] += 1
            continue
        records.extend(rows)
        traces.append(
            {
                "domain": "pmo_dependency_region",
                "lineage": route["task_family"],
                "source_group": f"pmo:{route['trace_identity']}",
                "route_id": route["trace_identity"],
                "states": states,
                "actions": actions,
                "regions": route["dependency_region_program"],
            }
        )
    summary = {
        "routes": len(corpus["routes"]),
        "converted_routes": len(traces),
        "subgoals": len(records),
        "failures": dict(sorted(failures.items())),
    }
    if summary["converted_routes"] != 85 or summary["subgoals"] != 115:
        raise RuntimeError(f"PMO converted census changed: {summary}")
    return traces, records, summary


def _training_events(trace: dict) -> tuple[list[PatchTrainingEvent], Counter]:
    states, actions = trace["states"], trace["actions"]
    system = editing_v2_semantic_rewrite_system()
    rows = []
    telemetry = Counter()
    for component_index, component in enumerate(trace["regions"]["components"]):
        indices = tuple(map(int, component["primitive_indices"]))[:2]
        if not indices:
            telemetry["empty_components"] += 1
            continue
        component_source = decode_state(states[indices[0]])
        first_current = component_source
        _, references, _, _ = _record_parts(actions[indices[0]], first_current)
        region = tuple(
            sorted(
                slot for slot in references if bool(is_element(component_source.atom_types[slot]))
            )
        )
        if not region:
            telemetry["no_live_region"] += 1
            continue
        previous = None
        for depth, primitive_index in enumerate(indices):
            current = decode_state(states[primitive_index])
            successor = decode_state(states[primitive_index + 1])
            rule, action = decode_action(actions[primitive_index])
            if rule not in RULES:
                telemetry[f"unsupported_rule:{rule}"] += 1
                continue
            replayed = system.apply(current, rule, action)
            if canonical_state_key(replayed) != canonical_state_key(successor):
                raise RuntimeError("training event failed exact executor replay")
            candidate = LegalSuccessor(
                rule, actions[primitive_index], successor, canonical_state_key(successor)
            )
            rows.append(
                PatchTrainingEvent(
                    domain=trace["domain"],
                    lineage=trace["lineage"],
                    route_id=trace["route_id"],
                    component_id=f"component-{component_index}",
                    component_events=len(indices),
                    depth=depth,
                    component_source=component_source,
                    current=current,
                    candidate=candidate,
                    region_slots=region,
                    mutable_slots=_mutable_after(component_source, current, region),
                    previous_rule=previous,
                )
            )
            previous = rule
            telemetry["events"] += 1
        telemetry["components"] += 1
    return rows, telemetry


def _route_teacher(trace: dict, records: list[dict]) -> dict:
    goal, bindings, _ = extract_structural_goal(trace["states"], trace["actions"])
    minimized = []
    for subgoal, binding in zip(goal.subgoals, bindings, strict=True):
        template, retained = minimize_subgoal(subgoal)
        minimized.append(
            (
                template,
                tuple(binding[index] for index in retained),
            )
        )
    source = decode_state(trace["states"][0])
    minimal_goal = StructuralGoal(
        tuple(materialize_template(template, source, binding) for template, binding in minimized)
    )
    matching_records = sorted(
        (row for row in records if row["route_id"] == trace["route_id"]),
        key=lambda row: row["subgoal_index"],
    )
    return {
        "teacher_id": identity(
            {
                "schema_version": "held_t4_compositional_teacher_v1",
                "endpoint": canonical_state_key(decode_state(trace["states"][-1])),
                "patches": [row[0].template_id for row in minimized],
            }
        ),
        "endpoint_state": trace["states"][-1],
        "goal": minimal_goal.payload(),
        "bindings": [list(row[1]) for row in minimized],
        "components": [
            {
                family: [_component_payload(value) for value in row["components"][family]]
                for family in ("whole_patch", *GRANULAR_COMPONENT_FAMILIES)
            }
            for row in matching_records
        ],
    }


def fit_all(output: Path, *, pmo_corpus: Path, pmo_result: Path) -> None:
    contract = _contract()
    if sha256(pmo_corpus) != contract["inputs"]["optional_pmo_dependency_region_corpus"]["sha256"]:
        raise ValueError("PMO corpus physical hash changed")
    if sha256(pmo_result) != contract["inputs"]["optional_pmo_dependency_region_result"]["sha256"]:
        raise ValueError("PMO result physical hash changed")
    load_sealed(pmo_result)
    t4_traces, t4_records, metadata = _t4_data()
    full_traces, full_records, full_summary = _full146_data(metadata)
    pmo_traces, pmo_records, pmo_summary = _pmo_data(pmo_corpus)
    split_rows = predeclared_source_folds(metadata)
    proposal = contract["decoder"]
    for split in split_rows:
        fold = int(split["fold"])
        train_sources = set(split["train_sources"])
        held_sources = set(split["test_sources"])
        if train_sources & held_sources:
            raise RuntimeError("whole-source split leakage")
        fit_traces = [
            row for row in (*t4_traces, *full_traces) if row["source_group"] in train_sources
        ]
        fit_traces.extend(pmo_traces)
        events = []
        training_telemetry = Counter()
        for trace in fit_traces:
            trace_events, telemetry = _training_events(trace)
            events.extend(trace_events)
            training_telemetry.update(telemetry)
        model = fit_compositional_patch_generator(
            events, variance_floor=float(contract["model"]["variance_floor"])
        )
        runtime = {
            "schema_version": RUNTIME_SCHEMA,
            "model": model.checkpoint(),
            "decoder": proposal,
            "fold": fold,
            "split_identity": identity(
                {
                    "schema_version": "whole_source_fold_identity_v1",
                    "fold": fold,
                    "train_sources": sorted(train_sources),
                    "held_sources": sorted(held_sources),
                }
            ),
            "contract_payload_sha256": identity(contract),
            "input_sha256": {
                name: value["sha256"]
                for name, value in contract["inputs"].items()
                if "sha256" in value
            },
            "new_oracle_calls": 0,
        }
        serialized = json.dumps(runtime, sort_keys=True).lower()
        for forbidden in (
            "route_id",
            "program_id",
            "template_id",
            "patch_id",
            "source_state",
            "endpoint_state",
            "smiles",
            "teacher_trace",
        ):
            if forbidden in serialized:
                raise RuntimeError(f"runtime checkpoint contains forbidden field: {forbidden}")
        held = [row for row in t4_traces if row["source_group"] in held_sources]
        by_source = defaultdict(list)
        for row in held:
            by_source[row["source_group"]].append(row)
        source_cases = []
        evaluation_cases = []
        for source_group in sorted(by_source):
            teachers = by_source[source_group]
            source_state = teachers[0]["states"][0]
            case_id = identity(
                {
                    "schema_version": "held_compositional_source_case_v1",
                    "fold": fold,
                    "source_state": source_state,
                }
            )
            source_cases.append({"source_case_id": case_id, "source_state": source_state})
            evaluation_cases.append(
                {
                    "source_case_id": case_id,
                    "source_group": source_group,
                    "source_state": source_state,
                    "teachers": [_route_teacher(row, t4_records) for row in teachers],
                }
            )
        train_records = [
            row for row in (*t4_records, *full_records) if row["source_group"] in train_sources
        ]
        train_records.extend(pmo_records)
        whole_vocab = ExactComponentVocabulary(
            component for row in train_records for component in row["components"]["whole_patch"]
        )
        fold_root = output / f"fold_{fold}"
        publish(fold_root / "runtime_checkpoint.json.gz", runtime, compressed=True)
        publish(
            fold_root / "source_manifest.json",
            {
                "schema_version": SOURCE_SCHEMA,
                "fold": fold,
                "source_cases": source_cases,
                "teacher_fields_present": False,
                "task_identity_present": False,
                "new_oracle_calls": 0,
            },
        )
        publish(
            fold_root / "evaluation_manifest.json.gz",
            {
                "schema_version": EVALUATION_SCHEMA,
                "fold": fold,
                "train_sources": sorted(train_sources),
                "held_sources": sorted(held_sources),
                "cases": evaluation_cases,
                "train_whole_patch_vocabulary": [
                    _component_payload(row) for row in whole_vocab.representatives
                ],
                "new_oracle_calls": 0,
            },
            compressed=True,
        )
        publish(
            fold_root / "fit_report.json",
            {
                "schema_version": "t4_compositional_structural_subgoal_fit_report_v1",
                "fold": fold,
                "training_sources": len(train_sources),
                "held_sources": len(held_sources),
                "training_trace_counts": dict(Counter(row["domain"] for row in fit_traces)),
                "training_event_counts": model.training_summary,
                "training_telemetry": dict(sorted(training_telemetry.items())),
                "auxiliary": {"full146": full_summary, "pmo": pmo_summary},
                "new_oracle_calls": 0,
            },
        )


def generate_all(input_root: Path, output: Path) -> None:
    _contract()
    folds = []
    for fold in range(3):
        runtime = load_sealed(input_root / f"fold_{fold}/runtime_checkpoint.json.gz")
        sources = load_sealed(input_root / f"fold_{fold}/source_manifest.json")
        if runtime.get("schema_version") != RUNTIME_SCHEMA:
            raise ValueError("compositional runtime schema mismatch")
        if sources.get("schema_version") != SOURCE_SCHEMA:
            raise ValueError("compositional source schema mismatch")
        if sources.get("teacher_fields_present") is not False:
            raise ValueError("compositional generation received teacher fields")
        model = CompositionalPatchGenerator.from_checkpoint(runtime["model"])
        decoder = runtime["decoder"]
        cases = []
        for index, case in enumerate(sources["source_cases"], 1):
            source = decode_state(case["source_state"])
            policies = []
            for policy_id, policy_model in (
                (POLICY_UNIFORM, None),
                (POLICY_LEARNED, model),
            ):
                began = perf_counter()
                candidates, telemetry = generate_compositional_patches(
                    source,
                    policy_model,
                    pool_size=int(decoder["candidate_pool_size"]),
                    first_event_beam=int(decoder["first_event_beam"]),
                    second_event_beam=int(decoder["second_event_beam"]),
                    second_event_expansion_per_prefix=int(
                        decoder["second_event_expansion_per_prefix"]
                    ),
                    per_rule_first_event_cap=int(decoder["per_rule_first_event_cap"]),
                )
                policies.append(
                    {
                        "policy_id": policy_id,
                        "candidates": [row.payload() for row in candidates],
                        "telemetry": telemetry,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "compositional_source_complete",
                            "fold": fold,
                            "source_index": index,
                            "policy": policy_id,
                            "candidates": len(candidates),
                            "wall_seconds": perf_counter() - began,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            cases.append(
                {
                    "source_case_id": case["source_case_id"],
                    "source_state": case["source_state"],
                    "policies": policies,
                }
            )
        folds.append(
            {
                "fold": fold,
                "training_identity": model.training_identity,
                "cases": cases,
            }
        )
    publish(
        output,
        {
            "schema_version": LOCK_SCHEMA,
            "folds": folds,
            "teacher_fields_present": False,
            "task_identity_present": False,
            "new_oracle_calls": 0,
        },
        compressed=output.suffix == ".gz",
    )


def _replay(source_state: dict, actions: tuple[dict, ...]) -> tuple[dict, ...]:
    system = editing_v2_semantic_rewrite_system()
    graph = decode_state(source_state)
    states = [source_state]
    for record in actions:
        rule, action = decode_action(record)
        graph = system.apply(graph, rule, action)
        states.append(encode_state(graph))
    return tuple(states)


def _generated_records(source_group: str, source_state: dict, candidate: dict) -> list[dict]:
    actions = tuple(candidate["actions"])
    states = _replay(source_state, actions)
    rows, _ = _records_from_trace(
        source_group=source_group,
        route_id=identity(candidate),
        states=states,
        actions=actions,
        metadata={"origin": "autonomous_compositional_patch"},
    )
    return rows


def _coverage(
    generated: list[dict], teachers: list[dict], train_vocab: ExactComponentVocabulary, cutoff: int
) -> dict:
    selected = generated[:cutoff]
    generated_components = defaultdict(list)
    for candidate in selected:
        for record in candidate["component_records"]:
            for family, rows in record.items():
                generated_components[family].extend(_component_from_payload(row) for row in rows)
    metrics = {}
    for family in ("whole_patch", *GRANULAR_COMPONENT_FAMILIES):
        vocabulary = ExactComponentVocabulary(generated_components[family])
        target = [
            _component_from_payload(value)
            for teacher in teachers
            for component in teacher["components"]
            for value in component[family]
        ]
        familiar = sum(vocabulary.contains(row) for row in target)
        metrics[family] = {
            "held_instances": len(target),
            "held_recovered": familiar,
            "held_coverage": familiar / len(target) if target else None,
        }
    generated_whole = generated_components["whole_patch"]
    unique = ExactComponentVocabulary(generated_whole)
    novel = sum(not train_vocab.contains(row) for row in unique.representatives)
    return {
        "families": metrics,
        "unique_patch_yield": unique.size,
        "novel_whole_patch_yield": novel,
        "novel_whole_patch_fraction": novel / unique.size if unique.size else 0,
        "unique_endpoint_yield": len({row["endpoint_key"] for row in selected}),
        "fixed_k_yield": len(selected) / cutoff,
    }


def _overall_summary(fold_reports: list[dict]) -> dict:
    """Reduce the frozen fold reports without changing any selection decision."""

    all_sources = [source for fold in fold_reports for source in fold["sources"]]
    policies = {}
    for policy_id in (POLICY_UNIFORM, POLICY_LEARNED):
        cutoffs = {}
        for cutoff in (8, 32, 128):
            name = str(cutoff)
            rows = [source["policies"][policy_id]["cutoffs"][name] for source in all_sources]
            quality_rows = [
                source
                for fold in fold_reports
                for source in fold["endpoint_and_transformation_quality"]["policies"][policy_id][
                    "per_source"
                ]
            ]
            family_counts = {
                family: {
                    "held_recovered": sum(
                        row["families"][family]["held_recovered"] for row in rows
                    ),
                    "held_instances": sum(
                        row["families"][family]["held_instances"] for row in rows
                    ),
                }
                for family in ("whole_patch", *GRANULAR_COMPONENT_FAMILIES)
            }
            for counts in family_counts.values():
                counts["instance_weighted_coverage"] = (
                    counts["held_recovered"] / counts["held_instances"]
                    if counts["held_instances"]
                    else None
                )
            granular_recovered = sum(
                family_counts[family]["held_recovered"] for family in GRANULAR_COMPONENT_FAMILIES
            )
            granular_instances = sum(
                family_counts[family]["held_instances"] for family in GRANULAR_COMPONENT_FAMILIES
            )
            cutoffs[name] = {
                "source_balanced": {
                    "exact_endpoint_recall": float(
                        np.mean([row["cutoffs"][name]["exact_recall"] for row in quality_rows])
                    ),
                    "transformation_equivalent_recall": float(
                        np.mean(
                            [row["cutoffs"][name]["transformation_recall"] for row in quality_rows]
                        )
                    ),
                    "exact_patch_recall": float(
                        np.mean([row["families"]["whole_patch"]["held_coverage"] for row in rows])
                    ),
                    "granular_component_coverage": float(
                        np.mean(
                            [
                                np.mean(
                                    [
                                        row["families"][family]["held_coverage"]
                                        for family in GRANULAR_COMPONENT_FAMILIES
                                    ]
                                )
                                for row in rows
                            ]
                        )
                    ),
                    "unique_patch_yield": float(
                        np.mean([row["unique_patch_yield"] for row in rows])
                    ),
                    "novel_whole_patch_yield": float(
                        np.mean([row["novel_whole_patch_yield"] for row in rows])
                    ),
                    "unique_endpoint_yield": float(
                        np.mean([row["unique_endpoint_yield"] for row in rows])
                    ),
                    "fixed_k_yield": float(np.mean([row["fixed_k_yield"] for row in rows])),
                },
                "instance_weighted": {
                    "exact_patch_recall": family_counts["whole_patch"][
                        "instance_weighted_coverage"
                    ],
                    "granular_component_coverage": granular_recovered / granular_instances,
                },
                "families": family_counts,
                "minimum_per_source": {
                    "novel_whole_patch_yield": min(row["novel_whole_patch_yield"] for row in rows),
                    "unique_endpoint_yield": min(row["unique_endpoint_yield"] for row in rows),
                    "fixed_k_yield": min(row["fixed_k_yield"] for row in rows),
                },
            }
        telemetry_rows = [source["policies"][policy_id]["telemetry"] for source in all_sources]
        telemetry = {
            key: sum(int(row.get(key, 0)) for row in telemetry_rows)
            for key in sorted({key for row in telemetry_rows for key in row})
        }
        compile_denominator = telemetry["legal_patch_compile_coverage_denominator"]
        realization_denominator = telemetry["exact_realization_precision_denominator"]
        policies[policy_id] = {
            "cutoffs": cutoffs,
            "work": telemetry,
            "legal_patch_compile_coverage": telemetry["legal_patch_compile_coverage_numerator"]
            / compile_denominator,
            "exact_realization_precision": telemetry["exact_realization_precision_numerator"]
            / realization_denominator,
        }
    learned = policies[POLICY_LEARNED]
    uniform = policies[POLICY_UNIFORM]
    improving_cutoffs = []
    for cutoff in (32, 128):
        name = str(cutoff)
        learned_metrics = learned["cutoffs"][name]["source_balanced"]
        uniform_metrics = uniform["cutoffs"][name]["source_balanced"]
        if (
            learned_metrics["granular_component_coverage"]
            > uniform_metrics["granular_component_coverage"]
            or learned_metrics["transformation_equivalent_recall"]
            > uniform_metrics["transformation_equivalent_recall"]
        ) and learned["exact_realization_precision"] >= uniform["exact_realization_precision"]:
            improving_cutoffs.append(cutoff)
    gates = {
        "all_held_sources_emit_unique_exactly_realized_patch": all(
            learned["cutoffs"]["128"]["minimum_per_source"][name] > 0
            for name in ("unique_endpoint_yield", "fixed_k_yield")
        ),
        "at_least_one_novel_whole_patch_per_held_source": learned["cutoffs"]["128"][
            "minimum_per_source"
        ]["novel_whole_patch_yield"]
        > 0,
        "generated_held_component_support_nonzero": learned["cutoffs"]["128"]["source_balanced"][
            "granular_component_coverage"
        ]
        > 0,
        "all_accepted_candidates_exactly_execute": all(
            policies[policy_id]["exact_realization_precision"] == 1
            for policy_id in (POLICY_UNIFORM, POLICY_LEARNED)
        ),
        "learned_improvement_gate": bool(improving_cutoffs),
        "learned_improving_cutoffs": improving_cutoffs,
        "exact_teacher_recovery_required": False,
    }
    gates["passed"] = all(
        value
        for key, value in gates.items()
        if key not in {"learned_improving_cutoffs", "exact_teacher_recovery_required"}
    )
    return {"policies": policies, "gates": gates}


def evaluate_all(input_root: Path, lock_path: Path, output: Path) -> None:
    _contract()
    lock = load_sealed(lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("teacher_fields_present") is not False:
        raise ValueError("candidate lock schema or separation invariant failed")
    fold_reports = []
    for locked_fold in lock["folds"]:
        fold = int(locked_fold["fold"])
        evaluation = load_sealed(input_root / f"fold_{fold}/evaluation_manifest.json.gz")
        if evaluation.get("schema_version") != EVALUATION_SCHEMA:
            raise ValueError("compositional evaluation schema mismatch")
        locked_cases = {row["source_case_id"]: row for row in locked_fold["cases"]}
        quality_cases = []
        source_metrics = []
        for reference in evaluation["cases"]:
            locked_case = locked_cases[reference["source_case_id"]]
            policies = []
            metric_policies = {}
            for policy in locked_case["policies"]:
                generated = []
                for candidate in policy["candidates"]:
                    records = _generated_records(
                        reference["source_group"], reference["source_state"], candidate
                    )
                    generated.append(
                        {
                            **candidate,
                            "endpoint_key": canonical_state_key(
                                decode_state(candidate["endpoint_state"])
                            ),
                            "component_records": [
                                {
                                    family: [
                                        _component_payload(value)
                                        for value in row["components"][family]
                                    ]
                                    for family in ("whole_patch", *GRANULAR_COMPONENT_FAMILIES)
                                }
                                for row in records
                            ],
                        }
                    )
                metric_policies[policy["policy_id"]] = {
                    "cutoffs": {
                        str(cutoff): _coverage(
                            generated,
                            reference["teachers"],
                            ExactComponentVocabulary(
                                _component_from_payload(row)
                                for row in evaluation["train_whole_patch_vocabulary"]
                            ),
                            cutoff,
                        )
                        for cutoff in (8, 32, 128)
                    },
                    "telemetry": policy["telemetry"],
                }
                policies.append(
                    {
                        "policy_id": policy["policy_id"],
                        "proposal_seconds": 0.0,
                        "attempts": [
                            {
                                "attempt_id": identity(candidate),
                                "rank": rank,
                                "status": "complete",
                                "endpoint_state": candidate["endpoint_state"],
                            }
                            for rank, candidate in enumerate(policy["candidates"], 1)
                        ],
                    }
                )
            source_metrics.append(
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
                    "policy_pools": policies,
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
                    "test_sources": evaluation["held_sources"],
                },
                "cases": quality_cases,
            },
            cutoffs=(8, 32, 128),
        )
        aggregate = {}
        for policy_id in (POLICY_UNIFORM, POLICY_LEARNED):
            aggregate[policy_id] = {"cutoffs": {}}
            for cutoff in (8, 32, 128):
                rows = [
                    row["policies"][policy_id]["cutoffs"][str(cutoff)] for row in source_metrics
                ]
                aggregate[policy_id]["cutoffs"][str(cutoff)] = {
                    "source_balanced": {
                        "exact_patch_recall": float(
                            np.mean(
                                [row["families"]["whole_patch"]["held_coverage"] for row in rows]
                            )
                        ),
                        "granular_component_coverage": float(
                            np.mean(
                                [
                                    np.mean(
                                        [
                                            row["families"][family]["held_coverage"]
                                            for family in GRANULAR_COMPONENT_FAMILIES
                                        ]
                                    )
                                    for row in rows
                                ]
                            )
                        ),
                        "unique_patch_yield": float(
                            np.mean([row["unique_patch_yield"] for row in rows])
                        ),
                        "novel_whole_patch_yield": float(
                            np.mean([row["novel_whole_patch_yield"] for row in rows])
                        ),
                        "unique_endpoint_yield": float(
                            np.mean([row["unique_endpoint_yield"] for row in rows])
                        ),
                        "fixed_k_yield": float(np.mean([row["fixed_k_yield"] for row in rows])),
                    }
                }
        fold_reports.append(
            {
                "schema_version": REPORT_SCHEMA,
                "fold": fold,
                "sources": source_metrics,
                "aggregate": aggregate,
                "endpoint_and_transformation_quality": quality,
                "new_oracle_calls": 0,
            }
        )
    publish(
        output,
        {
            "schema_version": RESULT_SCHEMA,
            "contract": {
                "path": str(CONTRACT.relative_to(ROOT)),
                "sha256": sha256(CONTRACT),
                "payload_sha256": identity(_contract()),
            },
            "folds": fold_reports,
            "overall": _overall_summary(fold_reports),
            "costs": {
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
            "implementation": {
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "generator_sha256": sha256(
                    ROOT / "src/compose_v4/control/compositional_structural_subgoal_generator.py"
                ),
                "runner_sha256": sha256(Path(__file__)),
                "realizer_sha256": sha256(REALIZER),
                "extractor_sha256": sha256(EXTRACTOR),
            },
            "new_oracle_calls": 0,
            "interpretation_limit": "Zero-oracle held-source structural generation only; no docking or IVG claim.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-all")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--pmo-corpus", type=Path, required=True)
    fit.add_argument("--pmo-result", type=Path, required=True)
    generate = commands.add_parser("generate-all")
    generate.add_argument("--input-root", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-all")
    evaluate.add_argument("--input-root", type=Path, required=True)
    evaluate.add_argument("--lock", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit-all":
        fit_all(args.output, pmo_corpus=args.pmo_corpus, pmo_result=args.pmo_result)
    elif args.command == "generate-all":
        generate_all(args.input_root, args.output)
    else:
        evaluate_all(args.input_root, args.lock, args.output)


if __name__ == "__main__":
    main()
