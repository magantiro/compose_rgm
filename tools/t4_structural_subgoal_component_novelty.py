#!/usr/bin/env python3
"""Grouped-source component-novelty and auxiliary-conversion audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
from rdkit import rdBase

from compose_v4.control.dependency_region_program import trace_structure
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import (
    EditProgram,
    ProgramExecutionError,
    execute_bound_program,
)
from compose_v4.control.edit_program import (
    attachment_bindings as program_bindings,
)
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    _extract_component,
    extract_structural_goal,
    instantiate_goal,
)
from compose_v4.control.structural_subgoal_components import (
    CORE_COMPONENT_FAMILIES,
    GRANULAR_COMPONENT_FAMILIES,
    AttributedComponent,
    ExactComponentVocabulary,
    component_families,
    dependency_motif,
)
from compose_v4.control.structural_subgoal_policy import (
    materialize_template,
    minimize_subgoal,
)
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import (
    predeclared_source_folds,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_structural_subgoal_audit import LIBRARY, teacher_traces

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_structural_subgoal_component_novelty_v1.json"
T4_CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
OUTPUT_SCHEMA = "t4_structural_subgoal_component_novelty_result_v1"
FAMILIES = (
    "whole_patch",
    *CORE_COMPONENT_FAMILIES,
    *GRANULAR_COMPONENT_FAMILIES,
    "target_radius_1_fragments",
    "target_radius_2_fragments",
    "target_graphlets_up_to_3",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed(path: Path) -> dict:
    raw = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    envelope = json.loads(raw)
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256", envelope.get("contract_sha256"))
    if not isinstance(payload, dict) or claimed != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    return payload


def publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite component-novelty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = (
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _contract() -> dict:
    envelope = json.loads(CONTRACT.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("contract_sha256") != identity(
        payload
    ):
        raise ValueError("component-novelty contract is not self-hashed")
    if any(
        payload["oracle"][field] != 0
        for field in (
            "calls_authorized",
            "docking_calls_authorized",
            "modal_launches_authorized",
        )
    ):
        raise ValueError("component-novelty contract authorizes external evaluation")
    for name, path in (
        ("teacher_library", LIBRARY),
        ("source_registry", SEEDS),
        ("t4_benchmark_contract", T4_CONTRACT),
        (
            "structural_subgoal_policy_contract",
            ROOT / "configs/t4_structural_subgoal_policy_v1.json",
        ),
        (
            "sealed_realizer_summary",
            ROOT / "diagnostics/t4_structural_subgoal/attempt_2/summary.json",
        ),
    ):
        if sha256(path) != payload["inputs"][name]["sha256"]:
            raise ValueError(f"component-novelty input hash changed: {path}")
    for row in payload["inputs"]["failed_policy_fold_reports"]:
        path = ROOT / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"failed-policy input hash changed: {path}")
    return payload


def _dependency_for_component(
    actions: tuple[dict, ...], regions: dict, component: dict
) -> AttributedComponent:
    return dependency_motif(
        actions,
        component,
        created_dependency_edges=tuple(regions["created_dependency_edges"]),
        cycle_dependency_edges=tuple(regions["cycle_dependency_edges"]),
    )


def _records_from_trace(
    *,
    source_group: str,
    route_id: str,
    states: tuple[dict, ...],
    actions: tuple[dict, ...],
    regions_override: dict | None = None,
    metadata: dict | None = None,
) -> tuple[list[dict], dict]:
    source, target = decode_state(states[0]), decode_state(states[-1])
    if regions_override is None:
        goal, bindings, regions = extract_structural_goal(states, actions)
    else:
        regions = regions_override
        structure = trace_structure(states, actions)
        extracted = [
            _extract_component(
                source,
                target,
                structure=structure,
                primitive_indices=tuple(map(int, component["primitive_indices"])),
            )
            for component in regions["components"]
        ]
        goal = StructuralGoal(tuple(row[0] for row in extracted))
        bindings = tuple(row[1] for row in extracted)
    if len(goal.subgoals) != len(regions["components"]):
        raise RuntimeError("structural subgoals and dependency components disagree")
    minimized, minimized_bindings = [], []
    for subgoal, binding in zip(goal.subgoals, bindings, strict=True):
        template, retained = minimize_subgoal(subgoal)
        minimized.append(template)
        minimized_bindings.append(tuple(binding[index] for index in retained))
    materialized = StructuralGoal(
        tuple(
            materialize_template(template, source, binding)
            for template, binding in zip(minimized, minimized_bindings, strict=True)
        )
    )
    reconstructed, _ = instantiate_goal(source, materialized, tuple(minimized_bindings))
    exact = canonical_state_key(reconstructed) == canonical_state_key(target)
    if not exact:
        raise RuntimeError("minimal structural components changed the exact endpoint")
    rows = []
    for index, (template, component) in enumerate(
        zip(minimized, regions["components"], strict=True)
    ):
        dependency = _dependency_for_component(actions, regions, component)
        rows.append(
            {
                "record_id": identity(
                    {
                        "schema_version": "structural_subgoal_component_record_v1",
                        "source_group": source_group,
                        "route_id": route_id,
                        "subgoal_index": index,
                        "template": template.payload(),
                    }
                ),
                "source_group": source_group,
                "route_id": route_id,
                "subgoal_index": index,
                "template_id": template.template_id,
                "template": template,
                "components": component_families(template, dependency),
                "metadata": dict(metadata or {}),
            }
        )
    return rows, {
        "subgoals": len(rows),
        "primitive_count": len(actions),
        "exact_target_reconstruction": exact,
        "runtime_length_supported": len(actions) <= 32,
    }


def _t4_records() -> tuple[list[dict], dict[str, dict]]:
    traces = teacher_traces()
    metadata = source_group_map(unseal(T4_CONTRACT), json.loads(SEEDS.read_text()))
    rows = []
    routes = []
    for teacher in traces:
        trace = teacher["trace"]
        route_rows, summary = _records_from_trace(
            source_group=teacher["source_group"],
            route_id=teacher["program_id"],
            states=tuple(trace["states"]),
            actions=tuple(trace["actions"]),
            metadata={"origin": "t4_complete_route"},
        )
        rows.extend(route_rows)
        routes.append(summary)
    if len(traces) != 77 or len(rows) != 147 or len(metadata) != 15:
        raise RuntimeError("T4 structural-subgoal census changed")
    return rows, metadata


def _class_counts(
    components: list[AttributedComponent],
) -> list[tuple[AttributedComponent, int]]:
    buckets: dict[tuple, list[list]] = defaultdict(list)
    for component in components:
        classes = buckets[component.bucket_key]
        for row in classes:
            vocabulary = ExactComponentVocabulary((row[0],))
            if vocabulary.contains(component):
                row[1] += 1
                break
        else:
            classes.append([component, 1])
    return [
        (row[0], row[1]) for key in sorted(buckets, key=str) for row in buckets[key]
    ]


def _family_metric(
    train_records: list[dict], held_records: list[dict], family: str
) -> tuple[dict, dict[str, bool]]:
    train = [
        component for row in train_records for component in row["components"][family]
    ]
    held = [
        component for row in held_records for component in row["components"][family]
    ]
    vocabulary = ExactComponentVocabulary(train)
    held_classes = _class_counts(held)
    held_instance_familiar = sum(vocabulary.contains(component) for component in held)
    held_unique_familiar = sum(
        vocabulary.contains(component) for component, _ in held_classes
    )
    held_vocabulary = ExactComponentVocabulary(held)
    train_transfer = sum(
        held_vocabulary.contains(component) for component in vocabulary.representatives
    )
    record_familiar = {}
    emitted_records = 0
    all_familiar = any_familiar = familiar_tokens = tokens = 0
    per_source = defaultdict(lambda: Counter(records=0))
    for row in held_records:
        components = row["components"][family]
        flags = [vocabulary.contains(component) for component in components]
        if not flags:
            record_familiar[row["record_id"]] = False
            per_source[row["source_group"]]["abstained_records"] += 1
            continue
        emitted_records += 1
        record_familiar[row["record_id"]] = all(flags)
        all_familiar += int(all(flags))
        any_familiar += int(any(flags))
        familiar_tokens += sum(flags)
        tokens += len(flags)
        source = per_source[row["source_group"]]
        source["records"] += 1
        source["all_familiar"] += int(all(flags))
        source["any_familiar"] += int(any(flags))
        source["familiar_tokens"] += sum(flags)
        source["tokens"] += len(flags)
    source_coverages = [
        values["familiar_tokens"] / values["tokens"]
        for values in per_source.values()
        if values["tokens"]
    ]
    train_classes = _class_counts(train)
    dominant = max((count for _, count in train_classes), default=0)
    metric = {
        "train_instances": len(train),
        "train_unique": vocabulary.size,
        "held_instances": len(held),
        "held_familiar_instances": held_instance_familiar,
        "held_novel_instances": len(held) - held_instance_familiar,
        "held_instance_coverage": held_instance_familiar / len(held) if held else None,
        "held_instance_novelty": (
            1 - held_instance_familiar / len(held) if held else None
        ),
        "held_unique": len(held_classes),
        "held_familiar_unique": held_unique_familiar,
        "held_unique_coverage": (
            held_unique_familiar / len(held_classes) if held_classes else None
        ),
        "train_unique_seen_in_held": train_transfer,
        "train_vocabulary_transfer_precision": (
            train_transfer / vocabulary.size if vocabulary.size else None
        ),
        "held_records": len(held_records),
        "held_records_with_components": emitted_records,
        "held_record_abstentions": len(held_records) - emitted_records,
        "held_records_all_components_familiar": all_familiar,
        "held_records_any_component_familiar": any_familiar,
        "held_record_all_component_coverage": (
            all_familiar / emitted_records if emitted_records else None
        ),
        "held_record_any_component_coverage": (
            any_familiar / emitted_records if emitted_records else None
        ),
        "source_balanced_component_coverage": (
            float(np.mean(source_coverages)) if source_coverages else None
        ),
        "train_dominant_class_fraction": dominant / len(train) if train else None,
        "degenerate_dominant_above_0_9": bool(train and dominant / len(train) > 0.9),
    }
    return metric, record_familiar


def _fold_metrics(train: list[dict], held: list[dict]) -> dict:
    family_metrics = {}
    familiarity = {}
    for family in FAMILIES:
        metric, flags = _family_metric(train, held, family)
        family_metrics[family] = metric
        familiarity[family] = flags
    strict_classes = Counter()
    granular_classes = Counter()
    source_strict_classes = defaultdict(Counter)
    source_granular_classes = defaultdict(Counter)
    train_template_ids = {row["template_id"] for row in train}
    serialized_overlap = 0
    for row in held:
        whole = familiarity["whole_patch"][row["record_id"]]
        all_strict = all(
            familiarity[family][row["record_id"]] for family in CORE_COMPONENT_FAMILIES
        )
        all_granular = all(
            familiarity[family][row["record_id"]]
            for family in GRANULAR_COMPONENT_FAMILIES
        )
        strict_name = (
            "whole_patch_familiar"
            if whole
            else (
                "novel_whole_patch_all_components_familiar"
                if all_strict
                else "novel_whole_patch_component_ood"
            )
        )
        granular_name = (
            "whole_patch_familiar"
            if whole
            else (
                "novel_whole_patch_all_components_familiar"
                if all_granular
                else "novel_whole_patch_component_ood"
            )
        )
        strict_classes[strict_name] += 1
        granular_classes[granular_name] += 1
        source_strict_classes[row["source_group"]][strict_name] += 1
        source_granular_classes[row["source_group"]][granular_name] += 1
        serialized_overlap += int(row["template_id"] in train_template_ids)
    denominator = len(held)
    per_source = {}
    for source in sorted(source_strict_classes):
        strict = source_strict_classes[source]
        granular = source_granular_classes[source]
        strict_total = sum(strict.values())
        granular_total = sum(granular.values())
        if strict_total != granular_total:
            raise RuntimeError("strict and granular diagnosis denominators disagree")
        per_source[source] = {
            "subgoals": strict_total,
            "strict_bundle": _diagnosis_rates(strict, strict_total),
            "granular": _diagnosis_rates(granular, granular_total),
        }
    return {
        "families": family_metrics,
        "combination_diagnosis": {
            "held_subgoals": denominator,
            "serialized_template_identifier_overlap": serialized_overlap,
            "strict_bundle": _diagnosis_rates(strict_classes, denominator),
            "granular": _diagnosis_rates(granular_classes, denominator),
        },
        "per_source_combination_diagnosis": per_source,
    }


def _diagnosis_rates(counts: Counter, denominator: int) -> dict:
    if denominator <= 0:
        raise ValueError("combination diagnosis requires held subgoals")
    result = dict(sorted(counts.items()))
    result.update(
        {
            "whole_patch_exact_isomorphic_coverage": counts["whole_patch_familiar"]
            / denominator,
            "novel_combination_fraction": counts[
                "novel_whole_patch_all_components_familiar"
            ]
            / denominator,
            "component_ood_fraction": counts["novel_whole_patch_component_ood"]
            / denominator,
        }
    )
    return result


def _remaining_full146_records(metadata: dict[str, dict]) -> tuple[list[dict], dict]:
    library = json.loads(LIBRARY.read_text())
    remaining = [
        row
        for row in library
        if [block["label"] for block in row["program"]["blocks"]]
        != ["compiled_complete_transformation"]
    ]
    if len(remaining) != 69:
        raise RuntimeError(
            f"expected 69 non-complete-route programs, found {len(remaining)}"
        )
    contract = unseal(T4_CONTRACT)
    source_graphs = {
        source_group: decode_state(contract["cells"][row["cell"]]["source_state"])
        for source_group, row in metadata.items()
    }
    records = []
    failures = Counter()
    applications = executable = converted = exact_context = 0
    program_ids = set()
    for library_row in remaining:
        program = EditProgram.from_payload(library_row["program"])
        program_ids.add(program.program_id)
        for source_group in library_row["source_groups"]:
            applications += 1
            source = source_graphs[source_group]
            census = program_bindings(
                program,
                source,
                max_bindings=64,
                max_visits=4096,
                contextual=True,
            )
            assignments = [
                assignment
                for assignment, distance in zip(
                    census.assignments, census.context_distances, strict=True
                )
                if distance == 0
            ]
            exact_context += int(bool(assignments))
            if not assignments:
                failures["no_exact_origin_context_binding"] += 1
                continue
            receipt = None
            for assignment in assignments:
                try:
                    _, receipt = execute_bound_program(
                        source,
                        program,
                        assignment,
                        max_primitives=32,
                        max_blocks=8,
                    )
                except (ProgramExecutionError, ValueError):
                    continue
                break
            if receipt is None:
                failures["no_exact_executable_origin_binding"] += 1
                continue
            executable += 1
            try:
                rows, _ = _records_from_trace(
                    source_group=source_group,
                    route_id=program.program_id,
                    states=tuple(receipt["states"]),
                    actions=tuple(receipt["actions"]),
                    metadata={"origin": "full146_non_complete_route"},
                )
            except (RuntimeError, ValueError) as error:
                failures[f"structural_conversion:{type(error).__name__}"] += 1
                continue
            converted += 1
            records.extend(rows)
    return records, {
        "programs": len(program_ids),
        "origin_applications": applications,
        "exact_context_binding_applications": exact_context,
        "exact_executable_applications": executable,
        "structurally_converted_applications": converted,
        "structural_conversion_coverage": (
            converted / applications if applications else 0
        ),
        "structural_conversion_precision": converted / executable if executable else 0,
        "subgoals": len(records),
        "failure_counts": dict(sorted(failures.items())),
    }


def _pmo_records(path: Path) -> tuple[list[dict], dict]:
    corpus = load_sealed(path)
    if corpus.get("schema_version") != "pmo_dependency_region_training_corpus_v2":
        raise ValueError("PMO dependency-region corpus schema changed")
    records = []
    failures = Counter()
    converted = exact = runtime_converted = 0
    for route in corpus["routes"]:
        try:
            rows, summary = _records_from_trace(
                source_group=f"pmo:{route['trace_identity']}",
                route_id=route["trace_identity"],
                states=tuple(route["states"]),
                actions=tuple(route["actions"]),
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
        converted += 1
        exact += int(summary["exact_target_reconstruction"])
        runtime_converted += int(
            route["dependency_region_program"]["runtime_length_supported"]
        )
        records.extend(rows)
    denominator = len(corpus["routes"])
    return records, {
        "routes": denominator,
        "runtime_length_routes": sum(
            row["dependency_region_program"]["runtime_length_supported"]
            for row in corpus["routes"]
        ),
        "converted_routes": converted,
        "converted_runtime_length_routes": runtime_converted,
        "exact_target_reconstructions": exact,
        "structural_conversion_coverage": converted / denominator if denominator else 0,
        "structural_conversion_precision": exact / converted if converted else 0,
        "subgoals": len(records),
        "failure_counts": dict(sorted(failures.items())),
    }


def _augmentation_metrics(
    base_train: list[dict],
    held: list[dict],
    remaining: list[dict],
    pmo: list[dict],
) -> dict:
    scenarios = {
        "t4_train_only": base_train,
        "remaining69_train_origins_only": remaining,
        "pmo_only": pmo,
        "t4_plus_remaining69": [*base_train, *remaining],
        "t4_plus_pmo": [*base_train, *pmo],
        "t4_plus_remaining69_plus_pmo": [*base_train, *remaining, *pmo],
    }
    result = {}
    for name, vocabulary_rows in scenarios.items():
        result[name] = {}
        for family in FAMILIES:
            metric, _ = _family_metric(vocabulary_rows, held, family)
            result[name][family] = {
                key: metric[key]
                for key in (
                    "train_instances",
                    "train_unique",
                    "held_instances",
                    "held_instance_coverage",
                    "held_unique_coverage",
                    "train_vocabulary_transfer_precision",
                    "held_record_all_component_coverage",
                    "source_balanced_component_coverage",
                )
            }
    return result


def _aggregate(folds: list[dict]) -> dict:
    families = {}
    for family in FAMILIES:
        metrics = [row["metrics"]["families"][family] for row in folds]
        held = sum(row["held_instances"] for row in metrics)
        familiar = sum(row["held_familiar_instances"] for row in metrics)
        unique = sum(row["held_unique"] for row in metrics)
        familiar_unique = sum(row["held_familiar_unique"] for row in metrics)
        families[family] = {
            "held_instances": held,
            "held_familiar_instances": familiar,
            "instance_weighted_coverage": familiar / held if held else None,
            "instance_weighted_novelty": 1 - familiar / held if held else None,
            "held_unique_within_fold_sum": unique,
            "held_familiar_unique_within_fold_sum": familiar_unique,
            "unique_within_fold_coverage": familiar_unique / unique if unique else None,
            "macro_fold_coverage": float(
                np.mean(
                    [
                        row["held_instance_coverage"]
                        for row in metrics
                        if row["held_instance_coverage"] is not None
                    ]
                )
            ),
            "source_balanced_coverage": float(
                np.mean(
                    [
                        row["source_balanced_component_coverage"]
                        for row in metrics
                        if row["source_balanced_component_coverage"] is not None
                    ]
                )
            ),
            "degenerate_in_any_fold": any(
                row["degenerate_dominant_above_0_9"] for row in metrics
            ),
        }
    combinations = {
        "strict_bundle": Counter(),
        "granular": Counter(),
    }
    held_subgoals = serialized_overlap = 0
    for fold in folds:
        row = fold["metrics"]["combination_diagnosis"]
        held_subgoals += row["held_subgoals"]
        serialized_overlap += row["serialized_template_identifier_overlap"]
        for resolution, counts in combinations.items():
            for key in (
                "whole_patch_familiar",
                "novel_whole_patch_all_components_familiar",
                "novel_whole_patch_component_ood",
            ):
                counts[key] += row[resolution].get(key, 0)
    return {
        "families": families,
        "combination_diagnosis": {
            "held_subgoals": held_subgoals,
            "serialized_template_identifier_overlap": serialized_overlap,
            **{
                resolution: _diagnosis_rates(counts, held_subgoals)
                for resolution, counts in combinations.items()
            },
        },
    }


def run(output: Path, *, pmo_corpus: Path, pmo_result: Path) -> dict:
    contract = _contract()
    for key, path in (
        ("optional_pmo_dependency_region_corpus", pmo_corpus),
        ("optional_pmo_dependency_region_result", pmo_result),
    ):
        if sha256(path) != contract["inputs"][key]["sha256"]:
            raise ValueError(f"PMO auxiliary input hash changed: {path}")
        load_sealed(path)
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).splitlines()
    if status:
        raise ValueError(f"component-novelty audit requires clean source: {status[:5]}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    t4_records, metadata = _t4_records()
    remaining_records, remaining_summary = _remaining_full146_records(metadata)
    pmo_records, pmo_summary = _pmo_records(pmo_corpus)
    splits = predeclared_source_folds(metadata)
    folds = []
    for split in splits:
        train_sources, held_sources = set(split["train_sources"]), set(
            split["test_sources"]
        )
        if train_sources & held_sources:
            raise RuntimeError("whole-source split leakage")
        train = [row for row in t4_records if row["source_group"] in train_sources]
        held = [row for row in t4_records if row["source_group"] in held_sources]
        remaining_train = [
            row for row in remaining_records if row["source_group"] in train_sources
        ]
        folds.append(
            {
                "fold": split["fold"],
                "train_sources": sorted(train_sources),
                "held_sources": sorted(held_sources),
                "train_routes": len({row["route_id"] for row in train}),
                "held_routes": len({row["route_id"] for row in held}),
                "train_subgoals": len(train),
                "held_subgoals": len(held),
                "metrics": _fold_metrics(train, held),
                "auxiliary_component_coverage": _augmentation_metrics(
                    train, held, remaining_train, pmo_records
                ),
            }
        )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "contract": {
            "path": str(CONTRACT.relative_to(ROOT)),
            "sha256": sha256(CONTRACT),
            "payload_sha256": identity(contract),
        },
        "evidence": "computed_zero_oracle_grouped_source_component_support_audit",
        "implementation": {
            "revision": revision,
            "working_tree_dirty": False,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "rdkit": rdBase.rdkitVersion,
            "core_path": "src/compose_v4/control/structural_subgoal_components.py",
            "core_sha256": sha256(
                ROOT / "src/compose_v4/control/structural_subgoal_components.py"
            ),
            "runner_path": "tools/t4_structural_subgoal_component_novelty.py",
            "runner_sha256": sha256(Path(__file__)),
        },
        "inputs": {
            "t4_complete_route_library": {
                "path": str(LIBRARY.relative_to(ROOT)),
                "sha256": sha256(LIBRARY),
            },
            "pmo_dependency_region_corpus": {
                "supplied_path": str(pmo_corpus),
                "logical_path": contract["inputs"][
                    "optional_pmo_dependency_region_corpus"
                ]["logical_path"],
                "sha256": sha256(pmo_corpus),
            },
            "pmo_dependency_region_result": {
                "supplied_path": str(pmo_result),
                "logical_path": contract["inputs"][
                    "optional_pmo_dependency_region_result"
                ]["logical_path"],
                "sha256": sha256(pmo_result),
            },
        },
        "t4_census": {
            "routes": len({row["route_id"] for row in t4_records}),
            "source_groups": len({row["source_group"] for row in t4_records}),
            "subgoals": len(t4_records),
            "folds": len(folds),
            "each_source_held_once": sorted(
                source for fold in folds for source in fold["held_sources"]
            )
            == sorted(metadata),
        },
        "component_definitions": contract["component_families"],
        "exact_equivalence": contract["exact_equivalence"],
        "folds": folds,
        "aggregate": _aggregate(folds),
        "auxiliary_conversion": {
            "full146_non_complete_routes": remaining_summary,
            "pmo_dependency_region_corpus": pmo_summary,
        },
        "decision": "component_support_measured_generator_architecture_not_selected",
        "costs": {
            "new_oracle_calls": 0,
            "new_docking_calls": 0,
            "modal_launches": 0,
            "model_fits": 0,
            "workers": 1,
            "device": "cpu",
        },
        "limitations": [
            "Component familiarity is exact attributed isomorphism under the frozen decomposition, not autonomous generation.",
            "Individual component overlap does not establish that a joint conditional generator can synthesize a valid unseen patch.",
            "The PMO artifact is answer-known training-only supervision supplied from a separate workspace path and is not merged or fitted here.",
            "The dependency motif uses teacher execution dependencies for a training-data support audit; the sealed structural realizer itself does not consume that teacher trace.",
            "No molecular utility, docking improvement or IVG comparison was measured.",
        ],
    }
    publish(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmo-corpus", type=Path, required=True)
    parser.add_argument("--pmo-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    began = perf_counter()
    result = run(args.output, pmo_corpus=args.pmo_corpus, pmo_result=args.pmo_result)
    print(
        json.dumps(
            {
                "aggregate": result["aggregate"],
                "auxiliary_conversion": result["auxiliary_conversion"],
                "costs": result["costs"],
                "operational_wall_seconds": perf_counter() - began,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
