"""Build the split-first exact PMO segmentation and generic negative-panel corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import resource
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v1 import (
    synthesize_dynamic_program_v1,
)
from compose_v4.experiments.pmo_exact_program_segmentation import (
    SCHEMA,
    TRAINING_SCHEMA,
    SegmentationConfig,
    segment_exact_trace,
    segmentation_summary,
)
from compose_v4.experiments.route_proposal_quality import SCHEMA as PROPOSAL_SCHEMA
from compose_v4.experiments.route_proposal_quality import evaluate as evaluate_proposals
from compose_v4.experiments.route_proposal_quality import transformation_equivalent
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.pmo_route_distillation import collect_routes, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("configs/pmo_exact_program_segmentation_v1.json")
DEFAULT_OUTPUT = Path("diagnostics/pmo_exact_program_segmentation/attempt_2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(root: Path) -> dict:
    path = root / CONTRACT
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "pmo_exact_program_segmentation_contract_v1"
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError(f"invalid exact PMO segmentation contract: {path}")
    if payload.get("outputs", {}).get("runtime_checkpoint") is not None:
        raise ValueError("this revision may not publish a runtime checkpoint")
    return payload


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    ready = _json_ready(payload)
    envelope = {"payload": ready, "payload_sha256": identity(ready)}
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    if compressed:
        path.write_bytes(gzip.compress(raw.encode(), mtime=0))
    else:
        path.write_text(raw)


def _deduplicate_routes(routes):
    grouped = defaultdict(list)
    for route in routes:
        grouped[route.trace_identity].append(route)
    result = []
    for trace_identity, members in sorted(grouped.items()):
        members.sort(key=lambda route: route.instance_identity)
        if (
            len({route.task_family for route in members}) != 1
            or len({route.lineage_identity for route in members}) != 1
        ):
            raise ValueError("one duplicate trace crosses a frozen split group")
        result.append((trace_identity, members[0], tuple(members)))
    return result


def _freeze_split(unique_routes, fold_rows: list[dict]) -> dict:
    family_fold = {}
    for row in fold_rows:
        for family in row["held_out_task_families"]:
            if family in family_fold:
                raise ValueError("a task family appears in multiple frozen folds")
            family_fold[family] = int(row["fold"])
    lineages = defaultdict(set)
    for _, route, _ in unique_routes:
        lineages[route.lineage_identity].add(route.task_family)
    if any(len(families) != 1 for families in lineages.values()):
        raise ValueError("one source lineage crosses task-family folds")
    observed = set().union(*lineages.values())
    if observed != set(family_fold):
        raise ValueError("frozen folds do not partition the admitted task families")
    lineage_fold = {
        lineage: family_fold[next(iter(families))]
        for lineage, families in sorted(lineages.items())
    }
    trace_fold = {
        trace_identity: lineage_fold[route.lineage_identity]
        for trace_identity, route, _ in unique_routes
    }
    return {
        "rule": "hold_out_whole_task_families_and_all_shared_base_lineages",
        "frozen_before_segmentation_or_panel_generation": True,
        "folds": fold_rows,
        "lineage_test_fold": lineage_fold,
        "trace_test_fold": trace_fold,
        "split_identity": identity(
            {
                "family_fold": family_fold,
                "lineage_fold": lineage_fold,
                "trace_fold": trace_fold,
            }
        ),
    }


def _segment_routes(unique_routes, split: dict, config: SegmentationConfig):
    result = []
    for trace_identity, route, members in unique_routes:
        segmentation = segment_exact_trace(route.states, route.actions, config)
        result.append(
            {
                "trace_identity": trace_identity,
                "lineage_identity": route.lineage_identity,
                "task_family": route.task_family,
                "test_fold": split["trace_test_fold"][trace_identity],
                "members": [
                    {
                        "instance_identity": member.instance_identity,
                        "collection": member.collection,
                        "task": member.task,
                        "member_id": member.member_id,
                        "source_path": member.source_path,
                        "evidence_role": member.evidence_role,
                    }
                    for member in members
                ],
                "source_state": route.states[0],
                "states": list(route.states),
                "actions": list(route.actions),
                "terminal_endpoint": route.endpoint,
                "segmentation": segmentation,
            }
        )
    return result


def _source_groups(segmented_routes: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for route in segmented_routes:
        grouped[route["lineage_identity"]].append(route)
    for lineage, routes in grouped.items():
        sources = {identity(route["source_state"]) for route in routes}
        folds = {route["test_fold"] for route in routes}
        families = {route["task_family"] for route in routes}
        if len(sources) != 1 or len(folds) != 1 or len(families) != 1:
            raise ValueError(f"source lineage is internally inconsistent: {lineage}")
    return dict(sorted(grouped.items()))


def _panel_seed(campaign_seed: int, lineage: str) -> int:
    return int(identity({"campaign_seed": campaign_seed, "lineage": lineage})[:16], 16)


def _generate_panel(lineage: str, routes: list[dict], configuration: dict) -> dict:
    source_state = routes[0]["source_state"]
    source = decode_state(source_state)
    source_key = canonical_state_key(source)
    teacher_graphs = [decode_state(route["states"][-1]) for route in routes]
    teacher_keys = {canonical_state_key(graph) for graph in teacher_graphs}
    rng = np.random.default_rng(_panel_seed(configuration["campaign_seed"], lineage))
    attempts = []
    seen = {}
    panel_cache = {}
    began = perf_counter()
    for index in range(configuration["attempts_per_lineage"]):
        attempt_id = f"generic-{index:04d}"
        try:
            _, program, assignment, trace, metadata = synthesize_dynamic_program_v1(
                source,
                rng,
                max_modules=configuration["max_modules"],
                max_primitives=configuration["max_primitives"],
                max_blocks=configuration["max_blocks"],
                panel_cache=panel_cache,
            )
        except ValueError as error:
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "rank": index + 1,
                    "status": "rejected",
                    "failure": str(error),
                    "endpoint_state": None,
                }
            )
            continue
        endpoint = decode_state(trace["states"][-1])
        endpoint_key = canonical_state_key(endpoint)
        if endpoint_key == source_key:
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "rank": index + 1,
                    "status": "rejected",
                    "failure": "canonical_self_event",
                    "endpoint_state": None,
                }
            )
            continue
        if endpoint_key in teacher_keys:
            relation = "exact_teacher_endpoint"
        elif any(
            transformation_equivalent(
                source,
                endpoint,
                teacher,
                radius=configuration["transformation_equivalence_radius"],
            )
            for teacher in teacher_graphs
        ):
            relation = "transformation_equivalent_teacher"
        else:
            relation = "generic_negative"
        duplicate_of = seen.setdefault(endpoint_key, attempt_id)
        attempts.append(
            {
                "attempt_id": attempt_id,
                "rank": index + 1,
                "status": "complete",
                "failure": None,
                "endpoint_state": trace["states"][-1],
                "trace": trace,
                "program": program.payload(),
                "assignment": list(assignment),
                "compiler_metadata": metadata,
                "teacher_relation": relation,
                "duplicate_of": None if duplicate_of == attempt_id else duplicate_of,
            }
        )
    return {
        "lineage_identity": lineage,
        "task_family": routes[0]["task_family"],
        "test_fold": routes[0]["test_fold"],
        "source_state": source_state,
        "teacher_trace_identities": [route["trace_identity"] for route in routes],
        "teacher_endpoint_states": [route["states"][-1] for route in routes],
        "seed": _panel_seed(configuration["campaign_seed"], lineage),
        "proposal_seconds": perf_counter() - began,
        "attempts": attempts,
    }


def _proposal_report(panels: list[dict], split: dict, configuration: dict) -> dict:
    by_fold = {}
    all_lineages = {panel["lineage_identity"] for panel in panels}
    for fold_row in split["folds"]:
        fold = int(fold_row["fold"])
        held_out = [panel for panel in panels if panel["test_fold"] == fold]
        test_sources = [panel["lineage_identity"] for panel in held_out]
        cases = []
        for panel in held_out:
            cases.append(
                {
                    "source_id": panel["lineage_identity"],
                    "source_state": panel["source_state"],
                    "teachers": [
                        {"teacher_id": trace, "endpoint_state": endpoint}
                        for trace, endpoint in zip(
                            panel["teacher_trace_identities"],
                            panel["teacher_endpoint_states"],
                            strict=True,
                        )
                    ],
                    "policy_pools": [
                        {
                            "policy_id": "task_blind_generic_dynamic_v1",
                            "proposal_seconds": panel["proposal_seconds"],
                            "attempts": [
                                {
                                    "attempt_id": row["attempt_id"],
                                    "rank": row["rank"],
                                    "status": row["status"],
                                    "endpoint_state": row["endpoint_state"],
                                }
                                for row in panel["attempts"]
                            ],
                        }
                    ],
                }
            )
        payload = {
            "schema_version": PROPOSAL_SCHEMA,
            "oracle_calls": 0,
            "split": {
                "train_sources": sorted(all_lineages - set(test_sources)),
                "calibration_sources": [],
                "test_sources": sorted(test_sources),
                "evaluation_role": "test",
            },
            "cases": cases,
        }
        by_fold[str(fold)] = evaluate_proposals(
            payload,
            cutoffs=(1, 5, 10, configuration["attempts_per_lineage"]),
            radius=configuration["transformation_equivalence_radius"],
        )
    complete = [
        row
        for panel in panels
        for row in panel["attempts"]
        if row["status"] == "complete"
    ]
    unique = [row for row in complete if row["duplicate_of"] is None]
    relations = Counter(row["teacher_relation"] for row in unique)
    return {
        "folds": by_fold,
        "attempts": sum(len(panel["attempts"]) for panel in panels),
        "complete_attempts": len(complete),
        "unique_complete": len(unique),
        "execution_precision": len(complete)
        / sum(len(panel["attempts"]) for panel in panels),
        "unique_endpoint_yield": len(unique)
        / sum(len(panel["attempts"]) for panel in panels),
        "unique_relation_counts": dict(sorted(relations.items())),
        "generic_negative_unique_yield": relations["generic_negative"]
        / sum(len(panel["attempts"]) for panel in panels),
        "teacher_injected_attempts": 0,
        "task_score_labels": 0,
        "proposal_seconds": sum(panel["proposal_seconds"] for panel in panels),
    }


def run(output: Path = ROOT / DEFAULT_OUTPUT, root: Path = ROOT) -> dict:
    if output.exists():
        raise ValueError(
            f"refusing to overwrite exact PMO segmentation output: {output}"
        )
    began = perf_counter()
    contract = _load_contract(root)
    source_manifest = contract["source_manifest"]
    manifest_path = root / source_manifest["path"]
    if sha256_file(manifest_path) != source_manifest["sha256"]:
        raise ValueError("sealed PMO route inclusion manifest changed")
    route_contract = load_contract(root)
    routes, source_audit = collect_routes(route_contract, root)
    unique_routes = _deduplicate_routes(routes)
    observed = {
        "route_instances": len(routes),
        "unique_traces": len(unique_routes),
        "lineages": len({route.lineage_identity for route in routes}),
        "primitive_transitions_with_multiplicity": sum(
            len(route.actions) for route in routes
        ),
        "deduplicated_transitions": sum(
            len(route.actions) for _, route, _ in unique_routes
        ),
    }
    expected = {
        key.removeprefix("expected_"): value
        for key, value in source_manifest.items()
        if key.startswith("expected_")
    }
    if observed != expected:
        raise ValueError(
            f"exact PMO source census changed: expected {expected}, got {observed}"
        )
    split = _freeze_split(unique_routes, contract["folds"])
    segmentation_config = SegmentationConfig(**contract["segmentation"])
    segmented = _segment_routes(unique_routes, split, segmentation_config)
    groups = _source_groups(segmented)
    if len(groups) != expected["lineages"]:
        raise ValueError("one or more frozen source lineages disappeared")
    panels = [
        _generate_panel(lineage, source_routes, contract["negative_panels"])
        for lineage, source_routes in groups.items()
    ]
    overall_segmentation = segmentation_summary(segmented)
    fold_segmentation = {
        str(row["fold"]): segmentation_summary(
            [route for route in segmented if route["test_fold"] == row["fold"]]
        )
        for row in contract["folds"]
    }
    panel_report = _proposal_report(panels, split, contract["negative_panels"])
    training_corpus = {
        "schema_version": TRAINING_SCHEMA,
        "artifact_role": "training_only_contains_exact_teacher_graphs_and_programs",
        "runtime_checkpoint": False,
        "split": split,
        "segmentation_configuration": contract["segmentation"],
        "negative_panel_configuration": contract["negative_panels"],
        "routes": segmented,
        "source_conditioned_panels": [
            {key: value for key, value in panel.items() if key != "proposal_seconds"}
            for panel in panels
        ],
        "task_scores_loaded": False,
        "new_oracle_calls": 0,
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        corpus_path = temporary / contract["outputs"]["training_corpus"]
        _publish(corpus_path, training_corpus, compressed=True)
        raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss = int(raw_peak if platform.system() == "Darwin" else raw_peak * 1024)
        report = {
            "schema_version": SCHEMA,
            "decision": "exact_corpus_and_panels_ready_complete_segment_decoder_gate_failed",
            "contract_path": str(CONTRACT),
            "contract_sha256": sha256_file(root / CONTRACT),
            "source_manifest": {
                "path": source_manifest["path"],
                "sha256": source_manifest["sha256"],
                "source_checks": source_audit["source_checks"],
            },
            "split": split,
            "census": observed,
            "segmentation": {
                "overall": overall_segmentation,
                "held_out_folds": fold_segmentation,
            },
            "negative_panels": panel_report,
            "outputs": {
                contract["outputs"]["training_corpus"]: sha256_file(corpus_path)
            },
            "gates": {
                "split_frozen_before_derivation": True,
                "zero_lineage_leakage": True,
                "exact_segment_replay_precision_one": overall_segmentation[
                    "exact_replay_precision"
                ]
                == 1.0,
                "nonzero_multi_action_coverage": overall_segmentation[
                    "multi_action_transition_coverage"
                ]
                > 0,
                "nonzero_generic_negative_yield": panel_report[
                    "unique_relation_counts"
                ].get("generic_negative", 0)
                > 0,
                "task_score_labels_absent": True,
                "runtime_checkpoint_absent": True,
            },
            "next_policy_gate": {
                "authorized_training_performed": False,
                "representation_supported_routes": overall_segmentation[
                    "complete_representation_routes"
                ],
                "runtime_length_reference_routes": overall_segmentation[
                    "runtime_length_routes"
                ],
                "complete_representation_coverage": overall_segmentation[
                    "complete_representation_routes"
                ]
                / overall_segmentation["runtime_length_routes"],
                "every_held_out_fold_has_complete_representation_support": all(
                    row["complete_representation_routes"] > 0
                    for row in fold_segmentation.values()
                ),
                "required_comparison": [
                    "balanced_marginal_segment_decoder",
                    "graph_region_dependency_conditioned_segment_decoder",
                    "same_source_contrastive_complete_candidate_ranker",
                ],
                "selection_status": (
                    "blocked_until_a_preregistered_generic_segmentation_revision_has_"
                    "support_in_every_held_out_fold"
                ),
            },
            "compute": {
                "device": "CPU",
                "workers": 1,
                "precision": "exact_integer_graph_replay_and_float64_metrics",
                "random_seed": contract["negative_panels"]["campaign_seed"],
                "total_seconds": perf_counter() - began,
                "peak_rss_bytes": peak_rss,
                "training_corpus_bytes": corpus_path.stat().st_size,
            },
            "implementation": {
                "revision": revision,
                "working_tree_dirty": dirty,
                "runner_sha256": sha256_file(Path(__file__)),
                "core_sha256": sha256_file(
                    root
                    / "src/compose_v4/experiments/pmo_exact_program_segmentation.py"
                ),
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "costs": {
                "new_oracle_calls": 0,
                "new_docking_calls": 0,
                "modal_launches": 0,
            },
            "limitations": [
                "All admitted routes are answer-known PMO development supervision.",
                "Structural segments are generic groupings, not semantic or mechanistic macros.",
                "Cross-segment created-handle dependencies require an autoregressive program state.",
                "Generic negative-panel recovery is a proposal-support diagnostic, not task optimization.",
                "No controller or runtime checkpoint was trained or selected.",
            ],
        }
        if not all(report["gates"].values()):
            raise RuntimeError(f"exact PMO segmentation gate failed: {report['gates']}")
        _publish(temporary / contract["outputs"]["result"], report)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_OUTPUT)
    result = run(parser.parse_args().output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "segmentation": result["segmentation"],
                "negative_panels": {
                    key: value
                    for key, value in result["negative_panels"].items()
                    if key != "folds"
                },
                "gates": result["gates"],
                "next_policy_gate": result["next_policy_gate"],
                "compute": result["compute"],
            },
            indent=2,
        )
    )
