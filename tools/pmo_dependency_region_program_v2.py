"""Build the sealed macro-free PMO dependency-region training corpus."""

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
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_dependency_region_program import (
    SCHEMA,
    TRAINING_SCHEMA,
    DependencyRegionConfig,
    dependency_region_program,
    dependency_region_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("configs/pmo_dependency_region_program_v2.json")
DEFAULT_OUTPUT = Path("diagnostics/pmo_dependency_region_program_v2/attempt_1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_envelope(path: Path, expected_payload_hash: str) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or envelope.get("payload_sha256") != expected_payload_hash
        or identity(payload) != expected_payload_hash
    ):
        raise ValueError(f"invalid self-hashed input envelope: {path}")
    return payload


def _load_gzip_envelope(path: Path, expected_payload_hash: str) -> dict:
    with gzip.open(path, "rt") as handle:
        envelope = json.load(handle)
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or envelope.get("payload_sha256") != expected_payload_hash
        or identity(payload) != expected_payload_hash
    ):
        raise ValueError(f"invalid self-hashed gzip input envelope: {path}")
    return payload


def _load_contract(root: Path) -> dict:
    path = root / CONTRACT
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "pmo_dependency_region_program_contract_v2"
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError(f"invalid PMO dependency-region contract: {path}")
    if payload.get("outputs", {}).get("runtime_checkpoint") is not None:
        raise ValueError("dependency-region v2 may not emit a runtime checkpoint")
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


def _validate_inputs(root: Path, contract: dict) -> tuple[dict, dict]:
    source = contract["source_v1"]
    result_path = root / source["result_path"]
    corpus_path = root / source["training_corpus_path"]
    if sha256_file(result_path) != source["result_sha256"]:
        raise ValueError("sealed PMO segmentation v1 result changed")
    if sha256_file(corpus_path) != source["training_corpus_sha256"]:
        raise ValueError("sealed PMO segmentation v1 training corpus changed")
    result = _load_json_envelope(result_path, source["result_payload_sha256"])
    corpus = _load_gzip_envelope(corpus_path, source["training_corpus_payload_sha256"])
    if result.get("decision") != (
        "exact_corpus_and_panels_ready_complete_segment_decoder_gate_failed"
    ):
        raise ValueError("v1 source is not the preserved failed representation gate")
    if (
        result.get("outputs", {}).get(corpus_path.name)
        != source["training_corpus_sha256"]
    ):
        raise ValueError("v1 result does not bind the exact training corpus")
    if corpus.get("runtime_checkpoint") is not False:
        raise ValueError("v1 source corpus role changed")
    if (
        corpus.get("split", {}).get("split_identity")
        != source["expected_split_identity"]
    ):
        raise ValueError("v1 frozen split identity changed")
    census = result.get("census", {})
    expected_census = {
        key.removeprefix("expected_"): value
        for key, value in source.items()
        if key
        in {
            "expected_route_instances",
            "expected_unique_traces",
            "expected_lineages",
            "expected_primitive_transitions_with_multiplicity",
            "expected_deduplicated_transitions",
        }
    }
    if census != expected_census:
        raise ValueError(f"v1 route census changed: {census}")
    panel = result.get("negative_panels", {})
    if (
        panel.get("attempts") != source["expected_negative_panel_attempts"]
        or panel.get("unique_complete")
        != source["expected_negative_panel_unique_complete"]
    ):
        raise ValueError("v1 negative panel census changed")
    return result, corpus


def _represent_routes(
    routes: list[dict], configuration: DependencyRegionConfig
) -> list[dict]:
    represented = []
    for route in routes:
        program = dependency_region_program(
            tuple(route["states"]), tuple(route["actions"]), configuration
        )
        represented.append(
            {
                key: route[key]
                for key in (
                    "trace_identity",
                    "lineage_identity",
                    "task_family",
                    "test_fold",
                    "members",
                    "source_state",
                    "states",
                    "actions",
                    "terminal_endpoint",
                )
            }
            | {"dependency_region_program": program}
        )
    return represented


def run(output: Path = ROOT / DEFAULT_OUTPUT, root: Path = ROOT) -> dict:
    if output.exists():
        raise ValueError(
            f"refusing to overwrite PMO dependency-region output: {output}"
        )
    began = perf_counter()
    contract = _load_contract(root)
    source_result, source_corpus = _validate_inputs(root, contract)
    source = contract["source_v1"]
    configuration = DependencyRegionConfig(
        runtime_maximum_primitives=contract["representation"][
            "runtime_maximum_primitives"
        ],
        runtime_maximum_components=contract["representation"][
            "runtime_maximum_components"
        ],
    )
    routes = _represent_routes(source_corpus["routes"], configuration)
    overall = dependency_region_summary(routes)
    held_out_folds = {
        str(fold["fold"]): dependency_region_summary(
            [route for route in routes if route["test_fold"] == fold["fold"]]
        )
        for fold in source_corpus["split"]["folds"]
    }
    if len(routes) != source["expected_unique_traces"]:
        raise ValueError("v2 representation lost a unique trace")
    if overall["primitive_transitions"] != source["expected_deduplicated_transitions"]:
        raise ValueError("v2 representation lost a primitive transition")
    if overall["runtime_length_routes"] != source["expected_runtime_length_routes"]:
        raise ValueError("v2 runtime-length census changed")

    panel_reference = {
        "labels_reused_without_regeneration": True,
        "representation_changes_endpoint_or_transformation_label": False,
        "source_result_path": source["result_path"],
        "source_result_sha256": source["result_sha256"],
        "source_result_payload_sha256": source["result_payload_sha256"],
        "source_training_corpus_path": source["training_corpus_path"],
        "source_training_corpus_sha256": source["training_corpus_sha256"],
        "source_training_corpus_payload_sha256": source[
            "training_corpus_payload_sha256"
        ],
        "attempts": source_result["negative_panels"]["attempts"],
        "complete_attempts": source_result["negative_panels"]["complete_attempts"],
        "unique_complete": source_result["negative_panels"]["unique_complete"],
        "unique_relation_counts": source_result["negative_panels"][
            "unique_relation_counts"
        ],
        "new_proposals": 0,
    }
    training_corpus = {
        "schema_version": TRAINING_SCHEMA,
        "artifact_role": (
            "training_only_exact_dependency_region_programs_no_runtime_checkpoint"
        ),
        "runtime_checkpoint": False,
        "source_v1": {
            "result_path": source["result_path"],
            "result_sha256": source["result_sha256"],
            "result_payload_sha256": source["result_payload_sha256"],
            "training_corpus_path": source["training_corpus_path"],
            "training_corpus_sha256": source["training_corpus_sha256"],
            "training_corpus_payload_sha256": source["training_corpus_payload_sha256"],
        },
        "split": source_corpus["split"],
        "representation_configuration": contract["representation"],
        "routes": routes,
        "negative_panel_reference": panel_reference,
        "task_scores_loaded": False,
        "new_oracle_calls": 0,
        "model_fit": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        corpus_path = temporary / contract["outputs"]["training_corpus"]
        _publish(corpus_path, training_corpus, compressed=True)
        raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss = int(raw_peak if platform.system() == "Darwin" else raw_peak * 1024)
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
        gates = {
            "sealed_v1_failed_gate_preserved": True,
            "frozen_split_identity_preserved": source_corpus["split"]["split_identity"]
            == source["expected_split_identity"],
            "exact_replay_coverage_one": overall["exact_replay_coverage"] == 1.0,
            "exact_replay_precision_one": overall["exact_replay_precision"] == 1.0,
            "all_runtime_length_routes_represented": overall[
                "complete_representation_routes"
            ]
            == source["expected_runtime_length_routes"],
            "nonzero_complete_support_every_fold": all(
                row["complete_representation_routes"] > 0
                for row in held_out_folds.values()
            ),
            "zero_cross_component_dependency_leakage": (
                overall["cross_component_created_dependency_edges"] == 0
                and overall["cross_component_cycle_dependency_edges"] == 0
            ),
            "runtime_budgets_unchanged": (
                configuration.runtime_maximum_primitives == 32
                and configuration.runtime_maximum_components == 8
            ),
            "negative_panel_reused_without_generation": panel_reference["new_proposals"]
            == 0,
            "runtime_checkpoint_absent": contract["outputs"]["runtime_checkpoint"]
            is None,
            "model_fit_absent": True,
        }
        report = {
            "schema_version": SCHEMA,
            "decision": (
                "dependency_region_representation_gate_passed_"
                "autoregressive_decoder_comparison_enabled"
                if all(gates.values())
                else "dependency_region_representation_gate_failed"
            ),
            "contract_path": str(CONTRACT),
            "contract_sha256": sha256_file(root / CONTRACT),
            "source_v1": {
                "decision": source_result["decision"],
                "result_path": source["result_path"],
                "result_sha256": source["result_sha256"],
                "result_payload_sha256": source["result_payload_sha256"],
                "training_corpus_path": source["training_corpus_path"],
                "training_corpus_sha256": source["training_corpus_sha256"],
                "training_corpus_payload_sha256": source[
                    "training_corpus_payload_sha256"
                ],
            },
            "split": {
                "split_identity": source_corpus["split"]["split_identity"],
                "folds": source_corpus["split"]["folds"],
                "frozen_before_v2_derivation": True,
            },
            "representation": {
                "configuration": contract["representation"],
                "overall": overall,
                "held_out_folds": held_out_folds,
            },
            "negative_panel_reference": panel_reference,
            "outputs": {
                contract["outputs"]["training_corpus"]: sha256_file(corpus_path)
            },
            "gates": gates,
            "next_policy_gate": {
                "authorized_training_performed": False,
                "autoregressive_decoder_comparison_enabled": all(gates.values()),
                "required_comparison": [
                    "balanced_marginal_dependency_region_decoder",
                    "graph_region_dependency_conditioned_autoregressive_decoder",
                    "same_source_contrastive_complete_candidate_ranker",
                ],
                "must_preserve_split_first_training": True,
            },
            "compute": {
                "device": "CPU",
                "workers": 1,
                "precision": "exact_integer_graph_replay_and_float64_metrics",
                "total_seconds": perf_counter() - began,
                "peak_rss_bytes": peak_rss,
                "training_corpus_bytes": corpus_path.stat().st_size,
            },
            "implementation": {
                "revision": revision,
                "working_tree_dirty": dirty,
                "runner_sha256": sha256_file(Path(__file__)),
                "core_sha256": sha256_file(
                    root / "src/compose_v4/experiments/pmo_dependency_region_program.py"
                ),
            },
            "costs": {
                "new_generic_proposals": 0,
                "new_oracle_calls": 0,
                "new_docking_calls": 0,
                "modal_launches": 0,
            },
            "limitations": [
                "All exact routes and reused panel labels are answer-known PMO development evidence.",
                "Components are connectivity groups, not semantic or mechanistic macros.",
                "Representation coverage does not establish learned proposal recall or PMO improvement.",
                "Routes over 32 primitives remain explicit abstentions.",
                "No controller or runtime checkpoint was trained or selected.",
            ],
        }
        if not all(gates.values()):
            raise RuntimeError(f"PMO dependency-region v2 gate failed: {gates}")
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
                "representation": result["representation"],
                "negative_panel_reference": result["negative_panel_reference"],
                "gates": result["gates"],
                "next_policy_gate": result["next_policy_gate"],
                "compute": result["compute"],
            },
            indent=2,
        )
    )
