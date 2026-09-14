#!/usr/bin/env python3
"""Fit and audit the answer-known T4 route-distilled proposal policy."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import (
    EditProgram,
    ProgramExecutionError,
    attachment_bindings,
    execute_bound_program,
)
from compose_v4.control.option_demonstrations import (
    DemonstrationFitConfig,
    batch_option_scores,
    descriptor_menu,
    fit_demonstration_actor,
    recognize_trace,
    source_weights,
)
from compose_v4.control.option_features import structural_option_features
from compose_v4.control.option_policy import option_actor_parameter_id
from compose_v4.control.route_distilled_program_policy import (
    CHECKPOINT_SCHEMA,
    GENERIC_MODULES,
    STAGE_DESCRIPTOR_NAMES,
    RouteDistilledProgramPolicy,
    default_reference,
    normalized_module_count_probabilities,
    option_family_weights,
    stage_descriptor,
)
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_program_vocabulary_audit import source_group_map

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
LIBRARY = (
    ROOT / "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
)
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
MAX_BINDINGS = 64
MAX_VISITS = 4096
MAX_PRIMITIVES = 32
MAX_BLOCKS = 8
EXPLORATION_FLOOR = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite route-distillation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    if compressed:
        temporary.write_bytes(gzip.compress(raw.encode(), mtime=0))
    else:
        temporary.write_text(raw)
    temporary.replace(path)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _teacher_traces(contract: dict, library: list[dict], groups: dict[str, dict]):
    complete = [
        row
        for row in library
        if len(row["program"]["blocks"]) == 1
        and row["program"]["blocks"][0]["label"] == "compiled_complete_transformation"
    ]
    if len(complete) != 77:
        raise ValueError(f"expected 77 complete teachers, found {len(complete)}")
    traces, exclusions = [], []
    for row in complete:
        program = EditProgram.from_payload(row["program"])
        accepted = None
        failures = Counter()
        for group in sorted(row["source_groups"]):
            cell = groups[group]["cell"]
            source = decode_state(contract["cells"][cell]["source_state"])
            census = attachment_bindings(
                program,
                source,
                max_bindings=MAX_BINDINGS,
                max_visits=MAX_VISITS,
                contextual=True,
            )
            for assignment in census.assignments:
                try:
                    product, trace = execute_bound_program(
                        source,
                        program,
                        assignment,
                        max_primitives=MAX_PRIMITIVES,
                        max_blocks=MAX_BLOCKS,
                    )
                except ProgramExecutionError as error:
                    failures[f"ProgramExecutionError:{error.step}"] += 1
                except ValueError as error:
                    failures[type(error).__name__] += 1
                else:
                    if trace["endpoint"] != canonical_state_key(product):
                        raise RuntimeError("teacher endpoint differs from exact replay")
                    accepted = {
                        "source_group": group,
                        "program_id": program.program_id,
                        "source": source,
                        "trace": trace,
                    }
                    break
            if accepted is not None:
                break
        if accepted is None:
            exclusions.append(
                {"program_id": program.program_id, "failures": dict(failures)}
            )
        else:
            traces.append(accepted)
    return traces, exclusions


def _rows(traces):
    rows, route_summaries = [], []
    for trace_index, teacher in enumerate(traces):
        trace = teacher["trace"]
        segments = recognize_trace(trace["states"], trace["actions"])
        route_rows = []
        for segment_index, segment in enumerate(segments):
            graph = decode_state(trace["states"][segment.start])
            actions = trace["actions"][segment.start : segment.stop]
            descriptor = stage_descriptor(graph, actions)
            row = {
                "decision_id": f"teacher-{trace_index}:segment-{segment_index}",
                "source_id": teacher["source_group"],
                "option": segment.option,
                "compound": segment.compound,
                "features": molecule_features(canonical_state_key(graph)).tolist(),
                "stage_descriptor": descriptor.tolist(),
                "primitive_edits": len(actions),
                "created_handle_dependencies": descriptor[
                    STAGE_DESCRIPTOR_NAMES.index("created_dependency_fraction")
                ],
            }
            rows.append(row)
            route_rows.append(row)
        route_summaries.append(
            {
                "teacher_index": trace_index,
                "primitive_edits": len(trace["actions"]),
                "high_level_decisions": len(segments),
                "compound_decisions": sum(segment.compound for segment in segments),
                "primitive_fallback_decisions": sum(
                    not segment.compound for segment in segments
                ),
                "compression_ratio": len(segments) / len(trace["actions"]),
                "created_handle_dependency_decisions": sum(
                    row["created_handle_dependencies"] > 0 for row in route_rows
                ),
                "exact_replay": True,
            }
        )
    return rows, route_summaries


def _probabilities(actor, rows, options, reference, *, floor):
    option_features = torch.from_numpy(
        np.stack([structural_option_features(name) for name in options])
    )
    states = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    with torch.inference_mode():
        scores = batch_option_scores(actor, states, option_features).numpy()
    scores -= scores.max(axis=1, keepdims=True)
    learned = np.asarray(reference)[None, :] * np.exp(scores)
    learned /= learned.sum(axis=1, keepdims=True)
    return floor * np.asarray(reference)[None, :] + (1 - floor) * learned


def _metrics(rows, probabilities, options, weights):
    labels = np.asarray([options.index(row["option"]) for row in rows])
    selected = probabilities[np.arange(len(rows)), labels]
    order = np.argsort(-probabilities, axis=1)
    ranks = np.asarray(
        [
            int(np.flatnonzero(order[index] == label)[0]) + 1
            for index, label in enumerate(labels)
        ]
    )
    return {
        "source_balanced_nll": float(-weights @ np.log(selected)),
        "source_balanced_mean_teacher_probability": float(weights @ selected),
        "source_balanced_top1": float(weights @ (ranks <= 1)),
        "source_balanced_top10": float(weights @ (ranks <= 10)),
        "unweighted_median_rank": float(np.median(ranks)),
        "unweighted_rank_at": {
            "1": float(np.mean(ranks <= 1)),
            "10": float(np.mean(ranks <= 10)),
            "100": float(np.mean(ranks <= 100)),
        },
    }


def _binding_prototypes(rows):
    weighted = defaultdict(list)
    row_weights = source_weights([row["source_id"] for row in rows])
    for row, source_weight in zip(rows, row_weights, strict=True):
        descriptor = np.asarray(row["stage_descriptor"], dtype=float)
        for family, conditional in option_family_weights(row["option"]).items():
            weighted[family].append((descriptor, float(source_weight) * conditional))
    global_rows = [
        (np.asarray(row["stage_descriptor"], dtype=float), float(weight))
        for row, weight in zip(rows, row_weights, strict=True)
    ]

    def moments(values):
        matrix = np.stack([row for row, _ in values])
        weights = np.asarray([weight for _, weight in values], dtype=float)
        weights /= weights.sum()
        mean = weights @ matrix
        variance = weights @ np.square(matrix - mean)
        return mean, np.maximum(np.sqrt(variance), 0.05)

    fallback = moments(global_rows)
    result = []
    for family in sorted(GENERIC_MODULES):
        mean, scale = moments(weighted[family]) if weighted[family] else fallback
        result.append(
            {"family": family, "mean": mean.tolist(), "scale": scale.tolist()}
        )
    return result


def run(
    output: Path,
    *,
    code_revision: str | None = None,
    working_tree_dirty: bool | None = None,
) -> dict:
    if output.exists():
        raise ValueError(f"preserve prior route-distillation result: {output}")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("route distillation requires pinned RDKit 2024.03.5")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    contract = unseal(CONTRACT)
    library = json.loads(LIBRARY.read_text())
    seed_registry = json.loads(SEEDS.read_text())
    groups = source_group_map(contract, seed_registry)
    traces, exclusions = _teacher_traces(contract, library, groups)
    if exclusions or len(traces) != 77:
        raise RuntimeError(
            f"teacher replay gate failed: admitted={len(traces)}, exclusions={exclusions}"
        )
    rows, route_summaries = _rows(traces)
    options = descriptor_menu()
    reference = np.asarray(default_reference(), dtype=float)
    weights = source_weights([row["source_id"] for row in rows])
    counts = np.zeros(len(options), dtype=float)
    for row, weight in zip(rows, weights, strict=True):
        counts[options.index(row["option"])] += weight
    marginal = (counts + reference) / (counts.sum() + reference.sum())
    config = DemonstrationFitConfig(
        hidden=64,
        updates=600,
        batch_size=64,
        learning_rate=0.003,
        base_floor=EXPLORATION_FLOOR,
        kl_penalty=0.01,
        seed=20260914,
    )
    fit_rows = [{**row, "role": "train"} for row in rows]
    actor, history = fit_demonstration_actor(
        [row["features"] for row in fit_rows],
        [options.index(row["option"]) for row in fit_rows],
        [row["source_id"] for row in fit_rows],
        options,
        reference,
        config=config,
    )
    learned_probabilities = _probabilities(
        actor, rows, options, reference, floor=EXPLORATION_FLOOR
    )
    reference_probabilities = np.broadcast_to(reference, learned_probabilities.shape)
    marginal_probabilities = np.broadcast_to(marginal, learned_probabilities.shape)
    metrics = {
        "generic_reference": _metrics(rows, reference_probabilities, options, weights),
        "source_balanced_marginal": _metrics(
            rows, marginal_probabilities, options, weights
        ),
        "context_conditioned_actor": _metrics(
            rows, learned_probabilities, options, weights
        ),
    }
    gate3 = metrics["context_conditioned_actor"]["source_balanced_nll"] < min(
        metrics["generic_reference"]["source_balanced_nll"],
        metrics["source_balanced_marginal"]["source_balanced_nll"],
    )
    route_counts = Counter(
        min(3, row["high_level_decisions"]) for row in route_summaries
    )
    module_count_probabilities = normalized_module_count_probabilities(
        sorted(route_counts.items())
    )
    training_identity = identity(
        {
            "schema_version": "t4_route_distillation_training_identity_v1",
            "library_sha256": sha256(LIBRARY),
            "contract_sha256": sha256(CONTRACT),
            "teacher_programs": len(traces),
            "decisions": len(rows),
            "configuration": asdict(config),
        }
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "evidence": "answer-known supervised T4 route proposal imitation",
        "state_dim": actor.state_dim,
        "option_dim": actor.option_dim,
        "hidden": config.hidden,
        "parameters": {
            name: tensor.detach().tolist()
            for name, tensor in sorted(actor.state_dict().items())
        },
        "parameter_identity": option_actor_parameter_id(actor),
        "options": list(options),
        "reference": reference.tolist(),
        "exploration_floor": EXPLORATION_FLOOR,
        "module_count_probabilities": list(module_count_probabilities),
        "binding_prototypes": _binding_prototypes(rows),
        "training_identity": training_identity,
    }
    # Runtime reconstruction is part of the gate, not deferred to deployment.
    policy = RouteDistilledProgramPolicy.from_checkpoint(checkpoint)
    test_distribution = policy.option_probabilities(traces[0]["source"])
    if np.any(test_distribution <= 0) or not np.isclose(test_distribution.sum(), 1.0):
        raise RuntimeError("runtime checkpoint does not produce a normalized law")
    forbidden_values = {
        *groups,
        *(row["original_seed"] for row in contract["cells"].values()),
        *(row["target"] for row in contract["cells"].values()),
        *(teacher["program_id"] for teacher in traces),
        *(teacher["trace"]["endpoint"] for teacher in traces),
    }
    checkpoint_strings = set(_strings(checkpoint))
    forbidden_hits = sorted(forbidden_values & checkpoint_strings)
    forbidden_keys = sorted(
        {
            value
            for value in checkpoint_strings
            if value
            in {
                "target",
                "protein",
                "seed_identity",
                "route_id",
                "endpoint",
                "source_atom",
                "program",
            }
        }
    )
    gate4 = not forbidden_hits and not forbidden_keys
    gates = {
        "teacher_replay": {
            "passed": len(traces) == 77 and not exclusions,
            "coverage": len(traces) / 77,
            "execution_precision": 1.0,
            "excluded": exclusions,
        },
        "compression": {
            "passed": all(row["exact_replay"] for row in route_summaries),
            "primitive_edits": sum(row["primitive_edits"] for row in route_summaries),
            "high_level_decisions": sum(
                row["high_level_decisions"] for row in route_summaries
            ),
            "compound_decisions": sum(
                row["compound_decisions"] for row in route_summaries
            ),
            "mean_compression_ratio": float(
                np.mean([row["compression_ratio"] for row in route_summaries])
            ),
            "primitive_fallbacks_preserved": sum(
                row["primitive_fallback_decisions"] for row in route_summaries
            ),
            "minimality_claimed": False,
        },
        "proposal_rank": {
            "passed": gate3,
            "metrics": metrics,
            "scope": "all-route fitted decision rank; not autonomous endpoint recall",
        },
        "runtime_provenance": {
            "passed": gate4,
            "forbidden_value_hits": forbidden_hits,
            "forbidden_key_hits": forbidden_keys,
            "runtime_route_rows": 0,
            "runtime_endpoint_rows": 0,
            "runtime_target_maps": 0,
        },
    }
    passed = all(row["passed"] for row in gates.values())
    if not passed:
        raise RuntimeError(f"route-distillation gate failed: {gates}")
    if code_revision is None:
        code_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    if working_tree_dirty is None:
        working_tree_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    report = {
        "schema_version": "t4_route_distillation_result_v1",
        "decision": "zero_oracle_gates_passed_scored_pilot_not_yet_run",
        "evidence": "answer-known all-route supervised proposal imitation",
        "gates": gates,
        "teacher_census": {
            "complete_routes": len(traces),
            "source_groups": len({row["source_group"] for row in traces}),
            "decisions": len(rows),
            "route_summaries": route_summaries,
        },
        "fit": {
            "configuration": asdict(config),
            "history": history,
            "module_count_probabilities": module_count_probabilities,
        },
        "inputs": {
            str(CONTRACT.relative_to(ROOT)): sha256(CONTRACT),
            str(LIBRARY.relative_to(ROOT)): sha256(LIBRARY),
            str(SEEDS.relative_to(ROOT)): sha256(SEEDS),
        },
        "implementation": {
            "revision": code_revision,
            "working_tree_dirty": working_tree_dirty,
            "script_sha256": sha256(Path(__file__)),
            "policy_sha256": sha256(
                ROOT / "src/compose_v4/control/route_distilled_program_policy.py"
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
            "device": "CPU",
            "threads": 1,
        },
        "costs": {"new_oracle_calls": 0, "new_docking_calls": 0},
        "limitations": [
            "All 77 complete T4 routes supervise the performance actor.",
            "The current high-level recognizer preserves most decisions as primitive fallbacks.",
            "Teacher-decision rank is not autonomous route or endpoint recall.",
            "The scored five-cell pilot remains prospective and no score match is guaranteed.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    publish(output / "actor.json.gz", checkpoint, compressed=True)
    report["outputs"] = {"actor.json.gz": sha256(output / "actor.json.gz")}
    publish(output / "result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics/t4_route_distillation/attempt_1",
    )
    result = run(parser.parse_args().output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "gates": result["gates"],
                "costs": result["costs"],
            },
            indent=2,
        )
    )
