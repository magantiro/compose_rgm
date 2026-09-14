"""Run the zero-oracle PMO legal WHERE/HOW successor-policy gate."""

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
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_dependency_region_policy_comparison import (
    route_events,
    weighted_teacher_events,
)
from compose_v4.experiments.pmo_legal_action_policy import (
    SCHEMA,
    RankerConfig,
    action_features,
    enumerate_rule_successors,
    fit_diagonal_contrastive_ranker,
    rank_successor_keys,
    score_features,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("configs/pmo_legal_action_policy_v1.json")
DEFAULT_OUTPUT = Path("diagnostics/pmo_legal_action_policy/attempt_1")


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
        or payload.get("schema_version") != "pmo_legal_action_policy_contract_v1"
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError(f"invalid PMO legal-action policy contract: {path}")
    return payload


def _load_envelope(path: Path, physical: str, payload_hash: str) -> dict:
    if sha256_file(path) != physical:
        raise ValueError(f"sealed physical hash mismatch: {path}")
    raw = (
        gzip.decompress(path.read_bytes()).decode()
        if path.suffix == ".gz"
        else path.read_text()
    )
    envelope = json.loads(raw)
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or envelope.get("payload_sha256") != identity(payload)
        or envelope["payload_sha256"] != payload_hash
    ):
        raise ValueError(f"sealed payload mismatch: {path}")
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


def _step_rows(routes: list[dict]) -> list[dict]:
    weights = {
        event.event_id: event.weight for event in weighted_teacher_events(routes)
    }
    rows = []
    for route in routes:
        route_event_rows = route_events(route, weight=1.0)
        for step, event in enumerate(route_event_rows):
            rows.append(
                {
                    "event_id": event.event_id,
                    "trace_identity": route["trace_identity"],
                    "lineage_identity": route["lineage_identity"],
                    "task_family": route["task_family"],
                    "test_fold": route["test_fold"],
                    "runtime_supported": event.runtime_supported,
                    "weight": weights[event.event_id],
                    "state": route["states"][step],
                    "teacher_successor_key": canonical_state_key(
                        decode_state(route["states"][step + 1])
                    ),
                    "rule": route["actions"][step]["executor_rule"],
                }
            )
    return rows


def _group_rows(rows: list[dict]) -> list[list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[identity(row["state"])].append(row)
    return [grouped[key] for key in sorted(grouped)]


def _sample_negatives(candidates, positive_key: str, event_id: str, maximum: int):
    negatives = [row for row in candidates if row[0] != positive_key]
    negatives.sort(
        key=lambda row: identity(
            {
                "schema_version": "pmo_legal_negative_order_v1",
                "event_id": event_id,
                "successor_key": row[0],
            }
        )
    )
    return negatives[:maximum]


def _prepare_feature_fibers(rows: list[dict]):
    """Enumerate and featurize each exact state/rule fiber exactly once."""

    fibers = {}
    for group in _group_rows(rows):
        state_identity = identity(group[0]["state"])
        graph = decode_state(group[0]["state"])
        for rule in sorted({row["rule"] for row in group}):
            candidates = enumerate_rule_successors(graph, rule)
            fibers[(state_identity, rule)] = tuple(
                (candidate.successor_key, action_features(graph, candidate))
                for candidate in candidates
            )
    return fibers


def _fit_fold(
    rows: list[dict], fold: int, config: RankerConfig, feature_fibers
) -> tuple[dict, dict]:
    training = [row for row in rows if row["test_fold"] != fold]
    differences = []
    census = {
        "events": len(training),
        "covered": 0,
        "no_legal_fiber": 0,
        "teacher_absent": 0,
        "no_negative": 0,
        "sampled_negative_pairs": 0,
    }
    for group in _group_rows(training):
        state_identity = identity(group[0]["state"])
        for row in group:
            candidates = feature_fibers[(state_identity, row["rule"])]
            if not candidates:
                census["no_legal_fiber"] += 1
                continue
            positive = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate[0] == row["teacher_successor_key"]
                ),
                None,
            )
            if positive is None:
                census["teacher_absent"] += 1
                continue
            census["covered"] += 1
            negatives = _sample_negatives(
                candidates,
                positive[0],
                row["event_id"],
                config.negatives_per_teacher,
            )
            if not negatives:
                census["no_negative"] += 1
                continue
            average_negative = np.mean(
                [candidate[1] for candidate in negatives],
                axis=0,
            )
            differences.append(
                (
                    positive[1] - average_negative,
                    row["weight"],
                )
            )
            census["sampled_negative_pairs"] += len(negatives)
    checkpoint = fit_diagonal_contrastive_ranker(differences, config)
    checkpoint["training_identity"] = identity(
        {
            "schema_version": "pmo_legal_action_training_identity_v1",
            "fold": fold,
            "events": len(training),
            "covered": census["covered"],
            "pairs": census["sampled_negative_pairs"],
            "configuration": config.__dict__,
        }
    )
    return checkpoint, census


def _evaluate_fold(
    rows: list[dict], fold: int, checkpoint: dict, feature_fibers
) -> tuple[list[dict], dict]:
    held_out = [row for row in rows if row["test_fold"] == fold]
    outcomes = []
    for group in _group_rows(held_out):
        state_identity = identity(group[0]["state"])
        for row in group:
            candidates = feature_fibers[(state_identity, row["rule"])]
            successor_keys = tuple(candidate[0] for candidate in candidates)
            scores = {
                successor_key: score_features(checkpoint, features)
                for successor_key, features in candidates
            }
            uniform_rank = rank_successor_keys(
                successor_keys, row["teacher_successor_key"]
            )
            learned_rank = rank_successor_keys(
                successor_keys,
                row["teacher_successor_key"],
                scores,
            )
            outcomes.append(
                {
                    "event_id": row["event_id"],
                    "task_family": row["task_family"],
                    "runtime_supported": row["runtime_supported"],
                    "weight": row["weight"],
                    "rule": row["rule"],
                    "fiber_size": len(candidates),
                    "teacher_covered": uniform_rank is not None,
                    "uniform_rank": uniform_rank,
                    "learned_rank": learned_rank,
                }
            )
    return outcomes, _metrics(outcomes)


def _metrics(rows: list[dict]) -> dict:
    def scope(selected):
        denominator = sum(row["weight"] for row in selected)
        covered = [row for row in selected if row["teacher_covered"]]
        covered_weight = sum(row["weight"] for row in covered)
        result = {
            "events": len(selected),
            "covered": len(covered),
            "coverage": covered_weight / denominator if denominator else 0.0,
            "mean_fiber_size": (
                sum(row["fiber_size"] for row in selected) / len(selected)
                if selected
                else None
            ),
        }
        for policy in ("uniform", "learned"):
            ranks = [row for row in covered if row[f"{policy}_rank"] is not None]
            weight = sum(row["weight"] for row in ranks)
            result[policy] = {
                "precision": len(ranks) / len(covered) if covered else None,
                "weighted_mean_rank": (
                    sum(row["weight"] * row[f"{policy}_rank"] for row in ranks) / weight
                    if weight
                    else None
                ),
                "weighted_mean_reciprocal_rank": (
                    sum(row["weight"] / row[f"{policy}_rank"] for row in ranks) / weight
                    if weight
                    else None
                ),
                **{
                    f"weighted_top_{cutoff}": (
                        sum(
                            row["weight"]
                            for row in ranks
                            if row[f"{policy}_rank"] <= cutoff
                        )
                        / weight
                        if weight
                        else None
                    )
                    for cutoff in (1, 5, 10, 32)
                },
            }
        return result

    return {
        "all": scope(rows),
        "runtime_supported": scope([row for row in rows if row["runtime_supported"]]),
        "long_route_local_only": scope(
            [row for row in rows if not row["runtime_supported"]]
        ),
    }


def _runtime_leakage(checkpoint: dict, teacher_values: set[str]) -> dict:
    forbidden = {
        "task",
        "family",
        "route",
        "lineage",
        "endpoint",
        "smiles",
        "slot",
        "teacher",
    }
    key_hits, strings = [], set()

    def walk(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                if any(term in str(key).lower() for term in forbidden):
                    key_hits.append(f"{path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            strings.add(value)

    walk(checkpoint)
    return {
        "forbidden_key_hits": sorted(key_hits),
        "forbidden_teacher_value_hits": sorted(strings & teacher_values),
    }


def _git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def run(output: Path = ROOT / DEFAULT_OUTPUT, root: Path = ROOT) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite PMO legal-action result: {output}")
    began = perf_counter()
    contract = _load_contract(root)
    source = contract["input"]
    corpus = _load_envelope(
        root / source["corpus_path"],
        source["corpus_sha256"],
        source["corpus_payload_sha256"],
    )
    policy_result = _load_envelope(
        root / source["policy_comparison_result_path"],
        source["policy_comparison_result_sha256"],
        source["policy_comparison_result_payload_sha256"],
    )
    if (
        policy_result["new_oracle_calls"] != 0
        or policy_result["decision"]["selected_policy"] is not None
        or policy_result["decision"]["scored_pilot_authorized"]
    ):
        raise ValueError("upstream PMO ranking gate decision changed")
    routes = list(corpus["routes"])
    if corpus.get("split", {}).get("split_identity") != contract["split_identity"]:
        raise ValueError("sealed PMO task-family split identity changed")
    rows = _step_rows(routes)
    observed = {
        "routes": len(routes),
        "events": len(rows),
        "runtime_supported_events": sum(row["runtime_supported"] for row in rows),
        "unique_states": len({identity(row["state"]) for row in rows}),
        "folds": sorted({row["test_fold"] for row in rows}),
    }
    if observed != contract["expected"]:
        raise ValueError(f"sealed legal-action census changed: {observed}")
    config = RankerConfig(**contract["ranker"])
    teacher_values = {
        str(value)
        for route in routes
        for value in (
            route["trace_identity"],
            route["lineage_identity"],
            route["terminal_endpoint"],
        )
    }
    folds, checkpoints, all_outcomes = [], [], []
    started = perf_counter()
    feature_fibers = _prepare_feature_fibers(rows)
    fiber_preparation_seconds = perf_counter() - started
    fit_seconds, evaluation_seconds = 0.0, 0.0
    for fold in range(3):
        started = perf_counter()
        checkpoint, training = _fit_fold(rows, fold, config, feature_fibers)
        fit_seconds += perf_counter() - started
        leakage = _runtime_leakage(checkpoint, teacher_values)
        if leakage["forbidden_key_hits"] or leakage["forbidden_teacher_value_hits"]:
            raise RuntimeError(f"legal-action checkpoint teacher leak: {leakage}")
        started = perf_counter()
        outcomes, metrics = _evaluate_fold(rows, fold, checkpoint, feature_fibers)
        evaluation_seconds += perf_counter() - started
        all_outcomes.extend(outcomes)
        folds.append(
            {
                "fold": fold,
                "held_out_task_families": sorted(
                    {row["task_family"] for row in rows if row["test_fold"] == fold}
                ),
                "training": training,
                "metrics": metrics,
            }
        )
        checkpoints.append({"fold": fold, "checkpoint": checkpoint})
    runtime = {
        "schema_version": "pmo_legal_action_fold_checkpoints_v1",
        "operator_support": "fixed_generic_active8_canonical_successors",
        "folds": checkpoints,
        "new_oracle_calls": 0,
    }
    payload = {
        "schema_version": SCHEMA,
        "contract": {
            "path": str(CONTRACT),
            "sha256": sha256_file(root / CONTRACT),
            "payload_sha256": identity(contract),
        },
        "input": source,
        "observed_census": observed,
        "folds": folds,
        "aggregate": _metrics(all_outcomes),
        "decision": {
            "selected_policy": None,
            "scored_pilot_authorized": False,
            "complete_program_decoder_authorized": False,
            "status": "one_step_where_how_gate_only",
        },
        "claim_boundary": (
            "held-family one-step legal-successor coverage and rank; not complete "
            "route recovery or PMO objective performance"
        ),
        "runtime_checkpoint": {
            "schema_version": runtime["schema_version"],
            "path": contract["outputs"]["runtime_checkpoint"],
        },
        "new_oracle_calls": 0,
        "task_scores_loaded": False,
        "compute": {
            "wall_seconds": perf_counter() - began,
            "fiber_preparation_seconds": fiber_preparation_seconds,
            "fit_seconds": fit_seconds,
            "evaluation_seconds": evaluation_seconds,
            "peak_rss_bytes": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if platform.system() == "Darwin"
                else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "hardware": platform.machine(),
            "numpy_version": np.__version__,
        },
        "code_revision": _git_revision(root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _publish(
            temporary / "runtime_fold_checkpoints.json.gz", runtime, compressed=True
        )
        checkpoint_path = temporary / "runtime_fold_checkpoints.json.gz"
        checkpoint_payload = json.loads(
            gzip.decompress(checkpoint_path.read_bytes()).decode()
        )["payload_sha256"]
        payload["runtime_checkpoint"].update(
            {
                "sha256": sha256_file(checkpoint_path),
                "payload_sha256": checkpoint_payload,
            }
        )
        _publish(temporary / "result.json", payload)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
