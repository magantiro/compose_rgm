"""Run the held-family PMO dependency-region policy comparison."""

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
    CHECKPOINT_SCHEMA,
    POLICIES,
    SCHEMA,
    PolicyConfig,
    checkpoint_envelope,
    event_outcomes,
    fit_fold_checkpoint,
    runtime_checkpoint_leakage,
    score_candidate,
    teacher_forced_metrics,
    weighted_teacher_events,
)
from compose_v4.experiments.pmo_dependency_region_program import (
    dependency_region_program,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("configs/pmo_dependency_region_policy_comparison_v1.json")
DEFAULT_OUTPUT = Path("diagnostics/pmo_dependency_region_policy_comparison/attempt_1")


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
        or payload.get("schema_version") != "pmo_dependency_region_policy_comparison_contract_v1"
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError(f"invalid PMO dependency-region policy contract: {path}")
    return payload


def _load_envelope(path: Path, physical: str, payload_hash: str) -> dict:
    if sha256_file(path) != physical:
        raise ValueError(f"sealed physical hash mismatch: {path}")
    raw = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    envelope = json.loads(raw)
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or envelope.get("payload_sha256") != identity(payload)
        or envelope["payload_sha256"] != payload_hash
    ):
        raise ValueError(f"sealed payload hash mismatch: {path}")
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


def _candidate_route(panel: dict, attempt: dict) -> dict | None:
    if attempt["status"] != "complete" or attempt.get("trace") is None:
        return None
    trace = attempt["trace"]
    states = trace["states"]
    actions = trace["actions"]
    if len(states) != len(actions) + 1:
        raise ValueError("complete panel trace has misaligned states/actions")
    route = {
        "trace_identity": f"generic:{panel['lineage_identity']}:{attempt['attempt_id']}",
        "lineage_identity": panel["lineage_identity"],
        "task_family": panel["task_family"],
        "test_fold": panel["test_fold"],
        "source_state": states[0],
        "states": states,
        "actions": actions,
    }
    route["dependency_region_program"] = dependency_region_program(states, actions)
    return route


def _endpoint_key(route: dict) -> str:
    state = route["states"][-1]
    return canonical_state_key(decode_state(state))


def _teacher_values(routes: list[dict], panels: list[dict]) -> set[str]:
    values = set()
    for route in routes:
        values.update(
            {
                str(route["trace_identity"]),
                str(route["lineage_identity"]),
                str(route["terminal_endpoint"]),
            }
        )
    for panel in panels:
        values.update(str(value) for value in panel["teacher_trace_identities"])
        values.update(str(value) for value in panel["teacher_endpoint_states"])
    return values


def _rank_panel(
    checkpoint: dict,
    panel: dict,
    teachers: list[dict],
    policy: str,
) -> dict:
    generic = [
        route
        for attempt in panel["attempts"]
        if (route := _candidate_route(panel, attempt)) is not None
    ]
    unique = {}
    for route in generic:
        unique.setdefault(_endpoint_key(route), ("generic", route))
    generic_unique = list(unique.values())
    generic_ranked = sorted(
        (
            {
                "origin": origin,
                "endpoint_key": endpoint,
                "score": score_candidate(checkpoint, route, policy),
            }
            for endpoint, (origin, route) in unique.items()
        ),
        key=lambda row: (-row["score"], row["endpoint_key"]),
    )
    injected = dict(unique)
    for route in teachers:
        injected[_endpoint_key(route)] = ("teacher", route)
    injected_ranked = sorted(
        (
            {
                "origin": origin,
                "endpoint_key": endpoint,
                "score": score_candidate(checkpoint, route, policy),
            }
            for endpoint, (origin, route) in injected.items()
        ),
        key=lambda row: (-row["score"], row["endpoint_key"]),
    )
    teacher_ranks = [
        index for index, row in enumerate(injected_ranked, start=1) if row["origin"] == "teacher"
    ]
    best_rank = min(teacher_ranks) if teacher_ranks else None
    return {
        "lineage_identity": panel["lineage_identity"],
        "task_family": panel["task_family"],
        "test_fold": panel["test_fold"],
        "generic_attempts": len(panel["attempts"]),
        "generic_complete": len(generic),
        "generic_unique": len(generic_unique),
        "generic_panel_exact_teacher_endpoints": sum(
            row["teacher_relation"] == "exact"
            for row in panel["attempts"]
            if row["status"] == "complete"
        ),
        "generic_panel_radius2_teacher_endpoints": sum(
            row["teacher_relation"] in {"exact", "radius_2"}
            for row in panel["attempts"]
            if row["status"] == "complete"
        ),
        "generic_top1_origin": generic_ranked[0]["origin"] if generic_ranked else None,
        "teacher_injected_candidates": len(injected_ranked),
        "teacher_endpoints": len(teacher_ranks),
        "best_teacher_rank": best_rank,
        "teacher_injected_coverage": bool(teacher_ranks),
        "teacher_reciprocal_rank": 1 / best_rank if best_rank else None,
        "teacher_top1": best_rank == 1 if best_rank else None,
        "teacher_top5": best_rank <= 5 if best_rank else None,
        "teacher_top10": best_rank <= 10 if best_rank else None,
    }


def _panel_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("panel metrics require rows")
    covered = [row for row in rows if row["teacher_injected_coverage"]]
    return {
        "panels": len(rows),
        "generic_attempts": sum(row["generic_attempts"] for row in rows),
        "generic_complete": sum(row["generic_complete"] for row in rows),
        "generic_unique": sum(row["generic_unique"] for row in rows),
        "generic_exact_recovery": sum(
            row["generic_panel_exact_teacher_endpoints"] > 0 for row in rows
        )
        / len(rows),
        "generic_radius2_recovery": sum(
            row["generic_panel_radius2_teacher_endpoints"] > 0 for row in rows
        )
        / len(rows),
        "teacher_injected_covered_panels": len(covered),
        "teacher_injected_coverage": len(covered) / len(rows),
        "teacher_injected_mean_reciprocal_rank": (
            sum(row["teacher_reciprocal_rank"] for row in covered) / len(covered)
            if covered
            else None
        ),
        "teacher_injected_top1": (
            sum(row["teacher_top1"] for row in covered) / len(covered) if covered else None
        ),
        "teacher_injected_top5": (
            sum(row["teacher_top5"] for row in covered) / len(covered) if covered else None
        ),
        "teacher_injected_top10": (
            sum(row["teacher_top10"] for row in covered) / len(covered) if covered else None
        ),
        "mean_best_teacher_rank": (
            sum(row["best_teacher_rank"] for row in covered) / len(covered) if covered else None
        ),
    }


def _teacher_metric_scopes(outcomes: tuple[dict, ...] | list[dict]) -> dict:
    rows = tuple(outcomes)
    runtime = tuple(row for row in rows if row["runtime_supported"])
    local_only = tuple(row for row in rows if not row["runtime_supported"])
    return {
        "all": teacher_forced_metrics(rows),
        "runtime_supported": teacher_forced_metrics(runtime),
        "long_route_local_only": teacher_forced_metrics(local_only),
    }


def _git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def run(output: Path = ROOT / DEFAULT_OUTPUT, root: Path = ROOT) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite PMO policy comparison: {output}")
    began = perf_counter()
    contract = _load_contract(root)
    source = contract["input"]
    corpus = _load_envelope(
        root / source["dependency_region_corpus_path"],
        source["dependency_region_corpus_sha256"],
        source["dependency_region_corpus_payload_sha256"],
    )
    panel_corpus = _load_envelope(
        root / source["panel_corpus_path"],
        source["panel_corpus_sha256"],
        source["panel_corpus_payload_sha256"],
    )
    routes = list(corpus["routes"])
    panels = list(panel_corpus["source_conditioned_panels"])
    expected = contract["expected"]
    observed = {
        "routes": len(routes),
        "runtime_supported_routes": sum(
            route["dependency_region_program"]["runtime_length_supported"] for route in routes
        ),
        "lineages": len({route["lineage_identity"] for route in routes}),
        "task_families": len({route["task_family"] for route in routes}),
        "panels": len(panels),
        "panel_attempts": sum(len(panel["attempts"]) for panel in panels),
    }
    if observed != expected:
        raise ValueError(f"sealed PMO policy census changed: {observed} != {expected}")
    if corpus["split"]["split_identity"] != contract["split_identity"]:
        raise ValueError("PMO policy split identity changed")

    config = PolicyConfig(**contract["models"])
    events = weighted_teacher_events(routes)
    teacher_values = _teacher_values(routes, panels)
    folds, teacher_outcomes = [], {policy: [] for policy in POLICIES}
    panel_rows = {policy: [] for policy in POLICIES}
    routes_by_lineage = defaultdict(list)
    for route in routes:
        if route["dependency_region_program"]["runtime_length_supported"]:
            routes_by_lineage[route["lineage_identity"]].append(route)
    fit_seconds = 0.0
    for fold_index in range(3):
        fit_began = perf_counter()
        checkpoint = fit_fold_checkpoint(events, fold_index, config)
        fit_seconds += perf_counter() - fit_began
        leakage = runtime_checkpoint_leakage(checkpoint, teacher_values)
        if leakage["forbidden_key_hits"] or leakage["forbidden_teacher_value_hits"]:
            raise RuntimeError(f"runtime checkpoint teacher leak: {leakage}")
        fold_panels = [panel for panel in panels if panel["test_fold"] == fold_index]
        fold_result = {
            "fold": fold_index,
            "held_out_task_families": sorted({panel["task_family"] for panel in fold_panels}),
            "test_panels": len(fold_panels),
            "test_routes": sum(route["test_fold"] == fold_index for route in routes),
            "policies": {},
        }
        for policy in POLICIES:
            outcomes = event_outcomes(checkpoint, events, policy)
            teacher_outcomes[policy].extend(outcomes)
            ranked = [
                _rank_panel(
                    checkpoint,
                    panel,
                    routes_by_lineage[panel["lineage_identity"]],
                    policy,
                )
                for panel in fold_panels
            ]
            panel_rows[policy].extend(ranked)
            fold_result["policies"][policy] = {
                "teacher_forced": _teacher_metric_scopes(outcomes),
                "shared_panels": _panel_metrics(ranked),
            }
        folds.append({"result": fold_result, "checkpoint": checkpoint})

    runtime = checkpoint_envelope([row["checkpoint"] for row in folds])
    aggregate = {
        policy: {
            "teacher_forced": _teacher_metric_scopes(teacher_outcomes[policy]),
            "shared_panels": _panel_metrics(panel_rows[policy]),
        }
        for policy in POLICIES
    }
    decision = {
        "selected_policy": None,
        "status": "zero_oracle_comparison_only_no_autonomous_decoder",
        "reason": (
            "the policies rank exact complete action traces on a shared legal candidate "
            "support but do not yet generate HOW parameters or legal bindings autonomously"
        ),
        "scored_pilot_authorized": False,
    }
    payload = {
        "schema_version": SCHEMA,
        "contract": {
            "path": str(CONTRACT),
            "sha256": sha256_file(root / CONTRACT),
            "payload_sha256": identity(contract),
        },
        "inputs": source,
        "split": corpus["split"],
        "policies": list(POLICIES),
        "observed_census": observed,
        "folds": [row["result"] for row in folds],
        "aggregate": aggregate,
        "decision": decision,
        "claim_boundary": {
            "teacher_forced": "held-task-family retrospective likelihood/rank",
            "generic_panels": "task-blind pre-existing proposals; no teacher injection",
            "teacher_injected_panels": (
                "diagnostic shared-candidate reranking with held-family teachers added only "
                "for ranking evaluation"
            ),
            "autonomous_complete_program_generation": "not implemented or measured",
        },
        "runtime_checkpoint": {
            "schema_version": CHECKPOINT_SCHEMA,
            "path": contract["outputs"]["runtime_checkpoint"],
        },
        "new_oracle_calls": 0,
        "task_scores_loaded": False,
        "compute": {
            "wall_seconds": perf_counter() - began,
            "fit_seconds": fit_seconds,
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
        _publish(temporary / "runtime_fold_checkpoints.json.gz", runtime, compressed=True)
        runtime_hash = sha256_file(temporary / "runtime_fold_checkpoints.json.gz")
        runtime_payload = json.loads(
            gzip.decompress((temporary / "runtime_fold_checkpoints.json.gz").read_bytes()).decode()
        )["payload_sha256"]
        payload["runtime_checkpoint"].update(
            {"sha256": runtime_hash, "payload_sha256": runtime_payload}
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
    arguments = parser.parse_args()
    payload = run(arguments.output)
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
