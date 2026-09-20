#!/usr/bin/env python3
"""Materialize the zero-oracle PMO complete-route Dynamic gate."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.pmo_complete_route_dynamic_gate import (
    SCHEMA,
    fit_complete_route_checkpoints,
    historical_recursive_fiber_replay,
    load_envelope,
    productive_continuation_ranks,
    run_matched_known_answer_support,
    teacher_forced_support,
)
from compose_v4.experiments.pmo_route_fiber_production_yield import (
    CORPUS,
    EXPECTED_SHA256,
    LEGAL_RUNTIME,
    fit_route_transition_checkpoint,
)

HISTORY = "diagnostics/pmo_online_policy/result_sealed.json"
OUTPUT = "diagnostics/pmo_complete_route_dynamic_gate_v1/attempt_1/result.json"
CHECKPOINT = (
    "diagnostics/pmo_complete_route_dynamic_gate_v1/attempt_1/runtime_fold_checkpoints.json"
)
OLD_FACTORIAL = {
    "configs/pmo_route_fiber_scored_pilot_v1.json",
    "diagnostics/pmo_route_fiber_scored_pilot_v1/candidate_locks.json",
    "diagnostics/pmo_route_fiber_scored_pilot_v1/preflight.json",
}


def _write_sealed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    envelope = {"payload": normalized, "payload_sha256": identity(normalized)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--limit-states", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    for relative in (CORPUS, LEGAL_RUNTIME):
        verify_file(root / relative, EXPECTED_SHA256[relative])
    corpus = load_envelope(root / CORPUS)
    legal = load_envelope(root / LEGAL_RUNTIME)
    historical = load_envelope(root / HISTORY)
    complete = fit_complete_route_checkpoints(corpus)
    old = fit_route_transition_checkpoint(corpus)
    support = teacher_forced_support(corpus)
    sampled = run_matched_known_answer_support(
        corpus,
        old["payload"],
        complete["payload"],
        legal,
        limit_states=args.limit_states,
        workers=args.workers,
    )
    ranks = productive_continuation_ranks(
        corpus, complete["payload"], legal, limit_states=args.limit_states
    )
    adaptive_replay = historical_recursive_fiber_replay(historical, adaptive=True)
    blind_replay = historical_recursive_fiber_replay(historical, adaptive=False)
    autonomous_complete_support = sum(
        row["arms"]["complete_route"]["exact_endpoint_support"]
        + row["arms"]["complete_route"]["transformation_equivalent_support"]
        for row in sampled.values()
    )
    known_productive_runtime_support = sum(
        row["same_runtime_replayed_routes"] for row in support.values()
    )
    reward_enrichment = adaptive_replay["best_reward"] > blind_replay["best_reward"]
    promoted = (
        known_productive_runtime_support > 0
        and adaptive_replay["multi_generation_lineage_observed"]
        and reward_enrichment
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation = (
        "src/compose_v4/experiments/pmo_complete_route_dynamic_gate.py",
        "tools/pmo_complete_route_dynamic_gate.py",
        "tests/test_pmo_complete_route_dynamic_gate.py",
    )
    payload = {
        "schema_version": SCHEMA,
        "decision": "PROMOTE_TO_SCORED_CONTRACT" if promoted else "NO_PROMOTION",
        "new_oracle_calls": 0,
        "modal_launches": 0,
        "scored_launch_authorized": False,
        "scope": {
            "tasks": list(sampled),
            "matched_candidate_budget_per_state_per_arm": 1,
            "state_limit": args.limit_states,
            "workers": args.workers,
            "seed": 20260919,
            "answer_known_support_is_not_autonomous_generalization": True,
            "historical_replay_is_not_prospective": True,
        },
        "inputs": {
            CORPUS: sha256_file(root / CORPUS),
            LEGAL_RUNTIME: sha256_file(root / LEGAL_RUNTIME),
            HISTORY: sha256_file(root / HISTORY),
            **{path: sha256_file(root / path) for path in sorted(OLD_FACTORIAL)},
        },
        "old_negative_factorial_immutable": True,
        "code_revision": revision,
        "implementation_sha256": {path: sha256_file(root / path) for path in implementation},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "machine": platform.machine(),
        },
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_payload_sha256": complete["payload_sha256"],
        "checkpoint_audit": complete["audit"],
        "teacher_forced_runtime_support": support,
        "matched_actual_production_support": sampled,
        "productive_continuation_rank": ranks,
        "historical_recursive_replay": {
            "fiber_control": adaptive_replay,
            "reward_blind": blind_replay,
            "matched_call_budget": adaptive_replay["maximum_calls"],
            "fiber_best_minus_blind_best": (
                adaptive_replay["best_reward"] - blind_replay["best_reward"]
            ),
        },
        "promotion_rule": {
            "known_productive_complete_program_support": (known_productive_runtime_support > 0),
            "known_productive_runtime_routes": known_productive_runtime_support,
            "clean_autonomous_exact_or_equivalent_support": autonomous_complete_support,
            "historically_improving_multi_generation_archive": adaptive_replay[
                "multi_generation_lineage_observed"
            ],
            "historical_reward_enrichment_over_blind": reward_enrichment,
            "structural_recall_alone_sufficient": False,
        },
    }
    _write_sealed(root / CHECKPOINT, complete["payload"])
    _write_sealed(root / OUTPUT, payload)
    print(json.dumps({"output": OUTPUT, "decision": payload["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
