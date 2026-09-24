"""Replay every accepted action and audit the frozen completed-interface pilot.

This reads the already finished units; it never samples, changes a molecule, or
rewrites the original summary.  It also checks the five-percentage-point
quality rule in the written pre-outcome contract independently of the runner.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from run_fragment_completed_interface_dev import (
    ARMS,
    CHECKPOINT,
    CONTRACT,
    MANIFEST,
    OUTPUT,
    PROMPTS,
    SAMPLER,
    _arm_summary,
    _sha256,
    _unit_path,
    _unit_row,
)

from compose_v4.benchmark.fragment_conditioned_sampler import (
    RegionLock,
    build_prompt_context,
)
from compose_v4.benchmark.fragment_constrained import load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.rewrite import operators
from compose_v4.rewrite.kernel import de_novo_rewrite_system

ROOT = Path(__file__).resolve().parents[1]
ACTION_TYPES = {
    name: getattr(operators, name)
    for name in (
        "AtomDelete",
        "AtomInsert",
        "AtomRestate",
        "BondInsert",
        "BondReorder",
        "BondReroute",
    )
}


def _action(record: dict):
    action_type = record["action_type"]
    if action_type not in ACTION_TYPES:
        raise ValueError(f"unsupported saved action type: {action_type}")
    payload = dict(record["payload"])
    if action_type == "AtomInsert":
        payload["neighbors"] = tuple(tuple(pair) for pair in payload["neighbors"])
    return ACTION_TYPES[action_type](**payload)


def _atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    prompts = load_genmol_prompts(MANIFEST)
    system = de_novo_rewrite_system()
    units = {}
    replay = {}
    metric_split = {}
    for arm, control in ARMS.items():
        for task, drug in PROMPTS:
            path = _unit_path(arm, task, drug)
            unit = json.loads(path.read_text())
            units[(arm, task.value, drug)] = unit
            row = _unit_row(unit, drug)
            raw_samples = [attempt["committed_smiles"] or "" for attempt in row["attempt_records"]]
            task_samples = [attempt["emitted_smiles"] or "" for attempt in row["attempt_records"]]
            raw_metrics = official_prompt_metrics(raw_samples, expected_samples=len(raw_samples))
            task_metrics = official_prompt_metrics(task_samples, expected_samples=len(task_samples))
            for key, value in task_metrics.items():
                if not math.isclose(value, row["official"][key], abs_tol=1e-9):
                    raise RuntimeError(f"saved official {key} differs from replay: {path}")
            metric_split[str(path)] = {
                "attempts": len(raw_samples),
                "chemically_valid_committed": row["committed_chemically_valid"],
                "full_task_success": row["emitted_nonempty"],
                "official_metrics_on_all_committed": raw_metrics,
                "official_metrics_on_constraint_emitted": task_metrics,
            }
            prompt = next(
                prompt for prompt in prompts if prompt.task is task and prompt.drug_name == drug
            )
            context = build_prompt_context(
                prompt, config=SAMPLER, control=control, linker_bridge_atoms=0
            )
            if context.start_smiles != row["start_smiles"]:
                raise RuntimeError(f"start SMILES differs: {path}")
            if list(context.locked_slots) != row["locked_slots"]:
                raise RuntimeError(f"locked slots differ: {path}")
            lock = RegionLock(
                context.start_state,
                context.locked_slots,
                released_pairs=context.attachment.released_pairs,
            )
            failures = []
            replayed = 0
            lock_steps = 0
            for attempt in row["attempt_records"]:
                state = context.start_state
                for event_index, record in enumerate(attempt["accepted_actions"]):
                    try:
                        state = system.apply(state, record["rule"], _action(record))
                    except Exception as error:  # noqa: BLE001 - preserve any replay failure
                        failures.append(
                            {
                                "attempt": attempt["attempt_index"],
                                "event": event_index,
                                "reason": f"executor replay: {type(error).__name__}: {error}",
                            }
                        )
                        break
                    lock_steps += 1
                    if not lock.permits(state):
                        failures.append(
                            {
                                "attempt": attempt["attempt_index"],
                                "event": event_index,
                                "reason": "accepted event violated exact region lock",
                            }
                        )
                        break
                else:
                    saved = attempt["committed_smiles"]
                    observed = molecular_graph_to_smiles(state) if attempt["events"] else None
                    if observed != saved:
                        failures.append(
                            {
                                "attempt": attempt["attempt_index"],
                                "reason": "replayed endpoint differs from committed endpoint",
                                "observed": observed,
                                "saved": saved,
                            }
                        )
                    else:
                        replayed += 1
            replay[str(path)] = {
                "file_sha256": _sha256(path),
                "attempts": len(row["attempt_records"]),
                "exact_endpoint_replays": replayed,
                "exact_lock_passing_steps": lock_steps,
                "failures": failures,
            }

    historical = _arm_summary(units, "historical")
    candidate = _arm_summary(units, "completed_site")
    corrected_quality_check = (
        candidate["official_quality_mean"] >= historical["official_quality_mean"] - 5.0
    )
    original_summary = json.loads((OUTPUT / "summary.json").read_text())
    all_replay = all(
        not row["failures"] and row["attempts"] == row["exact_endpoint_replays"]
        for row in replay.values()
    )
    checks = dict(original_summary["checks"])
    checks["quality_drop_at_most_five_points"] = corrected_quality_check
    checks["all_actions_replay_and_preserve_lock"] = all_replay
    report = {
        "schema_version": "fragment_completed_interface_replay_audit_v1",
        "contract_sha256": _sha256(CONTRACT),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "manifest_sha256": _sha256(MANIFEST),
        "original_summary_sha256": _sha256(OUTPUT / "summary.json"),
        "auditor_sha256": _sha256(ROOT / "tools/audit_fragment_completed_interface_dev.py"),
        "units": replay,
        "metric_split": metric_split,
        "original_checks": original_summary["checks"],
        "corrected_checks": checks,
        "quality_gate_correction": (
            "The pre-outcome document specified five percentage points, while "
            "the original runner compared 0.05 on a 0-100 quality scale. "
            "The original summary remains unchanged. Both readings pass for "
            "this result, and the overall qualification remains false."
        ),
        "qualified_for_held_test": all(checks.values()),
        "overall_attempts": sum(row["attempts"] for row in replay.values()),
        "overall_exact_replays": sum(row["exact_endpoint_replays"] for row in replay.values()),
        "overall_lock_passing_steps": sum(
            row["exact_lock_passing_steps"] for row in replay.values()
        ),
    }
    _atomic_json(OUTPUT / "replay_audit.json", report)
    print(
        json.dumps(
            {
                "attempts": report["overall_attempts"],
                "exact_replays": report["overall_exact_replays"],
                "lock_passing_steps": report["overall_lock_passing_steps"],
                "corrected_checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
