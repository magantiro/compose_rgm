"""Zero-oracle gate for route-distilled proposals plus PMO FiberControl.

This module intentionally stops before candidate generation when no split-clean
production route proposer exists.  The PMO route-distillation export is valuable
training supervision, but its sealed contract explicitly emitted neither a fitted
actor nor the compound-stage and binding targets required by the production
program sampler.  Treating that export as an executable policy would conflate
training support with runtime support.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from compose_v4.control.docking_value import identity

SCHEMA = "pmo_route_fiber_prelaunch_v1"
CONTRACT_SCHEMA = "pmo_route_fiber_pilot_contract_v1"

TRAINING_DATASET = "diagnostics/pmo_route_distillation/attempt_1/training_dataset.json.gz"
TRAINING_RESULT = "diagnostics/pmo_route_distillation/attempt_1/result.json"
DYNAMIC_RESULT = "diagnostics/pmo_dynamic_v21/result_v3.json"
INITIALIZATION = "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
CONTRACT = "configs/pmo_route_fiber_pilot_v1.json"

TASKS = (
    "celecoxib_rediscovery",
    "gsk3b",
    "perindopril_mpo",
)
ARMS = (
    "old_proposals_blind",
    "route_proposals_blind",
    "old_proposals_fiber_control",
    "route_proposals_fiber_control",
)

QUERIES_PER_ARM_TASK = 48
TOTAL_QUERY_CEILING = len(TASKS) * len(ARMS) * QUERIES_PER_ARM_TASK

EXPECTED_SHA256 = {
    TRAINING_DATASET: "4d280c13cc52493dbf1baa801bb6cc0c0ff7d956516d3edb67fe9433a2240b9e",
    TRAINING_RESULT: "ec30d1406c73011669f6a6e9661176f4d301652f2fa0479277a10bdb47f7c4d8",
    DYNAMIC_RESULT: "5cecaa308b89b3adc50ff8c552ec46f4db3f3a49e43fd780a7fce158f4dfdaa2",
    INITIALIZATION: "a1f7840df1f75567565de633c4df3d013c3165df3e43cbf2ac87b156d5f206ff",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(root: Path, relative: str) -> None:
    observed = sha256_file(root / relative)
    expected = EXPECTED_SHA256[relative]
    if observed != expected:
        raise ValueError(
            f"PMO route/FiberControl input changed: {relative}: "
            f"expected {expected}, observed {observed}"
        )


def _load_envelope(path: Path, *, compressed: bool = False) -> dict[str, Any]:
    opener = gzip.open if compressed else path.open
    arguments = (path, "rt") if compressed else ("r",)
    with opener(*arguments) as handle:
        envelope = json.load(handle)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid self-hashed envelope: {path}")
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"payload hash mismatch: {path}")
    return envelope["payload"]


def reward_to_control_score(reward: float) -> float:
    """Map PMO maximize rewards onto FiberControl's lower-is-better convention."""
    value = float(reward)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"PMO reward must be in [0, 1], observed {value}")
    return -value


def control_score_to_reward(score: float) -> float:
    value = -float(score)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"control score is outside the PMO reward image: {score}")
    return value


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(map(str, value))
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()


def _loto_audit(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = dataset.get("generic_rows")
    provenance = dataset.get("training_provenance")
    if not isinstance(rows, list) or not isinstance(provenance, list):
        raise TypeError("PMO route export is missing row-aligned supervision")
    if len(rows) != len(provenance) or len(rows) != 6143:
        raise ValueError("PMO route-export decision census changed")

    task_rows: Counter[str] = Counter()
    family_rows: Counter[str] = Counter()
    lineages_by_task: dict[str, set[str]] = defaultdict(set)
    all_tasks: set[str] = set()
    for index, (row, source) in enumerate(zip(rows, provenance, strict=True)):
        if row.get("row_index") != index or source.get("row_index") != index:
            raise ValueError("PMO route export lost row alignment")
        members = source.get("members") or []
        tasks = {str(member["task"]) for member in members}
        families = {str(member["task_family"]) for member in members}
        if not tasks or not families:
            raise ValueError("PMO route row lacks task/family provenance")
        for task in tasks:
            task_rows[task] += 1
            lineages_by_task[task].add(str(source["lineage_identity"]))
        for family in families:
            family_rows[family] += 1
        all_tasks.update(tasks)

    forbidden_generic = {
        "task",
        "task_family",
        "route_id",
        "endpoint",
        "smiles",
        "assignment",
        "source_graph",
        "program",
    }
    leaked = sorted(forbidden_generic & _recursive_keys(rows))
    if leaked:
        raise ValueError(f"generic PMO rows contain forbidden runtime fields: {leaked}")

    folds = {}
    for held_task in TASKS:
        if held_task not in all_tasks:
            raise ValueError(f"held PMO task has no route supervision: {held_task}")
        held_indices = []
        train_indices = []
        held_lineages = set(lineages_by_task[held_task])
        for index, source in enumerate(provenance):
            row_tasks = {str(member["task"]) for member in source["members"]}
            lineage = str(source["lineage_identity"])
            if held_task in row_tasks or lineage in held_lineages:
                held_indices.append(index)
            else:
                train_indices.append(index)
        train_tasks = sorted(all_tasks - {held_task})
        train_lineages = {str(provenance[index]["lineage_identity"]) for index in train_indices}
        if held_lineages & train_lineages:
            raise ValueError(f"lineage leakage in PMO fold: {held_task}")
        folds[held_task] = {
            "held_decisions": len(held_indices),
            "held_lineages": len(held_lineages),
            "train_decisions": len(train_indices),
            "train_lineages": len(train_lineages),
            "train_tasks": train_tasks,
            "held_task_absent_from_training": held_task not in train_tasks,
            "lineage_leakage": 0,
        }
    return {
        "rows": len(rows),
        "routes": len(dataset.get("route_manifest") or []),
        "tasks": sorted(all_tasks),
        "task_rows": dict(sorted(task_rows.items())),
        "family_rows": dict(sorted(family_rows.items())),
        "folds": folds,
        "generic_runtime_field_leaks": [],
    }


def pilot_policy() -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA,
        "scientific_problem": (
            "test whether split-clean route-derived structural proposals and "
            "within-task reward control complement each other on PMO"
        ),
        "primary_output": (
            "higher-is-better PMO score curves from a fixed two-by-two proposal "
            "source by selection-policy factorial"
        ),
        "tasks": list(TASKS),
        "task_roles": {
            "celecoxib_rediscovery": "exact rediscovery",
            "gsk3b": "learned bioactivity",
            "perindopril_mpo": "multi-property optimization",
        },
        "arms": list(ARMS),
        "queries_per_arm_task": QUERIES_PER_ARM_TASK,
        "total_query_ceiling": TOTAL_QUERY_CEILING,
        "automatic_retries": 0,
        "route_prior_split": "leave_exact_task_and_shared_lineages_out_before_fit",
        "same_candidate_budget_within_proposal_source": True,
        "same_generator_posthoc_baseline": True,
        "initialization_queries_count": True,
        "score_direction": "maximize",
        "fiber_control_adapter": "control_score=-pmo_reward",
        "online_reward_sharing": "none_across_tasks_or_arms",
        "runtime_route_or_endpoint_lookup": False,
        "checkpoint_forbidden_fields": [
            "task",
            "task_family",
            "route_id",
            "endpoint",
            "smiles",
            "source_graph",
            "assignment",
            "program",
        ],
        "zero_oracle_production_gate": {
            "exact_execution_precision": 1.0,
            "novel_valid_endpoint_per_initial_source": 1,
            "candidate_attempt_budget_matched": True,
            "no_teacher_injection": True,
            "loto_checkpoint_required_per_task": True,
        },
        "promotion": (
            "all production gates pass, then a separately locked scored launch may "
            "compare all four arms; this contract itself authorizes zero oracle calls"
        ),
    }


def contract_envelope() -> dict[str, Any]:
    payload = {
        **pilot_policy(),
        "authorization": (
            "2026-09-18 user authorized a bounded zero-oracle PMO route-prior "
            "plus FiberControl prelaunch on a few representative tasks"
        ),
        "inputs": EXPECTED_SHA256,
        "outputs": {"prelaunch": "diagnostics/pmo_route_fiber_pilot/prelaunch_v1.json"},
        "oracle_calls_authorized": 0,
        "modal_launch_authorized": False,
        "scored_pilot_requires_separate_authorization": True,
    }
    return {"payload": payload, "payload_sha256": identity(payload)}


def build_prelaunch(root: Path) -> dict[str, Any]:
    for relative in EXPECTED_SHA256:
        _verify_file(root, relative)
    dataset = _load_envelope(root / TRAINING_DATASET, compressed=True)
    route_result = _load_envelope(root / TRAINING_RESULT)
    dynamic_result = _load_envelope(root / DYNAMIC_RESULT)
    initialization = json.loads((root / INITIALIZATION).read_text())
    body = {key: value for key, value in initialization.items() if key != "lock_sha256"}
    if identity(body) != initialization.get("lock_sha256"):
        raise ValueError("PMO initialization lock changed")

    loto = _loto_audit(dataset)
    production_support = {
        "actor_training_performed": bool(dataset.get("actor_training_performed")),
        "module_count_targets_emitted": bool(dataset.get("module_count_targets_emitted")),
        "binding_prototype_targets_emitted": bool(dataset.get("binding_prototype_targets_emitted")),
        "compound_stage_count": int(route_result.get("compound_stage_decisions", 0) or 0),
        "production_loto_checkpoints_found": 0,
    }
    ready = all(
        (
            production_support["actor_training_performed"],
            production_support["module_count_targets_emitted"],
            production_support["binding_prototype_targets_emitted"],
            production_support["production_loto_checkpoints_found"] == len(TASKS),
        )
    )

    old = dynamic_result["arms"]["dynamic_v0"]
    historical = {
        task: {
            "charged_calls": old[task]["charged_oracle_calls"],
            "best_score": old[task]["best_score"],
            "official_10k_auc": old[task]["auc_top10_official_10k"],
        }
        for task in TASKS
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation = {
        path: sha256_file(root / path)
        for path in (
            "src/compose_v4/experiments/pmo_route_fiber_prelaunch.py",
            "tools/pmo_route_fiber_prelaunch.py",
            "docs/PMO_ROUTE_FIBER_PILOT.md",
            "tests/test_pmo_route_fiber_prelaunch.py",
        )
    }
    contract = contract_envelope()
    payload = {
        "schema_version": SCHEMA,
        "decision": (
            "READY_FOR_ZERO_ORACLE_PRODUCTION_COMPARISON"
            if ready
            else "BLOCKED_MISSING_SPLIT_CLEAN_PRODUCTION_ROUTE_PROPOSER"
        ),
        "new_oracle_calls": 0,
        "scored_launch_authorized": False,
        "contract": contract["payload"],
        "contract_payload_sha256": contract["payload_sha256"],
        "inputs": EXPECTED_SHA256,
        "code_revision": revision,
        "implementation_sha256": implementation,
        "route_export": {
            "schema_version": dataset.get("schema_version"),
            "artifact_role": dataset.get("artifact_role"),
            "evidence": dataset.get("evidence"),
        },
        "leave_one_task_out_audit": loto,
        "production_support": production_support,
        "initialization": {
            "count": initialization.get("count"),
            "task_information_present": any(
                "task" in row or "score" in row for row in initialization.get("candidates", [])
            ),
        },
        "historical_dynamic_v0_context": historical,
        "fiber_control_adapter": {
            "mapping": "control_score=-reward",
            "round_trip_examples": [
                {
                    "reward": reward,
                    "control_score": reward_to_control_score(reward),
                    "round_trip_reward": control_score_to_reward(reward_to_control_score(reward)),
                }
                for reward in (0.0, 0.5, 1.0)
            ],
        },
        "blocker": (
            None
            if ready
            else (
                "the sealed PMO export contains primitive route supervision but no "
                "fitted actor, module-count targets, binding prototypes, or per-task "
                "production checkpoints; a selector cannot be evaluated until the "
                "actual sampler can emit split-clean route-guided programs"
            )
        ),
        "next_zero_oracle_action": (
            "fit one generic program proposer per exact held task from training-only "
            "rows, then run equal-attempt old-versus-route production sampling on "
            "the immutable 16-source initialization before any PMO score is observed"
        ),
    }
    return {"payload": payload, "payload_sha256": identity(payload)}
