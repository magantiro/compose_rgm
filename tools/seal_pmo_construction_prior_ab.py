#!/usr/bin/env python3
"""Seal (or re-seal) the two construction-prior A/B arm contracts and their receipt.

The arms are matched by CONSTRUCTION rather than by inspection: one common payload is
built once and the only thing that varies between the two files is the ``arm`` block,
so "the arms differ in exactly one key" cannot drift into being false through an edit
to one file that was not mirrored into the other.

Re-run this after ANY edit to a pinned source file.  A contract whose
``implementation_sha256`` addresses bytes that no longer exist is an unresolvable pin,
and the launcher refuses to spawn on one -- which is the correct behaviour and also the
reason this has to be one command rather than a remembered procedure.

It refuses to touch anything once a launch receipt exists: after a scored run the
contracts are the record of what ran, and re-pinning them would rewrite that record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFF = "configs/pmo_ab_b_construction_prior_off_contract_v1.json"
ON = "configs/pmo_ab_b_construction_prior_on_contract_v1.json"
BASE = "configs/pmo_ab_b_memory_contract_v1.json"
RECEIPT = "diagnostics/pmo_construction_prior_ab/scored_authorization_v1.json"
LAUNCH = "diagnostics/pmo_construction_prior_ab/launch_receipt_v1.json"
DECISION_RULE_COMMIT = "ac077f7ee45d55d82ffcb38378ef9cce1a05b2ee"

CHECKPOINT_SHA256 = "24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4"
CHECKPOINT_STATUS = "DEVELOPMENT_ONLY_CONSTRUCTION_PRIOR_QUALIFICATION"

#: Added to the deployed arm's pins because each one decides WHICH MOLECULE the shallow
#: lane proposes.  Anything on the scoring path is pinned, on the same rule that put the
#: oracle adapter in: the difference between a number the oracle produced and a number
#: something else produced must not be free to drift.
ADDED_PINS = (
    "src/compose_v4/control/pmo_construction_prior.py",
    "src/compose_v4/control/learned_successor_prior.py",
    "src/compose_v4/control/dynamic_program_synthesis.py",
    "src/compose_v4/control/dynamic_program_synthesis_v21.py",
    "scripts/evaluate_tracelet_rollouts.py",
)

PRIOR_SPEC = {
    "schema_version": "pmo_construction_prior_spec_v1",
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "corpus_scope_hash": "3721d69851110fdd",
    "status": CHECKPOINT_STATUS,
    "floor": 0.05,
    "temperature": 1.0,
}

OFF_ARM = {
    "name": "B_cp_off",
    "comparison": "pmo_construction_prior_v1_control",
    "enable_online_memory": True,
    "construction_prior": None,
    "what_differs": (
        "NOTHING relative to the deployed B arm. `construction_prior` is ABSENT, which is "
        "the only byte-identical off for this seam: execute_task omits the key from "
        "optimizer_kwargs entirely rather than setting it to None, and "
        "synthesize_dynamic_program keeps v1's exact RNG stream when successor_prior is "
        "None. A 'uniform prior object' would NOT be a no-op -- any law object consumes "
        "rng.random where the unlawed draw consumes rng.permutation, so it reproduces the "
        "support and not the draws."
    ),
}

ON_ARM = {
    "name": "B_cp_on",
    "comparison": "pmo_construction_prior_v1_treated",
    "enable_online_memory": True,
    "construction_prior": PRIOR_SPEC,
    "what_differs": (
        "ONLY the construction prior. Same initialization lock, same controller seed "
        "20260920, same budget, same executor, same oracle, same allocator, same macro "
        "scheduling, same exploration policy, same online memory. The prior re-ranks the "
        "legal candidate list of the CONSTRUCTION draw by a task-independent trained "
        "editing law, with a 0.05 support floor so it is a re-ranking and never a filter; "
        "it reads no oracle, no task identity, no target and no score."
    ),
}


def _seal(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _common_payload() -> dict:
    base = json.loads((ROOT / BASE).read_text())["payload"]
    pins = sorted(set(base["implementation_sha256"]) | set(ADDED_PINS))
    return {
        "schema_version": "pmo_population_controller_v1_scored_contract",
        "authority": (
            f"{RECEIPT} is the only source of runtime scoring authority for this arm. The "
            "worker binds arm -> payload sha256 against that receipt and refuses any "
            "payload it does not name."
        ),
        "budget": {
            "automatic_retries": 0,
            "backfill": False,
            "candidate_calls_per_task": 234,
            "charged_calls_per_task": 250,
            "charged_calls_total": 250,
            "cpu_per_worker": 1,
            "gpu": False,
            "initialization_calls_per_task": 16,
            "max_rounds": 64,
            "queries_per_round": 16,
        },
        "checkpoint_publication_restriction": {
            "checkpoint": "ringcore_a7546e2_best.pt",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "status": CHECKPOINT_STATUS,
            "forbidden_by": "configs/ringcore_v1_checkpoint_selection.json",
            "forbidden_because": (
                "selected by hazard-inclusive GM loss, an explicitly forbidden criterion"
            ),
            "may_become_a_published_number": False,
            "applies_to_both_arms": (
                "The control carries the restriction too: it is the COMPARISON that is a "
                "development qualification, not merely the treated arm."
            ),
        },
        "controller_contract": base["controller_contract"],
        "controller_contract_sha256": base["controller_contract_sha256"],
        "corrected_source_capsule_manifest": base["corrected_source_capsule_manifest"],
        "corrected_worker_app": base["corrected_worker_app"],
        "corrected_worker_path": base["corrected_worker_path"],
        "development_only": True,
        "implementation_sha256": {rel: _file_sha256(rel) for rel in pins},
        "modal_launch_authorized": False,
        "oracle": base["oracle"],
        "oracle_calls_authorized": 0,
        "prior_scope_measured": {
            "prior_aware": [
                (
                    "shallow_program_channel, fresh-synthesis branch of "
                    "DynamicProgramOptimizer._mutate"
                ),
                (
                    "shallow_program_channel, warm-memory branch of "
                    "pmo_online_memory.memory_channel_proposal"
                ),
            ],
            "not_prior_aware": [
                "structured_program_channel (routes through dynamic_program_synthesis_v1)",
                "joint_dependency_region_jump",
                "recombination",
                "ProgramOptimizer._mutate current-state-edit branch",
                "the cold-start bootstrap pool",
            ],
            "note": (
                "This is a PARTIAL-COVERAGE mechanism by construction. A null result is a "
                "null about the shallow construction lane, not about a chemical prior in "
                "general."
            ),
        },
        "promotion": {
            "may_become_a_published_number": False,
            "decision_rule": (
                "diagnostics/pmo_construction_prior_ab/predeclared_decision_rule_v1.json, "
                f"committed at {DECISION_RULE_COMMIT} before this wiring existed and "
                "before any charged call"
            ),
        },
        "runtime": base["runtime"],
        "scientific_question": (
            "Does the learned, task-independent chemical prior over the CONSTRUCTION draw "
            "-- which measurably halves closed-loop manifold drift with zero oracle calls "
            "-- translate into a better scored PMO search, or is off-manifold drift an "
            "insufficient explanation of basin discovery?"
        ),
        "scored_launch_authorized": False,
        "status": "FROZEN_FAIL_CLOSED_AUTHORITY_IS_THE_SEALED_RECEIPT",
        "task_roles": {
            "celecoxib_rediscovery": (
                "development A/B substrate: the task whose matched 1k A/B is on record, "
                "whose manifold drift was measured, and whose plateau this prior is "
                "hypothesised to address"
            )
        },
        "tasks": ["celecoxib_rediscovery"],
        "wall_clock_confound": {
            "mechanism": (
                "each lane stops on 16 candidates or config.wall_seconds = 45.0, "
                "whichever first"
            ),
            "measured_construction_cost_ratio": 1.91,
            "measured_shallow_lane_seconds_in_deployed_B": {
                "median": 10.97, "max": 15.71, "budget": 45.0
            },
            "prediction": "the shallow lane stops on the candidate cap in BOTH arms",
            "verification": (
                "realized per-round shallow-lane seconds and pool sizes are compared "
                "across arms after the run"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-after-launch", action="store_true")
    args = parser.parse_args()
    if (ROOT / LAUNCH).exists() and not args.allow_after_launch:
        raise SystemExit(
            "a launch receipt exists: the contracts are now the record of what ran and "
            "re-pinning them would rewrite that record"
        )

    common = _common_payload()
    digests = {}
    for arm, relative in ((OFF_ARM, OFF), (ON_ARM, ON)):
        payload = {**common, "arm": arm}
        envelope = {"payload": payload, "payload_sha256": _seal(payload)}
        (ROOT / relative).write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        digests[arm["name"]] = envelope["payload_sha256"]

    receipt_path = ROOT / RECEIPT
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt.update({
        "schema_version": "pmo_construction_prior_ab_authorization_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": receipt.get(
            "authorized_at_utc", datetime.now(timezone.utc).isoformat()
        ),
        "resealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorizing_instruction": (
            "Owner instruction: freeze every PMO architecture change except the "
            "construction prior; run a matched celecoxib development A/B at 250 charged "
            "calls, deployed B versus B plus the full construction-lane prior, identical "
            "initialization and controller otherwise; the checkpoint may be used only as a "
            "development qualification while its publication restriction remains; do not "
            "modify donor allocation, 100-init, macro scheduling or exploration policy."
        ),
        "authorized_oracle_calls": {
            "per_arm": 250,
            "arms": 2,
            "total": 500,
            "hard_limit_note": (
                "500 charged calls and not one more. Each arm is a single Modal call with "
                "retries=0; the ledger budget is read from the arm's own sealed payload, "
                "and execute_task takes charged_calls_per_task as a REQUIRED argument with "
                "no default, so no runtime can inherit a budget it was not authorized for."
            ),
        },
        "arms": {
            name: {
                "contract": relative,
                "payload_sha256": digests[name],
                "charged_calls_per_task": 250,
                "tasks": ["celecoxib_rediscovery"],
                "construction_prior": (PRIOR_SPEC if name == "B_cp_on" else None),
            }
            for name, relative in (("B_cp_off", OFF), ("B_cp_on", ON))
        },
        "matched": (
            "Both payloads are built from ONE common dictionary, so they differ in exactly "
            "one top-level key, `arm`, by construction rather than by inspection. Within "
            "`arm` they differ only in `construction_prior` plus three descriptive strings."
        ),
        "development_only": {
            "may_become_a_published_number": False,
            "checkpoint": "ringcore_a7546e2_best.pt",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "label": CHECKPOINT_STATUS,
            "forbidden_by": "configs/ringcore_v1_checkpoint_selection.json",
            "forbidden_because": (
                "the checkpoint was selected by hazard-inclusive GM loss, an explicitly "
                "forbidden criterion for frozen results"
            ),
            "applies_to_both_arms": (
                "The control arm carries the restriction too. The COMPARISON is a "
                "development qualification of the mechanism."
            ),
        },
        "predeclared_decision_rule": {
            "path": "diagnostics/pmo_construction_prior_ab/predeclared_decision_rule_v1.json",
            "commit": DECISION_RULE_COMMIT,
            "declared_before": "the wiring change, the deploy, and any charged call",
        },
        "frozen_in_this_experiment": [
            "donor allocation", "initialization count (16)", "macro scheduling",
            "exploration policy", "parent allocator", "online memory mechanism",
            "executor", "oracle adapter", "controller seed 20260920",
        ],
    })
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sealed": digests, "receipt": RECEIPT}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
