"""Build and seal the all-23-task PMO diagnostic rung contract.  Launches nothing.

Run offline.  It constructs no ``tdc.Oracle``, opens no network connection and charges
no oracle call; the 23 task names are re-derived from the committed registry
verification artifact, whose own check is exact membership in ``tdc.metadata
.oracle_names`` rather than ``Oracle(name=...)`` construction (which fuzzy-matches at
threshold 0.8 and would resolve a near miss to a DIFFERENT oracle, silently).

The contract it writes is fail-closed: ``scored_launch_authorized`` is false,
``oracle_calls_authorized`` is 0 and ``status`` is PENDING.  Only the owner can
authorize spend, and only through a separate receipt that pins this payload's digest.

Deliberately NOT circular: the contract does not pin the authorization receipt.  The
receipt pins the contract.  A prior preparer wrote the receipt before the manifest that
pinned it while the receipt carried a moving ``authorized_at_utc``, so the launcher's
check could never pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.control.docking_value import identity  # noqa: E402
from compose_v4.experiments.pmo_diagnostic_rung_v1 import (  # noqa: E402
    CONTRACT,
    READOUT_CHECKPOINTS,
    SCHEMA,
    SPIN_UP_CALLS,
    SUITE,
    asset_backed_plan,
    rounds_for_budget,
    validate_budget,
    verify_suite_against_registry,
)
from compose_v4.experiments.pmo_population_v1 import (  # noqa: E402
    CHECKPOINTS,
    INIT_COUNT,
    QUERIES_PER_ROUND,
    configuration,
)

ASSET_ROOT = "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets"
SOURCE_CONTRACT = "configs/pmo_population_controller_v1.json"

# Every file on the path from a proposed molecule to a recorded number.  A file that
# decides whether a value is the oracle's output or PyTDC's swallowed default belongs
# here (that is why `pmo_oracle_assets.py` is pinned), as does the ledger that decides
# when the budget binds.
IMPLEMENTATION_FILES = (
    "modal_apps/pmo_diagnostic_rung_v1_app.py",
    "src/compose_v4/control/adaptive_program_optimizer.py",
    "src/compose_v4/control/bootstrap_pool_continuity.py",
    "src/compose_v4/control/dynamic_program_synthesis_v21.py",
    "src/compose_v4/control/pmo_credit.py",
    "src/compose_v4/control/pmo_population_controller.py",
    "src/compose_v4/control/program_campaign.py",
    "src/compose_v4/control/program_task.py",
    "src/compose_v4/experiments/pmo_diagnostic_rung_v1.py",
    "src/compose_v4/experiments/pmo_oracle_assets.py",
    "src/compose_v4/experiments/pmo_population_v1.py",
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(root: Path, budget: int) -> dict:
    registry = verify_suite_against_registry(root)
    declared = validate_budget(budget)
    assets = asset_backed_plan(root, ASSET_ROOT)
    source = json.loads((root / SOURCE_CONTRACT).read_text())["payload"]

    runnable = assets["runnable_tasks"]
    blocked = assets["blocked_pending_asset_pin"]

    return {
        "schema_version": SCHEMA,
        "scientific_question": (
            "We hold valid 250-call readings on three PMO tasks only (gsk3b 0.21, "
            "perindopril_mpo 0.486, celecoxib_rediscovery 0.196). Is COMPOSE weak across "
            "PMO generally, or badly matched to those three landscapes? A short rung "
            "across all 23 tasks separates the two and localizes the failure per task."
        ),
        "what_this_rung_is_not": (
            "Not a benchmark claim. The budget is far below the official 10000 calls, and "
            "pmo_top_ten_auc trapezoids up from (0,0), so a rung AUC is structurally "
            "depressed relative to a published figure and is reported within-rung only."
        ),
        "tasks": list(SUITE),
        "task_registry_verification": {
            "path": "diagnostics/pmo_task_registry_verification_v1.json",
            "sha256": _sha256(root / "diagnostics/pmo_task_registry_verification_v1.json"),
            **registry,
        },
        "budget": {
            "charged_calls_per_task": budget,
            "charged_calls_ceiling": budget * len(SUITE),
            "charged_calls_for_currently_runnable_tasks": budget * len(runnable),
            "initialization_calls": INIT_COUNT,
            "queries_per_round": QUERIES_PER_ROUND,
            "rounds": declared["rounds"],
            "adaptive_rounds": declared["adaptive_rounds"],
            "spin_up_calls": SPIN_UP_CALLS,
            "adaptive_calls": declared["adaptive_calls"],
            "adaptive_call_fraction": declared["adaptive_call_fraction"],
            "calls_include_initialization": True,
            "automatic_retries": 0,
            "backfill": False,
            "seeds": 1,
            "budget_authority": (
                "execute_task requires charged_calls_per_task and cross-checks it against "
                "this contract; the round count is derived from the budget by "
                "rounds_for_budget, so neither the ledger nor the campaign shape can be "
                "inherited from a module constant."
            ),
        },
        "readouts": {
            "primary": [f"best_at_{k}" for k in READOUT_CHECKPOINTS],
            "auc": {
                "key": "auc_top10_at_budget",
                "budget": budget,
                "comparability": "within-rung only; never against a published 10000-call AUC",
                "structural_discount_note": (
                    "a run holding constant top-10 level c scores c*(1 - frequency/(2*budget))"
                ),
            },
            "per_task_diagnostics": [
                "nonzero_score_rate",
                "improvement_probability_by_program_family",
                "improvement_probability_by_edit_scale",
                "productive_lineage_depth",
                "realization_refusal_causes",
                "archive_churn",
                "successful_basin_diversity",
            ],
        },
        "controller": {
            "source": "compose_v4.experiments.pmo_population_v1.configuration",
            "rationale": (
                "the strongest currently VALID controller: the same PmoPopulationController "
                "geometry that produced the only valid PMO readings on record"
            ),
            "seed": source["controller"]["seed"],
            "serialized": json.loads(json.dumps(asdict(configuration(source["controller"]["seed"])))),
            "optimizer_type": "compose_v4.control.pmo_population_controller.PmoPopulationController",
            "initial_batch_fn": (
                "compose_v4.control.dynamic_program_synthesis_v21.initial_dynamic_program_batch_v21"
            ),
        },
        "initialization": source["initialization"],
        "joint_checkpoint": source["joint_checkpoint"],
        "oracle": {
            "adapter": "native PyTDC Oracle(name=task) wrapped in AssetPinnedOracle",
            "environment": source["oracle"]["environment"],
            "source_files": source["oracle"]["source_files"],
            "direction": "maximize",
            "range": [0, 1],
            "prescreen": False,
            "calls_include_initialization": True,
            "assets": assets["tasks"],
            "asset_root": ASSET_ROOT,
            "capsule_contents": assets["capsule_contents"],
            "asset_policy": (
                "an asset-backed task is REFUSED unless its pinned pickle is present with "
                "the digest recorded here, and its positive control must pass before the "
                "first charged call; a blocked task charges zero"
            ),
            "positive_control_before_first_charged_call": True,
        },
        "task_partition": {
            "pure_rdkit_no_asset": assets["pure_rdkit_tasks"],
            "asset_backed": sorted(assets["tasks"]),
            "runnable_now": runnable,
            "blocked_pending_asset_pin": blocked,
            "blocked_tasks_charge_zero": True,
        },
        "reference_molecule_policy": {
            "prescreen": False,
            "reference_molecules_in_initialization": False,
            "reference_molecules_in_archive": False,
            "reference_molecules_in_selection": False,
            "initialization_is_task_independent": True,
            "evidence": (
                "the shared initialization block is loaded through _load_initialization, "
                "which raises if any candidate row carries a 'score' or 'task' key"
            ),
        },
        "implementation_sha256": {
            path: _sha256(root / path) for path in IMPLEMENTATION_FILES
        },
        "worker_app": "compose-pmo-diagnostic-rung-23task-v1",
        "worker_path": "modal_apps/pmo_diagnostic_rung_v1_app.py",
        "scored_launch_authorized": False,
        "modal_launch_authorized": False,
        "oracle_calls_authorized": 0,
        "status": "PENDING_OWNER_AUTHORIZATION",
        "authorization_note": (
            "Only the owner authorizes oracle spend. This payload is fail-closed; a "
            "separate receipt naming this payload_sha256 is the sole runtime authority."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--budget", type=int, required=True,
                        help="charged oracle calls per task")
    parser.add_argument("--write", action="store_true",
                        help="write the sealed contract; otherwise print it")
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    payload = build_payload(root, arguments.budget)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}

    if arguments.write:
        target = root / CONTRACT
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(envelope, sort_keys=True, indent=1) + "\n")
        # Re-read and re-verify: a contract that cannot be reopened by its own loader is
        # not sealed, it is merely written.
        from compose_v4.experiments.pmo_diagnostic_rung_v1 import load_contract

        load_contract(root)
        print(f"wrote {CONTRACT}")

    summary = {
        "payload_sha256": envelope["payload_sha256"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "charged_calls_per_task": payload["budget"]["charged_calls_per_task"],
        "charged_calls_ceiling_all_23": payload["budget"]["charged_calls_ceiling"],
        "charged_calls_runnable_now": payload["budget"][
            "charged_calls_for_currently_runnable_tasks"
        ],
        "runnable_now": len(payload["task_partition"]["runnable_now"]),
        "blocked_pending_asset_pin": payload["task_partition"]["blocked_pending_asset_pin"],
        "rounds": payload["budget"]["rounds"],
        "adaptive_rounds": payload["budget"]["adaptive_rounds"],
        "status": payload["status"],
        "scored_launch_authorized": payload["scored_launch_authorized"],
        "oracle_calls_authorized": payload["oracle_calls_authorized"],
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
