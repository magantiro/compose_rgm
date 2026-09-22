"""Build and seal the fa7_0 support-expansion contract from the working tree.

The contract is DERIVED, never hand-edited: every `runtime_inputs_sha256` entry is
recomputed from the file on disk, so a contract can only ever address the tree it
was built in.  Re-running this after any source change is how the pin stays honest,
and the launcher's own `_local_task` re-checks every hash plus `git diff` before a
launch, so a stale contract fails closed rather than running the wrong code.

Scientific fields are inherited from the authorized fa7 delta=0.6 arm
(`configs/t4_region_repair_rescue_fa7_d06_v1.json`, payload
`91101214587acb98ef39ba5511a21df379f33ac3d5f8d6f2c7f035219159bc1e`) so the docking
box, docking seed, evaluator hashes, controller seed, seed SMILES, delta, batch,
parent policy and value penalty are byte-identical to the arm this cell was
authorized under.  What this file adds is exactly the two executable mechanisms and
the single-cell narrowing; it inherits, it does not re-derive.

    python3 tools/build_t4_fa7_0_support_expansion_contract.py --status draft
    python3 tools/build_t4_fa7_0_support_expansion_contract.py --status authorized
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.continuation_profile import sha256_file  # noqa: E402
from compose_v4.experiments.t4_matched_pilot import seal, unseal  # noqa: E402

PARENT_CONTRACT = "configs/t4_region_repair_rescue_fa7_d06_v1.json"
PARENT_PAYLOAD_SHA256 = "91101214587acb98ef39ba5511a21df379f33ac3d5f8d6f2c7f035219159bc1e"
TARGET = "configs/t4_fa7_0_support_expansion_v1.json"
AUTHORIZED_STATUS = "AUTHORIZED_BY_OWNER_FOR_FA7_0_SUPPORT_EXPANSION"
DRAFT_STATUS = "DRAFT_PENDING_OWNER_AUTHORIZATION"

#: Every file whose bytes can change what this run computes.  The two law modules,
#: the fallback, the expansion policy and BOTH app files are pinned: a mechanism
#: that is not in this list is free to drift under a contract that claims to run it.
RUNTIME_INPUTS = (
    "diagnostics/t4_held_target_distillation_quality_v1/fa7_checkpoint.json",
    "docs/GENMOL_T4_SEEDS.json",
    "modal_apps/t4_fa7_0_support_expansion_app.py",
    "modal_apps/t4_fa7_0_support_expansion_base_app.py",
    "src/compose_v4/control/bridge_region_law.py",
    "src/compose_v4/control/complete_region_program.py",
    "src/compose_v4/control/completion_law_contract.py",
    "src/compose_v4/control/dynamic_program_synthesis.py",
    "src/compose_v4/control/fiber_control.py",
    "src/compose_v4/control/progressive_structured_sampler.py",
    "src/compose_v4/control/region_law_contract.py",
    "src/compose_v4/control/replace_completion_law.py",
    "src/compose_v4/control/route_distilled_goal_expert.py",
    "src/compose_v4/control/structural_subgoal_policy.py",
    "src/compose_v4/control/structural_subgoal_realizer.py",
    "src/compose_v4/control/zero_support_fallback.py",
    "src/compose_v4/experiments/t4_docking_adapter.py",
    "src/compose_v4/experiments/t4_fiber_campaign.py",
    "src/compose_v4/experiments/t4_integrated_route_fiber.py",
    "src/compose_v4/experiments/t4_integrated_route_fiber_v2.py",
    "src/compose_v4/experiments/t4_support_expansion.py",
)


#: The owner's total authorization for this cell, and every call already charged
#: against it by a launch of this arm that died. The ceiling is DERIVED from these
#: so the arithmetic cannot drift out of step with the record.
AUTHORIZED_TOTAL = 248
PRIOR_LAUNCHES = {
    "cancelled_first_launch_of_this_arm": 1,
    "preempted_second_launch_of_this_arm": 1,
    "keyerror_third_launch_of_this_arm": 1,
}


def build(status: str) -> dict:
    parent = unseal(ROOT / PARENT_CONTRACT)
    ceiling = AUTHORIZED_TOTAL - sum(PRIOR_LAUNCHES.values())
    inherited = {
        key: parent[key]
        for key in (
            "delta",
            "support",
            "docking_box",
            "docking_seed",
            "evaluator_sha256",
            "parents",
            "parent_explore",
            "batch",
            "exploration",
            "expert_floor_rounds",
            "route_scale_floor_rounds",
            "value_penalty",
            "phase_poll_seconds",
            "proposal_wait_seconds",
            "query_wait_seconds",
            "primary_baselines",
            "primary_model_output",
            "reported_ivg_source",
            "round_lock_policy",
        )
    }
    proposal = json.loads(json.dumps(parent["proposal"]))
    proposal["shallow"]["completion_law"] = "free_gate_margin_v1"
    cells = [row for row in parent["cells"] if row["cell"] == "fa7_0"]
    if len(cells) != 1:
        raise SystemExit("the parent contract does not carry exactly one fa7_0 cell")

    payload = {
        "schema_version": "t4_fa7_0_support_expansion_contract_v1",
        "status": status,
        **inherited,
        "cells": cells,
        "proposal": proposal,
        # DERIVED, never a literal: the authorized total less every call the prior
        # launches of this arm actually charged. Each dead launch charged exactly one
        # root docking, and they are subtracted rather than forgiven because the
        # authorization says "not one more".
        "charged_calls_per_cell": ceiling,
        "total_charged_call_ceiling": ceiling,
        "total_new_charged_call_ceiling": ceiling,
        # ---- The expansion ----
        # A ladder step is DRAWS PER LANE and is realised as parallel replicate
        # workers at the lane's own base draw count, never as one deeper worker.
        # 960 + 1920 + 3840 = 6720 is both the ladder sum and the per-event cap, so
        # the cap is the ladder rather than a second, looser bound.
        "support_expansion": {
            "draw_ladder": [960, 1920, 3840],
            "lanes": ["shallow", "anchored_replacement"],
            "zero_support_fallback": True,
            "stop_at_distinct_eligible": 4,
            "max_extra_draws_per_event": 6720,
            "wall_seconds": 7200.0,
        },
        "isolation": {
            "app": "compose-t4-fa7-0-support-expansion",
            "volume": "compose-t4-fa7-0-support-expansion",
            "output_prefix": "/fa7_0_support_expansion",
            "wrapper": "modal_apps/t4_fa7_0_support_expansion_app.py",
            "base_app": "modal_apps/t4_fa7_0_support_expansion_base_app.py",
            "guarantee": (
                "app, volume and output namespace differ from every other T4 arm, and "
                "the run id is content-addressed over the contract payload and the code "
                "revision, so this run cannot inherit the stale query lock that ended "
                "the predecessor attempt on resume"
            ),
        },
        "mechanism": {
            "completion_law": {
                "name": "replace-completion law (endpoint free-gate margin)",
                "field": "proposal.shallow.completion_law",
                "defect": (
                    "the completion half of segment_replace drew a length uniform on "
                    "1..8 and a uniform C/N/O chain, while every measured eligible "
                    "completion inserts one or two atoms, so most draw mass landed "
                    "where nothing is eligible"
                ),
                "measured": (
                    "480 draws per arm, arms differing only in this field: fa7_1 7->13, "
                    "braf_2 4->6, 5ht1b_0 20->35 eligible endpoints; no control "
                    "regressed; 6/6 mutations killed"
                ),
                "absent_means": "law=None, byte-identical to the historical draw",
                "source_branch": "t4-fa7-0-reachability-20260921 at 01baa8a6",
            },
            "support_expansion": {
                "name": "bounded support expansion on an empty candidate pool",
                "field": "support_expansion",
                "defect": (
                    "select_batch returns empty exactly when candidates is empty, so "
                    "the shipped loop's `if not selected` branch IS the cell's terminal "
                    "condition; fa7_0 terminated there at charged_calls=1, the seed's "
                    "own docking call, with 247 authorized calls never issued"
                ),
                "stages": [
                    "zero_support_fallback: excise each bridge-separated region through "
                    "the production executor and gate the EXECUTED endpoint directly, "
                    "which never enters the goal-abstraction layer where expand rebinds "
                    "a structural goal and gates what instantiate_goal returns",
                    "draw_ladder: escalating parallel replicates of the primary lanes",
                ],
                "charges_nothing": (
                    "expansion returns proposal records that are locked and docked "
                    "through the unchanged round path; no query is retried, no receipt "
                    "replaced and no call backfilled"
                ),
                "trigger_is_terminal_not_threshold": (
                    "every round that wrote a lock had at least one eligible candidate "
                    "and so never entered the branch, which is why the expansion is "
                    "structurally unreachable on any row that reached its budget"
                ),
                "consumption_check": (
                    "assert_support_expansion_is_consumed gates the terminal publish, so "
                    "a contract that declares an expansion and a runtime that does not "
                    "perform one cannot end the cell"
                ),
            },
        },
        "frozen_from": {
            "controller_contract": PARENT_CONTRACT,
            "controller_contract_payload_sha256": PARENT_PAYLOAD_SHA256,
            "allowed_adapter_changes": [
                "cells narrowed to fa7_0 alone",
                "proposal.shallow.completion_law added",
                "support_expansion block added",
                "separate app, base app, volume and output namespace",
                "proposal worker timeout 1800 -> 3600 s and cell timeout 6 -> 20 h, "
                "because an expansion event costs wall clock and a 480-draw worker "
                "with the completion law resolved is ~10 min on a fast core",
            ],
            "forbidden_changes": (
                "delta, proposal widths, the other proposal lanes, FiberControl "
                "features or fitting, expert-floor schedule, parent selection, "
                "exploration, endpoint gates, docking box, docking seed, evaluator, "
                "controller seed, retries or replacements"
            ),
        },
        "prior_charged_calls": {
            "fa7_0": 2,
            "cancelled_first_launch_of_this_arm": {
                "run_id": "837bb8c8dcca530f8e6538f0845a2866fe2b7ac00fcce85f9935e93ba9341578",
                "contract_payload_sha256": (
                    "b191a40b371328ef146c0ccc4f04616ad7814a277f0a7ce3470e6ab6ba25a89b"
                ),
                "charged_calls": 1,
                "what_it_charged": "the root docking of the seed, round 0",
                "why_cancelled": (
                    "the zero-support fallback labelled its records "
                    "proposal_lane='zero_support_fallback', which "
                    "t4_integrated_route_fiber._experts validates against the frozen "
                    "expert vocabulary and REJECTS, so attach_features would have raised "
                    "the first time the fallback produced an eligible endpoint -- a "
                    "failure that fires only on success. Cancelled at 1 charged call "
                    "rather than left to fail mid-budget and strand an unfinished lock"
                ),
                "subtracted_from_this_arms_ceiling": True,
            },
            "preempted_second_launch_of_this_arm": {
                "run_id": "4321b8b1edc8e73d6e99c04dfe557703e5e1ba30f2c187902229efb90880f82a",
                "contract_payload_sha256": (
                    "e3482c17c9966f01fefb32ea03ab8b6cf09db273fc8bfefeedb0188ecc91bea9"
                ),
                "charged_calls": 1,
                "what_it_charged": "the root docking of the seed, round 0, score -8.30",
                "how_it_died": (
                    "MEASURED from the app's own logs: 'Container terminated due to "
                    "preemption. Your Function will be restarted with the same input.' "
                    "The restart met its own round_000 lock with no checkpoint -- the "
                    "root lock is published BEFORE the root docking and the first "
                    "checkpoint was only written at the END of round one -- and "
                    "_resume_state correctly refused it as an ambiguous scored retry. "
                    "This arm's support expansion widened that window from minutes to "
                    "tens of minutes, turning a pre-existing hazard into the likely "
                    "outcome. Modal restarts a preempted container with the same input "
                    "EVEN AT retries=0"
                ),
                "what_it_reached_before_dying": (
                    "the logs show the normal round map, then the fallback map, then "
                    "the ladder map TWICE -- so the support expansion fired on the "
                    "empty pool and escalated to its second ladder step, which is the "
                    "behaviour this arm exists to produce"
                ),
                "repair": (
                    "the completed root call is now checkpointed before the round loop. "
                    "Round 0 is a completed round -- its one call is charged, scored and "
                    "recorded -- so checkpointing it admits no ambiguity: nothing is "
                    "re-docked and no uncertain call becomes free. The fail-closed guard "
                    "is unchanged"
                ),
                "subtracted_from_this_arms_ceiling": True,
            },
            "keyerror_third_launch_of_this_arm": {
                "run_id": "727d9db5b17855fbb7980945aa08ed2f52e589e9f0802132715017e25cf45105",
                "contract_payload_sha256": (
                    "fa4edee257da0c98f8f830987b7215bccdeccdd5d52987afd11ae532096c2bd6"
                ),
                "charged_calls": 1,
                "what_it_charged": "the root docking of the seed, round 0, score -8.8",
                "what_it_achieved": (
                    "THE MECHANISM WORKED. Logged: 'SUPPORT EXPANSION round=1 "
                    "stop=reached_target eligible=4 fallback=0 attempts=2 draws=2880'. "
                    "An empty candidate pool -- the state that ends the cell in the "
                    "shipped loop -- produced FOUR eligible endpoints from two ladder "
                    "steps. The root checkpoint added after the preempted launch also "
                    "held, so that repair is confirmed"
                ),
                "how_it_died": (
                    "KeyError('proposal_experts') while building the query rows, AFTER "
                    "the expansion had found its four endpoints, discarding all four. "
                    "The normal path gets that key from merge_expert_pools, which the "
                    "expansion path does not use because it deduplicates by endpoint "
                    "itself; expand's own records carry only proposal_lane"
                ),
                "repair": (
                    "normalize_expansion_records restores the one key the merge would "
                    "have supplied, from the lane that produced the record. A dry pass "
                    "now drives the whole expansion round body end to end -- both "
                    "stages, attach_features, select_batch, the query rows and the lock "
                    "-- and checks the selected rows against every key run_cell "
                    "subscripts off `row`, derived from run_cell's own source"
                ),
                "subtracted_from_this_arms_ceiling": True,
            },
            "note": (
                "the panel run charged 1 (the seed's own docking score, best -7.5) and "
                "the predecessor support-expansion arm charged 0; the 248 ceiling is "
                "the arm's authorized 250 less the measured prior calls, unchanged from "
                "the parent contract"
            ),
        },
        "authorization": {
            "scope": "fa7_0 at delta=0.6 only, up to 248 charged docking calls",
            "sentence": (
                "Run a support-expanded search on ONE cell -- fa7_0 at delta 0.6 -- and "
                "keep going until it produces an ELIGIBLE molecule with a docking score, "
                "or the budget is spent. AUTHORIZED SPEND: up to 248 charged docking "
                "calls, cell fa7_0 only. Not one more, no other cell."
            ),
            "granted_by": "owner, task brief 2026-09-22",
            "enforced_by": [
                "charged_calls_per_cell and total_charged_call_ceiling are DERIVED as "
                "the authorized 248 less every call the prior launches of this arm "
                "charged, so the arm's lifetime spend cannot exceed 248",
                "cells carries fa7_0 alone, and the wrapper refuses any other list",
                "the ledger binds on state.budget, which is initialised from "
                "charged_calls_per_cell and decremented once per docked query",
            ],
        },
        "claim_boundary": (
            "SEPARATE, VERSIONED SUPPORT-EXPANSION ARM for fa7_0 at delta=0.6, under its "
            "own contract identity, app and volume. Its proposal law DIFFERS from the v1 "
            "panel arm in two executable fields, and its round loop expands support on an "
            "empty candidate pool instead of terminating, so no result from it may be "
            "spliced into an unchanged-v1 panel table without being named as a "
            "support-expansion phase. Report benchmark eligibility and chemical "
            "plausibility as two numbers: an endpoint that deletes the amidine passes "
            "every threshold and is not a result anyone should defend."
        ),
        "trigger_rule": {
            "rule": (
                "expansion triggers on MEASURED candidate exhaustion only -- the empty "
                "pool the shipped loop would have ended on -- never on target identity "
                "and never on losing to the IVG comparator"
            ),
            "satisfied": True,
            "measured": (
                "fa7_0 terminated live at status=candidate_exhaustion with "
                "charged_calls=1 in round zero"
            ),
        },
        "offline_evidence": {
            "zero_oracle_calls": True,
            "kernel": "python 3.11.13 / rdkit 2024.03.5 / numpy 1.26.4 / torch 2.4.0",
            "note": (
                "measured yields are recorded in "
                "diagnostics/t4_fa7_0_support_expansion_offline_v1.json; they are "
                "proposal-side evidence and bound nothing about docking"
            ),
        },
        "runtime_inputs_sha256": {
            relative: sha256_file(ROOT / relative) for relative in sorted(RUNTIME_INPUTS)
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", choices=["draft", "authorized"], required=True)
    args = parser.parse_args()
    status = AUTHORIZED_STATUS if args.status == "authorized" else DRAFT_STATUS
    payload = build(status)
    seal(ROOT / TARGET, payload)
    # `seal` returns the FILE hash, not the payload hash -- the two differ and this
    # print labelled the wrong one on the first build. The payload hash is what the
    # launcher, the worker identity check and every citation of this arm mean.
    envelope = json.loads((ROOT / TARGET).read_text())
    revision = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    print(
        f"{TARGET}\n"
        f"  status          {status}\n"
        f"  payload_sha256  {envelope['payload_sha256']}\n"
        f"  file_sha256     {sha256_file(ROOT / TARGET)}\n"
        f"  built_at_rev    {revision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
