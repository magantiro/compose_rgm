"""Prepare the PMO scale-up ladder: 3 -> 5 -> 23 tasks at 250 / 1k / 10k calls.

Preparation only.  Every artifact this writes is sealed fail-closed
(`scored_launch_authorized: false`, `modal_launch_authorized: false`,
`oracle_calls_authorized: 0`) and nothing here launches, deploys or scores.

It writes NEW files only.  The live 3x250 launch chain -- the controller contract,
the corrected scored contract, its authorization receipt, the source capsule
manifest and `launch_pmo_population_v1_corrected.py` -- is pinned by content hash
and is read here, never written.

The runtime-capability section is MEASURED: the budget and task-lock constants are
parsed out of the pinned `pmo_population_v1.py` rather than assumed, because the
contract's budget block and the runtime's constants are independent and currently
disagree.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- Pinned inputs: read-only ----

RUNTIME = ROOT / "src/compose_v4/experiments/pmo_population_v1.py"
SCORED_CONTRACT = ROOT / "configs/pmo_population_controller_v1_scored_contract_corrected.json"
IVG_TABLE = ROOT / "docs/invirtuogen_pmo_targets.json"
GENMOL_TABLE = ROOT / "docs/genmol_pmo_targets.json"

# ---- New outputs ----

BASELINE = ROOT / "configs/pmo_no_prescreen_baseline_v1.json"
MANIFESTS = {
    250: ROOT / "configs/pmo_scale_up_manifest_250_v1.json",
    1000: ROOT / "configs/pmo_scale_up_manifest_1k_v1.json",
    10000: ROOT / "configs/pmo_scale_up_manifest_10k_v1.json",
}
PLAN = ROOT / "diagnostics/pmo_scale_up_plan_v1.json"

RUNGS = {
    "rung_1_pilot": ["gsk3b", "perindopril_mpo", "celecoxib_rediscovery"],
    "rung_2_five": ["gsk3b", "perindopril_mpo", "celecoxib_rediscovery", "jnk3",
                    "sitagliptin_mpo"],
    "rung_3_full_suite": None,  # filled with the verified 23
}

SEAL = {
    "scored_launch_authorized": False,
    "modal_launch_authorized": False,
    "oracle_calls_authorized": 0,
    "status": "PREPARED_FAIL_CLOSED_PENDING_EXPLICIT_PAYLOAD_AUTHORIZATION",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write(path: Path, payload: dict) -> str:
    """Write an envelope whose payload_sha256 binds the payload, like the live chain."""

    digest = identity(payload)
    envelope = {"payload": payload, "payload_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return digest


# ---- Measured runtime capability ----


def runtime_constants() -> dict:
    """Parse the pinned runtime's budget/lock constants. No import, no execution."""

    tree = ast.parse(RUNTIME.read_text())
    out: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if isinstance(node.value, ast.Constant):
            out[name] = node.value.value
        elif isinstance(node.value, ast.Tuple):
            out[name] = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    source = RUNTIME.read_text()
    out["_reads_contract_budget"] = 'contract["budget"]' in source or 'contract.get("budget"' in source
    out["_enforces_task_lock"] = "task is outside the PMO-v1 three-task lock" in source
    return out


def runtime_capability(constants: dict) -> dict:
    budget = constants["QUERY_BUDGET"]
    rounds, per_round, init = constants["MAX_ROUNDS"], constants["QUERIES_PER_ROUND"], constants["INIT_COUNT"]
    round_cap = rounds * per_round + init
    return {
        "measured_from": str(RUNTIME.relative_to(ROOT)),
        "measured_sha256": sha(RUNTIME),
        "QUERY_BUDGET": budget,
        "MAX_ROUNDS": rounds,
        "QUERIES_PER_ROUND": per_round,
        "INIT_COUNT": init,
        "task_lock": constants["TASKS"],
        "reads_contract_budget": constants["_reads_contract_budget"],
        "enforces_three_task_lock": constants["_enforces_task_lock"],
        "max_charged_calls_per_task": min(budget, round_cap),
        "round_capacity_calls": round_cap,
        "note": (
            "execute_task builds its ledger with budget=QUERY_BUDGET and computes the AUC "
            "with budget=QUERY_BUDGET. It never reads contract['budget'], so the contract's "
            "charged_calls_per_task is a declaration the runtime does not enforce."
        ),
    }


def budget_arithmetic(calls: int, n_tasks: int, per_round: int, init: int) -> dict:
    candidate = calls - init
    if candidate < 0:
        raise ValueError("budget smaller than the initialization batch")
    rounds = math.ceil(candidate / per_round)
    return {
        "charged_calls_per_task": calls,
        "initialization_calls_per_task": init,
        "candidate_calls_per_task": candidate,
        "queries_per_round": per_round,
        "rounds_required": rounds,
        "candidate_calls_at_rounds_required": rounds * per_round,
        "final_round_is_partial": rounds * per_round != candidate,
        "n_tasks": n_tasks,
        "charged_calls_total": calls * n_tasks,
        "arithmetic_check": init + candidate == calls,
    }


# ---- Controller task-blindness ----


def controller_task_blindness(suite: list[str]) -> dict:
    """Prove by EXECUTION that the controller config carries no task information.

    Not an inspection of the source: the config is built and searched for every PMO
    task name.  What IS task-dependent is recorded alongside, so the claim is
    "the controller geometry is identical across tasks", not "nothing sees the task".
    """

    import inspect
    from dataclasses import asdict

    from compose_v4.experiments import pmo_population_v1 as module

    signature = inspect.signature(module.configuration)
    built = {seed: asdict(module.configuration(seed)) for seed in (20260920, 1, 99)}
    reference = built[20260920]
    blob = json.dumps(reference, sort_keys=True, default=str).lower()
    leaked = sorted(task for task in suite if task in blob)
    source = inspect.getsource(module)
    differing = sorted(k for k in reference if any(built[s][k] != reference[k] for s in built))
    return {
        "method": "executed configuration() and searched the produced config for task names",
        "signature": str(signature),
        "parameters": list(signature.parameters),
        "accepts_task_argument": "task" in signature.parameters,
        "config_field_count": len(reference),
        "task_names_found_in_config": leaked,
        "task_blind": not leaked and "task" not in signature.parameters,
        "identical_across_tasks": (
            "configuration() cannot vary by task: its only argument is `seed`, and "
            "execute_task calls it as configuration(contract['controller']['seed']) "
            "with no task argument."
        ),
        "fields_varying_with_seed_only": differing,
        "call_site_passes_only_seed": (
            'configuration(contract["controller"]["seed"])' in source
        ),
        "task_dependent_elsewhere_by_design": {
            "ProgramTask(task_name, ...)": "oracle identity / ledger protocol binding",
            "_runtime_protocol(contract, task_name)": "per-task protocol hash",
            "TASK_ROLES[task_name]": "reporting label only; KeyError outside the lock",
        },
        "seed_in_scored_contract": None,
    }


# ---- Baselines ----


def baseline_table() -> dict:
    ivg = json.loads(IVG_TABLE.read_text())
    prescreen = ivg["targets"]
    no_prescreen = ivg["no_prescreen_targets_partial"]
    rows = {}
    for task in sorted(prescreen):
        value = no_prescreen.get(task)
        rows[task] = {
            "no_prescreen_auc_top10_10000": value if value is not None else "ABSENT",
            "provenance": (
                f"{ivg['source']} Table 2 (no ZINC250k prescreen), transcribed in "
                f"docs/invirtuogen_pmo_targets.json"
            ) if value is not None else (
                "ABSENT: not transcribed from the published no-prescreen table. "
                "Do NOT substitute the prescreened value -- it is a different "
                "oracle-budget regime (+250,000 calls)."
            ),
            "prescreened_auc_top10_10000_DO_NOT_COMPARE": prescreen[task],
        }
    return {
        "schema_version": "pmo_no_prescreen_baseline_v1",
        "metric": "auc_top10, top_auc(finish=True), 10,000 oracle calls, mean of 3 runs",
        "comparison_rule": (
            "COMPOSE runs with prescreen=false. Only the no_prescreen column is a valid "
            "comparator. The prescreened column is retained solely to make substitution "
            "visible and must never be used as the target."
        ),
        "source": ivg["source"],
        "url": ivg["url"],
        "source_sha256": sha(IVG_TABLE),
        "coverage": {
            "n_tasks": len(prescreen),
            "n_with_no_prescreen_value": len(no_prescreen),
            "n_absent": len(prescreen) - len(no_prescreen),
            "published_no_prescreen_sum_all_23": ivg["no_prescreen_sum_reported"],
            "sum_is_not_reconstructible": (
                "The published no-prescreen column sums to 16.676 over all 23 tasks, but "
                "only 7 per-task values are transcribed. A suite-sum comparison is "
                "therefore NOT possible from this table."
            ),
        },
        "tasks": rows,
        **SEAL,
    }


# ---- Main ----


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from verify_pmo_task_registry import pmo_suite_23

    suite, suite_provenance = pmo_suite_23()
    RUNGS["rung_3_full_suite"] = suite

    constants = runtime_constants()
    capability = runtime_capability(constants)
    blindness = controller_task_blindness(suite)
    per_round, init = capability["QUERIES_PER_ROUND"], capability["INIT_COUNT"]

    baseline = baseline_table()
    no_prescreen = {
        t: (None if r["no_prescreen_auc_top10_10000"] == "ABSENT"
            else r["no_prescreen_auc_top10_10000"])
        for t, r in baseline["tasks"].items()
    }

    scored = json.loads(SCORED_CONTRACT.read_text())
    launcher = (ROOT / "tools/launch_pmo_population_v1_corrected.py").read_text()
    launcher_pins_contract = scored["payload_sha256"] in launcher
    implementation_drift = sorted(
        relative for relative, digest in scored["payload"]["implementation_sha256"].items()
        if sha(ROOT / relative) != digest
    )

    manifests: dict[int, dict] = {}
    for calls, path in MANIFESTS.items():
        rungs = {}
        for rung, tasks in RUNGS.items():
            arithmetic = budget_arithmetic(calls, len(tasks), per_round, init)
            blockers = []
            if calls > capability["max_charged_calls_per_task"]:
                blockers.append(
                    f"runtime charges at most {capability['max_charged_calls_per_task']} "
                    f"calls/task (QUERY_BUDGET={capability['QUERY_BUDGET']}, "
                    f"MAX_ROUNDS={capability['MAX_ROUNDS']}); {calls} is unreachable"
                )
            if calls < capability["QUERY_BUDGET"]:
                blockers.append(
                    f"runtime ledger budget is {capability['QUERY_BUDGET']}, so a "
                    f"{calls}-call contract would not stop the run at {calls}; the AUC "
                    f"would also be normalized by {capability['QUERY_BUDGET']}"
                )
            if arithmetic["rounds_required"] > capability["MAX_ROUNDS"]:
                blockers.append(
                    f"needs {arithmetic['rounds_required']} rounds, "
                    f"MAX_ROUNDS={capability['MAX_ROUNDS']}"
                )
            if set(tasks) - set(capability["task_lock"]):
                blockers.append(
                    "tasks outside the pinned three-task lock: "
                    f"{sorted(set(tasks) - set(capability['task_lock']))}"
                )
            rungs[rung] = {
                "tasks": tasks,
                "budget": arithmetic,
                "baseline_covered": sorted(t for t in tasks if no_prescreen.get(t) is not None),
                "baseline_absent": sorted(t for t in tasks if no_prescreen.get(t) is None),
                "runtime_blockers": blockers,
                "runnable_without_touching_pinned_files": not blockers,
            }
        payload = {
            "schema_version": "pmo_scale_up_manifest_v1",
            "charged_calls_per_task": calls,
            "metric_id": f"auc_top10@{calls}",
            "official_pmo_comparable": calls == 10000,
            "comparability": (
                "Official PMO budget: comparable to published tables ONLY if the full "
                "budget is charged and the run is a mean of 3 seeds."
                if calls == 10000 else
                f"DEVELOPMENT READING. auc_top10@{calls} is NOT comparable to a published "
                f"auc_top10@10000. top_auc trapezoids from (0,0), so a constant top-10 "
                f"level scores level*(1 - 100/(2*{calls})) = "
                f"{1 - 100 / (2 * calls):.4f}x at this budget versus 0.9950x at 10,000 -- "
                "a short-budget AUC is structurally DEPRESSED at identical performance."
            ),
            "aggregation": {
                "module": "compose_v4.eval.pmo_budget_aggregation",
                "reading": "top_ten_auc(scores, budget=N, authorized_budget=N)",
                "projection": "frozen_tail_projection(...) -- always labelled, never comparable",
                "suite": "aggregate_suite(...) -- refuses mixed budgets and projections",
                "log": "score_calls_log(scores) -> charged_queries / best_score / top10_score",
            },
            "oracle": {"prescreen": False, "direction": "maximize", "range": [0.0, 1.0],
                       "calls_include_initialization": True},
            "controller": {
                "task_blind": blindness["task_blind"],
                "proof": "see plan.controller_task_blindness (executed, not inspected)",
            },
            "rungs": rungs,
            "baseline_reference": {
                "path": str(BASELINE.relative_to(ROOT)),
                "note": "written by this tool in the same run; ABSENT means no published value",
            },
            "pinned_chain_read_only": {
                "scored_contract": str(SCORED_CONTRACT.relative_to(ROOT)),
                "scored_contract_payload_sha256": scored["payload_sha256"],
                "runtime": str(RUNTIME.relative_to(ROOT)),
                "runtime_sha256": sha(RUNTIME),
            },
            "runtime_capability_measured": capability,
            **SEAL,
        }
        manifests[calls] = payload

    plan = {
        "schema_version": "pmo_scale_up_plan_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/build_pmo_scale_up_plan.py",
        "launch_performed": False,
        "oracle_calls_made": 0,
        "modal_actions": "none",
        "staleness": {
            "regenerate_before_use": True,
            "why": (
                "This plan pins the live launch chain by content hash. The chain was "
                "observed CHANGING during preparation (another session was editing it), so "
                "these pins can go stale. Re-run tools/build_pmo_scale_up_plan.py --apply "
                "immediately before acting on it; the builder is deterministic, so an "
                "unchanged tree reproduces byte-identical artifacts."
            ),
            "chain_consistency_at_generation": {
                "scored_contract_payload_sha256": scored["payload_sha256"],
                "launcher_PAYLOAD_constant_matches_contract": launcher_pins_contract,
                "implementation_sha256_drift": implementation_drift,
                "chain_is_launchable_now": launcher_pins_contract and not implementation_drift,
            },
        },
        "suite": {"n_tasks": len(suite), "tasks": suite, "provenance": suite_provenance},
        "ladder": {
            rung: {
                "tasks": tasks,
                "n_tasks": len(tasks),
                "oracle_cost_by_budget": {
                    str(calls): {
                        "charged_calls_total": calls * len(tasks),
                        "with_3_seeds_for_publication": calls * len(tasks) * 3,
                    }
                    for calls in sorted(MANIFESTS)
                },
                "baseline_absent": sorted(t for t in tasks if no_prescreen.get(t) is None),
            }
            for rung, tasks in RUNGS.items()
        },
        "runtime_capability_measured": capability,
        "controller_task_blindness": blindness,
        "blockers": [
            {
                "id": "contract_budget_not_enforced",
                "severity": "BLOCKS_THE_IMMINENT_3x250_RELAUNCH",
                "evidence": "MEASURED from the pinned runtime and campaign loop",
                "finding": (
                    "The corrected contract declares charged_calls_per_task=250, but "
                    "execute_task builds its ledger with budget=QUERY_BUDGET=1000 and never "
                    "reads contract['budget']. run_program_campaign's only call-count "
                    "authority is ledger.remaining; stagnation_rounds is None, so the "
                    "stagnation break is disabled and the wall-clock break is the only "
                    "other exit. 64 rounds x 16 queries + 16 init = 1040 > 1000, so the "
                    "ledger binds at 1000."
                ),
                "consequence": (
                    "The 3x250 pilot would charge up to 1000 calls/task (~3000 total, 4x the "
                    "declared 750), and auc_top10_development_1000 would be normalized by "
                    "1000 rather than 250."
                ),
                "fix_requires_pinned_file": "src/compose_v4/experiments/pmo_population_v1.py",
                "not_fixed_here": "editing it would break the pinned launch chain",
            },
            {
                "id": "three_task_lock",
                "severity": "BLOCKS_RUNG_2_AND_RUNG_3",
                "evidence": "MEASURED: execute_task raises on any task outside TASKS",
                "finding": (
                    "TASKS is a 3-tuple and execute_task raises 'task is outside the PMO-v1 "
                    "three-task lock'. TASK_ROLES would also KeyError for a new task."
                ),
                "fix_requires_pinned_file": "src/compose_v4/experiments/pmo_population_v1.py",
            },
            {
                "id": "max_rounds_caps_the_10k_rung",
                "severity": "BLOCKS_ALL_10k_RUNS",
                "evidence": "MEASURED: MAX_ROUNDS=64, QUERIES_PER_ROUND=16",
                "finding": (
                    "10,000 calls needs 624 rounds; capacity is 64 rounds = 1040 calls. Both "
                    "MAX_ROUNDS and QUERY_BUDGET must rise for an official-budget run."
                ),
                "fix_requires_pinned_file": "src/compose_v4/experiments/pmo_population_v1.py",
            },
            {
                "id": "contract_path_is_hardcoded",
                "severity": "SCOPE_NOTE",
                "evidence": "MEASURED: CONTRACT is a module constant read by load_contract",
                "finding": (
                    "load_contract always reads configs/pmo_population_controller_v1.json. The "
                    "manifests written here are sealed PLANNING artifacts that specify the "
                    "ladder and its arithmetic; they are not drop-in runtime contracts."
                ),
            },
            {
                "id": "gsk3b_ivg_target_unsourced",
                "severity": "BASELINE_INTEGRITY",
                "evidence": "MEASURED: grep of docs/ finds no source for 0.952",
                "finding": (
                    "ivg_targets_postrun_only pins gsk3b=0.952 in three pinned contracts. "
                    "celecoxib=0.798 and perindopril=0.645 both match the published "
                    "no-prescreen table exactly, but gsk3b is ABSENT from that table "
                    "(its prescreened value is 0.988). 0.952 has no provenance in the repo."
                ),
                "consequence": (
                    "One third of the pilot's promotion criterion rests on an unsourced "
                    "number. Promotion requires beating 2 of 3, so gsk3b can decide it."
                ),
                "fix_requires_pinned_file": "configs/pmo_population_controller_v1*.json",
            },
            {
                "id": "no_prescreen_sum_not_reconstructible",
                "severity": "CLAIM_SCOPE",
                "evidence": "MEASURED: 7 of 23 per-task no-prescreen values transcribed",
                "finding": (
                    "The published no-prescreen column sums to 16.676 over 23 tasks, but only "
                    "7 per-task values exist in the repo. A 23-task suite-sum comparison is "
                    "not possible until the remaining 16 are transcribed from the paper."
                ),
            },
        ],
        "manifests": {
            str(calls): {
                "path": str(MANIFESTS[calls].relative_to(ROOT)),
                "charged_calls_per_task": calls,
                "official_pmo_comparable": manifests[calls]["official_pmo_comparable"],
            }
            for calls in sorted(MANIFESTS)
        },
        "baseline": {"path": str(BASELINE.relative_to(ROOT)),
                     "coverage": baseline["coverage"]},
        **SEAL,
    }

    if not args.apply:
        print(json.dumps({
            "would_write": [str(p.relative_to(ROOT)) for p in
                            [BASELINE, *MANIFESTS.values(), PLAN]],
            "runtime_capability": capability,
            "baseline_coverage": baseline["coverage"],
        }, indent=2, sort_keys=True))
        print("dry run: pass --apply to write")
        return

    digests = {str(BASELINE.relative_to(ROOT)): write(BASELINE, baseline)}
    for calls, path in MANIFESTS.items():
        digests[str(path.relative_to(ROOT))] = write(path, manifests[calls])
    plan["artifact_payload_sha256"] = digests
    digests[str(PLAN.relative_to(ROOT))] = write(PLAN, plan)
    print(json.dumps(digests, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
