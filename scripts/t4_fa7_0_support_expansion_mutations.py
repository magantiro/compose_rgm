"""Mutation battery for the fa7_0 support-expansion wiring.

A structural or behavioural test is worth exactly what its mutations prove.  Each
mutation below removes ONE load-bearing piece of the mechanism and names the single
test that must turn red for it.  A mutation that fails to apply -- an indentation
mismatch, a quote style, a string that moved -- ABORTS rather than scoring as
killed, because a no-op `str.replace` reports a passing file as a killed mutation
and that has happened in this repository before.

Each mutation is run against ONLY the test it targets.  That is deliberate:
`test_every_pinned_runtime_input_matches_the_tree` would go red for EVERY mutation
of a pinned file, which would let a decorative test masquerade as a killed one.  A
guard whose test another guard also satisfies is not load-bearing.

Two positive controls run as well: the unmutated tree must be GREEN, and a
comment-only edit to a file outside the pin block must stay GREEN.  Without them,
"everything refused" is indistinguishable from "the harness is broken".

    PYTHONPATH=src python3 scripts/t4_fa7_0_support_expansion_mutations.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = "tests/test_t4_support_expansion.py"
BASE_APP = "modal_apps/t4_fa7_0_support_expansion_base_app.py"
LIBRARY = "src/compose_v4/experiments/t4_support_expansion.py"
SCRATCH = "scripts/t4_fa7_0_support_expansion_yield.py"

#: (name, relative path, old, new, the ONE test that must fail)
MUTATIONS = [
    (
        "run_cell_never_expands",
        BASE_APP,
        "            expansion = run_support_expansion(",
        "            expansion = _never_expand(",
        "test_run_cell_expands_support_before_it_can_terminate",
    ),
    (
        "terminal_publish_ungated",
        BASE_APP,
        "            assert_support_expansion_is_consumed(expansion)",
        "            pass  # mutation: terminal publish no longer gated",
        "test_the_terminal_publish_is_gated_on_the_expansion_having_run",
    ),
    (
        "image_bakes_the_parent_app",
        BASE_APP,
        '        ROOT / "modal_apps/t4_fa7_0_support_expansion_base_app.py",',
        '        ROOT / "modal_apps/t4_integrated_route_fiber_parp1_app.py",',
        "test_the_base_app_bakes_its_own_source_into_the_image",
    ),
    (
        "completion_law_not_threaded",
        BASE_APP,
        "            completion_law=completion_law,\n        )\n        telemetry = {",
        "        )\n        telemetry = {",
        "test_the_proposal_worker_threads_both_laws_and_the_fallback",
    ),
    (
        "lock_drops_expansion_telemetry",
        BASE_APP,
        '            "support_expansion_telemetry": _jsonable(expansion_telemetry),',
        "",
        "test_the_round_lock_carries_the_expansion_telemetry",
    ),
    (
        "missing_block_terminates_quietly",
        BASE_APP,
        "            policy = resolve_support_expansion(contract)",
        "            policy = None  # mutation: contract block never read",
        "test_a_missing_contract_block_raises_rather_than_terminating_quietly",
    ),
    (
        "draw_cap_ignored",
        LIBRARY,
        "            if spent + step > policy.max_extra_draws_per_event:",
        "            if False:",
        "test_the_draw_cap_binds_before_the_ladder_ends",
    ),
    (
        "wall_clock_ignored",
        LIBRARY,
        "            if clock() - started >= policy.wall_seconds:",
        "            if False:",
        "test_the_wall_clock_binds_and_is_checked_before_a_step_is_spent",
    ),
    (
        "fallback_stage_skipped",
        LIBRARY,
        "    if policy.zero_support_fallback and fallback is not None:",
        "    if False:",
        "test_the_fallback_runs_first_and_can_satisfy_the_target_alone",
    ),
    (
        "consumption_gate_accepts_a_no_op_event",
        LIBRARY,
        '    if outcome.stop_reason == "not_run":',
        "    if False:",
        "test_publishing_exhaustion_without_an_expansion_is_refused",
    ),
    (
        "already_seen_ignored",
        LIBRARY,
        "            if key in seen or key in found:",
        "            if key in found:",
        "test_already_seen_endpoints_are_not_counted_as_found",
    ),
]

#: (name, relative path, old, new) -- must stay GREEN on the named test.
POSITIVE_CONTROLS = [
    (
        "comment_only_edit_outside_the_pin_block",
        SCRATCH,
        "ZERO oracle calls, zero docking, zero Modal.",
        "ZERO oracle calls, zero docking, zero Modal.  (positive control)",
        "test_run_cell_expands_support_before_it_can_terminate",
    ),
]


def _pytest(selector: str) -> tuple[bool, str]:
    target = f"{TESTS}::{selector}" if selector else TESTS
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(ROOT / "src"),
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
            "HOME": str(Path.home()),
        },
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip().splitlines()[-1:][0] if (
        proc.stdout + proc.stderr
    ).strip() else ""


def _apply(relative: str, old: str, new: str) -> str:
    path = ROOT / relative
    original = path.read_text()
    if old not in original:
        raise SystemExit(
            f"ABORT: mutation text not found in {relative}; the battery would have "
            f"scored an unmutated file as a killed mutation.\n  looked for: {old!r}"
        )
    mutated = original.replace(old, new, 1)
    if mutated == original:
        raise SystemExit(f"ABORT: mutation of {relative} changed nothing")
    path.write_text(mutated)
    return original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results = {"schema_version": "t4_fa7_0_support_expansion_mutations_v1", "rows": []}

    baseline_ok, baseline_line = _pytest("")
    results["baseline_green"] = baseline_ok
    results["baseline_line"] = baseline_line
    print(f"[baseline] unmutated tree green={baseline_ok}  {baseline_line}", flush=True)
    if not baseline_ok:
        raise SystemExit("ABORT: the unmutated tree is already red; nothing below means anything")

    for name, relative, old, new, selector in MUTATIONS:
        original = _apply(relative, old, new)
        try:
            green, line = _pytest(selector)
        finally:
            (ROOT / relative).write_text(original)
        killed = not green
        results["rows"].append(
            {
                "mutation": name,
                "file": relative,
                "target_test": selector,
                "killed": killed,
                "pytest": line,
            }
        )
        print(f"[{'KILLED ' if killed else 'SURVIVED'}] {name} -> {selector}", flush=True)

    for name, relative, old, new, selector in POSITIVE_CONTROLS:
        original = _apply(relative, old, new)
        try:
            green, line = _pytest(selector)
        finally:
            (ROOT / relative).write_text(original)
        results["rows"].append(
            {
                "mutation": name,
                "file": relative,
                "target_test": selector,
                "positive_control": True,
                "stayed_green": green,
                "pytest": line,
            }
        )
        print(f"[{'GREEN  ' if green else 'FALSE-RED'}] control {name}", flush=True)

    killed = sum(1 for row in results["rows"] if row.get("killed"))
    total = sum(1 for row in results["rows"] if "killed" in row)
    controls = [row for row in results["rows"] if row.get("positive_control")]
    results["killed"] = killed
    results["mutations"] = total
    results["controls_green"] = all(row["stayed_green"] for row in controls)
    results["verdict"] = (
        "ALL_KILLED_CONTROLS_GREEN"
        if killed == total and results["controls_green"]
        else "INCOMPLETE"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\n{killed}/{total} killed, controls green={results['controls_green']}", flush=True)
    return 0 if results["verdict"] == "ALL_KILLED_CONTROLS_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
