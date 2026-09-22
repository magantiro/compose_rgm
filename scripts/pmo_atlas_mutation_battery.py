"""Mutation battery for the PMO atlas guards.

A guard is only evidence if removing it turns a NAMED test red.  This driver
copies the worktree, applies one production mutation at a time, runs the
focused suite, and records which tests failed.

Three disciplines this battery enforces, each paid for by a past failure:

* a mutation that does not change the file **aborts** the run rather than
  scoring as killed (a quote-mismatched ``str.replace`` is a silent no-op);
* a ``killed`` verdict must carry a NON-EMPTY list of failing test node ids,
  so "the suite errored for an unrelated reason" cannot read as "killed";
* a COSMETIC positive control that changes bytes and nothing else must stay
  GREEN, so "everything failed" cannot be mistaken for a working battery.

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/pmo_atlas_mutation_battery.py \
        --python ~/compose_region_pinned_env/bin/python \
        --output diagnostics/pmo_atlas_v1/mutation_battery.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODULE = "src/compose_v4/experiments/pmo_atlas_routes.py"
SUITE = "tests/test_pmo_atlas_routes.py"

DISCOVERY = "src/compose_v4/experiments/pmo_atlas_discovery.py"
BLIND_DRIVER = "scripts/pmo_atlas_blind_search.py"
PROBE_DRIVER = "scripts/pmo_atlas_entry_probe.py"
DISCOVERY_SUITE = "tests/test_pmo_atlas_discovery.py"


@dataclass(frozen=True)
class Mutation:
    name: str
    relative_path: str
    old: str
    new: str
    kind: str  # "negative" (must be killed) or "positive_control" (must stay green)
    expect_tests: tuple[str, ...] = ()


ROUTE_MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "bypass_input_hash_check",
        MODULE,
        "        if digest != artifact.file_sha256:",
        "        if False:",
        "negative",
        ("test_load_refuses_an_input_whose_bytes_moved",),
    ),
    Mutation(
        "bypass_payload_self_hash_check",
        MODULE,
        "    if recorded is not None and payload_sha256(payload) != recorded:",
        "    if False:",
        "negative",
        ("test_load_refuses_a_payload_that_disagrees_with_its_own_hash",),
    ),
    Mutation(
        "stop_comparing_interior_states",
        MODULE,
        "            if canonical_state_key(current) != canonical_state_key(recorded):\n"
        "                mismatches.append(index)",
        "            if False:\n"
        "                mismatches.append(index)",
        "negative",
        ("test_replay_detects_an_interior_divergence_not_only_the_endpoint",),
    ),
    Mutation(
        "let_exact_ignore_interior_mismatches",
        MODULE,
        "        and not mismatches\n        and not out_of_support",
        "        and True\n        and not out_of_support",
        "negative",
        ("test_replay_detects_an_interior_divergence_not_only_the_endpoint",),
    ),
    Mutation(
        "rebuild_checkpoints_by_reparsing_smiles",
        MODULE,
        "            found.append(\n                Checkpoint(",
        "            from compose_v4.chem.molecular_graph import smiles_to_molecular_graph\n"
        "            graph = smiles_to_molecular_graph(canonical_state_key(graph))\n"
        "            found.append(\n                Checkpoint(",
        "negative",
        ("test_checkpoints_are_committed_48_slot_states_inside_support",),
    ),
    Mutation(
        "clamp_the_anchor_short_of_the_endpoint",
        MODULE,
        "        index = max(1, min(total, round(fraction * total)))",
        "        index = max(1, min(total - 1, round(fraction * total)))",
        "negative",
        ("test_the_anchor_checkpoint_is_the_recorded_endpoint",),
    ),
    Mutation(
        "drop_the_checkpoint_fraction_range_check",
        MODULE,
        "        if not 0.0 <= fraction <= 1.0:",
        "        if False:",
        "negative",
        ("test_checkpoint_fraction_outside_the_unit_interval_is_refused",),
    ),
    Mutation(
        "make_the_regime_label_guard_a_no_op",
        MODULE,
        '    if payload.get("information_regime") != DEVELOPMENT_INFORMED_LABEL:',
        "    if False:",
        "negative",
        ("test_a_payload_without_the_regime_label_is_refused",),
    ),
    Mutation(
        "empty_the_prohibited_consumer_set",
        MODULE,
        "PROHIBITED_CONSUMERS: frozenset[str] = frozenset(\n    {\n"
        '        "scored_no_prescreen_run",',
        "PROHIBITED_CONSUMERS: frozenset[str] = frozenset(\n    {\n"
        '        "_disabled_scored_no_prescreen_run",',
        "negative",
        ("test_prohibited_consumers_are_refused_and_diagnostics_are_allowed",),
    ),
    Mutation(
        "count_programs_where_the_census_counts_sources",
        MODULE,
        '            "distinct_sources": len({route.source_smiles for route in routes}),',
        '            "distinct_sources": len(routes),',
        "negative",
        ("test_the_census_separates_programs_from_independent_lineages",),
    ),
    Mutation(
        "cosmetic_comment_only",
        MODULE,
        "# ---- Replay ----",
        "# ---- Replay ----\n# (cosmetic positive control: bytes change, behaviour does not)",
        "positive_control",
        (),
    ),
)

#: TEST C guards.  Every negative names the test it must turn red, and the two
#: driver mutations live at CALL SITES: a guard that is only tested at its
#: definition cannot show that production reaches it.
DISCOVERY_MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "bypass_blind_initialization_file_hash",
        DISCOVERY,
        "    if digest != BLIND_INITIALIZATION_SHA256:",
        "    if False:",
        "negative",
        ("test_an_initialization_whose_bytes_moved_is_refused",),
    ),
    Mutation(
        "bypass_blind_initialization_lock",
        DISCOVERY,
        '    if identity(body) != payload.get("lock_sha256"):',
        "    if False:",
        "negative",
        ("test_an_initialization_whose_lock_disagrees_with_its_content_is_refused",),
    ),
    Mutation(
        "bypass_the_atlas_leak_check",
        DISCOVERY,
        "    leaked = sorted(set(seen) & atlas)",
        "    leaked = []",
        "negative",
        ("test_an_atlas_molecule_in_the_initialization_is_refused",),
    ),
    Mutation(
        "allow_a_program_library_in_a_blind_run",
        DISCOVERY,
        "    if library:",
        "    if False:",
        "negative",
        ("test_a_program_library_is_refused_in_a_blind_run",),
    ),
    Mutation(
        "drop_the_blindness_guard_from_the_blind_driver",
        BLIND_DRIVER,
        "    blindness = assert_optimizer_blind(\n"
        "        initialization=initialization,\n"
        "        optimizer_kwargs=optimizer_kwargs,\n"
        "        library=(),\n"
        "        atlas=atlas,\n"
        "    )",
        '    blindness = {"leaked": []}\n    del atlas',
        "negative",
        ("test_the_blind_run_driver_calls_the_blindness_guard_before_any_charged_call",),
    ),
    Mutation(
        "let_a_probe_seed_come_from_the_atlas",
        PROBE_DRIVER,
        "    if row.endpoint in atlas:",
        "    if False:",
        "negative",
        ("test_a_probe_seed_from_the_atlas_is_refused",),
    ),
    Mutation(
        "let_a_probe_seed_arrive_without_its_state",
        PROBE_DRIVER,
        "    if row.state is None:",
        "    if False:",
        "negative",
        ("test_a_probe_seed_without_a_state_is_refused_rather_than_reparsed",),
    ),
    Mutation(
        "report_the_mean_approach_instead_of_the_nearest",
        DISCOVERY,
        "        value = max(similarities)",
        "        value = sum(similarities) / len(similarities)",
        "negative",
        ("test_nearest_approach_reports_the_maximum_not_an_average",),
    ),
    Mutation(
        "round_a_measurement_up_to_the_lowest_rung",
        DISCOVERY,
        '    label = reached[-1].label if reached else "below_" + ordered[0].label',
        "    label = reached[-1].label if reached else ordered[0].label",
        "negative",
        ("test_a_measurement_below_the_lowest_rung_is_not_rounded_up",),
    ),
    Mutation(
        "let_the_probe_geometry_drift_from_test_b",
        DISCOVERY,
        '        "initial_parent_fraction": 0.2,',
        '        "initial_parent_fraction": 0.5,',
        "negative",
        (
            "test_the_probe_runs_the_same_campaign_geometry_as_the_test_b_driver",
            "test_the_shared_geometry_helper_reproduces_the_test_b_driver",
        ),
    ),
    Mutation(
        "trust_the_parent_label_over_the_executed_trace",
        DISCOVERY,
        '                "parent_endpoint": executed or labelled,',
        '                "parent_endpoint": labelled or executed,',
        "negative",
        ("test_the_parent_comes_from_the_executed_trace_not_the_label",),
    ),
    Mutation(
        "count_the_seed_inside_the_probe_top_ten",
        DISCOVERY,
        "    new = {smiles: value for smiles, value in scores.items() if smiles != seed_smiles}",
        "    new = dict(scores)",
        "negative",
        ("test_a_run_that_only_preserves_its_seed_scores_no_lift",),
    ),
    Mutation(
        "cosmetic_comment_only",
        DISCOVERY,
        "# ---- Structural diagnostic ----",
        "# ---- Structural diagnostic ----\n"
        "# (cosmetic positive control: bytes change, behaviour does not)",
        "positive_control",
        (),
    ),
)

MUTATION_SETS: dict[str, tuple[str, tuple[Mutation, ...]]] = {
    "routes": (SUITE, ROUTE_MUTATIONS),
    "discovery": (DISCOVERY_SUITE, DISCOVERY_MUTATIONS),
}

_FAILED = re.compile(r"^(FAILED|ERROR) (\S+)", re.MULTILINE)


def _run_suite(
    root: Path, python: str, timeout: int, suite: str = SUITE
) -> tuple[int, tuple[str, ...], str]:
    env = {
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": "src:scripts",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
    }
    proc = subprocess.run(
        [python, "-m", "pytest", suite, "-q", "--no-header", "-p", "no:cacheprovider", "-rf"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = proc.stdout + proc.stderr
    failed = tuple(sorted({m.group(2).split("::")[-1] for m in _FAILED.finditer(output)}))
    return proc.returncode, failed, output[-2000:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", default="diagnostics/pmo_atlas_v1/mutation_battery.json")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--set", dest="mutation_set", default="routes",
                        choices=sorted(MUTATION_SETS))
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    suite, mutations = MUTATION_SETS[args.mutation_set]
    started = time.time()
    workspace = Path(tempfile.mkdtemp(prefix="pmo-atlas-mutation-"))
    tree = workspace / "tree"
    shutil.copytree(
        repo_root,
        tree,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"),
    )

    pristine = {
        mutation.relative_path: (tree / mutation.relative_path).read_text()
        for mutation in mutations
    }
    baseline_code, baseline_failed, baseline_tail = _run_suite(
        tree, args.python, args.timeout, suite
    )
    if baseline_code != 0:
        print("ABORT: the unmutated suite is not green; a battery on a red suite proves nothing")
        print(baseline_tail)
        return 2

    records: list[dict[str, Any]] = []
    for mutation in mutations:
        target = tree / mutation.relative_path
        original = target.read_text()
        mutated = original.replace(mutation.old, mutation.new, 1)
        if mutated == original:
            print(f"ABORT: mutation {mutation.name!r} did not apply; its anchor text moved")
            shutil.rmtree(workspace, ignore_errors=True)
            return 3
        target.write_text(mutated)
        code, failed, tail = _run_suite(tree, args.python, args.timeout, suite)
        target.write_text(original)

        if mutation.kind == "positive_control":
            verdict = "green_as_required" if code == 0 and not failed else "CONTROL_BROKEN"
        else:
            # A killed verdict REQUIRES a non-empty failing-test list.
            killed = code != 0 and bool(failed)
            expected_hit = set(mutation.expect_tests) <= set(failed)
            verdict = "killed" if killed and expected_hit else (
                "SURVIVED" if code == 0 else "killed_by_unexpected_test"
            )
        records.append(
            {
                "name": mutation.name,
                "kind": mutation.kind,
                "path": mutation.relative_path,
                "verdict": verdict,
                "exit_code": code,
                "failing_tests": list(failed),
                "expected_tests": list(mutation.expect_tests),
                "tail": tail if verdict not in {"killed", "green_as_required"} else None,
            }
        )
        print(f"{mutation.name:48s} {verdict:26s} failing={list(failed)}")

    for path, text in pristine.items():
        (tree / path).write_text(text)
    negatives = [r for r in records if r["kind"] == "negative"]
    controls = [r for r in records if r["kind"] == "positive_control"]
    payload = {
        "schema_version": "pmo_atlas_mutation_battery_v1",
        "information_regime": "DEVELOPMENT_INFORMED_DIAGNOSTIC",
        "suite": suite,
        "mutation_set": args.mutation_set,
        "python": args.python,
        "baseline_green": baseline_code == 0,
        "baseline_failing_tests": list(baseline_failed),
        "negatives_total": len(negatives),
        "negatives_killed": sum(1 for r in negatives if r["verdict"].startswith("killed")),
        "negatives_survived": [r["name"] for r in negatives if r["verdict"] == "SURVIVED"],
        "controls_green": sum(1 for r in controls if r["verdict"] == "green_as_required"),
        "controls_total": len(controls),
        "elapsed_seconds": round(time.time() - started, 2),
        "mutations": records,
    }
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    shutil.rmtree(workspace, ignore_errors=True)

    print(
        f"\nnegatives killed {payload['negatives_killed']}/{payload['negatives_total']}; "
        f"controls green {payload['controls_green']}/{payload['controls_total']}"
    )
    print(f"wrote {output}")
    return 0 if (
        payload["negatives_killed"] == payload["negatives_total"]
        and payload["controls_green"] == payload["controls_total"]
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
