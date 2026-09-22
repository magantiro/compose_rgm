#!/usr/bin/env python3
"""Mutation battery for the v3 path program and its suite wiring.

A guard is only load-bearing if removing it turns a NAMED test red.  This
battery edits production on a throwaway copy of the tree, runs the fragment
tests against it, and records which tests failed.

Three disciplines it enforces, each paid for by a real failure elsewhere in this
repository:

* **A mutation that does not APPLY must abort, never be scored.**  A mutation
  string written with single quotes against double-quoted source made
  ``str.replace`` a silent no-op, the unmutated file passed, and the guard read
  as weak.  Every mutation here asserts its anchor is present exactly once and
  that the file bytes changed.
* **KILLED requires a NON-EMPTY list of failing tests.**  A mutation that makes
  the suite error out for an unrelated reason is not evidence about the guard.
* **A cosmetic positive control must stay GREEN.**  Without a mutation that
  changes bytes and nothing else, "everything refused" cannot be distinguished
  from "the harness is broken".

Two of the mutations are deliberately the SAME wiring dropped at two different
hops, because two hops sharing one sink is exactly how a wiring mutation
survives a single consultation test.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = "src/compose_v4/benchmark/fragment_attachment_control.py"
SAMPLER = "src/compose_v4/benchmark/fragment_conditioned_sampler.py"
RUNNER = "tools/run_fragment_constrained_suite.py"
TEST_FILES = (
    "tests/test_fragment_attachment_control.py",
    "tests/test_fragment_official_suite.py",
)
COPY = (
    "src", "tools", "tests", "scripts", "data", "pyproject.toml",
    # Two official-suite tests read landed artifacts. ``diagnostics`` as a
    # whole is 542 MB, so only the subtree they open is copied; a battery
    # whose baseline is red for a missing fixture scores nothing.
    "diagnostics/fragment_official_suite_v2",
)

# (name, relative path, exact anchor, replacement, expected test, is_positive_control)
MUTATIONS: tuple[tuple[str, str, str, str, str, bool], ...] = (
    (
        "cosmetic_reformat_positive_control",
        CONTROL,
        "    # ---- Pathwise admission ----",
        "    # ---- Pathwise admission (cosmetic control, changes no behaviour) ----",
        "",
        True,
    ),
    (
        "path_program_active_for_a_single_core_prompt",
        CONTROL,
        "            and len(self._spec.lock_groups) >= 2\n        )",
        "            and len(self._spec.lock_groups) >= 1\n        )",
        "test_path_program_is_vacuous_for_a_single_core_prompt",
        False,
    ),
    (
        "path_program_on_by_default",
        CONTROL,
        "    path_program: bool = False",
        "    path_program: bool = True",
        "test_path_program_is_off_by_default",
        False,
    ),
    (
        "target_pinned_to_the_band_floor_instead_of_drawn",
        CONTROL,
        "        return int(rng.integers(low, high + 1))",
        "        return int(low)",
        "test_path_target_is_drawn_from_the_declared_band",
        False,
    ),
    (
        "unsatisfiable_predicate_blocks_again",
        CONTROL,
        "        if self.path_transaction_sites(predecessor) is None:\n            return True, \"\"",
        "        if False:\n            return True, \"\"",
        "test_an_unsatisfiable_path_predicate_does_not_block",
        False,
    ),
    (
        "free_valence_precondition_reinstated",
        CONTROL,
        "            for anchor in sorted(far):\n                if int(row[anchor]) <= 0:\n                    continue",
        "            for anchor in sorted(far):\n                if int(row[anchor]) <= 0:\n                    continue\n                if int(state.implicit_h_counts[anchor]) <= 0:\n                    continue",
        "test_a_saturated_anchor_no_longer_blocks_the_transaction",
        False,
    ),
    (
        "bridge_exchange_dropped_so_the_transaction_only_inserts",
        SAMPLER,
        "        after_open = system.apply(\n            after_insert,\n            \"bond_reroute\",\n            BondReroute(a=path_atom, b=far_anchor, u=free_slot, v=far_anchor),\n        )",
        "        after_open = after_insert",
        "test_the_transaction_lengthens_the_core_to_core_path",
        False,
    ),
    (
        "reroute_moves_the_path_atom_instead_of_the_new_atom",
        SAMPLER,
        "BondReroute(a=path_atom, b=far_anchor, u=free_slot, v=far_anchor),",
        "BondReroute(a=path_atom, b=far_anchor, u=path_atom, v=far_anchor),",
        "test_the_transaction_lengthens_the_core_to_core_path",
        False,
    ),
    (
        "payload_not_required_to_come_from_the_prior",
        SAMPLER,
        "        if not any(n[0] == path_atom for n in getattr(action, \"neighbors\", ())):\n            continue",
        "        pass",
        "test_the_transaction_payload_comes_from_the_prior",
        False,
    ),
    (
        "bridge_knob_dropped_where_run_task_calls_build_prompt_context",
        RUNNER,
        "                    linker_bridge_atoms=linker_bridge_atoms,\n                )",
        "                )",
        "test_the_bridge_knob_reaches_the_prompt_context_call_site",
        False,
    ),
    (
        "bridge_knob_dropped_where_main_calls_run_task",
        RUNNER,
        "        \"linker_bridge_atoms\": args.linker_bridge_atoms,\n    }\n\n\ndef protocol_block",
        "    }\n\n\ndef protocol_block",
        "test_the_bridge_knob_survives_the_args_to_run_task_hop",
        False,
    ),
    (
        "path_program_flag_ignored_by_the_runner",
        RUNNER,
        "        path_program=bool(args.path_program),",
        "        path_program=False,",
        "test_the_path_program_flag_reaches_the_controller_configuration",
        False,
    ),
    (
        "protocol_omits_the_seeded_bridge",
        RUNNER,
        "        \"linker_bridge_atoms\": args.linker_bridge_atoms,\n    }\n\n\ndef sampler_config_from_args",
        "    }\n\n\ndef sampler_config_from_args",
        "test_the_seeded_bridge_is_recorded_in_the_protocol_block",
        False,
    ),
)

FAIL_RE = re.compile(r"^(?:FAILED|ERROR) [^:]+::([A-Za-z0-9_\[\]-]+)", re.MULTILINE)


def _materialize(destination: Path) -> None:
    for entry in COPY:
        source = ROOT / entry
        if not source.exists():
            continue
        target = destination / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, target)


def _run_tests(tree: Path) -> tuple[bool, list[str], str]:
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": f"{tree / 'src'}:{tree / 'scripts'}:{tree / 'tools'}",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "HOME": str(Path.home()),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
         *TEST_FILES],
        cwd=tree, env=env, capture_output=True, text=True, timeout=3600, check=False,
    )
    output = proc.stdout + proc.stderr
    return proc.returncode == 0, sorted(set(FAIL_RE.findall(output))), output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="v3_mutation_baseline_") as tmp:
        tree = Path(tmp)
        _materialize(tree)
        green, failing, output = _run_tests(tree)
        if not green:
            raise SystemExit(
                "the UNMUTATED tree must be green before any mutation is scored; "
                f"failing: {failing}\n{output[-4000:]}"
            )
        baseline_failing = failing

    for name, relative, anchor, replacement, expected, positive in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix=f"v3_mutation_{name}_") as tmp:
            tree = Path(tmp)
            _materialize(tree)
            path = tree / relative
            before = path.read_text()
            occurrences = before.count(anchor)
            if occurrences != 1:
                raise SystemExit(
                    f"{name}: anchor occurs {occurrences} times in {relative}; a "
                    "mutation that cannot be applied unambiguously must ABORT, "
                    "never be scored -- a no-op replace reads as a surviving "
                    "mutation and indicts the tests for nothing"
                )
            after = before.replace(anchor, replacement)
            if after == before:
                raise SystemExit(f"{name}: replacement changed no bytes")
            path.write_text(after)

            green, failing, _ = _run_tests(tree)
            new_failures = [t for t in failing if t not in baseline_failing]
            if positive:
                verdict = "GREEN_AS_REQUIRED" if green else "CONTROL_BROKE_THE_HARNESS"
            elif not new_failures:
                verdict = "SURVIVED"
            elif expected in new_failures:
                verdict = "KILLED"
            else:
                verdict = "KILLED_BY_ANOTHER_TEST"
            results.append({
                "mutation": name,
                "file": relative,
                "positive_control": positive,
                "expected_test": expected or None,
                "all_failing_tests": new_failures,
                "verdict": verdict,
            })
            print(f"{name:58s} {verdict}", flush=True)

    killed = sum(1 for r in results if r["verdict"].startswith("KILLED"))
    total = sum(1 for r in results if not r["positive_control"])
    payload = {
        "schema": "compose_fragment_path_v3_mutation_battery_v1",
        "baseline_unmutated_tree_green": True,
        "baseline_failing_tests": baseline_failing,
        "rules": {
            "mutation_must_apply": "an ambiguous or no-op anchor ABORTS the run",
            "killed_requires": "a NON-EMPTY list of newly failing named tests",
            "positive_control": "a cosmetic edit must leave the suite green",
        },
        "mutations": results,
        "killed": killed,
        "total": total,
        "survivors": [r["mutation"] for r in results if r["verdict"] == "SURVIVED"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\n{killed} of {total} killed; wrote {args.output}")


if __name__ == "__main__":
    main()
