"""Mutation battery for the replace-completion law wiring.

A wiring test that cannot go red proves nothing.  This runner applies one
production mutation at a time on a COPY of the tree, runs a named test, and
requires that test to FAIL.  It also runs a positive control that perturbs
something irrelevant and must still PASS -- without such a control, "everything
refused" is indistinguishable from "the harness is broken", which has happened
in this repository before.

Every mutation is DIFFED before the test runs.  A mutation string that does not
match (a quoting slip, a reformatted call site) is a silent no-op that makes an
unmutated file look like a surviving mutant, so this refuses to report a verdict
when the edit did not apply.

ZERO oracle calls; this only runs pytest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA_VERSION = "t4_fa7_0_completion_law_mutations_v1"

DPS = "src/compose_v4/control/dynamic_program_synthesis.py"
LAW = "src/compose_v4/control/replace_completion_law.py"
TESTS = "tests/test_replace_completion_law.py"

#: (name, file, old, new, test, expectation)
#: expectation "red" = the mutation must break that test; "green" = control.
MUTATIONS = [
    (
        "hop1_dynamic_drops_the_keyword",
        DPS,
        """                product, stage = compile_generic_module(
                    current,
                    rng,
                    family,
                    region_law=region_law,
                    completion_law=completion_law,
                )""",
        """                product, stage = compile_generic_module(
                    current,
                    rng,
                    family,
                    region_law=region_law,
                )""",
        "test_dynamic_synthesizer_consults_the_completion_law",
        "red",
    ),
    (
        "hop2_named_sequence_drops_the_keyword",
        DPS,
        """        current, stage = compile_generic_module(
            current,
            rng,
            family,
            region_law=region_law,
            completion_law=completion_law,
        )""",
        """        current, stage = compile_generic_module(
            current,
            rng,
            family,
            region_law=region_law,
        )""",
        "test_named_sequence_consults_the_completion_law",
        "red",
    ),
    (
        "sink_ignores_the_law_entirely",
        DPS,
        "        if completion_law is None:",
        "        if True:",
        "test_dynamic_synthesizer_consults_the_completion_law",
        "red",
    ),
    (
        "law_becomes_a_filter_not_a_reranking",
        LAW,
        "            out[position] = max(self.floor, math.exp(value / self.temperature))",
        "            out[position] = math.exp(value / self.temperature)",
        "test_support_floor_actually_binds_not_merely_exists",
        "red",
    ),
    (
        "unrealizable_candidate_dropped_instead_of_floored",
        LAW,
        """            except (ValueError, KeyError, IndexError, TypeError, RuntimeError):
                out[position] = self.floor
                continue""",
        """            except (ValueError, KeyError, IndexError, TypeError, RuntimeError):
                out[position] = 0.0
                continue""",
        "test_a_raising_margin_is_floored_not_dropped",
        "red",
    ),
    (
        "tilt_points_the_wrong_way",
        LAW,
        "            out[position] = max(self.floor, math.exp(value / self.temperature))",
        "            out[position] = max(self.floor, math.exp(-value / self.temperature))",
        "test_a_better_margin_is_preferred_far_more_often_than_not",
        "red",
    ),
    (
        "CONTROL_cosmetic_comment_only",
        LAW,
        "SCHEMA_VERSION = \"t4_replace_completion_law_v1\"",
        "# a cosmetic edit that changes no behaviour\nSCHEMA_VERSION = \"t4_replace_completion_law_v1\"",
        "test_dynamic_synthesizer_consults_the_completion_law",
        "green",
    ),
]


def _run_test(tree: Path, test: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", f"{TESTS}::{test}", "-q", "--no-header"],
        cwd=tree,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONPATH": "src",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
            "HOME": str(Path.home()),
        },
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-400:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for name, rel, old, new, test, expectation in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            shutil.copytree(
                args.repo,
                tree,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc", "diagnostics", "logs"
                ),
            )
            target = tree / rel
            before = target.read_text()
            if old not in before:
                rows.append(
                    {
                        "mutation": name,
                        "applied": False,
                        "verdict": "HARNESS_ERROR",
                        "detail": "mutation string did not match; no verdict",
                    }
                )
                print(f"[{name}] HARNESS ERROR: mutation did not apply", flush=True)
                continue
            after = before.replace(old, new, 1)
            if after == before:
                rows.append(
                    {"mutation": name, "applied": False, "verdict": "HARNESS_ERROR"}
                )
                continue
            target.write_text(after)
            passed, tail = _run_test(tree, test)
            if expectation == "red":
                ok = not passed
                verdict = "KILLED" if ok else "SURVIVED"
            else:
                ok = passed
                verdict = "CONTROL_OK" if ok else "CONTROL_BROKEN"
            rows.append(
                {
                    "mutation": name,
                    "file": rel,
                    "test": test,
                    "expectation": expectation,
                    "applied": True,
                    "test_passed": passed,
                    "verdict": verdict,
                    "tail": tail if not ok else "",
                }
            )
            print(f"[{name}] {verdict}", flush=True)

    killed = sum(1 for r in rows if r.get("verdict") == "KILLED")
    survived = sum(1 for r in rows if r.get("verdict") == "SURVIVED")
    controls = sum(1 for r in rows if r.get("verdict") == "CONTROL_OK")
    broken = sum(
        1 for r in rows if r.get("verdict") in ("HARNESS_ERROR", "CONTROL_BROKEN")
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "killed": killed,
        "survived": survived,
        "controls_ok": controls,
        "harness_problems": broken,
        "verdict": "PASS" if survived == 0 and broken == 0 else "FAIL",
        "mutations": rows,
        "oracle_calls": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(
        f"\nkilled={killed} survived={survived} controls_ok={controls} "
        f"harness_problems={broken} -> {report['verdict']}"
    )


if __name__ == "__main__":
    main()
