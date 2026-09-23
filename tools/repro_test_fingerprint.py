"""Turn a pytest run into a comparable fingerprint, and diff two fingerprints.

The rule this exists to enforce: after a structural pass you may not say "pytest
fails, the cleanup broke it" NOR "those were probably pre-existing". You diff
against a fingerprint taken before the pass.

The trap it exists to avoid is specific and was hit once while producing this
baseline. Without ``--continue-on-collection-errors`` pytest stops at the FIRST
collection error and reports ``1 skipped, 1 error`` with ZERO failures, which
reads as a clean baseline. The real state behind that single error is thousands
of tests never run.

So comparability is a SET question, not a count question. The fingerprint
records the collected node ids, and ``compare`` asks which baseline nodes
DISAPPEARED. Requiring the count to match instead would cry wolf on every pass
that legitimately adds a test, and would still miss the case where a collection
error drops three tests while a new file adds six.

Usage::

    pytest tests/ -q --tb=no -rfE --continue-on-collection-errors > raw.txt
    pytest tests/ -q --collect-only --continue-on-collection-errors | grep :: > collected.txt
    python3 tools/repro_test_fingerprint.py --raw raw.txt --collected collected.txt \
        --out repro/test_fingerprint_v1.json
    python3 tools/repro_test_fingerprint.py --raw new.txt --collected new_collected.txt \
        --out new.json --compare repro/test_fingerprint_v1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

# "FAILED tests/test_x.py::test_y - AssertionError: ..." / "ERROR tests/test_z.py"
#
# Deliberately permissive after the kind. A parametrized node id can contain
# SPACES inside its brackets, and pytest truncates the short-summary line at the
# terminal width, so such a line carries neither a closing bracket nor the " - "
# separator. An end-anchored `\S+` pattern rejects exactly those lines and drops
# them from the fingerprint silently: measured here at 5 of 297 failures. The
# count guard below turns any future variant of that into a loud failure.
OUTCOME = re.compile(r"^(?P<kind>FAILED|ERROR)\s+(?P<rest>.+)$")
# "6296 tests collected", "6296 tests collected, 1 error"
COLLECTED = re.compile(r"(?P<n>\d+) tests? collected")
# "292 failed, 5806 passed, 101 errors in 1234.56s"
TALLY = re.compile(r"(?P<n>\d+) (?P<word>passed|failed|error|errors|skipped|xfailed|xpassed)")


def parse(raw: str) -> dict[str, Any]:
    failed: dict[str, str] = {}
    errored: dict[str, str] = {}
    candidate_lines = 0
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(("FAILED ", "ERROR ")):
            candidate_lines += 1
        match = OUTCOME.match(stripped)
        if not match:
            continue
        rest = match.group("rest")
        node, separator, detail = rest.partition(" - ")
        # The error signature, not the message: messages carry paths and numbers
        # that move between machines without any behaviour changing. A truncated
        # node id has no separator, and keeping the whole remainder is fine
        # because pytest truncates identically across runs, so the SET comparison
        # still holds.
        signature = detail.strip().split(":")[0][:120] if separator else ""
        (failed if match.group("kind") == "FAILED" else errored)[node.strip()] = signature

    parsed = len(failed) + len(errored)
    if parsed != candidate_lines:
        raise ValueError(
            f"fingerprint parser accounted for {parsed} of {candidate_lines} FAILED/ERROR lines. "
            "A fingerprint that silently drops entries is worse than no fingerprint; fix the "
            "parser rather than the expectation."
        )

    tail = "\n".join(raw.strip().splitlines()[-25:])
    tally: dict[str, int] = {}
    for match in TALLY.finditer(tail):
        word = match.group("word").rstrip("s")
        tally[word] = max(tally.get(word, 0), int(match.group("n")))
    collected = COLLECTED.search(raw)

    return {
        "schema_version": 1,
        "collected": int(collected.group("n")) if collected else None,
        "tally": tally,
        "failed_count": len(failed),
        "error_count": len(errored),
        "failed": dict(sorted(failed.items())),
        "errored": dict(sorted(errored.items())),
    }


def compare(baseline: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_bad = set(baseline["failed"]) | set(baseline["errored"])
    after_bad = set(after["failed"]) | set(after["errored"])
    regressions = sorted(after_bad - before_bad)
    repaired = sorted(before_bad - after_bad)

    # Comparability is a SET question, not a count question. Adding a test file
    # legitimately raises the count, so requiring equality would cry wolf on
    # every pass that adds a guard. What must never happen is a baseline node
    # DISAPPEARING, which is the signature of a collection error swallowing
    # tests while the run still reports zero failures.
    before_nodes = set(baseline.get("collected_nodes") or [])
    after_nodes = set(after.get("collected_nodes") or [])
    if before_nodes and after_nodes:
        vanished = sorted(before_nodes - after_nodes)
        added = sorted(after_nodes - before_nodes)
        comparable = not vanished
        basis = "node_id_set"
    else:
        # Fall back to counts, and then only a DROP is disqualifying.
        vanished, added = [], []
        comparable = (
            baseline["collected"] is not None
            and after["collected"] is not None
            and after["collected"] >= baseline["collected"]
        )
        basis = "collected_count_floor"

    if not comparable:
        verdict = "NOT_COMPARABLE_TESTS_DISAPPEARED"
    elif regressions:
        verdict = "TEST_REGRESSION"
    else:
        verdict = "NO_TEST_REGRESSION"
    return {
        "comparability_basis": basis,
        "baseline_collected": baseline["collected"],
        "after_collected": after["collected"],
        "collected_nodes_vanished": vanished[:50],
        "collected_nodes_vanished_count": len(vanished),
        "collected_nodes_added_count": len(added),
        "comparable": comparable,
        "baseline_failing_or_erroring": len(before_bad),
        "after_failing_or_erroring": len(after_bad),
        "new_regressions": regressions,
        "new_regression_count": len(regressions),
        "newly_passing": repaired,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="captured pytest stdout")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", help="baseline fingerprint to diff against")
    parser.add_argument(
        "--collected",
        help="file of collected node ids (pytest --collect-only -q | grep ::). Recorded so the "
        "comparison can ask which tests DISAPPEARED rather than whether the count changed.",
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    raw = pathlib.Path(args.raw).read_text(encoding="utf-8", errors="ignore")
    report = parse(raw)
    if args.collected:
        nodes = [
            line.strip()
            for line in pathlib.Path(args.collected).read_text(encoding="utf-8").splitlines()
            if "::" in line
        ]
        report["collected_nodes"] = sorted(set(nodes))
        report["collected_node_count"] = len(report["collected_nodes"])
    report["label"] = args.label
    report["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"collected={report['collected']} tally={report['tally']} "
        f"failed={report['failed_count']} errored={report['error_count']} -> {out}"
    )

    if args.compare:
        baseline = json.loads(pathlib.Path(args.compare).read_text(encoding="utf-8"))
        diff = compare(baseline, report)
        diff_path = out.with_name(out.stem + "_vs_baseline.json")
        diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(diff, indent=2)[:3000])
        return 0 if diff["verdict"] == "NO_TEST_REGRESSION" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
