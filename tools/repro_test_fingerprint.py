"""Turn a pytest run into a comparable fingerprint, and diff two fingerprints.

The rule this exists to enforce: after a structural pass you may not say "pytest
fails, the cleanup broke it" NOR "those were probably pre-existing". You diff
against a fingerprint taken before the pass.

The trap it exists to avoid is specific and was hit once while producing this
baseline. Without ``--continue-on-collection-errors`` pytest stops at the FIRST
collection error and reports ``1 skipped, 1 error`` with ZERO failures, which
reads as a clean baseline. The real state behind that single error is thousands
of tests never run. So the fingerprint records the collection summary line
FIRST, and ``compare`` refuses to produce a verdict when the collected count
moves, because two runs that collected different numbers of tests are not
comparable no matter how similar their failure lists look.

Usage::

    pytest tests/ -q --tb=no -rfE --continue-on-collection-errors > raw.txt
    python3 tools/repro_test_fingerprint.py --raw raw.txt --out repro/test_fingerprint_v1.json
    python3 tools/repro_test_fingerprint.py --raw new.txt --out new.json --compare base.json
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
OUTCOME = re.compile(r"^(?P<kind>FAILED|ERROR)\s+(?P<node>\S+)(?:\s+-\s+(?P<detail>.*))?$")
# "6296 tests collected", "6296 tests collected, 1 error"
COLLECTED = re.compile(r"(?P<n>\d+) tests? collected")
# "292 failed, 5806 passed, 101 errors in 1234.56s"
TALLY = re.compile(r"(?P<n>\d+) (?P<word>passed|failed|error|errors|skipped|xfailed|xpassed)")


def parse(raw: str) -> dict[str, Any]:
    failed: dict[str, str] = {}
    errored: dict[str, str] = {}
    for line in raw.splitlines():
        match = OUTCOME.match(line.strip())
        if not match:
            continue
        # The error signature, not the message: messages carry paths and numbers
        # that move between machines without any behaviour changing.
        detail = (match.group("detail") or "").strip()
        signature = detail.split(":")[0][:120] if detail else ""
        (failed if match.group("kind") == "FAILED" else errored)[match.group("node")] = signature

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

    comparable = (
        baseline["collected"] is not None
        and after["collected"] is not None
        and baseline["collected"] == after["collected"]
    )
    if not comparable:
        verdict = "NOT_COMPARABLE_COLLECTION_COUNT_MOVED"
    elif regressions:
        verdict = "TEST_REGRESSION"
    else:
        verdict = "NO_TEST_REGRESSION"
    return {
        "baseline_collected": baseline["collected"],
        "after_collected": after["collected"],
        "collection_counts_match": comparable,
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
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    raw = pathlib.Path(args.raw).read_text(encoding="utf-8", errors="ignore")
    report = parse(raw)
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
