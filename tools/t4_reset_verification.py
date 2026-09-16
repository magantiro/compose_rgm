"""Summarize actual verification artifacts; never converts a failure into a pass."""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal

ROOT = Path(__file__).resolve().parents[1]


def junit(path):
    if not path.exists():
        return {"status": "not_finished_or_not_run", "path": str(path)}
    root = ET.parse(path).getroot()  # Local pytest-generated XML only.
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        **counts,
        "status": "passed"
        if counts["tests"] > 0 and not counts["failures"] and not counts["errors"]
        else "failed",
        "failed_nodes": [
            f"{c.get('classname')}::{c.get('name')}"
            for c in root.iter("testcase")
            if c.find("failure") is not None or c.find("error") is not None
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = ROOT / "diagnostics/t4_objective_reset/verification"
    focused, full = junit(folder / "focused.xml"), junit(folder / "full_suite.xml")
    execution_path = folder / "full_suite_execution.json"
    if execution_path.exists() and full["status"] != "passed":
        full["execution"] = json.loads(execution_path.read_text())
    lint = json.loads((folder / "repository_lint.json").read_text())
    changed = set(
        subprocess.check_output(
            ["git", "diff", "2114c405", "--name-only", "--", "src/"], cwd=ROOT, text=True
        ).splitlines()
    )
    new_lint = [r for r in lint if str(Path(r["filename"]).relative_to(ROOT)) in changed]
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "configs/t4_objective_reset_runtime_v1.json",
        ROOT / "modal_apps/t4_objective_reset_app.py",
    ]
    receipt = {
        "schema_version": "t4_reset_verification_v1",
        "observed_at_utc": _stamp(),
        "focused": focused,
        "handoff_tests": junit(folder / "handoff.xml"),
        "full_suite": full,
        "repository_lint": {
            "count": len(lint),
            "violations_in_changed_source_files": len(new_lint),
            "sha256": sha256_file(folder / "repository_lint.json"),
        },
        "files_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
        "passed": focused["status"] == full["status"] == "passed" and not lint,
        "new_oracle_calls": 0,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "interpretation": "Inherited failures are distinguished from new regressions, not waived. No scored launch from a failed receipt.",
    }
    seal(args.output, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files_sha256"}, sort_keys=True))


if __name__ == "__main__":
    main()
