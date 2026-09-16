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


BASE_REVISION = "2114c405"

# A test that walks the repository can be broken by a file this branch merely adds,
# so those failures cannot be classified structurally and are listed for inspection.
SCANNING_MARKERS = ("rglob", "iterdir", "os.walk", "glob(")


def _git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def branch_footprint() -> dict:
    """Which pre-existing files this branch changed at all.

    If every executable change is a new file, no pre-existing module's behaviour
    moved, so a failure in a pre-existing test cannot have been introduced here.
    """
    rows = [
        line.split("\t", 1)
        for line in _git("diff", "--name-status", BASE_REVISION, "HEAD").splitlines()
        if line
    ]
    modified = sorted(path for status, path in rows if status != "A")
    executable_modified = sorted(
        path for path in modified if not path.endswith((".md", ".txt", ".rst"))
    )
    return {
        "base_revision": BASE_REVISION,
        "added": sorted(path for status, path in rows if status == "A"),
        "modified": modified,
        "executable_files_modified": executable_modified,
        "additions_only": not executable_modified,
    }


def failure_provenance(full: dict, footprint: dict) -> dict:
    """Separate this branch's own tests from inherited ones, structurally."""
    nodes = full.get("failed_nodes", [])
    branch_tests = {
        Path(path).stem
        for path in footprint["added"]
        if path.startswith("tests/") and path.endswith(".py")
    }
    own, inherited = [], []
    for node in nodes:
        module = node.split("::", 1)[0].split(".")[-1]
        (own if module in branch_tests else inherited).append(node)
    scanning = []
    for node in inherited:
        source = ROOT / "tests" / (node.split("::", 1)[0].split(".")[-1] + ".py")
        if source.exists():
            text = source.read_text()
            if any(marker in text for marker in SCANNING_MARKERS):
                scanning.append(node)
    return {
        "branch_test_files": sorted(branch_tests),
        "failing_nodes_from_this_branch": sorted(own),
        "failing_nodes_inherited": sorted(inherited),
        "inherited_failures_in_repository_scanning_tests": sorted(scanning),
        "structural_claim": (
            "every executable change on this branch is a new file, so no pre-existing "
            "module changed and an inherited failure cannot have been introduced here"
            if footprint["additions_only"]
            else "this branch modifies pre-existing executable files, so inherited "
            "failures cannot be classified structurally"
        ),
        "residual_risk": (
            "a test that walks the repository can be broken by a newly added file; those "
            "nodes are listed separately and must be read individually"
        ),
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

    def _relative(filename: str) -> str:
        """Lint rows record absolute paths, which a moved worktree makes foreign."""
        path = Path(filename)
        if path.is_relative_to(ROOT):
            return str(path.relative_to(ROOT))
        raise ValueError(
            f"repository lint was produced under a different root: {filename}. "
            "Regenerate it here with: .venv/bin/python -m ruff check --output-format json ."
        )

    new_lint = [r for r in lint if _relative(r["filename"]) in changed]
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "configs/t4_objective_reset_runtime_v1.json",
        ROOT / "modal_apps/t4_objective_reset_app.py",
    ]
    footprint = branch_footprint()
    receipt = {
        "schema_version": "t4_reset_verification_v1",
        "observed_at_utc": _stamp(),
        "branch_footprint": footprint,
        "failure_provenance": failure_provenance(full, footprint),
        "focused": focused,
        "handoff_tests": junit(folder / "handoff.xml"),
        "milestone_tests": junit(folder / "milestone.xml"),
        "full_suite": full,
        "full_suite_classification": (
            json.loads((folder / "full_suite_classification.json").read_text())["payload"]
            if (folder / "full_suite_classification.json").exists()
            else None
        ),
        "repository_lint": {
            "scope": "src/ only, matching the baseline this receipt is compared against",
            "count": len(lint),
            "violations_in_changed_source_files": len(new_lint),
            "sha256": sha256_file(folder / "repository_lint.json"),
            "whole_repository": {
                "scope": "the entire worktree, disclosed so the narrower number is not mistaken for it",
                "count": json.loads((folder / "repository_lint_all_summary.json").read_text())[
                    "count"
                ]
                if (folder / "repository_lint_all_summary.json").exists()
                else None,
                "summary": "diagnostics/t4_objective_reset/verification/repository_lint_all_summary.json",
            },
        },
        "files_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
        "passed": focused["status"] == full["status"] == "passed" and not lint,
        "branch_tests_passed": (
            focused["status"] == "passed"
            and junit(folder / "handoff.xml")["status"] == "passed"
            and junit(folder / "milestone.xml")["status"] == "passed"
        ),
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
