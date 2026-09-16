"""Classify a completed repository-wide suite run without converting failure into pass.

The reset milestone requires one repository-wide verification at the scientific-launch
boundary. That run is only useful if its failures can be attributed, so this tool
answers two separate questions and keeps them separate:

1. **Provenance.** Did this branch cause any of them? Settled structurally: if every
   executable change on the branch is a new file, no pre-existing module's behaviour
   moved, and only a test that walks the repository could still be affected by an
   addition. Those are listed by name for individual reading, never waved through.
2. **Cause.** Each failing node is bucketed by its own error text, and a second run of
   exactly the failing nodes in isolation separates real failures from ones that only
   appear in full-suite order.

Nothing here marks a failing suite as passing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "2114c405"

# Buckets are matched against the node's own error text, in order. The first three are
# environment facts this checkout cannot satisfy; the repository's production identity
# is pinned by the Modal image, not by this venv.
BUCKETS = (
    ("ringcore_catalog_drift", ("82fd910cafe2eeb7", "catalog fingerprint drifted")),
    ("torch_grad_mode_leak", ("does not require grad",)),
    ("missing_vendored_artifact", ("FileNotFoundError", "ModuleNotFoundError")),
    ("stale_bytecode_source_lookup", ("could not get source code",)),
)

SCANNING_MARKERS = ("rglob", "iterdir", "os.walk", "glob(")


def failing_nodes(path: Path) -> dict[str, str]:
    """Node identity to its error text, for every failure and every error."""
    root = ET.parse(path).getroot()
    result = {}
    for case in root.iter("testcase"):
        node = case.find("failure")
        if node is None:
            node = case.find("error")
        if node is None:
            continue
        node_id = "{}::{}".format(
            case.get("classname", "").replace(".", "/") + ".py", case.get("name")
        )
        result[node_id] = (node.get("message") or "") + "\n" + (node.text or "")
    return result


def counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def bucket(text: str) -> str:
    for name, markers in BUCKETS:
        if any(marker in text for marker in markers):
            return name
    return "unclassified"


def branch_footprint() -> dict:
    rows = [
        line.split("\t", 1)
        for line in subprocess.check_output(
            ["git", "diff", "--name-status", BASE_REVISION, "HEAD"], cwd=ROOT, text=True
        ).splitlines()
        if line
    ]
    modified = sorted(path for status, path in rows if status != "A")
    executable_modified = [p for p in modified if not p.endswith((".md", ".txt", ".rst"))]
    return {
        "base_revision": BASE_REVISION,
        "executable_files_added": sorted(
            path
            for status, path in rows
            if status == "A"
            and path.split("/")[0] in {"src", "tools", "tests", "modal_apps", "scripts"}
        ),
        "files_modified": modified,
        "executable_files_modified": executable_modified,
        "additions_only": not executable_modified,
    }


def scanning_tests(nodes) -> list[str]:
    """Failing nodes whose test module walks the repository.

    An added file cannot change a pre-existing module's behaviour, but it can change
    what a repository scan sees, so these are the only inherited failures that need
    individual reading.
    """
    found = []
    for node in nodes:
        source = ROOT / node.split("::", 1)[0]
        if source.exists() and any(marker in source.read_text() for marker in SCANNING_MARKERS):
            found.append(node)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-suite", type=Path, required=True)
    parser.add_argument("--recheck", type=Path, help="isolated rerun of exactly the failing nodes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full = failing_nodes(args.full_suite)
    footprint = branch_footprint()
    branch_tests = {
        path for path in footprint["executable_files_added"] if path.startswith("tests/")
    }
    from_branch = sorted(node for node in full if node.split("::", 1)[0] in branch_tests)

    # Residual risk, measured rather than argued: does any failure's own text name a
    # file this branch added? A repository scan that a new file broke would say so.
    added = [path for path in footprint["executable_files_added"] if not path.startswith("tests/")]
    naming_branch_file = {
        node: hits
        for node, text in full.items()
        if (hits := sorted(path for path in added if Path(path).name in text))
    }

    payload = {
        "schema_version": "t4_reset_suite_classification_v1",
        "full_suite": {
            "counts": counts(args.full_suite),
            "sha256": sha256_file(args.full_suite),
            "status": "failed",
            "buckets": dict(sorted(Counter(bucket(text) for text in full.values()).items())),
        },
        "provenance": {
            "branch_footprint": footprint,
            "branch_test_files": sorted(branch_tests),
            "failing_nodes_in_branch_test_files": from_branch,
            "inherited_failures_in_repository_scanning_tests": scanning_tests(full),
            "failures_whose_text_names_a_branch_added_file": naming_branch_file,
            "structural_claim": (
                "every executable change on this branch is a new file, so no pre-existing "
                "module changed and an inherited failure cannot have been introduced here"
                if footprint["additions_only"]
                else "this branch modifies pre-existing executable files, so inherited "
                "failures cannot be classified structurally"
            ),
        },
        "completed_at_utc": _stamp(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "interpretation": (
            "a failing repository-wide suite is reported as failing; this classification "
            "explains and attributes the failures and confers no launch authority"
        ),
    }

    if args.recheck:
        again = failing_nodes(args.recheck)
        recovered = sorted(set(full) - set(again))
        payload["isolated_recheck"] = {
            "counts": counts(args.recheck),
            "sha256": sha256_file(args.recheck),
            "nodes_rerun": len(full),
            "passed_in_isolation": len(recovered),
            "still_failing": len(again),
            "buckets": dict(sorted(Counter(bucket(text) for text in again.values()).items())),
            "recovered_nodes": recovered,
            "interpretation": (
                "a node that passes alone and fails in the full run is cross-test state "
                "pollution in that order, not a defect this branch introduced; it is "
                "still a real repository problem and is not waived"
            ),
        }

    seal(args.output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "provenance"}, indent=2)[:2400])
    print(
        json.dumps(
            {
                "additions_only": footprint["additions_only"],
                "failing_nodes_in_branch_test_files": from_branch,
                "scanning_tests_needing_manual_reading": payload["provenance"][
                    "inherited_failures_in_repository_scanning_tests"
                ],
                "failures_whose_text_names_a_branch_added_file": naming_branch_file,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
