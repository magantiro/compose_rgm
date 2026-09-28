"""Preserve local parity receipts and test outcomes; never generate molecules."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    return {
        "path": str(path),
        "sha256": sha(path),
        **{
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "errors", "failures", "skipped")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-artifacts", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="existing test-results directory; new report/receipt files only",
    )
    args = parser.parse_args()
    report = args.output / "verification.json"
    if report.exists() or (args.output / "parity").exists():
        raise FileExistsError("verification report or parity receipts already exist")
    root = Path.cwd()
    registry = json.loads((root / "experiments/fragments/assets.json").read_text())
    tasks = set(registry["task_assets"])
    receipts = {}
    for case in sorted(args.test_artifacts.glob("test_source_export*")):
        if case.is_symlink():
            continue
        source = case / "result"
        if not source.is_dir():
            continue
        parity = json.loads((source / "parity.json").read_text())
        task = parity["task"]
        if task in receipts or task not in tasks:
            raise ValueError(f"unexpected or duplicate parity task: {task}")
        target = args.output / "parity" / task
        target.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name in ("parity.json", "provenance.json", "result.json", "worker.log"):
            shutil.copyfile(source / name, target / name)
            hashes[name] = sha(target / name)
            if hashes[name] != sha(source / name):
                raise ValueError(f"copy changed: {source / name}")
        receipts[task] = {
            "passed": parity["passed"],
            "attempts": parity["attempts_checked"],
            "source": str(source),
            "files_sha256": hashes,
        }
    if set(receipts) != tasks:
        raise ValueError(f"missing parity tasks: {tasks - set(receipts)}")
    payload = {
        "schema": "compose_fragment_runtime_verification_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "reporter_sha256": sha(Path(__file__)),
        "adapter_sha256": {
            p.name: sha(p)
            for p in sorted((root / "src/compose_v4/experiments/fragments").glob("*.py"))
        },
        "task_manifest_sha256": sha(root / "experiments/fragments/manifest.json"),
        "asset_registry_sha256": sha(root / "experiments/fragments/assets.json"),
        "test_results": {p.stem: junit(p) for p in sorted(args.output.glob("*.xml"))},
        "parity": receipts,
        "scope": "local source-export saved-attempt parity and workflow tests, not a new benchmark campaign",
        "limitations": [
            "Full-suite collection fails on a pre-existing Python 3.11 protocol introspection test.",
            "Repository-wide Ruff still reports 513 pre-existing findings outside this package.",
            "Whole-panel generation, other hardware/OS, public asset distribution and raw-panel replay are not verified here.",
            "Original files are preserved; local copies do not provide off-machine backup.",
        ],
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(report)


if __name__ == "__main__":
    main()
