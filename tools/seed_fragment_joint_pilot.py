"""Reuse the identical seed-zero support prefix, without rerunning its panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summary_path = args.support_dir / "summary.json"
    manifest_path = args.support_dir / "manifest.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        not summary["support_pass"]
        or summary["mode"] != "support"
        or manifest["mode"] != "support"
        or manifest["seed"] != 0
        or manifest["attempts_per_prompt"] != 2
        or manifest["candidate_attempts"] != 8
        or len(summary["row_sha256"]) != 20
    ):
        raise ValueError("requires the complete qualified seed-zero support prefix")
    for path, expected in manifest["inputs"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"support input changed: {path}")
    copies = {}
    for name, expected in sorted(summary["row_sha256"].items()):
        row_path = args.support_dir / name
        if digest(row_path) != expected:
            raise ValueError(f"support row changed: {name}")
        row = json.loads(row_path.read_text())
        if len(row["attempts"]) != 2:
            raise ValueError("support row has unexpected prefix length")
        for index, attempt in enumerate(row["attempts"]):
            relative = f"attempts/{row['task']}_{row['drug']}_{index:03d}.json"
            path = args.support_dir / relative
            if json.loads(path.read_text()) != attempt or not attempt["complete"]:
                raise ValueError(f"support attempt not identical to sealed row: {relative}")
            if attempt["attempt_index"] != index or len(attempt["offered"]) != 8:
                raise ValueError("unexpected attempt identity or candidate count")
            if relative in copies:
                raise ValueError("duplicate support attempt")
            copies[relative] = path
    if len(copies) != 40:
        raise ValueError("support prefix does not contain exactly 40 attempts")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".pilot-prefix-") as temp:
        stage = Path(temp) / "complete"
        (stage / "attempts").mkdir(parents=True)
        for relative, path in sorted(copies.items()):
            shutil.copyfile(path, stage / relative)
        receipt = {
            "schema": "fragment_joint_support_prefix_reuse_v1",
            "role": "identical seed-zero panels reused; not independent support/pilot replication",
            "selection": "all 40 preview attempts, including any no-output attempts; no quality selection",
            "prefix_length_per_prompt": 2,
            "pilot_length_per_prompt": 20,
            "planned_new_attempts": 360,
            "source_sha256": {
                str(p.resolve()): digest(p) for p in [summary_path, manifest_path, Path(__file__)]
            },
            "attempt_sha256": {relative: digest(path) for relative, path in sorted(copies.items())},
            "python": platform.python_version(),
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        (stage / "prefix_reuse.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        stage.rename(args.output_dir)
    print(json.dumps({"reused_attempts": 40, "new_attempts": 360}))


if __name__ == "__main__":
    main()
