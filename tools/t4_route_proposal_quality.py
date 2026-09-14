#!/usr/bin/env python3
"""Evaluate frozen T4 route-proposal pools without calling an oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.route_proposal_quality import evaluate

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_payload(path):
    value = json.loads(path.read_text())
    if set(value) == {"payload", "payload_sha256"}:
        if identity(value["payload"]) != value["payload_sha256"]:
            raise ValueError(f"sealed proposal-quality input changed: {path}")
        return value["payload"]
    return value


def run(input_path, output_path):
    if output_path.exists():
        raise ValueError(
            f"refusing to overwrite proposal-quality result: {output_path}"
        )
    payload = load_payload(input_path)
    result = evaluate(payload)
    body = {
        **{key: value for key, value in result.items() if key != "report_sha256"},
        "inputs": {str(input_path): sha256_file(input_path)},
        "implementation": {
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
            "metric_source_sha256": sha256_file(
                ROOT / "src/compose_v4/experiments/route_proposal_quality.py"
            ),
            "runner_sha256": sha256_file(Path(__file__)),
        },
    }
    body["report_sha256"] = identity(body)
    envelope = {"payload": body, "payload_sha256": identity(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(output_path)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(
        json.dumps(
            {
                "report_sha256": result["report_sha256"],
                "policies": {
                    name: row["aggregate"] for name, row in result["policies"].items()
                },
                "oracle_calls": result["oracle_calls"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
