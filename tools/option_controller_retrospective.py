"""Run the zero-oracle retrospective option-controller gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.experiments.option_controller_retrospective import analyze

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    report = analyze(data, input_path=args.input, repo_root=ROOT)
    digest = publish_json(args.output, report)
    print(json.dumps({"output": str(args.output), "sha256": digest, "status": report["status"]}))


if __name__ == "__main__":
    main()
