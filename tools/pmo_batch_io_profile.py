"""Replay a completed PMO ledger to compare persistence barriers, not task scores."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from itertools import groupby
from pathlib import Path
from time import perf_counter

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_macro_probe import DurableScores
from compose_v4.experiments.t4_matched_pilot import _stamp, seal


def profile(input_path, output, root):
    data = json.loads(input_path.read_text())
    rows = data["oracle_rows"]
    if (
        data["status"] != "complete_bounded_development"
        or len(rows) != data["new_oracle_calls"]
        or [r["index"] for r in rows] != list(range(len(rows)))
        or any(r["status"] != "complete" for r in rows)
        or len({r["smiles"] for r in rows}) != len(rows)
    ):
        raise ValueError(f"{input_path}: expected a complete, ordered, unique oracle ledger")
    values = {r["smiles"]: r["score"] for r in rows}
    groups = [list(g) for _, g in groupby(rows, key=lambda r: (r["role"], r["lock_path"]))]
    results = {}
    with tempfile.TemporaryDirectory(prefix="compose-pmo-io-") as temporary:
        for mode in ("serial", "locked_batch"):
            folder, count, calls = Path(temporary) / mode, [0], []

            def barrier(count=count):
                count[0] += 1  # Count barriers; do NOT emulate network latency.

            def replay(s, calls=calls):
                calls.append(s)
                return values[s]  # Existing observation, never an oracle request.

            scores = DurableScores(folder, replay, len(rows), barrier, {})
            started = perf_counter()
            for i, group in enumerate(groups):
                lock = folder / f"lock_{i}.json"
                names = [r["smiles"] for r in group]
                seal(lock, {"smiles": names})
                scores.context = {"role": group[0]["role"], "lock_path": str(lock)}
                with redirect_stdout(StringIO()):
                    result = (
                        scores.score_many(names)
                        if mode == "locked_batch"
                        else [scores.score(s) for s in names]
                    )
                if [r["desirability"] for r in result] != [r["score"] for r in group]:
                    raise ValueError("serial/batch replay changed observed scores")
            if calls != [r["smiles"] for r in rows] or scores.meter.spent != len(rows):
                raise ValueError("serial/batch replay changed query order or accounting")
            results[mode] = {
                "replayed_calls": len(calls),
                "synchronous_barriers": count[0],
                "local_replay_seconds": perf_counter() - started,
                "score_order_sha256": identity([(r["smiles"], r["score"]) for r in scores.rows]),
            }
    sources = [
        "tools/pmo_batch_io_profile.py",
        "src/compose_v4/experiments/pmo_macro_probe.py",
        "src/compose_v4/experiments/pmo_trajectory_value.py",
        "src/compose_v4/benchmark/molleo_task3.py",
        "src/compose_v4/experiments/t4_matched_pilot.py",
        "src/compose_v4/experiments/t4_macro_beam.py",
    ]
    report = {
        "schema_version": "pmo_locked_batch_io_replay_v1",
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "implementation_sha256": {p: sha256_file(root / p) for p in sources},
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "uncommitted_development": True,
        "configuration": {
            "groups": [len(g) for g in groups],
            "seed": None,
            "ordering": "recorded query index",
        },
        "split": "replay of already observed development data; no model selection",
        "excluded_rows": 0,
        "software": {
            "python": platform.python_version(),
            **{p: importlib.metadata.version(p) for p in ("numpy", "rdkit")},
        },
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
        "precision": "existing Python float labels, exact equality",
        "at": _stamp(),
        "new_oracle_calls": 0,
        "results": results,
        "interpretation": "Exact replay/flush-count comparison, not measured Modal latency or a new PMO result; local duration excludes real volume synchronization.",
    }
    report["report_sha256"] = identity(report)
    publish_json(output, report)
    print(
        json.dumps(
            {"output": str(output), "groups": report["configuration"]["groups"], "results": results}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile(args.input.resolve(), args.output.resolve(), Path(__file__).resolve().parents[1])
