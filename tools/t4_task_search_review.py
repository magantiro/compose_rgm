"""Project a verified production lock into a compact, hash-bound audit report."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def review(directory: Path):
    script = Path(__file__)
    relative = script.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if committed != script.read_bytes():
        raise ValueError("commit the exact projection script before generating the report")
    paths = {"result": directory / "result.json", "lock": directory / "cache/candidate_lock.json"}
    result = json.loads(paths["result"].read_text())
    envelope = json.loads(paths["lock"].read_text())
    if hashlib.sha256(canonical(envelope["payload"])).hexdigest() != envelope["payload_sha256"]:
        raise ValueError("candidate lock envelope is corrupt")
    if result["verification"]["status"] != "verified" or result["candidate_lock_sha256"] != sha(
        paths["lock"]
    ):
        raise ValueError("result does not verify this physical candidate lock")
    lock = envelope["payload"]["lock"]
    if lock["schema_version"] != "t4_hierarchical_candidate_lock_v2":
        raise ValueError("foreign candidate schema")
    fields = (
        "smiles",
        "parent_lineage_id",
        "option",
        "primitive_depth",
        "r_release",
        "r_coherent",
        "r_change",
        "d_cycle_rank",
        "d_ring_systems",
        "d_heavy",
        "qed",
        "sa",
        "sim",
        "v",
        "predicted_docking",
        "cumulative_change",
    )
    pool = lock["pool"]
    report = {
        "schema_version": "t4_task_search_review_v1",
        "input_paths": {k: str(p.resolve()) for k, p in paths.items()},
        "input_sha256": {k: sha(p) for k, p in paths.items()},
        "producer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "producer_sha256": sha(script),
        "producer_path": relative,
        "scientific_code_revision": result["code_revision"],
        "software": {"python": platform.python_version(), "generation": lock["software"]},
        "seed": None,
        "seed_reason": "deterministic projection; generation seeds are in source configuration",
        "configuration": result["configuration"],
        "new_oracle_calls": 0,
        "decision": result["decision"],
        "pool": [{k: c[k] for k in fields} for c in pool],
        "take": [{k: c[k] for k in fields} for c in lock["take"]],
        "n_pool": len(pool),
        "n_feasible": sum(c["v"] == 0 for c in pool),
        "construction_options_completed": sum(
            e["stage"] == "how"
            and e.get("option_completed", False)
            and e["option"].startswith("construct:")
            for u in lock["work"]
            for e in u["path"]
        ),
        "primitive_depths": [sum(e["stage"] == "how" for e in u["path"]) for u in lock["work"]],
        "scope_range": [min(c["r_release"] for c in pool), max(c["r_release"] for c in pool)]
        if pool
        else None,
        "coherent_range": [min(c["r_coherent"] for c in pool), max(c["r_coherent"] for c in pool)]
        if pool
        else None,
        "positive_cycle_deltas": sum(c["d_cycle_rank"] > 0 for c in pool),
        "negative_cycle_deltas": sum(c["d_cycle_rank"] < 0 for c in pool),
        "ring_system_deltas": dict(sorted(Counter(str(c["d_ring_systems"]) for c in pool).items())),
        "overlapping_constraint_failures": {
            "qed": sum(c["qed"] < 0.6 for c in pool),
            "sa": sum(c["sa"] > 4 for c in pool),
            "similarity": sum(c["sim"] < 0.4 for c in pool),
        },
    }
    output = directory / "review.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_bytes(canonical(report) + b"\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = review(args.directory)
    print(
        {
            k: result[k]
            for k in ("decision", "n_pool", "n_feasible", "construction_options_completed")
        }
    )
