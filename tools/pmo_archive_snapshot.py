"""Read a completed prefix of a live archive pilot; no oracle or remote job calls."""

import argparse
import json
import platform
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
from rdkit import rdBase

from compose_v4.control.graph_geometry import structural_displacement
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", choices=("balanced", "adaptive"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.run):
        parser.error("--run must be a 64-character hexadecimal run identity")
    volume = modal.Volume.from_name("compose-v4-artifacts")
    prefix = f"pmo_archive_pilot/{args.run}/perindopril_mpo__{args.arm}__0"
    source_identity = args.output / "source_identity.json"
    identity = {"volume": "compose-v4-artifacts", "prefix": prefix}
    if source_identity.exists() and json.loads(source_identity.read_text()) != identity:
        raise ValueError("snapshot output belongs to another remote source")
    publish_json(source_identity, identity)
    paths = []

    def read(name):
        path = args.output / f"{name}.json"
        if not path.exists():
            data = b"".join(volume.read_file(f"{prefix}/{name}.json"))
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            unseal(temporary)
            temporary.replace(path)
        paths.append(path)
        return unseal(path)

    progress = read("progress_result")  # lock one finite prefix, even while job advances
    count = progress["attempt"] + 1

    def attempt(index):
        name = f"attempts/{index:04}"
        record, feedback = read(name), read(f"feedback/{index:04}")
        if record["status"] != feedback["status"] or record["status"] == "running":
            raise ValueError("snapshot is not a completed attempt")
        row = {**feedback, "id": name}
        if "candidate" in record:
            candidate = record["candidate"]
            source, product = (
                decode_search_state(record["source"]),
                decode_search_state(candidate["node"]),
            )
            measured = structural_displacement(
                source.graph, product.graph, source.lineage, product.lineage
            )
            if measured != candidate["structural_change"]:
                raise ValueError(
                    "recorded structural change differs from exact-state recomputation"
                )
            row.update(
                {
                    k: candidate[k]
                    for k in (
                        "smiles",
                        "chain",
                        "primitives",
                        "primitive_count",
                        "structural_change",
                        "topology",
                    )
                }
            )
            row["r_release"] = candidate["bundle"]["r_release"]
        return row

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(attempt, range(count)))
    completed = [r for r in rows if r["status"] == "complete"]
    constructors = [r for r in rows if (r["option"] or "").startswith("construct:")]
    for row in completed:
        row["descendants"] = sum(row["id"] in child.get("chain", [])[:-1] for child in completed)
    result = {
        "schema_version": "pmo_archive_snapshot_v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "volume": "compose-v4-artifacts",
        "prefix": prefix,
        "snapshot_attempts": count,
        "oracle_calls": progress["calls"],
        "best": progress["best"],
        "top10_mean": progress["top10_mean"],
        "source_hashes": {str(p.relative_to(args.output)): sha256_file(p) for p in sorted(paths)},
        "script_sha256": sha256_file(Path(__file__)),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "analysis_scope": {
            "split": "development",
            "workers": 4,
            "oracle_calls": 0,
            "random_sampling": False,
            "primitive_replay": "producer receipt only",
        },
        "completed": len(completed),
        "outcomes": dict(Counter(r["status"] for r in rows)),
        "constructor_attempts": len(constructors),
        "constructor_completed": sum(r["status"] == "complete" for r in constructors),
        "ring_increase_candidates": sum(
            r["structural_change"]["d_cycle_rank"] > 0 for r in completed
        ),
        "max_option_depth": max((len(r["chain"]) for r in completed), default=0),
        "max_primitive_depth": max((r["primitives"] for r in completed), default=0),
        "rows": rows,
        "interpretation": "Fixed completed-prefix operational diagnostic, not full-budget comparison. Topology recomputed from exact states; no scoring or winner inputs.",
    }
    publish_json(args.output / "summary.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "source_hashes")}, indent=2
        )
    )
    for row in constructors:
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "id",
                        "option",
                        "status",
                        "score",
                        "chain",
                        "structural_change",
                        "descendants",
                    )
                    if k in row
                }
            )
        )


if __name__ == "__main__":
    main()
