"""Bank and independently replay a completed five-source support certificate."""

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

from compose_v4.control.supported_route import exact_key
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import software
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    spawn_path = args.result.with_name("spawn.json")
    spawn = json.loads(spawn_path.read_text())
    snapshot_path = args.result.with_name("source_snapshot.json")
    snapshot = json.loads(snapshot_path.read_text())
    archive = args.result.with_name("source_snapshot.tar.gz")
    verify_file(archive, snapshot["archive_sha256"])
    sources = spawn["task"]["image_revision"]["serialized_sources"]
    with tarfile.open(archive, "r:gz") as tar:
        for p, expected in sources.items():
            if hashlib.sha256(tar.extractfile(p).read()).hexdigest() != expected:
                raise ValueError(f"executed source archive mismatch: {p}")
    if result["run_id"] != spawn["run_id"] or len(result["results"]) != 5:
        raise ValueError("route result/run census mismatch")
    system = editing_v2_semantic_rewrite_system()
    rows = []
    for row in result["results"]:
        graph = decode_state(row["source_state"])
        families = Counter()
        for step in row["steps"]:
            family, action = decode_action(step["action"])
            if exact_key(graph) != step["source_key"] or step["selected_mark_probability"] <= 0:
                raise ValueError("nonpositive or mismatched saved production witness")
            graph = system.apply(graph, family, action)
            if encode_state(graph) != step["state"]:
                raise ValueError("local exact-state primitive replay differs")
            families[family] += 1
        if canonical_state_key(graph) != row["endpoint"] or row["endpoint"] != row["target"]:
            raise ValueError("supported route endpoint does not equal known target")
        rows.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "source",
                        "status",
                        "original_steps",
                        "completed_original_steps",
                        "primitive_steps",
                        "seconds",
                        "law_work",
                    )
                },
                "independent_exact_replay": True,
                "families": dict(families),
                "lowered_transitions": sum(len(t["steps"]) > 1 for t in row["transitions"]),
            }
        )
    publish_json(
        args.output,
        {
            "schema_version": "route_support_report_v1",
            "run_id": result["run_id"],
            "inputs": {
                str(p): sha256_file(p) for p in (args.result, spawn_path, snapshot_path, archive)
            },
            "analyzer_sha256": sha256_file(Path(__file__)),
            "base_commit": result["code_revision"],
            "executed_revision": spawn["task"]["image_revision"],
            "analysis_software": software(),
            "proposal_probabilities": "recorded from qualified remote law; not recomputed by local torch",
            "rows": rows,
            "original_steps": sum(r["original_steps"] for r in rows),
            "supported_steps": sum(r["primitive_steps"] for r in rows),
            "oracle_calls": 0,
            "scope": "five answer-known paths, all exact endpoints reached; not blind discovery, shortest paths, or benchmark performance",
        },
    )
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
