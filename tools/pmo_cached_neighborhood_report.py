"""Audit paid cached-neighborhood results without repeating oracle calls."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.pmo_cached_neighborhood import query_subset


def report(directory, history_path):
    store = Store(directory, lambda: None)
    result, config, lock, generation = (
        store.read(k) for k in ("result", "configuration", "query_lock", "generation_lock")
    )
    for key in ("configuration", "query_lock"):
        verify_file(directory / f"{key}.json", result[f"{key}_sha256"])
    verify_file(directory / "generation_lock.json", lock["generation_sha256"])
    verify_file(history_path, lock["history_sha256"])
    snapshot = history_path.with_name("source_snapshot.tar.gz")
    verify_file(snapshot, lock["snapshot_sha256"])
    for path, digest in config["inputs"].items():
        verify_file(Path(path), digest)
    with tarfile.open(directory / "source_snapshot.tar.gz") as archive:
        for p, h in config["implementation"].items():
            if hashlib.sha256(archive.extractfile(p).read()).hexdigest() != h:
                raise ValueError(f"executed neighborhood source differs: {p}")
    history = json.loads(history_path.read_text())
    with tarfile.open(snapshot) as archive:
        prepared = json.load(archive.extractfile(history["configuration"]["prepared"]["path"]))
    known = dict(prepared["observed"])
    known.update({q["smiles"]: q["score"] for q in history["oracle_rows"]})
    if lock["new"] != query_subset(generation["products"], known):
        raise ValueError("queries do not reproduce the declared random subset")
    if [q["smiles"] for q in result["oracle_rows"]] != lock["new"]:
        raise ValueError("oracle sequence differs from the locked query sequence")
    for i, q in enumerate(result["oracle_rows"]):
        if (
            q["status"] != "complete"
            or q["smiles"] in known
            or q["lock_sha256"] != result["query_lock_sha256"]
            or store.read(f"oracle/{i:04}/result") != q
        ):
            raise ValueError("query receipt incomplete, repeated, or unlocked")
        known[q["smiles"]] = q["score"]
    source, system = decode_state(generation["source"]), editing_v2_semantic_rewrite_system()
    expected = {r["smiles"]: r for r in generation["products"] if r["smiles"] in known}
    if len(expected) != len(result["rows"]):
        raise ValueError("scored support coverage does not reproduce")
    for row in result["rows"]:
        if (
            row["score"] != known[row["smiles"]]
            or {k: v for k, v in row.items() if k != "score"} != expected[row["smiles"]]
        ):
            raise ValueError("scored endpoint changed after locking")
        a = row["witnesses"][0]
        codec = action_codec_v4 if a["schema_version"] == 4 else action_codec
        graph = system.apply(source, *codec.decode_action(a))
        if encode_state(graph) != row["state"] or canonical_state_key(graph) != row["smiles"]:
            raise ValueError("scored endpoint lacks exact production-executor replay")
    best = max(result["rows"], key=lambda r: r["score"])
    if best["score"] != result["summary"]["best_neighbor"]:
        raise ValueError("reported best differs from paid scores")
    return {
        "schema_version": "pmo_cached_neighborhood_audit_v1",
        "input_sha256": {
            str(p): sha256_file(p)
            for p in (
                directory / "result.json",
                directory / "configuration.json",
                directory / "source_snapshot.tar.gz",
                history_path,
            )
        },
        "analyzer_sha256": sha256_file(Path(__file__)),
        "checks": "executed source bytes, input hashes, locked uniform subset, exact paid-query sequence, charged union and all 1044 scored endpoint primitive replays",
        "summary": result["summary"],
        "parent_improvers": sum(
            r["score"] > result["summary"]["initial_score"] for r in result["rows"]
        ),
        "oracle_seconds": sum(q["oracle_seconds"] for q in result["oracle_rows"]),
        "reused_generation_seconds": generation["generation_seconds"],
        "historical_prescreen_calls": result["historical_prescreen_calls"],
        "historical_development_physical_calls": result["historical_development_physical_calls"],
        "top_candidates": [
            {k: v for k, v in r.items() if k != "state"} for r in result["rows"][:3]
        ],
        "decision": "Existing primitive support includes a better endpoint, but 1024 extra queries for this small gain is not an oracle-efficient controller. Retain data for selecting local edits; do not promote exhaustive local enumeration or remove broad options.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.input, args.history)
    publish_json(args.output, result)
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "oracle_seconds": result["oracle_seconds"],
                "parent_improvers": result["parent_improvers"],
            }
        )
    )


if __name__ == "__main__":
    main()
