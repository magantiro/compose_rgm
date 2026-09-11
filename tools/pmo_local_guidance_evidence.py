"""Replay the saved winning lineage and audit one matched-parent ranking loss."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np

from compose_v4.control.graph_geometry import topology
from compose_v4.control.local_endpoint_selector import select
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import software
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state

ROOT = Path(__file__).resolve().parents[1]


def ranking_evidence(selection, smiles):
    if len(selection["pool"]) != len(selection["predictions"]):
        raise ValueError("saved ranking has unequal molecule and prediction counts")
    ranked = sorted(
        zip(selection["pool"], selection["predictions"], strict=True),
        key=lambda row: (-row[1], row[0]),
    )
    if len({s for s, _ in ranked}) != len(ranked):
        raise ValueError("saved ranking contains duplicate canonical molecules")
    return {
        "pool_size": len(ranked),
        "present": smiles in selection["pool"],
        "rank": next((i + 1 for i, row in enumerate(ranked) if row[0] == smiles), None),
        "prediction": dict(ranked).get(smiles),
        "top16_cutoff": ranked[min(15, len(ranked) - 1)][1] if ranked else None,
    }


def evidence(directory, audit_path):
    path = directory / "result.json"
    result = json.loads(path.read_text())
    audit = json.loads(audit_path.read_text())
    verify_file(path, audit["input_paths_sha256"][str(path)])
    if result["status"] != "complete_development" or audit["run_id"] != result["run_id"]:
        raise ValueError("completed comparison and verified audit disagree")
    for name, digest in result["image_revision"]["serialized_sources"].items():
        if name.startswith("src/"):
            verify_file(ROOT / name, digest)
    with tarfile.open(directory / "source_snapshot.tar.gz") as source:
        prepared = json.loads(
            source.extractfile(result["configuration"]["prepared"]["path"]).read()
        )

    local_path, generic_path, selection_path = (
        directory / name
        for name in (
            "winning_local_program.json",
            "winning_generic_program.json",
            "round1_matched_local_selection.json",
        )
    )
    local, generic, selection = map(unseal, (local_path, generic_path, selection_path))
    system = editing_v2_semantic_rewrite_system()
    graph = decode_state(local["source"])
    initial_topology = topology(graph)
    executed = []
    for mark, target in zip(local["actions"], local["states"], strict=True):
        codec = action_codec_v4 if mark["schema_version"] == 4 else action_codec
        family, action = codec.decode_action(mark)
        graph = system.apply(graph, family, action)
        if encode_state(graph) != target:
            raise ValueError("local primitive replay changed its exact successor")
        executed.append(family)
    if canonical_state_key(graph) != local["smiles"]:
        raise ValueError("local primitive replay changed its canonical successor")
    precursor_topology = topology(graph)
    for event in generic["events"]:
        if event["source"]["graph"] != encode_state(graph):
            raise ValueError("generic continuation is not connected to the witnessed precursor")
        if "mark" in event:
            mark = event["mark"]
            codec = action_codec_v4 if mark["schema_version"] == 4 else action_codec
            family, action = codec.decode_action(mark)
            graph = system.apply(graph, family, action)
            executed.append(family)
        if event["product"]["graph"] != encode_state(graph):
            raise ValueError("generic event changed an unexecuted molecular state")
    winner = max(
        (p for rd in result["rounds"] for p in rd["arms"]["guided"]["proposals"] if p),
        key=lambda p: p["score"],
    )
    if (
        encode_state(graph) != winner["node"]["graph"]
        or winner["score"] != result["arms"]["guided"]["best"]
    ):
        raise ValueError("replayed endpoint is not the reported guided winner")
    first = result["rounds"][0]["arms"]
    slot = next(
        i
        for i, p in enumerate(first["guided"]["proposals"])
        if p and p["smiles"] == local["smiles"]
    )
    parent, precursor, missed = (
        result["initial_parents"][slot],
        first["guided"]["proposals"][slot],
        first["baseline"]["proposals"][slot],
    )
    if (
        local["source"] != encode_state(decode_search_state(parent["node"]).graph)
        or winner["chain"] != precursor["chain"] + [winner["id"]]
        or selection["chosen"] != precursor["smiles"]
        or selection["model_sha256"] != result["configuration"]["endpoint_model_sha256"]
    ):
        raise ValueError("saved selection, model or winner ancestry mismatch")
    rng = np.random.default_rng(
        np.random.SeedSequence([result["configuration"]["seed"], 1, slot, 777])
    )
    if select(selection["pool"], np.asarray(selection["predictions"]), rng) != selection["chosen"]:
        raise ValueError("saved top-sixteen selection differs from its frozen RNG draw")
    initial = result["initial_metrics"]["best"]
    positive = winner["score"] > max(initial, result["arms"]["baseline"]["best"])
    paths = [
        path,
        audit_path,
        local_path,
        generic_path,
        selection_path,
        directory / "source_snapshot.tar.gz",
    ]
    return {
        "schema_version": "pmo_local_guidance_evidence_v1",
        "run_id": result["run_id"],
        "input_sha256": {str(p): sha256_file(p) for p in paths},
        "analyzer_sha256": sha256_file(Path(__file__)),
        "executed_revision": result["image_revision"],
        "analysis_software": software(),
        "configuration": result["configuration"],
        "initial_best": initial,
        "arms": result["arms"],
        "decision": "positive_best_score_earns_unchanged_replication"
        if positive
        else "stop_unchanged_mixture",
        "replicated": False,
        "historical_prescreen_calls": prepared["historical_prescreen_calls"],
        "development_physical_calls_including_run": prepared[
            "historical_development_physical_calls"
        ]
        + result["new_oracle_calls"],
        "new_analysis_oracle_calls": 0,
        "new_reference_model_calls": 0,
        "winner": {
            "smiles": winner["smiles"],
            "score": winner["score"],
            "precursor_score": precursor["score"],
            "parent_score": parent["score"],
            "programs": ["local_endpoint_selector", generic["bundle"]["option"]],
            "primitive_families": executed,
            "replayed_primitives": len(executed),
            "initial_topology": initial_topology,
            "precursor_topology": precursor_topology,
            "final_topology": topology(graph),
        },
        "posthoc_matched_parent_diagnostic": {
            "round": 1,
            "slot": slot,
            "baseline_actual": missed["score"],
            "baseline_move": ranking_evidence(selection, missed["smiles"]),
            "selected_actual": precursor["score"],
            "selected_move": ranking_evidence(selection, precursor["smiles"]),
            "interpretation": "The selector missed an immediately better legal move but retained the precursor of the eventual winner. This is not evidence of a learned future-value model.",
        },
        "limits": [
            "One exposed warm development seed; no replicated or matched external performance claim.",
            "Guided top-ten mean and archive diversity are lower than baseline in this run.",
            "Parent-normalized structural-change ratios can exceed one after growth from small parents.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evidence(args.directory, args.audit)
    publish_json(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("decision", "arms", "winner", "posthoc_matched_parent_diagnostic")
            }
        )
    )


if __name__ == "__main__":
    main()
