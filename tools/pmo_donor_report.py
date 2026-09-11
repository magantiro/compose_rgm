"""Reconcile complete saved donor programs, labels and lineage, without rescoring."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.graph_geometry import structural_displacement
from compose_v4.control.region_rewrite import Lineage
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.directory / "result.json"
    result = json.loads(path.read_text())
    verify_file(args.directory / "batch.json", result["batch_sha256"])
    if identity(result["configuration"]) != result["configuration_sha256"]:
        raise ValueError("donor configuration identity mismatch")
    oracle_files = sorted((args.directory / "oracle").rglob("*.json"))
    oracle_rows = [json.loads(p.read_text()) for p in oracle_files]
    if len(oracle_rows) != result["physical_calls_this_probe"] or any(
        r["status"] != "complete" for r in oracle_rows
    ):
        raise ValueError("oracle count or completion mismatch")
    measured = {r["smiles"]: r["score"] for r in oracle_rows}
    if max(measured.values()) != result["best"]:
        raise ValueError("reported best differs from the measured archive")
    changes, inputs = [], {str(path): sha256_file(path)}
    for attempt in result["attempts"]:
        p = args.directory / f"attempts/{attempt['index']:04}.json"
        inputs[str(p)] = sha256_file(p)
        row = json.loads(p.read_text())["result"]
        if row["status"] != attempt["status"]:
            raise ValueError("attempt summary differs from raw receipt")
        if row["status"] != "compiled":
            continue
        first, last = decode_state(row["states"][0]), decode_state(row["states"][-1])
        if (
            len(row["states"]) != len(row["actions"]) + 1
            or canonical_state_key(last) != row["smiles"]
        ):
            raise ValueError("endpoint or primitive census mismatch")
        if measured[row["smiles"]] != attempt["score"]:
            raise ValueError("compiled candidate score differs from measured value")
        initial_lineage = Lineage.initial(np.flatnonzero(is_element(first.atom_types)))
        final_lineage = initial_lineage
        for action in row["actions"]:
            family, mark = decode_action(action)
            final_lineage = final_lineage.observe(family, mark)
        changes.append(
            {
                "attempt": attempt["index"],
                "score": attempt["score"],
                "primitives": row["primitive_steps"],
                "intended_release": row["released_fraction"],
                **structural_displacement(first, last, initial_lineage, final_lineage),
            }
        )
    for p in [
        args.directory / "batch.json",
        args.directory / "configuration.json",
        args.directory / "serialization_repair.json",
        *oracle_files,
    ]:
        inputs[str(p)] = sha256_file(p)
    summary = {
        k: result[k]
        for k in (
            "status",
            "configuration_sha256",
            "initial_best",
            "best",
            "initial_top10_mean",
            "top10_mean",
            "historical_prescreen_reported_calls",
            "historical_provenance_limit",
            "initial_supported_unique",
            "historical_parity_repeat_calls",
            "new_endpoint_calls",
            "physical_calls_this_probe",
            "status_counts",
            "proposal_seconds",
            "oracle_seconds",
            "top10",
            "decision",
        )
    }
    summary.update(
        schema_version="donor_probe_audit_v1",
        input_sha256=inputs,
        analyzer_sha256=sha256_file(Path(__file__)),
        new_oracle_calls=0,
        producer_configuration=result["configuration"],
        structural_changes=changes,
        compiled_programs=len(changes),
        cycle_rank_deltas=dict(Counter(r["d_cycle_rank"] for r in changes)),
        ring_system_deltas=dict(Counter(r["d_ring_systems"] for r in changes)),
        recorded_proposal_seconds_note="sum of completed attempt timings across original and resumed process; excludes failed serialization attempt and restart overhead",
        execution_support="executor verified by producer; learned-reference support unverified",
        candidate_novelty="94 not in the top-100 bank; full 249455 prescreen membership not yet checked",
    )
    publish_json(args.output, summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "best",
                    "top10_mean",
                    "compiled_programs",
                    "cycle_rank_deltas",
                    "proposal_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
