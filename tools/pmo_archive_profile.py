"""Zero-oracle profile of selection/setup using one saved production law.

This excludes model enumeration and Modal I/O. It does not estimate whole-run
throughput; the first four real attempts provide that measurement.
"""

import argparse
import cProfile
import json
import pstats
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import EXECUTABLE_PRODUCT_GATE, OptionContinuationKernel
from compose_v4.experiments.continuation_profile import ExecutorMeter, publish_json, sha256_file
from compose_v4.experiments.pmo_archive_pilot import load_contract
from compose_v4.experiments.t4_macro_beam import exact_archive_graph
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import encode_state

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--law", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(ROOT)
    if rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("profile requires pinned RDKit")
    graph = exact_archive_graph(contract["roots"][0])
    payload = unseal(args.law)
    if payload["source"] != encode_state(graph):
        raise ValueError("saved law does not match exact first root")
    pairs = [
        (action_codec_v4 if m["schema_version"] == 4 else action_codec).decode_action(m)
        for m in payload["marks"]
    ]

    def law(state):
        if encode_state(state) != payload["source"]:
            raise ValueError("profile cannot enumerate a new state")
        return (
            tuple(f for f, _ in pairs),
            tuple(a for _, a in pairs),
            tuple(payload["probabilities"]),
        )

    hierarchy = MolecularHierarchy(
        OptionContinuationKernel(
            law,
            editing_v2_semantic_rewrite_system(),
            max_executor_applications=None,
            product_gate=EXECUTABLE_PRODUCT_GATE,
        ),
        lazy_applicability=True,
        include_carbonyl_options=True,
    )
    node = MolecularSearchState.start(graph, budget=11, root_id="profile_first_root")
    meter, profiler, timings = ExecutorMeter(None), cProfile.Profile(), {}
    with meter.instrument(), profiler:
        start = perf_counter()
        where = hierarchy.row(node)
        selected = max(where.successors, key=lambda n: n.region.released_fraction)
        timings["where_seconds"] = perf_counter() - start
        start = perf_counter()
        what = hierarchy.row(selected)
        timings["what_cold_seconds"] = perf_counter() - start
        start = perf_counter()
        again = hierarchy.row(selected)
        timings["what_warm_seconds"] = perf_counter() - start
        if what.labels != again.labels or not np.array_equal(what.reference, again.reference):
            raise ValueError("cached applicability changed reference row")
        start = perf_counter()
        generic = what.successors[what.labels.index("generic")]
        following = hierarchy.sample_reference(generic, np.random.default_rng(0))
        timings["generic_first_primitive_seconds"] = perf_counter() - start
    stats = pstats.Stats(profiler)
    rows = [
        {
            "file": k[0],
            "line": k[1],
            "function": k[2],
            "calls": v[1],
            "self_seconds": v[2],
            "cumulative_seconds": v[3],
        }
        for k, v in sorted(stats.stats.items(), key=lambda item: -item[1][3])[:30]
    ]
    result = {
        "schema_version": "pmo_archive_cached_setup_profile_v1",
        "inputs": {"law_path": str(args.law), "law_sha256": sha256_file(args.law)},
        "contract_sha256": contract["contract_sha256"],
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": sha256_file(Path(__file__)),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "seed": 0,
        "root": 0,
        "region_rule": "largest release",
        "oracle_calls": 0,
        "new_model_enumerations": 0,
        "executor_calls": meter.calls,
        "timings": timings,
        "regions": len(where.labels),
        "options": len(what.labels),
        "release": selected.region.released_fraction,
        "generic_product": following is not None,
        "profile": rows,
        "interpretation": "local cached-law setup only; excludes model enumeration and Modal I/O",
    }
    publish_json(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("timings", "regions", "options", "executor_calls", "oracle_calls")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
