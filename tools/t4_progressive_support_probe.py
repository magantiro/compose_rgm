"""Conditional known-answer ring-panel probe; diagnostic only, zero oracle calls.

The teacher is supplied only to identify the diagnostic prefix and endpoint.
Nothing from this script or its result is loaded by the autonomous controller.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control import dynamic_program_synthesis_v1 as v1
from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from tools.t4_dynamic_v1 import _jak2_requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panels", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.panels <= 128:
        parser.error("declared diagnostic supports 1–128 panels")
    teacher = unseal(args.teacher_result)
    source, requests = _jak2_requests(teacher["champion"]["candidate"])
    current, first = v1.compile_pendant_delete(source, requests[0])
    _, desired = v1.compile_substituted_ring(current, requests[1])
    preferred = v1._following_context(current, first)
    rng = np.random.default_rng(np.random.SeedSequence([20260913, 260916, 2]))
    rows, pool_hashes, endpoints = [], set(), set()
    began = perf_counter()
    for index in range(args.panels):
        candidates = v1.enumerate_substituted_rings(
            current,
            rng,
            preferred_anchors=preferred,
            max_candidates=v1.MAX_CONTEXT_CANDIDATES,
            max_trials=v1.MAX_CONTEXT_TRIALS,
        )
        pool = [c.stage["endpoint"] for c in candidates]
        pool_hashes.add(identity(pool))
        endpoints.update(pool)
        ranks = [i + 1 for i, endpoint in enumerate(pool) if endpoint == desired["endpoint"]]
        selected = None
        if candidates:
            _, stage = v1._choose_bound_candidate(
                candidates, rng, family="construct_substituted_ring"
            )
            selected = stage["endpoint"]
        row = {
            "panel": index,
            "candidates": len(pool),
            "pool_sha256": identity(pool),
            "teacher_ranks": ranks,
            "selected_exact": selected == desired["endpoint"],
            "elapsed_seconds": perf_counter() - began,
        }
        rows.append(row)
        seal(args.output.with_suffix(".progress.json"), {"rows": rows, "new_oracle_calls": 0})
        if (index + 1) % 8 == 0:
            print(
                {
                    "panels": index + 1,
                    "distinct_pools": len(pool_hashes),
                    "unique_endpoints": len(endpoints),
                    "panels_with_answer": sum(bool(r["teacher_ranks"]) for r in rows),
                },
                flush=True,
            )
    code = [
        Path(__file__),
        ROOT / "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        ROOT / "src/compose_v4/control/progressive_structured_sampler.py",
        ROOT / "tools/t4_dynamic_v1.py",
    ]
    result = {
        "schema_version": "t4_progressive_conditional_support_v1",
        "rows": rows,
        "unique_panel_count": len(pool_hashes),
        "unique_endpoints": len(endpoints),
        "panels_with_answer": sum(bool(r["teacher_ranks"]) for r in rows),
        "chosen_answer": sum(r["selected_exact"] for r in rows),
        "teacher_endpoint": desired["endpoint"],
        "teacher_source_sha256": sha256_file(args.teacher_result),
        "inputs_sha256": {str(p): sha256_file(p) for p in code},
        "seed_sequence": [20260913, 260916, 2],
        "panels": args.panels,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "new_oracle_calls": 0,
        "completed_at_utc": _stamp(),
        "scope": "Known-answer prefix and family supplied. This is conditional parameter support, not autonomous full-route recovery or utility.",
    }
    seal(args.output, result)


if __name__ == "__main__":
    main()
