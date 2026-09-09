"""Recheck saved composite rewrites after the region-footprint repair; no search."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.option_continuation import OptionContinuationKernel, OptionState
from compose_v4.control.region import enumerate_regions
from compose_v4.control.region_rewrite import (
    Lineage,
    admissible_indices,
    context_preserved,
    touched_slots,
)
from compose_v4.experiments.t4_warm_continuation import canonical_slots
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from compose_v4.rewrite.tracelets import RingSystemRestate
from tools.ivg_winner_paths import digest, mapped_context, publish, sha

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = "3b7fb79a541b94dd5421f30ffc0516c5cdc4d9316845cd40561d799ec517f922"
REPAIRED = "src/compose_v4/control/region_rewrite.py"


def run(directory: Path, output: Path):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("use a clean committed worktree and an external output directory")
    baseline = directory / "audit.json"
    if sha(baseline) != BASELINE_SHA:
        raise ValueError("winner audit input hash mismatch")
    audit = json.loads(baseline.read_text())
    unchanged = {}
    for name, expected in audit["implementation_sha256"].items():
        if name != REPAIRED:
            if sha(ROOT / name) != expected:
                raise ValueError(f"unexpected changed dependency: {name}")
            unchanged[name] = expected
    if (
        rdBase.rdkitVersion != audit["software"]["rdkit"]
        or np.__version__ != audit["software"]["numpy"]
    ):
        raise ValueError("saved-state replay requires the audited local RDKit/NumPy runtime")
    started, rows, inputs = time.monotonic(), [], {str(baseline): BASELINE_SHA}
    for pair in audit["pairs"]:
        if pair["status"] != "witness_found":
            continue
        file = directory / pair["receipt"]
        if sha(file) != pair["receipt_sha256"]:
            raise ValueError(f"receipt hash mismatch: {file}")
        receipt = json.loads(gzip.decompress(file.read_bytes()))
        payload = receipt["payload"]
        if digest(payload) != receipt["payload_sha256"]:
            raise ValueError(f"receipt payload mismatch: {file}")
        path, annotations = payload["path"], payload["annotations"]
        for step, mark in enumerate(path["actions"]):
            family, action = decode_action(mark)
            if not isinstance(action, RingSystemRestate):
                continue
            inputs[str(file)] = pair["receipt_sha256"]
            graph, product = (decode_state(path["states"][i]) for i in (step, step + 1))
            smiles = annotations[step]["smiles"]
            mapping = canonical_slots(graph, smiles)
            regions = [r for r in enumerate_regions(smiles) if 1 <= r.size <= 24]
            passing, preserved = [], []
            for region in sorted(regions, key=lambda r: (r.size, repr(r.key()))):
                context = mapped_context(region, mapping)
                if admissible_indices((family,), (action,), context)[0]:
                    passing.append(region.size / graph.n_real_atoms)
                    if context_preserved(
                        graph, product, context.frozen, context.terminal_context_slots
                    ):
                        preserved.append((region, context))
            option_checks = {}
            kernel_calls = 0
            if preserved:
                region, context = preserved[0]
                kernel = OptionContinuationKernel(
                    lambda _, f=family, a=action: ((f,), (a,), (1.0,)),
                    editing_v2_semantic_rewrite_system(),
                    max_executor_applications=1,
                )
                for option in ("generic", "restate", "aromatize"):
                    node = OptionState(
                        graph,
                        graph,
                        context,
                        Lineage.initial(np.flatnonzero(is_element(graph.atom_types))),
                        option,
                        0,
                        1,
                        f"footprint:{pair['pair_id']}:{step}",
                    )
                    successors = kernel.row(node).successors
                    option_checks[option] = {
                        "successors": len(successors),
                        "exact_saved_product": bool(successors)
                        and all(
                            encode_state(s.graph) == path["states"][step + 1] for s in successors
                        ),
                    }
                kernel_calls = kernel.work.executor_applications
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "step": step,
                    "cells": sorted({r["cell"] for r in pair["references"]}),
                    "source_state_sha256": digest(path["states"][step]),
                    "expected_slots": sorted({v for c in action.changes for v in (c.a, c.b)}),
                    "actual_slots": sorted(touched_slots(action)),
                    "before_passing_regions": annotations[step]["next_edit"][
                        "touch_filter_regions"
                    ],
                    "after_passing_regions": len(passing),
                    "context_preserving_regions": len(preserved),
                    "scope_range": [min(passing), max(passing)] if passing else None,
                    "singleton_option_checks": option_checks,
                    "kernel_executor_applications": kernel_calls,
                }
            )
    result = {
        "schema_version": "t4_restate_footprint_repair_v1",
        "rows": rows,
        "input_sha256": dict(sorted(inputs.items())),
        "unchanged_dependencies": unchanged,
        "before_region_code_sha256": audit["implementation_sha256"][REPAIRED],
        "after_region_code_sha256": sha(ROOT / REPAIRED),
        "producer_sha256": sha(Path(__file__)),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {"machine": platform.machine(), "accelerator": "none", "workers": 1},
        "configuration": {
            "seed": None,
            "deterministic": True,
            "dtype": "exact saved integer graph arrays",
            "selection": "all RingSystemRestate actions in every saved successful witness",
            "reference": "diagnostic singleton mass 1, NOT learned R_theta",
            "max_region_atoms": 24,
            "new_searches": 0,
            "new_oracle_calls": 0,
            "learned_law_enumerations": 0,
        },
        "evidence_role": "development regression, no blind-discovery or pinned-Modal-runtime claim",
        "seconds": time.monotonic() - started,
    }
    publish(output, result)
    print(
        json.dumps(
            {
                "actions": len(rows),
                "admitted": sum(r["after_passing_regions"] > 0 for r in rows),
                "seconds": result["seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.directory, args.output)
