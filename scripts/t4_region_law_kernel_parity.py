"""Does the region-law evidence hold under the PRODUCTION chemistry kernel?

WHY THIS EXISTS
---------------
Every number the region-draw law produces is keyed on canonical SMILES: the law
enumerates bridge-separated regions, excises each, canonicalizes the child, and
ranks it by the free gate.  Both the ranking and the endpoint identity therefore
depend on which RDKit is loaded.

The production image pins **rdkit 2024.03.5**.  A developer laptop here runs
2026.03.6, and the two are known to disagree: on a Kekule-degenerate
hypervalent-sulfur ring they write DIFFERENT canonical SMILES for the same
molecule, and 2024.03.5 can raise ``Invariant Violation ... could not find
atom1`` from ``Chem.MolToSmiles`` on states the newer kernel accepts.  A gate
measured on the newer kernel is a statement about a chemistry the campaigns do
not run.

WHAT IT DOES
------------
Runs one kernel, writes one artifact: for every cell, every region the law
offers, the canonical SMILES of the child it produces, that child's similarity /
QED / SA, the production ``Fiber.check`` verdict, and the law's weight rank.
States the kernel REFUSES are recorded as refusals rather than crashing the run,
because "this kernel cannot express that state" is itself the finding.

``--compare`` then diffs two such artifacts and reports ``max_abs_delta`` per
scalar, canonical-SMILES disagreements, verdict flips and refusal counts.

ZERO ORACLE CALLS. Similarity, QED, SA and the structural gate are all free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = "t4_region_law_kernel_parity_v1"

#: The T4 proposal executor requires 48 slots.
PROPOSAL_SLOTS = 48

CELLS = {
    "configs/t4_held_target_distilled_braf_d06_250.json": ("braf_0", "braf_1", "braf_2"),
    "configs/t4_held_target_distilled_fa7_d06_250.json": ("fa7_0", "fa7_1", "fa7_2"),
}


def _kernel_identity() -> dict:
    import networkx
    import numpy
    import rdkit
    import scipy

    return {
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
    }


def _measure() -> dict:
    from compose_v4.chem.molecular_graph import (
        molecular_graph_to_smiles,
        smiles_to_molecular_graph,
    )
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.bridge_region_law import (
        FreeFeasibilityGate,
        RegionRealizationError,
        excise_region,
        free_gate_margin_law,
    )
    from compose_v4.control.dynamic_program_synthesis import MAX_SEGMENT_LENGTH
    from compose_v4.experiments.t4_fiber_campaign import Fiber

    rows = []
    for contract, wanted in CELLS.items():
        payload = json.loads(Path(contract).read_text())["payload"]
        delta = payload["delta"]
        for cell in payload["cells"]:
            if cell["cell"] not in wanted:
                continue
            source = cell["smiles"]
            graph = pad_molecular_graph(
                smiles_to_molecular_graph(source), PROPOSAL_SLOTS
            )
            fiber = Fiber(source, delta, support=payload["support"])
            law = free_gate_margin_law(
                FreeFeasibilityGate(delta=delta), source, maximum=None
            )
            regions = law.regions(graph)
            weights = law.weights(graph, regions)
            order = sorted(range(len(regions)), key=lambda at: -float(weights[at]))
            children = []
            refusals = []
            for rank, at in enumerate(order, start=1):
                region = regions[at]
                key = f"{'.'.join(str(s) for s in region.fragment)}@{region.anchor}"
                try:
                    child = excise_region(graph, region)
                except RegionRealizationError as error:
                    refusals.append(
                        {"region": key, "stage": "excise", "error": str(error)[:120]}
                    )
                    continue
                # The production kernel can REFUSE to canonicalize a state the
                # newer one writes happily; record that rather than crash.
                try:
                    smiles = molecular_graph_to_smiles(child)
                except Exception as error:  # noqa: BLE001 - the refusal IS the finding
                    refusals.append(
                        {
                            "region": key,
                            "stage": "canonicalize",
                            "error": f"{type(error).__name__}: {str(error)[:160]}",
                        }
                    )
                    continue
                if smiles is None:
                    refusals.append({"region": key, "stage": "canonicalize", "error": "None"})
                    continue
                try:
                    gate = fiber.check(smiles)
                except Exception as error:  # noqa: BLE001
                    refusals.append(
                        {
                            "region": key,
                            "stage": "gate",
                            "error": f"{type(error).__name__}: {str(error)[:160]}",
                        }
                    )
                    continue
                children.append(
                    {
                        "region": key,
                        "size": region.size,
                        "rank": rank,
                        "weight": round(float(weights[at]), 9),
                        "smiles": smiles,
                        "eligible": gate is not None,
                        "similarity": None if gate is None else round(gate["similarity"], 9),
                        "qed": None if gate is None else round(gate["qed"], 9),
                        "sa": None if gate is None else round(gate["sa"], 9),
                    }
                )
            rows.append(
                {
                    "cell": cell["cell"],
                    "delta": delta,
                    "source": source,
                    "source_heavy": int(graph.n_real_atoms),
                    "regions": len(regions),
                    "regions_within_v1_cap": sum(
                        1 for region in regions if region.size <= MAX_SEGMENT_LENGTH
                    ),
                    "eligible": sum(1 for row in children if row["eligible"]),
                    "best_eligible_rank": min(
                        (row["rank"] for row in children if row["eligible"]), default=None
                    ),
                    "refusals": refusals,
                    "children": children,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "kernel": _kernel_identity(),
        "oracle_calls": 0,
        "cells": rows,
    }


def _compare(left: dict, right: dict) -> dict:
    """Diff two kernels' artifacts. Any nonzero delta here invalidates a rank."""

    findings = []
    worst = {"similarity": 0.0, "qed": 0.0, "sa": 0.0}
    smiles_mismatch = verdict_flip = rank_move = 0
    for a, b in zip(left["cells"], right["cells"]):
        assert a["cell"] == b["cell"], "cell order must match"
        by_region_a = {row["region"]: row for row in a["children"]}
        by_region_b = {row["region"]: row for row in b["children"]}
        only_a = sorted(set(by_region_a) - set(by_region_b))
        only_b = sorted(set(by_region_b) - set(by_region_a))
        for region in sorted(set(by_region_a) & set(by_region_b)):
            rowa, rowb = by_region_a[region], by_region_b[region]
            if rowa["smiles"] != rowb["smiles"]:
                smiles_mismatch += 1
                findings.append(
                    {
                        "cell": a["cell"],
                        "region": region,
                        "kind": "canonical_smiles_disagreement",
                        "left": rowa["smiles"],
                        "right": rowb["smiles"],
                    }
                )
            if rowa["eligible"] != rowb["eligible"]:
                verdict_flip += 1
                findings.append(
                    {
                        "cell": a["cell"],
                        "region": region,
                        "kind": "verdict_flip",
                        "left": rowa["eligible"],
                        "right": rowb["eligible"],
                    }
                )
            if rowa["rank"] != rowb["rank"]:
                rank_move += 1
            for field, seen in worst.items():
                if rowa[field] is not None and rowb[field] is not None:
                    worst[field] = max(seen, abs(rowa[field] - rowb[field]))
        if only_a or only_b:
            findings.append(
                {
                    "cell": a["cell"],
                    "kind": "region_present_in_one_kernel_only",
                    "left_only": only_a,
                    "right_only": only_b,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION + "_compare",
        "left_kernel": left["kernel"],
        "right_kernel": right["kernel"],
        "max_abs_delta": worst,
        "canonical_smiles_disagreements": smiles_mismatch,
        "verdict_flips": verdict_flip,
        "rank_moves": rank_move,
        "refusals_left": sum(len(row["refusals"]) for row in left["cells"]),
        "refusals_right": sum(len(row["refusals"]) for row in right["cells"]),
        "best_eligible_rank_left": {
            row["cell"]: row["best_eligible_rank"] for row in left["cells"]
        },
        "best_eligible_rank_right": {
            row["cell"]: row["best_eligible_rank"] for row in right["cells"]
        },
        "eligible_left": {row["cell"]: row["eligible"] for row in left["cells"]},
        "eligible_right": {row["cell"]: row["eligible"] for row in right["cells"]},
        "parity": (
            smiles_mismatch == 0
            and verdict_flip == 0
            and rank_move == 0
            and max(worst.values()) == 0.0
        ),
        "findings": findings[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--compare-out", type=Path)
    args = parser.parse_args()
    if args.compare:
        left = json.loads(args.compare[0].read_text())
        right = json.loads(args.compare[1].read_text())
        report = _compare(left, right)
        print(json.dumps({k: v for k, v in report.items() if k != "findings"}, indent=1))
        if report["findings"]:
            print("\nFINDINGS:")
            print(json.dumps(report["findings"][:10], indent=1))
        if args.compare_out:
            args.compare_out.parent.mkdir(parents=True, exist_ok=True)
            args.compare_out.write_text(json.dumps(report, indent=1, sort_keys=True))
            print(f"\nwrote {args.compare_out}")
        return
    report = _measure()
    for row in report["cells"]:
        print(
            f"{row['cell']:>8}  regions={row['regions']:<4} eligible={row['eligible']:<3} "
            f"best_rank={row['best_eligible_rank']}  refusals={len(row['refusals'])}",
            flush=True,
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"wrote {args.out}  kernel={report['kernel']['rdkit']}")


if __name__ == "__main__":
    main()
