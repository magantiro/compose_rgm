"""Saved-path constraint accounting and bounded same-bundle continuation.

Neutral-reference engineering diagnostic only; no checkpoint or docking input.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.continuation import (
    ContinuationBudgetExceeded,
    FiniteHorizonContinuation,
    continuation_decision,
)
from compose_v4.control.option_continuation import OptionContinuationKernel, OptionState
from compose_v4.control.region import enumerate_regions
from compose_v4.control.region_rewrite import Lineage
from compose_v4.control.ring_program import (
    RingProgress,
    completed_construction,
    real_slots,
    ring_spec,
)
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, state_payload
from compose_v4.experiments.ring_construction_probe import ProbeRuntime
from compose_v4.experiments.t4_endpoint_selection import calculate_properties
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_warm_continuation import exact_context
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "max_executor_calls_per_bundle": 1024,
    "max_expansions": 1024,
    "max_terminal_evaluations": 512,
    "safety_seconds_per_bundle": 30,
    "delta": 0.4,
    "qed_min": 0.6,
    "sa_max": 4.0,
    "fingerprint": {"radius": 2, "fpSize": 2048},
    "reference": "neutral uniform broad-organic primitive enumerator, NOT R_theta",
    "terminal": "unchanged T4 endpoint feasibility indicator",
    "kappa": 1.0,
    "exploration": 0.1,
    "seed": None,
    "seed_reason": "exact deterministic backups; no trajectories sampled",
    "workers": 1,
    "oracle_calls": 0,
    "split": "seven inspected development bundles; no holdout claim",
    "winner_inputs": [],
}


def fingerprint_accounting(seed_bits, before_bits, after_bits):
    """Exact set identity; lost shared bits and new union bits are distinct."""
    seed, before, after = map(set, (seed_bits, before_bits, after_bits))
    if not seed or not before or not after:
        raise ValueError("fingerprint accounting requires nonempty molecular fingerprints")
    overlap_before, overlap_after = len(seed & before), len(seed & after)
    union_before, union_after = len(seed | before), len(seed | after)
    return {
        "seed_bits": len(seed),
        "before_bits": len(before),
        "after_bits": len(after),
        "shared_before": overlap_before,
        "shared_after": overlap_after,
        "union_before": union_before,
        "union_after": union_after,
        "lost_shared_bits": sorted((seed & before) - after),
        "gained_shared_bits": sorted((seed & after) - before),
        "new_nonseed_bits": sorted(after - before - seed),
        "removed_nonseed_bits": sorted(before - after - seed),
        "similarity_before": overlap_before / union_before,
        "similarity_after": overlap_after / union_after,
    }


class Properties:
    def __init__(self, seed):
        self.generator = rdFingerprintGenerator.GetMorganGenerator(**CONFIG["fingerprint"])
        self.seed_fp = self.generator.GetFingerprint(Chem.MolFromSmiles(seed))
        self.cache = {}

    def __call__(self, smiles):
        if smiles not in self.cache:
            mol = Chem.MolFromSmiles(smiles)
            result = calculate_properties(
                mol,
                seed_fp=self.seed_fp,
                generator=self.generator,
                sa_scorer=sascorer.calculateScore,
                delta=CONFIG["delta"],
                qed_min=CONFIG["qed_min"],
                sa_max=CONFIG["sa_max"],
            )
            if result is None:
                raise ValueError(f"invalid saved/executed molecule: {smiles}")
            self.cache[smiles] = result
        return self.cache[smiles]

    def bits(self, smiles):
        return self.generator.GetFingerprint(Chem.MolFromSmiles(smiles)).GetOnBits()


def saved_bundle(lock, warm, bundle, properties):
    trace = sorted(
        (
            t
            for w in lock["work"]
            for t in w["sampled_transitions"]
            if t["bundle_id"] == bundle["bundle_id"]
        ),
        key=lambda t: t["step"],
    )
    spec = ring_spec(bundle["option"])
    if spec.refine or len(trace) != spec.horizon:
        raise ValueError("diagnostic requires the saved zero-refinement completed program")
    parent = next(c for c in warm["archive"] if c["smiles"] == bundle["parent"])
    previous = parent["state"]
    path = [{"step": 0, "smiles": parent["smiles"], **properties(parent["smiles"])}]
    for i, step in enumerate(trace, 1):
        if step["step"] != i or step["source"] != previous:
            raise ValueError("saved program trace is discontinuous")
        smiles = canonical_state_key(decode_state(step["product"]))
        if smiles != step["canonical_product"]:
            raise ValueError("saved canonical product differs from exact state")
        path.append(
            {
                "step": i,
                "smiles": smiles,
                **properties(smiles),
                "fingerprint_delta": fingerprint_accounting(
                    properties.seed_fp.GetOnBits(),
                    properties.bits(path[-1]["smiles"]),
                    properties.bits(smiles),
                ),
            }
        )
        previous = step["product"]
    candidate = next(c for c in lock["pool"] if c["bundle_id"] == bundle["bundle_id"])
    for field in ("qed", "sa", "sim", "v"):
        if not math.isclose(path[-1][field], candidate[field], rel_tol=0, abs_tol=1e-7):
            raise ValueError(f"saved {field} differs from review environment")
    graph = decode_state(parent["state"])
    progress = RingProgress.from_payload(trace[-1]["ring_progress"])
    if not completed_construction(graph, decode_state(previous), progress, spec):
        raise ValueError("saved complete ring has no construction witness")
    region = next(
        r for r in enumerate_regions(parent["smiles"]) if sorted(r.atoms) == bundle["region_atoms"]
    )
    context = exact_context(graph, parent["smiles"], region)
    if sorted(context.locus) != bundle["region_slot_atoms"]:
        raise ValueError("region mapping changed; do not silently replay a different bundle")
    node = OptionState(
        graph,
        graph,
        context,
        Lineage.initial(real_slots(graph)),
        spec.option,
        0,
        spec.horizon,
        bundle["bundle_id"],
        ring_progress=RingProgress(),
    )
    return node, path


def probe(node, properties):
    started = perf_counter()
    runtime = ProbeRuntime(
        node.graph, CONFIG["max_executor_calls_per_bundle"], CONFIG["safety_seconds_per_bundle"]
    )
    kernel = OptionContinuationKernel(
        runtime.enumerate,
        runtime.system,
        max_executor_applications=CONFIG["max_executor_calls_per_bundle"],
    )
    terminals = {}

    def terminal(state):
        runtime.check_deadline()
        smiles = canonical_state_key(state.graph)
        prop = properties(smiles)
        terminals.setdefault(smiles, {"smiles": smiles, **prop, "state": state_payload(state)})
        return float(prop["v"] == 0)

    continuation = FiniteHorizonContinuation(
        kernel.row,
        terminal,
        OptionState.key,
        snapshot_id="neutral-t4-ring-feasibility-v1",
        max_expansions=CONFIG["max_expansions"],
        max_terminal_evaluations=CONFIG["max_terminal_evaluations"],
    )
    result = {"status": "incomplete", "reference_feasible_mass": None}
    try:
        with runtime.meter.instrument():
            row = kernel.row(node)
            values = continuation.successor_values(row, node.remaining)
            mass = float(np.dot(row.probabilities, values))
            decision = continuation_decision(
                row, node.remaining, continuation, fallback_values=np.ones(len(row.successors))
            )
            result.update(
                status="complete",
                reference_feasible_mass=mass,
                initial_decision=asdict(decision),
                one_guided_decision_then_reference_mass=float(
                    np.dot(decision.probabilities, values)
                ),
                initial_successors=[
                    {
                        "state": state_payload(s),
                        "reference_probability": p,
                        "feasible_value": float(h),
                        "properties": properties(canonical_state_key(s.graph)),
                    }
                    for s, p, h in zip(row.successors, row.probabilities, values, strict=True)
                ],
            )
    except (ContinuationBudgetExceeded, TimeoutError) as error:
        result.update(status=type(error).__name__, reason=str(error))
    return {
        **result,
        "terminals": [terminals[k] for k in sorted(terminals)],
        "terminal_feasible_unique": sum(t["v"] == 0 for t in terminals.values()),
        "continuation_work": asdict(continuation.work),
        "kernel_work": asdict(kernel.work),
        "executor_calls": runtime.meter.calls,
        "executor_receipts": runtime.meter.attempts,
        "seconds": perf_counter() - started,
    }


def refinement_start(initial, encoded_endpoint, progress):
    """Continue saved atom births in their recorded order, never from SMILES."""
    spec = replace(ring_spec(initial.option), refine=2)
    graph = decode_state(encoded_endpoint)
    if not completed_construction(initial.origin, graph, progress, spec):
        raise ValueError("refinement requires the exact completed construction")
    context, lineage = initial.context, initial.lineage
    for slot in progress.path:
        context = context.with_locus(slot)
        lineage = lineage.observe(
            "atom_insert", SimpleNamespace(slot=slot, atom_type=int(graph.atom_types[slot]))
        )
    return replace(
        initial,
        graph=graph,
        context=context,
        lineage=lineage,
        option=spec.option,
        step=spec.growth + 1,
        horizon=spec.horizon,
        ring_progress=progress,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--refine-saved",
        action="store_true",
        help="reuse six infeasible completed rings for two existing refinement steps",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"preserve previous result: {args.output}")
    warm = unseal(args.run / "warm_start.json")
    lock = unseal(args.run / "round_4/candidate_lock.json")
    if lock["task"]["delta"] != CONFIG["delta"]:
        raise ValueError("saved task differs from the declared diagnostic")
    properties = Properties(warm["archive"][0]["smiles"])
    bundles = [b for b in lock["bundles"] if ring_spec(b["option"]) is not None]
    if len(bundles) != 7:
        raise ValueError("expected the seven saved construction bundles")
    sources = [args.run / "warm_start.json", args.run / "round_4/candidate_lock.json"]
    code = [
        ROOT / p
        for p in subprocess.check_output(
            [
                "git",
                "ls-files",
                "src/compose_v4",
                "tools/t4_ring_feasibility.py",
                "docs/RING_FEASIBILITY_CONTINUATION.md",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
    ]
    report = {
        "schema_version": "t4_ring_feasibility_v1",
        "configuration": {
            **CONFIG,
            "mode": "refine_saved_two_steps" if args.refine_saved else "construction",
        },
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": {str(p): sha256_file(p) for p in sources},
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code},
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {
            "device": "CPU",
            "machine": platform.machine(),
            "precision": "float64 probabilities",
        },
        "bundles": [],
        "excluded_bundles": [],
        "status": "incomplete",
    }
    for bundle in sorted(bundles, key=lambda b: b["bundle_id"]):
        node, path = saved_bundle(lock, warm, bundle, properties)
        if args.refine_saved:
            if path[-1]["v"] == 0:
                report["excluded_bundles"].append(
                    {"bundle_id": node.bundle_id, "reason": "saved_construction_already_feasible"}
                )
                continue
            endpoint = next(
                t
                for w in lock["work"]
                for t in w["sampled_transitions"]
                if t["bundle_id"] == node.bundle_id and t["step"] == node.horizon
            )
            node = refinement_start(
                node, endpoint["product"], RingProgress.from_payload(endpoint["ring_progress"])
            )
        result = probe(node, properties)
        report["bundles"].append(
            {
                "bundle_id": node.bundle_id,
                "option": node.option,
                "initial": state_payload(node),
                "saved_path": path,
                "probe": result,
            }
        )
        publish_json(args.output, report)
        print(
            json.dumps(
                {
                    "bundle": node.bundle_id,
                    "option": node.option,
                    "status": result["status"],
                    "feasible": result["terminal_feasible_unique"],
                    "mass": result["reference_feasible_mass"],
                    "calls": result["executor_calls"],
                    "seconds": result["seconds"],
                }
            ),
            flush=True,
        )
    report["status"] = (
        "complete"
        if all(b["probe"]["status"] == "complete" for b in report["bundles"])
        else "bounded_incomplete"
    )
    publish_json(args.output, report)


if __name__ == "__main__":
    main()
