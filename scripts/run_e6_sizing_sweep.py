"""A2.1: run the preregistered exact-graph sizing sweep and record EVERY candidate.

The point of reporting every candidate -- qualifying or not, with its rejection reason -- is that the
selection then has no hidden search behind it. The rule (smallest qualifying candidate) and the window
(``exact_sizing`` in the frozen registry) are fixed before the numbers exist, and nothing about a
controller's performance enters.

    PYTHONPATH=src python3 scripts/run_e6_sizing_sweep.py \
        [--output diagnostics/exactness/e6_sizing_sweep.json] [--deadline-seconds 600]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from compose_v4.experiments.enumerable_ringcore import (
    SizingInfeasible,
    default_candidate_ladder,
    evaluate_candidate,
    select_candidate,
)
from compose_v4.experiments.registry import load_registry

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(_ROOT / "configs" / "experiment_registry.yaml"))
    parser.add_argument(
        "--output", default=str(_ROOT / "diagnostics" / "exactness" / "e6_sizing_sweep.json")
    )
    parser.add_argument("--deadline-seconds", type=float, default=600.0)
    arguments = parser.parse_args()

    registry = load_registry(arguments.registry)
    sizing = registry["exact_sizing"]
    n_min, n_max, e_max = int(sizing["N_min"]), int(sizing["N_max"]), int(sizing["E_max"])
    print(f"frozen window: {n_min} <= n_states <= {n_max}, n_edges <= {e_max}", flush=True)
    print(f"selection rule: {sizing['selection']}", flush=True)

    reports = []
    for candidate in default_candidate_ladder():
        print(f"measuring {candidate.candidate_id} ...", flush=True)
        report = evaluate_candidate(
            candidate,
            n_min=n_min,
            n_max=n_max,
            e_max=e_max,
            deadline_seconds=arguments.deadline_seconds,
        )
        statistics = report["statistics"]
        print(
            f"  states={statistics['n_states']} edges={statistics['n_edges']} "
            f"depth={statistics['depth']} stop={statistics['stop_reason']} "
            f"{statistics['seconds']}s qualifies={report['qualifies']}",
            flush=True,
        )
        if report["rejection_reasons"]:
            print(f"  rejected: {report['rejection_reasons']}", flush=True)
        reports.append(report)

    selection: dict | None = None
    infeasible_reason = None
    try:
        selection = select_candidate(reports)
        print(f"\nSELECTED: {selection['candidate_id']} "
              f"({selection['statistics']['n_states']} states, "
              f"{selection['statistics']['n_edges']} edges)", flush=True)
    except SizingInfeasible as error:
        infeasible_reason = str(error)
        print(f"\nINFEASIBLE: {infeasible_reason}", flush=True)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    payload = {
        "artifact": "e6_sizing_sweep",
        "commit": commit,
        "registry_protocol_content_hash": registry["protocol"]["protocol_freeze"]["content_hash"],
        "window": {"N_min": n_min, "N_max": n_max, "E_max": e_max},
        "selection_rule": sizing["selection"],
        "non_degeneracy_required": list(sizing["non_degeneracy_required"]),
        "operator_set": "ringcore_v1_compositional_cycle_ops (executor bond_insert/bond_delete)",
        "measured_size_mechanism": (
            "The reachable set is bounded by the seed's ATOM-SLOT COUNT: atom_insert fills a NULL slot and "
            "cannot extend the state array. Measured on cyclopropane, atom_insert fired 18 times without "
            "ever producing a 4-atom molecule. Slots are therefore the size dial."
        ),
        "selection_independent_of_controller_performance": True,
        "candidates": reports,
        "selected": selection,
        "infeasible_reason": infeasible_reason,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}", flush=True)
    return 0 if selection is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
