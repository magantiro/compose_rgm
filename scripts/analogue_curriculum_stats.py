#!/usr/bin/env python3
"""Curriculum-statistics report over the compiled one-cut MMP trace pool (deliverable of PAPER_MASTER_
PLAN.md sec 0b: curriculum bins are set FROM measured statistics, never preset easy/medium/hard).

Reads the two artifacts written by build_analogue_trace_pool.py -- the verified trace pool
(analogue_trace_pool.jsonl) and the per-pair diagnostics (analogue_pair_diagnostics.jsonl, which
includes failures so success-by-type is honest) -- and measures the curriculum axes: compiled
path-length distribution, ring complexity of the variable region, number of attachment sites,
operator-family usage, and compiler success rate by pair type. It then derives edit-budget curriculum
bins from the EMPIRICAL path-length quantiles (terciles) so the bins follow the data. Pure
post-processing (no RDKit/executor); writes analogue_curriculum_stats.json + a printed report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "diagnostics" / "composition"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hist(values) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _quantile_bins(lengths: list[int]) -> dict:
    """Empirical-tercile edit-budget bins over compiled path length (data-driven, not preset)."""
    if not lengths:
        return {"axis": "compiled_path_length", "method": "empirical_terciles",
                "edges": [], "bins": []}
    low, high = (int(round(edge)) for edge in np.percentile(lengths, [100 / 3, 200 / 3]))
    high = max(high, low)  # keep edges monotone under ties
    bins = [
        {"name": "local", "predicate": f"length <= {low}",
         "count": sum(1 for n in lengths if n <= low)},
        {"name": "lead_opt", "predicate": f"{low} < length <= {high}",
         "count": sum(1 for n in lengths if low < n <= high)},
        {"name": "scaffold", "predicate": f"length > {high}",
         "count": sum(1 for n in lengths if n > high)},
    ]
    return {"axis": "compiled_path_length", "method": "empirical_terciles (data-driven)",
            "edges": [low, high], "bins": bins}


def _success_by_pair_type(diagnostics: list[dict]) -> dict:
    grouped: dict[str, Counter] = {}
    for record in diagnostics:
        pair_type = record.get("pair_type") or "unknown"
        bucket = grouped.setdefault(pair_type, Counter())
        bucket["mined"] += 1
        bucket["forward_ok"] += int(record.get("forward_outcome") == "ok")
        bucket["reverse_ok"] += int(record.get("reverse_outcome") == "ok")
        bucket["both_ok"] += int(record.get("both_directions_ok", False))
    return {
        pair_type: {
            "mined": bucket["mined"],
            "forward_ok": bucket["forward_ok"],
            "forward_success_rate": bucket["forward_ok"] / bucket["mined"],
            "both_ok": bucket["both_ok"],
            "both_success_rate": bucket["both_ok"] / bucket["mined"],
        }
        for pair_type, bucket in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=OUT_DIR / "analogue_trace_pool.jsonl")
    parser.add_argument("--diagnostics", type=Path,
                        default=OUT_DIR / "analogue_pair_diagnostics.jsonl")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "analogue_curriculum_stats.json")
    args = parser.parse_args()

    pool = _read_jsonl(args.pool)
    diagnostics = _read_jsonl(args.diagnostics)
    if not pool:
        raise SystemExit(f"no trace pool at {args.pool}; run build_analogue_trace_pool.py first")

    lengths = [record["path_length"] for record in pool]
    variable_ring_atoms = [record["diagnostics"]["variable_ring_atoms"] for record in pool]
    attachment_sites = [record["diagnostics"]["attachment_count"] for record in pool]
    source_variable = [record["diagnostics"]["source_variable_heavy"] for record in pool]
    target_variable = [record["diagnostics"]["target_variable_heavy"] for record in pool]
    core_heavy = [record["diagnostics"]["constant_core_heavy"] for record in pool]

    operator_totals: Counter = Counter()
    per_trace_family = []
    for record in pool:
        histogram = record["operator_histogram"]
        operator_totals.update(histogram)
        per_trace_family.append(sum(histogram.values()))

    forward_ok = sum(1 for d in diagnostics if d.get("forward_outcome") == "ok")
    reverse_ok = sum(1 for d in diagnostics if d.get("reverse_outcome") == "ok")
    both_ok = sum(1 for d in diagnostics if d.get("both_directions_ok"))
    failure_categories = Counter(
        d.get("forward_outcome") for d in diagnostics if d.get("forward_outcome") != "ok")

    report = {
        "experiment": "analogue_curriculum_stats",
        "pool_path": str(args.pool),
        "diagnostics_path": str(args.diagnostics),
        "n_verified_traces": len(pool),
        "n_mined_pairs": len(diagnostics),
        "path_length": {
            "min": int(min(lengths)), "max": int(max(lengths)),
            "mean": float(np.mean(lengths)), "median": float(np.median(lengths)),
            "p25": float(np.percentile(lengths, 25)), "p75": float(np.percentile(lengths, 75)),
            "hist": _hist(lengths),
        },
        "variable_region_ring_complexity": {
            "ring_atom_hist": _hist(variable_ring_atoms),
            "mean_ring_atoms": float(np.mean(variable_ring_atoms)),
            "max_ring_atoms": int(max(variable_ring_atoms)),
            "ring_bearing_fraction": sum(1 for r in variable_ring_atoms if r > 0) / len(pool),
            "note": "one-cut A2.1 variable fragments are acyclic by construction (this is the "
                    "acyclic-variable curriculum bin); ring-bearing variable regions live in the "
                    "Murcko A2.3 candidate graph.",
        },
        "attachment_sites": {
            "hist": _hist(attachment_sites),
            "note": "one attachment per one-cut MMP (single-attachment bin).",
        },
        "variable_fragment_heavy": {
            "source_hist": _hist(source_variable),
            "target_hist": _hist(target_variable),
            "combined_mean": float(np.mean(source_variable + target_variable)),
            "combined_max": int(max(source_variable + target_variable)),
        },
        "constant_core_heavy": {
            "mean": float(np.mean(core_heavy)), "min": int(min(core_heavy)),
            "max": int(max(core_heavy)), "hist": _hist(core_heavy),
        },
        "operator_family_usage": {
            "totals": dict(operator_totals.most_common()),
            "families": len(operator_totals),
            "ops_per_trace_mean": float(np.mean(per_trace_family)),
            "delete_to_insert_ratio": operator_totals.get("atom_delete", 0)
            / max(operator_totals.get("atom_insert", 0), 1),
        },
        "compiler_success": {
            "forward_rate": forward_ok / max(len(diagnostics), 1),
            "reverse_rate": reverse_ok / max(len(diagnostics), 1),
            "both_direction_rate": both_ok / max(len(diagnostics), 1),
            "by_pair_type": _success_by_pair_type(diagnostics),
            "failure_categories": dict(failure_categories.most_common()),
        },
        "suggested_curriculum_bins": _quantile_bins(lengths),
        "secondary_bins_variable_heavy": _quantile_bins(
            [record["diagnostics"]["source_variable_heavy"]
             + record["diagnostics"]["target_variable_heavy"] for record in pool]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"=== curriculum statistics over {len(pool)} verified traces "
          f"({len(diagnostics)} mined pairs) ===", flush=True)
    pl = report["path_length"]
    print(f"path length: min={pl['min']} median={pl['median']:.0f} mean={pl['mean']:.1f} "
          f"max={pl['max']} | hist {pl['hist']}", flush=True)
    print(f"variable-region ring atoms: hist "
          f"{report['variable_region_ring_complexity']['ring_atom_hist']} "
          f"(ring-bearing {100 * report['variable_region_ring_complexity']['ring_bearing_fraction']:.1f}%)",
          flush=True)
    print(f"attachment sites: hist {report['attachment_sites']['hist']}", flush=True)
    print(f"operator usage: {report['operator_family_usage']['totals']} "
          f"(del:ins {report['operator_family_usage']['delete_to_insert_ratio']:.2f}, "
          f"{report['operator_family_usage']['ops_per_trace_mean']:.1f} ops/trace)", flush=True)
    print("compiler success by pair type:", flush=True)
    for pair_type, stats in report["compiler_success"]["by_pair_type"].items():
        print(f"   {pair_type:10s} mined={stats['mined']:3d} forward="
              f"{100 * stats['forward_success_rate']:5.1f}% both="
              f"{100 * stats['both_success_rate']:5.1f}%", flush=True)
    if failure_categories:
        print(f"failure categories: {dict(failure_categories.most_common())}", flush=True)
    bins = report["suggested_curriculum_bins"]
    print(f"data-driven edit-budget bins (tercile edges {bins['edges']} on path length):", flush=True)
    for spec in bins["bins"]:
        print(f"   {spec['name']:9s} [{spec['predicate']}] -> {spec['count']} traces", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
