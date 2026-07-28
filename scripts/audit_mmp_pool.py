#!/usr/bin/env python3
"""Data-starvation baseline audit of the mined MMP edit pool.

Quantifies how much diversity the historical training regime actually saw. The trainer loaded the pool via
``build_analogue_prior_records(pool, count=N)``, which reads the FIRST N LINES -- a contiguous prefix, not a
sample. If the pool is grouped by source molecule, the effective unique-source count of that prefix can be
far below N. This script measures that, censuses the full pool, and compares three fixed 5,000-row sets
(historical prefix / deterministic uniform / deterministic stratified) so the scaled sampler can be designed
from evidence rather than assumption.

This is a BASELINE AUDIT, not a decision gate -- scaling is already established as required.

Streams the file (it is ~0.5 GB) and never loads it whole.

Usage:
  python scripts/audit_mmp_pool.py --pool <edit_pool_full.jsonl> --output diagnostics/data/mmp_pool_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PREFIX_N = 5000  # the historical --analogue-trace-count


def _gini(counts: list[int]) -> float:
    """Concentration of records over sources/scaffolds. 0 = perfectly even, ->1 = all mass on one."""
    if not counts:
        return float("nan")
    xs = sorted(counts)
    n = len(xs)
    total = sum(xs)
    if total == 0:
        return float("nan")
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def _murcko(smiles: str) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:  # noqa: BLE001
        return None


def _step_rule(step: dict) -> str:
    """Pool records key the operator as 'rule'; be tolerant of 'rule_name' too."""
    return step.get("rule") or step.get("rule_name") or "?"


def _row_features(record: dict) -> dict:
    """Cheap per-row features -- no executor, no canonicalization. Prefers the record's own
    ``operator_histogram`` / ``path_length`` (authoritative) over recomputing from steps."""
    steps = record.get("steps", []) or []
    fams = Counter(record.get("operator_histogram") or {})
    if not fams:
        fams = Counter(_step_rule(s) for s in steps)
    # net atom-count change is exactly (#atom_insert - #atom_delete) for these traces
    d_atoms = fams.get("atom_insert", 0) - fams.get("atom_delete", 0)
    # net cycle-rank change: ring-closing events raise it, ring-opening events lower it
    d_cycle = (
        fams.get("bond_insert", 0) + fams.get("ring_system_grow", 0)
        - fams.get("bond_delete", 0) - fams.get("ring_system_delete", 0)
    )
    diag = record.get("diagnostics") or {}
    return {
        "source": record.get("source_smiles"),
        "target": record.get("target_key") or record.get("target_smiles"),
        "n_steps": int(record.get("path_length") or len(steps)),
        "families": fams,
        "d_atoms": d_atoms,
        "d_cycle": d_cycle,
        "layer": record.get("layer") or (record.get("metadata") or {}).get("layer"),
        "direction": record.get("direction"),
        "pair_type": diag.get("pair_type"),
        # transformation signature: the operator families in path order
        "signature": "|".join(_step_rule(s) for s in steps),
    }


def census(rows: list[dict], scaffolds: dict[str, str | None], label: str) -> dict:
    sources = [r["source"] for r in rows if r["source"]]
    targets = [r["target"] for r in rows if r["target"]]
    transitions = [(r["source"], r["target"]) for r in rows]
    src_counts = Counter(sources)
    scaf = [scaffolds.get(s) for s in sources]
    scaf_counts = Counter(s for s in scaf if s)
    n = len(rows)

    # longest contiguous run from a single source -> reveals grouping
    longest_run, run, prev = 0, 0, object()
    for s in sources:
        run = run + 1 if s == prev else 1
        longest_run = max(longest_run, run)
        prev = s
    # is the file grouped? (a source's rows all contiguous) / sorted?
    seen_blocks: dict[str, int] = defaultdict(int)
    prev = object()
    for s in sources:
        if s != prev:
            seen_blocks[s] += 1
        prev = s
    grouped = all(v == 1 for v in seen_blocks.values()) if seen_blocks else False
    sorted_by_source = sources == sorted(sources)

    top_shares = {}
    ordered = [c for _, c in src_counts.most_common()]
    if ordered and n:
        top1pct = max(1, len(ordered) // 100)
        top_shares = {
            "top_1pct_sources_share": round(sum(ordered[:top1pct]) / n, 4),
            "top_10_sources_share": round(sum(ordered[:10]) / n, 4),
            "top_100_sources_share": round(sum(ordered[:100]) / n, 4),
        }

    fam_total: Counter = Counter()
    for r in rows:
        fam_total.update(r["families"])

    return {
        "label": label,
        "rows": n,
        "unique_source_molecules": len(src_counts),
        "unique_target_molecules": len(set(targets)),
        "unique_canonical_transitions": len(set(transitions)),
        "unique_murcko_scaffolds": len(scaf_counts),
        "unique_transformation_signatures": len({r["signature"] for r in rows}),
        "duplicate_transition_rate": round(1.0 - len(set(transitions)) / n, 4) if n else None,
        "records_per_source_mean": round(n / max(len(src_counts), 1), 2),
        "source_frequency_gini": round(_gini(list(src_counts.values())), 4),
        "scaffold_frequency_gini": round(_gini(list(scaf_counts.values())), 4),
        **top_shares,
        "max_contiguous_run_one_source": longest_run,
        "file_grouped_by_source": grouped,
        "file_sorted_by_source": sorted_by_source,
        "path_length_distribution": dict(sorted(Counter(r["n_steps"] for r in rows).items())),
        "operator_family_counts": dict(fam_total.most_common()),
        "atom_count_change_distribution": dict(sorted(Counter(r["d_atoms"] for r in rows).items())),
        "cycle_rank_change_distribution": dict(sorted(Counter(r["d_cycle"] for r in rows).items())),
        "layer_counts": dict(Counter(r["layer"] for r in rows)),
        "direction_counts": dict(Counter(r["direction"] for r in rows)),
        "pair_type_counts": dict(Counter(r["pair_type"] for r in rows)),
    }


def select_uniform(all_rows: list[dict], k: int) -> list[int]:
    """Deterministic uniform selection over the WHOLE pool (even stride)."""
    n = len(all_rows)
    if n <= k:
        return list(range(n))
    step = n / k
    return [int(i * step) for i in range(k)]


def select_stratified(all_rows: list[dict], k: int) -> list[int]:
    """Deterministic stratified selection: round-robin over sources (caps per-source dominance),
    breaking ties by a stable hash so the choice is reproducible and not order-dependent."""
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(all_rows):
        by_source[r["source"]].append(i)
    order = sorted(
        by_source.items(), key=lambda kv: hashlib.sha256(str(kv[0]).encode()).hexdigest()
    )
    picked: list[int] = []
    depth = 0
    while len(picked) < k:
        progressed = False
        for _src, idxs in order:
            if depth < len(idxs):
                picked.append(idxs[depth])
                progressed = True
                if len(picked) >= k:
                    break
        if not progressed:
            break
        depth += 1
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--prefix-n", type=int, default=PREFIX_N)
    parser.add_argument("--scaffold-cap", type=int, default=40000,
                        help="cap unique sources scaffolded (Murcko is ~1ms each)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    with args.pool.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(_row_features(json.loads(line)))
    print(json.dumps({"phase": "pool_loaded", "rows": len(rows)}), flush=True)

    unique_sources = list({r["source"] for r in rows if r["source"]})
    scaffolds: dict[str, str | None] = {}
    for s in unique_sources[: args.scaffold_cap]:
        scaffolds[s] = _murcko(s)
    print(json.dumps({"phase": "scaffolds_computed",
                      "unique_sources": len(unique_sources),
                      "scaffolded": len(scaffolds)}), flush=True)

    prefix_idx = list(range(min(args.prefix_n, len(rows))))
    uniform_idx = select_uniform(rows, args.prefix_n)
    strat_idx = select_stratified(rows, args.prefix_n)

    report = {
        "pool_path": str(args.pool),
        "pool_sha256_prefix": hashlib.sha256(
            args.pool.read_bytes()[:1 << 20]
        ).hexdigest()[:16],
        "selection_implementation_hash": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()[:16],
        "prefix_n": args.prefix_n,
        "full_pool": census(rows, scaffolds, "full_pool"),
        "historical_prefix": census([rows[i] for i in prefix_idx], scaffolds, "historical_prefix_first_N"),
        "deterministic_uniform": census([rows[i] for i in uniform_idx], scaffolds, "deterministic_uniform"),
        "deterministic_stratified": census([rows[i] for i in strat_idx], scaffolds, "deterministic_stratified"),
        "selected_ids": {
            "historical_prefix": prefix_idx[:100],
            "deterministic_uniform": uniform_idx[:100],
            "deterministic_stratified": strat_idx[:100],
            "note": "first 100 of each shown; full index lists are reproducible from this script + pool",
        },
    }
    # headline comparison
    report["headline"] = {
        "historical_prefix_unique_sources": report["historical_prefix"]["unique_source_molecules"],
        "stratified_unique_sources": report["deterministic_stratified"]["unique_source_molecules"],
        "uniform_unique_sources": report["deterministic_uniform"]["unique_source_molecules"],
        "historical_prefix_unique_scaffolds": report["historical_prefix"]["unique_murcko_scaffolds"],
        "stratified_unique_scaffolds": report["deterministic_stratified"]["unique_murcko_scaffolds"],
        "full_pool_unique_sources": report["full_pool"]["unique_source_molecules"],
        "full_pool_unique_scaffolds": report["full_pool"]["unique_murcko_scaffolds"],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(json.dumps(report["headline"], indent=2, sort_keys=True))
    print(json.dumps({"full_pool": report["full_pool"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
