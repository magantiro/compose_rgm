#!/usr/bin/env python3
"""Benchmark-lead acceptance under the three corpus scopes.

The primary application edits real lead-optimization benchmark molecules. This reports how many of them are
even REPRESENTABLE under (1) the production broad-organic charge-preserving scope, (2) a broad-organic
neutral-only ablation, and (3) the old CNOF-neutral filter -- and, for every excluded lead, the exact
unsupported feature. Silently dropping S/P/Cl/Br/I- or charge-bearing leads would bias the benchmark, so
this coverage census is a prerequisite for the headline experiments.

Run: PYTHONPATH=src python scripts/benchmark_lead_scope_coverage.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import (
    IDX_TO_ELEMENT,
    MolecularGraphError,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_connected_or_null
from compose_v4.data.organic_corpus import (
    BROAD_ORGANIC_NEUTRAL_V1,
    BROAD_ORGANIC_V1,
    classify_smiles,
)

REPO = Path(__file__).resolve().parent.parent
_JIN = "configs/benchmarks/jin_iclr19_qed_test_exact_v1.csv"
_SMILES_COL = "canonical_nonisomeric_smiles"  # our representation drops stereo/isotopes


def _cnof_neutral_ok(smi: str, *, max_atoms: int = 48) -> bool:
    """The retired CNOF-neutral filter (C/N/O/F elements, neutral, connected, size-bounded)."""
    if "." in smi:
        return False
    try:
        graph = smiles_to_molecular_graph(smi)
    except (MolecularGraphError, ValueError):
        return False
    if graph.n_real_atoms == 0 or graph.n_real_atoms > max_atoms or not is_connected_or_null(graph):
        return False
    real = is_element(graph.atom_types)
    if np.any(graph.formal_charges[real] != 0):
        return False
    return all(IDX_TO_ELEMENT[int(t)] in {"C", "N", "O", "F"} for t in graph.atom_types[real])


def coverage(csv_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open()))
    n = len(rows)
    accepted = {"broad_organic_v1": 0, "broad_organic_neutral_v1": 0, "cnof_neutral": 0}
    excluded_broad: Counter = Counter()
    excluded_neutral: Counter = Counter()
    charged = 0
    for row in rows:
        smi = row[_SMILES_COL]
        ok_broad, reason_broad = classify_smiles(smi, BROAD_ORGANIC_V1)
        ok_neutral, reason_neutral = classify_smiles(smi, BROAD_ORGANIC_NEUTRAL_V1)
        ok_cnof = _cnof_neutral_ok(smi)
        accepted["broad_organic_v1"] += int(ok_broad)
        accepted["broad_organic_neutral_v1"] += int(ok_neutral)
        accepted["cnof_neutral"] += int(ok_cnof)
        if not ok_broad:
            excluded_broad[reason_broad] += 1
        if not ok_neutral:
            excluded_neutral[reason_neutral] += 1
        if int(row.get("formal_charge", "0")) != 0:
            charged += 1
    return {
        "benchmark": "jin_iclr19_qed_test_exact_v1",
        "smiles_column": _SMILES_COL,
        "n_leads": n,
        "accepted": accepted,
        "accepted_fraction": {k: round(v / n, 4) for k, v in accepted.items()},
        "broad_recovers_over_cnof": accepted["broad_organic_v1"] - accepted["cnof_neutral"],
        "excluded_features_broad_scope": dict(excluded_broad),
        "excluded_features_neutral_scope": dict(excluded_neutral),
        "leads_with_nonzero_net_charge": charged,
        "scope_hashes": {
            "broad_organic_v1": BROAD_ORGANIC_V1.scope_hash(),
            "broad_organic_neutral_v1": BROAD_ORGANIC_NEUTRAL_V1.scope_hash(),
        },
    }


def main() -> int:
    result = coverage(REPO / _JIN)
    n = result["n_leads"]
    print(f"{result['benchmark']}: {n} leads")
    for scope, count in result["accepted"].items():
        print(f"  {scope:28s}: {count}/{n} = {count / n:.1%}")
    print(f"  broad recovers over CNOF-neutral : {result['broad_recovers_over_cnof']} leads")
    print(f"  charged leads (kept by broad only): {result['leads_with_nonzero_net_charge']}/{n}")
    print(f"  excluded under broad             : {result['excluded_features_broad_scope'] or 'none'}")
    out = REPO / "diagnostics/composition/benchmark_lead_scope_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
