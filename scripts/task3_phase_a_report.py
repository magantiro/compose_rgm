#!/usr/bin/env python
"""Phase A discovery metrics, computed from the durable ledgers.

Reads ledgers rather than a run's return value, because a detached Modal launch
returns to the client immediately and its final summary is never printed
locally. The ledger is the record of what was actually charged, in order, so it
can answer every discovery question on its own -- which is the same property
that let two interrupted runs resume without losing work.

DEVELOPMENT-ONLY: official PROTOCOL (120 random ZINC-250k) on DEVELOPMENT seeds.
The five official initialization sets remain sealed.

THE OBJECTIVE IS DISCOVERY, NOT HYPERVOLUME. Early HV is printed last and
labelled, so it cannot drift into being read as the headline.

POWER WARNING, PRINTED WITH THE RESULT. The measured base rate of JNK3 activity
in random ZINC is 0.040% at >= 0.3, so ~2,380 search calls expect about ONE
chance active per arm. A one- or two-find difference between arms is Poisson
noise, not a result, and the script says so rather than leaving it to the reader.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from compose_v4.benchmark.molleo_task3 import hypervolume_qmc  # noqa: E402

ARMS = ("fixed-scalarization", "novelty-exploration")
THRESHOLDS = (0.3, 0.5)
#: Measured over 80,000 random ZINC molecules; see
#: diagnostics/task3_active_base_rate_BENCHMARK_PROPERTY.json
BASE_RATE = {0.3: 0.000400, 0.5: 0.000050}


def scaffold(smiles: str) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:  # noqa: BLE001
        return None


def analyse(path: Path, init_size: int) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    jnk3 = [row["v"][1] for row in rows]
    result = {"charged": len(rows)}
    for threshold in THRESHOLDS:
        hits = [i for i, value in enumerate(jnk3) if value >= threshold]
        # "Found by search" excludes anything already in the initialization,
        # which both arms share for a given seed and neither earned.
        searched = [i for i in hits if i >= init_size]
        scaffolds = {scaffold(rows[i]["smiles"]) for i in searched}
        scaffolds.discard(None)
        key = str(threshold).replace(".", "")
        result[f"first_active_{key}"] = (hits[0] + 1) if hits else None
        result[f"first_active_by_search_{key}"] = (searched[0] + 1) if searched else None
        result[f"actives_{key}"] = len(hits)
        result[f"actives_by_search_{key}"] = len(searched)
        result[f"scaffolds_by_search_{key}"] = len(scaffolds)
    result["best_jnk3"] = max(jnk3) if jnk3 else 0.0
    result["best_jnk3_in_init"] = max(jnk3[:init_size]) if jnk3 else 0.0
    result["early_hv"] = hypervolume_qmc(np.asarray([r["v"] for r in rows]),
                                         log2_samples=17)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledgers", type=Path, default=Path("/tmp/phase_a_ledgers"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--init-size", type=int, default=120)
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/task3_phase_a_discovery.json"))
    args = parser.parse_args()

    results: dict[str, list[dict]] = {}
    for arm in ARMS:
        for seed in args.seeds:
            path = args.ledgers / f"{arm}_seed{seed}.jsonl"
            if not path.exists():
                print(f"missing: {path}")
                continue
            row = analyse(path, args.init_size)
            row.update({"arm": arm, "seed": seed})
            results.setdefault(arm, []).append(row)

    search_calls = None
    print("PHASE A DISCOVERY -- development-only, official protocol on dev seeds")
    for arm, rows in results.items():
        rows.sort(key=lambda r: r["seed"])
        search_calls = rows[0]["charged"] - args.init_size
        print(f"\n{arm}")
        print(f"  charged                  {[r['charged'] for r in rows]}")
        print(f"  actives >=0.3 (total)    {[r['actives_03'] for r in rows]}")
        print(f"  actives >=0.3 BY SEARCH  {[r['actives_by_search_03'] for r in rows]}")
        print(f"  first active by search   {[r['first_active_by_search_03'] for r in rows]}")
        print(f"  active scaffolds         {[r['scaffolds_by_search_03'] for r in rows]}")
        moved = ["{:.2f}->{:.2f}".format(r["best_jnk3_in_init"], r["best_jnk3"])
                 for r in rows]
        print(f"  best jnk3 (init -> final) {moved}")
        print(f"  early HV (NOT the criterion) {[round(r['early_hv'], 4) for r in rows]}")

    # ---- power statement, printed with the result ------------------------
    if search_calls:
        for threshold in THRESHOLDS:
            expected = BASE_RATE[threshold] * search_calls
            key = str(threshold).replace(".", "")
            totals = {arm: sum(r[f"actives_by_search_{key}"] for r in rows)
                      for arm, rows in results.items()}
            n_seeds = max(len(rows) for rows in results.values())
            pooled_expected = expected * n_seeds
            print(f"\nPOWER at jnk3 >= {threshold}: {search_calls} search calls x "
                  f"{n_seeds} seeds expects {pooled_expected:.2f} chance actives "
                  f"per arm")
            for arm, total in totals.items():
                print(f"  {arm:<22} found {total}")
            gap = abs(list(totals.values())[0] - list(totals.values())[1]) if len(totals) == 2 else 0
            if pooled_expected > 0:
                # Poisson sd of a difference of two counts each ~pooled_expected
                sd = math.sqrt(2 * pooled_expected)
                print(f"  difference {gap} against a Poisson sd of {sd:.2f} on the "
                      f"difference -- {'NOT distinguishable from noise' if gap <= 2 * sd else 'larger than noise'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"STATUS": "DEVELOPMENT-ONLY -- official protocol, development seeds",
               "objective": "DISCOVERY, not hypervolume",
               "search_calls_per_arm": search_calls,
               "base_rate_used_for_power": BASE_RATE,
               "results": results}
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
