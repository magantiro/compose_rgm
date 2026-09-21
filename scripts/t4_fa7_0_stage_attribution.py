"""Which STAGE kills fa7_0: the region draw, the replacement, or the eligibility intersection?

`scripts/t4_fa7_0_reachability_probe.py` answers whether any production
configuration reaches an eligible endpoint.  If the answer is no, that alone does
not say where the path fails, and the three candidate stages call for three
different repairs.  This driver separates them on the production path.

THE DISCRIMINATOR
-----------------
Every known eligible witness for this cell is a net excision of 8 or 9 heavy
atoms that relieves the QED alerts by cutting the naphthoyl tail.  So bucket
EVERY endpoint the production path constructs by its net heavy-atom change and
ask, within each bucket, how close it came on each axis:

  * the path never constructs anything at -8/-9 .............. REGION DRAW kills it
  * it constructs the bucket, but nothing there passes sim ... REPLACEMENT kills it
  * it populates the bucket and passes each gate separately,
    but never both at once ........................ ELIGIBILITY INTERSECTION kills it

The census is over CONSTRUCTED endpoints, not eligible ones, which is the whole
point: `expand` returns only survivors, so the interior is invisible from its
return value.  The gate itself is the unmodified production `Fiber.check`,
wrapped and cross-checked exactly as in the probe -- `gate_disagreements` must
be 0 or the attribution below is void.

ZERO ORACLE CALLS.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem

from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.control.region_law_contract import (
    CONTRACT_FIELD,
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    resolve_region_law,
)

from t4_fa7_0_reachability_probe import _FunnelFiber

SCHEMA_VERSION = "t4_fa7_0_stage_attribution_v1"


def _bucket_census(rows: list[dict], source_heavy: int) -> dict:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("parsed"):
            continue
        buckets[source_heavy - row["heavy"]].append(row)
    out = {}
    for excision in sorted(buckets):
        group = buckets[excision]
        sim_pass = [r for r in group if r["pass_similarity"]]
        qed_pass = [r for r in group if r["pass_qed"]]
        both = [r for r in group if r["pass_similarity"] and r["pass_qed"]]
        out[str(excision)] = {
            "constructed": len(group),
            "best_similarity": round(max(r["similarity"] for r in group), 6),
            "best_qed": round(max(r["qed"] for r in group), 6),
            "best_sa": round(min(r["sa"] for r in group), 4),
            "pass_similarity": len(sim_pass),
            "pass_qed": len(qed_pass),
            "pass_similarity_and_qed": len(both),
            "best_qed_among_similarity_passing": (
                round(max(r["qed"] for r in sim_pass), 6) if sim_pass else None
            ),
            "best_similarity_among_qed_passing": (
                round(max(r["similarity"] for r in qed_pass), 6) if qed_pass else None
            ),
        }
    return out


def run(
    contract: Path,
    *,
    cell_name: str,
    draws: int,
    horizon: int,
    seed_offset: int,
    law_on: bool,
    witness_path: Path | None,
) -> dict:
    payload = unseal(contract)
    cell = next(c for c in payload["cells"] if c["cell"] == cell_name)
    shaped = json.loads(json.dumps(payload))
    if law_on:
        shaped["proposal"][CONTRACT_LANE][CONTRACT_FIELD] = FREE_GATE_MARGIN_V1
    else:
        shaped["proposal"][CONTRACT_LANE].pop(CONTRACT_FIELD, None)
    law = resolve_region_law(shaped, delta=shaped["delta"], reference_smiles=cell["smiles"])

    fiber = Fiber(cell["smiles"], payload["delta"], support=payload["support"])
    probe = _FunnelFiber(fiber)
    source_heavy = Chem.MolFromSmiles(cell["smiles"]).GetNumHeavyAtoms()
    seed = int(cell["controller_seed"]) + seed_offset
    started = time.time()
    records = expand(
        cell["smiles"],
        0.0,
        fiber=probe,
        rng=np.random.default_rng(seed),
        draws=draws,
        multi_region=True,
        horizon=horizon,
        proposal_lane=CONTRACT_LANE,
        region_law=law,
    )
    rows = list(probe.rows.values())

    witness_hits = []
    if witness_path and witness_path.exists():
        coverage = json.loads(witness_path.read_text())
        wits = coverage["cells"][cell_name]["witnesses"]
        built = {r["canonical"] for r in rows if r.get("canonical")}
        for w in wits:
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(w["witness_smiles"]))
            witness_hits.append(
                {
                    "witness": canonical,
                    "heavy_delta": w["heavy_delta"],
                    "proposal_mode": w["proposal_mode"],
                    "constructed_by_this_arm": canonical in built,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": str(contract),
        "cell": cell_name,
        "delta": payload["delta"],
        "support": payload["support"],
        "source_smiles": cell["smiles"],
        "source_heavy_atoms": source_heavy,
        "region_law": FREE_GATE_MARGIN_V1 if law is not None else None,
        "horizon": horizon,
        "draws": draws,
        "seed": seed,
        "eligible": len(records),
        "gate_calls": probe.calls,
        "gate_disagreements": len(probe.disagreements),
        "distinct_endpoints_constructed": len(rows),
        "excision_census": _bucket_census(rows, source_heavy),
        "witness_recovery": witness_hits,
        "seconds": round(time.time() - started, 1),
        "oracle_calls": 0,
        "docking_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--cell", default="fa7_0")
    parser.add_argument("--draws", type=int, default=480)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--law", choices=("on", "off"), default="on")
    parser.add_argument(
        "--witnesses",
        type=Path,
        default=Path("diagnostics/t4_region_prefix_coverage_v1.json"),
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = run(
        args.contract,
        cell_name=args.cell,
        draws=args.draws,
        horizon=args.horizon,
        seed_offset=args.seed_offset,
        law_on=args.law == "on",
        witness_path=args.witnesses,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(
        f"{report['cell']} eligible={report['eligible']} "
        f"constructed={report['distinct_endpoints_constructed']} "
        f"disagree={report['gate_disagreements']} {report['seconds']}s",
        flush=True,
    )
    for excision, row in report["excision_census"].items():
        if row["constructed"] >= 5:
            print(
                f"  cut {excision:>4}: n={row['constructed']:<5} "
                f"simPass={row['pass_similarity']:<4} qedPass={row['pass_qed']:<4} "
                f"both={row['pass_similarity_and_qed']:<3} "
                f"bestQED|sim={row['best_qed_among_similarity_passing']}"
            )


if __name__ == "__main__":
    main()
