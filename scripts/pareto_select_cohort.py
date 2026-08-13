"""Freeze the held-in cohort for the Pareto-control smoke. NO CONTROLLER RUNS.

Selection is entirely pre-control and outcome-independent, and the result is a
COMMITTED artifact with a SHA-256 so the source set cannot be quietly reselected
after seeing a result. That is the same discipline
`retarget_select_calibration_sources.py` used, and it is the reason the
retargeting calibration's cohort could be audited afterwards.

Eligibility, all decided before any controller exists:

  1. held-in only (`training_source_keys`);
  2. parses and canonicalizes under the frozen chemistry;
  3. heavy-atom count in [18, 38] -- the frozen cohort size band;
  4. does NOT already sit at the top of either objective, because a source with
     no headroom on an axis cannot show preference-dependent movement along it:
       * not already inside the developability region (QED >= 0.6 and
         1.0 <= cLogP <= 4.0), the same exclusion the retargeting cohort used;
       * not already potent (P(active) >= 0.5);
  5. DISJOINT from the sources the Stage 0 census measured, re-derived here from
     the census's own deterministic selection procedure so the disjointness is
     verifiable rather than asserted.

No source is excluded because a controller did badly on it -- no controller has
run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Mirrors scripts/pareto_tradeoff_census.py exactly, so the census's own source
#: list can be reconstructed and excluded.
CENSUS_SEED = 20260813
CENSUS_POOL_SAMPLE = 20000
CENSUS_SOURCES = 60

MIN_HEAVY, MAX_HEAVY = 18, 38
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)
POTENCY_LOGODDS_CEILING = 0.0  # P(active) = 0.5
CANONICAL_SLOTS = 48
COHORT_SEED = 20260814


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", type=Path,
                        default=REPO / "diagnostics/editing_v2_matched_validation_reserve_ids.json.gz")
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--scan", type=int, default=400)
    parser.add_argument("--out", type=Path,
                        default=REPO / "diagnostics/pareto_control_cohort.json")
    args = parser.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, QED

    from compose_v4.drd2_oracle import load_default_oracle

    RDLogger.DisableLog("rdApp.*")
    oracle = load_default_oracle(str(args.manifest))

    split = json.load(gzip.open(args.reserve, "rt"))
    held_in = sorted(split["training_source_keys"])
    del split  # the held-out list is never carried past this point

    # --- Reconstruct the census's sources so they can be excluded -----------
    census_rng = random.Random(CENSUS_SEED)
    pool = list(held_in)
    census_rng.shuffle(pool)
    census_pool = pool[:CENSUS_POOL_SAMPLE]
    heavy = []
    for smi in census_pool:
        mol = Chem.MolFromSmiles(smi)
        heavy.append(0 if mol is None else mol.GetNumHeavyAtoms())
    census_eligible = [s for s, h in zip(census_pool, heavy)
                       if MIN_HEAVY <= h <= MAX_HEAVY]
    census_rng.shuffle(census_eligible)
    census_sources = set(census_eligible[:CENSUS_SOURCES])
    print(f"census sources reconstructed for exclusion: {len(census_sources)}")

    # --- Select the cohort from a disjoint shuffle ---------------------------
    rng = random.Random(COHORT_SEED)
    candidates = [s for s in held_in if s not in census_sources]
    rng.shuffle(candidates)

    rejected = {"unparseable": 0, "size_band": 0, "already_developable": 0,
                "already_potent": 0}
    chosen: list[dict] = []
    scanned = 0
    for smi in candidates:
        if len(chosen) >= args.size or scanned >= args.scan:
            break
        scanned += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rejected["unparseable"] += 1
            continue
        n_heavy = mol.GetNumHeavyAtoms()
        if not (MIN_HEAVY <= n_heavy <= MAX_HEAVY):
            rejected["size_band"] += 1
            continue
        try:
            qed = float(QED.qed(mol))
        except Exception:  # noqa: BLE001
            rejected["unparseable"] += 1
            continue
        clogp = float(Crippen.MolLogP(mol))
        if qed >= QED_FLOOR and LOGP_BOX[0] <= clogp <= LOGP_BOX[1]:
            rejected["already_developable"] += 1
            continue
        logodds = float(oracle.margin_many([smi])[0])
        if logodds >= POTENCY_LOGODDS_CEILING:
            rejected["already_potent"] += 1
            continue
        chosen.append({"index": len(chosen), "source": smi,
                       "slots": CANONICAL_SLOTS, "heavy_atoms": int(n_heavy),
                       "qed": round(qed, 4), "clogp": round(clogp, 4),
                       "drd2_logodds": round(logodds, 4)})

    if len(chosen) < args.size:
        raise SystemExit(f"only {len(chosen)} of {args.size} sources found in "
                         f"{scanned} scanned; raise --scan")

    body = json.dumps(chosen, sort_keys=True).encode()
    payload = {
        "schema": "compose.pareto.control_cohort",
        "status": "FROZEN_BEFORE_ANY_CONTROLLER_RUN_OUTCOME_INDEPENDENT",
        "pool": "held-in training sources only",
        "criteria": {
            "heavy_atoms": [MIN_HEAVY, MAX_HEAVY],
            "exclude_already_developable": {"qed_floor": QED_FLOOR,
                                            "logp_box": list(LOGP_BOX)},
            "exclude_already_potent": {"drd2_logodds_ceiling": POTENCY_LOGODDS_CEILING,
                                       "equivalent_p_active": 0.5},
            "disjoint_from_stage0_census": True,
            "rule": ("a source with no headroom on an axis cannot show "
                     "preference-dependent movement along it; no source is "
                     "excluded for a controller outcome, because no controller "
                     "has run"),
        },
        "seed": COHORT_SEED,
        "scanned": scanned,
        "rejected": rejected,
        "census_sources_excluded": len(census_sources),
        "cohort_sha256": hashlib.sha256(body).hexdigest(),
        "sources": chosen,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"scanned {scanned}, rejected {rejected}")
    print(f"cohort {len(chosen)} sources, sha256 {payload['cohort_sha256'][:16]}")
    print(f"median heavy atoms {np.median([c['heavy_atoms'] for c in chosen]):.0f}, "
          f"median QED {np.median([c['qed'] for c in chosen]):.3f}, "
          f"median cLogP {np.median([c['clogp'] for c in chosen]):.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
