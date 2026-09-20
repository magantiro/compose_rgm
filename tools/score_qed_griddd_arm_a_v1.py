"""Re-derive the paper-era COMPOSE QED result from its banked per-source records.

NEW FILE.  It exists because the committed number had no producer in this
repository.  ``diagnostics/griddd_official800_k8.json`` carries ``solved: 446``
but no script in this tree writes it; the only aggregator present,
``scripts/hphi_official800_curve.py``, writes a DIFFERENT file
(``docs/OFFICIAL800_QED_CURVE.json``) into a different checkout
(``/Users/rmaganti/compose_v2_work``) and its committed copy is a 199-source
partial snapshot.  A number nobody can recompute is a number nobody can check.

READ-ONLY.  It fetches nothing and launches nothing: point ``--records`` at a
local directory of per-source records previously pulled with
``modal volume get compose-v4-artifacts editing_v2/r_theta_run/hphi_official800_k8``.
That is a volume read, not a Modal run.

WHAT IT CHECKS, BEYOND RE-COUNTING
----------------------------------
1. every record's ``source`` equals ``data/jin/qed_test.txt`` at its own index,
   so the panel identity is proved rather than assumed;
2. QED and Tanimoto are RECOMPUTED from the returned SMILES with RDKit and
   compared against the stored ``terminal_qed`` / ``terminal_sim``, so the count
   does not rest on fields the run wrote about itself;
3. the benchmark event is re-derived as ``QED >= 0.90 and Tanimoto >= 0.40``
   and compared against the stored ``success`` flag;
4. both denominators are reported -- scored records, and the full 800 with
   unscored sources counted as failures -- because they differ and a rate
   quoted without its denominator is not checkable.

The similarity definition is the benchmark's: Morgan radius 2, 2048 bits,
chirality off, Tanimoto, always against the ORIGINAL source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REGION_QED = 0.90
REGION_SIM = 0.40
PANEL = 800


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="local dir of per-source records")
    parser.add_argument("--sources", default="data/jin/qed_test.txt")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    panel = [ln.strip() for ln in Path(args.sources).read_text().split("\n") if ln.strip()]
    if len(panel) != PANEL:
        raise SystemExit(f"expected {PANEL} sources, found {len(panel)}")

    records: dict[int, dict] = {}
    for path in sorted(Path(args.records).rglob("*.json")):
        try:
            body = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "index" in body and "arms" in body:
            records[int(body["index"])] = body
    if not records:
        raise SystemExit("no per-source records found")

    panel_mismatch: list[int] = []
    flag_disagreements = 0
    unparseable = 0
    max_qed_delta = 0.0
    max_sim_delta = 0.0
    solved: list[int] = []
    slot_shapes: dict[int, int] = {}
    cumulative = [0] * args.k
    distinct_successful: list[int] = []

    for index, body in sorted(records.items()):
        if body.get("source") and body["source"] != panel[index]:
            panel_mismatch.append(index)
        source_mol = Chem.MolFromSmiles(panel[index])
        source_fp = generator.GetFingerprint(source_mol)
        candidates = sorted(body["arms"]["restart"]["candidates"], key=lambda c: int(c["k"]))
        slot_shapes[len(candidates)] = slot_shapes.get(len(candidates), 0) + 1
        hit_at = None
        unique_successes = set()
        for slot, candidate in enumerate(candidates[: args.k]):
            returned = candidate.get("returned")
            molecule = Chem.MolFromSmiles(returned) if returned else None
            if molecule is None:
                unparseable += 1
                continue
            qed = float(QED.qed(molecule))
            similarity = float(
                DataStructs.TanimotoSimilarity(source_fp, generator.GetFingerprint(molecule))
            )
            max_qed_delta = max(max_qed_delta, abs(qed - float(candidate["terminal_qed"])))
            max_sim_delta = max(max_sim_delta, abs(similarity - float(candidate["terminal_sim"])))
            derived = qed >= REGION_QED and similarity >= REGION_SIM
            if derived != bool(candidate["success"]):
                flag_disagreements += 1
            if derived:
                if hit_at is None:
                    hit_at = slot
                if returned != panel[index]:
                    unique_successes.add(returned)
        if hit_at is not None:
            solved.append(index)
            distinct_successful.append(len(unique_successes))
            for slot in range(hit_at, args.k):
                cumulative[slot] += 1

    scored = len(records)
    report = {
        "schema_version": "qed_griddd_arm_a_rederivation_v1",
        "panel": "jin_iclr19_qed_test_800",
        "k": args.k,
        "region": {"qed_min": REGION_QED, "tanimoto_min": REGION_SIM},
        "similarity": "Morgan r=2, 2048 bits, chirality off, Tanimoto to the original source",
        "scored_sources": scored,
        "unscored_sources": sorted(i for i in range(PANEL) if i not in records),
        "solved": len(solved),
        "rate_over_scored": len(solved) / scored,
        "rate_over_full_panel": len(solved) / PANEL,
        "cumulative_solved_at_k": cumulative,
        "slots_per_source_histogram": {str(k): v for k, v in sorted(slot_shapes.items())},
        "verification": {
            "panel_source_mismatches": panel_mismatch,
            "unparseable_returned_smiles": unparseable,
            "stored_flag_disagreements": flag_disagreements,
            "max_abs_qed_recompute_delta": max_qed_delta,
            "max_abs_similarity_recompute_delta": max_sim_delta,
        },
        "distinct_successful_non_source_returns": {
            "mean": (sum(distinct_successful) / len(distinct_successful))
            if distinct_successful
            else None,
            "sources_with_at_least_two": sum(1 for n in distinct_successful if n >= 2),
            "sources_with_exactly_one": sum(1 for n in distinct_successful if n == 1),
        },
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
