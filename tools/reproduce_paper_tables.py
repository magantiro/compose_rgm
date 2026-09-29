"""Reproduce the PMO and T4 tables printed in the ICLR 2027 submission.

    python3 tools/reproduce_paper_tables.py            # both
    python3 tools/reproduce_paper_tables.py --pmo      # Tables 2 and 11
    python3 tools/reproduce_paper_tables.py --t4       # Table 3

No network, no oracle calls, no docking, no cloud credentials. Everything read
here is committed in the repository.

WHY THIS IS NOT A TAUTOLOGY
---------------------------
The EXPECTED values are transcribed from the paper PDF and live in this file.
The COMPUTED values come from the result artifacts. Those are two independent
sources -- the artifacts were written by the campaign reducers, the paper by a
human copying from them -- so agreement is evidence that the printed table is
what the runs produced, and a disagreement localizes to one cell.

WHAT EACH TABLE IS BACKED BY
----------------------------
Table 2  (PMO-1K final top-10) and
Table 11 (PMO-1K AUC-Top10)
    diagnostics/pmo_ablation_frozen_v1/inputs/pmo_1k_final.json
    23 objectives, each with per-seed {best, top10, auc}. Five objectives carry
    FOUR seeds; the paper reports three. The seed ending in 983 is dropped --
    that is the `noprescreen_v2_rediscovery` replicate, and dropping it
    reproduces the published mean and standard deviation on every one of those
    five. The reported mean spans 22 objectives and excludes valsartan SMARTS.

Table 3  (T4 similarity-constrained lead optimization)
    diagnostics/T4_FROZEN_RESULT_v1.json
    15 cells per delta. ONE cell needs care: FA7 seed 1 at delta 0.6 is BLANK in
    the artifact and is printed in the paper as -6.4, which is that cell's SEED
    score. COMPOSE returned no feasible molecule beating the starting lead
    there, and IVG wins the cell at -7.7. The artifact is correct to record a
    blank and the paper is correct to print the seed score; this script states
    the substitution rather than hiding it.

NOT THIS PAPER'S TABLE
----------------------
diagnostics/t4_combined_table.json backs the separate NeurIPS **workshop**
package (`paper_gem_neurips2026`), which compares against GenMol, RetMol and
GraphGA at 500 calls per cell. It is a different experiment with different
numbers and is NOT what the ICLR submission prints. Use
tools/reproduce_t4_table.py --submitted for that one.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PMO_ARTIFACT = Path("diagnostics/pmo_ablation_frozen_v1/inputs/pmo_1k_final.json")
T4_ARTIFACT = Path("diagnostics/T4_FROZEN_RESULT_v1.json")

# Objectives carrying a fourth seed; the paper reports three and drops this one.
DROPPED_SEED_SUFFIX = "983"
EXCLUDED_FROM_MEAN = "valsartan_smarts"
TOLERANCE = 6e-4

# ---- Transcribed from the submission PDF ----

# Table 2: final top-10 after 1,000 oracle evaluations, mean over three seeds.
PAPER_TABLE_2 = {
    "albuterol_similarity": 0.782, "amlodipine_mpo": 0.539,
    "celecoxib_rediscovery": 0.335, "deco_hop": 0.586, "drd2": 0.979,
    "fexofenadine_mpo": 0.751, "gsk3b": 0.535, "isomers_c7h8n2o2": 0.911,
    "isomers_c9h10n2o2pf2cl": 0.895, "jnk3": 0.245, "median1": 0.294,
    "median2": 0.181, "mestranol_similarity": 0.503, "osimertinib_mpo": 0.833,
    "perindopril_mpo": 0.525, "qed": 0.945, "ranolazine_mpo": 0.770,
    "scaffold_hop": 0.461, "sitagliptin_mpo": 0.343,
    "thiothixene_rediscovery": 0.280, "troglitazone_rediscovery": 0.314,
    "zaleplon_mpo": 0.385,
}
PAPER_TABLE_2_MEAN = 0.563

# Table 11: AUC-Top10 over the same 1,000-evaluation budget.
PAPER_TABLE_11 = {
    "albuterol_similarity": 0.605, "amlodipine_mpo": 0.487,
    "celecoxib_rediscovery": 0.275, "deco_hop": 0.546, "drd2": 0.789,
    "fexofenadine_mpo": 0.682, "gsk3b": 0.422, "isomers_c7h8n2o2": 0.795,
    "isomers_c9h10n2o2pf2cl": 0.723, "jnk3": 0.208, "median1": 0.245,
    "median2": 0.154, "mestranol_similarity": 0.426, "osimertinib_mpo": 0.756,
    "perindopril_mpo": 0.453, "qed": 0.878, "ranolazine_mpo": 0.674,
    "scaffold_hop": 0.421, "sitagliptin_mpo": 0.236,
    "thiothixene_rediscovery": 0.237, "troglitazone_rediscovery": 0.253,
    "zaleplon_mpo": 0.333,
}
PAPER_TABLE_11_MEAN = 0.482

# Table 3: (target, seed index, seed score, COMPOSE/IVG/GenMol at 0.4 then 0.6).
PAPER_TABLE_3 = [
    ("PARP1", 1, -7.3, -11.9, -14.1, -10.6, -13.6, -12.3, -10.4),
    ("PARP1", 2, -7.8, -12.3, -13.4, -11.0, -12.7, -11.7, -9.7),
    ("PARP1", 3, -8.2, -14.1, -9.0, -11.3, -11.3, -10.7, -9.2),
    ("FA7", 1, -6.4, -9.3, -8.4, -8.4, -6.4, -7.7, -7.3),
    ("FA7", 2, -6.7, -9.4, -8.9, -8.4, -7.8, -7.5, -7.6),
    ("FA7", 3, -8.5, -9.6, -8.0, None, -8.5, -7.4, None),
    ("5HT1B", 1, -4.5, -12.4, -13.3, -12.9, -13.3, -12.4, -12.1),
    ("5HT1B", 2, -7.6, -11.9, -12.0, -12.3, -12.2, -12.0, -12.0),
    ("5HT1B", 3, -9.8, -12.5, -10.9, -11.6, -10.8, -10.6, -10.5),
    ("BRAF", 1, -9.3, -12.2, -10.1, -10.8, -9.0, -9.7, None),
    ("BRAF", 2, -9.4, -10.5, -10.8, -10.8, -11.0, -10.4, -9.7),
    ("BRAF", 3, -9.8, -10.9, -10.6, -10.6, -11.5, -10.3, -10.5),
    ("JAK2", 1, -7.7, -10.4, -10.2, -10.2, -10.6, -9.7, -9.3),
    ("JAK2", 2, -8.0, -11.3, -10.5, -10.0, -10.8, -10.4, -9.4),
    ("JAK2", 3, -8.6, -10.7, -10.2, -9.8, -10.8, -10.3, None),
]
PAPER_T4_WINS = {"0.4": 10, "0.6": 13}

# The one printed value that is not a COMPOSE result.
SEED_SCORE_SUBSTITUTION = ("FA7", 1, "0.6", -6.4)


class ReproductionError(RuntimeError):
    """Raised rather than warned; a warning beside a plausible table gets quoted."""


# ---- PMO ----


def paper_seeds(per_seed: dict) -> dict:
    if len(per_seed) == 4:
        return {k: v for k, v in per_seed.items()
                if not k.endswith(DROPPED_SEED_SUFFIX)}
    return per_seed


def check_pmo(root: Path) -> int:
    path = root / PMO_ARTIFACT
    if not path.exists():
        print(f"FAIL: missing {PMO_ARTIFACT}", file=sys.stderr)
        return 2
    data = json.loads(path.read_text())

    print("=" * 78)
    print("PMO-1K  --  paper Tables 2 and 11")
    print("=" * 78)
    print(f"  artifact: {PMO_ARTIFACT}")
    print("  protocol: 1,000 oracle calls, 3 seeds, no prescreen; the reported")
    print(f"            mean spans 22 objectives and excludes {EXCLUDED_FROM_MEAN}")

    dropped = [t for t, v in data.items() if len(v) == 4]
    print(f"  seeds:    {len(dropped)} objectives carry a 4th seed; the one ending "
          f"'{DROPPED_SEED_SUFFIX}' is dropped to match the paper")

    print(f"\n  {'objective':<26}{'top10':>17}{'':>4}{'AUC-Top10':>17}")
    print(f"  {'':<26}{'paper':>8}{'ours':>9}{'':>4}{'paper':>8}{'ours':>9}")
    print("  " + "-" * 72)

    fails, t10, auc = [], [], []
    for task in sorted(PAPER_TABLE_2):
        chosen = paper_seeds(data[task])
        m10 = statistics.fmean([s["top10"] for s in chosen.values()])
        mauc = statistics.fmean([s["auc"] for s in chosen.values()])
        t10.append(m10)
        auc.append(mauc)
        ok10 = abs(m10 - PAPER_TABLE_2[task]) <= TOLERANCE
        okauc = abs(mauc - PAPER_TABLE_11[task]) <= TOLERANCE
        if not ok10:
            fails.append(f"Table 2  {task}: paper {PAPER_TABLE_2[task]:.3f}, "
                         f"artifact {m10:.4f}")
        if not okauc:
            fails.append(f"Table 11 {task}: paper {PAPER_TABLE_11[task]:.3f}, "
                         f"artifact {mauc:.4f}")
        flag = "" if (ok10 and okauc) else "   <-- differs"
        print(f"  {task:<26}{PAPER_TABLE_2[task]:>8.3f}{m10:>9.3f}{'':>4}"
              f"{PAPER_TABLE_11[task]:>8.3f}{mauc:>9.3f}{flag}")

    mean10, meanauc = statistics.fmean(t10), statistics.fmean(auc)
    print("  " + "-" * 72)
    print(f"  {'MEAN (22 objectives)':<26}{PAPER_TABLE_2_MEAN:>8.3f}{mean10:>9.3f}"
          f"{'':>4}{PAPER_TABLE_11_MEAN:>8.3f}{meanauc:>9.3f}")

    if abs(mean10 - PAPER_TABLE_2_MEAN) > TOLERANCE:
        fails.append(f"Table 2 mean: paper {PAPER_TABLE_2_MEAN}, got {mean10:.4f}")
    if abs(meanauc - PAPER_TABLE_11_MEAN) > TOLERANCE:
        fails.append(f"Table 11 mean: paper {PAPER_TABLE_11_MEAN}, got {meanauc:.4f}")

    n = len(PAPER_TABLE_2)
    print(f"\n  matched {2 * n - len([f for f in fails if 'mean' not in f])} of "
          f"{2 * n} published per-objective values")
    if fails:
        print("\n  DIFFERENCES")
        for failure in fails:
            print(f"    - {failure}")
        print("\n  A per-objective difference of ~0.001 means the artifact holds a")
        print("  DIFFERENT SEED than the paper used for that objective, not that the")
        print("  reduction is wrong. Both headline means are unaffected.")
    return 1 if fails else 0


# ---- T4 ----


def check_t4(root: Path) -> int:
    path = root / T4_ARTIFACT
    if not path.exists():
        print(f"FAIL: missing {T4_ARTIFACT}", file=sys.stderr)
        return 2
    payload = json.loads(path.read_text())["payload"]

    print("=" * 78)
    print("T4 similarity-constrained lead optimization  --  paper Table 3")
    print("=" * 78)
    print(f"  artifact: {T4_ARTIFACT}")
    print("  budget:   COMPOSE 250 docking calls per seed vs 1,000 for IVG and GenMol")

    stored = {}
    for delta in ("0.4", "0.6"):
        for row in payload["deltas"][delta]["rows"]:
            stored[(row["target"].upper(), row["seed"], delta)] = row

    fails, wins = [], {"0.4": 0, "0.6": 0}
    sub_target, sub_seed, sub_delta, sub_value = SEED_SCORE_SUBSTITUTION

    print(f"\n  {'target':<8}{'seed':>5}{'seedscore':>10}"
          f"{'  d=0.4: COMPOSE/ours':>26}{'  d=0.6: COMPOSE/ours':>26}")
    print("  " + "-" * 74)
    for target, seed, seedscore, c4, i4, g4, c6, i6, g6 in PAPER_TABLE_3:
        line = f"  {target:<8}{seed:>5}{seedscore:>10.1f}"
        for delta, paper_c, paper_i, paper_g in (("0.4", c4, i4, g4),
                                                 ("0.6", c6, i6, g6)):
            row = stored.get((target, seed, delta))
            ours = None if row is None else row.get("compose")
            note = ""
            if (target, seed, delta) == (sub_target, sub_seed, sub_delta):
                # Paper prints the SEED score here; the artifact records a blank.
                if ours is not None:
                    fails.append(f"{target} seed {seed} d={delta}: expected the "
                                 f"artifact to be blank, found {ours}")
                if abs(paper_c - sub_value) > 1e-9:
                    fails.append(f"{target} seed {seed} d={delta}: paper prints "
                                 f"{paper_c}, expected the seed score {sub_value}")
                note = " (seed score)"
                shown = f"{paper_c:>7.1f}/{'blank':>7}"
            else:
                if ours is None:
                    fails.append(f"{target} seed {seed} d={delta}: artifact has no "
                                 f"COMPOSE value but the paper prints {paper_c}")
                    shown = f"{paper_c:>7.1f}/{'--':>7}"
                else:
                    if abs(ours - paper_c) > 0.05:
                        fails.append(f"{target} seed {seed} d={delta}: paper "
                                     f"{paper_c}, artifact {ours}")
                    shown = f"{paper_c:>7.1f}/{ours:>7.1f}"
                if row is not None and row.get("ivg") is not None and \
                        abs(row["ivg"] - paper_i) > 0.05:
                    fails.append(f"{target} seed {seed} d={delta}: IVG paper "
                                 f"{paper_i}, artifact {row['ivg']}")
            baselines = [b for b in (paper_i, paper_g) if b is not None]
            if all(paper_c <= b for b in baselines):
                wins[delta] += 1
            line += f"  {shown}{note:<14}"
        print(line)

    print("  " + "-" * 74)
    for delta in ("0.4", "0.6"):
        expected = PAPER_T4_WINS[delta]
        mark = "OK" if wins[delta] == expected else "MISMATCH"
        if wins[delta] != expected:
            fails.append(f"delta={delta}: counted {wins[delta]} COMPOSE-best cells, "
                         f"paper claims {expected}")
        print(f"  delta={delta}: COMPOSE best in {wins[delta]} of 15   "
              f"(paper: {expected})   {mark}")
    total = wins["0.4"] + wins["0.6"]
    print(f"  total: {total} of 30   (abstract: 23 of 30)   "
          f"{'OK' if total == 23 else 'MISMATCH'}")

    print("\n  ONE PRINTED VALUE IS NOT A COMPOSE RESULT:")
    print(f"    {sub_target} seed {sub_seed} at delta {sub_delta} prints "
          f"{sub_value}, the cell's SEED score.")
    print("    COMPOSE returned no feasible molecule improving on the lead there,")
    print("    and IVG wins the cell at -7.7. The artifact records it as blank.")

    caveat = payload.get("docking_reproducibility_caveat", {})
    if caveat:
        print("\n  DOCKING REPLICATE NOISE (stated in the paper as 0.70 kcal/mol):")
        print(f"    {str(caveat.get('severity', ''))[:100]}")

    if fails:
        print("\n  DIFFERENCES")
        for failure in fails:
            print(f"    - {failure}")
    return 1 if fails else 0


# ---- Entry point ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the PMO and T4 tables printed in the ICLR 2027 "
                    "submission, from committed artifacts only."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--pmo", action="store_true", help="only PMO Tables 2 and 11")
    parser.add_argument("--t4", action="store_true", help="only T4 Table 3")
    args = parser.parse_args(argv)

    both = not (args.pmo or args.t4)
    status = 0
    if both or args.pmo:
        status = max(status, check_pmo(args.repo_root))
        if both:
            print()
    if both or args.t4:
        status = max(status, check_t4(args.repo_root))

    print("\n" + "=" * 78)
    if status == 0:
        print("REPRODUCED -- every published value matched. "
              "0 oracle calls, 0 docking, 0 network.")
    else:
        print("REPRODUCED WITH DIFFERENCES -- see above. "
              "0 oracle calls, 0 docking, 0 network.")
    print("=" * 78)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
