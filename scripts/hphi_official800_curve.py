"""READ-ONLY aggregator for the OFFICIAL GrIDDD/Jin QED 800.

Modelled on scripts/hphi_valid128_curve.py, which produced the banked
128-source curve, and using the IDENTICAL success criterion:

    arm read : record['arms']['restart']['candidates']
    solved@k : any(c['success'] for c in candidates_sorted_by_k[:k])

`success` is written by run_source as in_region(terminal_qed, terminal_sim,
(0.90, 0.40)), i.e. QED >= 0.90 AND Morgan-Tanimoto >= 0.40 to the source, both
inclusive. This script RE-DERIVES that from the stored terminal_qed /
terminal_sim and asserts it agrees with the stored flag, so the number never
rests on a flag alone.

Nothing here writes to the Volume, resamples, retries or filters. A slot whose
trajectory never entered the region returned the unmodified source and stays in
the denominator as a failure.

    python scripts/hphi_official800_curve.py [--no-fetch]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

VOL = "compose-v4-artifacts"
REMOTE = "editing_v2/r_theta_run"
BASE_DIR = "hphi_official800_k8"          # k = 0..7
EXT_DIRS = [("hphi_official800_k812", "k8-12"),   # +4 trajectories, slots 8..11
            ("hphi_official800_k820", "k8-20")]   # the later K=20 extension
LOCAL = Path("/tmp/official800")
REGION = (0.90, 0.40)
N_SOURCES = 800


def fetch(sub: str) -> Path:
    """`modal volume get VOL <remote>/<sub> LOCAL` lands files at LOCAL/<sub>/."""
    LOCAL.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOL, f"{REMOTE}/{sub}",
                    str(LOCAL), "--force"], capture_output=True, text=True)
    return LOCAL / sub


def load_dir(sub: str) -> dict[int, list[dict]]:
    """Records land under <LOCAL>/<sub>/ regardless of the remote prefix."""
    out: dict[int, list[dict]] = {}
    for p in sorted((LOCAL / sub).glob("*.json")) if (LOCAL / sub).exists() else []:
        d = json.loads(p.read_text())
        arms = d.get("arms", {})
        if "restart" not in arms:
            print(f"  !! {p.name}: no restart arm", file=sys.stderr)
            continue
        out[int(d["index"])] = arms["restart"]["candidates"]
    return out


def main() -> None:
    if "--no-fetch" not in sys.argv:
        fetch(BASE_DIR)
        for sub, _ in EXT_DIRS:
            fetch(sub)

    base = load_dir(BASE_DIR)
    exts = {sub: load_dir(sub) for sub, _ in EXT_DIRS}

    cands: dict[int, list[dict]] = {}
    bad: list[str] = []
    for i, c in base.items():
        merged = list(c)
        for sub, _ in EXT_DIRS:
            merged += exts.get(sub, {}).get(i, [])
        merged = sorted(merged, key=lambda x: int(x["k"]))
        ks = [int(x["k"]) for x in merged]
        if ks != list(range(len(ks))):
            bad.append(f"src {i}: slot indices {ks}")
            continue
        cands[i] = merged
    for line in bad:
        print("  !! " + line, file=sys.stderr)

    n = len(cands)
    if n == 0:
        print("no records landed yet")
        return
    K = min(len(v) for v in cands.values())
    Kmax = max(len(v) for v in cands.values())

    # Re-derive the benchmark event; never trust the stored flag alone.
    disagree = 0
    for v in cands.values():
        for c in v:
            derived = (float(c["terminal_qed"]) >= REGION[0]
                       and float(c["terminal_sim"]) >= REGION[1])
            if derived != bool(c["success"]):
                disagree += 1
    print(f"landed {n}/{N_SOURCES} sources   slots/source {K}"
          f"{'' if K == Kmax else f'..{Kmax}'}   "
          f"criterion re-derivation disagreements: {disagree}")
    assert disagree == 0, "stored success flag disagrees with QED/sim re-derivation"

    cum = [sum(1 for v in cands.values()
               if any(c["success"] for c in v[:k + 1])) for k in range(K)]
    print(f"\n{'candidates':<12}" + "".join(f"{k + 1:>6}" for k in range(K)))
    print(f"{'cumulative':<12}" + "".join(f"{c:>6}" for c in cum))
    print(f"{'increment':<12}" + "".join(
        f"{(cum[0] if k == 0 else cum[k] - cum[k - 1]):>6}" for k in range(K)))
    print(f"{'rate %':<12}" + "".join(f"{cum[k] / n * 100:>6.1f}" for k in range(K)))

    print(f"\nsuccess@{K} = {cum[K - 1]}/{n} = {cum[K - 1] / n * 100:.1f}%"
          f"  ({'OFFICIAL 800 COMPLETE' if n == N_SOURCES else 'PROVISIONAL, partial panel'})")
    print("This is an oracle/candidate-EFFICIENCY operating point at K="
          f"{K}. GrIDDD's published 45.1% is at K=20 and is NOT matched by it.")

    runs = sum(len(v) for v in cands.values())
    ext = sum(1 for v in cands.values() for c in v if c.get("extinct"))
    unmod = sum(1 for v in cands.values() for c in v
                if float(c["terminal_sim"]) >= 0.9999)
    print(f"\nslots {runs}   extinct {ext} ({ext / runs * 100:.1f}%)"
          f"   returned-unmodified-source {unmod} ({unmod / runs * 100:.1f}%)")

    out = Path("/Users/rmaganti/compose_v2_work/docs/OFFICIAL800_QED_CURVE.json")
    out.write_text(json.dumps(
        {"panel": "griddd_jin_qed_test_800", "landed": n, "of": N_SOURCES,
         "K": K, "region": {"qed_min": REGION[0], "tanimoto_min": REGION[1]},
         "cumulative_solved": cum,
         "rate": {str(k + 1): cum[k] / n for k in range(K)},
         "slots": runs, "extinct": ext,
         "complete": n == N_SOURCES}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
