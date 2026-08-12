"""Open the sealed 67-pair panel under the preregistration and its amendment.

Reporting follows `diagnostics/editing_v2_sealed67_preregistration.json` and the
pre-outcome amendment that fixed the primary denominator at the 65
endpoint-clean pairs:

  PRIMARY      65 pairs whose source endpoint does not appear in the 24-pair
               development panel;
  SENSITIVITY  all 67 originally sealed pairs.

Exclusion is recovered here by joining shard `index` against the sealed panel
order and then against the amendment's `excluded_pairs`. The app passed the
flag into each container but rebuilt its result payload from an explicit key
list, so the flag never came back out; the criterion is mechanical and was
committed before any outcome was evaluated, so applying it at analysis time is
arithmetic, not a choice. The join is asserted endpoint-for-endpoint, not
assumed.

H1 is reported as a paired effect size with an exact interval, NOT as a
McNemar p-value. The verified controller always keeps greedy's action in the
shortlist and only overrides on strict improvement, so greedy-only wins are
structurally impossible: the discordance is one-sided BY CONSTRUCTION, and a
p-value would dress a design property as a finding. With b = 0 the paired
difference in recovery rate is exactly c/n, so a Clopper-Pearson interval on
c/n is the exact paired interval.

H2 is reported two ways because the development panel showed they are not
interchangeable: gain retention (a magnitude ratio) AND the actual rescue-set
overlap (which pairs).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ("greedy", "full", "sim1", "sim2", "hphi1", "ref1")


def interval(successes: int, total: int) -> tuple[float, float]:
    """Clopper-Pearson 95%. Exact, and correct at the boundary counts."""
    low = 0.0 if successes == 0 else stats.beta.ppf(0.025, successes, total - successes + 1)
    high = 1.0 if successes == total else stats.beta.ppf(0.975, successes + 1, total - successes)
    return float(low), float(high)


def summarise(rows: list[dict]) -> dict:
    greedy = sum(r["arms"]["greedy"]["recovered"] for r in rows)
    full = sum(r["arms"]["full"]["recovered"] for r in rows)
    headroom = full - greedy
    full_rescues = {r["index"] for r in rows
                    if r["arms"]["full"]["recovered"]
                    and not r["arms"]["greedy"]["recovered"]}

    out = {}
    for name in ARMS:
        recovered = sum(r["arms"][name]["recovered"] for r in rows)
        evaluations = sum(r["arms"][name]["continuation_evaluations"] for r in rows)
        universe = sum(r["arms"][name]["universe_candidates_seen"] for r in rows)
        rescues = {r["index"] for r in rows
                   if r["arms"][name]["recovered"]
                   and not r["arms"]["greedy"]["recovered"]}
        lost = [r["index"] for r in rows
                if r["arms"]["greedy"]["recovered"] and not r["arms"][name]["recovered"]]
        out[name] = {
            "recovered": recovered,
            "rate": recovered / len(rows),
            "extra_over_greedy": recovered - greedy,
            # magnitude ratio -- how much of the gain, not which pairs
            "gain_retention": (recovered - greedy) / headroom if headroom else None,
            # set overlap -- which pairs, the thing the 24 showed differs
            "rescue_overlap_with_full": len(rescues & full_rescues),
            "rescues_full_missed": sorted(rescues - full_rescues),
            "full_rescues_missed": sorted(full_rescues - rescues),
            "continuations_total": evaluations,
            "continuations_per_pair": evaluations / len(rows),
            "fraction_of_universe_evaluated": evaluations / universe if universe else None,
            "lost_versus_greedy": lost,
            "overrides": sum(r["arms"][name]["overrides"] for r in rows),
            "mean_best_similarity": float(np.mean([r["arms"][name]["best"] for r in rows])),
        }
    out["_meta"] = {
        "pairs": len(rows),
        "greedy_recovered": greedy,
        "full_recovered": full,
        "headroom": headroom,
        "full_rescue_indices": sorted(full_rescues),
    }
    return out


def paired(rows: list[dict], arm: str, base: str = "greedy") -> dict:
    """Paired binary comparison. b is structurally 0; see the module docstring."""
    b = [r["index"] for r in rows
         if r["arms"][base]["recovered"] and not r["arms"][arm]["recovered"]]
    c = [r["index"] for r in rows
         if r["arms"][arm]["recovered"] and not r["arms"][base]["recovered"]]
    n = len(rows)
    lo, hi = interval(len(c), n) if not b else (float("nan"), float("nan"))
    base_fail = sum(1 for r in rows if not r["arms"][base]["recovered"])
    r_lo, r_hi = interval(len(c), base_fail) if base_fail else (float("nan"), float("nan"))
    return {
        "arm": arm,
        "discordant_base_only": len(b),
        "discordant_arm_only": len(c),
        "difference_in_rate": len(c) / n,
        "difference_ci95": [lo, hi],
        "exact_because_b_is_zero": not b,
        "base_failures": base_fail,
        "rescue_rate_among_base_failures": len(c) / base_fail if base_fail else None,
        "rescue_rate_ci95": [r_lo, r_hi],
    }


def table(label: str, s: dict) -> None:
    n = s["_meta"]["pairs"]
    print(f"--- {label} (n={n}) ---")
    print(f"{'arm':>7} {'recov':>6} {'rate':>6} {'extra':>6} {'gain ret':>9} "
          f"{'overlap':>9} {'cont/pair':>10} {'% univ':>7} {'lost':>5}")
    head = s["_meta"]["headroom"]
    for name in ARMS:
        a = s[name]
        ret = "  n/a" if a["gain_retention"] is None else f"{a['gain_retention']:>8.0%}"
        ov = "n/a" if name == "greedy" else f"{a['rescue_overlap_with_full']}/{head}"
        print(f"{name:>7} {a['recovered']:>6} {a['rate']:>5.0%} "
              f"{a['extra_over_greedy']:>6} {ret:>9} {ov:>9} "
              f"{a['continuations_per_pair']:>10.1f} "
              f"{a['fraction_of_universe_evaluated']:>6.0%} "
              f"{len(a['lost_versus_greedy']):>5}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--seal", type=Path,
                        default=Path("diagnostics/editing_v2_controller_panel_seal.json"))
    parser.add_argument("--amendment", type=Path,
                        default=Path("diagnostics/editing_v2_sealed67_amendment.json"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    files = [f for f in glob.glob(str(args.shards / "**" / "*.json"), recursive=True)
             if "aggregate" not in f]
    rows = sorted((json.loads(Path(f).read_text()) for f in files), key=lambda r: r["index"])
    sealed = json.loads(args.seal.read_text())["sealed"]
    amendment = json.loads(args.amendment.read_text())

    if len(rows) != len(sealed):
        raise SystemExit(f"{len(rows)} shards but {len(sealed)} sealed pairs")

    # The index join is the whole basis for applying the amendment, so verify it
    # endpoint-for-endpoint rather than trusting enumeration order.
    for row in rows:
        pair = sealed[row["index"]]
        if (row["source"], row["target"]) != (pair["source"], pair["target"]):
            raise SystemExit(f"index {row['index']} does not match the sealed panel")

    excluded_keys = {(p["source"], p["target"]) for p in amendment["excluded_pairs"]}
    excluded = {i for i, p in enumerate(sealed) if (p["source"], p["target"]) in excluded_keys}
    if len(excluded) != amendment["counts"]["excluded"]:
        raise SystemExit(f"amendment names {amendment['counts']['excluded']} pairs, "
                         f"matched {len(excluded)}")
    primary = [r for r in rows if r["index"] not in excluded]
    print(f"{len(rows)} pairs run; excluded {sorted(excluded)}; primary {len(primary)}\n")

    prim, sens = summarise(primary), summarise(rows)
    table("PRIMARY -- 65 endpoint-clean pairs", prim)
    print()
    table("SENSITIVITY -- all 67 originally sealed", sens)

    print("\nH1  R_full > R_greedy, paired, on the primary 65")
    h1 = paired(primary, "full")
    print(f"  greedy {prim['_meta']['greedy_recovered']}/{len(primary)} "
          f"-> full {prim['_meta']['full_recovered']}/{len(primary)}")
    print(f"  discordant: full-only {h1['discordant_arm_only']}, "
          f"greedy-only {h1['discordant_base_only']} (structurally 0)")
    print(f"  difference in recovery rate {h1['difference_in_rate']:.1%} "
          f"[{h1['difference_ci95'][0]:.1%}, {h1['difference_ci95'][1]:.1%}] exact")
    print(f"  rescue rate among greedy's {h1['base_failures']} failures "
          f"{h1['rescue_rate_among_base_failures']:.1%} "
          f"[{h1['rescue_rate_ci95'][0]:.1%}, {h1['rescue_rate_ci95'][1]:.1%}]")

    print("\nH2  sim2 + verified rollout, the preregistered efficient controller")
    head = prim["_meta"]["headroom"]
    for name in ("sim2", "hphi1", "sim1", "ref1"):
        a = prim[name]
        print(f"  {name:>5}  gain retention {a['gain_retention']:>4.0%}   "
              f"rescue-set overlap {a['rescue_overlap_with_full']}/{head}   "
              f"own rescues full missed {len(a['rescues_full_missed'])}   "
              f"continuations {a['fraction_of_universe_evaluated']:.0%} of universe")

    print("\nby verified path length (primary 65)")
    print(f"{'steps':>6} {'n':>4} " + " ".join(f"{a:>7}" for a in ARMS))
    horizon = {}
    for steps in sorted({r["verified_steps"] for r in primary}):
        bucket = [r for r in primary if r["verified_steps"] == steps]
        counts = {a: sum(r["arms"][a]["recovered"] for r in bucket) for a in ARMS}
        horizon[str(steps)] = {"n": len(bucket), **counts}
        print(f"{steps:>6} {len(bucket):>4} " +
              " ".join(f"{counts[a]:>7}" for a in ARMS))

    kernel = sum(r["kernel_calls"] for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    print(f"\nkernel calls {kernel:,}   container time {seconds / 3600:.1f} h")

    violations = [a for a in ARMS if sens[a]["lost_versus_greedy"]]
    print("\nSAFETY -- greedy stays in every shortlist and the verified continuation")
    print("makes the final call, so no arm may lose a pair greedy recovers.")
    print(f"  {'HELD for every arm.' if not violations else f'VIOLATED by {violations} -- a BUG.'}")

    args.out.write_text(json.dumps({
        "schema": "compose.editing_v2.sealed67_result",
        "status": "SEALED_PANEL_OPENED_UNDER_PREREGISTRATION",
        "preregistration": "diagnostics/editing_v2_sealed67_preregistration.json",
        "amendment": "diagnostics/editing_v2_sealed67_amendment.json",
        "provenance": {
            "app_sha256_preregistered": ("232d66ce3823388e641159bbcc44d4eef29502c5"
                                         "f353ba986a2b8a68bc114d1f  (commit 3e1172b)"),
            "app_sha256_as_run": ("e9f48cbb85112465e8f1423bc84924fe4c56c663b3b5"
                                  "13a20edc3607f9b95c39  (commit 8d5c987)"),
            "why_they_differ": (
                "The preregistration and the amendment both recorded the app hash "
                "BEFORE the amendment was wired into the runner. `git diff 3e1172b "
                "8d5c987 -- modal_apps/h_phi_verified_hybrid_app.py` is the whole "
                "difference: it mounts the amendment file and stamps "
                "excluded_from_primary on each row for a printed count. run_pair, "
                "the six arms and every shortlist are byte-identical, so the "
                "controller as executed is the preregistered controller."),
            "known_defect": (
                "That wiring stamped the flag on the task but run_pair rebuilt its "
                "result from an explicit key list, so the flag never returned. "
                "Nothing behavioural read it, which is why the loss was silent. "
                "The split is recovered here by joining index against the sealed "
                "panel and the committed amendment, asserted endpoint-for-endpoint. "
                "The app has since been patched to emit the field."),
        },
        "excluded_indices": sorted(excluded),
        "primary": prim,
        "sensitivity": sens,
        "H1": h1,
        "H1_note": ("Paired effect size with an exact interval. No McNemar p-value: "
                    "greedy-only wins are impossible by construction, so the "
                    "one-sided discordance is a design property."),
        "H2": {name: {"gain_retention": prim[name]["gain_retention"],
                      "rescue_overlap_with_full": prim[name]["rescue_overlap_with_full"],
                      "rescues_full_missed": prim[name]["rescues_full_missed"],
                      "full_rescues_missed": prim[name]["full_rescues_missed"]}
               for name in ("sim1", "sim2", "hphi1", "ref1")},
        "by_verified_steps": horizon,
        "kernel_calls": kernel,
        "container_seconds": seconds,
        "safety_property_held": not violations,
        "per_pair": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
