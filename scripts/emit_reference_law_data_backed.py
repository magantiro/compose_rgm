"""Recompute the data-backed-only reference-law NLL from machine-readable provenance.

The mask is taken from ``editing_v2_family_lane_provenance.json`` -- no family is
named in this script.  No model evaluation and no training: this is a join of the
already-scored reference-law result against the emitted lane classification.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "diagnostics" / "editing_v2_experiment1_reference_law.json"
PROV   = ROOT / "diagnostics" / "editing_v2_family_lane_provenance.json"
DEST   = ROOT / "diagnostics" / "editing_v2_reference_law_data_backed.json"
ARMS = ("uniform_canonical", "empirical_family", "learned_family_uniform_id",
        "empirical_family_learned_id", "r_theta")
EXPECT = {"n": 2744, "empirical_family": 5.5477,
          "empirical_family_learned_id": 4.2462, "r_theta": 4.4907,
          "improvement_empirical_to_full": 1.0569}
TOL = 5e-4


def sha256_of(p: Path) -> str:
    h = hashlib.sha256(p.read_bytes()); return h.hexdigest()


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    S = json.loads(SCORES.read_text()); P = json.loads(PROV.read_text())
    fam_synth = {f: rec["synthetic_only"] for f, rec in P["families"].items()}

    def family_of(cell: str) -> str:
        return cell.split(":")[1] if ":" in cell else cell

    cells = S["by_capability_cell"]
    unknown = sorted({family_of(c) for c in cells} - set(fam_synth))
    if unknown:
        print(f"ERROR: families absent from the provenance artifact: {unknown}")
        return 1

    groups = {"data_backed": {}, "synthetic_only": {}}
    for c, v in cells.items():
        key = "synthetic_only" if fam_synth[family_of(c)] else "data_backed"
        groups[key][c] = v

    def agg(sub):
        n = sum(v["entries"] for v in sub.values())
        return n, {a: sum(v["entries"] * v[a] for v in sub.values()) / n for a in ARMS}

    out_groups = {}
    for name, sub in groups.items():
        n, A = agg(sub)
        A = {a: round(v, 6) for a, v in A.items()}
        out_groups[name] = {
            "n_scored_transitions": n, "n_capability_cells": len(sub),
            "families": sorted({family_of(c) for c in sub}),
            "nll": A,
            "improvement_empirical_to_full": round(
                A["empirical_family"] - A["r_theta"], 6),
            "improvement_empirical_to_identity_only": round(
                A["empirical_family"] - A["empirical_family_learned_id"], 6)}

    db = out_groups["data_backed"]
    checks, ok = [], True
    for k, want in EXPECT.items():
        got = db["n_scored_transitions"] if k == "n" else (
            db["improvement_empirical_to_full"] if k.startswith("improvement") else db["nll"][k])
        good = (got == want) if k == "n" else abs(got - want) <= TOL
        ok &= good
        checks.append({"quantity": k, "expected": want, "got": got, "pass": bool(good)})

    payload = {
        "schema": "compose.editing_v2.reference_law_data_backed", "schema_version": 1,
        "status": "PAPER_BEARING_DERIVED" if ok else "MISMATCH",
        "derivation": ("entries-weighted aggregate of the per-capability-cell reference-law "
                       "NLLs, partitioned by the synthetic_only flag emitted in the family-lane "
                       "provenance artifact. No model evaluation; no family named in this script."),
        "inputs": {
            "scores_artifact": str(SCORES.relative_to(ROOT)),
            "scores_sha256": sha256_of(SCORES),
            "scores_status": S.get("status"),
            "provenance_artifact": str(PROV.relative_to(ROOT)),
            "provenance_sha256": sha256_of(PROV),
            "provenance_rule": P["rule"],
            "synthetic_lane": P["synthetic_lane"],
            "data_backed_lanes": P["data_backed_lanes"]},
        "coverage": {
            "entries_scored_total": S["entries_scored"],
            "entries_in_reported_cells": sum(v["entries"] for v in cells.values()),
            "minimum_cell": S["minimum_cell"],
            "note": "cells below minimum_cell are unreported, not unscored"},
        "groups": out_groups,
        "reproduction_checks": checks,
        "generator": "scripts/emit_reference_law_data_backed.py",
        "generator_git_commit": git_commit()}
    DEST.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"mask taken from : {PROV.name}  (sha256 {payload['inputs']['provenance_sha256'][:12]}...)")
    print(f"synthetic-only families (from artifact): "
          f"{sorted(f for f, s in fam_synth.items() if s)}")
    print()
    for name, g in out_groups.items():
        print(f"{name:<16} n={g['n_scored_transitions']:>5}  " +
              "  ".join(f"{a.split('_')[0][:5]}={g['nll'][a]:.4f}" for a in ARMS) +
              f"   emp->full {g['improvement_empirical_to_full']:+.4f}")
    print()
    for c in checks:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['quantity']:<34}"
              f"expected {c['expected']}  got {c['got']}")
    print(f"\nwrote {DEST.relative_to(ROOT)}   status={payload['status']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
