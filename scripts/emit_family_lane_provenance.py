"""Emit the family x supervision-lane provenance artifact for the editing corpus.

The information already exists in the frozen Active8 run: each task directory
carries a RECEIPT.json naming its ``data_lane`` and ``partition_role``, and a
TASK_SUMMARY.json carrying the per-family and per-capability-cell transition
histograms.  This script joins them so that the synthetic/data-backed split used
by the reference-law analysis is reproducible from artifacts rather than from a
prose ledger entry.

Classification rule, applied mechanically:

    synthetic_only(F)  <=>  n_F(reversible_synthetic_walk) > 0
                            AND  sum over data-backed lanes of n_F(lane) == 0
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_LANE = "reversible_synthetic_walk"
#: Public (paper-facing) operator name for each model family that differs.
PUBLIC_NAME = {"cycle_insert": "cycle_close",
               "cycle_attach": "cycle_open",
               "ring_system_restate": "ring_aromaticity_restate"}
#: The S16 census this artifact must reproduce, family -> (transitions, chunks).
S16_EXPECTED = {"cycle_insert": (122_183, 102),
                "cycle_attach": (84_940, 102),
                "ring_system_restate": (16_380, 29)}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    tasks = sorted(ROOT.glob("local_runtime/active8/*/tasks/*"))
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_cell: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    chunks_per_family: dict[str, set[str]] = defaultdict(set)
    lanes: set[str] = set()
    summary_hashes: list[str] = []
    runs: set[str] = set()
    n_train = 0

    for task in tasks:
        receipt, summary = task / "RECEIPT.json", task / "TASK_SUMMARY.json"
        if not (receipt.exists() and summary.exists()):
            continue
        r = json.loads(receipt.read_text())
        if r.get("partition_role") != "train":
            continue                      # the split is what defines training supervision
        n_train += 1
        lane = r["data_lane"]
        lanes.add(lane)
        s = json.loads(summary.read_text())
        runs.add(s.get("run_identity_sha256", ""))
        summary_hashes.append(s.get("summary_sha256", ""))
        for fam, n in (s.get("action_family_histogram") or {}).items():
            by_family[fam][lane] += int(n)
            chunks_per_family[fam].add(task.name)
        for cell, n in (s.get("capability_cell_histogram") or {}).items():
            by_cell[cell][lane] += int(n)

    data_backed = sorted(lanes - {SYNTHETIC_LANE})
    families = {}
    for fam, counts in sorted(by_family.items()):
        synth = counts.get(SYNTHETIC_LANE, 0)
        backed = sum(counts.get(l, 0) for l in data_backed)
        families[fam] = {
            "public_name": PUBLIC_NAME.get(fam, fam),
            "by_lane": dict(sorted(counts.items())),
            "transitions_total": synth + backed,
            "transitions_synthetic": synth,
            "transitions_data_backed": backed,
            "chunks": len(chunks_per_family[fam]),
            "synthetic_only": bool(synth > 0 and backed == 0),
        }
    cells = {}
    for cell, counts in sorted(by_cell.items()):
        fam = cell.split(":")[1] if ":" in cell else cell
        cells[cell] = {"family": fam, "by_lane": dict(sorted(counts.items())),
                       "synthetic_only": families.get(fam, {}).get("synthetic_only")}

    # ---- reconciliation against the previously recorded census ---------------
    # The classification (synthetic-only vs data-backed) is the load-bearing
    # output and must reconcile exactly.  Transition COUNTS may differ from the
    # recorded census by the Active8 exclusion boundary: the summary histograms
    # count teacher actions before per-entry exclusions are applied, so a
    # post-exclusion census can be smaller.  A count difference is recorded, not
    # silently reconciled, and never changes a family's classification.
    problems, count_notes = [], []
    for fam, (n_exp, c_exp) in S16_EXPECTED.items():
        got = families.get(fam)
        if got is None:
            problems.append(f"{fam}: absent from the scan")
            continue
        if got["transitions_total"] != n_exp:
            count_notes.append(
                f"{fam}: histogram {got['transitions_total']:,} vs recorded census "
                f"{n_exp:,} (delta {got['transitions_total']-n_exp:+,}); classification unaffected")
        if got["chunks"] != c_exp:
            problems.append(f"{fam}: chunks {got['chunks']} != {c_exp}")
        if not got["synthetic_only"]:
            problems.append(f"{fam}: expected synthetic-only, got data-backed {got['transitions_data_backed']:,}")
    for fam, rec in families.items():
        if fam not in S16_EXPECTED and rec["synthetic_only"]:
            problems.append(f"{fam}: unexpectedly synthetic-only")

    out = {
        "schema": "compose.editing_v2.family_lane_provenance",
        "schema_version": 1,
        "status": "PROVENANCE_ARTIFACT" if not problems else "MISMATCH_WITH_RECORDED_CENSUS",
        "rule": ("synthetic_only(F) iff n_F(reversible_synthetic_walk) > 0 and "
                 "sum over data-backed lanes of n_F(lane) == 0"),
        "partition_role_scanned": "train",
        "train_chunks_scanned": n_train,
        "lanes_observed": sorted(lanes),
        "data_backed_lanes": data_backed,
        "synthetic_lane": SYNTHETIC_LANE,
        "active8_run_identity_sha256": sorted(x for x in runs if x),
        "input_summary_sha256_digest": hashlib.sha256(
            "".join(sorted(summary_hashes)).encode()).hexdigest(),
        "n_input_summaries": len(summary_hashes),
        "generator": "scripts/emit_family_lane_provenance.py",
        "generator_git_commit": git_commit(),
        "families": families,
        "capability_cells": cells,
        "reconciliation_problems": problems,
        "count_reconciliation_notes": count_notes,
        "recorded_census_compared_against": {k: {"transitions": v[0], "chunks": v[1]}
                                             for k, v in S16_EXPECTED.items()},
        "counting_boundary": ("summary histograms count teacher actions before "
                              "per-entry Active8 exclusions; a post-exclusion "
                              "census can therefore be smaller"),
    }
    dest = ROOT / "diagnostics" / "editing_v2_family_lane_provenance.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")

    print(f"train chunks scanned : {n_train}")
    print(f"lanes                : synthetic={SYNTHETIC_LANE}; data-backed={data_backed}")
    print(f"{'family':<24}{'total':>10}{'synthetic':>11}{'data-backed':>13}{'chunks':>8}  synthetic-only")
    for fam, rec in families.items():
        print(f"  {fam:<22}{rec['transitions_total']:>10,}{rec['transitions_synthetic']:>11,}"
              f"{rec['transitions_data_backed']:>13,}{rec['chunks']:>8}  {rec['synthetic_only']}")
    print()
    print("classification reconciliation:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("   ", p)
    if count_notes:
        print("count notes (do not affect classification):")
        for n in count_notes:
            print("   ", n)
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
