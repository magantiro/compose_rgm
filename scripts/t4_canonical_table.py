"""Render the canonical T4 table and the A/B/C ablation from the reconciliation.

Reads `diagnostics/t4_canonical_shared_controller_v1/reconciliation_v1.json` (built
from ROUND LOCKS by `scripts/t4_canonical_reconcile.py`) and writes the markdown
report. It adds the published InVirtuoGen column read-only, and prints the
cross-run docking caveat beside it every time, because `obabel --gen3D` is unseeded
and one T4 seed molecule has been measured at -7.5 / -8.30 / -8.8 across three runs:
per-cell margins against a different run are inside that noise, the arm-vs-arm
comparisons in this table are not, because they were computed in the same run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DIAG = ROOT / "diagnostics/t4_canonical_shared_controller_v1"
BASELINE = ROOT / "configs/t4_published_invirtuogen_baseline.json"

CAVEAT = (
    "Docking is not reproducible across runs: qvina02 is seeded, `obabel --gen3D` is "
    "not, and one T4 seed molecule has been measured at -7.5 / -8.30 / -8.8 across "
    "three runs. Every arm-vs-arm comparison below is WITHIN this run and is sound. "
    "The InVirtuoGen column is read-only context from a different run and a different "
    "docking invocation; per-cell margins against it are inside the noise, and only "
    "the aggregate is worth reading."
)


def _ivg() -> dict:
    payload = json.loads(BASELINE.read_text())
    payload = payload.get("payload", payload)
    table = {}
    for key, delta in (("delta_0_6", 0.6), ("delta_0_4", 0.4)):
        for target, values in (payload.get(key) or {}).items():
            for index, value in enumerate(values):
                table[(target, index, delta)] = value
    return table


def _cell_index(cell: str) -> int:
    return int(cell.split("_")[-2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconciliation", default=str(DIAG / "reconciliation_v1.json"))
    parser.add_argument("--out", default=str(DIAG / "T4_CANONICAL_RESULT_v1.md"))
    args = parser.parse_args()

    report = json.loads(Path(args.reconciliation).read_text())
    ivg = _ivg()
    rows = report["cells"]
    lines: list[str] = []
    add = lines.append

    add("# T4 canonical shared controller — result\n")
    add(f"Run id: `{report['run_id']}`  ")
    add(f"Controller identity: `{report['controller_identity_sha256']}`  ")
    add(f"Reconciled charged calls: **{report['total_reconciled_charged_calls']}** "
        "(from immutable round locks, not checkpoints)\n")
    add(f"> {CAVEAT}\n")

    add("## Arm summary\n")
    add("| arm | cells | complete | exhausted | charged calls | scored | sum best | "
        "expansion rounds | invalid-chemistry oracle calls |")
    add("|---|---|---|---|---|---|---|---|---|")
    for arm in sorted(report["by_arm"]):
        bucket = report["by_arm"][arm]
        add(f"| {arm} | {bucket['cells']} | {bucket['final']} | {bucket['exhausted']} | "
            f"{bucket['charged']} | {bucket['scored_cells']} | "
            f"{bucket['best_sum']:.1f} | {bucket['expansion_rounds']} | "
            f"{bucket['invalid_oracle_calls']} |")
    add("")

    # ---- The canonical table: arm C, all 30 cells ----
    add("## Canonical controller (arm C), all 30 cells\n")
    add("A best in (parentheses) is a RUNNING cell's current incumbent from its "
        "checkpoint, not a final result.\n")
    add("| target | seed | delta | status | best | root | IVG (other run) | calls | "
        "rounds | expansions | eligible/1k draws |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    c_rows = sorted(
        (row for row in rows if row["arm"] == "C"),
        key=lambda row: (-row["delta"], row["target"], row["cell"]),
    )
    for row in c_rows:
        index = _cell_index(row["cell"])
        reference = ivg.get((row["target"], index, row["delta"]))
        best = row.get("final_best")
        root = row.get("root_score")
        # A running cell's incumbent is shown in parentheses so a partial table cannot be
        # misread as a final one, and is never merged into `final_best`.
        provisional = best is None and row.get("checkpoint_best") is not None
        if provisional:
            best = row["checkpoint_best"]
        best_text = "" if best is None else (f"({best:.1f})" if provisional else f"{best:.1f}")
        root_text = "" if root is None else f"{root:.1f}"
        reference_text = "" if reference is None else f"{reference:.1f}"
        add(
            f"| {row['target'].upper()} | {index + 1} | {row['delta']} | "
            f"{row.get('status', '?')} | {best_text} | {root_text} | "
            f"{reference_text} | "
            f"{row.get('reconciled_charged_calls', 0)} | "
            f"{row.get('rounds') or row.get('checkpoint_rounds', 0)} | "
            f"{row.get('expansion_triggered_rounds', 0)} | "
            f"{row.get('eligible_per_1000_draws', '')} |"
        )
    add("")

    # ---- The ablation at delta 0.6 ----
    add("## A/B/C ablation at delta = 0.6 (same cells, same seeds, same run)\n")
    add("A vs B isolates coordinated structural programs. B vs C isolates adaptive "
        "use of structural support.\n")
    add("| target | seed | A local | B coordinated | C adaptive | B-A | C-B | "
        "C expansions |")
    add("|---|---|---|---|---|---|---|---|")
    by_key = {(row["arm"], row["cell"]): row for row in rows}
    cells06 = sorted(
        {row["cell"] for row in rows if row["arm"] == "A"},
        key=lambda cell: (cell.split("_")[0], cell),
    )
    deltas_ba: list[float] = []
    deltas_cb: list[float] = []
    for cell in cells06:
        values = {}
        for arm in ("A", "B", "C"):
            row = by_key.get((arm, cell)) or {}
            values[arm] = row.get("final_best")
        target = cell.split("_")[0]
        index = _cell_index(cell)
        ba = (values["B"] - values["A"]) if None not in (values["A"], values["B"]) else None
        cb = (values["C"] - values["B"]) if None not in (values["B"], values["C"]) else None
        if ba is not None:
            deltas_ba.append(ba)
        if cb is not None:
            deltas_cb.append(cb)
        expansions = (by_key.get(("C", cell)) or {}).get("expansion_triggered_rounds", 0)
        scores = " | ".join(
            "" if values[arm] is None else f"{values[arm]:.1f}" for arm in ("A", "B", "C")
        )
        ba_text = "" if ba is None else f"{ba:+.1f}"
        cb_text = "" if cb is None else f"{cb:+.1f}"
        add(f"| {target.upper()} | {index + 1} | {scores} | {ba_text} | {cb_text} "
            f"| {expansions} |")
    if deltas_ba and deltas_cb:
        mean_ba = sum(deltas_ba) / len(deltas_ba)
        mean_cb = sum(deltas_cb) / len(deltas_cb)
        add(f"| **mean** | | | | | **{mean_ba:+.2f}** | **{mean_cb:+.2f}** | |")
    add("")
    add("Negative is better (docking score). A negative B-A means the coordinated "
        "lanes helped; a negative C-B means the adaptive rule helped.\n")

    # ---- Mechanism telemetry ----
    add("## Mechanism telemetry (per arm, summed over cells)\n")
    add("| arm | proposal draws | eligible pool rows | eligible/1k draws | "
        "mean depth | max regions | atoms created | atoms deleted | ring-family "
        "candidates | worker failures |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ("A", "B", "C"):
        subset = [row for row in rows if row["arm"] == arm and row.get("locks")]
        if not subset:
            continue
        draws = sum(row.get("proposal_draws", 0) for row in subset)
        pool = sum(row.get("eligible_pool_rows", 0) for row in subset)
        depths = [row["mean_program_depth"] for row in subset if row.get("mean_program_depth")]
        add(
            f"| {arm} | {draws} | {pool} | "
            f"{round(1000.0 * pool / draws, 2) if draws else ''} | "
            f"{round(sum(depths) / len(depths), 2) if depths else ''} | "
            f"{max((row.get('max_regions_per_program') or 0) for row in subset)} | "
            f"{sum(row.get('atoms_created', 0) for row in subset)} | "
            f"{sum(row.get('atoms_deleted', 0) for row in subset)} | "
            f"{sum(row.get('ring_family_candidates', 0) for row in subset)} | "
            f"{sum(row.get('proposal_worker_failures', 0) for row in subset)} |"
        )
    add("")

    add("## Frontier-improvement attribution (synthesis-time lane tag)\n")
    add("| arm | lane | improvements |")
    add("|---|---|---|")
    for arm in ("A", "B", "C"):
        totals: dict[str, int] = {}
        for row in rows:
            if row["arm"] != arm:
                continue
            for lane, count in (row.get("frontier_improvement_attribution") or {}).items():
                totals[lane] = totals.get(lane, 0) + count
        for lane, count in sorted(totals.items(), key=lambda item: -item[1]):
            add(f"| {arm} | {lane} | {count} |")
    add("")

    # ---- The built-in replicate control ----
    # Arms B and C declare the SAME expert vocabulary and the SAME seed, and differ only
    # by the adaptive rule. On a delta=0.6 cell where that rule never fired, they
    # therefore ran an identical algorithm from an identical stream -- VERIFIED on the
    # locks: identical parents, identical candidate pools, identical query batches in
    # identical order. Any difference in their result is produced entirely by the
    # oracle, because `obabel --gen3D` is unseeded.
    #
    # So the run contains free replicate pairs, and their spread is the noise floor
    # below which NO per-cell margin in this table is interpretable.
    add("## Built-in replicate control: the noise floor of this pipeline\n")
    add("Arms B and C share a vocabulary and a seed and differ only by the adaptive "
        "rule. On a cell where that rule NEVER FIRED they ran an identical algorithm "
        "from an identical stream, so any difference between them is produced by the "
        "oracle alone (`obabel --gen3D` is unseeded). These pairs are a free replicate "
        "experiment, and their spread bounds what a per-cell margin can mean.\n")
    add("| cell | B best | C best | \\|B-C\\| | C expansions |")
    add("|---|---|---|---|---|")
    spreads = []
    for cell in cells06:
        b_row, c_row = by_key.get(("B", cell)) or {}, by_key.get(("C", cell)) or {}
        if c_row.get("expansion_triggered_rounds"):
            continue
        b_best, c_best = b_row.get("final_best"), c_row.get("final_best")
        if b_best is None or c_best is None:
            continue
        gap = abs(b_best - c_best)
        spreads.append(gap)
        add(f"| `{cell}` | {b_best:.1f} | {c_best:.1f} | {gap:.1f} | "
            f"{c_row.get('expansion_triggered_rounds', 0)} |")
    if spreads:
        spreads_sorted = sorted(spreads)
        median = spreads_sorted[len(spreads_sorted) // 2]
        add(f"| **n={len(spreads)}** | | | **median {median:.1f}, max "
            f"{max(spreads):.1f}** | |")
        add("")
        add(f"**Read every per-cell margin in this table against a noise floor of about "
            f"{median:.1f} kcal/mol.** Differences below it are not evidence of anything. "
            "The comparisons that ARE sound are the structural ones -- whether a cell "
            "completed its budget or exhausted, and how many calls it spent -- and the "
            "aggregate over cells.\n")
    else:
        add("")
        add("*No replicate pair is available yet: every completed B/C delta=0.6 pair "
            "either has not finished in both arms or had the expansion fire in C.*\n")

    add("## Expansion trigger census (arm C)\n")
    stops: dict[str, int] = {}
    triggered = added = 0
    for row in rows:
        if row["arm"] != "C":
            continue
        triggered += row.get("expansion_triggered_rounds", 0)
        added += row.get("expansion_endpoints_added", 0)
        for reason, count in (row.get("expansion_stop_reasons") or {}).items():
            stops[reason] = stops.get(reason, 0) + count
    add(f"- rounds that triggered the expansion: **{triggered}**")
    add(f"- distinct eligible endpoints the expansion added: **{added}**")
    add(f"- stop reasons: `{json.dumps(stops, sort_keys=True)}`")
    add("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
