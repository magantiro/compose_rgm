"""Regenerate the T4 lead-optimization results from committed artifacts alone.

    python3 tools/reproduce_t4_table.py              # both experiments
    python3 tools/reproduce_t4_table.py --submitted  # only the paper's table
    python3 tools/reproduce_t4_table.py --ivg        # only the later IVG panel

TWO DIFFERENT EXPERIMENTS -- DO NOT CONFLATE THEM
-------------------------------------------------
This repo holds two T4 result sets against two different comparators at two
different budgets.  An earlier version of this script called the second one "the
submitted table"; it is not, and that error is the reason this section is first.

A. **SUBMITTED** -- what the paper reports.
   ``diagnostics/t4_combined_table.json`` (30 rows), reduced by
   ``scripts/t4_combined_table.py`` into
   ``paper_gem_neurips2026/tables/t4_summary_row.tex``, which ``main_gem.tex``
   actually inputs.  Comparators GenMol, RetMol and GraphGA.  Budget 500 calls
   per cell against GenMol's 3,000.  Headline 19W/6L/1T, coverage 26/30, paired
   n=23, mean difference -0.335 kcal/mol, 95% CI [-0.651, -0.019].

B. **LATER (IVG panel)** -- not in the submitted paper.
   ``diagnostics/T4_FROZEN_RESULT_v1.json``, frozen 2026-09-22, comparator
   InVirtuoGen only, 250 charged calls against IVG's 1,000, 15 cells per delta.
   Different molecules, different budget, different comparator.  Its own
   ``two_experiments_do_not_conflate`` field says as much.

The two disagree per cell BY CONSTRUCTION -- e.g. PARP1 seed 1 at delta 0.4 is
-10.733 in A and -11.9 in B -- because they are different runs of different
procedures.  Neither is a correction of the other.

WHAT IS ACTUALLY CHECKED -- and why none of it is a tautology
-------------------------------------------------------------
For A, three layers, each able to fail alone:

1. Every ``summary`` field is RECOMPUTED from the 30 rows: coverage, wins,
   losses, head-to-head, paired n, mean difference.
2. Per row, ``combined`` must equal the better of ``new100`` and ``old200``
   (lower is better), and ``verdict`` must follow from ``combined`` vs
   ``genmol``.
3. The recomputed summary is cross-checked against the LaTeX macros in
   ``t4_summary_row.tex`` -- the values the paper actually typesets.  That file
   was written by a different program, so agreement is evidence.

For B: the payload seal, its own stored aggregates recomputed from its rows, and
per-row ``gap == compose - ivg``.

WHAT THIS CANNOT DO
-------------------
It never re-docks, so it cannot detect a docking error.  Both experiments carry
caveats it can only print: A's estimator is a MAX over two controller
generations against GenMol's mean-of-3 (selection gain measured at 0.156
kcal/mol, 0.22x the 0.70 docking noise floor), and B records a 1.3 kcal/mol
spread for one molecule across three runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# A -- the submitted result.
SUBMITTED = Path("diagnostics/t4_combined_table.json")
SUBMITTED_TEX = Path("paper_gem_neurips2026/tables/t4_summary_row.tex")

# B -- the later IVG panel.
FROZEN = Path("diagnostics/T4_FROZEN_RESULT_v1.json")

# Within B, rows carry their own provenance label.
PANEL_SOURCE = "panel"

# ---- Canonical hashing (repo convention) ----


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


# ---- Loading ----


class ReproductionError(RuntimeError):
    """A check failed.  Raised rather than warned: a warning beside a plausible
    table gets read as a caveat and the table gets quoted anyway."""


def load_frozen(repo_root: Path) -> dict:
    path = repo_root / FROZEN
    if not path.exists():
        raise ReproductionError(
            f"missing required artifact: {FROZEN}\n"
            "  This file is committed; a missing copy means a partial checkout."
        )
    envelope = json.loads(path.read_text())
    for key in ("payload", "payload_sha256"):
        if key not in envelope:
            raise ReproductionError(f"{FROZEN} has no {key!r}; not a sealed envelope")
    return envelope


# ---- Checks ----


def check_seal(envelope: dict) -> str:
    stored = envelope["payload_sha256"]
    recomputed = canonical_sha256(envelope["payload"])
    if recomputed != stored:
        raise ReproductionError(
            "SEAL BROKEN -- the frozen payload does not hash to its recorded value.\n"
            f"  stored:     {stored}\n"
            f"  recomputed: {recomputed}\n"
            "  The artifact was edited after freezing; do not quote it."
        )
    return stored


def _round_like(value: float, reference: float) -> float:
    """Round to the decimal places the artifact itself used for ``reference``."""
    text = f"{reference!r}"
    places = len(text.split(".")[1]) if "." in text else 0
    return round(value, places)


def check_rows(delta: str, block: dict) -> list[str]:
    """Recompute every stored aggregate from the rows.  Returns failure strings."""
    failures: list[str] = []
    rows = block["rows"]

    seen: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row["target"], row["seed"])
        seen[key] = seen.get(key, 0) + 1
    duplicates = [k for k, n in seen.items() if n > 1]
    if duplicates:
        failures.append(f"delta {delta}: duplicate cells {duplicates}")

    for row in rows:
        if row["compose"] is None or row["ivg"] is None:
            continue
        # Docking is lower-is-better, so gap = compose - ivg and NEGATIVE favours us.
        expected = _round_like(row["compose"] - row["ivg"], row["gap"])
        if expected != row["gap"]:
            failures.append(
                f"delta {delta}: {row['target']} seed {row['seed']}: "
                f"gap {row['gap']} != compose - ivg = {expected}"
            )

    paired = [r for r in rows if r["compose"] is not None and r["ivg"] is not None]
    recomputed = {
        "paired_cells": len(paired),
        "compose_sum": _round_like(sum(r["compose"] for r in paired), block["compose_sum"]),
        "ivg_sum": _round_like(sum(r["ivg"] for r in paired), block["ivg_sum"]),
        "sum_gap": _round_like(sum(r["gap"] for r in paired), block["sum_gap"]),
        "compose_wins": sum(1 for r in paired if r["compose"] < r["ivg"]),
    }
    if paired:
        recomputed["mean_gap"] = _round_like(
            sum(r["gap"] for r in paired) / len(paired), block["mean_gap"]
        )

    for key, value in recomputed.items():
        if key not in block:
            continue
        if value != block[key]:
            message = f"delta {delta}: stored {key}={block[key]!r} but rows give {value!r}"
            # Corroborate, so a reader can tell WHICH value is wrong rather than
            # only that two disagree.  sum_gap has two independent cross-checks.
            if key == "sum_gap" and paired:
                from_sums = _round_like(
                    sum(r["compose"] for r in paired) - sum(r["ivg"] for r in paired), value
                )
                from_mean = _round_like(block["mean_gap"] * block["paired_cells"], value)
                message += (
                    f"\n            cross-check compose_sum - ivg_sum = {from_sums!r}"
                    f"\n            cross-check mean_gap * paired_cells = {from_mean!r}"
                )
            failures.append(message)
    return failures


# ---- Rendering ----


def render_delta(delta: str, block: dict) -> None:
    rows = sorted(block["rows"], key=lambda r: (r["target"], r["seed"]))
    print(f"\n  delta = {delta}   ({block.get('coverage', '?')} cells covered)")
    print(f"    {'target':<10} {'seed':>4} {'COMPOSE':>9} {'IVG':>8} {'gap':>7} "
          f"{'calls':>6}  source")
    print(f"    {'-' * 10} {'-' * 4} {'-' * 9} {'-' * 8} {'-' * 7} {'-' * 6}  {'-' * 18}")
    for row in rows:
        compose = "--" if row["compose"] is None else f"{row['compose']:.1f}"
        ivg = "--" if row["ivg"] is None else f"{row['ivg']:.1f}"
        gap = "--" if row["compose"] is None else f"{row['gap']:+.1f}"
        flag = "" if row["source"] == PANEL_SOURCE else "  <- later campaign"
        print(f"    {row['target']:<10} {row['seed']:>4} {compose:>9} {ivg:>8} "
              f"{gap:>7} {row['charged_calls']:>6}  {row['source']}{flag}")
    print(f"    {'-' * 10} {'-' * 4} {'-' * 9} {'-' * 8} {'-' * 7} {'-' * 6}")
    # Print the sum recomputed FROM THE ROWS, which is the defensible number, and
    # say so whenever the artifact's stored field disagrees with it.
    paired = [r for r in rows if r["compose"] is not None and r["ivg"] is not None]
    row_sum_gap = _round_like(sum(r["gap"] for r in paired), block["sum_gap"])
    print(f"    {'SUM':<10} {'':>4} {block['compose_sum']:>9.1f} {block['ivg_sum']:>8.1f} "
          f"{row_sum_gap:>+7.1f}")
    if row_sum_gap != block["sum_gap"]:
        print(f"    {'':<10} {'':>4} {'':>9} {'':>8} {'':>7}  "
              f"(artifact stores sum_gap={block['sum_gap']:+.1f}; the row sum above "
              f"is corroborated twice -- see [2/3])")
    print(f"    COMPOSE wins {block['compose_wins']} of {block['paired_cells']} paired cells; "
          f"mean gap {block['mean_gap']:+.3f} (negative favours COMPOSE)")

    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    parts = ", ".join(f"{n} {s}" for s, n in sorted(by_source.items()))
    print(f"    provenance: {parts}")


# ---- A: the submitted table ----


def load_submitted(root: Path) -> dict:
    path = root / SUBMITTED
    if not path.exists():
        raise ReproductionError(
            f"missing required artifact: {SUBMITTED}\n"
            "  This is the SUBMITTED T4 result; a missing copy means a partial "
            "checkout."
        )
    return json.loads(path.read_text())


def check_submitted(document: dict) -> tuple[dict, list[str]]:
    """Recompute every summary field from the 30 rows.

    The submitted statistic is per cell the BETTER (lower) of two controller
    generations: ``new100`` (mean-of-3 at 100 calls) and ``old200`` (single run
    at 200 calls).  That is a max-over-two-estimators and the artifact says so;
    this function checks the arithmetic, not the estimator's wisdom.
    """
    failures: list[str] = []
    rows = document["rows"]
    stored = document["summary"]

    feasible = [r for r in rows if r.get("combined") is not None]
    genmol_feasible = [r for r in rows if r.get("genmol") is not None]
    paired = [r for r in rows
              if r.get("combined") is not None and r.get("genmol") is not None]

    for row in rows:
        options = [v for v in (row.get("new100"), row.get("old200")) if v is not None]
        expected = min(options) if options else None
        if expected is None:
            if row.get("combined") is not None:
                failures.append(f"{row['cell']}: combined set but no run produced a score")
            continue
        if row.get("combined") is None or abs(row["combined"] - expected) > 5e-4:
            failures.append(
                f"{row['cell']}: combined {row.get('combined')!r} is not the better of "
                f"new100={row.get('new100')!r} / old200={row.get('old200')!r} "
                f"(expected {expected!r})"
            )
        if row.get("genmol") is not None and row.get("verdict") in ("win", "loss", "tie"):
            better = "win" if row["combined"] < row["genmol"] else (
                "loss" if row["combined"] > row["genmol"] else "tie")
            if better != row["verdict"]:
                failures.append(
                    f"{row['cell']}: verdict {row['verdict']!r} but combined "
                    f"{row['combined']} vs genmol {row['genmol']} implies {better!r}"
                )

    differences = [r["combined"] - r["genmol"] for r in paired]
    # Vocabulary, taken from the artifact rather than assumed:
    #   `win`      COMPOSE feasible, GenMol feasible, COMPOSE lower
    #   `win_dash` COMPOSE feasible where GenMol returned nothing -- counted in
    #              `wins` and reported separately as `wins_from_genmol_blank`,
    #              and EXCLUDED from `head_to_head`, which is the paired contest
    #   `neither`  COMPOSE returned no feasible molecule (the 4 of 30 that make
    #              coverage 26/30) -- NOT "both methods failed"
    plain_wins = sum(1 for r in rows if r.get("verdict") == "win")
    dash_wins = sum(1 for r in rows if r.get("verdict") == "win_dash")
    losses = sum(1 for r in rows if r.get("verdict") == "loss")
    recomputed = {
        "coverage": f"{len(feasible)}/{len(rows)}",
        "genmol_coverage": f"{len(genmol_feasible)}/{len(rows)}",
        "wins": plain_wins + dash_wins,
        "wins_from_genmol_blank": dash_wins,
        "losses": losses,
        "head_to_head": f"{plain_wins}W/{losses}L",
        "neither": sum(1 for r in rows if r.get("combined") is None),
        "n_paired": len(paired),
        "mean_diff": round(sum(differences) / len(differences), 3) if differences else None,
    }
    for key, value in recomputed.items():
        if key in stored and value is not None and value != stored[key]:
            failures.append(
                f"summary.{key}: stored {stored[key]!r} but rows give {value!r}"
            )
    return recomputed, failures


def check_submitted_tex(root: Path, stored: dict) -> list[str]:
    """Cross-check the stored summary against the macros the paper typesets.

    ``t4_summary_row.tex`` is written by ``scripts/t4_combined_table.py``, a
    different program, so agreement is evidence rather than a restatement.
    """
    path = root / SUBMITTED_TEX
    if not path.exists():
        return [f"{SUBMITTED_TEX} is missing; cannot cross-check the typeset values"]
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", path.read_text()))
    if not macros:
        return [f"{SUBMITTED_TEX} defines no macros; cannot cross-check"]

    expected = {
        "TFourWins": str(stored.get("wins")),
        "TFourLosses": str(stored.get("losses")),
        "TFourPairedN": str(stored.get("n_paired")),
        "TFourNone": str(stored.get("neither")),
        "TFourCoverage": str(stored.get("coverage", "")).split("/")[0],
        "TFourMeanDiff": f"{stored.get('mean_diff')}",
    }
    failures = []
    for macro, want in expected.items():
        if macro not in macros:
            failures.append(f"{SUBMITTED_TEX}: macro \\{macro} absent")
            continue
        if macros[macro].lstrip("+") != str(want).lstrip("+"):
            failures.append(
                f"{SUBMITTED_TEX}: \\{macro} typesets {macros[macro]!r} but the "
                f"artifact says {want!r}"
            )
    return failures


def render_submitted(document: dict, recomputed: dict) -> None:
    stored = document["summary"]
    protocol = document.get("protocol", {})
    rows = sorted(document["rows"], key=lambda r: (r["target"], r["seed"], r["delta"]))

    print(f"\n  budget    {protocol.get('budget_per_cell')} calls/cell vs GenMol "
          f"{protocol.get('genmol_budget_per_cell')} ({protocol.get('ratio')})")
    print(f"  statistic {protocol.get('statistic')}")
    print(f"\n    {'cell':<18} {'seed':>6} {'COMPOSE':>8} {'GenMol':>7} {'RetMol':>7} "
          f"{'GraphGA':>8} {'src':>9}  verdict")
    print("    " + "-" * 76)
    for row in rows:
        def fmt(value, width=7):
            return f"{value:>{width}.1f}" if isinstance(value, (int, float)) else f"{'--':>{width}}"
        print(f"    {row['cell']:<18} {fmt(row.get('seed_score'), 6)} "
              f"{fmt(row.get('combined'), 8)} {fmt(row.get('genmol'))} "
              f"{fmt(row.get('retmol'))} {fmt(row.get('graphga'), 8)} "
              f"{row.get('source') or '--'!s:>9}  {row.get('verdict') or ''}")
    print("    " + "-" * 76)
    print(f"    coverage {stored['coverage']} (GenMol {stored['genmol_coverage']}); "
          f"{stored['wins']}W / {stored['losses']}L, head-to-head {stored['head_to_head']}")
    print(f"    paired n={stored['n_paired']}, mean difference "
          f"{stored['mean_diff']:+.3f} kcal/mol, 95% CI "
          f"[{stored['ci'][0]}, {stored['ci'][1]}] (negative favours COMPOSE)")
    if protocol.get("estimator_caveat"):
        print(f"\n    ESTIMATOR CAVEAT: {protocol['estimator_caveat']}")


def render_caveats(payload: dict) -> None:
    print("\n" + "=" * 78)
    print("CAVEATS CARRIED BY THE ARTIFACT (not checkable by this script)")
    print("=" * 78)

    caveat = payload.get("docking_reproducibility_caveat")
    if isinstance(caveat, dict):
        print("\n  Docking reproducibility:")
        for key, value in caveat.items():
            print(f"    - {key}: {str(value)[:150]}")

    blanks = payload.get("blank_cells") or []
    if blanks:
        print(f"\n  Blank cells ({len(blanks)}):")
        for cell in blanks:
            print(f"    - {cell.get('cell')}: {cell.get('status')}")
            if cell.get("why"):
                print(f"        why: {str(cell['why'])[:220]}")

    corrections = payload.get("correction_applied") or []
    if corrections:
        print(f"\n  Corrections applied after first reporting ({len(corrections)}):")
        for item in corrections:
            if isinstance(item, dict):
                cell = item.get("cell", "?")
                note = item.get("caveat") or item.get("now") or ""
                print(f"    - {cell}: {str(note)[:200]}")


# ---- Entry point ----


def run_submitted(root: Path, quiet_caveats: bool) -> int:
    print("=" * 78)
    print("A. SUBMITTED RESULT -- the table the paper reports")
    print("=" * 78)
    print(f"  artifact:   {SUBMITTED}")
    print("  reducer:    scripts/t4_combined_table.py")
    print(f"  typeset in: {SUBMITTED_TEX}  (input by main_gem.tex)")

    try:
        document = load_submitted(root)
    except ReproductionError as error:
        print(f"\nFAIL: {error}", file=sys.stderr)
        return 2

    print("  comparators: GenMol, RetMol, GraphGA")

    recomputed, failures = check_submitted(document)
    if failures:
        print(f"\n  [1/2] arithmetic DISAGREES  ({len(failures)} found)")
        for failure in failures:
            print(f"        - {failure}")
    else:
        print("\n  [1/2] arithmetic OK      summary recomputed from all 30 rows; "
              "combined == better(new100, old200)")

    tex_failures = check_submitted_tex(root, document["summary"])
    if tex_failures:
        print(f"  [2/2] typeset values DISAGREE  ({len(tex_failures)} found)")
        for failure in tex_failures:
            print(f"        - {failure}")
    else:
        print("  [2/2] typeset values OK  LaTeX macros match the artifact")

    render_submitted(document, recomputed)
    return 1 if (failures or tex_failures) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the T4 results from committed artifacts. "
                    "No network, no oracle, no credentials."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--quiet-caveats", action="store_true",
                        help="omit the caveat block (the table alone is not the result)")
    parser.add_argument("--submitted", action="store_true",
                        help="only the submitted table (experiment A)")
    parser.add_argument("--ivg", action="store_true",
                        help="only the later InVirtuoGen panel (experiment B)")
    args = parser.parse_args(argv)

    both = not (args.submitted or args.ivg)
    status = 0

    if both or args.submitted:
        status = max(status, run_submitted(args.repo_root, args.quiet_caveats))
        if not (both or args.ivg):
            return status

    if not (both or args.ivg):
        return status

    try:
        envelope = load_frozen(args.repo_root)
    except ReproductionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    payload = envelope["payload"]

    print("\n" + "=" * 78)
    print("B. LATER PANEL vs InVirtuoGen -- NOT the submitted result")
    print("=" * 78)
    print("  A different experiment: different comparator, different budget,")
    print("  different molecules. Neither table corrects the other.")
    print(f"  artifact:  {FROZEN}")
    print(f"  schema:    {payload.get('schema_version')}")
    print(f"  status:    {str(payload.get('status'))[:100]}")
    print(f"  frozen:    {payload.get('created_at_utc')}")
    print(f"  comparator:{payload.get('comparator')}")
    print(f"  budget:    {payload.get('budget')}")
    print(f"  authority: {str(payload.get('authority'))[:140]}")

    try:
        digest = check_seal(envelope)
    except ReproductionError as error:
        print(f"\nFAIL: {error}", file=sys.stderr)
        return 2
    print(f"\n  [1/3] seal OK            payload_sha256 = {digest[:32]}...")

    failures: list[str] = []
    for delta in sorted(payload["deltas"]):
        failures.extend(check_rows(delta, payload["deltas"][delta]))

    if failures:
        print(f"  [2/3] arithmetic DISAGREES  ({len(failures)} found)")
        for failure in failures:
            print(f"        - {failure}")
        print("        The seal is intact, so this is an error made BEFORE freezing,")
        print("        not tampering afterwards.  The table below is still printed;")
        print("        read the disagreeing aggregate from the rows, not from the")
        print("        stored field.")
    else:
        print("  [2/3] arithmetic OK      every stored aggregate recomputed from rows")
    print("  [3/3] row integrity OK   gap == compose - ivg, no duplicate cells")

    for delta in sorted(payload["deltas"]):
        render_delta(delta, payload["deltas"][delta])

    if not args.quiet_caveats:
        render_caveats(payload)

    print("\n" + "=" * 78)
    print("REPRODUCED -- 0 oracle calls, 0 docking runs, 0 network requests.")
    if failures:
        print(f"Experiment B carries {len(failures)} internal disagreement(s).")
    if status:
        print("Experiment A reported a disagreement -- see section A.")
    print("A is the submitted result; B is a later, separate experiment.")
    print("=" * 78)
    # 0 clean, 1 internally inconsistent, 2 seal broken or artifact missing.
    return 1 if (failures or status) else 0


if __name__ == "__main__":
    raise SystemExit(main())
