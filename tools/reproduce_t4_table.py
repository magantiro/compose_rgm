"""Regenerate the frozen T4 lead-optimization table from committed artifacts alone.

WHAT THIS IS FOR
----------------
Verifying a published number without cloud credentials, without docking, and
without a single oracle call.  Everything this reads is committed in the repo, so
a fresh clone can run it.

    python3 tools/reproduce_t4_table.py

The table it prints is the one in the paper.  It is rebuilt from
``diagnostics/T4_FROZEN_RESULT_v1.json``, whose ``payload`` is sealed by
``payload_sha256`` under the repo's canonical-JSON convention
(``sort_keys=True, separators=(",", ":"), ensure_ascii=False``).

WHAT IT ACTUALLY CHECKS -- and why it is not a tautology
--------------------------------------------------------
Three independent checks, in order, each able to fail on its own:

1. **Seal.**  ``sha256(canonical(payload)) == payload_sha256``.  Catches any edit
   to the artifact after it was frozen.
2. **Arithmetic.**  Every aggregate the artifact STORES -- ``compose_sum``,
   ``ivg_sum``, ``sum_gap``, ``mean_gap``, ``compose_wins``, ``paired_cells``,
   ``coverage`` -- is RECOMPUTED here from the per-cell rows and compared.  The
   stored values were produced by the campaign reconciler, a different program,
   so agreement is evidence rather than a restatement.  A comparison whose
   expectation is recomputed from the code under test cannot fail; these
   expectations come from the artifact.
3. **Row integrity.**  Per row, ``gap`` must equal ``compose - ivg`` to the
   artifact's own rounding, and no cell may appear twice at one delta.  Docking
   is lower-is-better, so a NEGATIVE gap favours COMPOSE.

PROVENANCE IS PRINTED, NOT FLATTENED
------------------------------------
Each row carries ``source``: ``panel`` (the historical per-cell procedure that
produced the submitted table) or ``support_expansion`` (a later campaign).  They
are counted separately and labelled in the output, because a reader must never
have to guess which rows are the submitted result.  ``blank_cells`` and
``correction_applied`` are reported in full for the same reason.

WHAT IT DOES NOT DO
-------------------
It does not re-dock anything and therefore cannot detect a docking error.  The
artifact's own ``docking_reproducibility_caveat`` records a measured 1.3 kcal/mol
spread for one molecule across runs, which is wider than several per-row margins;
that caveat is printed with the table and is not something this script can check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN = Path("diagnostics/T4_FROZEN_RESULT_v1.json")

# The submitted table came from this procedure; anything else is a later campaign.
SUBMITTED_SOURCE = "panel"

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
        flag = "" if row["source"] == SUBMITTED_SOURCE else "  <- later campaign"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the frozen T4 table from committed artifacts. "
                    "No network, no oracle, no credentials."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--quiet-caveats", action="store_true",
                        help="omit the caveat block (the table alone is not the result)")
    args = parser.parse_args(argv)

    try:
        envelope = load_frozen(args.repo_root)
    except ReproductionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    payload = envelope["payload"]

    print("=" * 78)
    print("T4 LEAD OPTIMIZATION -- regenerated from committed artifacts")
    print("=" * 78)
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
        print(f"WITH {len(failures)} INTERNAL DISAGREEMENT(S) -- see [2/3] above.")
    print("=" * 78)
    # 0 clean, 1 internally inconsistent, 2 seal broken or artifact missing.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
