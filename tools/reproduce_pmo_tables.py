"""Recompute PMO development scores from committed oracle ledgers alone.

WHAT THIS IS FOR
----------------
Re-deriving every PMO number from the charged-call receipts, without cloud
credentials, without PyTDC, and without a single new oracle call.

    PYTHONPATH=src python3 tools/reproduce_pmo_tables.py
    PYTHONPATH=src python3 tools/reproduce_pmo_tables.py --task celecoxib_rediscovery

PMO HAS NO FROZEN PAPER TABLE
-----------------------------
Unlike T4 (``tools/reproduce_t4_table.py``, sealed frozen artifact), PMO is
development evidence.  Nothing here is a submitted result, and this script says
so on every run.  What it gives you is the exact arithmetic behind each
development claim, recomputed from the receipts rather than restated from a
summary someone wrote afterwards.

WHAT A LEDGER IS
----------------
A directory carrying ``oracle/manifest.json`` (task name, budget, protocol) and
``oracle/query_NNNNNN/result.json``, one receipt per CHARGED call::

    {"index": 5, "endpoint": "<SMILES>", "score": 0.189..., "role": "initialization",
     "status": "complete", "receipt_id": "...", "lock_id": "..."}

Receipts are the accounting record: an initialization molecule costs a call just
as a proposal does, and both count against the budget.

WHAT IS RECOMPUTED
------------------
``best``, the top-ten mean, and the AUC -- the last through the PRODUCTION
``compose_v4.control.program_task.pmo_top_ten_auc``, not a local transcription,
so this cannot drift from what the campaign scored.  A self-test runs first: a
constant sequence ``c`` must give ``c * (1 - frequency / (2 * budget))``, which
is the closed form of the trapezoid rule from (0, 0).  If that fails the
environment is wrong and no number below is trustworthy.

THE AUC TRAP THIS SCRIPT REFUSES TO LET YOU FALL INTO
-----------------------------------------------------
``top_auc`` trapezoids up from (0, 0), so a short-budget AUC is structurally
DEPRESSED: a run holding a constant top-ten level ``c`` scores
``c * (1 - frequency / (2 * budget))``.  At the official 10,000-call budget that
removes 0.5%; at 1,000 it removes 5%; **at 250 it removes 20%**.  Every AUC
printed below its official budget is therefore labelled, and the script states
the depression explicitly.  Never scale one of these and compare it to a
published 10,000-call figure.

OPTIONAL CHEMISTRY
------------------
With RDKit importable, a manifold column (median QED, median SA, % on-manifold)
is added.  Without it the score table is unchanged and the column is reported as
UNAVAILABLE rather than silently omitted.  This matters because three PMO
oracles (gsk3b, jnk3, drd2) are ML predictors that reward leaving the drug-like
manifold, so a headline score on those tasks must be read beside its chemistry.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS = "diagnostics"

# The official PMO budget every published comparison uses.
OFFICIAL_BUDGET = 10_000
AUC_FREQUENCY = 100

# Oracles that are ML predictors over fingerprints, where a high score is
# reachable by leaving the drug-like manifold.  Measured: r(score, QED) = -0.705
# on gsk3b and -0.750 on jnk3.
PREDICTOR_ORACLES = {"gsk3b", "jnk3", "drd2"}

# On-manifold = plausible chemistry.  Applied ONLY to READ a result; PMO is
# no-prescreen and nothing is filtered during search.
QED_FLOOR = 0.5
SA_CEILING = 5.0


class ReproductionError(RuntimeError):
    """A check failed.  Raised rather than warned."""


# ---- Production AUC, imported not transcribed ----


def load_auc():
    try:
        from compose_v4.control.program_task import pmo_top_ten_auc
    except ImportError as error:
        raise ReproductionError(
            f"cannot import the production AUC: {error}\n"
            "  Run with PYTHONPATH=src (numpy is required)."
        ) from error
    return pmo_top_ten_auc


def self_test(auc) -> None:
    """A constant sequence has a closed form.  If this fails, stop."""
    for level, budget in ((0.5, 250), (0.25, 1000)):
        expected = level * (1 - AUC_FREQUENCY / (2 * budget))
        got = auc([level] * budget, budget=budget, frequency=AUC_FREQUENCY)
        if abs(got - expected) > 1e-9:
            raise ReproductionError(
                "AUC self-test FAILED -- the environment does not reproduce the "
                f"closed form.\n  level={level} budget={budget} "
                f"expected={expected!r} got={got!r}"
            )


# ---- Ledger discovery ----


def find_ledgers(root: Path) -> list[Path]:
    """Any directory with an oracle manifest and at least one receipt.

    Scans the filesystem rather than git, so a fresh clone naturally sees only
    committed ledgers and a working tree sees whatever is present locally.
    """
    found = []
    for manifest in sorted((root / DIAGNOSTICS).rglob("oracle/manifest.json")):
        ledger = manifest.parent.parent
        if any(manifest.parent.glob("query_*/result.json")):
            found.append(ledger)
    return found


def tracked_prefixes(root: Path) -> set[str] | None:
    """Directories git tracks under diagnostics/.

    A LOCAL run sees ledgers a fresh clone will not -- several campaign
    workspaces are deliberately gitignored.  Without this the script would
    report numbers nobody else can reproduce, silently.  ``None`` means the
    question could not be answered (no git, or not a repository).
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", DIAGNOSTICS],
            capture_output=True, check=True, timeout=120,
        ).stdout.decode()
    except (OSError, subprocess.SubprocessError):
        return None
    prefixes: set[str] = set()
    for entry in out.split("\0"):
        if not entry:
            continue
        parts = entry.split("/")
        for depth in range(1, len(parts)):
            prefixes.add("/".join(parts[:depth]))
    return prefixes


def zero_charged_declaration(ledger: Path, root: Path) -> str | None:
    """Does a committed report near this ledger declare it charged NOTHING?

    A zero-oracle integration gate produces a ledger that is shaped exactly like
    a scored run -- same schema, same task name, same oracle_protocol -- while
    its scores come from a non-production scorer.  One such workspace here holds
    a celecoxib 'best' of 0.9167 against a real-run best of ~0.25.  Dropped into
    a score table that reads as a breakthrough.

    Detected from EVIDENCE (a sibling or parent report that declares
    ``new_charged_oracle_calls == 0`` or a zero-charged ``evidence_role``),
    never from the directory being called 'dryrun' -- a naming convention is not
    a guarantee, and the next such workspace may be named anything.
    """
    current = ledger
    for _ in range(3):
        if current == root or not current.parent.is_relative_to(root):
            break
        current = current.parent
        for report in sorted(current.glob("*.json")):
            try:
                payload = json.loads(report.read_text())
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            role = str(payload.get("evidence_role", ""))
            charged = payload.get("new_charged_oracle_calls")
            if charged == 0 or "zero_charged" in role or "zero_oracle" in role:
                return f"{report.relative_to(root)}: {role or 'new_charged_oracle_calls=0'}"
    return None


def read_ledger(ledger: Path) -> dict | None:
    manifest = json.loads((ledger / "oracle" / "manifest.json").read_text())
    task = manifest.get("task") or {}
    if task.get("kind") != "pmo":
        return None

    receipts = []
    skipped = 0
    for path in sorted((ledger / "oracle").glob("query_*/result.json")):
        try:
            receipts.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            skipped += 1
    if not receipts:
        return None

    receipts.sort(key=lambda r: r.get("index", 0))
    incomplete = [r for r in receipts if r.get("status") != "complete"]
    scored = [r for r in receipts if r.get("status") == "complete" and r.get("score") is not None]

    return {
        "path": ledger,
        "task": task.get("name", "?"),
        "budget": manifest.get("budget"),
        "protocol": task.get("oracle_protocol"),
        "receipts": receipts,
        "scored": scored,
        "incomplete": len(incomplete),
        "unreadable": skipped,
    }


# ---- Scoring ----


def score_ledger(entry: dict, auc) -> dict:
    """Recompute best / top-ten / AUC in CHARGE ORDER.

    A frontier metric is only defined on the charged sequence -- the archive has
    no order, so an archive-wide figure belongs to no checkpoint.
    """
    scores = [float(r["score"]) for r in entry["scored"]]
    charged = len(scores)
    result = {"charged": charged, "best": max(scores) if scores else None}
    result["top10"] = (
        statistics.fmean(sorted(scores)[-10:]) if scores else None
    )

    budget = entry["budget"] or charged
    try:
        result["auc"] = auc(scores, budget=budget, frequency=AUC_FREQUENCY)
    except ValueError as error:
        result["auc"] = None
        result["auc_error"] = str(error)
    result["auc_budget"] = budget
    result["depression"] = AUC_FREQUENCY / (2 * budget) if budget else None
    return result


def load_chemistry():
    """Return ``(Chem, QED, sascorer)`` or ``None`` when RDKit is unavailable.

    ``sascorer`` ships in RDKit's Contrib tree, which is not on ``sys.path`` by
    default, so it is reached through ``RDConfig.RDContribDir``.  ImportError
    and OSError are both reachable here (missing package, missing Contrib
    directory), and either one means the same thing to the caller: no chemistry
    column.  The score table does not depend on this.
    """
    try:
        import os
        import sys as _sys

        from rdkit import Chem, RDLogger
        from rdkit.Chem import QED, RDConfig

        contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib not in _sys.path:
            _sys.path.append(contrib)
        import sascorer  # type: ignore[import-not-found]
    except (ImportError, OSError, AttributeError):
        return None
    RDLogger.DisableLog("rdApp.*")
    return Chem, QED, sascorer


def chemistry(entry: dict) -> dict | None:
    """Median QED / SA and on-manifold share.  None when RDKit is absent."""
    loaded = load_chemistry()
    if loaded is None:
        return None
    Chem, QED, sascorer = loaded

    qeds, sas, on = [], [], 0
    for receipt in entry["scored"]:
        mol = Chem.MolFromSmiles(receipt.get("endpoint") or "")
        if mol is None:
            continue
        q = QED.qed(mol)
        s = sascorer.calculateScore(mol)
        qeds.append(q)
        sas.append(s)
        if q >= QED_FLOOR and s <= SA_CEILING:
            on += 1
    if not qeds:
        return None
    return {
        "median_qed": statistics.median(qeds),
        "median_sa": statistics.median(sas),
        "on_manifold": on / len(qeds),
        "n": len(qeds),
    }


# ---- Rendering ----


def render(entries: list[tuple[dict, dict, dict | None]], root: Path) -> None:
    by_task: dict[str, list] = {}
    for entry, scores, chem in entries:
        by_task.setdefault(entry["task"], []).append((entry, scores, chem))

    for task in sorted(by_task):
        rows = sorted(by_task[task], key=lambda t: str(t[0]["path"]))
        flag = "  [ML-PREDICTOR ORACLE -- read the chemistry column]" if task in PREDICTOR_ORACLES else ""
        print(f"\n  {task}{flag}")
        header = (f"    {'ledger':<48} {'git':>3} {'calls/budget':>13} {'best':>7} "
                  f"{'top10':>7} {'AUC':>7}")
        if any(c for _, _, c in rows):
            header += f" {'medQED':>7} {'medSA':>6} {'onMan':>6}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for entry, scores, chem in rows:
            name = str(entry["path"].relative_to(root))
            if len(name) > 48:
                name = "..." + name[-45:]
            best = "--" if scores["best"] is None else f"{scores['best']:.4f}"
            top10 = "--" if scores["top10"] is None else f"{scores['top10']:.4f}"
            auc = "--" if scores["auc"] is None else f"{scores['auc']:.4f}"
            # A partial run's AUC integrates few points over the full declared
            # budget, so it is tiny for reasons that have nothing to do with
            # search quality.  Showing charged/budget is what makes that legible.
            span = f"{scores['charged']}/{scores['auc_budget']}"
            mark = {True: "yes", False: "NO", None: "?"}[entry["tracked"]]
            line = (f"    {name:<48} {mark:>3} {span:>13} {best:>7} "
                    f"{top10:>7} {auc:>7}")
            if chem:
                line += (f" {chem['median_qed']:>7.3f} {chem['median_sa']:>6.2f} "
                         f"{chem['on_manifold'] * 100:>5.1f}%")
            elif any(c for _, _, c in rows):
                line += f" {'--':>7} {'--':>6} {'--':>6}"
            print(line)
            notes = []
            if entry["incomplete"] or entry["unreadable"]:
                notes.append(f"{entry['incomplete']} incomplete, "
                             f"{entry['unreadable']} unreadable receipt(s)")
            if scores["charged"] < (scores["auc_budget"] or 0):
                notes.append(f"PARTIAL: {scores['charged']} of "
                             f"{scores['auc_budget']} budgeted calls -- its AUC is "
                             f"not comparable to a completed run")
            if entry["tracked"] is False:
                notes.append("NOT COMMITTED: a fresh clone will not have this ledger")
            for note in notes:
                print(f"    {'':<48}   ^ {note}")

        budgets = {s["auc_budget"] for _, s, _ in rows}
        for budget in sorted(b for b in budgets if b and b < OFFICIAL_BUDGET):
            pct = AUC_FREQUENCY / (2 * budget) * 100
            print(f"    NOTE: AUC at budget {budget} is structurally depressed "
                  f"{pct:.1f}% vs the official {OFFICIAL_BUDGET}-call budget.")


# ---- Entry point ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute PMO scores from committed oracle ledgers. "
                    "No network, no PyTDC, no new oracle calls."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--task", help="restrict to one PMO task name")
    parser.add_argument("--no-chemistry", action="store_true",
                        help="skip the RDKit manifold column even if available")
    parser.add_argument("--include-zero-charged", action="store_true",
                        help="ALSO show ledgers whose own report declares they "
                             "charged no oracle calls (their scores are not "
                             "production scores)")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("PMO -- recomputed from committed oracle ledgers")
    print("=" * 78)
    print("  STATUS: development evidence. There is NO frozen PMO paper table;")
    print("          nothing below is a submitted result.")

    try:
        auc = load_auc()
        self_test(auc)
    except ReproductionError as error:
        print(f"\nFAIL: {error}", file=sys.stderr)
        return 2
    print("  [1/2] AUC self-test OK   production pmo_top_ten_auc reproduces the "
          "closed form")

    ledgers = find_ledgers(args.repo_root)
    if not ledgers:
        print("\nFAIL: no PMO oracle ledgers found under diagnostics/.",
              file=sys.stderr)
        print("  Expected <dir>/oracle/manifest.json + oracle/query_*/result.json",
              file=sys.stderr)
        return 2

    prefixes = tracked_prefixes(args.repo_root)
    entries, unreadable, excluded = [], 0, []
    chem_available = None
    for ledger in ledgers:
        try:
            entry = read_ledger(ledger)
        except (OSError, json.JSONDecodeError):
            unreadable += 1
            continue
        if entry is None:
            continue
        relative = str(ledger.relative_to(args.repo_root))
        entry["tracked"] = None if prefixes is None else (relative in prefixes)
        declaration = zero_charged_declaration(ledger, args.repo_root)
        if declaration is not None and not args.include_zero_charged:
            excluded.append((relative, declaration))
            continue
        if args.task and entry["task"] != args.task:
            continue
        scores = score_ledger(entry, auc)
        chem = None if args.no_chemistry else chemistry(entry)
        if chem is not None:
            chem_available = True
        elif chem_available is None and not args.no_chemistry:
            chem_available = False
        entries.append((entry, scores, chem))

    if not entries:
        print(f"\nNo PMO ledgers matched"
              f"{' for task ' + args.task if args.task else ''}.", file=sys.stderr)
        return 2

    total_calls = sum(s["charged"] for _, s, _ in entries)
    print(f"  [2/2] ledgers read OK    {len(entries)} ledger(s), "
          f"{total_calls} charged calls total")
    if unreadable:
        print(f"        {unreadable} ledger(s) unreadable and SKIPPED")
    if excluded:
        print(f"        {len(excluded)} ledger(s) EXCLUDED -- their own committed "
              f"report declares zero charged oracle calls, so their scores are "
              f"not production scores:")
        for path, why in excluded:
            print(f"          - {path}")
            print(f"              evidence: {why}")
        print("        (pass --include-zero-charged to show them anyway)")
    untracked = [e for e, _, _ in entries if e["tracked"] is False]
    if untracked:
        print(f"        {len(untracked)} ledger(s) are NOT COMMITTED and will be "
              f"absent from a fresh clone (git column below)")
    elif prefixes is None:
        print("        git unavailable: cannot say which ledgers a fresh clone "
              "would have")
    if args.no_chemistry:
        print("        chemistry column: DISABLED by --no-chemistry")
    elif chem_available:
        print("        chemistry column: RDKit available")
    else:
        print("        chemistry column: UNAVAILABLE (RDKit not importable) -- "
              "score table is unaffected")

    render(entries, args.repo_root)

    print("\n" + "=" * 78)
    print("RECOMPUTED -- 0 new oracle calls, 0 network requests.")
    print("Never scale a short-budget AUC and compare it to a published "
          f"{OFFICIAL_BUDGET}-call figure.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
