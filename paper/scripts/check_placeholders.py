#!/usr/bin/env python3
"""FINAL-mode gate: fail if the manuscript is not author-ready.

Checks (all must pass for `make paper-final`):
  1. No pending result in results_registry.yaml.
  2. No \\METHODCHECK{...} anywhere in the TeX sources.
  3. No \\pendingfigure{...} (every figure must render from a real artifact).
  4. No stray XXX / TODO / FIXME / PLACEHOLDER tokens in non-comment TeX.
  5. Main-body page count (intro..conclusion) <= PAGE_LIMIT, via the
     `mainbodyend` sentinel label in main.aux.
  6. No undefined citations or references in main.log.
  7. No reference to a stale-results manifest.

Exit code 0 = ready; nonzero = not ready (with a report). This is a source/
artifact gate; it does not itself compile the paper (compile_paper.sh does).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PAPER = Path(__file__).resolve().parent.parent
PAGE_LIMIT = 9
TEX_DIRS = ["sections", "appendix", "figures", "tables"]
STRAY_TOKENS = re.compile(r"(?<![A-Za-z])(TODO|FIXME|PLACEHOLDER)(?![A-Za-z])")


def _tex_files() -> list[Path]:
    files = [PAPER / "main.tex"]
    for d in TEX_DIRS:
        files += sorted((PAPER / d).glob("*.tex"))
    return [f for f in files if f.exists()]


def _strip_comments(line: str) -> str:
    out, esc = [], False
    for ch in line:
        if ch == "\\" and not esc:
            esc = True
            out.append(ch)
            continue
        if ch == "%" and not esc:
            break
        esc = False
        out.append(ch)
    return "".join(out)


def check_pending_results(failures: list[str]) -> None:
    reg = yaml.safe_load((PAPER / "results_registry.yaml").read_text()) or {}
    pending = []
    for key, rec in (reg.get("results", {}) or {}).items():
        if isinstance(rec, dict) and rec.get("status", "pending") != "final":
            pending.append(key)
    # grids are always pending until filled per-cell; treat any grid as pending
    if reg.get("grids"):
        pending.append("<grid cells: %d grids>" % len(reg["grids"]))
    if pending:
        failures.append(f"{len(pending)} pending result key(s): " + ", ".join(map(str, pending[:12]))
                        + (" ..." if len(pending) > 12 else ""))


def check_tex_tokens(failures: list[str]) -> None:
    methodcheck, pendingfig, stray = [], [], []
    for f in _tex_files():
        for i, raw in enumerate(f.read_text().splitlines(), 1):
            line = _strip_comments(raw)
            if "\\METHODCHECK" in line:
                methodcheck.append(f"{f.name}:{i}")
            if "\\pendingfigure" in line:
                pendingfig.append(f"{f.name}:{i}")
            if STRAY_TOKENS.search(line):
                stray.append(f"{f.name}:{i}")
    if methodcheck:
        failures.append("\\METHODCHECK present: " + ", ".join(methodcheck))
    if pendingfig:
        failures.append("\\pendingfigure present (unrendered figure): " + ", ".join(pendingfig))
    if stray:
        failures.append("stray TODO/FIXME/PLACEHOLDER: " + ", ".join(stray))


def check_page_limit(failures: list[str]) -> None:
    aux = PAPER / "main.aux"
    if not aux.exists():
        failures.append("main.aux missing (compile first); cannot verify page limit")
        return
    m = re.search(r"\\newlabel\{mainbodyend\}\{\{[^}]*\}\{(\d+)\}", aux.read_text())
    if not m:
        failures.append("mainbodyend sentinel not found in main.aux; cannot verify page limit")
        return
    page = int(m.group(1))
    if page > PAGE_LIMIT:
        failures.append(f"main body ends on page {page} > {PAGE_LIMIT}-page limit")


def check_log(failures: list[str]) -> None:
    log = PAPER / "main.log"
    if not log.exists():
        failures.append("main.log missing (compile first)")
        return
    text = log.read_text(errors="ignore")
    if re.search(r"Citation `[^']+' .*undefined", text):
        failures.append("undefined citation(s) in main.log")
    if re.search(r"Reference `[^']+' on page \d+ undefined", text):
        failures.append("undefined reference(s) in main.log")


def check_stale_refs(failures: list[str]) -> None:
    for f in _tex_files():
        txt = f.read_text()
        if "STALE_RESULTS_MANIFEST" in txt or "composition_learned_vs_uniform" in txt:
            failures.append(f"reference to a stale-results artifact in {f.name}")


def main() -> int:
    failures: list[str] = []
    check_pending_results(failures)
    check_tex_tokens(failures)
    check_page_limit(failures)
    check_log(failures)
    check_stale_refs(failures)
    if failures:
        print("check_placeholders: NOT READY FOR FINAL\n")
        for msg in failures:
            print("  - " + msg)
        print("\n(These are expected while results are pending; use `make paper-draft`.)")
        return 1
    print("check_placeholders: OK -- no placeholders, no METHODCHECK, within page limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
