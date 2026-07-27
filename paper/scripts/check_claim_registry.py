#!/usr/bin/env python3
"""Validate CLAIM_LEDGER.md and the registry/macro cross-references.

Checks:
  1. CLAIM_LEDGER.md parses as a table with the required 9 columns and every
     data row is complete.
  2. Every `Result key` cited by a claim exists in results_registry.yaml
     (or is a config key, or the literal `implementation`/`n/a`).
  3. Every \\Result{KEY}, \\ResultCI{KEY}, and \\ResultSentence{KEY} used in the
     TeX resolves to a results_registry key (scalar, grid-expanded, or sentence).
  4. Every \\ConfigValue{KEY} used in the TeX resolves to a config_registry key.

Exit 0 if consistent, else nonzero with a report. Safe to run any time (it does
not require the paper to be compiled).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PAPER = Path(__file__).resolve().parent.parent
REQUIRED_COLS = ["ID", "Exact proposed claim", "Type", "Assumptions", "Evidence",
                 "Main/Appendix", "Result key", "Status", "Overclaim risk"]
TEX_DIRS = ["sections", "appendix", "figures", "tables"]


def tex_files() -> list[Path]:
    files = [PAPER / "main.tex"]
    for d in TEX_DIRS:
        files += sorted((PAPER / d).glob("*.tex"))
    return [f for f in files if f.exists()]


def registry_result_keys() -> set[str]:
    reg = yaml.safe_load((PAPER / "results_registry.yaml").read_text()) or {}
    keys = set((reg.get("results", {}) or {}).keys())
    for grid in reg.get("grids", []) or []:
        for metric in grid["metrics"]:
            for method in grid["methods"]:
                keys.add(f"{grid['prefix']}_{metric['key']}_{method['key']}")
    return keys


def config_keys() -> set[str]:
    cfg = yaml.safe_load((PAPER / "config_registry.yaml").read_text()) or {}
    return {k for k, v in cfg.items() if isinstance(v, dict) and "value" in v}


def parse_ledger(failures: list[str]) -> list[dict]:
    path = PAPER / "CLAIM_LEDGER.md"
    if not path.exists():
        failures.append("CLAIM_LEDGER.md not found")
        return []
    rows = []
    header = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):  # separator row
            continue
        if header is None:
            header = cells
            missing = [c for c in REQUIRED_COLS if c not in header]
            if missing:
                failures.append(f"CLAIM_LEDGER header missing columns: {missing}")
            continue
        if len(cells) != len(header):
            failures.append(f"ragged ledger row ({len(cells)} vs {len(header)} cols): {cells[:1]}")
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def main() -> int:
    failures: list[str] = []
    res_keys = registry_result_keys()
    cfg_keys = config_keys()
    rows = parse_ledger(failures)

    # 1-2. ledger completeness + result-key existence
    for row in rows:
        rid = row.get("ID", "?")
        for col in REQUIRED_COLS:
            if not row.get(col):
                failures.append(f"claim {rid}: empty '{col}'")
        rk = row.get("Result key", "").strip()
        for key in re.split(r"[,\s]+", rk):
            key = key.strip("` ")
            if not key or key.lower() in {"implementation", "n/a", "none", "--"}:
                continue
            if key not in res_keys and key not in cfg_keys:
                failures.append(f"claim {rid}: result key '{key}' not in any registry")

    # 3-4. macro references resolve
    macro = re.compile(r"\\(Result|ResultCI|ResultSentence|ConfigValue)\{([^}]+)\}")
    for f in tex_files():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            # ignore macro DEFINITIONS in math_commands.tex
            if f.name == "math_commands.tex":
                continue
            for mm in macro.finditer(line):
                kind, key = mm.group(1), mm.group(2).strip()
                if kind == "ConfigValue":
                    if key not in cfg_keys:
                        failures.append(f"{f.name}:{i}: \\ConfigValue{{{key}}} not in config_registry")
                else:
                    if key not in res_keys:
                        failures.append(f"{f.name}:{i}: \\{kind}{{{key}}} not in results_registry")

    if failures:
        print("check_claim_registry: INCONSISTENT\n")
        for msg in failures[:60]:
            print("  - " + msg)
        if len(failures) > 60:
            print(f"  ... and {len(failures)-60} more")
        return 1
    print(f"check_claim_registry: OK -- {len(rows)} claims, all result/config keys resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
