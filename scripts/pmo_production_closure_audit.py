"""Which mechanisms can the scored PMO run actually reach?

This repository has twice paid for a mechanism that was built, tested, merged and INERT:
the T4 region law behind a keyword no production caller passed, and
`OnlineProposalMemory.allocation_priority`, whose docstring named the gap it closed while
having zero production call sites.  Both were found by inspection after the fact.  This
makes the check a command.

WHAT IT MEASURES, AND THE TWO WAYS A MODULE CAN BE ABSENT
----------------------------------------------------------
It imports a production ENTRY POINT and records the `compose_v4` modules that import
brings in.  A module outside that closure cannot be called at import time -- but a
DEFERRED import inside a function body would not appear either, and this repo contains
exactly that pattern (`pmo_option_particles.py` imports `donor_memory` inside a function).
So absence is established two ways:

  1. the module is not in `sys.modules` after importing the entry point, AND
  2. no source line of any module in that closure mentions it

Reporting only (1) would call a deferred import "absent".  Reporting only (2) would be
confused by an unrelated LOCAL VARIABLE of the same name -- which is not hypothetical:
`adaptive_program_optimizer` binds a local called `donor_program` that has nothing to do
with the module, so the source scan alone reads as a hit.  Both signals are reported
separately and the verdict names which one fired.

This audit is about REACHABILITY, not execution.  A module in the closure is imported, not
necessarily called -- the standing rule that imports are not calls (the PMO path imports
torch and instantiates no model) applies in this direction too.  Absence is the strong
conclusion here; presence is only a prerequisite.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENTRY = "compose_v4.experiments.pmo_population_v1"

#: Mechanisms worth asking about for a PMO discovery claim, each with what it would supply.
WATCHED = {
    "compose_v4.control.donor_program":
        "molecular donor recombination -- pendant exchange between two complete molecules, "
        "with the retained core preserved by assertion after replay",
    "compose_v4.control.donor_memory":
        "the scored donor pool and its cut distribution (uniform_oriented_single_bridge)",
    "compose_v4.control.pmo_online_memory":
        "arm B -- donor REGION associations and edit-outcome memory biasing the region law",
    "compose_v4.control.pmo_discovery":
        "arm C -- basin-stratified exploration floor and frontier credit",
    "compose_v4.control.pmo_transport_correspondence":
        "the G->T correspondence planner (retain_core, R_delete, H_install, alpha, D)",
    "compose_v4.control.pmo_transport_staging":
        "the stage splitter and ordering selector under the realization ceiling",
    "compose_v4.control.bridge_region_law":
        "the T4 region repair -- conditioned, uncapped bridge-separated region draw",
}


def audit(entry: str) -> dict:
    importlib.import_module(entry)
    closure = {
        name: Path(module.__file__)
        for name, module in sorted(sys.modules.items())
        if name.startswith("compose_v4") and getattr(module, "__file__", None)
    }
    sources = {name: path.read_text() for name, path in closure.items()}

    rows = []
    for module, supplies in sorted(WATCHED.items()):
        leaf = module.rsplit(".", 1)[-1]
        in_closure = module in closure
        # A dotted reference cannot be a local variable, so it is the sound source signal.
        dotted = re.compile(re.escape(module))
        # The bare leaf is reported separately BECAUSE it collides with local variables.
        bare = re.compile(rf"\b{re.escape(leaf)}\b")
        dotted_hits, bare_hits = [], []
        for name, text in sources.items():
            if name == module:
                continue
            for index, line in enumerate(text.splitlines(), 1):
                if dotted.search(line):
                    dotted_hits.append({"module": name, "line": index, "text": line.strip()[:120]})
                elif bare.search(line):
                    bare_hits.append({"module": name, "line": index, "text": line.strip()[:120]})
        rows.append({
            "module": module,
            "supplies": supplies,
            "in_import_closure": in_closure,
            "dotted_references_in_closure": len(dotted_hits),
            "bare_name_references_in_closure": len(bare_hits),
            "dotted_examples": dotted_hits[:5],
            "bare_name_examples": bare_hits[:5],
            "reachable": in_closure or bool(dotted_hits),
            "bare_name_only": (not in_closure) and (not dotted_hits) and bool(bare_hits),
        })
    return {
        "schema_version": "pmo_production_closure_audit_v1",
        "oracle_calls": 0,
        "entry_point": entry,
        "closure_modules": len(closure),
        "scope_note": (
            "reachability, not execution: a module in the closure is imported, not "
            "necessarily called"
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", default=DEFAULT_ENTRY)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = audit(args.entry)
    print(f"{report['entry_point']}: {report['closure_modules']} compose_v4 modules in closure\n")
    for row in report["rows"]:
        if row["in_import_closure"]:
            mark = "REACHABLE  (imported)"
        elif row["dotted_references_in_closure"]:
            mark = f"REACHABLE  (deferred, {row['dotted_references_in_closure']} refs)"
        elif row["bare_name_only"]:
            mark = f"ABSENT     (bare-name collisions only: {row['bare_name_references_in_closure']})"
        else:
            mark = "ABSENT"
        print(f"  {mark:<46} {row['module']}")
    absent = [r["module"] for r in report["rows"] if not r["reachable"]]
    print(f"\nunreachable from the scored entry point: {len(absent)}")
    for module in absent:
        print(f"  {module}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
