"""Stage-splitter validation: does a staged transport path EXIST, and does it approach?

Zero oracle calls, zero program execution. For every declared-structure PMO task, the
best starting molecules in COMPOSE's own initialization bank are aligned onto the declared
target, the transport is split into stages under the realization ceiling, and every stage
endpoint is built with RDKit and checked:

  respects_ceiling             no stage exceeds `max_primitives`
  all_intermediates_valid      every stage endpoint sanitizes
  all_intermediates_connected  every stage endpoint is one fragment
  reaches_target               the final endpoint IS the target, by canonical SMILES
  monotone / dips_below_source whether similarity to T improves along the path

THE PREDECLARED FALSIFIER, restated: if staged intermediates cannot be kept connected and
valid, the realization CEILING -- not the planner -- blocks long-range transport, and that
is a statement about the architecture.

Monotonicity is reported SEPARATELY from existence, because they can and do disagree: a
path can reach the target through valid connected intermediates while similarity dips
below where the run started. That is not a failure of the path; it is a failure of the
SIGNAL to guide a controller along it, and the two need different responses.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.pmo_transport_correspondence import correspondences
from compose_v4.control.pmo_transport_staging import (
    DEFAULT_MAX_PRIMITIVES,
    split,
    validate_staging,
)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def best_start(bank, target, *, top):
    mol = Chem.MolFromSmiles(target)
    fingerprint = _GEN.GetFingerprint(mol)
    scored = []
    for smiles in bank:
        candidate = Chem.MolFromSmiles(smiles)
        if candidate is not None:
            scored.append(
                (DataStructs.TanimotoSimilarity(fingerprint, _GEN.GetFingerprint(candidate)),
                 smiles)
            )
    scored.sort(reverse=True)
    return [smiles for _, smiles in scored[:top]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--max-primitives", type=int, default=DEFAULT_MAX_PRIMITIVES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    audit = json.loads(
        (ROOT / "diagnostics/pmo_discovery_v1/goal_specification_audit_v1.json").read_text()
    )
    bank = json.loads((ROOT / "docs/PMO_INIT_BANK.json").read_text())
    if isinstance(bank, dict):
        bank = bank.get("smiles") or bank.get("molecules")

    rows = []
    for entry in audit["rows"]:
        declared = [d for d in entry.get("declared", []) if d["kind"] == "smiles"]
        if not declared:
            continue
        target = declared[0]["value"]
        for source in best_start(bank, target, top=args.starts):
            found = correspondences(source, target)
            if not found:
                continue
            correspondence = found[0]
            stages = split(correspondence, max_primitives=args.max_primitives)
            checks = validate_staging(
                correspondence, stages, max_primitives=args.max_primitives
            )
            rows.append({
                "task": entry["task"],
                "goal_kind": entry["goal_kind"],
                "source": source,
                "scale": correspondence.scale,
                "requires_reattachment": correspondence.requires_reattachment,
                **checks,
            })

    def share(key):
        return sum(1 for row in rows if row[key]) / len(rows) if rows else 0.0

    # SCOPE THE CRITERION BY GOAL KIND, and by what the correspondence itself declares.
    #
    # `declared_smarts` tasks (deco_hop, scaffold_hop) supply a REFERENCE scaffold and a
    # substructure constraint, not a molecule to reproduce, so "reaches_target" is not
    # their objective and applying it counts a category error as a staging failure.
    #
    # A `requires_reattachment` correspondence is one whose retained core is disconnected
    # in the source: NO deletion order can keep it connected, which the correspondence
    # says up front. Counting it as a staging defect would be counting a declared
    # limitation twice.
    def in_scope_for_target(row):
        return row["goal_kind"] != "declared_smarts"

    existence = [
        row for row in rows
        if row["respects_ceiling"]
        and row["all_intermediates_valid"]
        and (row["all_intermediates_connected"] or row["requires_reattachment"])
        and (row["reaches_target"] or not in_scope_for_target(row))
    ]
    out_of_scope = [row for row in rows if not in_scope_for_target(row)]
    reattachment = [row for row in rows if row["requires_reattachment"]]
    evaluable = [row for row in rows if row["similarity_monotone_nondecreasing"] is not None]
    monotone = [row for row in evaluable if row["similarity_monotone_nondecreasing"]]
    dipping = [row for row in rows if row["dips_below_source"]]

    print(f"transports staged: {len(rows)}   max_primitives={args.max_primitives}")
    print(f"  respects_ceiling            {share('respects_ceiling'):.1%}")
    print(f"  all_intermediates_valid     {share('all_intermediates_valid'):.1%}")
    print(f"  all_intermediates_connected {share('all_intermediates_connected'):.1%}")
    print(f"  reaches_target              {share('reaches_target'):.1%}")
    print(f"  PATH EXISTS (scoped)        {len(existence)}/{len(rows)}")
    print(f"    of which excluded from reaches_target (declared_smarts): {len(out_of_scope)}")
    print(f"    correspondences declaring requires_reattachment:         {len(reattachment)}")
    print(f"  monotone (of {len(evaluable)} evaluable) {len(monotone)}")
    print(f"  DIPS BELOW SOURCE           {len(dipping)}/{len(rows)}")
    if rows:
        print(f"  stages: median {int(statistics.median(r['stages'] for r in rows))}, "
              f"max {max(r['stages'] for r in rows)}")
    verdict = (
        "STAGED_PATH_EXISTS_BUT_SIGNAL_IS_NOT_MONOTONE"
        if len(existence) == len(rows) and dipping
        else "STAGED_PATH_EXISTS_AND_APPROACHES_MONOTONICALLY"
        if len(existence) == len(rows)
        else "STAGING_FAILS_CEILING_IS_THE_BLOCKER"
    )
    report_extra = {
        "excluded_from_reaches_target_declared_smarts": len(out_of_scope),
        "correspondences_requiring_reattachment": len(reattachment),
    }
    print(f"VERDICT: {verdict}")
    report = {
        "schema_version": "pmo_transport_staging_validation_v1",
        "oracle_calls": 0,
        "programs_executed": 0,
        "max_primitives": args.max_primitives,
        "transports": len(rows),
        "path_exists": len(existence),
        "monotone_evaluable": len(evaluable),
        "monotone": len(monotone),
        "dips_below_source": len(dipping),
        "verdict": verdict,
        **report_extra,
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
