"""Can a DIFFERENT alignment of the same (G, T) pair be staged monotonically?

Zero oracle calls, zero program execution. This is the discriminating experiment for the
stage-splitter finding, and both branches lead somewhere:

  IF some alignments are monotone, the planner should select alignments on TRAJECTORY
  SHAPE rather than on scale, and the compiler is unblocked.

  IF none are, a score-greedy controller provably cannot follow a transport. That is a
  statement about the OBJECTIVE rather than a limitation of COMPOSE, and it makes the
  basin exploration floor load-bearing -- budget protected from score-based selection --
  rather than merely free.

`correspondences` already returns the top-K alignments of a pair; they are genuinely
different transports, not re-spellings of one, because a symmetric molecule admits several
mappings of the same core. Each is staged independently and its trajectory measured.

WHY DEPTH AND WIDTH, not just whether a dip occurs. A transport dipping 2% for one stage
is navigable by almost any selection tolerance; one dipping 40% for two consecutive stages
needs genuinely protected budget. The distribution is the difference between a tunable
parameter and an architectural requirement, so it is reported rather than collapsed into a
boolean.
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
    parser.add_argument("--alignments", type=int, default=8)
    parser.add_argument("--max-primitives", type=int, default=DEFAULT_MAX_PRIMITIVES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    audit = json.loads(
        (ROOT / "diagnostics/pmo_discovery_v1/goal_specification_audit_v1.json").read_text()
    )
    bank = json.loads((ROOT / "docs/PMO_INIT_BANK.json").read_text())
    if isinstance(bank, dict):
        bank = bank.get("smiles") or bank.get("molecules")

    pairs = []
    for entry in audit["rows"]:
        declared = [d for d in entry.get("declared", []) if d["kind"] == "smiles"]
        if not declared or entry["goal_kind"] == "declared_smarts":
            # declared_smarts supplies a reference scaffold, not a molecule to reproduce,
            # so a trajectory toward it is not the objective and would pollute the answer.
            continue
        target = declared[0]["value"]
        for source in best_start(bank, target, top=args.starts):
            pairs.append((entry["task"], entry["goal_kind"], source, target))

    rows = []
    for task, goal_kind, source, target in pairs:
        found = correspondences(source, target, top_k=args.alignments)
        if not found:
            continue
        alignments = []
        for index, correspondence in enumerate(found):
            for interleave in (False, True):
                stages = split(
                    correspondence,
                    max_primitives=args.max_primitives,
                    interleave=interleave,
                )
                checks = validate_staging(
                    correspondence, stages, max_primitives=args.max_primitives
                )
                alignments.append({
                    "alignment": index,
                    "interleave": interleave,
                    "scale": correspondence.scale,
                    "retain_core": len(correspondence.retain_core),
                    "stages": checks["stages"],
                    "monotone": checks["similarity_monotone_nondecreasing"],
                    "dips_below_source": checks["dips_below_source"],
                    "dip_depth_absolute": checks["dip_depth_absolute"],
                    "dip_depth_relative": checks["dip_depth_relative"],
                    "dip_width_stages": checks["dip_width_stages"],
                    "reaches_target": checks["reaches_target"],
                    "connected": checks["all_intermediates_connected"],
                    "valid": checks["all_intermediates_valid"],
                })
        usable = [
            a for a in alignments
            if a["valid"] and a["connected"] and a["reaches_target"]
        ]
        monotone = [a for a in usable if a["monotone"] is not False]
        # What the CURRENT planner does: first alignment, prune-first order.
        default = next(a for a in alignments if a["alignment"] == 0 and not a["interleave"])
        interleaved = next(
            (a for a in alignments if a["alignment"] == 0 and a["interleave"]), None
        )
        rows.append({
            "task": task,
            "goal_kind": goal_kind,
            "source": source,
            "alignments_examined": len(alignments),
            "usable_alignments": len(usable),
            "monotone_alignments": len(monotone),
            "any_monotone": bool(monotone),
            # Honest separation: an alternative ALIGNMENT only exists for symmetric
            # molecules -- 26 of 35 real pairs admit exactly one -- so "not rescued by an
            # alignment" is definitionally true for most pairs and is NOT evidence.
            "alternative_alignment_available": len(found) > 1,
            "default_is_monotone": default["monotone"] is not False,
            "interleaved_is_monotone": (
                None if interleaved is None else interleaved["monotone"] is not False
            ),
            "interleaved_dip_depth_relative": (
                None if interleaved is None else interleaved["dip_depth_relative"]
            ),
            "interleaved_preserves_endpoint": (
                None
                if interleaved is None
                else interleaved["reaches_target"] == default["reaches_target"]
                and interleaved["valid"]
                and interleaved["connected"]
            ),
            "default_dip_depth_relative": default["dip_depth_relative"],
            "default_dip_width_stages": default["dip_width_stages"],
            "alignments": alignments,
        })

    dipping = [r for r in rows if not r["default_is_monotone"]]
    with_alternative = [r for r in dipping if r["alternative_alignment_available"]]
    rescued_alignment = [r for r in with_alternative if r["any_monotone"]]
    rescued_interleave = [r for r in dipping if r["interleaved_is_monotone"]]
    interleave_safe = [r for r in dipping if r["interleaved_preserves_endpoint"]]
    depths = [
        r["default_dip_depth_relative"] for r in dipping
        if r["default_dip_depth_relative"] is not None
    ]
    widths = [
        r["default_dip_width_stages"] for r in dipping
        if r["default_dip_width_stages"] is not None
    ]

    print(f"pairs: {len(rows)}   alignments per pair: up to {args.alignments}")
    print(f"  default (prune-first) dips:    {len(dipping)}/{len(rows)}")
    print(f"  dipping pairs WITH an alternative alignment at all: {len(with_alternative)}")
    print(f"    rescued by another alignment: {len(rescued_alignment)}/{len(with_alternative)}")
    print(f"  RESCUED BY INTERLEAVING:       {len(rescued_interleave)}/{len(dipping)}")
    print(f"    of which endpoint preserved: {len(interleave_safe)}/{len(dipping)}")
    if depths:
        print(f"  dip depth (relative to source): median {statistics.median(depths):.1%}"
              f"  min {min(depths):.1%}  max {max(depths):.1%}")
        shallow = sum(1 for d in depths if d <= 0.10)
        print(f"    <=10% deep: {shallow}/{len(depths)}    >25% deep: "
              f"{sum(1 for d in depths if d > 0.25)}/{len(depths)}")
    if widths:
        print(f"  dip width (consecutive stages): median {statistics.median(widths)}"
              f"  max {max(widths)}")
    verdict = (
        "NO_DIPPING_PAIRS"
        if not dipping
        else "INTERLEAVING_REMOVES_EVERY_DIP"
        if len(rescued_interleave) == len(dipping)
        else "INTERLEAVING_REMOVES_MOST_DIPS"
        if len(rescued_interleave) >= 0.5 * len(dipping)
        else "DIPS_SURVIVE_REORDERING_THE_OBJECTIVE_CANNOT_GUIDE_TRANSPORT"
    )
    print(f"VERDICT: {verdict}")
    report = {
        "schema_version": "pmo_transport_alignment_selection_v1",
        "oracle_calls": 0,
        "programs_executed": 0,
        "pairs": len(rows),
        "default_dipping": len(dipping),
        "dipping_with_an_alternative_alignment": len(with_alternative),
        "rescued_by_alternative_alignment": len(rescued_alignment),
        "rescued_by_interleaving": len(rescued_interleave),
        "interleaving_preserves_endpoint": len(interleave_safe),
        "dip_depth_relative_median": statistics.median(depths) if depths else None,
        "dip_width_stages_median": statistics.median(widths) if widths else None,
        "verdict": verdict,
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
