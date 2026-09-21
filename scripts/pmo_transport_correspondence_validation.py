"""Step 1 validation: the (G, T) correspondence on real declared targets.

Zero oracle calls, zero program execution. For every declared-structure PMO task, the
best starting molecule in COMPOSE's own initialization bank is aligned onto the declared
target and every offline check is run:

  partitions_source                  retain_core and R_delete cover G exactly, disjointly
  core_is_ring_complete_substructure the core is a real substructure of BOTH endpoints,
                                     with no ring left partly in and partly out
  deletion_connectivity_satisfied_or_reattachment_declared
                                     removing R_delete in the recorded order never
                                     disconnects what survives -- OR the correspondence
                                     declares that its retained core is disconnected in
                                     the source, which no deletion order can fix and which
                                     makes the transport an excision PLUS a reattachment
  zero_scale_implies_identical       a transport of size zero means the same molecule

The reported `scale` is the honest transport size: atoms deleted + atoms installed +
bond orders changed inside the retained core. It is LARGER than a raw MCS reading,
because the core is pruned to ring-complete and bond-order changes are counted -- both
corrections raise the estimate rather than lower it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.pmo_transport_correspondence import correspondences, validate

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def best_start(bank: list[str], target: str, *, top: int) -> list[str]:
    """The `top` bank molecules most similar to the target. Objective-blind: similarity
    to a DECLARED target is task structure, not oracle output."""
    mol = Chem.MolFromSmiles(target)
    fingerprint = _GEN.GetFingerprint(mol)
    scored = []
    for smiles in bank:
        candidate = Chem.MolFromSmiles(smiles)
        if candidate is None:
            continue
        scored.append(
            (DataStructs.TanimotoSimilarity(fingerprint, _GEN.GetFingerprint(candidate)), smiles)
        )
    scored.sort(reverse=True)
    return [smiles for _, smiles in scored[:top]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    audit = json.loads(
        (ROOT / "diagnostics/pmo_discovery_v1/goal_specification_audit_v1.json").read_text()
    )
    bank = json.loads((ROOT / "docs/PMO_INIT_BANK.json").read_text())
    if isinstance(bank, dict):
        bank = bank.get("smiles") or bank.get("molecules")

    rows, failures, no_core = [], [], []
    for entry in audit["rows"]:
        declared = [d for d in entry.get("declared", []) if d["kind"] == "smiles"]
        if not declared:
            continue
        target = declared[0]["value"]
        for source in best_start(bank, target, top=args.starts):
            found = correspondences(source, target)
            if not found:
                no_core.append({"task": entry["task"], "source": source})
                continue
            for correspondence in found:
                checks = validate(correspondence)
                row = {
                    "task": entry["task"],
                    "goal_kind": entry["goal_kind"],
                    "source": source,
                    "retain_core": len(correspondence.retain_core),
                    "r_delete": len(correspondence.r_delete),
                    "h_install": len(correspondence.h_install),
                    "core_bond_changes": len(correspondence.core_bond_changes),
                    "alpha": len(correspondence.alpha),
                    "scale": correspondence.scale,
                    "checks": checks,
                }
                rows.append(row)
                # Judge on the CONDITIONED key. `deletion_keeps_source_connected` and
                # `requires_reattachment` are reported for diagnosis, not as verdicts.
                decisive = {
                    key: value
                    for key, value in checks.items()
                    if key not in ("deletion_keeps_source_connected", "requires_reattachment")
                }
                row["requires_reattachment"] = correspondence.requires_reattachment
                if not all(decisive.values()):
                    failures.append(row)

    by_task: dict[str, list[int]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row["scale"])
    for task, scales in sorted(by_task.items()):
        print(f"  {task:<30} alignments={len(scales):<3} "
              f"min_scale={min(scales):<3} median_scale={int(statistics.median(scales))}")
    scales = [row["scale"] for row in rows]
    print(
        f"\ncorrespondences={len(rows)}  tasks={len(by_task)}  "
        f"VALIDATION FAILURES={len(failures)}  no_shared_core={len(no_core)}"
    )
    if scales:
        print(
            f"scale: min {min(scales)}  median {int(statistics.median(scales))}  max {max(scales)}"
        )
    verdict = "STEP1_VALIDATED" if rows and not failures else "STEP1_VALIDATION_FAILED"
    print(f"VERDICT: {verdict}")
    report = {
        "schema_version": "pmo_transport_correspondence_validation_v1",
        "oracle_calls": 0,
        "programs_executed": 0,
        "correspondences": len(rows),
        "validation_failures": failures,
        "pairs_with_no_shared_ring_complete_core": no_core,
        "scale_summary": (
            {
                "min": min(scales),
                "median": int(statistics.median(scales)),
                "max": max(scales),
            }
            if scales
            else None
        ),
        "verdict": verdict,
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if verdict == "STEP1_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
