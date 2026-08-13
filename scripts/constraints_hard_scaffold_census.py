"""Stage-1 protected-core census — the MODEL-FREE half.

Measures the protected-object definition on a held-in source cohort using only
RDKit. It needs **no** `R_theta` checkpoint, **no** successor kernel, **no**
oracle and **no** Gate-0 authenticated chain, so it runs locally on CPU while
the fiber-dependent half of the census stays blocked (see
`docs/workstreams/constraints-hard/SCAFFOLD_FEASIBILITY.md`).

What it can establish
---------------------
Whether the frozen protected object — the atom- and bond-labeled Bemis-Murcko
scaffold of the source — is a *usable* core: non-empty, not the whole molecule,
and leaving enough structure outside itself to edit.

What it CANNOT establish
------------------------
Anything about the successor fiber: support retention `|F_C(x)|/|F(x)|`,
mask-empty rate, or the frequency of legal core-violating alternatives. Those
need the kernel. Do not read this file as the Stage-1 verdict.

This census can genuinely FAIL. If Bemis-Murcko scaffolds are systematically
above the frozen upper fraction band, the protected object is too large to
leave room to act, and the correct response is to report that — **not** to
widen the band.

Usage
-----
    python3 scripts/constraints_hard_scaffold_census.py \
        --cohort diagnostics/retarget_calibration_cohort.json \
        --out diagnostics/constraints_hard_scaffold_census.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ARTIFACT_STATUS = "SMOKE_HELD_IN"
PROTECTED_OBJECT_RULE = "atom_and_bond_labeled_bemis_murcko_scaffold_v1"

# ---------------------------------------------------------------------------
# Frozen eligibility thresholds.
#
# REUSED VERBATIM from Lane 2, `src/compose_v4/experiments/pathwise_constraints.py`
# (branch codex/compose-pathwise-constraints, tip bcc4a40). Per the project rule
# "reuse a frozen threshold; do not mint one from the result", these are NOT
# re-derived here and MUST NOT be widened to make this census pass.
#
# Note the protected object differs: Lane 2 protects the largest fused ring
# system (`MOTIF_RULE_VERSION = "largest_ring_system_v1"`); this lane protects
# the Bemis-Murcko scaffold, which is a SUPERSET (ring systems plus the linkers
# joining them). Applying Lane 2's band to a strictly larger object is the
# conservative direction and is declared as a reuse, not a new rule.
# ---------------------------------------------------------------------------
MIN_CORE_ATOMS = 6  # Lane 2 MIN_MOTIF_ATOMS
CORE_FRACTION_BAND = (0.20, 0.70)  # Lane 2 MOTIF_FRACTION_BAND
MIN_FREE_ATOMS = 8  # Lane 2 MIN_FREE_ATOMS
HEAVY_ATOM_BAND = (18, 38)  # Lane 2 HEAVY_ATOM_BAND

THRESHOLD_PROVENANCE = {
    "MIN_CORE_ATOMS": "Lane 2 pathwise_constraints.MIN_MOTIF_ATOMS, reused verbatim",
    "CORE_FRACTION_BAND": "Lane 2 pathwise_constraints.MOTIF_FRACTION_BAND, reused verbatim",
    "MIN_FREE_ATOMS": "Lane 2 pathwise_constraints.MIN_FREE_ATOMS, reused verbatim",
    "HEAVY_ATOM_BAND": "Lane 2 pathwise_constraints.HEAVY_ATOM_BAND, reused verbatim",
}


def murcko_atom_indices(mol: Chem.Mol) -> frozenset[int]:
    """Bemis-Murcko scaffold as atom indices **in the parent molecule**.

    Parent indices are what we need, not a detached scaffold molecule: the
    exact-label SMARTS must be written from the PARENT's atom labels, because
    `GetScaffoldForMol` returns a new molecule whose H counts and sometimes
    aromaticity perception differ from the parent at the attachment points.

    Three rules, matching RDKit's `MurckoDecompose` convention:

    1. every ring atom is kept;
    2. linker atoms are kept - obtained by iteratively pruning terminal
       non-ring atoms until a fixpoint;
    3. **exocyclic atoms joined to a kept atom by a double bond are kept.**

    Rule 3 is not optional and was the bug this function's cross-check caught:
    omitting it drops ring carbonyl oxygens, disagreeing with RDKit on 16 of 30
    held-in sources. `scaffold_parent_index_crosscheck` in the census output is
    the regression guard - it compares this count against
    `GetScaffoldForMol(mol).GetNumHeavyAtoms()` on every source.
    """
    ring_info = mol.GetRingInfo()
    if not any(ring_info.NumAtomRings(a.GetIdx()) > 0 for a in mol.GetAtoms()):
        return frozenset()

    keep = {a.GetIdx() for a in mol.GetAtoms() if ring_info.NumAtomRings(a.GetIdx()) > 0}

    # Rules 1 + 2: prune terminal non-ring atoms to a fixpoint.
    alive = {a.GetIdx() for a in mol.GetAtoms()}
    changed = True
    while changed:
        changed = False
        for idx in sorted(alive):
            if idx in keep:
                continue
            degree = sum(
                1
                for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
                if nbr.GetIdx() in alive
            )
            if degree <= 1:
                alive.discard(idx)
                changed = True

    # Rule 3: re-attach exocyclic double-bonded atoms.
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in alive and j not in alive:
            alive.add(j)
        elif j in alive and i not in alive:
            alive.add(i)

    return frozenset(alive)


def core_bond_indices(mol: Chem.Mol, atoms: frozenset[int]) -> frozenset[int]:
    return frozenset(
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms
    )


def census_one(record: dict[str, Any]) -> dict[str, Any]:
    smiles = str(record["source"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"source": smiles, "eligible": False, "reason": "unparseable"}

    heavy = mol.GetNumHeavyAtoms()
    core = murcko_atom_indices(mol)
    core_bonds = core_bond_indices(mol, core)
    n_core = len(core)
    free_atoms = heavy - n_core
    free_bonds = mol.GetNumBonds() - len(core_bonds)
    fraction = (n_core / heavy) if heavy else 0.0

    # Cross-check against RDKit's own scaffold, as a bug detector on the
    # parent-index computation. A disagreement is recorded, never smoothed.
    try:
        rd_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        rd_n = rd_scaffold.GetNumHeavyAtoms() if rd_scaffold is not None else 0
    except Exception:  # noqa: BLE001
        rd_n = -1
    scaffold_agrees = rd_n == n_core

    # How much larger is the Bemis-Murcko core than Lane 2's protected motif
    # (largest fused ring system)? Recorded so the two lanes' constraints can be
    # compared without re-running either.
    ring_info = mol.GetRingInfo()
    systems: list[set[int]] = []
    for ring in ring_info.AtomRings():
        cur = set(ring)
        merged = [cur]
        for s in systems:
            if s & cur:
                merged.append(s)
        new = set().union(*merged)
        systems = [s for s in systems if not (s & cur)] + [new]
    largest_ring_system = max((len(s) for s in systems), default=0)

    reasons: list[str] = []
    if n_core == 0:
        reasons.append("empty_scaffold_acyclic_source")
    if n_core < MIN_CORE_ATOMS:
        reasons.append("core_too_small")
    if not (CORE_FRACTION_BAND[0] <= fraction <= CORE_FRACTION_BAND[1]):
        reasons.append("core_fraction_outside_band")
    if free_atoms < MIN_FREE_ATOMS:
        reasons.append("too_few_editable_atoms_outside_core")
    if not (HEAVY_ATOM_BAND[0] <= heavy <= HEAVY_ATOM_BAND[1]):
        reasons.append("heavy_atoms_outside_band")

    return {
        "source": smiles,
        "heavy_atoms": heavy,
        "core_atoms": n_core,
        "core_bonds": len(core_bonds),
        "core_fraction": round(fraction, 4),
        "free_atoms": free_atoms,
        "free_bonds": free_bonds,
        "largest_ring_system_atoms": largest_ring_system,
        "core_minus_largest_ring_system": n_core - largest_ring_system,
        "rdkit_scaffold_heavy_atoms": rd_n,
        "scaffold_parent_index_crosscheck": scaffold_agrees,
        "eligible": not reasons,
        "reasons": reasons,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stat(key: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        vals = sorted(r[key] for r in subset if key in r)
        if not vals:
            return {"n": 0}
        mid = len(vals) // 2
        median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
        return {
            "n": len(vals),
            "mean": round(sum(vals) / len(vals), 4),
            "median": round(median, 4),
            "min": vals[0],
            "max": vals[-1],
        }

    parsed = [r for r in rows if "heavy_atoms" in r]
    eligible = [r for r in parsed if r["eligible"]]
    reason_counts: dict[str, int] = {}
    for r in parsed:
        for reason in r.get("reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    crosscheck_failures = [
        r["source"] for r in parsed if not r["scaffold_parent_index_crosscheck"]
    ]

    return {
        "sources": len(rows),
        "parsed": len(parsed),
        "eligible": len(eligible),
        "eligible_fraction": round(len(eligible) / len(parsed), 4) if parsed else 0.0,
        "ineligibility_reasons": dict(sorted(reason_counts.items())),
        "core_fraction_all": stat("core_fraction", parsed),
        "core_fraction_eligible": stat("core_fraction", eligible),
        "core_atoms_all": stat("core_atoms", parsed),
        "free_atoms_all": stat("free_atoms", parsed),
        "free_atoms_eligible": stat("free_atoms", eligible),
        "core_minus_largest_ring_system": stat("core_minus_largest_ring_system", parsed),
        "scaffold_parent_index_crosscheck_failures": crosscheck_failures,
    }


def sweep_pool(pool_path: Path) -> dict[str, Any]:
    """Applicability of the frozen protected-object rule across the held-in pool.

    HELD-IN ONLY. Reads `training_source_keys` and never touches
    `reserve_source_keys`; the held-out reserve is not opened by this script.
    """
    import gzip
    import statistics as st
    from collections import Counter

    with gzip.open(pool_path, "rt") as handle:
        pool = json.load(handle)
    if "reserve_source_keys" in pool:
        del pool["reserve_source_keys"]  # structurally refuse to look
    keys = pool["training_source_keys"]

    reasons: Counter[str] = Counter()
    eligible = 0
    fractions: list[float] = []
    frees: list[int] = []
    for smiles in keys:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            reasons["unparseable"] += 1
            continue
        heavy = mol.GetNumHeavyAtoms()
        n_core = len(murcko_atom_indices(mol))
        free = heavy - n_core
        fraction = (n_core / heavy) if heavy else 0.0
        fractions.append(fraction)
        frees.append(free)
        local: list[str] = []
        if n_core == 0:
            local.append("empty_scaffold_acyclic_source")
        if n_core < MIN_CORE_ATOMS:
            local.append("core_too_small")
        if not (CORE_FRACTION_BAND[0] <= fraction <= CORE_FRACTION_BAND[1]):
            local.append("core_fraction_outside_band")
        if free < MIN_FREE_ATOMS:
            local.append("too_few_editable_atoms_outside_core")
        if not (HEAVY_ATOM_BAND[0] <= heavy <= HEAVY_ATOM_BAND[1]):
            local.append("heavy_atoms_outside_band")
        if not local:
            eligible += 1
        reasons.update(local)

    parsed = len(fractions)
    return {
        "pool_path": str(pool_path),
        "pool_key": "training_source_keys",
        "held_out_opened": False,
        "sources": len(keys),
        "parsed": parsed,
        "eligible": eligible,
        "eligible_fraction": round(eligible / parsed, 4) if parsed else 0.0,
        "core_fraction_median": round(st.median(fractions), 4) if fractions else None,
        "core_fraction_mean": round(st.mean(fractions), 4) if fractions else None,
        "free_atoms_median": st.median(frees) if frees else None,
        "free_atoms_mean": round(st.mean(frees), 4) if frees else None,
        "ineligibility_reasons_not_mutually_exclusive": {
            key: {"count": value, "fraction": round(value / parsed, 4)}
            for key, value in reasons.most_common()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="optional held-in pool .json.gz; adds a whole-pool applicability sweep",
    )
    args = parser.parse_args()

    payload = json.loads(args.cohort.read_text())
    rows = [census_one(rec) for rec in payload["sources"]]
    summary = summarize(rows)
    pool_sweep = sweep_pool(args.pool) if args.pool else None

    cohort_digest = hashlib.sha256(args.cohort.read_bytes()).hexdigest()
    result = {
        "schema": "compose.constraints_hard.scaffold_census",
        "schema_version": 1,
        "status": ARTIFACT_STATUS,
        "held_out_opened": False,
        "scope": (
            "MODEL-FREE ONLY. No kernel, no R_theta, no oracle, no Gate-0. "
            "Establishes nothing about the successor fiber."
        ),
        "protected_object_rule": PROTECTED_OBJECT_RULE,
        "cohort_path": str(args.cohort),
        "cohort_file_sha256": cohort_digest,
        "cohort_declared_sha256": payload.get("cohort_sha256"),
        "cohort_status": payload.get("status"),
        "thresholds": {
            "MIN_CORE_ATOMS": MIN_CORE_ATOMS,
            "CORE_FRACTION_BAND": list(CORE_FRACTION_BAND),
            "MIN_FREE_ATOMS": MIN_FREE_ATOMS,
            "HEAVY_ATOM_BAND": list(HEAVY_ATOM_BAND),
        },
        "threshold_provenance": THRESHOLD_PROVENANCE,
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "rdkit_pin_matches_production": Chem.rdBase.rdkitVersion == "2024.03.5",
        "summary": summary,
        "held_in_pool_applicability": pool_sweep,
        "per_source": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    if pool_sweep is not None:
        print("\n--- held-in pool applicability ---")
        print(json.dumps(pool_sweep, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
