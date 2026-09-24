"""Locate benchmark-visible fragment misses in the frozen qualification capture.

This is an exploratory integrity audit of previously held prompts, not a model
selection gate. Any repair informed by these molecules needs a fresh held test.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS
from run_fragment_constrained_suite import MANIFEST, audit_queries, contains_all_fragments

from compose_v4.benchmark.fragment_constrained import load_genmol_prompts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/fragment_single_interface_qualification_v1/quality_denominators.json"
OUTPUT = ROOT / "diagnostics/fragment_rdkit_preservation_audit_v1/result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bond_agnostic_overlap(query: Chem.Mol, product: Chem.Mol) -> dict:
    match = rdFMCS.FindMCS(
        [query, product],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        timeout=5,
    )
    result = {
        "query_atoms": query.GetNumAtoms(),
        "query_bonds": query.GetNumBonds(),
        "mcs_atoms": match.numAtoms,
        "mcs_bonds": match.numBonds,
        "canceled": bool(match.canceled),
    }
    if (
        match.canceled
        or match.numAtoms != query.GetNumAtoms()
        or match.numBonds != query.GetNumBonds()
    ):
        return result
    pattern = Chem.MolFromSmarts(match.smartsString)
    query_match = query.GetSubstructMatch(pattern)
    product_matches = product.GetSubstructMatches(
        pattern, uniquify=False, useChirality=False, maxMatches=1000
    )
    if not query_match or not product_matches:
        raise RuntimeError("full bond-agnostic MCS did not embed in both molecules")
    candidates = []
    for product_match in product_matches:
        mapped = dict(zip(query_match, product_match, strict=True))
        atom_differences = []
        bond_differences = []
        for source_index, target_index in mapped.items():
            source_atom = query.GetAtomWithIdx(source_index)
            target_atom = product.GetAtomWithIdx(target_index)
            before = (source_atom.GetFormalCharge(), source_atom.GetIsAromatic())
            after = (target_atom.GetFormalCharge(), target_atom.GetIsAromatic())
            if before != after:
                atom_differences.append(
                    {
                        "query_atom": source_index,
                        "product_atom": target_index,
                        "element": source_atom.GetSymbol(),
                        "query_charge_aromatic": before,
                        "product_charge_aromatic": after,
                    }
                )
        for source_bond in query.GetBonds():
            target_bond = product.GetBondBetweenAtoms(
                mapped[source_bond.GetBeginAtomIdx()],
                mapped[source_bond.GetEndAtomIdx()],
            )
            if target_bond is None:
                raise RuntimeError("MCS mapping lost an internal query bond")
            before = (str(source_bond.GetBondType()), source_bond.GetIsAromatic())
            after = (str(target_bond.GetBondType()), target_bond.GetIsAromatic())
            if before != after:
                bond_differences.append(
                    {
                        "query_atoms": [
                            source_bond.GetBeginAtomIdx(),
                            source_bond.GetEndAtomIdx(),
                        ],
                        "product_atoms": [
                            target_bond.GetBeginAtomIdx(),
                            target_bond.GetEndAtomIdx(),
                        ],
                        "query_bond_state": before,
                        "product_bond_state": after,
                    }
                )
        candidates.append(
            (len(atom_differences) + len(bond_differences), atom_differences, bond_differences)
        )
    _, atom_differences, bond_differences = min(candidates, key=lambda row: row[0])
    result["atom_state_differences"] = atom_differences
    result["bond_state_differences"] = bond_differences
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    source = json.loads(SOURCE.read_text())
    prompts = {
        (prompt.task.value, prompt.drug_name): prompt for prompt in load_genmol_prompts(MANIFEST)
    }
    units = []
    misses = []
    for relative_path, expected_hash in sorted(source["unit_sha256"].items()):
        path = ROOT / relative_path
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"frozen unit hash mismatch: {relative_path}")
        unit = json.loads(path.read_text())
        task, drug = unit["task"], unit["drug"]
        prompt = prompts[(task, drug)]
        queries = audit_queries(prompt)
        rows = unit["result"]["per_drug"][drug]
        if len(rows) != 1:
            raise RuntimeError(f"expected one row for {relative_path}")
        row = rows[0]
        committed = [a for a in row["attempt_records"] if a["committed_smiles"]]
        observed = sum(contains_all_fragments(a["committed_smiles"], queries) for a in committed)
        if observed != row["committed_fragment_preserving"]:
            raise RuntimeError(f"preservation count drift in {relative_path}")
        units.append(
            {
                "path": relative_path,
                "sha256": actual_hash,
                "attempts": row["attempts"],
                "committed": len(committed),
                "rdkit_preserving": observed,
            }
        )
        for attempt in committed:
            smiles = attempt["committed_smiles"]
            if contains_all_fragments(smiles, queries):
                continue
            product = Chem.MolFromSmiles(smiles)
            if product is None:
                raise RuntimeError(f"committed SMILES did not parse: {relative_path}")
            fragment_checks = []
            for fragment, query in zip(prompt.fragments, queries, strict=True):
                exact = product.GetSubstructMatches(
                    query, uniquify=False, useChirality=False, maxMatches=1000
                )
                fragment_checks.append(
                    {
                        "prompt_fragment": fragment,
                        "exact_matches": len(exact),
                        "bond_agnostic_overlap": _bond_agnostic_overlap(query, product),
                    }
                )
            misses.append(
                {
                    "unit": relative_path,
                    "arm": unit["arm"],
                    "task": task,
                    "drug": drug,
                    "attempt_index": attempt["attempt_index"],
                    "committed_smiles": smiles,
                    "fragments": fragment_checks,
                }
            )

    result = {
        "schema_version": "fragment_rdkit_preservation_audit_v1",
        "evidence_role": (
            "Exploratory diagnosis of previously held prompts. Do not select or "
            "qualify a repaired controller on these same examples."
        ),
        "source_sha256": _sha256(SOURCE),
        "manifest_sha256": _sha256(MANIFEST),
        "auditor_sha256": _sha256(Path(__file__)),
        "rdkit_version": rdBase.rdkitVersion,
        "units": units,
        "misses": misses,
        "totals": {
            "units": len(units),
            "attempts": sum(unit["attempts"] for unit in units),
            "committed": sum(unit["committed"] for unit in units),
            "rdkit_preserving": sum(unit["rdkit_preserving"] for unit in units),
            "nonpreserving": len(misses),
        },
    }
    _atomic_json(OUTPUT, result)
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
