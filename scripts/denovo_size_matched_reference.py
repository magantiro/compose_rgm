"""Compare saved de novo molecules to deterministic size-matched GuacaMol rows."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

import rdkit
from denovo_postring_quality_audit import _grid, _group_summary, characterize, sha256
from rdkit import Chem

SCHEMA = "denovo_size_matched_reference_v1"
EXPECTED_TRAIN_SHA256 = "70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791"


def select_training_pool(
    path: Path, pool_size: int, allowed_elements: frozenset[str] | None = None
) -> dict[int, list[str]]:
    """Choose corpus rows without using any generated molecule or metric."""

    def rank(text: str) -> tuple[bytes, str]:
        return hashlib.blake2b(text.encode(), digest_size=8).digest(), text

    with path.open() as handle:
        ranked = heapq.nsmallest(pool_size, (rank(line.strip()) for line in handle if line.strip()))
    buckets: dict[int, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for _, text in ranked:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        if allowed_elements is not None and any(
            atom.GetSymbol() not in allowed_elements for atom in mol.GetAtoms()
        ):
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        buckets[mol.GetNumHeavyAtoms()].append(canonical)
    return buckets


def audit(
    records_path: Path,
    train_path: Path,
    output: Path,
    pool_size: int,
    per_generated: int,
    allowed_elements: frozenset[str] | None = None,
) -> dict:
    if sha256(train_path) != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"GuacaMol training SHA-256 mismatch: {train_path}")
    generated = json.loads(records_path.read_text())
    pool = select_training_pool(train_path, pool_size, allowed_elements)
    by_arm_size: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in generated:
        if row["group"] != "NO_ENDPOINT":
            by_arm_size[(row["arm"], row["heavy_atoms"])].append(row)
    matched: list[dict] = []
    missing = []
    for (arm, size), rows in sorted(by_arm_size.items()):
        candidates = pool.get(size, [])
        for position, row in enumerate(sorted(rows, key=lambda item: item["index"])):
            selected = candidates[position * per_generated : (position + 1) * per_generated]
            if len(selected) < per_generated:
                missing.append(
                    {
                        "arm": arm,
                        "index": row["index"],
                        "heavy_atoms": size,
                        "available": len(candidates),
                    }
                )
            for offset, smiles in enumerate(selected):
                pseudo = {
                    "smiles": smiles,
                    "index": row["index"],
                    "trajectory_seed": offset,
                    "events": 0,
                    "event_rules": [],
                    "valid_state": True,
                    "connected": True,
                }
                reference = characterize(pseudo, f"train_for_{arm}", train_path)
                if reference["heavy_atoms"] != size:
                    raise RuntimeError("size-matched training identity drift")
                reference["matched_generated_group"] = row["group"]
                reference["matched_generated_index"] = row["index"]
                matched.append(reference)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for arm in sorted({row["arm"] for row in generated}):
        for group in sorted({row["group"] for row in generated if row["arm"] == arm}):
            if group == "NO_ENDPOINT":
                continue
            ours = [row for row in generated if row["arm"] == arm and row["group"] == group]
            refs = [
                row
                for row in matched
                if row["arm"] == f"train_for_{arm}" and row["matched_generated_group"] == group
            ]
            grouped[arm][group] = {
                "generated": _group_summary(ours),
                "size_matched_train": _group_summary(refs),
            }
    output.mkdir(parents=True, exist_ok=True)
    grid = _grid(
        [row for row in matched if row["arm"] == "train_for_C1"],
        output / "c1_matched_train_random.png",
        maximum=100,
    )
    report = {
        "schema_version": SCHEMA,
        "evidence_class": "computed_exploratory_local_rdkit",
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256(Path(__file__)),
        "inputs": {
            "generated_records": {"path": str(records_path), "sha256": sha256(records_path)},
            "training_corpus": {
                "path": str(train_path),
                "sha256": sha256(train_path),
                "rows": 500000,
            },
        },
        "configuration": {
            "pool_size": pool_size,
            "matches_per_generated": per_generated,
            "pool_selection": "smallest BLAKE2b-8 rank of full SMILES",
            "allowed_elements": sorted(allowed_elements) if allowed_elements is not None else None,
        },
        "versions": {"python": platform.python_version(), "rdkit": rdkit.__version__},
        "pool_size_by_heavy_atoms": dict(sorted(pool.items(), key=lambda item: item[0])),
        "matched_reference_count": len(matched),
        "missing_matches": missing,
        "grid_molecule_count": grid,
        "group_comparison": grouped,
    }
    # Keep counts in the public report, not all 50,000 sampled training SMILES.
    report["pool_size_by_heavy_atoms"] = {str(key): len(value) for key, value in pool.items()}
    (output / "matched_reference_records.json").write_text(
        json.dumps(matched, indent=1, sort_keys=True) + "\n"
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=50000)
    parser.add_argument("--per-generated", type=int, default=3)
    parser.add_argument(
        "--allowed-elements",
        nargs="+",
        default=None,
        help="Restrict reference molecules to the generated representation's elements",
    )
    args = parser.parse_args()
    report = audit(
        args.records,
        args.train,
        args.output,
        args.pool_size,
        args.per_generated,
        frozenset(args.allowed_elements) if args.allowed_elements is not None else None,
    )
    print(
        json.dumps(
            {
                "matched_reference_count": report["matched_reference_count"],
                "missing_matches": report["missing_matches"],
                "group_comparison": report["group_comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
