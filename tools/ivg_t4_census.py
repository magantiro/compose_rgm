"""Development-only structural census of every released similarity-constrained T4 cell.

No training, generator guidance, fragment extraction, or oracle calls. Downloaded
sources stay in the ignored cache, with immutable URLs and hashes in the report.
Run with PYTHONPATH=src .venv/bin/python tools/ivg_t4_census.py --download.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.eval.ring_taxonomy import ring_taxonomy_report

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb"
REPO = "invirtuolabs/InVirtuoGen_results"
CSV_DIR = "results/lead_optimization/sim_constraint"
SELECTOR = "in_virtuo_reinforce/evaluation/results_table_lead.py"
TARGETS = ("5ht1b", "braf", "fa7", "jak2", "parp1")
EXPECTED = tuple(
    f"{CSV_DIR}/docking_{target}_idx{idx}_thr{thr}.csv"
    for target in TARGETS
    for idx in range(3)
    for thr in (4, 6)
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def source_inventory(cache: Path, download: bool) -> list[dict]:
    tree_path = cache / "tree.json"
    tree_url = f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"
    if download and not tree_path.exists():
        publish(tree_path, fetch(tree_url))
    tree = json.loads(tree_path.read_text())
    if tree["sha"] != REVISION or tree["truncated"]:
        raise ValueError(f"{tree_path}: expected complete tree for {REVISION}")
    entries = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
    actual = sorted(p for p in entries if p.startswith(CSV_DIR + "/") and p.endswith(".csv"))
    if actual != sorted(EXPECTED):
        raise ValueError(f"{tree_path}: unexpected T4 cell inventory: {actual}")

    def obtain(relative: str) -> dict:
        path = cache / relative
        url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{relative}"
        if download and not path.exists():
            publish(path, fetch(url))
        payload = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if blob != entries[relative]["sha"]:
            raise ValueError(f"{path}: upstream Git blob mismatch")
        return {"path": relative, "url": url, "git_blob": blob, "sha256": sha256(path)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(obtain, (*EXPECTED, SELECTOR, "LICENSE")))
    return sorted(manifest, key=lambda row: row["path"])


def select_winners(rows: list[dict[str, str]], delta: float, source: str) -> dict:
    """Match upstream strict filtering and per-run maximum; preserve all co-best ties."""
    groups: dict[int, list[dict]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for line, row in enumerate(rows, 2):
        try:
            seed = int(row["seed"])
            values = {key: float(row[key]) for key in ("docking score", "qed", "sa", "sim")}
            smiles = row["smiles"]
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f"{source}:{line}: malformed required field: {row}") from error
        if not smiles or not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{source}:{line}: empty SMILES or nonfinite score: {row}")
        groups[seed]  # Preserve runs with no feasible row.
        reasons = []
        if values["qed"] <= 0.6:
            reasons.append("qed_not_gt_0.6")
        if values["sa"] >= 4:
            reasons.append("sa_not_lt_4")
        if values["sim"] <= delta:
            reasons.append("sim_not_gt_delta")
        if reasons:
            rejected["+".join(reasons)] += 1
        else:
            groups[seed].append({"line": line, "smiles": smiles, **values})
    runs = []
    for seed, feasible in sorted(groups.items()):
        best = max((row["docking score"] for row in feasible), default=None)
        distinct = {}
        for row in feasible:
            if row["docking score"] != best:
                continue
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                raise ValueError(f"{source}:{row['line']}: invalid selected SMILES")
            canonical = Chem.MolToSmiles(mol)
            if canonical not in distinct:
                distinct[canonical] = {"canonical_smiles": canonical, "source_rows": []}
            distinct[canonical]["source_rows"].append(row)
        runs.append(
            {
                "run_seed": seed,
                "feasible_rows": len(feasible),
                "reported_docking_score": -best if best is not None else None,
                "winners": [distinct[key] for key in sorted(distinct)],
            }
        )
    return {
        "row_count": len(rows),
        "rejection_reason_counts": dict(sorted(rejected.items())),
        "runs": runs,
    }


def structure(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid endpoint SMILES: {smiles}")
    taxonomy = ring_taxonomy_report((smiles,))
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
        "has_specified_atom_or_bond_stereo": any(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()
        )
        or any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()),
        "elements": dict(sorted(Counter(atom.GetSymbol() for atom in mol.GetAtoms()).items())),
        "cycle_rank": int(taxonomy["means"]["cycle_rank"]),
        "ring_system_count": int(taxonomy["means"]["ring_systems"]),
        "aromatic_rings": taxonomy["ring_class_counts"]["aromatic"]["count"],
        "ring_sizes": sorted(len(ring) for ring in mol.GetRingInfo().AtomRings()),
        "taxonomy": taxonomy,
    }


def summarize(records: list[dict]) -> dict:
    # One vote per distinct canonical endpoint, not per CSV duplicate or plateau step.
    unique = {record["canonical_smiles"]: record["structure"] for record in records}
    if not unique:
        return {"unique_structures": 0}
    taxonomy = ring_taxonomy_report(tuple(sorted(unique)))
    return {
        "cell_run_structure_records": len(records),
        "unique_structures": len(unique),
        "taxonomy": taxonomy,
        "multiple_ring_systems": sum(d["ring_system_count"] >= 2 for d in unique.values()),
        "above_40_heavy_atoms": sum(d["heavy_atoms"] > 40 for d in unique.values()),
        "nonzero_formal_charge": sum(d["formal_charge"] != 0 for d in unique.values()),
        "specified_stereochemistry": sum(
            d["has_specified_atom_or_bond_stereo"] for d in unique.values()
        ),
        "cycle_rank_counts": dict(
            sorted(Counter(str(d["cycle_rank"]) for d in unique.values()).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/ivg_t4" / REVISION)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "diagnostics/ivg_t4_census/census.json"
    )
    args = parser.parse_args()
    sources = source_inventory(args.cache, args.download)
    seeds_path = ROOT / "docs/GENMOL_T4_SEEDS.json"
    seeds = json.loads(seeds_path.read_text())
    by_target = {target: [row for row in seeds if row["target"] == target] for target in TARGETS}
    if any(len(rows) != 3 for rows in by_target.values()):
        raise ValueError(f"{seeds_path}: expected three starting molecules per target")
    cells, all_winners = [], []
    fingerprints = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for relative in EXPECTED:
        filename = Path(relative).stem
        _, target, idx_text, thr_text = filename.split("_")
        idx, delta = int(idx_text[3:]), int(thr_text[3:]) / 10
        starting = by_target[target][idx]
        initial = structure(starting["smiles"])
        initial_fp = fingerprints.GetFingerprint(Chem.MolFromSmiles(starting["smiles"]))
        with (args.cache / relative).open(newline="") as handle:
            cell = select_winners(list(csv.DictReader(handle)), delta, relative)
        cell.update(
            {
                "cell": filename,
                "target": target,
                "source_idx": idx,
                "delta": delta,
                "source_smiles": starting["smiles"],
                "source_structure": initial,
            }
        )
        for run in cell["runs"]:
            for winner in run["winners"]:
                winner["structure"] = structure(winner["canonical_smiles"])
                winner["recomputed_similarity_to_registered_source"] = float(
                    DataStructs.TanimotoSimilarity(
                        initial_fp,
                        fingerprints.GetFingerprint(Chem.MolFromSmiles(winner["canonical_smiles"])),
                    )
                )
                winner["endpoint_delta_from_source"] = {
                    key: winner["structure"][key] - initial[key]
                    for key in (
                        "heavy_atoms",
                        "cycle_rank",
                        "ring_system_count",
                        "aromatic_rings",
                        "formal_charge",
                    )
                }
                all_winners.append({"target": target, **winner})
        cell["max_reported_similarity_discrepancy"] = max(
            (
                abs(row["sim"] - winner["recomputed_similarity_to_registered_source"])
                for run in cell["runs"]
                for winner in run["winners"]
                for row in winner["source_rows"]
            ),
            default=None,
        )
        cells.append(cell)
    dependencies = (
        "tools/ivg_t4_census.py",
        "src/compose_v4/eval/ring_taxonomy.py",
        "docs/GENMOL_T4_SEEDS.json",
    )
    report = {
        "schema_version": "ivg_t4_endpoint_census_v1",
        "evidence_role": "computed development-only endpoint census; not prospective validation",
        "upstream_revision": REVISION,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_and_input_sha256": {p: sha256(ROOT / p) for p in dependencies},
        "upstream_sources": sources,
        "inventory_sha256": sha256(args.cache / "tree.json"),
        "configuration": {
            "selection": "upstream strict sim > delta, qed > 0.6, sa < 4; max docking score per seed",
            "ties": "all canonically distinct co-best endpoints, with every duplicate source row",
            "aggregate_weighting": "unique canonical endpoint; not an equal-run or population estimate",
            "seed_mapping": "target-local file idx indexes target-filtered GENMOL_T4_SEEDS registry",
            "source_mapping_check": "compare reported winner similarity with Morgan r2/2048 Tanimoto; not used for winner selection",
            "ring_basis": "RDKit symmetrized SSSR; cycle rank reported separately",
            "randomness": "none; no random seed or stochastic computation",
            "split": "all inspected cells are development evidence, not an untouched final holdout",
            "additional_oracle_calls": 0,
            "precision": "integer topology counts; float64 reported scores",
        },
        "access_basis": {
            "origin": f"https://github.com/{REPO}",
            "use": "public release, research-only endpoint inspection; no commercial permission inferred",
            "license_notice": "LICENSE has NC-SA link/header but NC body; preserve exact notice",
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": "CPU",
        },
        "census_counts": {
            "cells": len(cells),
            "rows": sum(cell["row_count"] for cell in cells),
            "observed_runs": sum(len(cell["runs"]) for cell in cells),
            "runs_without_feasible_winner": sum(
                not run["winners"] for cell in cells for run in cell["runs"]
            ),
            "co_best_source_rows": sum(len(w["source_rows"]) for w in all_winners),
        },
        "all_targets": summarize(all_winners),
        "by_target": {t: summarize([w for w in all_winners if w["target"] == t]) for t in TARGETS},
        "cells": cells,
        "limitations": [
            "Only similarity-constrained T4; unconstrained T4 and other tasks are not pooled.",
            "Scores and feasibility are reported by IVG; no redocking or oracle-parity claim.",
            "Endpoint deltas do not establish edit histories, reachability, or causal docking effects.",
            "Ring-taxonomy labels overlap; fused includes shared-atom cases also labelled bridged.",
            "40-atom, charge, and stereo flags are partial support checks, not executor admission.",
            "Canonical identity retains specified stereochemistry; no 3D or stereo-generation claim.",
            "No winner fragments, frequencies, or labels are fed to the controller or a learned model.",
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if json.loads(payload) != report:
        raise RuntimeError("census serialization round-trip failed")
    publish(args.output, payload.encode())
    print(
        json.dumps(
            {
                "counts": report["census_counts"],
                "all_targets": report["all_targets"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
