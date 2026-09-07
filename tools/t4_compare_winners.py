"""Describe the locked T4 audit molecules against the existing IVG snapshot.

Development diagnostic only: no generation, inference guidance, docking, or
parameter selection. The output is a structural comparison, not a causal edit
prescription or a verified executor path. Ring semantics come from COMPOSE's
existing taxonomy and macro machinery. SMILES are endpoint analysis inputs only.

Run: PYTHONPATH=src .venv/bin/python tools/t4_compare_winners.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, Draw, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.macro_engine import ring_systems
from compose_v4.eval.ring_taxonomy import ring_taxonomy_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/t4_winner_comparison"
INPUTS = (
    "diagnostics/t4_three_level_option_audit.json",
    "diagnostics/ivg_winners.json",
    "diagnostics/ivg_molecules/docking_parp1_idx0_thr4.csv",
    "diagnostics/t4_oracle_calibration.json",
    "diagnostics/pmo_puct_tournament.json",
    "diagnostics/pmo_bo_tournament.json",
    "src/compose_v4/eval/ring_taxonomy.py",
    "src/compose_v4/control/macro_engine.py",
    "modal_apps/pmo_control_app.py",
    "modal_apps/genmol_t4_opt_app.py",
    "tools/t4_compare_winners.py",
)


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(smiles: str, seed_fp, delta: float) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid diagnostic molecule: {smiles!r}")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    taxonomy = ring_taxonomy_report((smiles,))
    systems = []
    for group in ring_systems(mol):
        atoms = sorted(set().union(*group))
        systems.append(
            {
                "ring_sizes": sorted(len(ring) for ring in group),
                "ring_atoms": len(atoms),
                "ring_elements": dict(
                    sorted(Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in atoms).items())
                ),
                "ring_fragment_smiles": Chem.MolFragmentToSmiles(mol, atomsToUse=atoms),
            }
        )
    qed = float(QED.qed(mol))
    sa = float(sascorer.calculateScore(mol))
    sim = float(DataStructs.TanimotoSimilarity(seed_fp, fp.GetFingerprint(mol)))
    return {
        "canonical_smiles": Chem.MolToSmiles(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
        "elements": dict(sorted(Counter(a.GetSymbol() for a in mol.GetAtoms()).items())),
        "cycle_rank": int(taxonomy["means"]["cycle_rank"]),
        "ring_system_count": len(systems),
        "ring_sizes": sorted(len(r) for r in mol.GetRingInfo().AtomRings()),
        "aromatic_rings": taxonomy["ring_class_counts"]["aromatic"]["count"],
        "ring_systems": sorted(systems, key=lambda x: (x["ring_atoms"], x["ring_fragment_smiles"])),
        "qed": qed,
        "sa": sa,
        "similarity_to_seed": sim,
        "feasible": qed >= 0.6 and sa <= 4.0 and sim >= delta,
    }


def main() -> None:
    audit = read_json(INPUTS[0])
    if audit["cell"] != "compose_iclr_macro_prior_parp1_0_d0.4" or not audit["complete"]:
        raise ValueError("expected completed locked parp1 seed0 delta0.4 audit")
    winners = read_json(INPUTS[1])["parp1_s0_d0.4"]["winners"]
    with (ROOT / INPUTS[2]).open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    calibration = next(
        row for row in read_json(INPUTS[3]) if row["target"] == "parp1" and row["idx"] == 0
    )
    records = [
        {"id": "seed", "smiles": audit["seed"], "score": None},
        {"id": "compose_best_feasible", "smiles": audit["best_smiles"], "score": audit["best_ds"]},
    ]
    for index, winner in enumerate(winners):
        matches = [
            i + 2
            for i, row in enumerate(source_rows)
            if row["smiles"] == winner["smiles"]
            and abs(float(row["docking score"]) + winner["ds"]) < 1e-8
        ]
        if not matches:
            raise ValueError(f"winner {index} absent from source CSV at its recorded score")
        records.append(
            {
                "id": f"ivg_snapshot_{index + 1}",
                "smiles": winner["smiles"],
                "score": winner["ds"],
                "source_csv_lines": matches,
                "reported_constraints": {k: winner[k] for k in ("qed", "sa", "sim")},
                "historical_redock_scores": [
                    row["ours"] for row in calibration["pairs"] if row["smiles"] == winner["smiles"]
                ],
            }
        )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(audit["seed"]))
    for row in records:
        row["structure"] = describe(row["smiles"], seed_fp, audit["delta"])
        if row["id"].startswith("ivg_"):
            for old, new in (("qed", "qed"), ("sa", "sa"), ("sim", "similarity_to_seed")):
                if abs(row["reported_constraints"][old] - row["structure"][new]) > 1e-6:
                    raise ValueError(f"constraint mismatch for {row['id']}: {old}")
    puct_runs = read_json(INPUTS[4])["runs"]
    report = {
        "schema_version": "t4_winner_structural_comparison_v1",
        "evidence_role": "development-only endpoint analysis; no causal intervention",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": {p: sha256(ROOT / p) for p in INPUTS},
        "sa_score_data_sha256": sha256(Path(sascorer.__file__).parent / "fpscores.pkl.gz"),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": platform.machine(),
        "configuration": {
            "cell": audit["cell"],
            "comparison_selection": "audit best feasible vs all five existing snapshot winners",
            "fingerprint": {"radius": 2, "bits": 2048},
            "qed_min": 0.6,
            "sa_max": 4.0,
            "similarity_min": audit["delta"],
            "randomness": "none in structural analysis; 2D depiction is illustrative",
            "precision": "integer topology counts; float64 descriptors",
            "exclusions": [],
            "source_csv_row_count": len(source_rows),
            "additional_oracle_calls": 0,
        },
        "limitations": [
            "The existing winner snapshot is not the three-run paper aggregate.",
            "Source CSV identity is hashed locally; upstream revision/license lineage is not embedded in the historical snapshot.",
            "Historical redocking lacks the current audit's full input-hash provenance; scores are reported context only.",
            "Ring fragments are descriptive views, not proposals or executor paths.",
            "Endpoints do not establish reachability, a valid path, causal docking improvement, or pose contacts.",
            "The inspected cell is development evidence; final generalization needs a separate untouched panel.",
        ],
        "records": records,
        "historical_mcts_inventory": {
            "task_count": len({r["task"] for r in puct_runs}),
            "run_count": len(puct_runs),
            "budgets": sorted({r["budget"] for r in puct_runs}),
            "runs_with_ledger_call_mismatch": sum(
                r["ledger_n"] != r["calls_spent"] for r in puct_runs
            ),
            "completed_bo_runs_in_local_tournament": len(read_json(INPUTS[5])["runs"]),
            "interpretation": "existence evidence only; not used to rank search algorithms",
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if json.loads(payload) != report:
        raise RuntimeError("comparison JSON round-trip failed")
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / "comparison.json.tmp"
    temporary.write_text(payload)
    temporary.replace(OUT / "comparison.json")
    legends = []
    for row in records:
        d = row["structure"]
        score = "seed" if row["score"] is None else f"recorded DS {row['score']}"
        legends.append(
            f"{row['id']} | {score}\n{d['heavy_atoms']} atoms, cycle rank {d['cycle_rank']}, {d['ring_system_count']} systems"
        )
        print(
            row["id"],
            row["score"],
            {
                k: d[k]
                for k in (
                    "heavy_atoms",
                    "cycle_rank",
                    "ring_system_count",
                    "ring_sizes",
                    "aromatic_rings",
                    "qed",
                    "sa",
                    "similarity_to_seed",
                )
            },
        )
    grid = Draw.MolsToGridImage(
        [Chem.MolFromSmiles(row["smiles"]) for row in records],
        molsPerRow=3,
        subImgSize=(500, 360),
        legends=legends,
    )
    grid.save(OUT / "structures.png")
    print("historical_mcts_inventory", report["historical_mcts_inventory"])
    print("saved", OUT / "comparison.json")


if __name__ == "__main__":
    main()
