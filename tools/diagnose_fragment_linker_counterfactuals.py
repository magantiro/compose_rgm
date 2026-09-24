"""Inspect locked linker outputs and fixed, manual connector replacements.

This is a post-generation diagnostic. The replacements are never presented as
sampler outputs or used to select a benchmark method. All fixed connectors are
reported, including refusals, with no QED/SA-based selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import QED, Descriptors, Draw, rdMolDescriptors

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    load_genmol_prompts,
    sascorer,
)
from compose_v4.benchmark.fragment_linker_assembly import (
    assemble_linker_program,
    linker_fidelity,
)

# Frozen before calculating any manual-connector scores. These are simple
# two-ended edits, not an optimized or chemically screened fragment panel.
MANUAL_CONNECTORS = (
    "[1*]C[2*]",
    "[1*]CC[2*]",
    "[1*]O[2*]",
    "[1*]N[2*]",
    "[1*]CO[2*]",
    "[1*]CNC[2*]",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sa_parts(mol: Chem.Mol) -> dict[str, float | int]:
    """Expose the published RDKit SA terms, asserting exact scorer agreement."""
    score = float(sascorer.calculateScore(mol))
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2).GetNonzeroElements()
    counts = sum(fp.values())
    fragment = sum(sascorer._fscores.get(k, -4.0) * n for k, n in fp.items()) / counts
    atoms = mol.GetNumAtoms()
    chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    macrocycles = sum(len(ring) > 8 for ring in mol.GetRingInfo().AtomRings())
    penalties = {
        "size": atoms**1.005 - atoms,
        "stereo": math.log10(chiral + 1),
        "bridgehead": math.log10(bridge + 1),
        "spiro": math.log10(spiro + 1),
        "macrocycle": math.log10(2) if macrocycles else 0.0,
    }
    density = math.log(atoms / len(fp)) * 0.5 if atoms > len(fp) else 0.0
    raw = fragment - sum(penalties.values()) + density
    reconstruction = 11.0 - (raw + 5.0) / 6.5 * 9.0
    if reconstruction > 8:
        reconstruction = 8.0 + math.log(reconstruction - 8.0)
    reconstruction = max(1.0, min(10.0, reconstruction))
    if abs(reconstruction - score) > 1e-10:
        raise AssertionError(f"SA component reconstruction drift: {reconstruction} != {score}")
    return {
        "sa": score,
        "fragment_mean": fragment,
        "density_bonus": density,
        "complexity_penalty": sum(penalties.values()),
        "size_penalty": penalties["size"],
        "stereo_penalty": penalties["stereo"],
        "bridgehead_penalty": penalties["bridgehead"],
        "spiro_penalty": penalties["spiro"],
        "macrocycle_penalty": penalties["macrocycle"],
        "chiral_centers": chiral,
        "bridgehead_atoms": bridge,
        "spiro_atoms": spiro,
    }


def _metrics(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"invalid/disconnected diagnostic molecule: {smiles}")
    qed = float(QED.qed(mol))
    parts = _sa_parts(mol)
    return {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "qed": qed,
        **parts,
        "qed_pass": qed >= 0.6,
        "sa_pass": parts["sa"] <= 4.0,
        "quality_pass": qed >= 0.6 and parts["sa"] <= 4.0,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "rings": rdMolDescriptors.CalcNumRings(mol),
    }


def _colors(fidelity: dict) -> dict[int, tuple[float, float, float]]:
    blue = (0.52, 0.72, 0.97)
    green = (0.53, 0.85, 0.66)
    orange = (1.0, 0.74, 0.42)
    result = {int(a): orange for a in fidelity["linker_path"][1:-1]}
    result.update({int(a): blue for a in fidelity["core_maps"][0]})
    result.update({int(a): green for a in fidelity["core_maps"][1]})
    return result


def _grid(rows: list[dict], path: Path, *, per_row: int = 4) -> None:
    mols = [Chem.MolFromSmiles(row["metrics"]["smiles"]) for row in rows]
    legends = [
        f"{row['label']}\nQED {row['metrics']['qed']:.3f}; "
        f"SA {row['metrics']['sa']:.3f}; linker {row['fidelity']['linker_internal_atoms']}"
        for row in rows
    ]
    colors = [_colors(row["fidelity"]) for row in rows]
    Draw.MolsToGridImage(
        mols,
        molsPerRow=per_row,
        subImgSize=(430, 320),
        legends=legends,
        highlightAtomLists=[list(x) for x in colors],
        highlightAtomColors=colors,
    ).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--prompt-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    prompts = load_genmol_prompts(args.prompt_csv)
    prompt = next(
        p for p in prompts if p.drug_name == args.drug and p.task == FragmentTask.LINKER_DESIGN
    )
    lock_path = args.pilot_dir / "locks" / f"{args.drug}.json"
    lock = json.loads(lock_path.read_text())
    samples = lock["samples"]
    if len(samples) != 20:
        raise ValueError(f"expected 20 locked attempts, got {len(samples)}")
    inputs = [
        args.prompt_csv,
        lock_path,
        args.pilot_dir / "manifest.json",
        Path(__file__),
        Path(QED.__file__),
        Path(sascorer.__file__),
    ]
    generated = []
    for index, smiles in enumerate(samples):
        attempt = args.pilot_dir / "attempts" / f"{args.drug}_{index:03d}.json"
        expected = lock["attempt_hashes"][f"attempts/{args.drug}_{index:03d}.json"]
        if _sha256(attempt) != expected:
            raise ValueError(f"pilot attempt hash mismatch: {attempt}")
        data = json.loads(attempt.read_text())
        if data["panel"]["selected_smiles"] != smiles:
            raise ValueError(f"locked sample/attempt disagreement: {attempt}")
        fidelity = linker_fidelity(prompt, smiles)
        if not fidelity["satisfied"] or not data["selected_exact_core_path_fidelity"]:
            raise ValueError(f"pilot core/path fidelity failed: {attempt}")
        generated.append(
            {
                "label": f"attempt {index:02d}",
                "attempt_index": index,
                "metrics": _metrics(smiles),
                "fidelity": fidelity,
            }
        )
        inputs.append(attempt)
    manual = []
    for rooted in MANUAL_CONNECTORS:
        try:
            compiled = assemble_linker_program(prompt, rooted)
            fidelity = linker_fidelity(prompt, compiled.smiles)
            if not fidelity["satisfied"]:
                raise ValueError(f"counterfactual core/path refusal: {fidelity['reason']}")
            manual.append(
                {
                    "label": rooted,
                    "connector": rooted,
                    "metrics": _metrics(compiled.smiles),
                    "fidelity": fidelity,
                    "exact_compose_program": True,
                    "not_sampler_output": True,
                }
            )
        except ValueError as error:
            manual.append({"label": rooted, "connector": rooted, "refusal": str(error)})
    original = _metrics(prompt.original_smiles)
    result = {
        "schema": "fragment_linker_manual_counterfactual_v2",
        "role": "post-generation diagnosis only; never a sampler, benchmark row, or QED/SA-guided selection",
        "drug": args.drug,
        "task": prompt.task.value,
        "original_drug_control": {**original, "linker_admissible": False},
        "generated": generated,
        "manual_fixed_connector_panel": manual,
        "summary": {
            "locked_attempts": len(generated),
            "qed_pass": sum(x["metrics"]["qed_pass"] for x in generated),
            "sa_pass": sum(x["metrics"]["sa_pass"] for x in generated),
            "quality_pass": sum(x["metrics"]["quality_pass"] for x in generated),
            "mean_qed": sum(x["metrics"]["qed"] for x in generated) / len(generated),
            "mean_sa": sum(x["metrics"]["sa"] for x in generated) / len(generated),
            "mean_linker_atoms": sum(x["fidelity"]["linker_internal_atoms"] for x in generated)
            / len(generated),
            "manual_panel_quality_pass": sum(
                x.get("metrics", {}).get("quality_pass", False) for x in manual
            ),
        },
        "configuration": {
            "connectors": MANUAL_CONNECTORS,
            "quality_thresholds": {"qed_min": 0.6, "sa_max": 4.0},
            "pilot_attempts": 20,
            "manual_selection": "none; every predeclared connector reported",
        },
        "provenance": {
            "input_sha256": {str(p.resolve()): _sha256(p) for p in inputs},
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
            "execution": {
                "hardware": "cpu",
                "precision": "rdkit double",
                "seeds": "none; locked inputs and fixed manual panel",
            },
            "exclusions": 0,
        },
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output_dir.parent, prefix=".linker-diagnostic-"
    ) as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        for start in range(0, len(generated), 10):
            _grid(generated[start : start + 10], stage / f"locked_outputs_{start:02d}.png")
        _grid([x for x in manual if "metrics" in x], stage / "manual_connectors.png", per_row=3)
        result["provenance"]["image_sha256"] = {
            x.name: _sha256(x) for x in sorted(stage.glob("*.png"))
        }
        (stage / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stage.rename(args.output_dir)
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "manual": [
                    {
                        "connector": x["connector"],
                        "qed": x.get("metrics", {}).get("qed"),
                        "sa": x.get("metrics", {}).get("sa"),
                        "refusal": x.get("refusal"),
                    }
                    for x in manual
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
