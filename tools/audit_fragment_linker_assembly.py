"""No-score all-prompt semantics and bridge-free compiler support audit."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from fetch_official_fragment_evaluator import OFFICIAL_BLOBS
from rdkit import Chem, rdBase
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_linker_assembly import assemble_linker_program
from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def representation_neutral(text):
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"unparseable benchmark input: {text}")
    Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
    return Chem.MolToSmiles(molecule)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    prompt_path = root / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    if (
        physical_sha256(prompt_path)
        != "a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9"
    ):
        raise ValueError("released GenMol prompt asset changed")
    prompts = [p for p in load_genmol_prompts(prompt_path) if p.task is FragmentTask.LINKER_DESIGN]
    cached_inputs = []
    for name in ("references/frags_downstream.csv", "in_virtuo_gen/evaluation/downstream.py"):
        relative, expected = OFFICIAL_BLOBS[name]
        path = root / ".official_eval_cache/pkg" / relative
        if physical_sha256(path) != expected:
            raise ValueError(f"pinned official source drift: {path}")
        cached_inputs.append(path)
    with cached_inputs[0].open(newline="") as handle:
        ivg_rows = list(csv.DictReader(handle))
    ivg = {representation_neutral(row["smiles"]): row for row in ivg_rows}
    if len(ivg) != 10:
        raise ValueError("IVG source must contain ten distinct drug identities")
    pairs, results = [], []
    for prompt in prompts:
        upstream = ivg[representation_neutral(prompt.original_smiles)]
        official = ast.literal_eval(upstream["Scaffold morphing"])
        equivalent = sorted(map(representation_neutral, official)) == sorted(
            map(representation_neutral, prompt.fragments)
        )
        pairs.append(
            {
                "drug": prompt.drug_name,
                "genmol_fragments": prompt.fragments,
                "ivg_fragments": official,
                "same_graph_interfaces_without_stereo": equivalent,
            }
        )
        for length in (2, 3, 4):
            connector = "[1*]" + "C" * length + "[2*]"
            row = {"drug": prompt.drug_name, "connector": connector, "length": length}
            try:
                candidate = assemble_linker_program(prompt, connector)
            except ValueError as error:
                row.update(status="abstained", reason_type=type(error).__name__, reason=str(error))
            else:
                if candidate.provenance["fidelity"]["linker_internal_atoms"] != length:
                    raise RuntimeError("assembled connector differs from declared length")
                if length == 2:
                    morphing = assemble_linker_program(
                        replace(prompt, task=FragmentTask.SCAFFOLD_MORPHING), connector
                    )
                    if candidate.trace != morphing.trace:
                        raise RuntimeError("identical linker/morphing prompts changed the program")
                row.update(
                    status="complete",
                    smiles=candidate.smiles,
                    provenance=candidate.provenance,
                    trace=candidate.trace,
                )
            results.append(row)
    code_paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if source and name.startswith("compose_v4") and str(source).endswith(".py"):
            code_paths.add(Path(source).resolve())
    inputs = {
        str(p): physical_sha256(p) for p in sorted(code_paths | {prompt_path, *cached_inputs})
    }
    result = {
        "schema": "fragment_linker_assembly_support_v1",
        "evidence_role": "deterministic content/compiler correctness; not learned generation or quality",
        "configuration": {
            "connectors": ["[1*]CC[2*]", "[1*]CCC[2*]", "[1*]CCCC[2*]"],
            "max_atoms": 40,
            "slots": 48,
            "max_primitives": 32,
            "max_blocks": 8,
        },
        "oracle_calls": 0,
        "quality_evaluations": 0,
        "learned_model_calls": 0,
        "randomness": "none; deterministic fixed fixtures and frozen manifest order",
        "inputs": inputs,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "device": "cpu",
        },
        "prompt_comparisons": pairs,
        "equivalent_prompt_pairs": sum(p["same_graph_interfaces_without_stereo"] for p in pairs),
        "attempts": len(results),
        "complete": sum(r["status"] == "complete" for r in results),
        "results": results,
    }
    _atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("equivalent_prompt_pairs", "attempts", "complete")}))


if __name__ == "__main__":
    main()
