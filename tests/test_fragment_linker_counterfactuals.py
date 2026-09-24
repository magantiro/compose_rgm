"""Focused checks for post-generation linker diagnosis."""

from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    load_genmol_prompts,
    sascorer,
)
from compose_v4.benchmark.fragment_linker_assembly import (
    assemble_linker_program,
    linker_fidelity,
)
from tools.diagnose_fragment_linker_counterfactuals import (
    MANUAL_CONNECTORS,
    _sa_parts,
)


@pytest.mark.parametrize("smiles", ["CCO", "C1CCCCC1", "c1ccccc1C(=O)N"])
def test_sa_decomposition_reconstructs_authoritative_scorer(smiles: str) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    assert _sa_parts(molecule)["sa"] == pytest.approx(sascorer.calculateScore(molecule), abs=1e-10)


def test_fixed_manual_connectors_are_exact_valid_linkers() -> None:
    csv = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    if not csv.exists():
        pytest.skip("documented GenMol fragment prompt asset absent")
    prompt = next(
        p
        for p in load_genmol_prompts(csv)
        if p.drug_name == "CYCLOTHIAZIDE" and p.task == FragmentTask.LINKER_DESIGN
    )
    for connector in MANUAL_CONNECTORS:
        program = assemble_linker_program(prompt, connector)
        fidelity = linker_fidelity(prompt, program.smiles)
        assert fidelity["satisfied"]
        assert fidelity["linker_internal_atoms"] >= 1
        assert Chem.MolFromSmiles(program.smiles) is not None
