"""Protect diagnostic score parity and locked-core counterfactual semantics."""

import pytest
from rdkit import Chem
from rdkit.Chem import QED

from compose_v4.benchmark.fragment_constrained import FragmentPrompt, FragmentTask
from tools.diagnose_fragment_qed import decompose
from tools.fragment_decoration_counterfactuals import simplify


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1"])
def test_qed_decomposition_reconstructs_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert decompose(mol)["qed"] == pytest.approx(QED.qed(mol), abs=1e-12)


def test_single_pendant_replacement_preserves_core():
    prompt = FragmentPrompt(
        "fixture", "CCCc1ccccc1", FragmentTask.SCAFFOLD_DECORATION, ("[1*]c1ccccc1",)
    )
    assert simplify("CCCc1ccccc1", prompt) == "Cc1ccccc1"


def test_two_interfaces_remain_decorated_without_changing_core():
    prompt = FragmentPrompt(
        "fixture", "CCCc1ccc(CCO)cc1", FragmentTask.SCAFFOLD_DECORATION, ("[1*]c1ccc([2*])cc1",)
    )
    result = simplify(prompt.original_smiles, prompt, all_decorations=True)
    assert result == Chem.MolToSmiles(Chem.MolFromSmiles("Cc1ccc(C)cc1"))


def test_two_boundary_region_is_not_silently_cut_as_pendant():
    prompt = FragmentPrompt(
        "fixture", "c1ccc2c(c1)CCC2", FragmentTask.SCAFFOLD_DECORATION, ("[1*]c1ccccc1[2*]",)
    )
    with pytest.raises(ValueError, match="single-boundary"):
        simplify(prompt.original_smiles, prompt)
