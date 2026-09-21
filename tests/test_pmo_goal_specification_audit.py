"""The goal-extraction classifier reads declared structure the way TDC writes it.

These tests carry their OWN miniature module source, so they do not depend on the pinned
PyTDC sdist being present in a local cache. They pin the three patterns that the first
version of the audit got wrong, each of which under-reported declared content and would
have made the transport architecture look less applicable than it is:

  1. a keyword naming a MODULE CONSTANT rather than a literal (`median1` passes
     `target_smiles_1=camphor_smiles`);
  2. a reference structure assigned to a LOCAL variable inside a function body, which is
     how every composite MPO task and `valsartan_smarts` declare theirs;
  3. a task defined as a CLASS rather than a function or assignment (`jnk3`).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pmo_goal_specification_audit import (
    _classify_structure,
    _declared_in_body,
    _module_constants,
    audit,
)

MODULE = '''
camphor_smiles = "CC1(C)C2CCC1(C)C(=O)C2"
menthol_smiles = "CC(C)C1CCC(C)CC1O"

median1 = median_meta(target_smiles_1=camphor_smiles, target_smiles_2=menthol_smiles)
celecoxib_rediscovery = rediscovery_meta(target_smiles="CC1=CC=C(C=C1)C(F)(F)F", fp="ECFP4")
isomers_c7h8n2o2 = isomer_meta(target_smiles="C7H8N2O2", means="geometric")


def osimertinib_mpo(test_smiles):
    osimertinib_smiles = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"
    return osimertinib_smiles


def valsartan_smarts(test_smiles):
    valsartan_smarts = "CN(C=O)Cc1ccc(c2ccccc2)cc1"
    return valsartan_smarts


class jnk3:
    def __init__(self):
        self.model = "jnk3_current"
'''


def _rows(tasks):
    path = Path(__file__).parent / "_tmp_goal_module.py"
    path.write_text(MODULE)
    try:
        return {row["task"]: row for row in audit(path, tasks)["rows"]}
    finally:
        path.unlink(missing_ok=True)


def test_a_keyword_naming_a_module_constant_is_resolved():
    """`target_smiles_1=camphor_smiles` is a NAME. Reading only literals reported
    median1 as declaring nothing, which is false."""
    row = _rows(["median1"])["median1"]
    assert row["goal_kind"] == "declared_target_pair"
    values = {item["value"] for item in row["declared"]}
    assert "CC1(C)C2CCC1(C)C(=O)C2" in values, "the camphor constant was not resolved"
    assert "CC(C)C1CCC(C)CC1O" in values, "the menthol constant was not resolved"


def test_a_reference_assigned_to_a_local_inside_a_function_is_found():
    """Every composite MPO declares its reference this way. Scanning keywords only
    classified all six as black-box."""
    row = _rows(["osimertinib_mpo"])["osimertinib_mpo"]
    assert row["goal_kind"] == "declared_reference_structure_in_composite"
    assert row["declared"] and row["declared"][0]["kind"] == "smiles"


def test_a_smarts_is_separated_from_a_smiles_by_its_variable_name():
    """A SMARTS such as "CN(C=O)Cc1ccc(c2ccccc2)cc1" also parses as SMILES, so RDKit
    alone cannot separate them; TDC's naming does."""
    row = _rows(["valsartan_smarts"])["valsartan_smarts"]
    assert row["goal_kind"] == "declared_smarts"
    assert {item["kind"] for item in row["declared"]} == {"smarts"}
    assert _classify_structure("x_smarts", "CN(C=O)Cc1ccc(c2ccccc2)cc1") == "smarts"
    assert _classify_structure("x_smiles", "CN(C=O)Cc1ccc(c2ccccc2)cc1") == "smiles"


def test_a_class_backed_task_resolves_rather_than_reporting_unresolved():
    """`jnk3` is a CLASS. Scanning only Assign and FunctionDef reported it unresolved,
    which is indistinguishable from a task the audit failed to read."""
    row = _rows(["jnk3"])["jnk3"]
    assert row["goal_kind"] == "black_box_scalar_only"
    assert row["declared"] == []


def test_a_formula_is_not_mistaken_for_a_structure():
    row = _rows(["isomers_c7h8n2o2"])["isomers_c7h8n2o2"]
    assert row["goal_kind"] == "declared_molecular_formula"
    assert _classify_structure("target_smiles", "C7H8N2O2") == "formula"


def test_module_constants_only_collects_strings():
    constants = _module_constants(ast.parse(MODULE))
    assert constants["camphor_smiles"] == "CC1(C)C2CCC1(C)C(=O)C2"
    assert "median1" not in constants, "a call result is not a string constant"


def test_an_unknown_task_is_reported_unresolved_not_black_box():
    """The two must stay distinguishable: black-box is a finding, unresolved is a gap."""
    row = _rows(["not_a_real_task"])["not_a_real_task"]
    assert row["goal_kind"] == "unresolved"


def test_declared_structures_are_deduplicated_but_order_preserved():
    found = _declared_in_body(
        ast.parse(
            'a_smiles = "CCCCCO"\nb_smiles = "CCCCCO"\nc_smiles = "CCCCCCO"'
        ),
        {},
    )
    assert [item["value"] for item in found] == ["CCCCCO", "CCCCCCO"]


def test_very_short_strings_are_not_read_as_declared_structures():
    """A deliberate guard, found by the dedup test above rather than documented first.

    Many short TDC literals parse as valid molecules -- "CC" and "CCO" among them -- and
    appear as fingerprint names, modifier labels and means. Admitting them would fill the
    audit with structures no task declares. The cost is that a genuinely tiny declared
    target would be missed; no PMO task has one, and `unresolved` would surface it.
    """
    assert _classify_structure("x_smiles", "CCO") is None
    assert _classify_structure("x_smiles", "CCCCCO") == "smiles"
