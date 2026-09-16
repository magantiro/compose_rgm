from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.complete_region_patch_policy import (
    ConditionalPatchPolicy,
    PatchPolicyTrainingRow,
    SourceRegionContext,
    decode_patch_stream,
    encode_patch_stream,
    fit_conditional_patch_policy,
    patch_stream_support,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import atom_signature, environment
from compose_v4.control.structural_subgoal import StructuralSubgoal
from tools.t4_complete_region_patch_policy import CONTRACT, load_contract


def _append_carbon_patch() -> StructuralSubgoal:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[0] = 2
    hydrogens[0] = 4
    source = MolecularGraph(atom_types, charges, hydrogens, bonds)
    return StructuralSubgoal(
        input_atoms=(atom_signature(source, 0),),
        input_bonds=((0,),),
        environments=(environment(source, 0),),
        target_atoms=((2, 0, 3, 1),),
        output_atoms=((2, 0, 3, 1),),
        target_bonds=((0, 1), (1, 0)),
    )


def test_complete_patch_stream_roundtrips_whole_structural_goal() -> None:
    patch = _append_carbon_patch()
    tokens = encode_patch_stream(patch)

    decoded = decode_patch_stream(SourceRegionContext.from_subgoal(patch), tokens)

    assert decoded == patch
    assert patch_stream_support(patch) == (True, None)
    factors = {token.factor for token in tokens}
    assert factors == {
        "atom_attributes",
        "attachments",
        "control",
        "dependencies",
    }
    assert not any("executor" in token.kind or "primitive" in token.kind for token in tokens)


def test_complete_patch_stream_rejects_value_outside_fixed_support() -> None:
    patch = _append_carbon_patch()
    tokens = list(encode_patch_stream(patch))
    index = next(i for i, token in enumerate(tokens) if token.kind == "output_formal_charge")
    tokens[index] = replace(tokens[index], value=9)

    try:
        decode_patch_stream(SourceRegionContext.from_subgoal(patch), tokens)
    except ValueError as error:
        assert "unsupported complete-patch token" in str(error)
    else:
        raise AssertionError("out-of-support charge token was accepted")


def test_complete_region_patch_policy_contract_is_self_hashed_and_zero_cost() -> None:
    payload = load_contract(CONTRACT)
    assert payload["gate0_support"]["require_teacher_patch_grammar_support"] == 147
    assert payload["generated_object"]["program_decisions"] == [1, 4]
    assert all(value == 0 for value in payload["costs"].values())


def test_complete_region_patch_policy_contract_mutation_fails(tmp_path: Path) -> None:
    envelope = json.loads(CONTRACT.read_text())
    envelope["payload"]["costs"]["oracle_calls_authorized"] = 1
    envelope["contract_sha256"] = identity(envelope["payload"])
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="external cost"):
        load_contract(mutated)


def test_complete_region_patch_policy_tool_has_no_external_runtime_import() -> None:
    tree = ast.parse(Path("tools/t4_complete_region_patch_policy.py").read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any(name == "modal" or name.startswith("modal.") for name in imports)
    assert not any("oracle" in name or "docking_oracle" in name for name in imports)


def test_split_first_complete_patch_policy_roundtrip_and_score() -> None:
    patch = _append_carbon_patch()
    row = PatchPolicyTrainingRow(
        source_group="source-a",
        route_id="route-a",
        region_index=0,
        route_region_count=1,
        context=SourceRegionContext.from_subgoal(patch),
        tokens=encode_patch_stream(patch),
        control_after="stop",
    )

    policy = fit_conditional_patch_policy([row], smoothing_alpha=0.25)
    restored = ConditionalPatchPolicy.from_checkpoint(policy.checkpoint())
    learned = restored.score_stream(row.context, row.tokens, learned=True)
    marginal = restored.score_stream(row.context, row.tokens, learned=False)

    assert learned["supported"] is True
    assert marginal["supported"] is True
    assert learned["nll"] <= marginal["nll"]
    serialized = json.dumps(restored.checkpoint(), sort_keys=True)
    for forbidden in ("source-a", "route-a", "source_state", "endpoint", "actions"):
        assert forbidden not in serialized
