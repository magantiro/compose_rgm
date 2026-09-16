import ast
import json
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from tools.t4_complete_region_runtime_audit import DEFAULT_CONTRACT, load_contract


def test_complete_region_contract_is_self_hashed_and_zero_cost() -> None:
    payload = load_contract(DEFAULT_CONTRACT)
    assert payload["frozen_protocol"]["exact_teacher_routes"] == 77
    assert payload["declared_support"]["program_decisions"] == [1, 4]
    assert all(value == 0 for value in payload["costs"].values())


def test_complete_region_contract_mutation_fails(tmp_path: Path) -> None:
    envelope = json.loads(DEFAULT_CONTRACT.read_text())
    envelope["payload"]["costs"]["oracle_calls_authorized"] = 1
    envelope["contract_sha256"] = identity(envelope["payload"])
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="nonzero external cost"):
        load_contract(mutated)


def test_complete_region_audit_has_no_external_runtime_import() -> None:
    source = Path("tools/t4_complete_region_runtime_audit.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "modal" or name.startswith("modal.") for name in imports)
    assert not any("oracle" in name or "docking_oracle" in name for name in imports)
