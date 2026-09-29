"""Shared offline fixtures. Expensive chemistry imports stay inside their fixture."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def current_e6_inputs(tmp_path_factory):
    """A fresh bounded graph audit, isolated from the immutable July receipts.

    These are test inputs, not a replacement benchmark result or launch contract.
    The production executor rebuilds the complete six-slot carbon graph. Only the
    fixture readiness contract binds this newly computed, non-scored audit.
    """
    from compose_v4.experiments import e6_a2_3_readiness, e6_graph_audit

    repository = Path(__file__).resolve().parents[1]
    fixture_root = tmp_path_factory.mktemp("fixture_e6_current_inputs")
    frozen_contract = e6_a2_3_readiness.load_a2_3_readiness_contract(
        repository / "configs/exact_control_a2_3_readiness_v1.json"
    )
    inputs = frozen_contract["inputs"]
    paths = {
        *e6_graph_audit._IMPLEMENTATION_SOURCES,
        *e6_a2_3_readiness._IMPLEMENTATION_SOURCES,
        "tests/test_e6_graph_audit.py",
        *(value for name, value in inputs.items() if name.endswith("_path")),
    }
    for relative in sorted(paths):
        # Rebuild the upstream receipt below; never copy its historical numbers.
        if relative == inputs["a2_2b_artifact_path"]:
            continue
        destination = fixture_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)

    registry_path = fixture_root / inputs["registry_path"]
    graph_contract = e6_graph_audit.load_a2_2b_contract(
        fixture_root / inputs["a2_2b_contract_path"], registry_path=registry_path
    )
    graph_audit = e6_graph_audit.run_a2_2b_graph_audit(
        graph_contract, registry_path=registry_path, repo_root=fixture_root
    )
    e6_graph_audit.freeze_a2_2b_artifact(
        graph_audit,
        fixture_root / inputs["a2_2b_artifact_path"],
        contract=graph_contract,
        registry_path=registry_path,
        repo_root=fixture_root,
    )
    inputs["expected_a2_2b_artifact_sha256"] = graph_audit["artifact_sha256"]
    inputs["expected_graph_audit_hash"] = graph_audit["graph_audit_hash"]
    frozen_contract["contract_sha256"] = e6_a2_3_readiness.contract_self_hash(frozen_contract)
    contract_path = fixture_root / "configs/fixture_e6_readiness.json"
    contract_path.write_text(json.dumps(frozen_contract, sort_keys=True))
    contract = e6_a2_3_readiness.load_a2_3_readiness_contract(contract_path)
    assert contract["paper_claim_authorized"] is False
    assert contract["solver_execution_authorized"] is False
    return fixture_root, contract, graph_contract, graph_audit
