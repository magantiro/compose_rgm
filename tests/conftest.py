"""Shared offline fixtures. Expensive chemistry imports stay inside their fixture."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def require_local_artifacts():
    """Skip absent external evidence in a source export, or fail in strict mode.

    Present but altered artifacts are never skipped: the experiment's own hash
    verifier still decides whether those inputs are authentic.
    """

    def require(paths: list[Path], *, purpose: str) -> None:
        missing = [str(path) for path in paths if not path.is_file()]
        if not missing:
            return
        message = f"{purpose} requires external local artifact(s): " + ", ".join(missing)
        if os.environ.get("COMPOSE_REQUIRE_EXTERNAL_ASSETS") == "1":
            pytest.fail(message)
        pytest.skip(message)

    return require


@pytest.fixture
def require_software_versions():
    """Route serialized-model tests to their exact recorded software kernel."""

    def require(expected: dict[str, str], *, purpose: str) -> None:
        mismatches = []
        for distribution, version in sorted(expected.items()):
            try:
                if distribution == "rdkit":
                    # Recorded chemistry identities use RDKit's own version
                    # spelling (for example 2025.09.6), not PEP-440's
                    # normalized distribution spelling (2025.9.6).
                    from rdkit import rdBase

                    observed = rdBase.rdkitVersion
                else:
                    observed = importlib.metadata.version(distribution)
            except (ImportError, importlib.metadata.PackageNotFoundError):
                observed = "absent"
            if observed != version:
                mismatches.append(f"{distribution}={observed} (need {version})")
        if not mismatches:
            return
        message = f"{purpose} requires a separate kernel: " + ", ".join(mismatches)
        if os.environ.get("COMPOSE_REQUIRE_ALTERNATE_KERNELS") == "1":
            pytest.fail(message)
        pytest.skip(message)

    return require


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
