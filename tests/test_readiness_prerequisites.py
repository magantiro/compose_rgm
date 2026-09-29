"""Missing research inputs are visible, and strict verification fails closed."""

from __future__ import annotations

import pytest


def test_external_artifact_gate_distinguishes_absence_from_strict_failure(
    tmp_path, monkeypatch, require_local_artifacts
):
    missing = tmp_path / "not-distributed.json"
    monkeypatch.delenv("COMPOSE_REQUIRE_EXTERNAL_ASSETS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="not-distributed.json"):
        require_local_artifacts([missing], purpose="frozen campaign")
    monkeypatch.setenv("COMPOSE_REQUIRE_EXTERNAL_ASSETS", "1")
    with pytest.raises(pytest.fail.Exception, match="not-distributed.json"):
        require_local_artifacts([missing], purpose="frozen campaign")


def test_alternate_kernel_gate_uses_exact_rdkit_runtime_version(
    monkeypatch, require_software_versions
):
    from rdkit import rdBase

    require_software_versions({"rdkit": rdBase.rdkitVersion}, purpose="chemistry kernel")
    monkeypatch.delenv("COMPOSE_REQUIRE_ALTERNATE_KERNELS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="requires a separate kernel"):
        require_software_versions({"missing-research-package": "1.0"}, purpose="legacy oracle")
    monkeypatch.setenv("COMPOSE_REQUIRE_ALTERNATE_KERNELS", "1")
    with pytest.raises(pytest.fail.Exception, match="requires a separate kernel"):
        require_software_versions({"missing-research-package": "1.0"}, purpose="legacy oracle")
