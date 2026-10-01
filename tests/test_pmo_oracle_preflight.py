"""The PMO chemistry preflight does not spend an oracle evaluation."""

from __future__ import annotations

import json
import sys

import pytest

from compose_v4.experiments import pmo_oracle_worker


def test_wrong_chemistry_kernel_is_rejected(monkeypatch):
    versions = {"PyTDC": "1.1.15", "rdkit": "2024.3.5", "numpy": "1.26.4"}
    monkeypatch.setattr(pmo_oracle_worker.sys, "version_info", (3, 11))
    monkeypatch.setattr(
        pmo_oracle_worker.importlib.metadata,
        "version",
        lambda name: versions[name],
    )
    with pytest.raises(ValueError, match="not pinned"):
        pmo_oracle_worker.check_oracle_environment()


def test_cli_preflight_never_constructs_an_oracle(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight constructed an oracle")

    monkeypatch.setattr(pmo_oracle_worker, "_real_oracle", forbidden)
    monkeypatch.setattr(
        pmo_oracle_worker,
        "check_oracle_environment",
        lambda: (object(), {"PyTDC": "1.1.15", "rdkit": "2023.9.6", "numpy": "1.26.4"}),
    )
    monkeypatch.setattr(sys, "argv", ["pmo_oracle_worker", "--check-env"])
    pmo_oracle_worker.main()
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
