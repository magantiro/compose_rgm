"""Compatibility checks for immutable publication on legacy Modal Volume v1."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from compose_v4.data import immutable_artifact


def test_immutable_publication_falls_back_when_hard_links_are_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.json"

    def unsupported_link(_source: str, _destination: Path) -> None:
        raise OSError(errno.EPERM, "hard links unsupported")

    monkeypatch.setattr(immutable_artifact.os, "link", unsupported_link)

    assert immutable_artifact.write_bytes_if_absent(destination, b"payload\n") is True
    assert destination.read_bytes() == b"payload\n"
    assert immutable_artifact.write_bytes_if_absent(destination, b"payload\n") is False
    with pytest.raises(immutable_artifact.ImmutableArtifactError, match="different bytes"):
        immutable_artifact.write_bytes_if_absent(destination, b"collision\n")
