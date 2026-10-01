"""The T4 downloader must publish only checked MOOD bytes."""

from __future__ import annotations

import hashlib
import io
import stat
from pathlib import Path

import pytest

from tools import fetch_t4_docking_assets as fetch


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_offline_check_does_not_contact_network(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network used")
    )
    row = fetch.ensure_asset(
        tmp_path,
        relative_path="receptors/parp1.pdbqt",
        source_path="receptors/parp1.pdbqt",
        expected_sha256=_hash(b"receptor"),
        download=False,
    )
    assert row["status"] == "missing"
    assert not (tmp_path / "receptors").exists()


def test_download_checks_hash_and_sets_executable_mode(tmp_path: Path, monkeypatch):
    calls = []

    def open_fake(request, *, timeout: int):
        calls.append((request.full_url, request.get_header("User-agent"), timeout))
        return io.BytesIO(b"binary")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", open_fake)
    kwargs = {
        "relative_path": "bin/qvina02",
        "source_path": "qvina02",
        "expected_sha256": _hash(b"binary"),
        "download": True,
        "executable": True,
    }
    assert fetch.ensure_asset(tmp_path, **kwargs)["status"] == "downloaded"
    assert fetch.ensure_asset(tmp_path, **kwargs)["status"] == "ready"
    assert calls == [(fetch.SOURCE_URL + "qvina02", fetch.USER_AGENT, 60)]
    path = tmp_path / "bin/qvina02"
    assert path.read_bytes() == b"binary"
    assert path.stat().st_mode & stat.S_IXUSR


def test_wrong_download_does_not_publish(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong")
    )
    with pytest.raises(ValueError, match="download identity mismatch"):
        fetch.ensure_asset(
            tmp_path,
            relative_path="receptors/parp1.pdbqt",
            source_path="receptors/parp1.pdbqt",
            expected_sha256=_hash(b"right"),
            download=True,
        )
    assert not (tmp_path / "receptors/parp1.pdbqt").exists()


def test_existing_wrong_file_is_preserved(tmp_path: Path):
    path = tmp_path / "receptors/parp1.pdbqt"
    path.parent.mkdir()
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        fetch.ensure_asset(
            tmp_path,
            relative_path="receptors/parp1.pdbqt",
            source_path="receptors/parp1.pdbqt",
            expected_sha256=_hash(b"right"),
            download=False,
        )
    assert path.read_bytes() == b"wrong"


def test_existing_binary_needs_execute_permission(tmp_path: Path):
    path = tmp_path / "bin/qvina02"
    path.parent.mkdir()
    path.write_bytes(b"binary")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="lacks execute permission"):
        fetch.ensure_asset(
            tmp_path,
            relative_path="bin/qvina02",
            source_path="qvina02",
            expected_sha256=_hash(b"binary"),
            download=False,
            executable=True,
        )
    assert path.read_bytes() == b"binary"


def test_manifest_contains_exact_docking_files(tmp_path: Path):
    result = fetch.fetch_assets(tmp_path, download=False)
    assert result["source_revision"] == fetch.SOURCE_REVISION
    assert result["qvina_platform"] == "linux_x86_64"
    assert set(result["assets"]) == {
        "qvina02",
        "receptor_5ht1b",
        "receptor_braf",
        "receptor_fa7",
        "receptor_jak2",
        "receptor_parp1",
    }
    assert all(row["status"] == "missing" for row in result["assets"].values())
