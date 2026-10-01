"""The PMO asset fetcher must publish only hash-verified files."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from tools import fetch_pmo_oracle_assets as fetch


def _entry(data: bytes) -> dict:
    return {
        "relative_path": "oracle/test_current.pkl",
        "sha256": hashlib.sha256(data).hexdigest(),
        "dataverse_file_id": 123,
    }


def test_offline_check_reports_missing_without_network(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network used")
    )
    result = fetch.ensure_asset(tmp_path, _entry(b"model"), download=False)
    assert result["status"] == "missing"
    assert not (tmp_path / "oracle").exists()


def test_download_is_checked_and_never_overwrites(tmp_path: Path, monkeypatch):
    calls = []

    def open_fake(request, *, timeout: int):
        calls.append((request.full_url, request.get_header("User-agent"), timeout))
        return io.BytesIO(b"model")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", open_fake)
    entry = _entry(b"model")
    assert fetch.ensure_asset(tmp_path, entry, download=True)["status"] == "downloaded"
    assert fetch.ensure_asset(tmp_path, entry, download=True)["status"] == "ready"
    assert calls == [(fetch.DATAVERSE_URL + "123", fetch.USER_AGENT, 60)]
    assert (tmp_path / "oracle/test_current.pkl").read_bytes() == b"model"


def test_wrong_download_never_publishes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong")
    )
    with pytest.raises(ValueError, match="download identity mismatch"):
        fetch.ensure_asset(tmp_path, _entry(b"model"), download=True)
    assert not (tmp_path / "oracle/test_current.pkl").exists()
    assert not list((tmp_path / "oracle").iterdir())


def test_existing_wrong_file_is_preserved(tmp_path: Path):
    target = tmp_path / "oracle/test_current.pkl"
    target.parent.mkdir()
    target.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        fetch.ensure_asset(tmp_path, _entry(b"model"), download=False)
    assert target.read_bytes() == b"wrong"


def test_asset_path_refuses_parent_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid oracle asset path"):
        fetch.asset_path(tmp_path, "../outside.pkl")
