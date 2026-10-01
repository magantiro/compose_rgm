"""The external fragment evaluator must be fetched at an exact identity."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from tools import fetch_fragment_evaluator as fetch


def _entry(data: bytes) -> dict:
    return {
        "path": "evaluator/pkg/in_virtuo_gen/train_utils/metrics.py",
        "sha256": hashlib.sha256(data).hexdigest(),
        "origin": fetch.SOURCE_IDENTITY + "in_virtuo_gen/train_utils/metrics.py",
    }


def test_offline_check_never_contacts_network(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network used")
    )
    assert fetch.ensure_asset(tmp_path, _entry(b"code"), download=False)["status"] == "missing"
    assert not (tmp_path / "evaluator").exists()


def test_download_uses_pinned_source_and_hash(tmp_path: Path, monkeypatch):
    calls = []

    def open_fake(request, *, timeout: int):
        calls.append((request.full_url, request.get_header("User-agent"), timeout))
        return io.BytesIO(b"code")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", open_fake)
    entry = _entry(b"code")
    assert fetch.ensure_asset(tmp_path, entry, download=True)["status"] == "downloaded"
    assert fetch.ensure_asset(tmp_path, entry, download=True)["status"] == "ready"
    assert calls == [
        (fetch.SOURCE_URL + "in_virtuo_gen/train_utils/metrics.py", fetch.USER_AGENT, 60)
    ]


def test_wrong_bytes_do_not_publish(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong")
    )
    with pytest.raises(ValueError, match="download identity mismatch"):
        fetch.ensure_asset(tmp_path, _entry(b"code"), download=True)
    assert not (tmp_path / "evaluator/pkg/in_virtuo_gen/train_utils/metrics.py").exists()


def test_manifest_origin_cannot_redirect_source():
    entry = _entry(b"code")
    entry["origin"] = "different_repo@" + fetch.SOURCE_REVISION + ":metrics.py"
    with pytest.raises(ValueError, match="unexpected evaluator origin"):
        fetch.source_path(entry)
    entry = _entry(b"code")
    entry["path"] = "evaluator/pkg/other.py"
    with pytest.raises(ValueError, match="does not match origin"):
        fetch.source_path(entry)


def test_prompt_table_adds_only_the_pinned_terminal_newline(tmp_path: Path):
    source = tmp_path / "evaluator/pkg/references/fragments.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"header\nrow")
    source_entry = {
        "path": "evaluator/pkg/references/fragments.csv",
        "sha256": _entry(b"header\nrow")["sha256"],
    }
    prompts = {
        "path": "prompts.csv",
        "sha256": _entry(b"header\nrow\n")["sha256"],
        "source_sha256": source_entry["sha256"],
    }
    result = fetch.ensure_prompts(tmp_path, prompts, source_entry, download=True)
    assert result["status"] == "derived"
    assert (tmp_path / "prompts.csv").read_bytes() == b"header\nrow\n"
    assert (
        fetch.ensure_prompts(tmp_path, prompts, source_entry, download=False)["status"] == "ready"
    )


def test_prompt_derivation_requires_verified_source(tmp_path: Path):
    with pytest.raises(ValueError, match="verified upstream fragment table is missing"):
        fetch.ensure_prompts(
            tmp_path,
            {
                "path": "prompts.csv",
                "sha256": _entry(b"row\n")["sha256"],
                "source_sha256": _entry(b"row")["sha256"],
            },
            {"path": "evaluator/pkg/references/fragments.csv", "sha256": _entry(b"row")["sha256"]},
            download=True,
        )


def test_manifest_lists_six_upstream_inputs_and_prompts(tmp_path: Path):
    result = fetch.fetch_assets(tmp_path, download=False)
    assert set(result["assets"]) == fetch.EVALUATOR_KEYS | {"prompts"}
    assert result["source_revision"] == fetch.SOURCE_REVISION
    assert all(row["status"] == "missing" for row in result["assets"].values())
