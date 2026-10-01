"""Public task configurations must name one frozen molecular reference."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_task_configs_share_the_nll_reference_identity():
    fragment = _read("experiments/fragments/assets.json")
    shared = _read("experiments/reference/model.json")
    pmo = _read("experiments/pmo/example.json")
    t4 = _read("experiments/t4/example.json")
    checkpoint_sha256 = fragment["assets"]["checkpoint"]["sha256"]
    catalog_sha256 = fragment["assets"]["catalog"]["sha256"]
    catalog_fingerprint = fragment["catalog_fingerprint"]
    from compose_v4.experiments.fragments.generation_worker import EXPECTED_CATALOG_FINGERPRINT

    assert catalog_fingerprint == EXPECTED_CATALOG_FINGERPRINT
    assert checkpoint_sha256 == shared["checkpoint"]["sha256"]
    assert catalog_sha256 == shared["catalog"]["sha256"]
    for task in (
        pmo,
        t4,
        _read("experiments/pmo/uniform_chain.json"),
        _read("experiments/pmo/created_atom_rebinding.json"),
    ):
        assert task["reference"]["sha256"] == checkpoint_sha256
        assert task["reference"]["catalog_fingerprint"] == catalog_fingerprint
        assert task["reference"]["catalog_sha256"] == catalog_sha256
    assert pmo["guidance"]["mode"] == "active"
    assert t4["search"]["guidance"]["mode"] == "active"
