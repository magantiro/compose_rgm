"""Small offline checks for the resumable official superstructure wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_fragment_superstructure_official_v2 import (
    atomic_json,
    identity,
    load_contract,
    read_shard,
)


def test_contract_self_hash_roundtrip_and_tamper(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    payload = {"schema": "compose_fragment_superstructure_official_v2", "seeds": [0, 1, 2]}
    atomic_json(contract, {"payload": payload, "payload_sha256": identity(payload)})
    assert load_contract(contract) == (payload, identity(payload))
    atomic_json(
        contract, {"payload": {**payload, "seeds": [0]}, "payload_sha256": identity(payload)}
    )
    with pytest.raises(ValueError, match="self-hash mismatch"):
        load_contract(contract)


def test_resume_rejects_a_shard_that_lost_the_fragment(tmp_path: Path) -> None:
    path = tmp_path / "shard.json"
    row = {
        "seed": 0,
        "attempts": 100,
        "attempt_records": [{} for _ in range(100)],
        "chemical_samples": ["C"] * 100,
        "emitted_samples": ["C"] * 100,
        "committed_endpoints": 100,
        "committed_chemically_valid": 100,
        "committed_fragment_preserving": 99,
        "emitted_nonempty": 100,
        "official": {"validity": 100.0},
    }
    shard = {
        "schema": "compose_fragment_official_suite_v2",
        "protocol": {"samples_per_prompt": 100, "seed_list": [0]},
        "sampler": {"config": {"max_events": 32}},
        "attachment_control": {
            "config": {
                "enabled": True,
                "condition_initial_locked_family": True,
                "hard_lock_effective_chemistry": True,
            }
        },
        "results": {"superstructure_generation": {"per_drug": {"TEST": [row]}}},
    }
    path.write_text(json.dumps(shard))
    with pytest.raises(RuntimeError, match="fragment loss"):
        read_shard(path, "TEST", 0, {"sampler_config": {"max_events": 32}})
    row["committed_fragment_preserving"] = 100
    path.write_text(json.dumps(shard))
    assert read_shard(path, "TEST", 0, {"sampler_config": {"max_events": 32}}) == row


def test_resume_rejects_wrong_seed_even_if_every_molecule_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "shard.json"
    shard = {
        "schema": "compose_fragment_official_suite_v2",
        "protocol": {"samples_per_prompt": 100, "seed_list": [1]},
    }
    path.write_text(json.dumps(shard))
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        read_shard(path, "TEST", 0, {"sampler_config": {"max_events": 32}})
