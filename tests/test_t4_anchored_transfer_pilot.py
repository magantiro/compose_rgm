import hashlib
import json
from pathlib import Path

from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_anchored_transfer_lock import verify_transfer_lock
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


def test_transfer_pilot_contract_binds_the_frozen_lock():
    contract_path = ROOT / "configs/t4_anchored_transfer_pilot_v1.json"
    envelope = json.loads(contract_path.read_text())
    payload = envelope["payload"]
    assert (
        hashlib.sha256(canonical_bytes(payload)).hexdigest()
        == envelope["payload_sha256"]
    )

    lock_path = ROOT / payload["inputs"]["candidate_lock"]
    assert sha256_file(lock_path) == payload["inputs"]["candidate_lock_sha256"]
    publication = unseal(lock_path)
    candidate_lock = publication["assessment"]["lock"]
    verify_transfer_lock(candidate_lock)
    assert candidate_lock["lock_id"] == payload["candidate_lock_id"]
    assert candidate_lock["charged_calls"] == payload["charged_call_limit"] == 12
    assert candidate_lock["automatic_retries"] == 0
    assert candidate_lock["replacement_after_scoring"] is False


def test_transfer_pilot_binds_implementation_inputs():
    payload = unseal(ROOT / "configs/t4_anchored_transfer_pilot_v1.json")
    inputs = payload["inputs"]
    pairs = (
        ("lock_module", "lock_module_sha256"),
        ("modal_app", "modal_app_sha256"),
        ("docking_wrapper", "docking_wrapper_sha256"),
    )
    for path_key, hash_key in pairs:
        assert sha256_file(ROOT / inputs[path_key]) == inputs[hash_key]
