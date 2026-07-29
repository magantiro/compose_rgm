"""Negative tests for the strict RING_CORE_V1 checkpoint-identity gate (requirement 1 of the rollout-harness
hardening). A base-B checkpoint, wrong max_atoms, stale operator registry, cycle ops disabled, ring macros
enabled, legacy grow enabled, or missing metadata must FAIL LOUDLY -- never be analyzed as RingCore."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ring_core_identity import (  # noqa: E402
    BASE_B_SHA256,
    SCHEDULER_CONFIG_HASH,
    SCOPE_HASH,
    RingCoreIdentityError,
    recompute_scheduler_config_hash,
    scheduler_config_hash_from_args,
    verify_checkpoint_identity,
)


def _valid_payload() -> dict:
    return {
        "state_dict": {},
        "training_backend": "factorized_marks",
        "corpus_scope_hash": SCOPE_HASH,
        "enable_cycle_ops": True,
        "enable_ring_macros": False,
        "enable_ring_grow_macro": False,
        "corrupted_prior_mix": True,
        "max_atoms": 40,
        "global_step": 500,
        "initialization_source_sha256": BASE_B_SHA256,
        "rate_factorization": "hierarchical",
    }


def test_valid_ring_core_payload_passes():
    report = verify_checkpoint_identity(_valid_payload(), checkpoint_path=None)
    assert report["identity_ok"] is True
    assert report["enable_cycle_ops"] and not report["enable_ring_grow_macro"]


def test_missing_checkpoint_file_fails():
    with pytest.raises(RingCoreIdentityError, match="not found"):
        verify_checkpoint_identity(_valid_payload(), checkpoint_path=Path("/nonexistent/ckpt.pt"))


def test_base_b_checkpoint_fails():
    payload = _valid_payload()
    del payload["enable_cycle_ops"]
    del payload["enable_ring_grow_macro"]
    payload["corrupted_prior_mix"] = False
    with pytest.raises(RingCoreIdentityError):
        verify_checkpoint_identity(payload, checkpoint_path=None)


def test_wrong_max_atoms_fails():
    payload = _valid_payload()
    payload["max_atoms"] = 48
    with pytest.raises(RingCoreIdentityError, match="max_atoms"):
        verify_checkpoint_identity(payload, checkpoint_path=None)


def test_stale_operator_registry_fails():
    payload = _valid_payload()
    payload["operator_registry_hash"] = "deadbeefdeadbeef"
    with pytest.raises(RingCoreIdentityError, match="operator_registry_hash|operator-registry"):
        verify_checkpoint_identity(payload, checkpoint_path=None)


def test_cycle_ops_disabled_fails():
    payload = _valid_payload()
    payload["enable_cycle_ops"] = False
    with pytest.raises(RingCoreIdentityError, match="cycle"):
        verify_checkpoint_identity(payload, checkpoint_path=None)


def test_ring_macros_enabled_fails():
    payload = _valid_payload()
    payload["enable_ring_macros"] = True
    with pytest.raises(RingCoreIdentityError, match="macros"):
        verify_checkpoint_identity(payload, checkpoint_path=None)


def test_legacy_grow_enabled_fails():
    payload = _valid_payload()
    payload["enable_ring_grow_macro"] = True
    with pytest.raises(RingCoreIdentityError, match="grow"):
        verify_checkpoint_identity(payload, checkpoint_path=None)


def test_wrong_scope_fails():
    payload = _valid_payload()
    payload["corpus_scope_hash"] = "0000000000000000"
    with pytest.raises(RingCoreIdentityError, match="scope"):
        verify_checkpoint_identity(payload, checkpoint_path=None)


def test_missing_metadata_fails():
    payload = _valid_payload()
    del payload["state_dict"]
    with pytest.raises(RingCoreIdentityError, match="missing"):
        verify_checkpoint_identity(payload, checkpoint_path=None)


# ---- Owner-locked production scheduler (decision 2026-07-28) ----


def test_frozen_scheduler_dict_matches_constant():
    # the frozen PRODUCTION_SCHEDULER must hash to the pinned SCHEDULER_CONFIG_HASH.
    # ACTIVE contract is the owner-approved 16,000-step schedule (2026-07-29); the 3,000-step schedule
    # (0b832985c65de1cc) is SUPERSEDED and is asserted to FAIL in test_approved_scheduler_contract.py.
    assert recompute_scheduler_config_hash() == SCHEDULER_CONFIG_HASH == "dafd4b5092414394"


def test_locked_scheduler_args_reproduce_hash():
    # a launch using the owner-locked scheduler values reproduces the frozen hash
    assert (
        scheduler_config_hash_from_args(
            warmup_steps=500,
            schedule_steps=16000,
            minimum_learning_rate_fraction=0.05,
            peak_learning_rate=3e-4,
            weight_decay=1e-5,
        )
        == SCHEDULER_CONFIG_HASH
    )


@pytest.mark.parametrize(
    "override",
    [
        {"schedule_steps": 30000},  # base-B de-novo horizon (never decays over a warm-start window)
        {"schedule_steps": 3000},  # the SUPERSEDED pre-benchmark lock -- now drift
        {"schedule_steps": 12000},  # a considered but unapproved horizon
        {"schedule_steps": 2000},  # over-aggressive cooling
        {"schedule_steps": 500},  # the superseded all-warmup preflight
        {"warmup_steps": 0},  # no warmup
        {"peak_learning_rate": 2e-3},  # the gate default lr, not the locked 3e-4
        {"weight_decay": 0.0},  # the gate default wd, not the locked 1e-5
        {"minimum_learning_rate_fraction": 1.0},  # no decay floor
    ],
)
def test_scheduler_drift_changes_hash(override):
    # any drift from the locked scheduler yields a different hash -> the gate launch guard aborts
    base = dict(
        warmup_steps=500,
        schedule_steps=16000,
        minimum_learning_rate_fraction=0.05,
        peak_learning_rate=3e-4,
        weight_decay=1e-5,
    )
    base.update(override)
    assert scheduler_config_hash_from_args(**base) != SCHEDULER_CONFIG_HASH


def test_declared_scheduler_hash_mismatch_fails_identity():
    # a RingCore checkpoint self-declaring a drifted scheduler hash fails the strict identity gate
    payload = _valid_payload()
    payload["scheduler_config_hash"] = "deadbeefdeadbeef"
    with pytest.raises(RingCoreIdentityError, match="scheduler_config_hash"):
        verify_checkpoint_identity(payload, checkpoint_path=None)


def test_declared_locked_scheduler_hash_passes_identity():
    payload = _valid_payload()
    payload["scheduler_config_hash"] = SCHEDULER_CONFIG_HASH
    report = verify_checkpoint_identity(payload, checkpoint_path=None)
    assert report["identity_ok"] is True
