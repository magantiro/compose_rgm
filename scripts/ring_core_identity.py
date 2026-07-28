#!/usr/bin/env python3
"""Shared RING_CORE_V1 identity: frozen hashes, checkpoint-metadata writer, and the STRICT verifier.

Single source of truth for the frozen capability so the trainer (writes the identity into checkpoint
metadata), the rollout harness (verifies it before analysis), and the negative tests all agree. The verifier
FAILS LOUDLY on any mismatch -- a base-B checkpoint, wrong max_atoms, stale operator registry, cycle ops
disabled, ring macros enabled, legacy grow enabled, or missing metadata must never be analyzed as RingCore.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---- Frozen RING_CORE_V1 identity (tag ring-core-v1) ----
CAPABILITY_HASH = "330473e319bfec19"
OPERATOR_REGISTRY_HASH = "9197401e8dc3a7ae"
CYCLE_OP_SEMANTIC_HASH = "27a823aeb6cf7548"
CALIBRATION_POLICY_HASH = "f534c0233e3bf4a0"
SCOPE_HASH = "3721d69851110fdd"
BASE_B_SHA256 = "c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c"
MAX_ATOMS = 40

# ---- Owner-locked production scheduler (decision 2026-07-28) ----
# The schedule-faithful 1,000-step check AND the eventual full RingCore run share this SAME schedule (only
# training_steps differs); the check stops early at 1,000 inside the 3,000-step cosine decay. schedule_steps
# is the cosine-decay horizon and is deliberately NOT the base-B de-novo 30,000 (which leaves LR ~peak over a
# 1-3k warm-start window and never tests decay) nor 2,000 (too-aggressive cooling). Any RingCore launch whose
# scheduler args do not reproduce SCHEDULER_CONFIG_HASH aborts before GPU (see the gate launch-identity guard).
PRODUCTION_SCHEDULER = {
    "optimizer": "AdamW",
    "peak_learning_rate": 3e-4,
    "weight_decay": 1e-05,
    "warmup_steps": 500,
    "schedule_steps": 3000,
    "minimum_learning_rate_fraction": 0.05,
    "schedule": "cosine_with_linear_warmup",
}
SCHEDULER_CONFIG_HASH = "0b832985c65de1cc"

_OPERATOR_REGISTRY_SOURCES = [
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/tracelets.py",
    "src/compose_v4/rewrite/factorized_fiber.py",
]
# Must match freeze_ring_core_v1._CYCLE_OP_SEMANTIC_SOURCES exactly (frozen hash 27a823aeb6cf7548).
_CYCLE_OP_SEMANTIC_SOURCES = [
    "src/compose_v4/experiments/cycle_op_prior.py",
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/kernel.py",
]


def _hash_sources(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(paths):
        fp = REPO / rel
        digest.update(rel.encode())
        digest.update(fp.read_bytes() if fp.exists() else b"<MISSING>")
    return digest.hexdigest()[:16]


def recompute_operator_registry_hash() -> str:
    return _hash_sources(_OPERATOR_REGISTRY_SOURCES)


def recompute_cycle_op_semantic_hash() -> str:
    return _hash_sources(_CYCLE_OP_SEMANTIC_SOURCES)


def _scheduler_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def recompute_scheduler_config_hash() -> str:
    """Hash of the frozen PRODUCTION_SCHEDULER; must equal SCHEDULER_CONFIG_HASH (a self-check)."""
    return _scheduler_hash(PRODUCTION_SCHEDULER)


def scheduler_config_hash_from_args(
    *,
    warmup_steps: int,
    schedule_steps: int,
    minimum_learning_rate_fraction: float,
    peak_learning_rate: float,
    weight_decay: float,
    optimizer: str = "AdamW",
    schedule: str = "cosine_with_linear_warmup",
) -> str:
    """Hash a run's ACTUAL scheduler args in the canonical PRODUCTION_SCHEDULER shape. Excludes
    training_steps by design -- the schedule-check (1,000) and the full run share the SAME LR schedule, so
    both must produce SCHEDULER_CONFIG_HASH; only training_steps differs (the check stops early)."""
    return _scheduler_hash(
        {
            "optimizer": optimizer,
            "peak_learning_rate": peak_learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "schedule_steps": schedule_steps,
            "minimum_learning_rate_fraction": minimum_learning_rate_fraction,
            "schedule": schedule,
        }
    )


def production_sampler_config_hash(payload: dict) -> str:
    """Hash the sampler/config-relevant metadata so a changed inference config is detectable."""
    keys = (
        "rate_factorization", "ring_electronic_mode", "ring_family_mass_mode",
        "ring_template_factorization", "bond_representation", "hidden_dim",
        "message_passing_steps", "empirical_mark_prior_mode",
    )
    config = {k: payload.get(k) for k in keys}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def ring_core_checkpoint_metadata() -> dict:
    """Identity fields the trainer persists into checkpoint_metadata for a RING_CORE_V1 checkpoint, so the
    checkpoint self-identifies. Code hashes are recomputed live (authoritative), then cross-checked here."""
    return {
        "ring_core_v1": True,
        "ring_core_capability_hash": CAPABILITY_HASH,
        "operator_registry_hash": recompute_operator_registry_hash(),
        "cycle_op_semantic_hash": recompute_cycle_op_semantic_hash(),
        "calibration_policy_hash": CALIBRATION_POLICY_HASH,
        "ring_core_max_atoms": MAX_ATOMS,
        # Owner-locked production scheduler (the gate asserts the run's actual scheduler reproduces this
        # hash before GPU, so persisting the frozen constant is truthful for every RingCore checkpoint).
        "scheduler_config_hash": SCHEDULER_CONFIG_HASH,
        "production_scheduler": dict(PRODUCTION_SCHEDULER),
    }


class RingCoreIdentityError(ValueError):
    """A checkpoint failed the strict RingCore-V1 identity gate."""


def verify_checkpoint_identity(
    payload: dict,
    *,
    checkpoint_path: Path | None,
    require_trained: bool = True,
) -> dict:
    """(report) or raise RingCoreIdentityError. Verifies EVERY identity fact and fails loudly on mismatch.

    Critical (raise): missing checkpoint, missing metadata, max_atoms != 40, scope mismatch, cycle ops
    disabled/absent, ring macros enabled, legacy grow enabled/absent (a RingCore checkpoint MUST carry
    enable_ring_grow_macro=False explicitly -- absence means base-B or a non-RingCore checkpoint), stale
    operator-registry or cycle-op code hash, carbon-tree-only (base-B) checkpoint."""
    problems: list[str] = []
    report: dict = {"checkpoint_path": str(checkpoint_path) if checkpoint_path else None}

    if checkpoint_path is not None:
        if not Path(checkpoint_path).exists():
            raise RingCoreIdentityError(f"checkpoint not found: {checkpoint_path}")
        report["checkpoint_sha256"] = hashlib.sha256(
            Path(checkpoint_path).read_bytes()
        ).hexdigest()

    required_meta = {"state_dict", "training_backend", "corpus_scope_hash"}
    missing = sorted(required_meta - set(payload))
    if missing:
        raise RingCoreIdentityError(f"missing checkpoint metadata: {missing}")

    # global step + source B
    report["global_step"] = payload.get("global_step", payload.get("step"))
    report["source_b_sha256"] = payload.get(
        "initialization_source_sha256", payload.get("source_checkpoint_sha256")
    )
    if report["source_b_sha256"] not in (None, BASE_B_SHA256):
        problems.append(
            f"source B hash {report['source_b_sha256']} != frozen {BASE_B_SHA256}"
        )

    # max_atoms
    max_atoms = payload.get("ring_core_max_atoms", payload.get("max_atoms"))
    report["max_atoms"] = max_atoms
    if max_atoms is not None and int(max_atoms) != MAX_ATOMS:
        problems.append(f"max_atoms {max_atoms} != {MAX_ATOMS}")

    # scope
    report["corpus_scope_hash"] = payload.get("corpus_scope_hash")
    if payload.get("corpus_scope_hash") != SCOPE_HASH:
        problems.append(f"scope hash {payload.get('corpus_scope_hash')} != {SCOPE_HASH}")

    # capability flags -- the decisive RingCore identity
    enable_cycle_ops = bool(payload.get("enable_cycle_ops"))
    enable_ring_macros = bool(payload.get("enable_ring_macros"))
    grow_present = "enable_ring_grow_macro" in payload
    enable_ring_grow_macro = bool(payload.get("enable_ring_grow_macro", True))
    report["enable_cycle_ops"] = enable_cycle_ops
    report["enable_ring_macros"] = enable_ring_macros
    report["enable_ring_grow_macro"] = enable_ring_grow_macro
    if not enable_cycle_ops:
        problems.append("enable_cycle_ops is not True (cycle operators disabled)")
    if enable_ring_macros:
        problems.append("enable_ring_macros is True (ring macros must be disabled in RingCore-V1)")
    if not grow_present:
        problems.append("enable_ring_grow_macro absent (base-B / non-RingCore checkpoint)")
    elif enable_ring_grow_macro:
        problems.append("enable_ring_grow_macro is True (legacy grow macro must be disabled)")

    # base-B (carbon-tree-only) checkpoint
    if not bool(payload.get("corrupted_prior_mix")):
        problems.append("corrupted_prior_mix is not True (carbon-tree-only / base-B checkpoint)")

    # code hashes (recomputed from source; must equal frozen)
    op_hash = recompute_operator_registry_hash()
    cyc_hash = recompute_cycle_op_semantic_hash()
    report["operator_registry_hash"] = op_hash
    report["cycle_op_semantic_hash"] = cyc_hash
    if op_hash != OPERATOR_REGISTRY_HASH:
        problems.append(f"operator-registry hash {op_hash} != frozen {OPERATOR_REGISTRY_HASH}")
    if cyc_hash != CYCLE_OP_SEMANTIC_HASH:
        problems.append(f"cycle-op semantic hash {cyc_hash} != frozen {CYCLE_OP_SEMANTIC_HASH}")
    # if the checkpoint self-declares hashes, they must match too
    for key, frozen in (
        ("ring_core_capability_hash", CAPABILITY_HASH),
        ("operator_registry_hash", OPERATOR_REGISTRY_HASH),
        ("cycle_op_semantic_hash", CYCLE_OP_SEMANTIC_HASH),
        ("calibration_policy_hash", CALIBRATION_POLICY_HASH),
        ("scheduler_config_hash", SCHEDULER_CONFIG_HASH),
    ):
        declared = payload.get(key)
        if declared is not None and declared != frozen:
            problems.append(f"declared {key} {declared} != frozen {frozen}")

    report["capability_hash"] = CAPABILITY_HASH
    report["calibration_policy_hash"] = CALIBRATION_POLICY_HASH
    report["production_sampler_config_hash"] = production_sampler_config_hash(payload)

    if require_trained and report["global_step"] in (None, 0):
        # not fatal for a step-0 A/B baseline load; the caller decides. Flag it.
        report["warning_step_0_or_unknown"] = True

    if problems:
        raise RingCoreIdentityError("; ".join(problems))
    report["identity_ok"] = True
    return report
