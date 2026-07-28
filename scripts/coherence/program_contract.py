#!/usr/bin/env python3
"""Program-coherence audit §2: the ONE authoritative production-contract object + deterministic fingerprint.

Assembles every contract-critical field of the RING_CORE_V1 production system from the authoritative code
constants (not chat/memory) and computes a single ``production_contract_fingerprint``. Production paths
(trainer, validator, sampler, loader, dry-launch, rollout harness, paper-config generator) should each emit +
verify this fingerprint so a drifted consumer fails loudly. Read-only: derives from existing constants only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ring_core_identity as _rci  # noqa: E402
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    MARK_RULE_NAMES,
    OperatorCapabilities,
)

REPO = Path(__file__).resolve().parent.parent.parent


def _hash_sources(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(paths):
        fp = REPO / rel
        digest.update(rel.encode())
        digest.update(fp.read_bytes() if fp.exists() else b"<MISSING>")
    return digest.hexdigest()[:16]


def build_contract() -> dict:
    caps = OperatorCapabilities(
        compute_ring_grow_support=False, compute_ring_restates=True,
        compute_cyclic_graft=True, compute_ring_opening=True,
    )
    return {
        "program": "RING_CORE_V1",
        # 1. state space
        "state_space_version": "broad_organic_v1",
        "max_atoms": _rci.MAX_ATOMS,
        "atom_vocabulary": "ORGANIC_VOCABULARY",
        "corpus_scope_hash": BROAD_ORGANIC_V1.scope_hash(),
        "charge_policy": "retain_representable_charges__charge_preserving",
        "standardization_hash": _hash_sources([
            "src/compose_v4/chem/molecular_graph.py", "src/compose_v4/chem/state.py"]),
        # 2. operators / executor / successor
        "operator_registry_hash": _rci.recompute_operator_registry_hash(),
        "mark_rule_names": list(MARK_RULE_NAMES),
        "capability_hash": _rci.CAPABILITY_HASH,
        "cycle_op_semantic_hash": _rci.recompute_cycle_op_semantic_hash(),
        "executor_hash": _hash_sources(["src/compose_v4/rewrite/kernel.py"]),
        "canonical_successor_hash": _hash_sources(["src/compose_v4/rewrite/kernel.py"]),
        # 3. ring process
        "enable_cycle_ops": True,
        "enable_ring_macros": False,
        "enable_ring_grow_macro": False,
        "ring_catalog_fingerprint": "639ff6078c32d43c",
        # 4. data recipe
        "corruption_policy_hash": _hash_sources([
            "src/compose_v4/rewrite/source_corruption.py",
            "src/compose_v4/experiments/corrupted_source_prior.py"]),
        "teacher_filter_version": _rci_teacher_filter_version(),
        "eval_operator_capability_fingerprint": caps.fingerprint(),
        "calibration_policy_hash": _rci.CALIBRATION_POLICY_HASH,
        "denovo_keep": 0,
        "sampler_mixture": {"corruption": 0.55, "mmp": 0.45, "denovo": 0.0},
        "curriculum_bin_edges": [5, 9, 13],
        # 5. model / warm-start / checkpoint
        "model_architecture_version": "factorized_tracelet_rate_model_v1",
        "base_b_sha256": _rci.BASE_B_SHA256,
        "editing_process_type": "fixed_step_embedded_jump_chain__source_conditioned",
        "evaluation_batch_cache_format_version": 2,
    }


def _rci_teacher_filter_version() -> int:
    from compose_v4.rewrite.source_corruption import TEACHER_REPRESENTABILITY_FILTER_VERSION
    return int(TEACHER_REPRESENTABILITY_FILTER_VERSION)


def contract_fingerprint(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:24]


def main() -> int:
    contract = build_contract()
    fp = contract_fingerprint(contract)
    out_dir = REPO / "diagnostics/coherence"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "program_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (out_dir / "program_contract_fingerprint.txt").write_text(fp + "\n")
    print(json.dumps({"production_contract_fingerprint": fp,
                      "operator_registry_hash": contract["operator_registry_hash"],
                      "capability_hash": contract["capability_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
