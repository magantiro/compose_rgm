#!/usr/bin/env python3
"""Build the RING_CORE_V1 scaled edit manifest (zero-mixture: denovo weight 0, NO de-novo path cache).

Bases on the measured broad-organic mining statistics in the existing max_atoms=40 manifest (corruption/MMP
layer weights + curriculum bin edges + census), then corrects the provenance for the compositional-ring core:
disables the legacy ring_system_grow, enables the compositional cycle_close/cycle_open (cycle_insert/
cycle_attach) production families, injects the frozen RingCore-V1 capability + hashes (from ring_core_identity),
declares the cycle-op supervision folded into the corruption layer at train time, and points the MMP pool at
the production volume path. The gate consumes only locked_mixture.{layer_weights,curriculum_bin_edges}; the
rest is the provenance the §1/§2 manifest gates verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.model.factorized_tracelet_rate_model import production_enabled_families

import ring_core_identity as I
from compose_v4.model.factorized_tracelet_rate_model import OperatorCapabilities
from compose_v4.rewrite.source_corruption import TEACHER_REPRESENTABILITY_FILTER_VERSION

# The RING_CORE_V1 eval/training editing capabilities (grow off; restate/graft/ring-opening on).
_RING_CORE_CAPABILITIES = OperatorCapabilities(
    compute_ring_grow_support=False, compute_ring_restates=True,
    compute_cyclic_graft=True, compute_ring_opening=True,
)

REPO = Path(__file__).resolve().parent.parent
_BASE = REPO / "diagnostics/composition/scaled_edit_data_manifest_40.json"
_VOLUME_POOL = "/artifacts/edit_mining_full_broad_40/edit_pool_full.jsonl"

# RingCore-V1 production-enabled operator families: NO ring_system_grow (legacy macro disabled); the
# compositional cycle_close/cycle_open live on the cycle_insert/cycle_attach slots.
_RING_CORE_PRODUCTION_ENABLED = production_enabled_families(
    enable_cycle_ops=True, enable_ring_grow_macro=False
)


def build(base_path: Path, out_path: Path) -> dict:
    base = json.loads(base_path.read_text())
    manifest = dict(base)
    manifest["kind"] = "ring_core_v1_scaled_edit_manifest"
    manifest["manifest_version"] = "ring_core_v1"
    manifest["source_commit_tag"] = "ring-core-v1"

    # Capability provenance (frozen RING_CORE_V1 identity).
    manifest["ring_core_capability"] = {
        "capability_hash": I.CAPABILITY_HASH,
        "operator_registry_hash": I.recompute_operator_registry_hash(),
        "cycle_op_semantic_hash": I.recompute_cycle_op_semantic_hash(),
        "calibration_policy_hash": I.CALIBRATION_POLICY_HASH,
        "enable_cycle_ops": True,
        "enable_ring_macros": False,
        "enable_ring_grow_macro": False,
        "max_atoms": I.MAX_ATOMS,
        "corpus_scope_hash": I.SCOPE_HASH,
        "base_b_sha256": I.BASE_B_SHA256,
    }
    # Operator registry corrected for the compositional core.
    manifest["operator_registry"] = {
        "mark_rule_names": list(base.get("operator_registry", {}).get("mark_rule_names", [])),
        "production_enabled": list(_RING_CORE_PRODUCTION_ENABLED),
        "legacy_disabled": ["ring_system_grow"],
        "hash": I.recompute_operator_registry_hash(),
    }
    # Cycle-op supervision: generated in-memory at train time (--cycle-op-mix) and folded into the corruption
    # layer, both directions, round-trip verified; calibrated per the frozen calibration policy.
    # Teacher-in-exact-candidates filter (declared production data contract): unrepresentable corruption
    # teachers (measured: ~2% -- all inverse/grow ring_system_restate on fused rings) are excluded at the
    # data source so every selected teacher belongs to the model's exact dynamic candidate set.
    manifest["teacher_representability_filter"] = {
        "version": TEACHER_REPRESENTABILITY_FILTER_VERSION,
        "invariant": "every selected teacher action belongs to the exact dynamic candidate set for its state",
        "eval_operator_capability_fingerprint": _RING_CORE_CAPABILITIES.fingerprint(),
        "characterization": "diagnostics/production_preflight/teacher_filter_characterization.json",
    }
    manifest["cycle_op_supervision"] = {
        "generator": "compose_v4.experiments.cycle_op_prior.build_cycle_op_records",
        "families": ["cycle_insert", "cycle_attach"],
        "executor_rule_names": ["bond_insert", "bond_delete"],
        "both_directions": True,
        "folded_into_layer": "corruption",
        "calibration_policy_hash": I.CALIBRATION_POLICY_HASH,
        "cycle_family_selected_target_rate": 0.61,
        "note": "cycle records are regenerated in-memory at train time from the corruption source SMILES, "
        "so their count scales with --corrupted-prior-count; the corruption layer weight covers them.",
    }
    # Zero-mixture: no de-novo layer (denovo weight 0); the gate forces denovo_keep=0 under --scaled-manifest.
    lm = dict(manifest.get("locked_mixture", {}))
    lm.setdefault("layer_weights", {})
    lm["layer_weights"] = {k: v for k, v in lm["layer_weights"].items() if k != "denovo"}
    lm["denovo_weight"] = 0.0
    lm["zero_mixture"] = True
    manifest["locked_mixture"] = lm

    # Point the MMP pool at the production volume path (the base manifest carried a stale scratch path).
    mmp = dict(manifest.get("mmp_pool", {}))
    mmp["volume_path"] = _VOLUME_POOL
    manifest["mmp_pool"] = mmp

    manifest["notes"] = (
        "RING_CORE_V1 zero-mixture manifest: denovo_weight=0 so no de-novo path cache is opened; cycle ops "
        "define ring support (legacy ring_system_grow disabled). Measured corruption/MMP statistics + bin "
        "edges inherited from the broad-organic max_atoms=40 mining. NON_SCIENTIFIC_PREFLIGHT provenance."
    )
    manifest["manifest_self_sha256"] = None
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    manifest["manifest_self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=_BASE)
    parser.add_argument(
        "--out", type=Path, default=REPO / "diagnostics/production_preflight/ring_core_v1_scaled_manifest.json"
    )
    args = parser.parse_args()
    m = build(args.base, args.out)
    print(json.dumps({
        "kind": m["kind"],
        "capability_hash": m["ring_core_capability"]["capability_hash"],
        "operator_registry_hash": m["operator_registry"]["hash"],
        "production_enabled": m["operator_registry"]["production_enabled"],
        "denovo_weight": m["locked_mixture"]["denovo_weight"],
        "layer_weights": m["locked_mixture"]["layer_weights"],
        "mmp_volume_path": m["mmp_pool"]["volume_path"],
        "output": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
