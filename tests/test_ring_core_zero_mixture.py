"""Local fixture-based regression tests for the RING_CORE_V1 zero-mixture (--scaled-manifest) gate path
(owner mandate §5). These prove the LOGIC without the full GuacaMol volume: the scaled manifest is
zero-mixture (denovo_weight=0, no grow, cycle ops enabled, frozen hashes); the manifest sampler builds NO
de-novo layer at denovo_keep=0; the production ring catalog reconstructs deterministically from the seeds
(not de-novo paths); and the instrumentation counters behave. The corpus-dependent proof (a real gate run
emits denovo_path_compile_calls=0 + reaches the first editing batch) is the CPU Modal dry-launch (§8)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compose_v4.experiments import zero_mixture_instrumentation as zmi  # noqa: E402

_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "diagnostics/production_preflight/ring_core_v1_scaled_manifest.json"
)


def test_instrumentation_counters():
    zmi.reset()
    assert zmi.snapshot()["denovo_path_compile_calls"] == 0
    ok, _ = zmi.zero_mixture_ok()
    assert not ok  # edit counters not yet positive
    zmi.bump("edit_dataset_constructions")
    zmi.bump("edit_batches_emitted")
    zmi.bump("validation_edit_batches_emitted")
    ok, report = zmi.zero_mixture_ok()
    assert ok and report["denovo_all_zero"] and report["edit_path_reached"]
    zmi.bump("denovo_path_compile_calls")
    ok2, _ = zmi.zero_mixture_ok()
    assert not ok2  # any de-novo work breaks the invariant
    zmi.reset()


def test_manifest_is_zero_mixture_ring_core():
    from ring_core_identity import (
        CALIBRATION_POLICY_HASH,
        CAPABILITY_HASH,
        recompute_operator_registry_hash,
    )

    m = json.loads(_MANIFEST.read_text())
    assert m["locked_mixture"]["denovo_weight"] == 0.0
    assert "denovo" not in m["locked_mixture"]["layer_weights"]
    pe = m["operator_registry"]["production_enabled"]
    assert "ring_system_grow" not in pe
    assert "cycle_insert" in pe and "cycle_attach" in pe
    cap = m["ring_core_capability"]
    assert cap["capability_hash"] == CAPABILITY_HASH
    assert cap["operator_registry_hash"] == recompute_operator_registry_hash()
    assert cap["calibration_policy_hash"] == CALIBRATION_POLICY_HASH
    assert cap["enable_cycle_ops"] and not cap["enable_ring_macros"]
    assert not cap["enable_ring_grow_macro"]


def test_scaled_sampler_has_no_denovo_layer_at_denovo_keep_zero():
    # At denovo_keep=0 the "denovo" layer key is never created, so no de-novo record can be sampled -- the
    # zero-mixture invariant at the sampler boundary. Uses real edit records (corruption + cycle).
    import numpy as np

    from compose_v4.experiments.cycle_op_prior import build_cycle_op_records
    from train_tracelet_cnof_gate import _build_scaled_manifest_sampler

    records, _ = build_cycle_op_records(
        ["c1ccc2ccccc2c1CCN", "C1CCNCC1C(=O)O", "CSc1ccc(N)cc1CC"], n_slots=40, seed=0
    )
    edit_records = tuple(records)
    assert len(edit_records) >= 8
    n_corruption = len(edit_records) // 2
    sampler = _build_scaled_manifest_sampler(
        edit_records, denovo_keep=0, n_corruption=n_corruption, manifest_path=str(_MANIFEST), seed=0
    )
    assert sampler is not None
    rng = np.random.default_rng(0)
    for _ in range(200):
        idx = int(sampler.draw(rng))
        assert 0 <= idx < len(edit_records)  # every draw is an edit record; no de-novo region exists


def test_seed_ring_catalog_reconstructs_deterministically():
    from train_tracelet_cnof_gate import _build_ring_core_seed_ring_catalog
    from compose_v4.rewrite.typed_ring_catalog import ring_catalog_fingerprint

    a = _build_ring_core_seed_ring_catalog(40)
    b = _build_ring_core_seed_ring_catalog(40)
    assert ring_catalog_fingerprint(a) == ring_catalog_fingerprint(b)  # deterministic
    assert len(a.statistics()) > 0
