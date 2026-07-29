"""A packed path must be INDISTINGUISHABLE from a replayed one on every accessor the trainer uses.

"The states match" is too weak a claim. The trainer reaches ``record.path.trace``, ``state_at``,
``path_length`` and ``operational_jump_rate``; ``_sample_tracelet_progress`` additionally reaches
``marginal`` and ``sample_progress``. Any of those diverging changes training while the state arrays look
fine, so each is compared directly against a replayed ``TraceProgressCTMC`` built from the same trace.

Uses real molecules (the analogue pool sample) rather than synthetic toys, because the failure modes that
have actually bitten this repo -- charge, aromaticity, fused rings, slot-stable deletes -- do not appear
in small hand-built graphs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from compose_v4.data.packed_trace_store import (  # noqa: E402
    PackedStoreError,
    PackedTraceProgress,
    build_packed_entry,
    pack_path,
    read_packed_shard,
    unpack_path,
    write_packed_shard,
)
from compose_v4.experiments.analogue_prior import rewrite_trace_from_record  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key  # noqa: E402
from compose_v4.rewrite.progress import TraceProgressCTMC  # noqa: E402

POOL = REPO / "diagnostics/composition/analogue_trace_pool.jsonl"


def _pairs(limit=25):
    """(replayed, packed) path pairs built from the same real traces."""
    if not POOL.exists():
        pytest.skip("local analogue pool sample unavailable")
    out = []
    for line in POOL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            trace = rewrite_trace_from_record(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
        replayed = TraceProgressCTMC(trace)
        out.append((replayed, unpack_path(trace, pack_path(replayed))))
        if len(out) >= limit:
            break
    if not out:
        pytest.skip("no rebuildable pool records")
    return out


def test_path_length_matches():
    for replayed, packed in _pairs():
        assert packed.path_length == replayed.path_length


def test_state_at_matches_exactly_at_every_progress():
    """Exact array equality AND canonical key equality, at every position including the endpoint."""
    for replayed, packed in _pairs():
        for progress in range(replayed.path_length + 1):
            a, b = packed.state_at(progress), replayed.state_at(progress)
            assert np.array_equal(a.atom_types, b.atom_types), f"atom_types differ at {progress}"
            assert np.array_equal(a.bonds, b.bonds), f"bonds differ at {progress}"
            assert np.array_equal(a.formal_charges, b.formal_charges), f"charges differ at {progress}"
            assert canonical_state_key(a) == canonical_state_key(b)


def test_trace_and_steps_are_preserved():
    for replayed, packed in _pairs():
        assert packed.trace is replayed.trace
        assert len(packed.trace.steps) == replayed.path_length
        for a, b in zip(packed.trace.steps, replayed.trace.steps):
            assert a.rule_name == b.rule_name and a.action == b.action


def test_marginal_matches_at_every_time():
    for replayed, packed in _pairs():
        for t in (0.0, 0.01, 0.3, 0.5, 0.77, 0.99, 1.0):
            assert np.allclose(packed.marginal(t), replayed.marginal(t), atol=1e-12, rtol=1e-12)


def test_operational_jump_rate_matches_at_every_progress():
    """The hazard term -- terminal supervision depends on it, and ~49% of draws are terminal."""
    for replayed, packed in _pairs():
        for progress in range(replayed.path_length + 1):
            assert packed.operational_jump_rate(progress) == pytest.approx(
                replayed.operational_jump_rate(progress), rel=1e-12, abs=1e-12
            )


def test_jump_rate_matches():
    for replayed, packed in _pairs(limit=10):
        for progress in range(replayed.path_length):
            for t in (0.2, 0.6, 0.9):
                assert packed.jump_rate(progress, t) == pytest.approx(
                    replayed.jump_rate(progress, t), rel=1e-12, abs=1e-12
                )


def test_sample_progress_distribution_matches():
    """Same RNG stream must yield the same progress -- inherited method, so this should be exact."""
    for replayed, packed in _pairs(limit=10):
        for t in (0.25, 0.75):
            r1 = np.random.default_rng(17)
            r2 = np.random.default_rng(17)
            for _ in range(200):
                assert packed.sample_progress(t, r2) == replayed.sample_progress(t, r1)


def test_states_property_matches():
    for replayed, packed in _pairs(limit=8):
        assert len(packed.states) == len(replayed.states)
        for a, b in zip(packed.states, replayed.states):
            assert canonical_state_key(a) == canonical_state_key(b)


# ---- fail-closed behaviour --------------------------------------------------------------------------


def test_wrong_state_count_is_refused():
    replayed, _ = _pairs(limit=1)[0]
    encoded = pack_path(replayed)
    with pytest.raises(PackedStoreError, match="expected one per progress position"):
        unpack_path(replayed.trace, encoded[:-1])


def test_mismatched_endpoint_is_refused():
    """Pairing a store with the wrong trace must fail loudly, not train on a silent mismatch."""
    pairs = _pairs(limit=6)
    (path_a, _), (path_b, _) = pairs[0], next(
        (p for p in pairs[1:] if p[0].path_length == pairs[0][0].path_length), (None, None)
    )
    if path_b is None:
        pytest.skip("no second trace of equal length in the sample")
    with pytest.raises(PackedStoreError, match="endpoint state does not equal"):
        unpack_path(path_a.trace, pack_path(path_b))


def test_roundtrip_through_a_written_shard(tmp_path):
    if not POOL.exists():
        pytest.skip("local analogue pool sample unavailable")
    records = [json.loads(line) for line in POOL.read_text().splitlines() if line.strip()][:6]
    from compose_v4.rewrite.trace_shard import encode_trace_record

    entries, expected = [], []
    for record in records:
        try:
            trace = rewrite_trace_from_record(record)
        except Exception:  # noqa: BLE001
            continue
        replayed = TraceProgressCTMC(trace)
        expected.append(replayed)
        entries.append(
            build_packed_entry(
                encode_trace_record(
                    trace, n_slots=40, seed=0, trace_id=f"t{len(entries)}",
                    partition="train", layer="mmp",
                ),
                replayed,
            )
        )
    if not entries:
        pytest.skip("no rebuildable pool records")

    shard = tmp_path / "packed_0000.jsonl.gz"
    manifest = write_packed_shard(shard, entries, provenance={"capability_hash": "abc123"})
    assert manifest["entries"] == len(entries)
    assert manifest["states"] == sum(p.path_length + 1 for p in expected)

    loaded = list(read_packed_shard(shard, expected_provenance={"capability_hash": "abc123"}))
    assert len(loaded) == len(entries)
    for (_, packed), replayed in zip(loaded, expected):
        assert isinstance(packed, PackedTraceProgress)
        assert packed.path_length == replayed.path_length
        for progress in range(replayed.path_length + 1):
            assert canonical_state_key(packed.state_at(progress)) == canonical_state_key(
                replayed.state_at(progress)
            )


def test_provenance_mismatch_is_refused(tmp_path):
    """A store built under different capability flags enumerates different candidates -- refuse it."""
    if not POOL.exists():
        pytest.skip("local analogue pool sample unavailable")
    from compose_v4.rewrite.trace_shard import encode_trace_record

    record = json.loads(POOL.read_text().splitlines()[0])
    replayed = TraceProgressCTMC(rewrite_trace_from_record(record))
    shard = tmp_path / "packed_0000.jsonl.gz"
    write_packed_shard(
        shard,
        [
            build_packed_entry(
                encode_trace_record(
                    replayed.trace, n_slots=40, seed=0, trace_id="t0",
                    partition="train", layer="mmp",
                ),
                replayed,
            )
        ],
        provenance={"capability_hash": "built_under_this"},
    )
    with pytest.raises(PackedStoreError, match="provenance mismatch on capability_hash"):
        list(read_packed_shard(shard, expected_provenance={"capability_hash": "trained_under_that"}))


def test_missing_manifest_is_refused(tmp_path):
    shard = tmp_path / "packed_0000.jsonl.gz"
    shard.write_bytes(b"")
    with pytest.raises(PackedStoreError, match="no manifest"):
        list(read_packed_shard(shard))


# ---- sampler contract: the closed form must not outlive its assumptions ------------------------------


def test_sampler_contract_records_the_scheduler_identity():
    from compose_v4.data.packed_trace_store import PROGRESS_SAMPLER_VERSION, sampler_contract

    contract = sampler_contract()
    assert contract["scheduler_type"] == "PowerSurvivalScheduler"
    assert contract["scheduler_power"] == 1.0
    assert contract["progress_sampler_version"] == PROGRESS_SAMPLER_VERSION
    assert len(contract["time_sampling_implementation_hash"]) == 16


def test_non_unit_scheduler_power_blocks_the_packed_path(monkeypatch):
    """A future scheduler change must invalidate the closed form, not silently reuse unit-power logic."""
    import compose_v4.rewrite.progress as progress_module
    from compose_v4.data.packed_trace_store import PackedStoreError, assert_closed_form_applies

    original = progress_module.PowerSurvivalScheduler
    # NB: subclassing and setting `power = 2.0` would NOT work -- the dataclass __init__ reassigns the
    # original field default. Patch with a factory that builds a genuinely non-unit scheduler.
    monkeypatch.setattr(progress_module, "PowerSurvivalScheduler", lambda: original(power=2.0))
    with pytest.raises(PackedStoreError, match="assumes alpha\\(t\\) == t"):
        assert_closed_form_applies()


def test_stored_contract_mismatch_is_refused_on_load(tmp_path, monkeypatch):
    """A store built under one sampling law must not load under another."""
    if not POOL.exists():
        pytest.skip("local analogue pool sample unavailable")
    import compose_v4.data.packed_trace_store as store
    from compose_v4.rewrite.trace_shard import encode_trace_record

    record = json.loads(POOL.read_text().splitlines()[0])
    replayed = TraceProgressCTMC(rewrite_trace_from_record(record))
    shard = tmp_path / "packed_0000.jsonl.gz"
    store.write_packed_shard(
        shard,
        [
            store.build_packed_entry(
                encode_trace_record(
                    replayed.trace, n_slots=40, seed=0, trace_id="t0",
                    partition="train", layer="mmp",
                ),
                replayed,
            )
        ],
        provenance={},
    )
    monkeypatch.setattr(store, "PROGRESS_SAMPLER_VERSION", 99)
    with pytest.raises(store.PackedStoreError, match="sampler contract mismatch"):
        list(store.read_packed_shard(shard))


def test_sentinel_replay_is_deterministic_and_passes_on_a_good_store(tmp_path):
    """Fixed first-N replay: same entries every run, so a pass/failure means the same thing each time."""
    if not POOL.exists():
        pytest.skip("local analogue pool sample unavailable")
    import compose_v4.data.packed_trace_store as store
    from compose_v4.rewrite.trace_shard import encode_trace_record

    entries = []
    for i, line in enumerate(POOL.read_text().splitlines()[:5]):
        if not line.strip():
            continue
        replayed = TraceProgressCTMC(rewrite_trace_from_record(json.loads(line)))
        entries.append(
            store.build_packed_entry(
                encode_trace_record(
                    replayed.trace, n_slots=40, seed=0, trace_id=f"t{i}",
                    partition="train", layer="mmp",
                ),
                replayed,
            )
        )
    shard = tmp_path / "packed_0000.jsonl.gz"
    store.write_packed_shard(shard, entries, provenance={})
    first = store.sentinel_replay_check(shard, entries=3)
    second = store.sentinel_replay_check(shard, entries=3)
    assert first == second, "the sentinel must be deterministic, not sampled"
    assert first["sentinel_entries_replayed"] == 3
