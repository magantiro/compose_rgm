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

import hashlib
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from compose_v4.data.packed_trace_store import (
    PackedStoreError,
    PackedTraceProgress,
    build_packed_entry,
    pack_path,
    read_addressed_packed_shard,
    read_frozen_source_addressed_packed_shard,
    read_packed_shard,
    unpack_path,
    write_packed_shard,
)
from compose_v4.experiments.analogue_prior import rewrite_trace_from_record
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteTrace

_FIXTURE = Path(__file__).resolve().parent / "fixtures/analogue_trace_pool_sample.jsonl"
_FULL = Path(__file__).resolve().parent.parent / "diagnostics/composition/analogue_trace_pool.jsonl"
# Prefer the COMMITTED fixture so these tests run in a clean checkout -- the authoritative launch gate
# runs from a clean worktree, where an untracked pool would silently skip them.
POOL = _FIXTURE if _FIXTURE.exists() else _FULL


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
        except Exception:  # noqa: BLE001, S112
            continue
        replayed = TraceProgressCTMC(trace)
        out.append((replayed, unpack_path(trace, pack_path(replayed))))
        if len(out) >= limit:
            break
    if not out:
        pytest.skip("no rebuildable pool records")
    return out


def _write_address_fixture(
    tmp_path: Path,
    *,
    trace_ids: tuple[str, ...],
    layer: str = "mmp_analogue",
) -> tuple[Path, list[dict]]:
    """Write real packed rows with controlled immutable trace-envelope IDs."""

    if not POOL.exists():
        pytest.skip("local analogue pool sample unavailable")
    from compose_v4.rewrite.trace_shard import encode_trace_record

    entries: list[dict] = []
    for line in POOL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            trace = rewrite_trace_from_record(json.loads(line))
        except Exception:  # noqa: BLE001, S112
            continue
        replayed = TraceProgressCTMC(trace)
        entries.append(
            build_packed_entry(
                encode_trace_record(
                    trace,
                    n_slots=40,
                    seed=0,
                    trace_id=trace_ids[len(entries)],
                    partition="train",
                    layer=layer,
                ),
                replayed,
            )
        )
        if len(entries) == len(trace_ids):
            break
    if len(entries) != len(trace_ids):
        pytest.skip("not enough rebuildable pool records")
    shard = tmp_path / "shard_0000.jsonl.gz"
    write_packed_shard(shard, entries, provenance={"capability_hash": "address-test"})
    return shard, entries


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
            assert np.array_equal(a.formal_charges, b.formal_charges), (
                f"charges differ at {progress}"
            )
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
    (path_a, _), (path_b, _) = (
        pairs[0],
        next((p for p in pairs[1:] if p[0].path_length == pairs[0][0].path_length), (None, None)),
    )
    if path_b is None:
        pytest.skip("no second trace of equal length in the sample")
    with pytest.raises(PackedStoreError, match="endpoint state does not equal"):
        unpack_path(path_a.trace, pack_path(path_b))


def test_endpoint_implicit_h_mismatch_is_refused() -> None:
    replayed, _ = _pairs(limit=1)[0]
    target = replayed.trace.target
    mismatched_target = type(target)(
        atom_types=target.atom_types.copy(),
        formal_charges=target.formal_charges.copy(),
        implicit_h_counts=target.implicit_h_counts.copy(),
        bonds=target.bonds.copy(),
    )
    mismatched_target.implicit_h_counts[0] += 1
    trace = RewriteTrace(
        source=replayed.trace.source,
        target=mismatched_target,
        steps=replayed.trace.steps,
        metadata=replayed.trace.metadata,
    )
    with pytest.raises(PackedStoreError, match="endpoint state does not equal"):
        PackedTraceProgress(trace, replayed.states)


def test_roundtrip_through_a_written_shard(tmp_path):
    if not POOL.exists():
        pytest.skip("local analogue pool sample unavailable")
    records = [json.loads(line) for line in POOL.read_text().splitlines() if line.strip()][:6]
    from compose_v4.rewrite.trace_shard import encode_trace_record

    entries, expected = [], []
    for record in records:
        try:
            trace = rewrite_trace_from_record(record)
        except Exception:  # noqa: BLE001, S112
            continue
        replayed = TraceProgressCTMC(trace)
        expected.append(replayed)
        entries.append(
            build_packed_entry(
                encode_trace_record(
                    trace,
                    n_slots=40,
                    seed=0,
                    trace_id=f"t{len(entries)}",
                    partition="train",
                    layer="mmp",
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


def test_addressed_reader_roundtrips_exact_row_identity_and_projects_legacy_pairs(
    tmp_path,
):
    shard, entries = _write_address_fixture(
        tmp_path,
        trace_ids=("trace-a", "trace-b", "trace-c"),
    )
    addressed = list(
        read_addressed_packed_shard(
            shard,
            expected_provenance={"capability_hash": "address-test"},
        )
    )

    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    assert [row.address.entry_index for row in addressed] == [0, 1, 2]
    assert len({row.address for row in addressed}) == len(addressed)
    assert len({row.address.trace_id for row in addressed}) == len(addressed)
    for index, (row, entry) in enumerate(zip(addressed, entries)):
        envelope = entry["trace"]
        assert row.address.packed_shard_content_sha256 == digest
        assert row.address.packed_shard_name == shard.name
        assert row.address.entry_index == index
        assert row.address.trace_id == envelope["trace_id"]
        assert row.address.layer == envelope["layer"]
        assert row.address.partition == envelope["partition"]
        assert row.address.source_key == envelope["source_key"]
        assert row.address.target_key == envelope["target_key"]
        assert row.address.path_length == envelope["path_length"]
        assert row.path.trace is row.trace
        assert row.path.path_length == row.address.path_length

    with pytest.raises(FrozenInstanceError):
        addressed[0].address.entry_index = 99

    legacy = list(
        read_packed_shard(
            shard,
            expected_provenance={"capability_hash": "address-test"},
        )
    )
    assert len(legacy) == len(addressed)
    for (legacy_trace, legacy_path), row in zip(legacy, addressed):
        assert legacy_trace.steps == row.trace.steps
        assert legacy_path.path_length == row.path.path_length
        for progress in range(row.path.path_length + 1):
            legacy_state = legacy_path.state_at(progress)
            addressed_state = row.path.state_at(progress)
            assert np.array_equal(
                legacy_state.atom_types,
                addressed_state.atom_types,
            )
            assert np.array_equal(legacy_state.bonds, addressed_state.bonds)
            assert np.array_equal(
                legacy_state.formal_charges,
                addressed_state.formal_charges,
            )


def test_addressed_reader_rejects_duplicate_trace_ids_without_tightening_legacy_reader(
    tmp_path,
):
    shard, _ = _write_address_fixture(
        tmp_path,
        trace_ids=("duplicate-id", "duplicate-id"),
    )
    with pytest.raises(PackedStoreError, match="duplicate trace_id"):
        list(read_addressed_packed_shard(shard))
    assert len(list(read_packed_shard(shard))) == 2


def test_addressed_reader_requires_identity_fields_but_legacy_reader_remains_compatible(
    tmp_path,
):
    shard, entries = _write_address_fixture(tmp_path, trace_ids=("legacy-row",))
    entries[0]["trace"].pop("layer")
    write_packed_shard(shard, entries, provenance={"capability_hash": "address-test"})

    assert len(list(read_packed_shard(shard))) == 1
    with pytest.raises(PackedStoreError, match="'layer'.*non-empty string"):
        list(read_addressed_packed_shard(shard))


def test_addressed_reader_keeps_rdkit_endpoint_checks_on_explicit_audit_path(
    tmp_path,
):
    shard, entries = _write_address_fixture(tmp_path, trace_ids=("tampered-key",))
    entries[0]["trace"]["source_key"] = "not-the-decoded-source"
    write_packed_shard(shard, entries, provenance={"capability_hash": "address-test"})

    # The production path binds the stored envelope to exact shard bytes but
    # performs no per-row RDKit canonicalization.
    addressed = list(read_addressed_packed_shard(shard))
    assert addressed[0].address.source_key == "not-the-decoded-source"

    # Corpus certification can request sampled or exhaustive semantic replay.
    with pytest.raises(PackedStoreError, match="source_key.*decoded"):
        list(read_addressed_packed_shard(shard, verify_fraction=1.0))


def test_addressed_reader_reconciles_contiguous_indices_with_manifest_count(tmp_path):
    shard, _ = _write_address_fixture(
        tmp_path,
        trace_ids=("trace-a", "trace-b"),
    )
    manifest_path = shard.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(PackedStoreError, match="manifest declares 3 entries"):
        list(read_addressed_packed_shard(shard))
    assert len(list(read_packed_shard(shard))) == 2


def test_frozen_source_reader_migrates_exact_bytes_without_authorizing_stale_overlay(
    tmp_path,
):
    shard, _ = _write_address_fixture(tmp_path, trace_ids=("frozen-source",))
    manifest_path = shard.with_suffix(".manifest.json")
    overlay_path = Path(str(shard) + ".provenance.json")
    overlay_path.write_text(
        json.dumps(
            {
                "schema": "compose.data.provenance_overlay",
                "schema_version": 1,
                "upgrade_implementation_hash": "historical-contract",
            },
            sort_keys=True,
        )
    )
    shard_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    overlay_sha256 = hashlib.sha256(overlay_path.read_bytes()).hexdigest()

    with pytest.raises(PackedStoreError, match="invalid provenance overlay"):
        list(read_addressed_packed_shard(shard))

    migrated = list(
        read_frozen_source_addressed_packed_shard(
            shard,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_overlay_sha256=overlay_sha256,
        )
    )
    assert len(migrated) == 1
    assert migrated[0].address.packed_shard_content_sha256 == shard_sha256

    with pytest.raises(PackedStoreError, match="manifest SHA-256 mismatch"):
        list(
            read_frozen_source_addressed_packed_shard(
                shard,
                expected_shard_sha256=shard_sha256,
                expected_manifest_sha256="0" * 64,
                expected_overlay_sha256=overlay_sha256,
            )
        )
    with pytest.raises(PackedStoreError, match="provenance-overlay SHA-256 mismatch"):
        list(
            read_frozen_source_addressed_packed_shard(
                shard,
                expected_shard_sha256=shard_sha256,
                expected_manifest_sha256=manifest_sha256,
                expected_overlay_sha256="0" * 64,
            )
        )


def test_frozen_source_range_reader_preserves_addresses_and_decodes_only_range(
    tmp_path,
    monkeypatch,
):
    import compose_v4.data.packed_trace_store as packed_store

    shard, _ = _write_address_fixture(
        tmp_path,
        trace_ids=("range-a", "range-b", "range-c", "range-d"),
    )
    manifest_path = shard.with_suffix(".manifest.json")
    shard_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    decoded_trace_ids = []
    original_decode = packed_store.decode_packed_trace

    def counted_decode(record, states):
        decoded_trace_ids.append(record["trace_id"])
        return original_decode(record, states)

    monkeypatch.setattr(packed_store, "decode_packed_trace", counted_decode)
    ranged = list(
        read_frozen_source_addressed_packed_shard(
            shard,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_overlay_sha256=None,
            entry_start=1,
            entry_stop=3,
        )
    )
    assert [row.address.entry_index for row in ranged] == [1, 2]
    assert [row.address.trace_id for row in ranged] == ["range-b", "range-c"]
    assert decoded_trace_ids == ["range-b", "range-c"]


def test_frozen_source_range_reader_still_reconciles_full_manifest_census(
    tmp_path,
):
    shard, _ = _write_address_fixture(
        tmp_path,
        trace_ids=("range-a", "range-b", "range-c"),
    )
    manifest_path = shard.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"] = 4
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(PackedStoreError, match="manifest declares 4 entries"):
        list(
            read_frozen_source_addressed_packed_shard(
                shard,
                expected_shard_sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
                expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                expected_overlay_sha256=None,
                entry_start=1,
                entry_stop=2,
            )
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
                    replayed.trace,
                    n_slots=40,
                    seed=0,
                    trace_id="t0",
                    partition="train",
                    layer="mmp",
                ),
                replayed,
            )
        ],
        provenance={"capability_hash": "built_under_this"},
    )
    with pytest.raises(PackedStoreError, match="provenance mismatch on capability_hash"):
        list(
            read_packed_shard(shard, expected_provenance={"capability_hash": "trained_under_that"})
        )


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
                    replayed.trace,
                    n_slots=40,
                    seed=0,
                    trace_id="t0",
                    partition="train",
                    layer="mmp",
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
                    replayed.trace,
                    n_slots=40,
                    seed=0,
                    trace_id=f"t{i}",
                    partition="train",
                    layer="mmp",
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


# ---- manifest naming: the convention must match what the precompile app actually writes ---------------


def test_manifest_path_uses_the_repo_convention():
    """`.jsonl.gz` -> `.jsonl.manifest.json`, NOT `.jsonl.gz.manifest.json`.

    Regression: source_shard_fingerprint originally built the path by string concatenation, so it looked
    for `shard_0000.jsonl.gz.manifest.json` while precompile_edit_data_app writes
    `shard_0000.jsonl.manifest.json` via Path.with_suffix. Every shard of the Modal build failed on it.
    The local fixture had masked the bug because staging it RENAMED the manifest to the wrong convention.
    """
    from compose_v4.data.packed_trace_store import manifest_path_for

    assert (
        manifest_path_for(Path("a/b/shard_0000.jsonl.gz")).name == "shard_0000.jsonl.manifest.json"
    )
    assert manifest_path_for(Path("shard_0007.jsonl.gz")).name == "shard_0007.jsonl.manifest.json"


def test_fingerprint_reads_a_manifest_written_the_way_the_precompile_app_writes_it(tmp_path):
    """Build the sidecar exactly as precompile_edit_data_app does and require the fingerprint to find it."""
    from compose_v4.data.packed_trace_store import source_shard_fingerprint

    shard = tmp_path / "shard_0000.jsonl.gz"
    shard.write_bytes(b"")
    # precompile_edit_data_app.compile_shard: shard_path.with_suffix(".manifest.json")
    shard.with_suffix(".manifest.json").write_text(json.dumps({"content_sha256": "abc123def456"}))
    assert source_shard_fingerprint(shard) == "abc123def456"


def test_packed_store_manifest_is_discoverable_by_the_same_rule(tmp_path):
    """The packed store must use ONE convention with the audit shards, so either can be fingerprinted."""
    import compose_v4.data.packed_trace_store as store

    if not POOL.exists():
        pytest.skip("local analogue pool sample unavailable")
    from compose_v4.rewrite.trace_shard import encode_trace_record

    replayed = TraceProgressCTMC(
        rewrite_trace_from_record(json.loads(POOL.read_text().splitlines()[0]))
    )
    shard = tmp_path / "packed_0000.jsonl.gz"
    store.write_packed_shard(
        shard,
        [
            store.build_packed_entry(
                encode_trace_record(
                    replayed.trace,
                    n_slots=40,
                    seed=0,
                    trace_id="t0",
                    partition="train",
                    layer="mmp",
                ),
                replayed,
            )
        ],
        provenance={},
    )
    assert (tmp_path / "packed_0000.jsonl.manifest.json").exists()
    assert not (tmp_path / "packed_0000.jsonl.gz.manifest.json").exists()
    assert len(list(store.read_packed_shard(shard))) == 1


def test_no_module_reimplements_the_manifest_path():
    """One convention, one helper.

    The sidecar-naming bug had TWO homes: the library helper and a duplicate string-concatenation copy
    inside the Modal reducer. Fixing only the library left the reducer reporting all 45 packed shards as
    missing. This forbids the duplicate from reappearing anywhere.
    """
    import re

    repo = Path(__file__).resolve().parent.parent
    offenders = [
        str(path.relative_to(repo))
        for path in list((repo / "src").rglob("*.py")) + list((repo / "modal_apps").rglob("*.py"))
        if re.search(r'\+\s*"\.manifest\.json"', path.read_text())
    ]
    assert not offenders, (
        f"these modules build a manifest path by concatenation instead of manifest_path_for: {offenders}"
    )
