"""The corpus loader must refuse exactly the damage that would train silently.

Every rejection tested here corresponds to a real failure mode: a state that
decodes to something else, a duplicate id resolved by read order, a fiber
describing a different molecule than its entry, or an empty read reported as
success. None of those raise on their own -- each produces a run that looks
healthy and optimizes the wrong thing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.data.corpus_training_library import (
    CorpusTrainingLibraryError,
    load_corpus_training_library,
    teacher_fiber_from_library_payload,
)

SOURCE_SHA = "a" * 64
TARGET_SHA = "b" * 64


def _alias(family: str = "atom_delete", table: str = "atom_delete", coordinate=(4,)) -> dict:
    return {
        "family_name": family,
        "table_name": table,
        "coordinate": list(coordinate),
    }


def _fiber(
    *,
    source_key: str = "CCO",
    target_key: str = "CO",
    aliases=None,
    source_sha: str = SOURCE_SHA,
    target_sha: str = TARGET_SHA,
) -> dict:
    return {
        "source_key": source_key,
        "target_key": target_key,
        "target_state_sha256": target_sha,
        "aliases": list(aliases if aliases is not None else [_alias()]),
        "state_support": {
            "source_key": source_key,
            "source_state_sha256": source_sha,
            "virtual_aliases": [],
        },
    }


def _state(n: int = 3) -> dict:
    """A real slot-exact payload, so round-trip verification is exercised."""

    return {
        "n_slots": n,
        "atom_types": [6] * n,
        "formal_charges": [0] * n,
        "implicit_h_counts": ([3] + [2] * (n - 2) + [3]) if n > 1 else [4],
        "bonds": [[i, i + 1, 1] for i in range(n - 1)],
    }


def _entry(entry_id: str, **overrides) -> dict:
    fiber = overrides.pop("teacher_successor_fiber", _fiber())
    entry = {
        "p50_entry_sha256": entry_id,
        "p50_compiled_entry_sha256": "f" + entry_id[1:],
        "teacher_successor_fiber": fiber,
        "source_state_sha256": SOURCE_SHA,
        "target_state_sha256": TARGET_SHA,
        "production_successor_alias_multiplicity": len(fiber["aliases"]) if fiber else 0,
        "exact_state": _state(),
        "support_time_hex": "0x1.0p+0",
        "model_family": "atom_delete",
        "capability_cell_id": "editing_v2_active8_v1:atom_delete:leaf_atom_removal",
    }
    entry.update(overrides)
    return entry


def _write_slice(root: Path, name: str, entries: list[dict]) -> None:
    directory = root / "chunks" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ENTRIES.json").write_text(
        json.dumps({"entries": entries, "entry_count": len(entries)})
    )


def _load(root: Path, **kwargs):
    # Round-trip verification stays ON: the fixture state is slot-exact, so a
    # regression in the guard would show up here rather than at corpus scale.
    kwargs.setdefault("verify_state_roundtrip", True)
    return load_corpus_training_library(
        [root], allow_reapable_roots=True, **kwargs
    )


def test_reads_entries_across_slices_in_deterministic_order(tmp_path: Path) -> None:
    _write_slice(tmp_path, "s2", [_entry("c" * 64), _entry("a" * 64)])
    _write_slice(tmp_path, "s1", [_entry("b" * 64)])
    library = _load(tmp_path)
    assert len(library) == 3
    assert [e.entry_id for e in library.entries] == ["a" * 64, "b" * 64, "c" * 64]
    assert library.slice_count == 2


def test_missing_teacher_fiber_is_refused(tmp_path: Path) -> None:
    """A terminal row carries no molecular jump; training it supervises nothing."""

    _write_slice(tmp_path, "s", [_entry("a" * 64, teacher_successor_fiber=None)])
    with pytest.raises(CorpusTrainingLibraryError, match="no teacher successor fiber"):
        _load(tmp_path)


def test_identical_duplicate_is_kept_once(tmp_path: Path) -> None:
    """The real library HAS 4 duplicate ids across two roots."""

    _write_slice(tmp_path, "s1", [_entry("a" * 64)])
    _write_slice(tmp_path, "s2", [_entry("a" * 64)])
    library = _load(tmp_path)
    assert len(library) == 1


def test_conflicting_duplicate_is_refused(tmp_path: Path) -> None:
    """Otherwise the winner is decided by directory read order."""

    _write_slice(tmp_path, "s1", [_entry("a" * 64)])
    _write_slice(
        tmp_path,
        "s2",
        [_entry("a" * 64, teacher_successor_fiber=_fiber(target_key="CCC"))],
    )
    with pytest.raises(CorpusTrainingLibraryError, match="twice with different content"):
        _load(tmp_path)


def test_fiber_disagreeing_on_the_source_state_is_refused(tmp_path: Path) -> None:
    _write_slice(
        tmp_path,
        "s",
        [_entry("a" * 64, teacher_successor_fiber=_fiber(source_sha="c" * 64))],
    )
    with pytest.raises(CorpusTrainingLibraryError, match="source state"):
        _load(tmp_path)


def test_fiber_disagreeing_on_the_target_state_is_refused(tmp_path: Path) -> None:
    _write_slice(
        tmp_path,
        "s",
        [_entry("a" * 64, teacher_successor_fiber=_fiber(target_sha="c" * 64))],
    )
    with pytest.raises(CorpusTrainingLibraryError, match="target state"):
        _load(tmp_path)


def test_alias_multiplicity_mismatch_is_refused(tmp_path: Path) -> None:
    """18.2% of teacher fibers are multi-alias, so this count is load-bearing."""

    _write_slice(
        tmp_path,
        "s",
        [_entry("a" * 64, production_successor_alias_multiplicity=3)],
    )
    with pytest.raises(CorpusTrainingLibraryError, match="teacher aliases"):
        _load(tmp_path)


def test_held_out_sources_are_dropped_and_counted(tmp_path: Path) -> None:
    _write_slice(
        tmp_path,
        "s",
        [
            _entry("a" * 64, teacher_successor_fiber=_fiber(source_key="CCO")),
            _entry("b" * 64, teacher_successor_fiber=_fiber(source_key="HELDOUT")),
        ],
    )
    library = _load(tmp_path, excluded_sources=["HELDOUT"])
    assert len(library) == 1
    assert library.excluded_entry_count == 1
    assert library.entries[0].source_key == "CCO"


def test_empty_read_is_refused_rather_than_reported_as_success(tmp_path: Path) -> None:
    """The silent-zero guard: a parse that matches nothing is a bug, not a result."""

    (tmp_path / "chunks").mkdir()
    with pytest.raises(CorpusTrainingLibraryError, match="refusing to report an empty library"):
        _load(tmp_path)


def test_all_entries_excluded_is_also_refused(tmp_path: Path) -> None:
    _write_slice(tmp_path, "s", [_entry("a" * 64)])
    with pytest.raises(CorpusTrainingLibraryError, match="refusing to report an empty library"):
        _load(tmp_path, excluded_sources=["CCO"])


def test_reapable_roots_are_refused_by_default(tmp_path: Path) -> None:
    """pytest's tmp_path lives where the OS reaps; the guard must fire."""

    _write_slice(tmp_path, "s", [_entry("a" * 64)])
    with pytest.raises(Exception) as error:
        load_corpus_training_library([tmp_path], verify_state_roundtrip=False)
    assert "corpus chunk root" in str(error.value) or "durable" in str(error.value).lower()


def test_inputs_for_preserves_requested_order_and_repetition(tmp_path: Path) -> None:
    """The sampler owns order; storage must not reimpose its own."""

    _write_slice(tmp_path, "s", [_entry("a" * 64), _entry("b" * 64)])
    library = _load(tmp_path)
    requested = ["b" * 64, "a" * 64, "b" * 64]
    _states, fibers, entries = library.inputs_for(requested)
    assert [e.entry_id for e in entries] == requested
    assert len(fibers) == 3


def test_unknown_entry_id_is_refused(tmp_path: Path) -> None:
    _write_slice(tmp_path, "s", [_entry("a" * 64)])
    library = _load(tmp_path)
    with pytest.raises(CorpusTrainingLibraryError, match="outside the library"):
        library.inputs_for(["z" * 64])


def test_malformed_alias_coordinate_is_refused(tmp_path: Path) -> None:
    """A coerced or negative coordinate would index a real action table."""

    bad = _fiber(aliases=[{"family_name": "atom_delete", "table_name": "atom_delete",
                          "coordinate": [-1]}])
    _write_slice(tmp_path, "s", [_entry("a" * 64, teacher_successor_fiber=bad)])
    with pytest.raises(CorpusTrainingLibraryError, match="coordinate is malformed"):
        _load(tmp_path)


def test_states_stay_serialized_rather_than_parsed(tmp_path: Path) -> None:
    """Holding parsed dicts measured 8.1 kB/entry -- the decoded cost.

    A parsed ``dict`` of small ints costs about as much as the decoded state,
    so keeping one to avoid the other saves nothing. The saving only exists
    while the payload stays as bytes, and nothing else in the type system says
    so, hence this test.
    """

    _write_slice(tmp_path, "s", [_entry("a" * 64)])
    entry = _load(tmp_path).entries[0]
    assert isinstance(entry.state_json, bytes)
    assert entry.state_payload() == _state()
    assert entry.state() is not None


def test_state_that_does_not_round_trip_is_refused(tmp_path: Path) -> None:
    """A slot layout that decodes to something else trains normally and wrongly.

    Here the payload claims 3 slots but lists a bond order the re-encode will
    not reproduce, so decode/encode is not the identity.
    """

    broken = _state()
    broken["bonds"] = [[0, 1, 1], [1, 2, 1], [0, 2, 0]]  # a zero-order bond is dropped
    _write_slice(tmp_path, "s", [_entry("a" * 64, exact_state=broken)])
    with pytest.raises(CorpusTrainingLibraryError, match="decode round trip"):
        _load(tmp_path)


def test_round_trip_check_can_be_skipped(tmp_path: Path) -> None:
    broken = _state()
    broken["bonds"] = [[0, 1, 1], [1, 2, 1], [0, 2, 0]]
    _write_slice(tmp_path, "s", [_entry("a" * 64, exact_state=broken)])
    assert len(_load(tmp_path, verify_state_roundtrip=False)) == 1


def test_stream_key_is_the_manifest_id_not_the_compiled_id(tmp_path: Path) -> None:
    """MEASURED: the two ids are never equal across the whole library.

    The prepared manifest's sequence is keyed on p50_entry_sha256, so that is
    what a training stream resolves against. Keying the library on the compile
    -side id instead resolves nothing -- which fails loudly, but only once
    something actually tries to draw a batch.
    """

    _write_slice(tmp_path, "s", [_entry("a" * 64)])
    library = _load(tmp_path)
    entry = library.entries[0]
    assert entry.entry_id == "a" * 64
    assert entry.compiled_entry_id == "f" + "a" * 63
    assert entry.entry_id != entry.compiled_entry_id
    assert "a" * 64 in library.index_by_entry_id
    assert entry.compiled_entry_id in library.index_by_compiled_entry_id


def test_decoder_agrees_with_the_p50_runtime_decoder() -> None:
    """Pin the duplicated decoder against the one it mirrors.

    This module re-implements the P50 runtime's private fiber decoder rather
    than importing a private symbol from a frozen module. Two decoders can
    drift, so assert they produce the identical object on a real payload shape.
    """

    p50 = pytest.importorskip("compose_v4.experiments.editing_v2_process_v2_p50_runtime")
    payload = _fiber(
        aliases=[
            _alias("atom_delete", "atom_delete", (4,)),
            _alias("bond_reroute", "reroute", (1, 2)),
        ]
    )
    assert teacher_fiber_from_library_payload(payload) == p50._teacher_fiber_from_payload(payload)
