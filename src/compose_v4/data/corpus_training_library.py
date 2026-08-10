"""Corpus-scale training inputs read from the frozen chunk library.

WHY THIS EXISTS
---------------
``load_semantic_t1_prepared_inputs`` reads ONE canonical JSON holding a full
``successor_partition`` per entry, bounded by ``MAX_BYTES = 2 GiB``. MEASURED,
a partition is a median 208.6 kB, so the 151,082-entry library would need
~31 GB in that format -- the loader would reject it outright. That path is a
PANEL artifact by construction and cannot carry the corpus.

It also does not need to. MEASURED by reading the actual training step: the
optimizer consumes ``forward_teacher_successor_batch(model, batch, fibers)``.
The partition is bound and discarded; only ``_metric_rows`` reads it, under
``model.eval()``. What training needs per entry is the decoded source state and
the teacher successor fiber -- and the frozen chunk library ALREADY stores both
in ``ENTRIES.json``:

    ENTRIES.json across the four roots      354.9 MB
      teacher_successor_fiber                76.9 MB
      exact_state payloads                   88.0 MB
    BATCH.pt (pre-collated, not read here)  ~14.5 GB

So the corpus training inputs are ~355 MB of JSON, not 31 GB.

RAW PAYLOADS RESIDENT, STATES DECODED ON DEMAND
-----------------------------------------------
MEASURED: a decoded state costs 8.30 kB against 0.56 kB for its raw payload --
decoding inflates 15x, so holding decoded states would cost 1.20 GB against
0.09 GB. Decoding is also cheap: 19 us per state, so a 64-example minibatch
decodes in 1.2 ms, which is noise beside a GPU step. This module therefore
keeps raw payloads and decodes per minibatch.

WHAT THIS REFUSES
-----------------
Every check here exists because its failure is SILENT. A missing teacher fiber,
a conflicting duplicate id, a state that does not survive a decode/encode round
trip, or a fiber describing a different source than its entry would all produce
a training run that looks healthy and optimizes the wrong thing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from compose_v4.data.durable_path import require_durable_path
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state

#: Filename holding per-entry metadata inside one compiled slice directory.
ENTRIES_FILENAME = "ENTRIES.json"


class CorpusTrainingLibraryError(RuntimeError):
    """The frozen chunk library cannot be read as training inputs."""


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CorpusTrainingLibraryError(f"{field} is not a lowercase SHA-256")
    return text


def alias_from_library_payload(value: object) -> TeacherSuccessorAlias:
    """Decode one stored alias, refusing anything malformed.

    Kept strict deliberately: a coordinate silently coerced from a float or a
    negative would index a real action table and train a wrong operand.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "family_name",
        "table_name",
        "coordinate",
    }:
        raise CorpusTrainingLibraryError("a stored teacher alias is malformed")
    coordinate = value["coordinate"]
    if (
        not isinstance(value["family_name"], str)
        or not value["family_name"]
        or not isinstance(value["table_name"], str)
        or not value["table_name"]
        or not isinstance(coordinate, list)
        or not coordinate
        or any(type(item) is not int or item < 0 for item in coordinate)
    ):
        raise CorpusTrainingLibraryError("a stored teacher alias coordinate is malformed")
    return TeacherSuccessorAlias(
        family_name=str(value["family_name"]),
        table_name=str(value["table_name"]),
        coordinate=tuple(coordinate),
    )


def teacher_fiber_from_library_payload(value: object) -> TeacherSuccessorFiber:
    """Decode one stored ``teacher_successor_fiber``.

    This mirrors the P50 runtime's private decoder. The duplication is
    deliberate -- importing a private symbol from a frozen module would couple
    this loader to that module's error type -- but two decoders can drift, so
    ``test_corpus_training_library`` asserts the two agree on real payloads.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "source_key",
        "target_key",
        "target_state_sha256",
        "aliases",
        "state_support",
    }:
        raise CorpusTrainingLibraryError("a stored teacher fiber is malformed")
    support = value["state_support"]
    aliases = value["aliases"]
    if (
        not isinstance(support, Mapping)
        or set(support) != {"source_key", "source_state_sha256", "virtual_aliases"}
        or not isinstance(aliases, list)
        or not isinstance(support["virtual_aliases"], list)
    ):
        raise CorpusTrainingLibraryError("a stored state support is malformed")
    try:
        state_support = StateProductiveSupport(
            source_key=str(support["source_key"]),
            source_state_sha256=_require_sha256(
                support["source_state_sha256"], field="stored source state"
            ),
            virtual_aliases=tuple(
                alias_from_library_payload(alias) for alias in support["virtual_aliases"]
            ),
        )
        return TeacherSuccessorFiber(
            source_key=str(value["source_key"]),
            target_key=str(value["target_key"]),
            target_state_sha256=_require_sha256(
                value["target_state_sha256"], field="stored target state"
            ),
            aliases=tuple(alias_from_library_payload(alias) for alias in aliases),
            state_support=state_support,
        )
    except ValueError as error:
        raise CorpusTrainingLibraryError("a stored teacher fiber is invalid") from error


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One training transition, with its source state still encoded."""

    entry_id: str
    state_payload: Mapping[str, Any]
    teacher_fiber: TeacherSuccessorFiber
    support_time_hex: str
    model_family: str
    capability_cell_id: str

    @property
    def source_key(self) -> str:
        return self.teacher_fiber.source_key

    def state(self) -> Any:
        """Decode this entry's source state. MEASURED at 19 us."""

        return decode_state(self.state_payload)


@dataclass(frozen=True, slots=True)
class CorpusTrainingLibrary:
    """Every eligible training entry, ordered deterministically by id."""

    entries: tuple[CorpusEntry, ...]
    index_by_entry_id: Mapping[str, int]
    #: Entries dropped because their source is held out under split precedence.
    excluded_entry_count: int
    #: Slice directories read.
    slice_count: int

    def __len__(self) -> int:
        return len(self.entries)

    def entry(self, entry_id: str) -> CorpusEntry:
        try:
            return self.entries[self.index_by_entry_id[entry_id]]
        except KeyError as error:
            raise CorpusTrainingLibraryError(
                "training stream references an entry outside the library"
            ) from error

    def inputs_for(
        self, entry_ids: Sequence[str]
    ) -> tuple[tuple[Any, ...], tuple[TeacherSuccessorFiber, ...], tuple[CorpusEntry, ...]]:
        """Decode one minibatch, in the ORDER REQUESTED.

        The sampler decides order and repetition; this must not reimpose its
        own, or the optimizer would see a different sequence while every count
        stayed identical.
        """

        selected = tuple(self.entry(entry_id) for entry_id in entry_ids)
        return (
            tuple(entry.state() for entry in selected),
            tuple(entry.teacher_fiber for entry in selected),
            selected,
        )


def _slice_entry_files(root: Path) -> list[Path]:
    return sorted(root.rglob(ENTRIES_FILENAME))


def load_corpus_training_library(
    roots: Iterable[Path | str],
    *,
    excluded_sources: Iterable[str] = (),
    verify_state_roundtrip: bool = True,
    allow_reapable_roots: bool = False,
) -> CorpusTrainingLibrary:
    """Read the frozen chunk roots into training inputs, refusing silent damage.

    ``verify_state_roundtrip`` decodes and re-encodes every state, which costs
    ~19 us each (~3 s corpus-wide) and holds no extra memory because each
    decode is discarded. It defaults on because a state that does not round
    trip trains normally and wrongly.
    """

    resolved: list[Path] = []
    for root in roots:
        path = Path(root)
        resolved.append(
            path if allow_reapable_roots else require_durable_path(path, role="corpus chunk root")
        )

    excluded = frozenset(str(source) for source in excluded_sources)
    by_id: dict[str, CorpusEntry] = {}
    payload_by_id: dict[str, str] = {}
    slice_count = 0
    excluded_count = 0

    for root in resolved:
        if not root.is_dir():
            raise CorpusTrainingLibraryError(f"corpus chunk root is not a directory: {root}")
        for entries_file in _slice_entry_files(root):
            slice_count += 1
            try:
                document = json.loads(entries_file.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise CorpusTrainingLibraryError(
                    f"compiled slice is unreadable: {entries_file}"
                ) from error
            raw_entries = document.get("entries")
            if not isinstance(raw_entries, list):
                raise CorpusTrainingLibraryError(
                    f"compiled slice has no entry list: {entries_file}"
                )
            for raw in raw_entries:
                entry_id = str(raw["p50_compiled_entry_sha256"])
                fiber_payload = raw.get("teacher_successor_fiber")
                if fiber_payload is None:
                    # A terminal row carries no molecular jump to supervise.
                    raise CorpusTrainingLibraryError(
                        f"compiled entry {entry_id} has no teacher successor fiber"
                    )

                # A duplicate id is only safe when the two records agree. The
                # library HAS 4 of these across two roots; a conflicting pair
                # would otherwise resolve by read order.
                canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
                if entry_id in payload_by_id:
                    if payload_by_id[entry_id] != canonical:
                        raise CorpusTrainingLibraryError(
                            f"compiled entry {entry_id} appears twice with different content"
                        )
                    continue
                payload_by_id[entry_id] = canonical

                fiber = teacher_fiber_from_library_payload(fiber_payload)
                if fiber.state_support.source_state_sha256 != _require_sha256(
                    raw["source_state_sha256"], field="entry source state"
                ):
                    raise CorpusTrainingLibraryError(
                        f"compiled entry {entry_id} disagrees with its fiber on the source state"
                    )
                if fiber.target_state_sha256 != _require_sha256(
                    raw["target_state_sha256"], field="entry target state"
                ):
                    raise CorpusTrainingLibraryError(
                        f"compiled entry {entry_id} disagrees with its fiber on the target state"
                    )
                multiplicity = int(raw["production_successor_alias_multiplicity"])
                if len(fiber.aliases) != multiplicity:
                    raise CorpusTrainingLibraryError(
                        f"compiled entry {entry_id} records {multiplicity} teacher aliases "
                        f"but its fiber holds {len(fiber.aliases)}"
                    )

                if fiber.source_key in excluded:
                    excluded_count += 1
                    continue

                state_payload = raw["exact_state"]
                if verify_state_roundtrip and encode_state(decode_state(state_payload)) != state_payload:
                    raise CorpusTrainingLibraryError(
                        f"compiled entry {entry_id} state does not survive a decode round trip"
                    )

                by_id[entry_id] = CorpusEntry(
                    entry_id=entry_id,
                    state_payload=state_payload,
                    teacher_fiber=fiber,
                    support_time_hex=str(raw["support_time_hex"]),
                    model_family=str(raw["model_family"]),
                    capability_cell_id=str(raw["capability_cell_id"]),
                )

    if not slice_count:
        raise CorpusTrainingLibraryError(
            "no compiled slice was found under any corpus root; refusing to "
            "report an empty library as a successful load"
        )
    if not by_id:
        raise CorpusTrainingLibraryError(
            f"read {slice_count} compiled slices but kept no entry; refusing to "
            "report an empty library as a successful load"
        )

    ordered = tuple(by_id[entry_id] for entry_id in sorted(by_id))
    return CorpusTrainingLibrary(
        entries=ordered,
        index_by_entry_id=MappingProxyType(
            {entry.entry_id: index for index, entry in enumerate(ordered)}
        ),
        excluded_entry_count=excluded_count,
        slice_count=slice_count,
    )


__all__ = [
    "ENTRIES_FILENAME",
    "CorpusEntry",
    "CorpusTrainingLibrary",
    "CorpusTrainingLibraryError",
    "alias_from_library_payload",
    "load_corpus_training_library",
    "teacher_fiber_from_library_payload",
]
