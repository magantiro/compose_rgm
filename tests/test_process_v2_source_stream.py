"""The cache-fed Process-V2 source stream, driven end to end.

Everything here runs against real published artifacts: a real twenty-cell V1
migration payload, a real committed chunk-cache generation over it, and a real
cache-fed rebind run whose decisions were proved by the production executor.
Nothing is substituted, because the properties under test are exactly the ones a
substituted resolver cannot make false:

* the stream joins the committed cache rows to the committed rebind decisions by
  the frozen ``(v1_task_identity_sha256, entry_index)`` key, and accounts for
  every entry;
* deleting the V1 payload after the cache is published changes nothing, because
  no path here opens a packed shard;
* an upstream-rejected trace is present in the census and carries no chemistry a
  candidate evaluator could consume;
* the stream and the published identity reopen the same committed generation,
  and both are deterministic.

The negative controls matter as much as the assertions.  The "no packed shard is
opened" test carries a positive control that opens one, so the instrument is
demonstrably able to see the read it asserts the absence of; and the
never-evaluated invariant is attacked by construction, so it fails if the guard
is removed rather than merely being observed to hold.
"""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    JOIN_KEY_FIELDS,
    UPSTREAM_REJECTED,
)
from compose_v4.data.editing_v2_process_v2_active8_source import (
    ProcessV2Active8SourceError,
    ProcessV2SourceEntry,
    resolve_process_v2_active8_source_inventory,
    validate_process_v2_active8_source_identity,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    COMPLETION_FILENAME as CACHE_COMPLETION_FILENAME,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.data.semantic_packed_trace_store import (
    SHARD_FILENAME as SEMANTIC_SHARD_FILENAME,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_process_v2_chunk_fed_rebind as chunk_fed  # noqa: E402

ROOT = _REPO_ROOT

# Both are autouse fixtures of the modules above; rebinding them here is how a
# pytest module inherits another module's fixtures without a conftest.
_effective_mask_authority = chunk_fed._effective_mask_authority
_register_extra_fixture_traces = chunk_fed._register_extra_fixture_traces


# ---- One real committed chain: payload, cache, decisions ----------------------


class _Chain:
    """The three real artifacts the source joins, and where they are mounted."""

    def __init__(self, payload, cache_plan, rebind_plan) -> None:
        self.payload = payload
        self.cache_plan = cache_plan
        self.rebind_plan = rebind_plan

    @property
    def artifact_root(self) -> Path:
        return self.payload.artifact_root

    def resolve(self, **kwargs: Any):
        return resolve_process_v2_active8_source_inventory(
            self.rebind_plan,
            artifact_root=self.artifact_root,
            repo_root=ROOT,
            **kwargs,
        )


@pytest.fixture
def chain(tmp_path: Path) -> _Chain:
    payload, _binding, cache_plan, _cache_completion = chunk_fed._cached_payload(tmp_path)
    rebind_plan = chunk_fed._cache_fed_plan(payload, cache_plan)
    chunk_fed._execute_and_reduce(payload, rebind_plan)
    return _Chain(payload, cache_plan, rebind_plan)


def _stream_digest(inventory) -> str:
    """An address-free digest of exactly what the stream decided to hand out.

    Built from the join key, the record identity and the category rather than
    from any run-level hash, so it is comparable across two resolutions of the
    same committed artifacts.
    """

    return canonical_sha256(
        [
            [
                list(entry.join_key),
                entry.v1_identity.record_sha256,
                entry.rejection_category,
                entry.v1_record is not None,
            ]
            for entry in inventory.iter_source_entries()
        ]
    )


# ---- The join itself ----------------------------------------------------------


def test_the_stream_joins_every_cached_row_to_its_recorded_decision(chain: _Chain) -> None:
    inventory = chain.resolve()
    entries = list(inventory.iter_source_entries())

    # Every entry of the corpus, exactly once, under the frozen join key.
    assert len(entries) == inventory.counts["source_entries"] == 60
    keys = [entry.join_key for entry in entries]
    assert len(set(keys)) == len(keys)
    expected = [
        (source.v1_task_identity_sha256, index)
        for source in inventory.sources
        for index in range(source.v1_entry_count)
    ]
    assert sorted(keys) == sorted(expected)

    admitted = [entry for entry in entries if entry.admitted]
    rejected = [entry for entry in entries if not entry.admitted]
    assert len(admitted) == inventory.counts["admitted_entries"]
    assert len(rejected) == inventory.counts["rejected_entries"]
    assert rejected, "the fixture must carry a real upstream rejection"


def test_each_entry_carries_the_payload_and_decision_provenance_under_distinct_names(
    chain: _Chain,
) -> None:
    """The two identities are what a consumer must not be able to confuse.

    The V1 identity is the superseded one the chemistry was built under; the V2
    identity is the live one the decision was made under. They are different
    values on different objects, and every admitted entry carries both.
    """

    inventory = chain.resolve()
    assert (
        inventory.v1_payload_process_identity_sha256 != inventory.process_v2_identity_sha256
    )
    seen_admitted = 0
    for entry in inventory.iter_source_entries():
        assert entry.v1_identity.process_identity_sha256 == (
            inventory.v1_payload_process_identity_sha256
        )
        if entry.admitted:
            seen_admitted += 1
            assert entry.v2_admission.process_v2_identity_sha256 == (
                inventory.process_v2_identity_sha256
            )
            assert entry.v2_admission.admitted is True
    assert seen_admitted == inventory.counts["admitted_entries"]


def test_the_stream_is_deterministic_and_ordered_by_the_frozen_join_key(
    chain: _Chain,
) -> None:
    inventory = chain.resolve()
    first = [entry.join_key for entry in inventory.iter_source_entries()]
    second = [entry.join_key for entry in chain.resolve().iter_source_entries()]

    assert JOIN_KEY_FIELDS == ("v1_task_identity_sha256", "entry_index")
    assert first == sorted(first)
    assert first == second
    assert _stream_digest(inventory) == _stream_digest(chain.resolve())


def test_a_stream_that_does_not_yield_its_own_census_is_refused(chain: _Chain) -> None:
    """The stream accounts for every entry or refuses; there is no partial mode.

    A source that yielded what it could match would silently redefine the
    corpus, and every downstream census would reconcile against the smaller
    number. The published census is the thing the stream is checked against,
    so it is the census that is moved here.
    """

    inventory = chain.resolve()
    object.__setattr__(
        inventory, "counts", {**inventory.counts, "source_entries": 61}
    )
    with pytest.raises(
        ProcessV2Active8SourceError, match="yielded 60 source_entries"
    ):
        list(inventory.iter_source_entries())


# ---- Acceptance 6: an upstream-rejected trace is never candidate-evaluated ----


def test_an_upstream_rejected_trace_is_in_the_census_and_carries_no_chemistry(
    chain: _Chain,
) -> None:
    inventory = chain.resolve()
    rejected = [entry for entry in inventory.iter_source_entries() if not entry.admitted]
    assert rejected

    for entry in rejected:
        # Present, categorised, and reason-coded: the census can account for it.
        assert entry.rejection_category == UPSTREAM_REJECTED
        assert entry.rejection["exclusion_code"] in inventory.rejected_traces_by_code
        assert entry.v1_identity.trace_id
        assert entry.join_key[1] == entry.v1_identity.entry_index
        # And nothing a candidate evaluator could consume.
        assert entry.v1_record is None
        assert entry.addressed is None
        assert entry.v2_admission is None

    census = inventory.rejected_traces_by_code
    assert sum(census.values()) == len(rejected) == inventory.counts["rejected_entries"]


def test_an_upstream_rejected_entry_cannot_be_constructed_with_evaluable_chemistry(
    chain: _Chain,
) -> None:
    """The invariant is structural, so removing it is a failure, not a style change.

    Observing that the stream happens to withhold the chemistry would still pass
    if the withholding were deleted from one of two branches. Attacking the
    constructor proves the rule is enforced where the object is built.
    """

    inventory = chain.resolve()
    entries = list(inventory.iter_source_entries())
    admitted = next(entry for entry in entries if entry.admitted)
    rejected = next(entry for entry in entries if not entry.admitted)

    with pytest.raises(ProcessV2Active8SourceError, match="candidate evaluator"):
        ProcessV2SourceEntry(
            v1_identity=rejected.v1_identity,
            rejection_category=UPSTREAM_REJECTED,
            v2_admission=None,
            rejection=rejected.rejection,
            v1_record=admitted.v1_record,
            addressed=admitted.addressed,
        )
    with pytest.raises(ProcessV2Active8SourceError, match="must carry its decoded record"):
        ProcessV2SourceEntry(
            v1_identity=admitted.v1_identity,
            rejection_category=None,
            v2_admission=admitted.v2_admission,
            rejection=None,
            v1_record=None,
            addressed=None,
        )


# ---- Acceptance 2: the V1 payload is not on the production path ---------------


def test_the_source_and_its_stream_survive_deleting_the_v1_payload(chain: _Chain) -> None:
    """The strongest statement of "no raw fallback": the shards need not exist.

    The V1 payload is bound at plan time, by hash. After that the corpus is the
    committed cache, so the resolution, the published identity and every
    streamed entry are reproduced here with the entire payload removed from the
    artifact root. A surviving raw read would not merely be slow; it would be
    impossible.
    """

    before = chain.resolve()
    identity_before = before.identity()
    digest_before = _stream_digest(before)

    shutil.rmtree(chain.payload.payload_root)
    assert not chain.payload.payload_root.exists()

    after = chain.resolve()
    assert canonical_bytes(after.identity()) == canonical_bytes(identity_before)
    assert _stream_digest(after) == digest_before
    assert validate_process_v2_active8_source_identity(
        after.identity(), repo_root=ROOT
    ) == after.identity()


def test_no_packed_shard_is_opened_by_the_resolution_the_identity_or_the_stream(
    chain: _Chain, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    real_path_open = Path.open
    real_gzip_open = gzip.open

    def path_open(path_self, *args: Any, **kwargs: Any):
        opened.append(Path(path_self).name)
        return real_path_open(path_self, *args, **kwargs)

    def gzip_open(filename: Any, *args: Any, **kwargs: Any):
        opened.append(Path(filename).name)
        return real_gzip_open(filename, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(gzip, "open", gzip_open)

    inventory = chain.resolve()
    inventory.identity()
    streamed = sum(1 for _entry in inventory.iter_source_entries())

    assert streamed == 60
    assert opened, "the instrument recorded no reads at all"
    assert SEMANTIC_SHARD_FILENAME not in opened

    # The positive control: the admitted source's own record iterator reads the
    # raw shard, so the instrument is demonstrably able to see the read whose
    # absence is asserted above. It is also why the stream does not use it.
    opened.clear()
    next(iter(inventory.admitted_source.iter_records()))
    assert SEMANTIC_SHARD_FILENAME in opened


# ---- The stream and the identity reopen the same committed generation --------


def test_the_identity_names_the_committed_generation_the_stream_reads(
    chain: _Chain,
) -> None:
    inventory = chain.resolve()
    identity = inventory.identity()
    completion = inventory.cache_generation.completion
    binding = chain.rebind_plan["cache_binding"]

    assert identity["cache_completion_sha256"] == completion["completion_sha256"]
    assert identity["cache_completion_sha256"] == binding["cache_completion_sha256"]
    assert identity["cache_physical_identity_sha256"] == (
        binding["cache_physical_identity_sha256"]
    )
    assert identity["cache_semantic_identity_sha256"] == (
        binding["cache_semantic_identity_sha256"]
    )
    # The historical payload provenance survives the payload: it is read off the
    # generation, which recorded it when the rows were cached.
    assert identity["v1_migration_completion_sha256"] == (
        completion["cache_semantic_identity"]["migration_completion_sha256"]
    )


def test_an_uncommitted_cache_generation_cannot_be_resolved(chain: _Chain) -> None:
    run_root = chain.artifact_root / str(
        chain.cache_plan["run_artifact_root"]
    ).removeprefix("/artifacts/")
    (run_root / CACHE_COMPLETION_FILENAME).unlink()
    with pytest.raises(ProcessV2Active8SourceError, match="cannot be opened"):
        chain.resolve()


def test_an_oracle_geometry_run_is_refused_as_a_production_source(
    tmp_path: Path,
) -> None:
    """There is no raw-shard path INTO the source, not merely none taken.

    An oracle-geometry rebind is a correctness comparison rather than evidence,
    and it carries no cache binding at all, so the refusal is structural.
    """

    payload, _binding, cache_plan, _completion = chunk_fed._cached_payload(tmp_path)
    oracle_plan = chunk_fed._oracle_plan(payload)
    chunk_fed._execute_and_reduce(payload, oracle_plan)
    with pytest.raises(
        ProcessV2Active8SourceError, match="does not bind a committed chunk cache"
    ):
        resolve_process_v2_active8_source_inventory(
            oracle_plan, artifact_root=payload.artifact_root, repo_root=ROOT
        )
    # And the cache-fed run over the same payload resolves, so the refusal is
    # about the geometry rather than about this fixture.
    cache_fed_plan = chunk_fed._cache_fed_plan(payload, cache_plan)
    chunk_fed._execute_and_reduce(payload, cache_fed_plan)
    assert resolve_process_v2_active8_source_inventory(
        cache_fed_plan, artifact_root=payload.artifact_root, repo_root=ROOT
    ).counts["source_entries"] == 60


def test_a_cached_chunk_whose_bytes_moved_is_refused_by_the_stream(
    chain: _Chain,
) -> None:
    """The stream verifies the rows it joins rather than trusting the manifest."""

    inventory = chain.resolve()
    assert sum(1 for _entry in inventory.iter_source_entries()) == 60

    source_root = chain.artifact_root / str(
        chain.cache_plan["run_artifact_root"]
    ).removeprefix("/artifacts/") / "sources"
    victim = next(iter(sorted(source_root.rglob("chunk-*.jsonl.gz"))))
    original = victim.read_bytes()
    victim.write_bytes(original + b"\x00")
    with pytest.raises(ProcessV2Active8SourceError, match="unreadable"):
        list(chain.resolve(verify_chunk_bytes=False).iter_source_entries())
    victim.write_bytes(original)
    assert sum(1 for _entry in chain.resolve().iter_source_entries()) == 60
