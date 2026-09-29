"""Protocol-only downstream fixtures for the frozen Process-V2 interfaces.

This module is deliberately *upstream of every later scientific decision*.  It
proves that the interfaces frozen in
``compose_v4.data.editing_v2_process_v2_schema`` are usable by a layer that does
not exist yet, without encoding any Active8 candidate semantics, Gate 0 policy,
T1 panel rule, or P50 threshold.  Nothing here decides anything scientific; if a
fixture below could only be written by inventing a scientific value, it was not
written.

Two things are proven.

**A fake decision index satisfies the structural protocol.**
:class:`~compose_v4.data.editing_v2_process_v2_schema.StructuralDecisionIndex`
exists so a Process-V2 decision index never subclasses a V1 loader that would
revalidate V1 schemas and live identities, which is what makes a V1 loader
reject the historical payload by construction.  A protocol only delivers that if
it is genuinely satisfiable *structurally*, so the fake below inherits from
nothing at all and its MRO is asserted to prove it.  The census helper
``require_census_reconciles`` is exercised in both directions, because a protocol
member named ``counts`` is a name, not a proof: the runtime check is by name
only, and a decision index whose census does not reconcile would pass
``isinstance`` unchanged.

**The real Process-V2 chain projects into that protocol.**  A monkeypatched
resolver would prove nothing about production behaviour, so the second half
builds a tiny synthetic V1 payload with the production writers, runs the real
rebind through the real
``compose_v4.rewrite.process_v2_atom_delete`` admission authority, reduces it,
resolves the real admitted source, and projects the published result into the
frozen protocol.  Its run, completion and task identities are fixture-derived,
and its artifact namespace, source revision and trace metadata each carry an
explicit fixture marker, so no artifact this module produces can be mistaken for
the real migration; the production completion SHAs and counts quoted in the
runnable-chain handoff are asserted absent from every published byte.

One identity is deliberately *not* synthetic.  The pinned V1 process identity is
the real one the payload was written under, because the mechanism under test is
the pinned-identity acceptance path: the payload is read through a pinned
identity rather than through whatever identity happens to be live.  A fixture
that pinned a made-up identity would exercise nothing.

The V1 payload builders are imported from ``tests/test_editing_process_v2_rebind``
rather than rebuilt.  Rebuilding them would create a second payload writer whose
drift from the first is invisible, which is the duplication the repository
registry rule forbids.

Scope, stated rather than implied
---------------------------------

The projection ``_PublishedRebindIndex`` below is a *fixture*.  It is not a
proposal for, nor a preview of, the Process-V2 decision-index loader that
Section 5 of the runnable-chain specification requires; that loader is a later
workstream's deliverable and will carry candidate decisions this projection does
not have.  The projection reads only fields the production rebind writer already
publishes, and its one non-mechanical choice is which of the two published
process identities becomes ``process_identity_sha256``.  That choice is fixed by
the frozen schema, not by this module: the V2 identity is the identity the
decisions were computed under, and binding the superseded V1 one instead is
exactly the conflation the schema docstring names.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from compose_v4.data.editing_process_v2_admitted_source import (
    resolve_process_v2_admitted_source,
)
from compose_v4.data.editing_process_v2_rebind import reduce_process_v2_rebind
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    StructuralDecisionIndex,
    require_census_reconciles,
)
from compose_v4.data.editing_v2_semantic_active8_decision_source import (
    EditingV2SemanticActive8DecisionIndex,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    EditingV2SemanticActive8SourceInventory,
)

# The V1 payload fixture builders are not importable as a package.  pytest
# already puts ``tests`` on ``sys.path`` under the default prepend import mode;
# the entry is added explicitly so this module also imports under
# ``importmode=importlib`` and outside pytest.  This mirrors
# ``tests/test_process_v2_rebind_end_to_end.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_editing_process_v2_rebind as v1_fixture

# ---- Fixture provenance and the values a fixture may never carry ----

# Every identity this module produces is derived from the fixture builders, and
# the artifact namespace they publish under is fixture-only.
FIXTURE_ARTIFACT_PREFIX = "/artifacts/rebind_fixture"
FIXTURE_SOURCE_COMMIT = "a" * 40
FIXTURE_SOURCE_TREE = "b" * 40
FIXTURE_TRACE_METADATA_MARKER = "fixture"

# The exact current semantic migration completion, quoted from Section 2 of the
# runnable-chain handoff.  A fixture that reproduced any of these would be
# indistinguishable from the real artifact in a grep, a log, or a manifest, so
# they are asserted absent rather than merely avoided by convention.  They are
# recorded here as forbidden literals only; this module makes no claim about
# them and never reads the real completion.
FORBIDDEN_PRODUCTION_SHA256 = (
    "2c4ebb5ea8dc55c942978c1548d99f6a7b34bc8a3b7d6a425c814cc0de27dcc7",
    "a7698306bf1977a6a4cf3d7003948084729a428dd609f5b44058b9ace160ce24",
)
FORBIDDEN_PRODUCTION_COUNTS = (695638, 646779, 48859)

# The frozen public interface, independent of typing.Protocol's private
# implementation attributes (which differ between Python 3.11 and 3.12).
REQUIRED_INDEX_MEMBERS = frozenset(
    {
        "process_identity_sha256",
        "completion_sha256",
        "counts",
        "rejected_traces_by_code",
        "identity",
    }
)


# ---- A fake V2 decision index, satisfying the protocol structurally ----


class _FakeDecisionIndex:
    """A decision index that inherits from nothing.

    Not a dataclass and not a subclass: the point of the protocol is that a V2
    index can satisfy it without acquiring a base class, so the fake has to be
    able to demonstrate that.  Its numbers are arbitrary fixture values chosen
    only to reconcile; they encode no candidate semantics and no policy.
    """

    def __init__(
        self,
        *,
        source: int = 7,
        admitted: int = 5,
        rejected: int = 2,
        process_identity_sha256: str = "1" * 64,
        completion_sha256: str = "2" * 64,
    ) -> None:
        self._source = source
        self._admitted = admitted
        self._rejected = rejected
        self._process_identity_sha256 = process_identity_sha256
        self._completion_sha256 = completion_sha256

    @property
    def process_identity_sha256(self) -> str:
        return self._process_identity_sha256

    @property
    def completion_sha256(self) -> str:
        return self._completion_sha256

    def counts(self) -> Mapping[str, int]:
        return {
            "source_entries": self._source,
            "admitted_entries": self._admitted,
            "rejected_entries": self._rejected,
        }

    def rejected_traces_by_code(self) -> Mapping[str, int]:
        return {"fixture_reason_code": self._rejected} if self._rejected else {}

    def identity(self) -> Mapping[str, Any]:
        return {
            "fixture": True,
            "process_identity_sha256": self._process_identity_sha256,
            "completion_sha256": self._completion_sha256,
            "counts": dict(self.counts()),
        }


def test_a_fake_index_satisfies_the_protocol_without_inheriting_a_v1_loader() -> None:
    """Without this, the protocol could be satisfiable only by subclassing.

    ``StructuralDecisionIndex`` exists so a Process-V2 index never inherits a V1
    loader that revalidates V1 schemas and live identities.  If the only way to
    become an instance were to inherit something, the protocol would have
    delivered the opposite of its purpose and nothing would notice.
    """

    fake = _FakeDecisionIndex()
    assert isinstance(fake, StructuralDecisionIndex)

    # Structural, absolutely: the fake's whole ancestry is itself and ``object``.
    assert type(fake).__mro__ == (_FakeDecisionIndex, object)
    inherited = [base for base in type(fake).__mro__ if base is not object]
    assert not [base for base in inherited if base.__module__.startswith("compose_v4")]
    for v1_class in (
        EditingV2SemanticActive8SourceInventory,
        EditingV2SemanticActive8DecisionIndex,
    ):
        assert v1_class not in type(fake).__mro__
    assert StructuralDecisionIndex not in type(fake).__mro__


def test_require_census_reconciles_accepts_the_fixture_and_refuses_an_unbalanced_census() -> None:
    """Without this the census helper could be vacuous in either direction.

    A helper that accepted everything would let ``admitted + rejected != source``
    reach a downstream stage; a helper that accepted nothing would make every
    caller work around it.  Both directions are measured on the same object.
    """

    fake = _FakeDecisionIndex()
    require_census_reconciles(fake.counts(), label="fake index")

    unbalanced = _FakeDecisionIndex(source=7, admitted=5, rejected=1)
    with pytest.raises(ProcessV2SchemaError) as refusal:
        require_census_reconciles(unbalanced.counts(), label="unbalanced fixture")
    # The refusal names the measured numbers, so an operator can act on it.
    assert "does not reconcile" in str(refusal.value)
    assert "5 admitted" in str(refusal.value) and "1 rejected" in str(refusal.value)
    assert "7 source" in str(refusal.value)

    # An omitted field is a distinct refusal, not a silent zero.
    with pytest.raises(ProcessV2SchemaError, match="omits"):
        require_census_reconciles(
            {"source_entries": 7, "admitted_entries": 5}, label="partial fixture"
        )


def test_a_v1_shaped_object_is_not_a_structural_decision_index() -> None:
    """Without this the protocol could be satisfied by any V1 loader by accident.

    The shape is taken from the real V1 classes rather than invented, so a V1
    loader that later grows the missing members fails here instead of quietly
    becoming an acceptable V2 decision index.
    """

    required = REQUIRED_INDEX_MEMBERS
    public_members = {name for name in vars(StructuralDecisionIndex) if not name.startswith("_")}
    assert public_members == required

    # Measured: both real V1 loaders lack exactly the same three members.
    for v1_class in (
        EditingV2SemanticActive8SourceInventory,
        EditingV2SemanticActive8DecisionIndex,
    ):
        assert required - set(dir(v1_class)) == {
            "completion_sha256",
            "rejected_traces_by_code",
            "identity",
        }

    # An object carrying exactly the V1 inventory's own field names is not an
    # instance.  The field names come from the real dataclass, so this cannot
    # drift into testing a straw man.
    v1_shaped = SimpleNamespace(
        **{field.name: None for field in dataclass_fields(EditingV2SemanticActive8SourceInventory)}
    )
    assert not isinstance(v1_shaped, StructuralDecisionIndex)


@pytest.mark.parametrize(
    "dropped",
    sorted(REQUIRED_INDEX_MEMBERS),
)
def test_every_protocol_member_is_load_bearing_for_the_instance_check(dropped: str) -> None:
    """Without this the protocol could be satisfiable while missing a member.

    A protocol whose members are not each individually required would let a
    downstream stage bind an index that cannot answer one of the five questions
    the stage is about to ask, and discover it at the call site rather than at
    the boundary.  Dropping each member in turn is the only way to show that all
    five are enforced rather than only the first one checked.
    """

    members = sorted(REQUIRED_INDEX_MEMBERS)
    partial = type(
        "PartialIndex",
        (),
        {member: (lambda self: {}) for member in members if member != dropped},
    )()
    assert not isinstance(partial, StructuralDecisionIndex)

    complete = type("CompleteIndex", (), {member: (lambda self: {}) for member in members})()
    assert isinstance(complete, StructuralDecisionIndex)


def test_the_runtime_protocol_check_is_by_name_so_the_census_check_stays_mandatory() -> None:
    """Without this, ``isinstance`` would be mistaken for a census proof.

    A runtime-checkable protocol checks member *presence*, not callability and
    not any invariant.  An object whose ``counts`` is a plain attribute, and
    whose census does not reconcile, is still an instance.  That is why
    ``require_census_reconciles`` is a separate obligation at every boundary that
    carries a census, and why a later stage must call it rather than infer it.
    """

    by_name_only = SimpleNamespace(
        process_identity_sha256="1" * 64,
        completion_sha256="2" * 64,
        counts={"source_entries": 7, "admitted_entries": 5, "rejected_entries": 1},
        rejected_traces_by_code={},
        identity={},
    )
    assert isinstance(by_name_only, StructuralDecisionIndex)
    assert not callable(by_name_only.counts)
    with pytest.raises(ProcessV2SchemaError, match="does not reconcile"):
        require_census_reconciles(by_name_only.counts, label="by-name-only index")


# ---- The real Process-V2 chain, driven end to end ----


class _PublishedRebindIndex:
    """A fixture projection of a published rebind completion into the protocol.

    Fixture only.  See this module's docstring: this is not the Process-V2
    decision-index loader, and it computes nothing.  It reads published fields
    and binds the V2 process identity, never the superseded V1 one.
    """

    def __init__(self, completion: Mapping[str, Any]) -> None:
        self._completion = completion

    @property
    def process_identity_sha256(self) -> str:
        return str(self._completion["process_v2_identity_sha256"])

    @property
    def completion_sha256(self) -> str:
        return str(self._completion["completion_sha256"])

    def counts(self) -> Mapping[str, int]:
        return {key: int(value) for key, value in self._completion["counts"].items()}

    def rejected_traces_by_code(self) -> Mapping[str, int]:
        return dict(self._completion["rejected_traces_by_code"])

    def identity(self) -> Mapping[str, Any]:
        return dict(self._completion)


def _completed_fixture_chain(tmp_path: Path, *, tasks=v1_fixture._V1_TASKS):
    """Build a tiny V1 payload, run the real rebind, reduce, and resolve."""

    payload = v1_fixture._build_v1_payload(tmp_path / "artifacts", tasks=tasks)
    plan = v1_fixture._plan_for(payload)
    v1_fixture._execute_all(payload, plan)
    completion = reduce_process_v2_rebind(
        plan, artifact_root=payload.artifact_root, repo_root=v1_fixture.ROOT
    )
    source = resolve_process_v2_admitted_source(
        plan, artifact_root=payload.artifact_root, repo_root=v1_fixture.ROOT
    )
    return payload, plan, completion, source


def _integers(value: object) -> Iterator[int]:
    """Every integer reachable in a decoded JSON document."""

    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _integers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _integers(item)


def test_the_real_process_v2_chain_projects_into_the_frozen_protocol(
    tmp_path: Path,
) -> None:
    """Without this the protocol is proven only against a fake.

    A protocol that fits a hand-written stand-in but not the shape the
    production writers actually publish would be discovered by the first real
    consumer, after the layers above it were already built against it.  This
    drives the real rebind and the real admitted-source resolver and projects
    what they published, so the fit is measured rather than assumed.
    """

    # The real production admission authority, not a substitute.  If
    # ``process_v2_atom_delete_mask`` were absent the shared fixture module would
    # install its own oracle and this test would silently stop driving
    # production code.
    assert v1_fixture.AUTHORITY_SUBSTITUTED is False

    _payload, plan, completion, source = _completed_fixture_chain(tmp_path)

    # Present is not the same as exercised.  The published census proves the
    # admission authority was evaluated at every progress state and returned a
    # non-empty fiber, so this fixture drove production code rather than merely
    # importing it.
    census = completion["process_v2_atom_delete_census"]
    assert census["states_evaluated"] == 10
    assert census["candidate_slots"] > 0

    index = _PublishedRebindIndex(completion)
    assert isinstance(index, StructuralDecisionIndex)
    require_census_reconciles(index.counts(), label="published rebind completion")

    # Measured on this payload: four traces in, four admitted, none rejected.
    assert index.counts()["source_entries"] == 4
    assert index.counts()["admitted_entries"] == 4
    assert index.counts()["rejected_entries"] == 0
    assert index.rejected_traces_by_code() == {}

    # The identity bound is the V2 one, and the two identities stay distinct.
    live_v2 = str(plan["process_v2_identity"]["process_identity_sha256"])
    pinned_v1 = str(plan["pinned_process_identity"]["process_identity_sha256"])
    assert index.process_identity_sha256 == live_v2
    assert index.process_identity_sha256 != pinned_v1
    assert completion["pinned_process_identity_sha256"] == pinned_v1

    # The resolver ran on the same published completion the projection reads.
    assert source.identity()["completion_sha256"] == index.completion_sha256
    assert dict(source.counts) == dict(index.counts())


def test_a_rejecting_payload_still_reconciles_through_the_protocol(
    tmp_path: Path,
) -> None:
    """Without this the census check is only ever exercised on ``rejected == 0``.

    A reconciliation that holds when one side of the sum is zero is nearly
    unfalsifiable.  This payload contains one trace whose teacher is outside the
    Process-V2 fiber, so the published census carries a nonzero rejection and a
    reason code, and the protocol projection must still reconcile.
    """

    _payload, _plan, completion, _source = _completed_fixture_chain(
        tmp_path, tasks=v1_fixture._V1_TASKS_WITH_EXCLUSION
    )
    index = _PublishedRebindIndex(completion)
    require_census_reconciles(index.counts(), label="rejecting rebind completion")
    assert index.counts()["rejected_entries"] == 1
    assert index.counts()["source_entries"] == 5
    assert sum(index.rejected_traces_by_code().values()) == 1


def test_the_fixture_chain_carries_explicit_fixture_provenance(tmp_path: Path) -> None:
    """Without this a fixture artifact is indistinguishable from a real one.

    Section 5 requires fixture-only run, completion and task identities with
    explicit fixture provenance.  A synthetic artifact that reached a volume, a
    log or a manifest without a marker could be read as evidence, so the markers
    are asserted rather than assumed to be present.
    """

    payload, plan, completion, _source = _completed_fixture_chain(tmp_path)

    # The artifact namespace is fixture-only.
    assert str(plan["run_artifact_root"]).startswith(f"{FIXTURE_ARTIFACT_PREFIX}/")

    # The source revision is synthetic and can never name a real git object.
    assert plan["source_revision"]["commit"] == FIXTURE_SOURCE_COMMIT
    assert plan["source_revision"]["tree"] == FIXTURE_SOURCE_TREE

    # Every V1 task identity is a fixture identity, derived by the fixture
    # builder from an explicit fixture string rather than measured from a run.
    fixture_task_ids = {
        v1_fixture._v1_task_identity(index) for index in range(len(v1_fixture._V1_TASKS))
    }
    assert {
        str(task["v1_task_identity_sha256"]) for task in plan["v1_payload_binding"]["v1_tasks"]
    } == fixture_task_ids

    # The marker survives the migration into the V1 records the rebind reads.
    for index in range(len(payload.v1_task_identities)):
        for record in v1_fixture._read_v1_records(v1_fixture._semantic_dir(payload, index)):
            assert FIXTURE_TRACE_METADATA_MARKER in record["metadata"]

    # Nothing published grants anything.
    assert completion["training_authorized"] is False
    assert completion["gate_zero_run"] is False


def test_the_fixture_chain_never_reuses_a_production_identity_or_count(
    tmp_path: Path,
) -> None:
    """Without this a fixture could collide with the real migration's identity.

    Section 5 forbids reusing the production completion SHAs or counts.  A
    collision would make a synthetic run indistinguishable from the exact
    current semantic migration in any downstream comparison, which is the
    failure mode content addressing exists to prevent.  Every published byte is
    scanned for the SHAs, and every published integer for the counts, because a
    count hidden inside a nested census would not appear in a naive text scan
    and a text scan for a decimal count would false-positive inside a hash.
    """

    payload, plan, completion, source = _completed_fixture_chain(tmp_path)
    run_root = v1_fixture._run_root(payload, plan)

    published = [path for path in run_root.rglob("*") if path.is_file()]
    assert published, "the fixture published nothing to scan"
    compressed = [path for path in published if path.suffix == ".gz"]
    assert compressed, "the proof shards are gzip; a scan that never decompresses is vacuous"
    for path in published:
        # Decompressed, because the proof shards are gzip and a raw-byte scan of
        # a compressed shard would find nothing no matter what it contained.
        blob = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        for forbidden in FORBIDDEN_PRODUCTION_SHA256:
            assert forbidden.encode() not in blob, path

    documents: list[Mapping[str, Any]] = [
        dict(completion),
        dict(plan),
        dict(source.identity()),
    ]
    for document in documents:
        encoded = json.dumps(document)
        for forbidden in FORBIDDEN_PRODUCTION_SHA256:
            assert forbidden not in encoded
        observed = set(_integers(document))
        assert not observed & set(FORBIDDEN_PRODUCTION_COUNTS)
