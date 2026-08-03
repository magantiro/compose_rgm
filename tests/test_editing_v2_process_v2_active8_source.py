"""The join between the Process-V2 overlay and the V1 lane/role inventory.

The two halves each have their own authority and their own tests. What is new,
and therefore what is tested here, is the JOIN: that the overlay and the
inventory describe the same payload, that the per-task census reconciles, and
that the two process identities stay distinguishable.

The failure this guards is specific and silent. An admitted overlay proved
against one migration, paired with a different migration's packed shards, yields
a corpus whose chemistry and whose admission decisions came from different runs.
Every count would reconcile run-wide, because the reduction only ever sees its
own overlay; only a cross-check against the shard identities catches it.

The two resolvers are substituted here rather than driven end to end. That is
deliberate: driving them needs a full V1 migration payload, which
`tests/test_process_v2_rebind_end_to_end.py` covers, and it would test the
resolvers rather than the join. Substituting them isolates the code this module
actually adds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data import editing_v2_process_v2_active8_source as joiner
from compose_v4.data.editing_v2_process_v2_active8_source import (
    ACTIVE8_SOURCE_IDENTITY_FIELDS,
    SOURCE_SCHEMA,
    SOURCE_SCHEMA_VERSION,
    SOURCE_SELF_HASH_FIELD,
    SOURCE_STATUS,
    ProcessV2Active8SourceError,
    ProcessV2Active8SourceIdentityError,
    resolve_process_v2_active8_source_inventory,
    validate_process_v2_active8_source_identity,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    ProcessV2SchemaError,
    canonical_sha256,
    require_no_granted_authority,
)

_ROOT = Path(__file__).resolve().parents[1]

# The V1 payload fixture builders are not importable as a package; pytest already
# puts ``tests`` on ``sys.path`` under the default prepend import mode.
if str(_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tests"))

import test_editing_process_v2_admitted_source as adapter_fixture  # noqa: E402

_effective_mask_authority = adapter_fixture._effective_mask_authority

# The SUPERSEDED V1 payload identity. Named for its lineage: the chain
# verifier refuses a historical value carried under a live-sounding name,
# and it is right to -- that is how a dead identity gets read as current.
_SUPERSEDED_V1_IDENTITY = "6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7"
_V2_IDENTITY = "0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd"

_TASKS = (
    {"task": "a" * 64, "shard": "1" * 64, "lane": "observed_local_analogue", "split": "train"},
    {"task": "b" * 64, "shard": "2" * 64, "lane": "reversible_synthetic_walk", "split": "train"},
)


# ---- Stand-ins for the two authorities ----


class _Source:
    def __init__(self, task: str, shard: str, lane: str, split: str) -> None:
        self.task_identity_sha256 = task
        self.semantic_shard_sha256 = shard
        self.semantic_manifest_sha256 = "3" * 64
        self.data_lane = lane
        self.partition_role = split


class _V1Inventory:
    def __init__(self, sources) -> None:
        self.sources = tuple(sources)
        self.migration_completion_sha256 = "4" * 64


class _Admitted:
    def __init__(self, *, results, counts, plan) -> None:
        self.completion = {"result_inventory": results, "completion_sha256": "5" * 64}
        self.counts = counts
        self.rejected_traces_by_code = {"atom_delete_outside_process_v2_mask": 1}
        self.plan = plan

    def identity(self) -> dict[str, Any]:
        return {"admitted_source_sha256": "6" * 64}


def _plan(tasks=_TASKS) -> dict[str, Any]:
    return {
        "v1_payload_binding": {
            "v1_tasks": [
                {
                    "v1_task_identity_sha256": task["task"],
                    "v1_semantic_shard_sha256": task["shard"],
                    "data_lane": task["lane"],
                    "split": task["split"],
                    "v1_entries": 4,
                }
                for task in tasks
            ]
        },
        "pinned_process_identity": {"process_identity_sha256": _SUPERSEDED_V1_IDENTITY},
        "process_v2_identity": {"process_identity_sha256": _V2_IDENTITY},
    }


def _install(monkeypatch, *, admitted, inventory) -> None:
    monkeypatch.setattr(
        joiner, "resolve_process_v2_admitted_source", lambda *a, **k: admitted
    )
    monkeypatch.setattr(
        joiner, "resolve_editing_v2_semantic_active8_sources", lambda *a, **k: inventory
    )


def _resolve(plan):
    return resolve_process_v2_active8_source_inventory(
        plan,
        v1_migration_completion_path=Path("/artifacts/x/COMPLETE.json"),
        artifact_root=Path("/artifacts"),
        repo_root=Path("/repo"),
    )


def _admitted_for(tasks=_TASKS, *, admitted_each=3, rejected_each=1, plan=None):
    results = [
        {
            "v1_task_identity_sha256": task["task"],
            "counts": {"admitted_entries": admitted_each, "rejected_entries": rejected_each},
        }
        for task in tasks
    ]
    total = len(tasks)
    counts = {
        "source_entries": total * (admitted_each + rejected_each),
        "admitted_entries": total * admitted_each,
        "rejected_entries": total * rejected_each,
        "admitted_states": total * admitted_each * 2,
        "admitted_transitions": total * admitted_each,
    }
    return _Admitted(results=results, counts=counts, plan=plan or _plan(tasks))


def _inventory_for(tasks=_TASKS):
    return _V1Inventory(
        _Source(task["task"], task["shard"], task["lane"], task["split"]) for task in tasks
    )


# ---- The join succeeds and keeps both identities distinguishable ----


def test_the_join_reconciles_and_keeps_both_identities_separate(monkeypatch) -> None:
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for())
    resolved = _resolve(_plan())

    assert resolved.v1_payload_process_identity_sha256 == _SUPERSEDED_V1_IDENTITY
    assert resolved.process_v2_identity_sha256 == _V2_IDENTITY
    # The two must never be collapsed into one field: that is how a V1 artifact
    # comes to be read as a V2 one.
    assert resolved.v1_payload_process_identity_sha256 != resolved.process_v2_identity_sha256

    assert len(resolved.sources) == len(_TASKS)
    for source in resolved.sources:
        assert source.admitted_entry_count + source.rejected_entry_count == (
            source.v1_entry_count
        )

    identity = resolved.identity()
    assert identity["schema"] == SOURCE_SCHEMA
    assert identity["status"] == SOURCE_STATUS
    assert identity["v1_payload_process_identity_sha256"] == _SUPERSEDED_V1_IDENTITY
    assert identity["process_v2_identity_sha256"] == _V2_IDENTITY
    assert len(identity["process_v2_active8_source_sha256"]) == 64


def test_the_source_identity_grants_nothing(monkeypatch) -> None:
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for())
    identity = _resolve(_plan()).identity()
    for flag in (
        "training_authorized",
        "gate_zero_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
    ):
        assert identity[flag] is False, flag


# ---- The cross-checks this module exists for ----


def test_a_different_task_set_is_refused(monkeypatch) -> None:
    """An overlay proved against one migration, paired with another's shards."""

    other = (
        {"task": "c" * 64, "shard": "1" * 64, "lane": "observed_local_analogue", "split": "train"},
        _TASKS[1],
    )
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for(other))
    with pytest.raises(ProcessV2Active8SourceError, match="different\n?\\s*task sets|different task sets"):
        _resolve(_plan())


def test_a_different_semantic_shard_for_the_same_task_is_refused(monkeypatch) -> None:
    """Same task identity, different chemistry: the silent case."""

    drifted = (
        {**_TASKS[0], "shard": "9" * 64},
        _TASKS[1],
    )
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for(drifted))
    with pytest.raises(ProcessV2Active8SourceError, match="different semantic shard"):
        _resolve(_plan())


def test_a_lane_or_split_disagreement_is_refused(monkeypatch) -> None:
    relabelled = (
        {**_TASKS[0], "split": "validation"},
        _TASKS[1],
    )
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for(relabelled))
    with pytest.raises(ProcessV2Active8SourceError, match="lane or split disagrees"):
        _resolve(_plan())


def test_a_per_task_census_that_does_not_reconcile_is_refused(monkeypatch) -> None:
    """Run-wide totals can reconcile while a single task does not."""

    admitted = _admitted_for(admitted_each=2, rejected_each=1)  # 3 != v1_entries 4
    _install(monkeypatch, admitted=admitted, inventory=_inventory_for())
    with pytest.raises(ProcessV2Active8SourceError, match="admits 2 and rejects 1 of 4"):
        _resolve(_plan())


def test_the_joined_total_must_account_for_every_source_entry(monkeypatch) -> None:
    """The per-task sum and the published run census must agree."""

    admitted = _admitted_for()
    admitted.counts = {**admitted.counts, "source_entries": 99}
    _install(monkeypatch, admitted=admitted, inventory=_inventory_for())
    with pytest.raises(ProcessV2Active8SourceError, match="does not account for every source entry"):
        _resolve(_plan())


# ---- Schema 2: the published identity, checked by its owning validator ----
#
# The tests above substitute both resolvers, which isolates the join.  The tests
# below drive a REAL admitted source through the identity builder, because the
# identity's whole new claim is about the values the embedded descriptor carries,
# and a stand-in descriptor cannot make that claim wrong.


@pytest.fixture(scope="module")
def real_join(tmp_path_factory: pytest.TempPathFactory):
    """One real published overlay, joined to a V1 inventory naming its own tasks."""

    root = tmp_path_factory.mktemp("active8_source_payload")
    fixture = adapter_fixture._build_and_prove(
        root / "artifacts", adapter_fixture._TASKS_WITH_EXCLUSION
    )
    admitted = adapter_fixture._resolve(fixture)
    inventory = _V1Inventory(
        _Source(
            str(task["v1_task_identity_sha256"]),
            str(task["v1_semantic_shard_sha256"]),
            str(task["data_lane"]),
            str(task["split"]),
        )
        for task in fixture.plan["v1_payload_binding"]["v1_tasks"]
    )
    return fixture.plan, admitted, inventory


def _resolve_real(monkeypatch, real_join):
    plan, admitted, inventory = real_join
    _install(monkeypatch, admitted=admitted, inventory=inventory)
    return _resolve(plan)


def _reseal(payload: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in sorted(payload.items())
        if key != SOURCE_SELF_HASH_FIELD
    }
    return dict(sorted({**body, SOURCE_SELF_HASH_FIELD: canonical_sha256(body)}.items()))


def test_a_real_source_identity_validates_and_passes_the_recursive_authority_guard(
    monkeypatch, real_join
) -> None:
    identity = _resolve_real(monkeypatch, real_join).identity()
    assert identity["schema_version"] == SOURCE_SCHEMA_VERSION == 2
    assert set(identity) == set(ACTIVE8_SOURCE_IDENTITY_FIELDS)
    # The shared vocabulary-free guard, over the whole nested object.
    require_no_granted_authority(identity, label="the joined identity")
    assert (
        validate_process_v2_active8_source_identity(identity, repo_root=_ROOT) == identity
    )
    # And the embedded descriptor is the real one, verbatim.
    _plan_, admitted, _inventory = real_join
    assert identity["admitted_source_identity"] == admitted.identity()


def test_the_real_identity_publishes_all_seven_authority_fields(
    monkeypatch, real_join
) -> None:
    """Version 1 published four, so three could not be read as false at all."""

    identity = _resolve_real(monkeypatch, real_join).identity()
    for name in AUTHORITY_FIELDS:
        assert identity[name] is False, name
        assert identity["admitted_source_identity"][name] is False, name
    assert len([key for key in identity if key.endswith("_authorized")]) == 7


def test_the_real_identity_is_sorted_at_every_depth(monkeypatch, real_join) -> None:
    identity = _resolve_real(monkeypatch, real_join).identity()

    def _mappings(node: object):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from _mappings(value)
        elif isinstance(node, list):
            for value in node:
                yield from _mappings(value)

    seen = list(_mappings(identity))
    assert len(seen) >= 6, "the embedded descriptor and the source rows must be reached"
    for node in seen:
        assert list(node) == sorted(node), node


def test_a_grant_nested_in_the_embedded_descriptor_or_a_source_row_is_refused(
    monkeypatch, real_join
) -> None:
    """Both nesting shapes: a mapping, and a mapping inside a list."""

    identity = _resolve_real(monkeypatch, real_join).identity()

    nested_mapping = _reseal(
        {
            **identity,
            "admitted_source_identity": {
                **identity["admitted_source_identity"],
                "gate_zero_authorized": True,
            },
        }
    )
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="grants authority"):
        validate_process_v2_active8_source_identity(nested_mapping, repo_root=_ROOT)

    rows = [dict(row) for row in identity["sources"]]
    rows[0]["t1_authorized"] = True
    nested_in_list = _reseal({**identity, "sources": rows})
    with pytest.raises(
        ProcessV2Active8SourceIdentityError, match="grants authority"
    ) as raised:
        validate_process_v2_active8_source_identity(nested_in_list, repo_root=_ROOT)
    assert "sources[0].t1_authorized" in str(raised.value)
    # The shared guard reaches it directly too, which is what makes the two
    # independent rather than one calling the other.
    with pytest.raises(ProcessV2SchemaError, match="sources\\[0\\].t1_authorized"):
        require_no_granted_authority(nested_in_list, label="the joined identity")


def test_the_embedded_descriptor_is_checked_by_its_own_owner(
    monkeypatch, real_join
) -> None:
    """Delegation is the property, so the descriptor must be broken ONLY internally.

    Every joint statement below still holds -- same Process-V2 identity, same V1
    payload identity, same census, same rejection census -- so nothing in this
    module's own checks can see the defect. Only calling the admitted source's
    validator does.
    """

    identity = _resolve_real(monkeypatch, real_join).identity()
    descriptor = {
        key: value
        for key, value in sorted(identity["admitted_source_identity"].items())
        if key != "admitted_source_sha256"
    }
    descriptor["adapter_implementation_sha256"] = "c" * 64
    descriptor = dict(
        sorted(
            {**descriptor, "admitted_source_sha256": canonical_sha256(descriptor)}.items()
        )
    )
    broken = _reseal({**identity, "admitted_source_identity": descriptor})

    # The joint statements are untouched, which is what makes this a delegation test.
    assert descriptor["process_v2_identity_sha256"] == broken["process_v2_identity_sha256"]
    assert descriptor["pinned_process_identity_sha256"] == (
        broken["v1_payload_process_identity_sha256"]
    )
    assert dict(descriptor["counts"]) == dict(broken["counts"])

    with pytest.raises(
        ProcessV2Active8SourceIdentityError, match="embeds an admitted-source identity"
    ):
        validate_process_v2_active8_source_identity(broken, repo_root=_ROOT)


def test_a_descriptor_from_another_run_is_refused(monkeypatch, real_join) -> None:
    """Every per-half check passes; only the joint statement is false."""

    identity = _resolve_real(monkeypatch, real_join).identity()
    for top_level, nested, message in (
        ("process_v2_identity_sha256", None, "was resolved under"),
        ("v1_payload_process_identity_sha256", None, "but its embedded"),
    ):
        broken = _reseal({**identity, top_level: "8" * 64})
        assert nested is None
        with pytest.raises(ProcessV2Active8SourceIdentityError, match=message):
            validate_process_v2_active8_source_identity(broken, repo_root=_ROOT)


def test_collapsing_the_two_identities_into_one_value_is_refused(
    monkeypatch, real_join
) -> None:
    """Collapsed CONSISTENTLY, so the per-half comparison agrees and passes.

    Collapsing only the top-level name is caught by the descriptor comparison, so
    the interesting case is the one where both sides say the same wrong thing --
    which is exactly what a consumer that "simplified" the two names into one
    would produce.
    """

    identity = _resolve_real(monkeypatch, real_join).identity()
    live_v2 = identity["process_v2_identity_sha256"]
    descriptor = {
        key: value
        for key, value in sorted(identity["admitted_source_identity"].items())
        if key != "admitted_source_sha256"
    }
    descriptor["pinned_process_identity_sha256"] = live_v2
    descriptor = dict(
        sorted(
            {
                **descriptor,
                "admitted_source_sha256": canonical_sha256(descriptor),
            }.items()
        )
    )
    collapsed = _reseal(
        {
            **identity,
            "admitted_source_identity": descriptor,
            "v1_payload_process_identity_sha256": live_v2,
        }
    )
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="under both identity names"):
        validate_process_v2_active8_source_identity(collapsed, repo_root=_ROOT)


def test_a_census_the_embedded_descriptor_does_not_carry_is_refused(
    monkeypatch, real_join
) -> None:
    identity = _resolve_real(monkeypatch, real_join).identity()
    counts = dict(sorted({**identity["counts"], "admitted_states": 999}.items()))
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="does not carry"):
        validate_process_v2_active8_source_identity(
            _reseal({**identity, "counts": counts}), repo_root=_ROOT
        )
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="rejection census"):
        validate_process_v2_active8_source_identity(
            _reseal({**identity, "rejected_traces_by_code": {}}), repo_root=_ROOT
        )


def test_per_source_rows_must_reconcile_against_the_global_census(
    monkeypatch, real_join
) -> None:
    """A run-wide census can reconcile while its rows do not."""

    identity = _resolve_real(monkeypatch, real_join).identity()
    rows = [dict(row) for row in identity["sources"]]
    # Relabel one rejection as an admission. The row still reconciles against its
    # own v1_entry_count, every count stays non-negative, and the global
    # source == admitted + rejected identity still holds, so only the
    # row-to-census comparison can catch it.
    target = next(
        index for index, row in enumerate(rows) if row["rejected_entry_count"] >= 1
    )
    rows[target]["admitted_entry_count"] += 1
    rows[target]["rejected_entry_count"] -= 1
    assert rows[target]["admitted_entry_count"] + rows[target]["rejected_entry_count"] == (
        rows[target]["v1_entry_count"]
    )
    assert sum(row["v1_entry_count"] for row in rows) == identity["counts"]["source_entries"]
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="admitted_entry_count totals"):
        validate_process_v2_active8_source_identity(
            _reseal({**identity, "sources": rows}), repo_root=_ROOT
        )


def test_a_row_that_does_not_reconcile_internally_is_refused(
    monkeypatch, real_join
) -> None:
    identity = _resolve_real(monkeypatch, real_join).identity()
    rows = [dict(row) for row in identity["sources"]]
    rows[0]["v1_entry_count"] += 1
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="admits .* and rejects"):
        validate_process_v2_active8_source_identity(
            _reseal({**identity, "sources": rows}), repo_root=_ROOT
        )


def test_duplicate_or_unordered_source_rows_are_refused(
    monkeypatch, real_join
) -> None:
    identity = _resolve_real(monkeypatch, real_join).identity()
    rows = [dict(row) for row in identity["sources"]]
    assert len(rows) >= 2, "the fixture must have more than one lane/role"

    duplicated = _reseal({**identity, "sources": [rows[0], dict(rows[0])]})
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="one task is one row"):
        validate_process_v2_active8_source_identity(duplicated, repo_root=_ROOT)

    unordered = _reseal({**identity, "sources": list(reversed(rows))})
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="out of order"):
        validate_process_v2_active8_source_identity(unordered, repo_root=_ROOT)


def test_schema_version_one_is_refused_as_an_incompatibility(
    monkeypatch, real_join
) -> None:
    identity = _resolve_real(monkeypatch, real_join).identity()
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="incompatible with 2"):
        validate_process_v2_active8_source_identity(
            _reseal({**identity, "schema_version": 1}), repo_root=_ROOT
        )


def test_a_missing_extra_or_coerced_field_is_refused(monkeypatch, real_join) -> None:
    identity = _resolve_real(monkeypatch, real_join).identity()

    missing = _reseal(
        {k: v for k, v in identity.items() if k != "v1_migration_completion_sha256"}
    )
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="missing"):
        validate_process_v2_active8_source_identity(missing, repo_root=_ROOT)

    extra = _reseal({**identity, "materialized_later": "0" * 64})
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="unexpected"):
        validate_process_v2_active8_source_identity(extra, repo_root=_ROOT)

    uppercase = _reseal(
        {**identity, "v1_migration_completion_sha256": "A" * 64}
    )
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="lowercase hex"):
        validate_process_v2_active8_source_identity(uppercase, repo_root=_ROOT)

    rows = [dict(row) for row in identity["sources"]]
    rows[0]["v1_entry_count"] = str(rows[0]["v1_entry_count"])
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="not an exact int"):
        validate_process_v2_active8_source_identity(
            _reseal({**identity, "sources": rows}), repo_root=_ROOT
        )


def test_an_unsorted_identity_is_refused_even_though_it_self_hashes(
    monkeypatch, real_join
) -> None:
    identity = _resolve_real(monkeypatch, real_join).identity()
    shuffled = dict(reversed(list(identity.items())))
    assert shuffled == identity
    assert shuffled[SOURCE_SELF_HASH_FIELD] == canonical_sha256(
        {k: v for k, v in shuffled.items() if k != SOURCE_SELF_HASH_FIELD}
    )
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="sorted"):
        validate_process_v2_active8_source_identity(shuffled, repo_root=_ROOT)


def test_a_disagreeing_self_hash_is_refused(monkeypatch, real_join) -> None:
    identity = _resolve_real(monkeypatch, real_join).identity()
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="self-hash"):
        validate_process_v2_active8_source_identity(
            {**identity, SOURCE_SELF_HASH_FIELD: "2" * 64}, repo_root=_ROOT
        )


def test_the_taxonomy_version_in_the_cell_namespace_is_not_renamed() -> None:
    """`editing_v2_active8_v1` is a taxonomy version, not a schema version.

    Bumping the Active8 SOURCE schema to 2 must not drag the cell namespace with
    it: the two are different kinds of version and only look alike.
    """

    registry = (_ROOT / "configs/editing_v2_process_v2_capability_cells.json").read_text()
    assert "editing_v2_active8_v1" in registry
    roles = (
        _ROOT / "configs/editing_v2_process_v2_development_cell_roles.json"
    ).read_text()
    assert "editing_v2_active8_v1" in roles


def test_a_non_object_identity_is_refused() -> None:
    with pytest.raises(ProcessV2Active8SourceIdentityError, match="must be an object"):
        validate_process_v2_active8_source_identity([], repo_root=_ROOT)


