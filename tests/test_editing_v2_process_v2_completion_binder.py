"""The exact completion binder refuses every way a corpus can silently shrink.

``tests/test_editing_process_v2_rebind.py`` proves that ``bind_v1_semantic_payload``
validates whatever payload it is pointed at.  That is the wrong question for the
Process-V2 chain, whose failure mode is *nineteen of twenty task directories
bind cleanly*.  This module asks the right one: does the binder require the
exact ``SEMANTIC_MIGRATION_COMPLETE.json``, and does everything derive from it?

The fixture is a real twenty-cell migration run.  Each of the five declared data
lanes crossed with each of the four partition roles gets a real packed shard
built by the production materializer from real molecules through the real
executor, and several cells carry a genuinely V1-rejected trace, so the
completion's ``source``, ``admitted`` and ``rejected`` counts are measured rather
than asserted and ``source > admitted`` the way the production completion's are.

Also pinned here: the module's mirrored completion field set is checked against
the *sealing site's own source*, so if ``reduce_semantic_trace_migration`` ever
adds or renames a key this test fails instead of the binder silently accepting a
document it no longer understands.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from compose_v4.data.editing_process_v2_rebind import TASK_DIRNAME
from compose_v4.data.editing_v2_process_v2_completion_binder import (
    BINDING_SCHEMA,
    BINDING_STATUS,
    COMPLETION_FIELDS,
    EXPECTED_LANE_ROLE_GRID,
    PRODUCTION_COMPLETION_ARTIFACT_PATH,
    PRODUCTION_COMPLETION_EXPECTATION,
    PRODUCTION_COMPLETION_VOLUME_PATH,
    ProcessV2CompletionBindingError,
    ProcessV2CompletionExpectation,
    ProcessV2CompletionIncomplete,
    bind_exact_semantic_migration_completion,
    validate_exact_completion_binding_payload,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_FILENAME as MIGRATION_COMPLETION_FILENAME,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_SCHEMA as MIGRATION_COMPLETION_SCHEMA,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_SCHEMA_VERSION as MIGRATION_COMPLETION_SCHEMA_VERSION,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_STATUS as MIGRATION_COMPLETION_STATUS,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    PLAN_FILENAME as MIGRATION_PLAN_FILENAME,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    RECEIPT_FILENAME as V1_RECEIPT_FILENAME,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
)
from compose_v4.rewrite.operators import AtomDelete

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_editing_process_v2_rebind as v1_fixture  # noqa: E402

PAYLOAD_ARTIFACT_PATH = v1_fixture.PAYLOAD_ARTIFACT_PATH
COMPLETION_ARTIFACT_PATH = f"{PAYLOAD_ARTIFACT_PATH}/{MIGRATION_COMPLETION_FILENAME}"

# A charged lead whose only recorded edit deletes a methyl off its quaternary
# nitrogen.  The V1 semantic runtime refuses it on charge, so the cells that
# carry it publish a real rejection and the fixture's completion census has
# ``source > admitted`` without any count being invented.
_CHARGE_VIOLATING = "binder_charge_violating_demethylation"
_EXTRA_TRACES = {
    _CHARGE_VIOLATING: ("C[N+](C)(C)CC(=O)[O-]", (("atom_delete", AtomDelete(0)),)),
}

# Every cell gets at least one admissible trace; every fourth gets a second one
# plus the V1-rejected charge violation, so the census has both `admitted > 1`
# and `source > admitted` in a subset of cells rather than uniformly.
_CELL_TRACES = tuple(
    ("cyclize_hexane", _CHARGE_VIOLATING, "cyclize_fluoropentane")
    if index % 4 == 0
    else ("cyclize_fluoropentane",)
    for index in range(len(EXPECTED_LANE_ROLE_GRID))
)
_TASKS = tuple(
    (lane, role, _CELL_TRACES[index])
    for index, (lane, role) in enumerate(EXPECTED_LANE_ROLE_GRID)
)


@pytest.fixture(autouse=True)
def _register_extra_fixture_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, definition in _EXTRA_TRACES.items():
        monkeypatch.setitem(v1_fixture._FIXTURE_TRACES, name, definition)


# ---- A real twenty-cell migration run -----------------------------------------


def _read_receipt(payload_root: Path, identity: str) -> dict:
    return json.loads((payload_root / TASK_DIRNAME / identity / V1_RECEIPT_FILENAME).read_bytes())


def _seal(body: Mapping[str, object], *, field: str) -> dict:
    return {**dict(body), field: canonical_sha256(dict(body))}


def build_migration_run(
    artifact_root: Path,
    *,
    tasks: Sequence[tuple[str, str, tuple[str, ...]]] = _TASKS,
    drop_task_index: int | None = None,
) -> tuple[v1_fixture._V1Payload, dict, ProcessV2CompletionExpectation]:
    """Build the payload and seal a completion describing exactly it."""

    payload = v1_fixture._build_v1_payload(artifact_root, tasks=tuple(tasks))
    results = []
    totals = {"source": 0, "admitted": 0, "rejected": 0}
    for index, identity in enumerate(payload.v1_task_identities):
        receipt = _read_receipt(payload.payload_root, identity)
        counts = {field: int(receipt["counts"][field]) for field in ("source", "admitted", "rejected")}
        for field in counts:
            totals[field] += counts[field]
        results.append(
            {
                "task_identity_sha256": identity,
                "data_lane": receipt["data_lane"],
                "split": receipt["split"],
                "source_artifact_path": f"/artifacts/v1_source/shard_{index:04d}.jsonl.gz",
                "output_artifact_path": f"{PAYLOAD_ARTIFACT_PATH}/{TASK_DIRNAME}/{identity}",
                "receipt_sha256": receipt["receipt_sha256"],
                "semantic_shard_sha256": receipt["semantic_shard_sha256"],
                "semantic_manifest_sha256": receipt["semantic_manifest_sha256"],
                "counts": counts,
            }
        )

    plan_path = payload.payload_root / MIGRATION_PLAN_FILENAME
    plan_path.write_bytes(canonical_bytes({"fixture": "migration-plan"}) + b"\n")
    body = {
        "schema": MIGRATION_COMPLETION_SCHEMA,
        "schema_version": MIGRATION_COMPLETION_SCHEMA_VERSION,
        "status": MIGRATION_COMPLETION_STATUS,
        "training_authorized": False,
        "gate_zero_run": False,
        "run_identity_sha256": hashlib.sha256(b"fixture-run").hexdigest(),
        "plan_sha256": hashlib.sha256(b"fixture-plan").hexdigest(),
        "plan_file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "source_revision_sha256": hashlib.sha256(b"fixture-revision").hexdigest(),
        "source_inventory_sha256": hashlib.sha256(b"fixture-inventory").hexdigest(),
        # The fixture payload was just built, so its pinned identities are the
        # ones the builders emit here. Production pins the *superseded* pair;
        # what matters is that the binder reads the pinned values it is given
        # and never consults the live loader, which
        # ``test_the_binder_reads_a_payload_whose_live_identity_is_gone`` proves.
        "process_identity_sha256": v1_fixture.editing_v2_process_identity()[
            "process_identity_sha256"
        ],
        "builder_identity_sha256": v1_fixture.semantic_packed_builder_identity()["identity_sha256"],
        "task_inventory_sha256": hashlib.sha256(b"fixture-tasks").hexdigest(),
        "task_count": len(results),
        "result_inventory": results,
        "result_inventory_sha256": canonical_sha256(results),
        "counts": totals,
        "rejections_by_code": {},
        "accepted_family_histogram": {},
    }
    completion = _seal(body, field="completion_sha256")
    if drop_task_index is not None:
        dropped = completion["result_inventory"][drop_task_index]["task_identity_sha256"]
        remaining = [
            entry
            for entry in completion["result_inventory"]
            if entry["task_identity_sha256"] != dropped
        ]
        completion = reseal_completion(completion, result_inventory=remaining)
    write_completion(payload, completion)
    return payload, completion, expectation_for(completion)


def reseal_completion(completion: Mapping[str, object], **overrides: object) -> dict:
    """Reseal a completion around an override, so it is internally consistent."""

    body = {key: item for key, item in completion.items() if key != "completion_sha256"}
    body.update(overrides)
    if "result_inventory" in overrides:
        body["result_inventory_sha256"] = canonical_sha256(body["result_inventory"])
        body["task_count"] = len(body["result_inventory"])
    return _seal(body, field="completion_sha256")


def write_completion(payload: v1_fixture._V1Payload, completion: Mapping[str, object]) -> Path:
    path = payload.payload_root / MIGRATION_COMPLETION_FILENAME
    path.write_bytes(canonical_bytes(completion) + b"\n")
    return path


def expectation_for(completion: Mapping[str, object]) -> ProcessV2CompletionExpectation:
    counts = completion["counts"]
    return ProcessV2CompletionExpectation(
        completion_artifact_path=COMPLETION_ARTIFACT_PATH,
        run_identity_sha256=str(completion["run_identity_sha256"]),
        completion_sha256=str(completion["completion_sha256"]),
        process_identity_sha256=str(completion["process_identity_sha256"]),
        builder_identity_sha256=str(completion["builder_identity_sha256"]),
        result_inventory_sha256=str(completion["result_inventory_sha256"]),
        task_count=int(completion["task_count"]),
        source_traces=int(counts["source"]),
        admitted_traces=int(counts["admitted"]),
        rejected_traces=int(counts["rejected"]),
    )


def bind(payload: v1_fixture._V1Payload, expectation: ProcessV2CompletionExpectation):
    process_identity, builder_identity = v1_fixture._pinned_identities()
    return bind_exact_semantic_migration_completion(
        completion_artifact_path=expectation.completion_artifact_path,
        artifact_root=payload.artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
        expectation=expectation,
    )


# ---- 1. The happy path binds the exact twenty cells ----------------------------


def test_the_exact_completion_binds_twenty_cells_and_the_admitted_census(
    tmp_path: Path,
) -> None:
    payload, completion, expectation = build_migration_run(tmp_path / "artifacts")
    binding = bind(payload, expectation)

    assert len(binding.source_inventory) == 20
    assert sorted(
        (entry["data_lane"], entry["split"]) for entry in binding.source_inventory
    ) == sorted(EXPECTED_LANE_ROLE_GRID)
    assert binding.payload_root_artifact_path == PAYLOAD_ARTIFACT_PATH

    # The load-bearing count. The rebind reads the admitted records; the
    # migration's source count is strictly larger and is never the input.
    assert binding.counts["source"] > binding.counts["admitted"] > 0
    assert binding.counts["rejected"] > 0
    assert binding.rebind_source_entries == binding.counts["admitted"]
    assert int(binding.v1_payload_binding["v1_entry_count"]) == binding.counts["admitted"]
    assert completion["counts"] == dict(binding.counts)

    payload_body = binding.as_payload()
    assert payload_body["schema"] == BINDING_SCHEMA
    assert payload_body["status"] == BINDING_STATUS
    assert validate_exact_completion_binding_payload(payload_body) == payload_body
    assert payload_body["counts"]["rebind_source_entries"] == binding.counts["admitted"]
    assert all(
        value is False
        for key, value in payload_body.items()
        if key.endswith("_authorized")
    )


def test_the_binding_payload_is_byte_stable_across_two_independent_binds(
    tmp_path: Path,
) -> None:
    payload, _completion, expectation = build_migration_run(tmp_path / "artifacts")
    first = bind(payload, expectation).as_payload()
    second = bind(payload, expectation).as_payload()
    assert canonical_bytes(first) == canonical_bytes(second)


# ---- 2. A subset is refused, not silently accepted -----------------------------


def test_nineteen_of_twenty_task_directories_cannot_bind(tmp_path: Path) -> None:
    payload, _completion, expectation = build_migration_run(tmp_path / "artifacts")
    victim = payload.v1_task_identities[7]
    for entry in sorted((payload.payload_root / TASK_DIRNAME / victim).rglob("*"), reverse=True):
        entry.unlink() if entry.is_file() else entry.rmdir()
    (payload.payload_root / TASK_DIRNAME / victim).rmdir()

    with pytest.raises(ProcessV2CompletionIncomplete, match="not exactly the completion"):
        bind(payload, expectation)


def test_a_completion_that_declares_nineteen_cells_is_refused(tmp_path: Path) -> None:
    payload, _completion, expectation = build_migration_run(
        tmp_path / "artifacts", drop_task_index=3
    )
    # Declared expectation still says twenty, which is the production case.
    with pytest.raises(ProcessV2CompletionIncomplete, match="19 of 20 expected"):
        bind(payload, ProcessV2CompletionExpectation(**{**expectation.__dict__, "task_count": 20}))
    # Even a caller who lowered its own expectation to nineteen is refused,
    # because the lane/role grid is read from the corpus contract.
    with pytest.raises(ProcessV2CompletionIncomplete, match="lane/role grid disagrees"):
        bind(payload, expectation)


def test_an_unexpected_extra_task_directory_is_refused(tmp_path: Path) -> None:
    payload, _completion, expectation = build_migration_run(tmp_path / "artifacts")
    intruder = payload.payload_root / TASK_DIRNAME / ("f" * 64)
    intruder.mkdir()
    with pytest.raises(ProcessV2CompletionIncomplete, match="extras="):
        bind(payload, expectation)


def test_a_completion_missing_a_lane_role_cell_is_refused(tmp_path: Path) -> None:
    payload, completion, _expectation = build_migration_run(tmp_path / "artifacts")
    inventory = [dict(entry) for entry in completion["result_inventory"]]
    # Duplicate one cell over another: still twenty entries, still twenty task
    # directories, but the grid no longer covers every lane/role.
    inventory[1]["data_lane"] = inventory[0]["data_lane"]
    inventory[1]["split"] = inventory[0]["split"]
    mutated = reseal_completion(completion, result_inventory=inventory)
    write_completion(payload, mutated)
    with pytest.raises(ProcessV2CompletionIncomplete, match="lane/role grid disagrees"):
        bind(payload, expectation_for(mutated))


# ---- 3. Identity, hash and census mutations ------------------------------------


def test_an_equal_record_count_with_a_different_shard_hash_is_refused(
    tmp_path: Path,
) -> None:
    payload, completion, _expectation = build_migration_run(tmp_path / "artifacts")
    inventory = [dict(entry) for entry in completion["result_inventory"]]
    inventory[5]["semantic_shard_sha256"] = "0" * 64
    mutated = reseal_completion(completion, result_inventory=inventory)
    write_completion(payload, mutated)
    with pytest.raises(ProcessV2CompletionBindingError, match="semantic_shard_sha256"):
        bind(payload, expectation_for(mutated))


def test_a_completion_whose_admitted_census_is_not_the_shard_census_is_refused(
    tmp_path: Path,
) -> None:
    payload, completion, _expectation = build_migration_run(tmp_path / "artifacts")
    inventory = [dict(entry) for entry in completion["result_inventory"]]
    # Internally reconciled, so only the join against the real shard census
    # catches it.
    inventory[0]["counts"] = {"source": 1000, "admitted": 999, "rejected": 1}
    mutated = reseal_completion(completion, result_inventory=inventory)
    write_completion(payload, mutated)
    with pytest.raises(ProcessV2CompletionBindingError, match=r"disagrees .*\['entries'\]"):
        bind(payload, expectation_for(mutated))


def test_a_completion_census_that_is_not_the_sum_of_its_tasks_is_refused(
    tmp_path: Path,
) -> None:
    payload, completion, _expectation = build_migration_run(tmp_path / "artifacts")
    counts = dict(completion["counts"])
    mutated = reseal_completion(
        completion,
        counts={**counts, "source": counts["source"] + 7, "rejected": counts["rejected"] + 7},
    )
    write_completion(payload, mutated)
    with pytest.raises(ProcessV2CompletionBindingError, match="not the sum of its twenty"):
        bind(payload, expectation_for(mutated))


def test_a_resealed_completion_that_moves_a_hash_fails_its_declared_expectation(
    tmp_path: Path,
) -> None:
    payload, completion, expectation = build_migration_run(tmp_path / "artifacts")
    mutated = reseal_completion(completion, run_identity_sha256="1" * 64)
    write_completion(payload, mutated)
    # Resealing keeps the document self-consistent; only the declared
    # expectation catches it.
    with pytest.raises(ProcessV2CompletionBindingError, match="declared expectation"):
        bind(payload, expectation)


def test_a_completion_naming_another_process_identity_is_refused(tmp_path: Path) -> None:
    payload, completion, _expectation = build_migration_run(tmp_path / "artifacts")
    live = v1_fixture.editing_v2_process_identity()["process_identity_sha256"]
    assert live != SUPERSEDED_V1_PROCESS_IDENTITY_SHA256
    mutated = reseal_completion(
        completion, process_identity_sha256=SUPERSEDED_V1_PROCESS_IDENTITY_SHA256
    )
    write_completion(payload, mutated)
    with pytest.raises(ProcessV2CompletionBindingError, match="pinned identities"):
        bind(payload, expectation_for(mutated))


def test_the_binder_reads_a_payload_whose_live_identity_is_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical-pinned means pinned: the live loader is never consulted.

    Production's payload was built under a process identity that has since been
    superseded.  Making the live check fail outright is the sharpest available
    stand-in, and the binding still resolves because every identity on the path
    comes from the pinned objects the caller supplied.
    """

    import compose_v4.data.semantic_packed_trace_store as semantic_store

    payload, _completion, expectation = build_migration_run(tmp_path / "artifacts")

    def superseded(_expected: str) -> dict:
        raise v1_fixture.EditingV2ProcessIdentityError("live identity has been superseded")

    monkeypatch.setattr(semantic_store, "require_editing_v2_process_identity", superseded)
    binding = bind(payload, expectation)
    assert len(binding.source_inventory) == 20


def test_a_non_self_consistent_completion_is_refused(tmp_path: Path) -> None:
    payload, completion, expectation = build_migration_run(tmp_path / "artifacts")
    broken = {**completion, "task_count": 19}
    write_completion(payload, broken)
    with pytest.raises(ProcessV2CompletionBindingError, match="self-hash"):
        bind(payload, expectation)


def test_a_non_canonical_completion_encoding_is_refused(tmp_path: Path) -> None:
    payload, completion, expectation = build_migration_run(tmp_path / "artifacts")
    path = payload.payload_root / MIGRATION_COMPLETION_FILENAME
    path.write_bytes(json.dumps(completion, indent=2).encode("utf-8"))
    with pytest.raises(ProcessV2CompletionBindingError, match="canonically serialized"):
        bind(payload, expectation)


def test_an_absent_completion_refuses_rather_than_scanning_directories(
    tmp_path: Path,
) -> None:
    payload, _completion, expectation = build_migration_run(tmp_path / "artifacts")
    (payload.payload_root / MIGRATION_COMPLETION_FILENAME).unlink()
    with pytest.raises(ProcessV2CompletionIncomplete, match="run root inventory disagrees"):
        bind(payload, expectation)


def test_a_completion_path_other_than_the_declared_one_is_refused(tmp_path: Path) -> None:
    payload, completion, _expectation = build_migration_run(tmp_path / "artifacts")
    wrong = ProcessV2CompletionExpectation(
        **{
            **expectation_for(completion).__dict__,
            "completion_artifact_path": f"{PAYLOAD_ARTIFACT_PATH}/OTHER.json",
        }
    )
    with pytest.raises(ProcessV2CompletionBindingError, match="declared expected path"):
        bind_exact_semantic_migration_completion(
            completion_artifact_path=COMPLETION_ARTIFACT_PATH,
            artifact_root=payload.artifact_root,
            pinned_process_identity=v1_fixture._pinned_identities()[0],
            pinned_builder_identity=v1_fixture._pinned_identities()[1],
            expectation=wrong,
        )


# ---- 4. The mirrored schema cannot drift from its producer ---------------------


def test_the_mirrored_completion_field_set_equals_the_sealing_sites_own_keys() -> None:
    """Read the reducer's ``body`` literal rather than trusting the mirror."""

    source = (
        _REPO_ROOT / "src" / "compose_v4" / "data" / "semantic_trace_migration_mapreduce.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    reducer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "reduce_semantic_trace_migration"
    )
    sealed: set[str] | None = None
    for node in ast.walk(reducer):
        if not isinstance(node, ast.Dict):
            continue
        keys = {key.value for key in node.keys if isinstance(key, ast.Constant)}
        if "result_inventory_sha256" in keys and "counts" in keys:
            sealed = keys
            break
    assert sealed is not None, "the migration reducer no longer seals a literal completion body"
    assert sealed | {"completion_sha256"} == set(COMPLETION_FIELDS)


def test_the_production_expectation_is_pinned_and_mount_separated() -> None:
    assert PRODUCTION_COMPLETION_ARTIFACT_PATH == f"/artifacts{PRODUCTION_COMPLETION_VOLUME_PATH}"
    assert PRODUCTION_COMPLETION_VOLUME_PATH.endswith(f"/{MIGRATION_COMPLETION_FILENAME}")
    assert PRODUCTION_COMPLETION_EXPECTATION.task_count == 20
    assert PRODUCTION_COMPLETION_EXPECTATION.admitted_traces == 646_779
    assert PRODUCTION_COMPLETION_EXPECTATION.source_traces == 695_638
    assert (
        PRODUCTION_COMPLETION_EXPECTATION.admitted_traces
        + PRODUCTION_COMPLETION_EXPECTATION.rejected_traces
        == PRODUCTION_COMPLETION_EXPECTATION.source_traces
    )
    assert (
        PRODUCTION_COMPLETION_EXPECTATION.process_identity_sha256
        == SUPERSEDED_V1_PROCESS_IDENTITY_SHA256
    )


def test_a_binding_payload_claiming_the_source_count_as_its_input_is_refused() -> None:
    """The exact confusion the previous handoff made, refused mechanically."""

    body = {
        "schema": BINDING_SCHEMA,
        "schema_version": 1,
        "status": BINDING_STATUS,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "counts": {
            "source": 695_638,
            "admitted": 646_779,
            "rejected": 48_859,
            "rebind_source_entries": 695_638,
        },
    }
    payload = {**body, "binding_sha256": canonical_sha256(body)}
    with pytest.raises(ProcessV2CompletionBindingError, match="migration input count"):
        validate_exact_completion_binding_payload(payload)
