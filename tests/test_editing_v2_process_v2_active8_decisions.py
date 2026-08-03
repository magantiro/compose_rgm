"""The Process-V2 Active8 map/reduce, driven end to end on a real chain.

Nothing here is monkeypatched into agreement.  The fixture builds a real
twenty-cell V1 semantic migration with the production materializer, caches it
with the production chunk cache, rebinds it with the production Process-V2
rebind, resolves the production admitted source, constructs the Process-V2
scratch model through the frozen model/process contract, and only then runs the
Active8 decision stage over it.  Every number asserted below is measured from
those artifacts.

The fixture cell is chosen so all three census categories are populated by real
chemistry rather than by construction:

* ``cyclize_hexane`` -- rebind-admitted and Active8-accepted;
* ``open_cyclohexane`` -- a connected-nonleaf ring-atom deletion, which only the
  Process-V2 fiber can represent, rebind-admitted and Active8-accepted;
* ``open_benzene`` -- an aromatic connected-nonleaf deletion the Process-V2
  rebind refuses, so it arrives already rejected and is never candidate-
  evaluated;
* ``active8_ring_reorder`` -- rebind-admitted, and Active8-excluded because its
  MIDDLE action reorders a ring bond, which the production model's peripheral
  bond-reorder mask does not carry.

Acceptance tests 7, 8 and 9 live here.  Tests 3, 4 and 5 are in
``test_editing_v2_process_v2_active8_policy``; test 10 is in
``test_editing_v2_process_v2_active8_index``.
"""

from __future__ import annotations

import gzip
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_process_v2_admitted_source import (
    resolve_process_v2_admitted_source,
)
from compose_v4.data.editing_process_v2_rebind import (
    bind_process_v2_chunk_cache_generation,
    execute_process_v2_rebind_task,
    plan_process_v2_rebind,
    reduce_process_v2_rebind,
    write_process_v2_rebind_plan,
)
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_EXCLUDED,
    JOIN_KEY_FIELDS,
    TRACE_KEY_FIELDS,
    UPSTREAM_REJECTED,
)
from compose_v4.data.editing_v2_process_v2_active8_mapreduce import (
    COMPLETION_FILENAME,
    COUNT_FIELDS,
    DECISION_FILENAME,
    RECEIPT_FILENAME,
    TASK_DIRNAME,
    ProcessV2Active8Incomplete,
    ProcessV2Active8MapReduceError,
    completed_process_v2_active8_task_ids,
    execute_process_v2_active8_decision_task,
    iter_process_v2_active8_decision_rows,
    plan_process_v2_active8_decisions,
    reduce_process_v2_active8_decisions,
    run_process_v2_active8_map,
    validate_process_v2_active8_plan,
    write_process_v2_active8_plan,
)
from compose_v4.data.editing_v2_process_v2_active8_runtime import (
    build_process_v2_active8_runtime,
    load_process_v2_active8_model_config,
    runtime_identity_resolver,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    execute_process_v2_chunk_cache_task,
    plan_process_v2_chunk_cache,
    reduce_process_v2_chunk_cache,
    write_process_v2_chunk_cache_plan,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.experiments.editing_v2_semantic_runtime import SemanticScratchModelConfig
from compose_v4.rewrite.operators import AtomDelete, BondInsert, BondReorder

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_editing_process_v2_rebind as v1_fixture  # noqa: E402
import test_editing_v2_process_v2_completion_binder as binder_fixture  # noqa: E402

ROOT = _REPO_ROOT
RECORDS_PER_CHUNK = 2

#: Rebind-admitted and Active8-excluded, at its MIDDLE step: closing the ring is
#: supported, deleting the pendant methyl is supported, and reordering a RING
#: bond is not, because the production bond-reorder mask is peripheral-only.
_ACTIVE8_EXCLUDED_TRACE = "active8_ring_reorder"
_EXTRA_TRACES = {
    _ACTIVE8_EXCLUDED_TRACE: (
        "CCCCCCC",
        (
            ("bond_insert", BondInsert(0, 5, 1)),
            ("bond_reorder", BondReorder(0, 1, 2)),
            ("atom_delete", AtomDelete(6)),
        ),
    ),
}

#: Four traces, one per outcome the census has to distinguish, in every cell of
#: the twenty-cell grid.  The molecules repeat across cells on purpose: the
#: production candidate checker caches by exact source state, so twenty cells
#: cost what one cell costs plus the reads.
_CELL = (
    "cyclize_hexane",
    "open_cyclohexane",
    "open_benzene",
    _ACTIVE8_EXCLUDED_TRACE,
)
_TASKS = tuple((lane, role, _CELL) for lane, role, _names in binder_fixture._TASKS)


@pytest.fixture(autouse=True)
def _require_the_production_admission_authority() -> None:
    """Refuse to run against the fixture module's stand-in mask.

    ``test_editing_process_v2_rebind`` installs an independent oracle only while
    ``process_v2_atom_delete_mask`` is absent.  If that ever happened here the
    upstream rejections below would be produced by a stand-in, and this module
    would silently stop driving production code.
    """

    if v1_fixture.AUTHORITY_SUBSTITUTED:
        pytest.fail("the production Process-V2 admission authority is absent")


# ---- The real chain -----------------------------------------------------------


@dataclass(frozen=True)
class Chain:
    """Every artifact the Active8 stage is planned against, already published."""

    payload: Any
    artifact_root: Path
    cache_plan: dict[str, Any]
    cache_completion: dict[str, Any]
    rebind_plan: dict[str, Any]
    rebind_completion: dict[str, Any]
    admitted_source_identity: dict[str, Any]
    runtime: Any


def _small_model_config() -> SemanticScratchModelConfig:
    """The frozen architecture, narrowed to what a bounded fixture can run.

    Read from the frozen decision-runtime contract and then reduced in exactly
    three coordinates -- width, depth and slot count -- so the modes, the mark
    dimension, the vocabulary width and the RingCore catalog stay the production
    ones.  The narrowing is stated because it is real: this proves the stage, not
    the production model's capacity.
    """

    frozen = load_process_v2_active8_model_config(repo_root=ROOT)
    return SemanticScratchModelConfig(
        initialization_seed=frozen.initialization_seed,
        max_atoms=v1_fixture.SLOTS,
        hidden_dim=32,
        message_passing_steps=2,
        mark_dim=frozen.mark_dim,
        dtype=frozen.dtype,
        atom_vocabulary_class_count=frozen.atom_vocabulary_class_count,
        catalog_fingerprint=frozen.catalog_fingerprint,
    )


def build_chain(tmp_path: Path, *, tasks=_TASKS) -> Chain:
    """Build every upstream artifact with the production writers."""

    artifact_root = tmp_path / "artifacts"
    payload, _completion, expectation = binder_fixture.build_migration_run(
        artifact_root, tasks=tasks
    )
    binding = binder_fixture.bind(payload, expectation)

    cache_plan = plan_process_v2_chunk_cache(
        binding,
        repo_root=ROOT,
        output_artifact_prefix="/artifacts/active8_cache",
        records_per_chunk=RECORDS_PER_CHUNK,
    )
    write_process_v2_chunk_cache_plan(cache_plan, artifact_root=payload.artifact_root)
    for task in cache_plan["tasks"]:
        execute_process_v2_chunk_cache_task(
            cache_plan, task["task_identity_sha256"], artifact_root=payload.artifact_root
        )
    cache_completion = reduce_process_v2_chunk_cache(
        cache_plan, artifact_root=payload.artifact_root
    )

    cache_binding = bind_process_v2_chunk_cache_generation(
        str(cache_plan["run_artifact_root"]),
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    process_identity, builder_identity = v1_fixture._pinned_identities()
    v1_binding = v1_fixture.bind_v1_semantic_payload(
        payload_root_artifact_path=v1_fixture.PAYLOAD_ARTIFACT_PATH,
        artifact_root=payload.artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    rebind_plan = plan_process_v2_rebind(
        v1_binding,
        source_revision=v1_fixture._source_revision(),
        repo_root=ROOT,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
        cache_binding=cache_binding,
        output_artifact_prefix="/artifacts/active8_rebind",
        entries_per_task=RECORDS_PER_CHUNK,
    )
    write_process_v2_rebind_plan(
        rebind_plan, artifact_root=payload.artifact_root, repo_root=ROOT
    )
    for task in rebind_plan["tasks"]:
        execute_process_v2_rebind_task(
            rebind_plan,
            task["task_identity_sha256"],
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )
    rebind_completion = reduce_process_v2_rebind(
        rebind_plan, artifact_root=payload.artifact_root, repo_root=ROOT
    )
    admitted = resolve_process_v2_admitted_source(
        rebind_plan, artifact_root=payload.artifact_root, repo_root=ROOT
    )
    return Chain(
        payload=payload,
        artifact_root=payload.artifact_root,
        cache_plan=cache_plan,
        cache_completion=cache_completion,
        rebind_plan=rebind_plan,
        rebind_completion=rebind_completion,
        admitted_source_identity=dict(admitted.identity()),
        runtime=build_process_v2_active8_runtime(
            model_config=_small_model_config(), repo_root=ROOT
        ),
    )


def active8_plan(chain: Chain, *, prefix: str = "/artifacts/active8_decisions") -> dict[str, Any]:
    return plan_process_v2_active8_decisions(
        cache_run_artifact_root=str(chain.cache_plan["run_artifact_root"]),
        rebind_plan=chain.rebind_plan,
        rebind_completion=chain.rebind_completion,
        admitted_source_identity=chain.admitted_source_identity,
        model_runtime_identity=chain.runtime.identity,
        artifact_root=chain.artifact_root,
        repo_root=ROOT,
        output_artifact_prefix=prefix,
    )


def execute_active8(
    chain: Chain, plan: dict[str, Any], *, order: list[int] | None = None
) -> None:
    indices = list(range(len(plan["tasks"]))) if order is None else order
    for index in indices:
        execute_process_v2_active8_decision_task(
            plan,
            plan["tasks"][index]["task_identity_sha256"],
            artifact_root=chain.artifact_root,
            repo_root=ROOT,
            model=chain.runtime.model,
            candidate_checker=chain.runtime.checker,
            model_runtime_identity_resolver=runtime_identity_resolver(chain.runtime),
        )


def run_active8(
    chain: Chain, *, prefix: str = "/artifacts/active8_decisions", order: list[int] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = active8_plan(chain, prefix=prefix)
    write_process_v2_active8_plan(plan, artifact_root=chain.artifact_root, repo_root=ROOT)
    execute_active8(chain, plan, order=order)
    completion = reduce_process_v2_active8_decisions(
        plan, artifact_root=chain.artifact_root, repo_root=ROOT
    )
    return plan, completion


def mounted(chain: Chain, artifact_path: str) -> Path:
    return chain.artifact_root / str(artifact_path).removeprefix("/artifacts/")


@pytest.fixture(name="chain", scope="module")
def _chain_fixture(tmp_path_factory: pytest.TempPathFactory) -> Chain:
    """One built chain shared by every test in this module.

    Module-scoped deliberately: building it runs a real migration, a real cache,
    a real rebind and a real model construction, and every test below reads the
    same immutable artifacts.  The Active8 stage itself is re-run per test into
    its own artifact prefix, so no test observes another's output.
    """

    monkeypatch = pytest.MonkeyPatch()
    for name, definition in {**binder_fixture._EXTRA_TRACES, **_EXTRA_TRACES}.items():
        monkeypatch.setitem(v1_fixture._FIXTURE_TRACES, name, definition)
    try:
        return build_chain(tmp_path_factory.mktemp("active8_chain"))
    finally:
        monkeypatch.undo()


# ---- Acceptance test 7: the bounded independent oracle -------------------------


def _oracle_rows(chain: Chain) -> dict[tuple[str, int], dict[str, Any]]:
    """Decide every entry a second time, without the map/reduce.

    The oracle shares the policy and the checker -- those are the production
    decision under test and must not be reimplemented -- but reaches the rows a
    different way: it walks the cache generation's targets directly, joins the
    rebind decisions itself, and never opens a published Active8 artifact.  What
    it independently produces is the per-entry outcome and the four candidate
    counts, which is exactly what the published rows are compared against.
    """

    from compose_v4.data.editing_v2_process_v2_active8_mapreduce import (
        _chunk_target_for_task,
        read_upstream_rebind_decisions,
    )
    from compose_v4.data.editing_v2_process_v2_active8_policy import (
        evaluate_process_v2_active8_trace,
        upstream_rejected_trace_decision,
    )
    from compose_v4.data.editing_v2_process_v2_chunk_cache import (
        read_process_v2_chunk_target,
    )

    plan = active8_plan(chain, prefix="/artifacts/active8_oracle")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for task in plan["tasks"]:
        upstream = read_upstream_rebind_decisions(task, artifact_root=chain.artifact_root)
        target = _chunk_target_for_task(task)
        source_output = mounted(chain, target.source_artifact_path)
        for read in read_process_v2_chunk_target(
            source_output,
            target=target,
            expected_process_identity=plan["pinned_process_identity"],
            repo_root=ROOT,
        ):
            admitted, upstream_row = upstream[read.entry_index]
            if admitted:
                decision = evaluate_process_v2_active8_trace(
                    read.addressed,
                    candidate_check=chain.runtime.checker,
                    policy=chain.runtime.policy,
                )
            else:
                decision = upstream_rejected_trace_decision(
                    trace_id=read.addressed.address.trace_id,
                    rejection_code=str(upstream_row["exclusion_code"]),
                    policy=chain.runtime.policy,
                )
            rows[(str(task["v1_task_identity_sha256"]), read.entry_index)] = {
                "trace_id": read.addressed.address.trace_id,
                "category": decision.category,
                "active8_status": decision.active8_status,
                "marks": [
                    None
                    if item.candidate_evidence is None
                    else (
                        item.candidate_evidence.raw_mark_count,
                        item.candidate_evidence.canonical_successor_count,
                        item.candidate_evidence.matching_mark_count,
                        item.candidate_evidence.successor_alias_count,
                        item.candidate_evidence.action_sha256,
                        item.candidate_evidence.target_state_sha256,
                    )
                    for item in decision.action_decisions
                ],
            }
    return rows


def test_every_published_count_agrees_with_the_bounded_independent_oracle(
    chain: Chain,
) -> None:
    """Acceptance 7. Without this the counts are asserted only against themselves.

    Raw mark counts, canonical successor counts, alias multiplicity, the teacher
    mark identity and the teacher successor identity are all published per
    action.  A test that recomputed them from the published row would compare a
    value with itself; this recomputes them from the model, through a second
    traversal that never opens an Active8 artifact.
    """

    plan, completion = run_active8(chain, prefix="/artifacts/active8_oracle_run")
    oracle = _oracle_rows(chain)
    assert oracle, "the oracle decided nothing, so the comparison would be vacuous"

    published: dict[tuple[str, int], dict[str, Any]] = {}
    for row in iter_process_v2_active8_decision_rows(
        plan, artifact_root=chain.artifact_root, repo_root=ROOT
    ):
        key = (str(row["v1_task_identity_sha256"]), int(row["entry_index"]))
        assert key not in published
        published[key] = row
    assert set(published) == set(oracle)

    for key, expected in sorted(oracle.items()):
        row = published[key]
        assert row["trace_id"] == expected["trace_id"], key
        assert row["category"] == expected["category"], key
        assert row["active8_status"] == expected["active8_status"], key
        observed = [
            None
            if action["candidate_evidence"] is None
            else (
                action["candidate_evidence"]["raw_mark_count"],
                action["candidate_evidence"]["canonical_successor_count"],
                action["candidate_evidence"]["matching_mark_count"],
                action["candidate_evidence"]["successor_alias_count"],
                action["candidate_evidence"]["action_sha256"],
                action["candidate_evidence"]["target_state_sha256"],
            )
            for action in row["actions"]
        ]
        assert observed == expected["marks"], key

    # Measured on this payload, so the comparison above is not over a degenerate
    # census: all three categories are populated, and the alias multiplicity is
    # genuinely greater than one somewhere.
    counts = completion["active8_counts"]
    assert counts["upstream_rejected_entries"] > 0
    assert counts["active8_accepted_entries"] > 0
    assert counts["active8_excluded_entries"] > 0
    assert counts["successor_alias_multiplicity"] > counts["matching_candidate_marks"]


def test_the_census_reconciles_exactly_as_the_seam_declares(chain: Chain) -> None:
    """Without this the four categories could drift apart without notice."""

    _plan, completion = run_active8(chain, prefix="/artifacts/active8_census")
    counts = completion["active8_counts"]
    assert set(counts) == set(COUNT_FIELDS)
    assert set(ACTIVE8_CENSUS_FIELDS) <= set(counts)
    assert counts["source_entries"] == (
        counts["upstream_rejected_entries"]
        + counts["active8_accepted_entries"]
        + counts["active8_excluded_entries"]
    )
    # The structural projection a smaller consumer reads must reconcile too.
    structural = completion["structural_counts"]
    assert structural["source_entries"] == (
        structural["admitted_entries"] + structural["rejected_entries"]
    )
    assert structural["admitted_entries"] == counts["active8_accepted_entries"]
    # Twenty cells of four traces, every one accounted for exactly once.
    assert counts["source_entries"] == 20 * len(_CELL)
    assert counts["upstream_rejected_entries"] == 20
    assert counts["active8_excluded_entries"] == 20
    assert counts["active8_accepted_entries"] == 40

    # The two categories stay distinct in the published reason histograms.
    assert set(completion["upstream_rejection_reason_histogram"]) == {
        "atom_delete_outside_process_v2_mask"
    }
    assert set(completion["active8_exclusion_reason_histogram"]) == {
        "teacher_mark_absent_from_process_v2_marked_law"
    }
    assert completion["training_authorized"] is False
    assert completion["gate_zero_authorized"] is False


# ---- Acceptance test 8: determinism under concurrency and order ---------------


def _published_bytes(chain: Chain, plan: dict[str, Any]) -> dict[str, bytes]:
    run_root = mounted(chain, plan["run_artifact_root"])
    return {
        str(path.relative_to(run_root)): path.read_bytes()
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
    }


def test_concurrency_and_completion_order_reduce_byte_identically(chain: Chain) -> None:
    """Acceptance 8. Without this the reduction could depend on the schedule.

    ONE plan is executed four times -- once in plan order, then under container
    bounds 1, 20 and 40 with the completion order shuffled by a seeded
    permutation each time -- and every published byte is compared: the plan, the
    receipts, the decision shards and the completion.  That is deliberately
    stronger than comparing the completion, which a schedule-dependent receipt
    would survive.

    The plan is held fixed rather than re-planned per bound because the run
    identity content-addresses the output prefix, so two prefixes produce two
    legitimately different sets of file names and a byte comparison across them
    could only compare content-free digests.  The run root is removed between
    schedules so each one publishes from nothing.
    """

    prefix = "/artifacts/active8_determinism"
    plan = active8_plan(chain, prefix=prefix)
    run_root = mounted(chain, plan["run_artifact_root"])

    def publish_under(*, order: list[int] | None, max_containers: int | None) -> dict[str, bytes]:
        if run_root.exists():
            shutil.rmtree(run_root)
        write_process_v2_active8_plan(
            plan, artifact_root=chain.artifact_root, repo_root=ROOT
        )
        if max_containers is not None:
            wave_sizes: list[int] = []

            def submit(wave: tuple[str, ...]) -> list[str]:
                wave_sizes.append(len(wave))
                return list(wave)

            submitted = run_process_v2_active8_map(
                plan,
                submit=submit,
                max_map_containers=max_containers,
                repo_root=ROOT,
            )
            assert max(wave_sizes) <= max_containers
            assert sorted(submitted) == sorted(
                task["task_identity_sha256"] for task in plan["tasks"]
            )
        execute_active8(chain, plan, order=order)
        reduce_process_v2_active8_decisions(
            plan, artifact_root=chain.artifact_root, repo_root=ROOT
        )
        return _published_bytes(chain, plan)

    reference = publish_under(order=None, max_containers=None)
    assert len(reference) == 2 * len(plan["tasks"]) + 2, sorted(reference)

    for max_containers in (1, 20, 40):
        order = list(range(len(plan["tasks"])))
        random.Random(20260803 + max_containers).shuffle(order)
        assert order != sorted(order), "the shuffle must actually reorder"
        observed = publish_under(order=order, max_containers=max_containers)
        assert observed == reference, max_containers


def test_a_second_identical_plan_reduces_to_the_same_completion(chain: Chain) -> None:
    """Without this determinism is proven only across schedules, not across runs."""

    first = active8_plan(chain, prefix="/artifacts/active8_repeat")
    second = active8_plan(chain, prefix="/artifacts/active8_repeat")
    assert canonical_bytes(first) == canonical_bytes(second)


# ---- Acceptance test 9: everything incomplete fails closed --------------------


def test_a_missing_task_publishes_no_completion(chain: Chain) -> None:
    """Acceptance 9, missing. Without this a partial run could be sealed."""

    plan = active8_plan(chain, prefix="/artifacts/active8_missing")
    write_process_v2_active8_plan(plan, artifact_root=chain.artifact_root, repo_root=ROOT)
    execute_active8(chain, plan, order=list(range(1, len(plan["tasks"]))))
    with pytest.raises(ProcessV2Active8Incomplete):
        reduce_process_v2_active8_decisions(
            plan, artifact_root=chain.artifact_root, repo_root=ROOT
        )
    assert not (mounted(chain, plan["run_artifact_root"]) / COMPLETION_FILENAME).exists()


def test_an_extra_task_object_publishes_no_completion(chain: Chain) -> None:
    """Acceptance 9, extra. Without this an unplanned result could be counted."""

    plan, _completion = run_active8(chain, prefix="/artifacts/active8_extra")
    task_root = mounted(chain, plan["run_artifact_root"]) / TASK_DIRNAME
    first = plan["tasks"][0]["task_identity_sha256"]
    shutil.copytree(task_root / first, task_root / ("f" * 64))
    with pytest.raises(ProcessV2Active8MapReduceError, match="unexpected objects"):
        completed_process_v2_active8_task_ids(
            plan, artifact_root=chain.artifact_root, repo_root=ROOT
        )


def test_a_duplicated_decision_row_publishes_no_completion(chain: Chain) -> None:
    """Acceptance 9, duplicate. Without this one entry could be decided twice."""

    plan = active8_plan(chain, prefix="/artifacts/active8_duplicate")
    write_process_v2_active8_plan(plan, artifact_root=chain.artifact_root, repo_root=ROOT)
    execute_active8(chain, plan)
    output = mounted(chain, plan["tasks"][0]["output_artifact_path"])
    raw = gzip.decompress((output / DECISION_FILENAME).read_bytes())
    duplicated = raw + raw.splitlines(keepends=True)[0]
    (output / DECISION_FILENAME).write_bytes(_gzip(duplicated))
    with pytest.raises(ProcessV2Active8MapReduceError):
        reduce_process_v2_active8_decisions(
            plan, artifact_root=chain.artifact_root, repo_root=ROOT
        )
    assert not (mounted(chain, plan["run_artifact_root"]) / COMPLETION_FILENAME).exists()


def test_a_corrupted_decision_shard_publishes_no_completion(chain: Chain) -> None:
    """Acceptance 9, corrupted. Without this a damaged shard could still reduce."""

    plan = active8_plan(chain, prefix="/artifacts/active8_corrupt")
    write_process_v2_active8_plan(plan, artifact_root=chain.artifact_root, repo_root=ROOT)
    execute_active8(chain, plan)
    output = mounted(chain, plan["tasks"][0]["output_artifact_path"])
    (output / DECISION_FILENAME).write_bytes(b"not gzip at all")
    with pytest.raises(ProcessV2Active8MapReduceError):
        reduce_process_v2_active8_decisions(
            plan, artifact_root=chain.artifact_root, repo_root=ROOT
        )
    assert not (mounted(chain, plan["run_artifact_root"]) / COMPLETION_FILENAME).exists()


def test_a_partially_published_task_is_not_a_result_and_does_not_block_a_retry(
    chain: Chain,
) -> None:
    """Acceptance 9, partial. Without this half a task could count as a task.

    A task publishes by directory rename, so the only thing a hard termination
    can leave behind is the hidden staging sibling.  It must be invisible to the
    completion scan and must not collide with the retry that follows it.
    """

    plan = active8_plan(chain, prefix="/artifacts/active8_partial")
    write_process_v2_active8_plan(plan, artifact_root=chain.artifact_root, repo_root=ROOT)
    execute_active8(chain, plan)
    task = plan["tasks"][0]
    output = mounted(chain, task["output_artifact_path"])
    staging = output.parent / f".{output.name}.abandoned.staging"
    shutil.copytree(output, staging)
    (staging / RECEIPT_FILENAME).unlink()
    shutil.rmtree(output)

    assert task["task_identity_sha256"] not in completed_process_v2_active8_task_ids(
        plan, artifact_root=chain.artifact_root, repo_root=ROOT
    )
    with pytest.raises(ProcessV2Active8Incomplete):
        reduce_process_v2_active8_decisions(
            plan, artifact_root=chain.artifact_root, repo_root=ROOT
        )

    execute_active8(chain, plan, order=[0])
    assert task["task_identity_sha256"] in completed_process_v2_active8_task_ids(
        plan, artifact_root=chain.artifact_root, repo_root=ROOT
    )


def test_a_mismatched_receipt_publishes_no_completion(chain: Chain) -> None:
    """Acceptance 9, mismatched. Without this a resealed receipt could pass.

    The receipt is resealed correctly around a moved count, so it is internally
    self-consistent and only disagrees with the decision rows it summarizes.
    That is the case a self-hash check alone cannot catch, which is why the
    validator recomputes the census from the rows.
    """

    plan = active8_plan(chain, prefix="/artifacts/active8_mismatch")
    write_process_v2_active8_plan(plan, artifact_root=chain.artifact_root, repo_root=ROOT)
    execute_active8(chain, plan)
    output = mounted(chain, plan["tasks"][0]["output_artifact_path"])
    receipt = json.loads((output / RECEIPT_FILENAME).read_bytes())
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    body["counts"] = {
        **body["counts"],
        "raw_candidate_marks": int(body["counts"]["raw_candidate_marks"]) + 1,
    }
    resealed = {**body, "receipt_sha256": canonical_sha256(body)}
    (output / RECEIPT_FILENAME).write_bytes(canonical_bytes(resealed) + b"\n")
    with pytest.raises(ProcessV2Active8MapReduceError, match="census disagrees"):
        reduce_process_v2_active8_decisions(
            plan, artifact_root=chain.artifact_root, repo_root=ROOT
        )
    assert not (mounted(chain, plan["run_artifact_root"]) / COMPLETION_FILENAME).exists()


def test_the_reduction_requires_the_exact_published_plan_bytes(chain: Chain) -> None:
    """Without this a run could reduce against a plan nobody published."""

    plan = active8_plan(chain, prefix="/artifacts/active8_planless")
    execute_active8(chain, plan)
    with pytest.raises(ProcessV2Active8MapReduceError, match="published plan bytes"):
        reduce_process_v2_active8_decisions(
            plan, artifact_root=chain.artifact_root, repo_root=ROOT
        )


def test_a_completed_task_is_reused_exactly_rather_than_recomputed(chain: Chain) -> None:
    """Without this a restart could publish a second, differing result."""

    plan = active8_plan(chain, prefix="/artifacts/active8_reuse")
    write_process_v2_active8_plan(plan, artifact_root=chain.artifact_root, repo_root=ROOT)
    identity = plan["tasks"][0]["task_identity_sha256"]
    first = execute_process_v2_active8_decision_task(
        plan,
        identity,
        artifact_root=chain.artifact_root,
        repo_root=ROOT,
        model=chain.runtime.model,
        candidate_checker=chain.runtime.checker,
        model_runtime_identity_resolver=runtime_identity_resolver(chain.runtime),
    )
    second = execute_process_v2_active8_decision_task(
        plan,
        identity,
        artifact_root=chain.artifact_root,
        repo_root=ROOT,
        model=chain.runtime.model,
        candidate_checker=chain.runtime.checker,
        model_runtime_identity_resolver=runtime_identity_resolver(chain.runtime),
    )
    assert first["reused"] is False
    assert second["reused"] is True
    assert {k: v for k, v in first.items() if k != "reused"} == {
        k: v for k, v in second.items() if k != "reused"
    }


# ---- The bindings the plan refuses --------------------------------------------


def test_a_plan_cannot_bind_a_rebind_of_another_cache_generation(chain: Chain) -> None:
    """Without this Active8 could decide rows a different rebind never saw."""

    other_cache = plan_process_v2_chunk_cache(
        binder_fixture.bind(
            chain.payload, binder_fixture.expectation_for(_read_completion(chain))
        ),
        repo_root=ROOT,
        output_artifact_prefix="/artifacts/active8_other_cache",
        records_per_chunk=RECORDS_PER_CHUNK + 1,
    )
    write_process_v2_chunk_cache_plan(other_cache, artifact_root=chain.artifact_root)
    for task in other_cache["tasks"]:
        execute_process_v2_chunk_cache_task(
            other_cache, task["task_identity_sha256"], artifact_root=chain.artifact_root
        )
    reduce_process_v2_chunk_cache(other_cache, artifact_root=chain.artifact_root)
    with pytest.raises(ProcessV2Active8MapReduceError, match="different chunk-cache"):
        plan_process_v2_active8_decisions(
            cache_run_artifact_root=str(other_cache["run_artifact_root"]),
            rebind_plan=chain.rebind_plan,
            rebind_completion=chain.rebind_completion,
            admitted_source_identity=chain.admitted_source_identity,
            model_runtime_identity=chain.runtime.identity,
            artifact_root=chain.artifact_root,
            repo_root=ROOT,
            output_artifact_prefix="/artifacts/active8_wrong_cache",
        )


def test_a_plan_cannot_bind_an_admitted_source_of_another_run(chain: Chain) -> None:
    """Without this the plan's provenance could name a corpus it never read."""

    foreign = {
        **chain.admitted_source_identity,
        "run_identity_sha256": "0" * 64,
    }
    with pytest.raises(ProcessV2Active8MapReduceError, match="another Process-V2 rebind run"):
        plan_process_v2_active8_decisions(
            cache_run_artifact_root=str(chain.cache_plan["run_artifact_root"]),
            rebind_plan=chain.rebind_plan,
            rebind_completion=chain.rebind_completion,
            admitted_source_identity=foreign,
            model_runtime_identity=chain.runtime.identity,
            artifact_root=chain.artifact_root,
            repo_root=ROOT,
            output_artifact_prefix="/artifacts/active8_foreign_source",
        )


def test_the_plan_binds_the_v2_identity_and_keeps_the_v1_one_distinct(
    chain: Chain,
) -> None:
    """Without this the superseded payload identity could be read as current."""

    plan = active8_plan(chain, prefix="/artifacts/active8_identities")
    live_v2 = str(chain.rebind_plan["process_v2_identity"]["process_identity_sha256"])
    pinned_v1 = str(chain.rebind_plan["pinned_process_identity"]["process_identity_sha256"])
    assert plan["process_v2_identity"]["process_identity_sha256"] == live_v2
    assert plan["pinned_process_identity"]["process_identity_sha256"] == pinned_v1
    assert live_v2 != pinned_v1
    assert plan["policy"]["process_identity_sha256"] == live_v2
    for task in plan["tasks"]:
        assert task["pinned_process_identity_sha256"] == pinned_v1


def test_every_published_row_exposes_the_whole_trace_key(chain: Chain) -> None:
    """The seam's correction, enforced on the production rows.

    A consumer addresses one trace by ``(v1_task_identity_sha256, entry_index,
    trace_id)``.  If a published row omitted any of the three, the consumer
    would have to reconstruct the key from somewhere else, which is where a
    bare-id lookup gets reintroduced.
    """

    plan, _completion = run_active8(chain, prefix="/artifacts/active8_keys")
    keys: set[tuple[str, int, str]] = set()
    for row in iter_process_v2_active8_decision_rows(
        plan, artifact_root=chain.artifact_root, repo_root=ROOT
    ):
        assert set(TRACE_KEY_FIELDS) <= set(row)
        assert tuple(row[field] for field in JOIN_KEY_FIELDS) == (
            row["v1_task_identity_sha256"],
            row["entry_index"],
        )
        key = tuple(row[field] for field in TRACE_KEY_FIELDS)
        assert key not in keys
        keys.add(key)
    assert len(keys) == 20 * len(_CELL)
    # Measured, and recorded rather than assumed: this fixture's trace ids are
    # content-addressed, so they are globally distinct here and a bare-id lookup
    # would NOT collide on this payload. That is a property of the fixture, not
    # of the corpus -- ids are only guaranteed unique within a V1 task -- so the
    # collision itself is proven on a constructed pair in
    # ``test_editing_v2_process_v2_active8_index``. What is proven here is that
    # the production rows carry the whole key, which is what makes that lookup
    # possible at all.
    assert len({key[2] for key in keys}) == len(keys)


def test_the_two_rejection_categories_are_never_merged(chain: Chain) -> None:
    """Without this an unevaluated trace would look like a refused one."""

    plan, _completion = run_active8(chain, prefix="/artifacts/active8_categories")
    by_category: dict[str | None, list[dict[str, Any]]] = {}
    for row in iter_process_v2_active8_decision_rows(
        plan, artifact_root=chain.artifact_root, repo_root=ROOT
    ):
        by_category.setdefault(row["category"], []).append(row)
    assert set(by_category) == {None, UPSTREAM_REJECTED, ACTIVE8_EXCLUDED}
    for row in by_category[UPSTREAM_REJECTED]:
        assert row["active8_status"] == "not_evaluated"
        assert row["actions"] == []
        assert row["active8_exclusions"] == []
        assert row["upstream_rejection_code"]
    for row in by_category[ACTIVE8_EXCLUDED]:
        assert row["active8_status"] == "excluded"
        assert row["active8_exclusions"]
        assert row["upstream_rejection_code"] is None
        assert row["progress_rows"] == []
    for row in by_category[None]:
        assert row["active8_status"] == "accepted"
        assert len(row["progress_rows"]) == row["address"]["path_length"] + 1


# ---- Helpers ------------------------------------------------------------------


def _gzip(raw: bytes) -> bytes:
    import io

    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


def _read_completion(chain: Chain) -> dict[str, Any]:
    from compose_v4.data.semantic_trace_migration_mapreduce import (
        COMPLETION_FILENAME as MIGRATION_COMPLETION_FILENAME,
    )

    return json.loads(
        (chain.payload.payload_root / MIGRATION_COMPLETION_FILENAME).read_bytes()
    )


def test_the_active8_plan_and_completion_grant_nothing(chain: Chain) -> None:
    """Every authority field, at every level, published false."""

    plan, completion = run_active8(chain, prefix="/artifacts/active8_authority")
    for document in (plan, completion):
        for field in (
            "training_authorized",
            "gate_zero_authorized",
            "t1_authorized",
            "bounded_p50_authorized",
            "long_training_authorized",
            "checkpoint_selection_authorized",
            "final_test_selection_authorized",
        ):
            assert document[field] is False, field
    encoded = json.dumps([plan, completion])
    assert "p50_authorized\"" not in encoded.replace("bounded_p50_authorized\"", "")
    validate_process_v2_active8_plan(plan, repo_root=ROOT)
