"""The whole Process-V2 chain, run end to end, as one test.

WHY THIS TEST IS THE ONE THAT DECIDES
--------------------------------------
Three concerns were built in parallel against a frozen seam: the cache-fed
source, the Active8 decision stage and index, and Gate 0.  Each has its own
suite and each is green.  That proves every piece matches the contract that was
written for it.  It does not prove the contract matches reality, because the
contract and the three implementations share one author for the parts that
matter, and a seam that is wrong in the same way on both sides agrees with
itself.

So this file drives the production chain with nothing stubbed:

    cache construction -> rebind -> Active8 decisions -> decision index -> Gate 0

and asserts the properties that only exist once the pieces are joined.  Anything
provable inside one agent's boundary belongs in that agent's suite, not here.

WHAT IT ASSERTS THAT NO PER-STAGE SUITE CAN
-------------------------------------------
* the chain reaches a sealed Gate-0 decision at all;
* no raw packed shard is opened after the cache is published, measured with a
  positive control so the instrument is shown able to see the read whose
  absence is claimed;
* Gate 0 decodes each ELIGIBLE cache chunk once and its decode count does not
  scale with traces or with transitions -- the property whose absence was the
  original defect, and the only form of the bound that is not vacuous;
* no sealed-role chunk is decoded at all, so held-out molecular states are never
  opened rather than opened and discarded;
* ``terminal`` on a structural assignment describes the teacher's SOURCE
  progress position, proven on a real one-step transition whose successor IS
  terminal.

THE BOUND THIS FILE USED TO CARRY WAS VACUOUS TWICE OVER
---------------------------------------------------------
It asserted ``decodes <= accepted`` where ``accepted`` counted all 40 accepted
traces while only 10 are decision-eligible, so it carried a free 4x before it
bound anything; and it was measured on a fixture whose chunks hold 2 records
against a production 2048, so it could show a per-trace loop but never what one
decode costs.  The assertions below use the SELECTED inventory -- the eligible
tasks and the eligible traces -- and pin the decode count to the chunk count.
"""

from __future__ import annotations

import copy
import gzip
import inspect
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import test_editing_v2_process_v2_active8_decisions as stage
from compose_v4.data.editing_v2_process_v2_active8_decision_index import (
    ProcessV2Active8DecisionIndex,
    resolve_process_v2_active8_decision_index,
)
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_CENSUS_FIELDS,
    ProcessV2Active8Index,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    StructuralDecisionIndex,
)
from compose_v4.experiments import editing_v2_process_v2_gate_zero as gate_zero
from compose_v4.experiments.editing_v2_process_v2_gate_zero import (
    COMPLETION_FILENAME,
    DECISION_FILENAME,
    EVIDENCE_FILENAME,
    ROLE_FILTER_PARAMETER,
    SUPPORT_EVIDENCE_FIELD,
    run_process_v2_gate_zero,
)

ROOT = stage.ROOT


# ---- The role filter Gate 0 requires, until the index publishes it ----


class _PendingRoleFilteredIndex(ProcessV2Active8DecisionIndex):
    """The role-filtered bulk stream, standing in for the index's own.

    TEMPORARY, AND ITS REMOVAL IS ENFORCED.  Gate 0 refuses any index whose bulk
    stream cannot be restricted to the decision-eligible partition roles BEFORE
    it opens a cache chunk, because filtering afterwards means the held-out
    molecular states were decoded and then discarded.  The index is gaining that
    filter in a separate workstream; until it lands, this subclass supplies it so
    the join can be measured, and
    ``test_the_role_filtered_stream_is_still_a_local_stand_in`` FAILS the moment
    the real one exists, which is when this class must be deleted.

    It reimplements nothing: it selects tasks by their declared ``split`` and
    delegates to the production stream, which is the whole change the index
    needs.  The selection is made on a shallow copy so the shared module-scoped
    index is never mutated.
    """

    def iter_accepted_transitions(self, *, partition_roles: tuple[str, ...] | None = None):
        if partition_roles is None:
            yield from super().iter_accepted_transitions()
            return
        selected = [
            task for task in self._plan["tasks"] if str(task["split"]) in partition_roles
        ]
        filtered = copy.copy(self)
        filtered._plan = {**self._plan, "tasks": selected}
        yield from ProcessV2Active8DecisionIndex.iter_accepted_transitions(filtered)


def test_the_role_filtered_stream_is_still_a_local_stand_in() -> None:
    """The ratchet that makes the stand-in above impossible to leave behind."""

    parameters = inspect.signature(
        ProcessV2Active8DecisionIndex.iter_accepted_transitions
    ).parameters
    assert ROLE_FILTER_PARAMETER not in parameters, (
        "the decision index now publishes its own role-filtered bulk stream: delete "
        "_PendingRoleFilteredIndex and drive Gate 0 with the real index directly"
    )


@pytest.fixture(autouse=True)
def _require_the_production_admission_authority() -> None:
    """Reuse the stage module's own refusal, not a copy of it.

    That module refuses to run against the rebind fixture's stand-in atom-delete
    mask. An end-to-end test that quietly ran against a stand-in would be the
    worst possible green: the chain would join perfectly around a mask the
    production runtime does not use.
    """

    stage._require_the_production_admission_authority.__wrapped__()


@pytest.fixture(name="chain", scope="module")
def _chain(tmp_path_factory: pytest.TempPathFactory) -> stage.Chain:
    """Built exactly as the stage module builds it, extra traces and all."""

    monkeypatch = pytest.MonkeyPatch()
    for name, definition in {
        **stage.binder_fixture._EXTRA_TRACES,
        **stage._EXTRA_TRACES,
    }.items():
        monkeypatch.setitem(stage.v1_fixture._FIXTURE_TRACES, name, definition)
    try:
        return stage.build_chain(tmp_path_factory.mktemp("e2e"))
    finally:
        monkeypatch.undo()


@pytest.fixture(name="index", scope="module")
def _index(chain: stage.Chain) -> Any:
    """The real index, reopened from published bytes rather than handed over."""

    plan, _completion = stage.run_active8(chain, prefix="/artifacts/e2e_decisions")
    root = stage.mounted(chain, plan["run_artifact_root"])
    index = resolve_process_v2_active8_decision_index(
        plan_path=root / "PROCESS_V2_ACTIVE8_PLAN.json",
        completion_path=root / "PROCESS_V2_ACTIVE8_COMPLETE.json",
        artifact_root=chain.artifact_root,
        repo_root=ROOT,
    )
    # Everything about the chain stays real; only the role filter is supplied.
    index.__class__ = _PendingRoleFilteredIndex
    return index


@pytest.fixture(name="contract", scope="module")
def _contract() -> gate_zero.FrozenProcessV2GateZeroContract:
    return gate_zero.load_process_v2_gate_zero_contract(repo_root=ROOT)


def _gate_zero(chain: stage.Chain, index: Any, name: str) -> dict[str, Any]:
    output = chain.artifact_root / name
    return run_process_v2_gate_zero(
        index,
        repo_root=ROOT,
        artifact_root=chain.artifact_root,
        output_directory=output,
    )


def _eligible_inventory(index: Any, contract: Any) -> dict[str, Any]:
    """The SELECTED cache inventory, from the index's own resolved metadata.

    Decision metadata is read for every role -- that is what keeps the census and
    the sealed inventories complete -- so the eligible task set can be counted
    without opening anything.
    """

    roles = set(contract.decision_roles)
    eligible_tasks: set[str] = set()
    tasks_with_traces: set[str] = set()
    traces = 0
    transitions = 0
    sealed_tasks: set[str] = set()
    for row in index.iter_resolved_traces():
        if row["partition_role"] in roles:
            eligible_tasks.add(row["task_identity_sha256"])
            if row["rejection_category"] is None:
                tasks_with_traces.add(row["task_identity_sha256"])
                traces += 1
                transitions += row["accepted_transition_count"]
        else:
            sealed_tasks.add(row["task_identity_sha256"])
    return {
        "eligible_tasks": len(eligible_tasks),
        "chunks_with_eligible_traces": len(tasks_with_traces),
        "eligible_traces": traces,
        "eligible_transitions": transitions,
        "sealed_tasks": len(sealed_tasks),
    }


def _record_decodes(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every production chunk read, with the target it was asked for."""

    import compose_v4.data.editing_v2_process_v2_active8_decision_index as index_module

    decodes: list[Any] = []
    real_reader = index_module.read_process_v2_chunk_target

    def counting_reader(*args, **kwargs):  # type: ignore[no-untyped-def]
        decodes.append(kwargs.get("target"))
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(index_module, "read_process_v2_chunk_target", counting_reader)
    return decodes


# ---- The chain reaches a sealed decision ----


def test_the_chain_runs_from_cache_construction_to_a_sealed_gate_zero_decision(
    chain: stage.Chain, index: Any
) -> None:
    """Acceptance 1. The seam is only proven by the pieces joining."""

    # The index the parallel work was built against, satisfied by the real thing.
    assert isinstance(index, StructuralDecisionIndex)
    assert isinstance(index, ProcessV2Active8Index)
    assert "V1" not in [cls.__name__ for cls in type(index).__mro__]

    result = _gate_zero(chain, index, "e2e_gate_zero")
    decision = result["decision"]
    counts = result["evidence"]["counts"]

    # A result alone proves nothing: Gate 0 published FAIL while skipping every
    # transition, because the index was handed a transformed view it could not
    # recognise, refused all of them, and the loop counted each refusal as a
    # classification failure and continued. The decision was still in
    # {PASS, FAIL}, so asserting only that is satisfied by a gate doing nothing.
    assert counts["structural_assignments"] > 0, "Gate 0 assigned nothing"
    assert counts["observed_capability_cells"] > 0, "Gate 0 observed no cell"
    assert counts["classification_failures"] == 0, (
        f"{counts['classification_failures']} transitions were refused; a chain "
        "whose own index rejects its own transitions is not a working chain"
    )
    assert decision["structural_result"] in {"PASS", "FAIL"}

    output = chain.artifact_root / "e2e_gate_zero"
    for filename in (EVIDENCE_FILENAME, DECISION_FILENAME, COMPLETION_FILENAME):
        assert (output / filename).is_file(), filename

    # A completed gate, either way, authorizes nothing.
    for field in AUTHORITY_FIELDS:
        assert decision[field] is False, field
        assert result["completion"][field] is False, field


def test_the_gate_zero_census_is_the_index_census(chain: stage.Chain, index: Any) -> None:
    """The two stages must be describing one corpus, not two.

    A Gate 0 that recounted the corpus itself could disagree with the index and
    still publish, and the disagreement would look like a chemistry finding.
    """

    evidence = _gate_zero(chain, index, "e2e_census")["evidence"]
    published = index.counts()
    for field in ACTIVE8_CENSUS_FIELDS:
        assert field in published, field
    source = published["source_entries"]
    assert source == (
        published["upstream_rejected_entries"]
        + published["active8_accepted_entries"]
        + published["active8_excluded_entries"]
    )
    # Gate 0 names its own census in its own vocabulary; what must agree is the
    # corpus size, not the spelling.
    assert evidence["counts"]["resolved_traces"] == source
    assert (
        evidence["counts"]["upstream_rejected_traces"]
        == published["upstream_rejected_entries"]
    )
    # Sealed roles are hashed without being decoded, so their inventories must
    # still be present and distinct -- "not decoded" is not "not accounted for".
    inventories = evidence["sealed_nondecision_role_inventory_sha256"]
    assert set(inventories) == set(evidence["sealed_nondecision_partition_roles"])
    assert len(set(inventories.values())) == len(inventories)


# ---- No raw source after cache publication ----


def test_no_raw_packed_shard_is_opened_after_the_cache_is_published(
    chain: stage.Chain, index: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 1's second half, and the reason the cache exists.

    Instrumented with a positive control: the same recorder is shown to SEE a
    raw read when one genuinely happens, so the absence asserted below is a
    measurement rather than an instrument that was never wired up.
    """

    opened: list[str] = []
    real_gzip_open = gzip.open

    def recording_gzip_open(filename, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(Path(str(filename)).name)
        return real_gzip_open(filename, *args, **kwargs)

    monkeypatch.setattr(gzip, "open", recording_gzip_open)
    _gate_zero(chain, index, "e2e_no_raw")

    assert opened, "the recorder saw nothing at all, so it proves nothing"
    assert "traces.jsonl.gz" not in opened

    # Positive control: the raw shard IS reachable, and this recorder sees it.
    raw = sorted(chain.artifact_root.rglob("traces.jsonl.gz"))
    assert raw, "the fixture publishes no raw shard, so the assertion above is vacuous"
    with recording_gzip_open(raw[0], "rb") as handle:
        handle.read(1)
    assert "traces.jsonl.gz" in opened


# ---- The join must not reinstate the triangular rescan ----


def test_gate_zero_decodes_each_eligible_chunk_once_and_no_sealed_chunk(
    chain: stage.Chain, index: Any, contract: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect neither module can see alone, bounded against the right numbers.

    The previous bound compared the decode count to all 40 ACCEPTED traces while
    only 10 are decision-eligible, so it permitted four decodes per eligible
    trace before failing; and its fixture holds 2 records per chunk against a
    production 2048, so it could never show what a decode costs.  What is
    asserted here instead is the shape: the decode count equals the number of
    eligible chunks and is strictly below both the trace count and the transition
    count, which is what "does not scale with traces or transitions" means when
    only one corpus size is available.
    """

    inventory = _eligible_inventory(index, contract)
    decodes = _record_decodes(monkeypatch)
    _gate_zero(chain, index, "e2e_decode_count")

    print(
        "\n  gate-zero chunk decodes="
        f"{len(decodes)} eligible_tasks={inventory['eligible_tasks']} "
        f"chunks_with_eligible_traces={inventory['chunks_with_eligible_traces']} "
        f"eligible_traces={inventory['eligible_traces']} "
        f"eligible_transitions={inventory['eligible_transitions']} "
        f"sealed_tasks={inventory['sealed_tasks']}"
    )

    # 1. The selected inventory is non-empty, or every bound below is vacuous.
    assert inventory["eligible_tasks"] > 0
    assert inventory["chunks_with_eligible_traces"] > 1, (
        "one eligible chunk cannot distinguish a per-chunk decode from a per-trace "
        "one; the fixture must carry several"
    )
    assert inventory["eligible_traces"] > 0
    # 2. The instrument must be able to see a decode at all.
    assert decodes, "no chunk decode was observed, so this bound proves nothing"

    # 3. Decodes are bounded by the ACTUAL selected cache-chunk inventory, and
    #    reach exactly the chunks that hold an eligible trace.
    assert len(decodes) <= inventory["eligible_tasks"]
    assert len(decodes) == inventory["chunks_with_eligible_traces"]
    distinct = {(target.split, target.data_lane, target.chunk_index) for target in decodes}
    assert len(distinct) == len(decodes), "a chunk was decoded more than once"

    # 4. The load-bearing one: the decode count does not scale with traces or
    #    with transitions. Both strictly exceed it, so a per-trace or
    #    per-transition loop could not produce this number.
    assert inventory["eligible_traces"] > len(decodes)
    assert inventory["eligible_transitions"] > len(decodes)

    # 5. Sealed roles have zero reads. Not "were not counted" -- not opened.
    sealed = set(contract.sealed_roles)
    read_by_role = Counter(target.split for target in decodes)
    assert not (sealed & set(read_by_role)), (
        f"held-out molecular states were decoded: {dict(read_by_role)}"
    )
    assert set(read_by_role) <= set(contract.decision_roles)


def test_gate_zero_refuses_an_index_whose_bulk_stream_cannot_be_role_filtered(
    chain: stage.Chain, index: Any, contract: Any
) -> None:
    """The refusal that keeps "never opened" from decaying into "never counted".

    The negative control is the REAL index class as it stands today: its bulk
    stream takes no role filter, so it can only be restricted after the held-out
    chunks are decoded.  Gate 0 refuses it, and refuses it before consuming
    anything -- which is why this is checked by signature and not by name.
    """

    unfiltered = copy.copy(index)
    unfiltered.__class__ = ProcessV2Active8DecisionIndex
    assert isinstance(unfiltered, ProcessV2Active8Index)
    with pytest.raises(gate_zero.ProcessV2GateZeroError, match=ROLE_FILTER_PARAMETER):
        gate_zero.build_process_v2_gate_zero_evidence(unfiltered, contract=contract)


# ---- `terminal` is the SOURCE progress position ----


def _one_step_transition(index: Any, contract: Any) -> dict[str, Any]:
    """A real accepted transition that is the ONLY step of its trace.

    Its successor is therefore the terminal progress position, which is exactly
    the case that used to be published as a terminal assignment.
    """

    lengths = {
        (row["v1_task_identity_sha256"], row["entry_index"], row["trace_id"]): row["path_length"]
        for row in index.iter_resolved_traces()
        if row["partition_role"] in set(contract.decision_roles)
        and row["rejection_category"] is None
    }
    for published in index.iter_accepted_transitions(
        partition_roles=tuple(contract.decision_roles)
    ):
        key = (
            published["v1_task_identity_sha256"],
            published["entry_index"],
            published["trace_id"],
        )
        if lengths.get(key) == 1 and published["successor_is_terminal"] is True:
            return dict(published)
    raise AssertionError("the fixture holds no one-step decision-eligible trace")


def test_terminal_describes_the_source_position_of_a_real_final_transition(
    index: Any, contract: Any
) -> None:
    """The correction, on the transition that used to prove the bug."""

    published = _one_step_transition(index, contract)
    assert published["successor_is_terminal"] is True
    assert published["source_progress_index"] == 0
    assert published["successor_progress_index"] == 1

    view = gate_zero._structural_view(published, contract=contract)
    assert view["terminal"] is False, (
        "`terminal` is the teacher's SOURCE progress position; a published "
        "transition exists because its source has an outgoing step, so it is "
        "never terminal -- not even when its successor is"
    )
    assert view["progress_index"] == 0


def test_no_accepted_transition_of_the_real_chain_is_a_terminal_assignment(
    chain: stage.Chain, index: Any
) -> None:
    """The count that was equal to the number of accepted traces, and is now zero."""

    evidence = _gate_zero(chain, index, "e2e_terminal")["evidence"]
    assert evidence["counts"]["terminal_assignments"] == 0
    assert evidence["checks"]["terminal_assignment_count_is_zero"] is True
    # Non-vacuous: every accepted trace ends on a transition whose SUCCESSOR is
    # terminal, so before the correction this count equalled the trace count.
    assert evidence["counts"]["decision_eligible_traces"] > 0


# ---- `require_every_teacher_supported` ----


def test_every_production_teacher_carries_exact_successor_support_evidence(
    index: Any, contract: Any
) -> None:
    """The requirement is live on the production path, not skipped for want of a field."""

    seen = 0
    for published in index.iter_accepted_transitions(
        partition_roles=tuple(contract.decision_roles)
    ):
        view = gate_zero._structural_view(published, contract=contract)
        assert SUPPORT_EVIDENCE_FIELD in view
        assert view[SUPPORT_EVIDENCE_FIELD] >= 1
        assert "has_exact_successor_support" not in gate_zero._teacher_failures(
            view, contract=contract
        )
        seen += 1
    assert seen > 0, "no teacher was examined, so this proves nothing"


def test_a_teacher_without_exact_successor_support_cannot_receive_a_cell(
    index: Any, contract: Any
) -> None:
    """The contract declares `require_every_teacher_supported`; this is it firing.

    The frozen support predicate needs a matching mark that produces the
    teacher's EXACT successor state.  Nothing else Gate 0 reads implies it, so
    with the count zeroed the teacher must be reported as a requirement failure
    rather than silently receiving a capability cell.
    """

    def unsupported(transition, **kwargs):  # type: ignore[no-untyped-def]
        view = dict(gate_zero._structural_view(transition, **kwargs))
        view[SUPPORT_EVIDENCE_FIELD] = 0
        return view

    evidence = gate_zero.build_process_v2_gate_zero_evidence(
        index, contract=contract, view=unsupported
    )
    failures = evidence["teacher_requirement_failure_counts"]["has_exact_successor_support"]
    assert failures == evidence["counts"]["decision_eligible_transitions"] > 0
    assert evidence["checks"]["every_decision_eligible_teacher_has_exact_successor_support"] is (
        False
    )
    assert evidence["structural_result"] == "FAIL"
    receipts = [
        receipt
        for receipt in evidence["classification_failure_receipts"]
        if receipt["failure_type"] == "teacher_has_exact_successor_support"
    ]
    assert receipts and receipts[0]["capability_cell_id"] is not None


# ---- The evidence strata: one derivation, and a proven bin set ----


def test_the_frozen_strata_bins_tile_the_count_range_exactly_once(contract: Any) -> None:
    """The property that is real, and is not a transition's to prove."""

    for field, bins in contract.strata_bins.items():
        assert gate_zero.strata_bin_defects(bins) == (), field


@pytest.mark.parametrize(
    ("bins", "fragment"),
    [
        ((("a", 1, 3), ("b", 5, None)), "do not meet"),
        ((("a", 1, 6), ("b", 5, None)), "do not meet"),
        ((("a", 1, 3), ("b", 4, 9)), "unbinned"),
        ((("a", 2, None),), "does not start at 1"),
        ((("a", 1, 3), ("a", 4, None)), "repeats a stratum id"),
        ((("a", 1, None), ("b", 2, None)), "unbounded above"),
    ],
)
def test_a_bin_set_that_does_not_tile_the_count_range_is_refused(
    bins: tuple[Any, ...], fragment: str
) -> None:
    """Negative controls, so the proof above is not simply agreeing with itself."""

    defects = gate_zero.strata_bin_defects(bins)
    assert any(fragment in defect for defect in defects), defects
    declaration = {
        "bins": [
            {"id": identifier, "minimum": minimum, "maximum": maximum}
            for identifier, minimum, maximum in bins
        ]
    }
    with pytest.raises(gate_zero.ProcessV2GateZeroError, match="tile the count range"):
        gate_zero._bins(declaration, field="fixture")


def test_the_production_strata_are_derived_once_and_published_with_their_bins(
    chain: stage.Chain, index: Any, contract: Any
) -> None:
    """Gate 0 authors the strata; the evidence says so and carries the bins."""

    evidence = _gate_zero(chain, index, "e2e_strata")["evidence"]
    assert evidence["evidence_strata_authority"] == (
        "gate_zero_derives_each_stratum_once_from_index_verified_counts"
    )
    published = evidence["frozen_evidence_strata"]
    assert set(published) == set(contract.strata_bins)
    for field, bins in contract.strata_bins.items():
        assert published[field] == [
            {"id": identifier, "minimum": minimum, "maximum": maximum}
            for identifier, minimum, maximum in bins
        ]
    assert evidence["teacher_requirement_failure_counts"][
        "declares_the_frozen_evidence_strata"
    ] == 0


# ---- What one complete pass costs, measured rather than projected ----


def test_report_the_complete_gate_zero_path_cost(
    chain: stage.Chain, index: Any, contract: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One complete Gate-0 pass, instrumented.  Reports; it does not assert a time.

    WHAT THIS COVERS, EXACTLY: the resolved-trace scan over every role, the
    role-filtered chunk decode, the structural view (ActionV4 decode plus the
    capability-cell classification of each action against its exact source and
    successor states) and the aggregation.  It does NOT cover a production chunk:
    this fixture holds 2 records per chunk against a production 2048, and its
    molecules are fixture-sized.  It does NOT cover successor re-enumeration,
    because Gate 0 performs none -- Active8 owns candidate enumeration and Gate 0
    consumes its evidence.  A projection from these numbers to a full corpus is
    therefore a projection of Gate 0's OWN per-transition cost only, and this
    repository has a recorded history of component benchmarks over-predicting
    path throughput by an order of magnitude.
    """

    inventory = _eligible_inventory(index, contract)
    decodes = _record_decodes(monkeypatch)
    # The contract load is O(1) in the corpus -- it validates and rehashes the
    # chain -- so it is timed apart from the pass that scales with it.
    load_wall = time.perf_counter()
    gate_zero.load_process_v2_gate_zero_contract(repo_root=ROOT)
    load_wall = time.perf_counter() - load_wall

    tracemalloc.start()
    cpu = time.process_time()
    wall = time.perf_counter()
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, contract=contract)
    wall = time.perf_counter() - wall
    cpu = time.process_time() - cpu
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    counts = evidence["counts"]
    transitions = counts["decision_eligible_transitions"]
    print(
        "\n  gate-zero complete pass (corpus-scaling work only):"
        f"\n    resolved rows (all roles) = {counts['resolved_traces']}"
        f"\n    eligible traces           = {counts['decision_eligible_traces']}"
        f"\n    eligible transitions      = {transitions}"
        f"\n    chunk decodes             = {len(decodes)}"
        f" (eligible chunks {inventory['chunks_with_eligible_traces']})"
        f"\n    successor evaluations     = 0 (Gate 0 re-enumerates nothing)"
        f"\n    cpu  = {cpu:.4f} s  ({1000 * cpu / max(transitions, 1):.3f} ms/transition,"
        f" {1000 * cpu / max(counts['resolved_traces'], 1):.3f} ms/resolved row)"
        f"\n    wall = {wall:.4f} s"
        f"\n    peak python allocation = {peak / 1024:.1f} KiB"
        f"\n    contract load (O(1), excluded above) = {load_wall:.4f} s"
    )
    assert transitions > 0
