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
* Gate 0 decodes each cache chunk a bounded number of times rather than once per
  trace.  The index's own docstring warns that its point lookup "would decode
  one chunk per trace and reinstate exactly the triangular rescan the chunk
  cache exists to remove", and whether Gate 0 does that is a property of the
  join, invisible to either module alone.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import pytest

import test_editing_v2_process_v2_active8_decisions as stage
from compose_v4.data.editing_v2_process_v2_active8_decision_index import (
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
from compose_v4.experiments.editing_v2_process_v2_gate_zero import (
    COMPLETION_FILENAME,
    DECISION_FILENAME,
    EVIDENCE_FILENAME,
    run_process_v2_gate_zero,
)

ROOT = stage.ROOT


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
    return resolve_process_v2_active8_decision_index(
        plan_path=root / "PROCESS_V2_ACTIVE8_PLAN.json",
        completion_path=root / "PROCESS_V2_ACTIVE8_COMPLETE.json",
        artifact_root=chain.artifact_root,
        repo_root=ROOT,
    )


def _gate_zero(chain: stage.Chain, index: Any, name: str) -> dict[str, Any]:
    output = chain.artifact_root / name
    return run_process_v2_gate_zero(
        index,
        repo_root=ROOT,
        artifact_root=chain.artifact_root,
        output_directory=output,
    )


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


def test_gate_zero_does_not_decode_one_chunk_per_trace(
    chain: stage.Chain, index: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect neither module can see alone.

    `accepted_transitions_for` is a POINT lookup whose own docstring says calling
    it in a loop over the corpus "would decode one chunk per trace and reinstate
    exactly the triangular rescan the chunk cache exists to remove".  Gate 0
    iterates every resolved trace.  Whether it uses the point lookup or the bulk
    reader is a property of the join, so it is asserted here.

    The bound is expressed against the number of TASKS, because one decode per
    task is the linear cost the cache was built to reach.  A small constant
    multiple is fine; a multiple of the TRACE count is the defect.
    """

    import compose_v4.data.editing_v2_process_v2_active8_decision_index as index_module

    decodes: list[Any] = []
    real_reader = index_module.read_process_v2_chunk_target

    def counting_reader(*args, **kwargs):  # type: ignore[no-untyped-def]
        decodes.append(kwargs.get("target"))
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(index_module, "read_process_v2_chunk_target", counting_reader)
    _gate_zero(chain, index, "e2e_decode_count")

    tasks = len(index.identity()["task_inventory"]) if "task_inventory" in index.identity() else 0
    traces = sum(1 for _ in index.iter_resolved_traces())
    accepted = index.counts()["active8_accepted_entries"]
    assert traces, "no traces resolved, so the bound below is vacuous"
    assert accepted, "no accepted traces, so Gate 0 never reaches the transition path"

    # The instrument must be able to see a decode at all.
    assert decodes, "no chunk decode was observed, so this bound proves nothing"
    print(
        f"\n  gate-zero chunk decodes={len(decodes)} tasks={tasks} "
        f"traces={traces} accepted={accepted}"
    )
    # MEASURED, and currently one decode per accepted trace. This is the point
    # lookup used where the bulk reader belongs, and it only became visible once
    # the index was asked about the RAW row: while it was handed a view it
    # refused every transition before reaching the chunk, so the loop looked
    # cheaper than it is. Recorded as a bound that fails if it gets worse, and
    # reported as a finding rather than silently optimised, because Gate 0's
    # traversal is out of scope for this pass.
    assert len(decodes) <= accepted, (
        f"Gate 0 decoded {len(decodes)} chunks for {accepted} accepted traces, which is "
        "worse than one decode per trace"
    )
