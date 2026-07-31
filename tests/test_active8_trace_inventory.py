from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.chem.molecular_graph import (
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.active8_trace_inventory import (
    ACTIVE8_FAMILIES,
    Active8SourceShard,
    Active8TraceInventoryError,
    ExactCandidateEvidence,
    ProductionExactCandidateChecker,
    accepted_trace_keys,
    build_active8_trace_inventory,
    inventory_record_for_trace,
    implementation_identity,
    load_active8_trace_admission,
)
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
    read_addressed_packed_shard,
    write_packed_shard,
)
from compose_v4.experiments.factorized_successor_training import (
    rewrite_action_codec_sha256,
)
from compose_v4.rewrite.action_codec import encode_action
from compose_v4.experiments.production_successor_kernel import (
    FactorizedMarkedLaw,
    ScoredRewriteMark,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomDelete, AtomInsert
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import encode_state
from compose_v4.rewrite.tracelets import RingSystemDelete


def _addressed(steps: tuple[RewriteStep, ...]) -> AddressedPackedTrace:
    state = pad_molecular_graph(smiles_to_molecular_graph("CC"), 8)
    states = tuple(state for _ in range(len(steps) + 1))
    trace = RewriteTrace(
        source=state,
        target=state,
        steps=steps,
        metadata={"fixture": "active8-whole-trace"},
    )
    path = PackedTraceProgress(trace, states)
    address = PackedTraceAddress(
        packed_shard_content_sha256="1" * 64,
        packed_shard_name="fixture.jsonl.gz",
        entry_index=7,
        trace_id="trace-7",
        layer="corruption",
        partition="train",
        source_key="CC",
        target_key="CC",
        path_length=len(steps),
    )
    return AddressedPackedTrace(address=address, trace=trace, path=path)


def _accept_exact_teacher(
    addressed: AddressedPackedTrace,
    step_index: int,
) -> ExactCandidateEvidence:
    step = addressed.trace.steps[step_index]
    return ExactCandidateEvidence(
        supported=True,
        action_sha256=rewrite_action_codec_sha256(
            step.rule_name,
            step.action,
        ),
    )


def _real_delete_addressed() -> AddressedPackedTrace:
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 8)
    action = AtomDelete(1)
    target = de_novo_rewrite_system().apply(source, "atom_delete", action)
    trace = RewriteTrace(
        source=source,
        target=target,
        steps=(RewriteStep("atom_delete", action),),
        metadata={"fixture": "active8-production-checker"},
    )
    return AddressedPackedTrace(
        address=PackedTraceAddress(
            packed_shard_content_sha256="2" * 64,
            packed_shard_name="checker-fixture.jsonl.gz",
            entry_index=0,
            trace_id="checker-trace",
            layer="corruption",
            partition="validation",
            source_key=canonical_state_key(source),
            target_key=canonical_state_key(target),
            path_length=1,
        ),
        trace=trace,
        path=PackedTraceProgress(trace, (source, target)),
    )


def _scored_delete(action: AtomDelete, coordinate: int) -> ScoredRewriteMark:
    return ScoredRewriteMark(
        family_name="atom_delete",
        table_name="atom_delete",
        executor_rule_name="atom_delete",
        action=action,
        coordinate=(coordinate,),
        log_probability=-0.6931471805599453,
    )


def test_production_checker_executes_only_the_exact_teacher_mark(
    monkeypatch,
) -> None:
    addressed = _real_delete_addressed()
    source = addressed.path.state_at(0)
    target = addressed.path.state_at(1)
    marked_law = FactorizedMarkedLaw(
        source_key=canonical_state_key(source),
        marks=(
            _scored_delete(AtomDelete(0), 0),
            _scored_delete(AtomDelete(1), 1),
        ),
        total_hazard=1.0,
        family_log_probabilities=(0.0,),
        enabled_families=("atom_delete",),
    )
    monkeypatch.setattr(
        "compose_v4.data.active8_trace_inventory.enumerate_factorized_marked_law",
        lambda model, state, time: marked_law,
    )

    class ExactTeacherRuntime:
        def __init__(self) -> None:
            self.applied: list[tuple[str, object]] = []

        def apply(self, state, rule_name, action):
            self.applied.append((rule_name, action))
            if action != AtomDelete(1):
                raise AssertionError("an unrelated legal mark was executed")
            return target

    runtime = ExactTeacherRuntime()
    evidence = ProductionExactCandidateChecker(
        object(),
        cache_size=1,
        system=runtime,
    )(addressed, 0)

    assert evidence == ExactCandidateEvidence(
        supported=True,
        action_sha256=rewrite_action_codec_sha256(
            "atom_delete",
            AtomDelete(1),
        ),
    )
    assert runtime.applied == [("atom_delete", AtomDelete(1))]


def test_production_checker_rejects_an_exact_action_with_wrong_slot_output(
    monkeypatch,
) -> None:
    addressed = _real_delete_addressed()
    source = addressed.path.state_at(0)
    marked_law = FactorizedMarkedLaw(
        source_key=canonical_state_key(source),
        marks=(_scored_delete(AtomDelete(1), 1),),
        total_hazard=1.0,
        family_log_probabilities=(0.0,),
        enabled_families=("atom_delete",),
    )
    monkeypatch.setattr(
        "compose_v4.data.active8_trace_inventory.enumerate_factorized_marked_law",
        lambda model, state, time: marked_law,
    )

    class WrongSlotRuntime:
        @staticmethod
        def apply(state, rule_name, action):
            return de_novo_rewrite_system().apply(
                state,
                rule_name,
                AtomDelete(0),
            )

    evidence = ProductionExactCandidateChecker(
        object(),
        cache_size=1,
        system=WrongSlotRuntime(),
    )(addressed, 0)

    assert not evidence.supported
    assert evidence.reason == "teacher_not_in_exact_candidates"


def test_active8_contract_is_exact_and_ordered() -> None:
    assert ACTIVE8_FAMILIES == (
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    )


def test_implementation_identity_covers_transitive_candidate_and_executor_sources(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    baseline = implementation_identity(repo_root=repository_root)
    copied_root = tmp_path / "source-snapshot"
    for relative in baseline["sources"]:
        source = repository_root / relative
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    assert implementation_identity(repo_root=copied_root) == baseline

    semantic_sources = (
        "src/compose_v4/model/factorized_tracelet_rate_model.py",
        "src/compose_v4/rewrite/operators.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/factorized_fiber.py",
    )
    assert set(semantic_sources).issubset(baseline["sources"])
    for relative in semantic_sources:
        path = copied_root / relative
        original = path.read_bytes()
        path.write_bytes(original + b"\n# active8 lineage mutation probe\n")
        changed = implementation_identity(repo_root=copied_root)
        assert changed["implementation_sha256"] != baseline["implementation_sha256"]
        assert changed["sources"][relative] != baseline["sources"][relative]
        path.write_bytes(original)


def test_disallowed_middle_step_excludes_the_entire_trace() -> None:
    addressed = _addressed(
        (
            RewriteStep("atom_delete", AtomDelete(1)),
            RewriteStep(
                "ring_system_delete",
                RingSystemDelete(
                    system_atoms=(0, 1),
                    retained_system_atoms=(0,),
                    bond_deletions=(),
                    atom_deletions=(1,),
                    atom_payloads=(),
                    bond_reorders=(),
                    source_aromatic_edges=(),
                    aromatic_edges=(),
                    topology_class="fixture",
                ),
            ),
            RewriteStep("atom_delete", AtomDelete(0)),
        )
    )

    record = inventory_record_for_trace(
        addressed,
        exact_candidate_checker=_accept_exact_teacher,
    )

    assert record["decision"] == "excluded"
    assert record["progress_rows"] == []
    assert record["exclusions"] == [
        {
            "step_index": 1,
            "executor_rule": "ring_system_delete",
            "family": "ring_system_delete",
            "reason": "disallowed_family",
        }
    ]
    # The physical trace identity remains auditable even though no row survives.
    assert record["trace_key"] == {
        "packed_shard_content_sha256": "1" * 64,
        "entry_index": 7,
        "trace_id": "trace-7",
    }


def test_accepted_trace_retains_every_progress_state_and_terminal() -> None:
    addressed = _addressed(
        (
            RewriteStep("atom_delete", AtomDelete(1)),
            RewriteStep("atom_delete", AtomDelete(0)),
        )
    )
    record = inventory_record_for_trace(
        addressed,
        exact_candidate_checker=_accept_exact_teacher,
    )

    assert record["decision"] == "accepted"
    assert len(record["progress_rows"]) == 3
    assert [row["progress_index"] for row in record["progress_rows"]] == [0, 1, 2]
    assert [row["is_terminal"] for row in record["progress_rows"]] == [
        False,
        False,
        True,
    ]
    assert record["progress_rows"][-1]["teacher_family"] is None
    assert record["progress_rows"][-1]["teacher_action_sha256"] is None
    assert all(len(row["exact_state_sha256"]) == 64 for row in record["progress_rows"])


def test_exact_candidate_failure_excludes_trace_without_partial_rows() -> None:
    addressed = _addressed((RewriteStep("atom_delete", AtomDelete(1)),))

    def reject(
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> ExactCandidateEvidence:
        return ExactCandidateEvidence(
            supported=False,
            action_sha256=None,
            reason="teacher_not_in_exact_candidates",
            detail="fixture",
        )

    record = inventory_record_for_trace(
        addressed,
        exact_candidate_checker=reject,
    )
    assert record["decision"] == "excluded"
    assert record["progress_rows"] == []
    assert record["exclusions"][0]["reason"] == ("teacher_not_in_exact_candidates")


@pytest.mark.parametrize(
    ("step", "reason"),
    (
        (RewriteStep("mystery_operator", object()), "unknown_or_legacy_executor_rule"),
        (
            RewriteStep(
                "atom_insert",
                AtomInsert(
                    slot=3,
                    atom_type=1,
                    formal_charge=0,
                    implicit_h_count=0,
                    neighbors=((0, 1), (1, 1)),
                ),
            ),
            "unsupported_multi_neighbor_atom_insert",
        ),
    ),
)
def test_unknown_and_structurally_unsupported_steps_fail_closed(
    step: RewriteStep,
    reason: str,
) -> None:
    record = inventory_record_for_trace(
        _addressed((step,)),
        exact_candidate_checker=_accept_exact_teacher,
    )
    assert record["decision"] == "excluded"
    assert record["progress_rows"] == []
    assert record["exclusions"][0]["reason"] == reason


def test_candidate_evidence_cannot_name_another_action() -> None:
    addressed = _addressed((RewriteStep("atom_delete", AtomDelete(1)),))

    def wrong_identity(
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> ExactCandidateEvidence:
        return ExactCandidateEvidence(
            supported=True,
            action_sha256="f" * 64,
        )

    record = inventory_record_for_trace(
        addressed,
        exact_candidate_checker=wrong_identity,
    )
    assert record["decision"] == "excluded"
    assert record["progress_rows"] == []
    assert record["exclusions"][0]["reason"] == ("candidate_evidence_action_identity_mismatch")


def _packed_entry(
    trace_id: str,
    steps: tuple[RewriteStep, ...],
) -> dict:
    state = pad_molecular_graph(smiles_to_molecular_graph("CC"), 8)
    return {
        "trace": {
            "trace_id": trace_id,
            "layer": "corruption",
            "partition": "train",
            "source_key": "CC",
            "target_key": "CC",
            "path_length": len(steps),
            "steps": [{"action": encode_action(step.rule_name, step.action)} for step in steps],
            "metadata": {},
        },
        "states": [encode_state(state) for _ in range(len(steps) + 1)],
    }


def test_immutable_inventory_reader_exposes_only_whole_accepted_traces(
    tmp_path,
) -> None:
    allowed = (RewriteStep("atom_delete", AtomDelete(1)),)
    disallowed_middle = (
        RewriteStep("atom_delete", AtomDelete(1)),
        RewriteStep(
            "ring_system_delete",
            RingSystemDelete(
                system_atoms=(0, 1),
                retained_system_atoms=(0,),
                bond_deletions=(),
                atom_deletions=(1,),
                atom_payloads=(),
                bond_reorders=(),
                source_aromatic_edges=(),
                aromatic_edges=(),
                topology_class="fixture",
            ),
        ),
        RewriteStep("atom_delete", AtomDelete(0)),
    )
    shard = tmp_path / "packed" / "train" / "fixture.jsonl.gz"
    write_packed_shard(
        shard,
        [
            _packed_entry("accepted", allowed),
            _packed_entry("excluded-middle", disallowed_middle),
        ],
        provenance={"capability_hash": "active8-fixture"},
        deterministic_gzip=True,
    )
    source_manifest = {"artifact": "fixture-unified", "shards": 1}
    source_manifest_path = tmp_path / "UNIFIED_PACKED_MANIFEST.json"
    source_manifest_path.write_text(json.dumps(source_manifest, sort_keys=True))
    output = tmp_path / "inventory"
    manifest = build_active8_trace_inventory(
        (
            Active8SourceShard(
                manifest_layer="general_corruption",
                envelope_layer="corruption",
                partition="train",
                relative_path="train/fixture.jsonl.gz",
                path=shard,
            ),
        ),
        exact_candidate_checker=_accept_exact_teacher,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        support_contract_sha256="a" * 64,
        output_dir=output,
    )

    assert manifest["counts"]["traces"] == 2
    assert manifest["counts"]["accepted_traces"] == 1
    assert manifest["counts"]["excluded_traces"] == 1
    assert manifest["counts"]["accepted_progress_rows"] == 2
    keys = accepted_trace_keys(output / "ACTIVE8_TRACE_INVENTORY.json")
    assert len(keys) == 1
    only = next(iter(keys))
    assert only[1:] == (0, "accepted")

    manifest_path = output / "ACTIVE8_TRACE_INVENTORY.json"
    admission = load_active8_trace_admission(
        manifest_path,
        expected_manifest_file_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        expected_inventory_sha256=manifest["inventory_sha256"],
        expected_effective_source_corpus_cache_sha256=manifest["source_identity"][
            "effective_source_corpus_cache_sha256"
        ],
    )
    addressed = tuple(read_addressed_packed_shard(shard))
    assert admission.is_accepted(addressed[0].address)
    assert not admission.is_accepted(addressed[1].address)
    admission.assert_complete_source_shard(
        packed_shard_name=shard.name,
        layer="corruption",
        partition="train",
        observed_digest=addressed[0].address.packed_shard_content_sha256,
        observed_entries=2,
    )
    admission.assert_partition_shards(
        "train",
        {("corruption", "train", shard.name)},
    )
    with pytest.raises(
        Active8TraceInventoryError,
        match="trace ID disagrees",
    ):
        admission.is_accepted(
            SimpleNamespace(
                packed_shard_content_sha256=(addressed[0].address.packed_shard_content_sha256),
                entry_index=0,
                trace_id="substituted-trace",
            )
        )
    with pytest.raises(
        Active8TraceInventoryError,
        match="file SHA-256 mismatch",
    ):
        load_active8_trace_admission(
            manifest_path,
            expected_manifest_file_sha256="f" * 64,
        )

    original_manifest = manifest_path.read_bytes()
    original_decision_shard = (output / manifest["shards"][0]["inventory_shard"]).read_bytes()
    with pytest.raises(
        Active8TraceInventoryError,
        match="immutable active-8 inventory already exists",
    ):
        build_active8_trace_inventory(
            (
                Active8SourceShard(
                    manifest_layer="general_corruption",
                    envelope_layer="corruption",
                    partition="train",
                    relative_path="train/fixture.jsonl.gz",
                    path=shard,
                ),
            ),
            exact_candidate_checker=_accept_exact_teacher,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            support_contract_sha256="a" * 64,
            output_dir=output,
        )
    assert manifest_path.read_bytes() == original_manifest
    assert (
        output / manifest["shards"][0]["inventory_shard"]
    ).read_bytes() == original_decision_shard
