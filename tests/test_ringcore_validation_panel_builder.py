from __future__ import annotations

import copy
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.production_edit_corpus import LayeredEditCorpus
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.hierarchical_sampler import (
    CYCLE_OPS,
    GENERAL_CORRUPTION,
    MMP_ANALOGUE,
    build_layered_sampler,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (
    load_json_object,
)
from compose_v4.experiments.ringcore_validation_panel import (
    ValidationPanelError,
    validate_validation_panel_artifact,
)
from compose_v4.experiments.ringcore_validation_panel_builder import (
    build_family_forensics_validation_panel,
    build_production_validation_panel,
    panel_build_identity,
    panel_sampler_implementation_sha256,
    validate_panel_corpus_sampler,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ringcore_v1_successor_leaderboard_v1.json"
INVENTORY_PATH = (
    ROOT / "diagnostics" / "coherence" / "ringcore_v1_scientific_a7546e2_frozen_inventory.json"
)
_PACKED_SHA256 = "4" * 64


def _record(*, layer: str, entry_index: int) -> PathRecord:
    source = pad_molecular_graph(smiles_to_molecular_graph("C"), 6)
    target = pad_molecular_graph(smiles_to_molecular_graph("CC"), 6)
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
        flexible_size=True,
    )
    path = TraceProgressCTMC(trace)
    address = PackedTraceAddress(
        packed_shard_content_sha256=_PACKED_SHA256,
        packed_shard_name=f"shard_{entry_index:04d}.jsonl.gz",
        entry_index=entry_index,
        trace_id=f"trace-{layer}-{entry_index}",
        layer=layer,
        partition="validation",
        source_key=canonical_state_key(path.state_at(0)),
        target_key=canonical_state_key(path.state_at(path.path_length)),
        path_length=path.path_length,
    )
    return PathRecord(
        target_key=address.target_key,
        path=path,
        corpus_address=address,
    )


def _fixture() -> tuple[dict, dict, LayeredEditCorpus]:
    config = load_json_object(CONFIG_PATH)
    config["panels"]["production_law"]["initial_draws"] = 12
    config["panels"]["production_law"]["expanded_draws"] = 24
    config["panels"]["family_forensics"]["active_families"] = ["atom_insert"]
    config["panels"]["family_forensics"]["initial_minimum_nonterminal_examples_per_family"] = 2
    config["panels"]["family_forensics"][
        "expanded_minimum_nonterminal_examples_per_ambiguous_family"
    ] = 4
    config["panels"]["family_forensics"]["maximum_stream_draws"] = 128
    raw_layers = {
        GENERAL_CORRUPTION: (_record(layer="corruption", entry_index=0),),
        CYCLE_OPS: (_record(layer="cycle_ops", entry_index=1),),
        MMP_ANALOGUE: (_record(layer="mmp_analogue", entry_index=2),),
    }
    for layer in raw_layers:
        config["validation_data"]["layers"][layer]["records"] = 1
    weights = {
        layer: float(config["validation_data"]["layers"][layer]["trace_draw_weight"])
        for layer in raw_layers
    }
    sampler = build_layered_sampler(
        raw_layers,
        layer_weights=weights,
        path_length_bins=(5, 9, 13),
        cold_element_floor=0.0,
        seed=0,
    )
    records = tuple(
        record
        for layer in (GENERAL_CORRUPTION, CYCLE_OPS, MMP_ANALOGUE)
        for record in raw_layers[layer]
    )
    corpus = LayeredEditCorpus(
        records=records,
        layer_bounds={
            GENERAL_CORRUPTION: (0, 1),
            CYCLE_OPS: (1, 2),
            MMP_ANALOGUE: (2, 3),
        },
        sampler=sampler,
        provenance={"partition": "validation"},
    )
    return config, load_json_object(INVENTORY_PATH), corpus


def test_sampler_implementation_hash_matches_frozen_protocol() -> None:
    config, _inventory, _corpus = _fixture()
    observed = panel_sampler_implementation_sha256()
    assert observed == config["validation_data"]["record_sampling"]["implementation_sha256"]
    assert len(observed) == 64


def test_production_panel_is_deterministic_exact_addressed_prefix() -> None:
    config, inventory, corpus = _fixture()
    first = build_production_validation_panel(
        corpus,
        stage="initial",
        config=config,
        inventory=inventory,
    )
    second = build_production_validation_panel(
        corpus,
        stage="initial",
        config=config,
        inventory=inventory,
    )

    assert first == second
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert len(first["rows"]) == 12
    assert [row["stream_draw_index"] for row in first["rows"]] == list(range(12))
    assert all(row["partition"] == "validation" for row in first["rows"])
    assert all(row["exact_state_ref"]["shard_sha256"] == _PACKED_SHA256 for row in first["rows"])
    assert {row["layer"] for row in first["rows"]} <= {
        GENERAL_CORRUPTION,
        CYCLE_OPS,
        MMP_ANALOGUE,
    }
    for row in first["rows"]:
        if row["terminal"]:
            assert row["semantic_cell_id"] is None
            assert row["teacher_family"] is None
        else:
            assert row["semantic_cell_id"] is not None
            assert row["teacher_family"] == "atom_insert"
            assert row["teacher_action_sha256"] is not None
    validate_validation_panel_artifact(
        first,
        config=config,
        inventory=inventory,
    )


def test_expanded_panel_retains_the_exact_initial_prefix() -> None:
    config, inventory, corpus = _fixture()
    initial = build_production_validation_panel(
        corpus,
        stage="initial",
        config=config,
        inventory=inventory,
    )
    expanded = build_production_validation_panel(
        corpus,
        stage="expanded",
        config=config,
        inventory=inventory,
    )

    for left, right in zip(
        initial["rows"],
        expanded["rows"][: len(initial["rows"])],
        strict=True,
    ):
        assert left == right


def test_forensics_panel_retains_exact_family_target_in_stream_order() -> None:
    config, inventory, corpus = _fixture()
    panel = build_family_forensics_validation_panel(
        corpus,
        requested_nonterminal_examples_by_family={"atom_insert": 2},
        config=config,
        inventory=inventory,
    )

    assert len(panel["rows"]) == 2
    assert all(not row["terminal"] for row in panel["rows"])
    assert all(row["teacher_family"] == "atom_insert" for row in panel["rows"])
    assert [row["stream_draw_index"] for row in panel["rows"]] == sorted(
        row["stream_draw_index"] for row in panel["rows"]
    )
    assert panel["draw_contract"]["family_nonterminal_census"] == {"atom_insert": 2}
    validate_validation_panel_artifact(
        panel,
        config=config,
        inventory=inventory,
    )


def test_builder_refuses_sampler_or_partition_drift() -> None:
    config, inventory, corpus = _fixture()
    wrong_bins = copy.deepcopy(corpus.sampler)
    wrong_bins.path_length_bins = (1, 2, 4)
    drifted = LayeredEditCorpus(
        records=corpus.records,
        layer_bounds=corpus.layer_bounds,
        sampler=wrong_bins,
        provenance=corpus.provenance,
    )
    with pytest.raises(ValidationPanelError, match="path-length bins"):
        validate_panel_corpus_sampler(drifted, config=config)

    first = corpus.records[0]
    address = first.corpus_address
    assert address is not None
    train_address = PackedTraceAddress(
        packed_shard_content_sha256=address.packed_shard_content_sha256,
        packed_shard_name=address.packed_shard_name,
        entry_index=address.entry_index,
        trace_id=address.trace_id,
        layer=address.layer,
        partition="train",
        source_key=address.source_key,
        target_key=address.target_key,
        path_length=address.path_length,
    )
    records = list(corpus.records)
    records[0] = PathRecord(
        target_key=first.target_key,
        path=first.path,
        corpus_address=train_address,
    )
    wrong_partition = LayeredEditCorpus(
        records=tuple(records),
        layer_bounds=corpus.layer_bounds,
        sampler=corpus.sampler,
        provenance=corpus.provenance,
    )
    with pytest.raises(ValidationPanelError, match="outside validation"):
        build_production_validation_panel(
            wrong_partition,
            stage="initial",
            config=config,
            inventory=inventory,
        )


def test_panel_build_identity_is_compact_and_hash_bound() -> None:
    config, inventory, corpus = _fixture()
    panel = build_production_validation_panel(
        corpus,
        stage="initial",
        config=config,
        inventory=inventory,
    )
    identity = panel_build_identity(
        config=config,
        inventory=inventory,
        panel=panel,
    )

    assert identity["panel_artifact_sha256"] == panel["artifact_sha256"]
    assert identity["row_count"] == 12
    assert identity["nonterminal_row_count"] + identity["terminal_row_count"] == 12
