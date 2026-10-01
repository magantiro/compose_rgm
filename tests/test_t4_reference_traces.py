"""Trace capture preserves the generated pool and records actual post-intervention programs."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from compose_v4.control.reference_programs import t4_program_input
from compose_v4.control.structural_subgoal_realizer import RealizerConfig
from compose_v4.experiments import t4_fiber_campaign as t4
from compose_v4.rewrite.trace_shard import decode_state


def generate(*, capture, seed=11):
    # Keep the compiler unit fixture independent of a QED/SA or docking target.
    gate = SimpleNamespace(delta=0.4, check=lambda smiles: {"smiles": smiles})
    rng = np.random.default_rng(seed)
    rows = t4.expand(
        "CCO",
        -8.0,
        gate,
        rng,
        draws=1,
        horizon=1,
        include_realized_actions=capture,
        realizer_config=RealizerConfig(maximum_expansions=64) if capture else None,
    )
    return rows, rng.bit_generator.state


def test_captured_traces_do_not_change_candidates_order_or_rng():
    baseline, baseline_rng = generate(capture=False)
    captured, captured_rng = generate(capture=True)
    assert baseline and captured
    extra = {
        "source_state",
        "trace_origin",
        "trace_compilation",
        "realized_actions",
        "realized_endpoint_key",
    }
    assert [{k: v for k, v in row.items() if k not in extra} for row in captured] == baseline
    assert baseline_rng == captured_rng
    realized = [row for row in captured if "realized_actions" in row]
    assert realized, "test must exercise exact program replay, not only abstentions"
    assert any(row["families"] != ["base"] for row in realized)
    for row in realized:
        assert row["trace_origin"] == "compiled_structural_goal"
        program = t4_program_input(row, candidate_id=row["smiles"])
        assert program.trace_sha256
        assert program == t4_program_input(
            row, candidate_id=row["smiles"], source=decode_state(row["source_state"])
        )


def test_abstention_is_not_candidate_filtering(monkeypatch):
    monkeypatch.setattr(
        t4,
        "realize_structural_goal",
        lambda *a, **k: {
            "status": "search_limit_abstention",
            "expanded": 64,
            "attempted": 100,
        },
    )
    baseline, baseline_rng = generate(capture=False)
    captured, captured_rng = generate(capture=True)
    assert [row["smiles"] for row in captured] == [row["smiles"] for row in baseline]
    assert baseline_rng == captured_rng
    for row in captured:
        assert row["trace_compilation"]["status"] == "search_limit_abstention"
        assert t4_program_input(row, candidate_id=row["smiles"]).trace_json is None


def test_recorded_source_cannot_be_silently_replaced():
    rows, _ = generate(capture=True)
    row = next(row for row in rows if row.get("realized_actions"))
    source = decode_state(row["source_state"])
    changed = source.implicit_h_counts.copy()
    changed[0] += 1
    with pytest.raises(ValueError, match="differs from recorded source"):
        t4_program_input(
            row, candidate_id=row["smiles"], source=replace(source, implicit_h_counts=changed)
        )
    with pytest.raises(ValueError, match="exact source_state"):
        t4_program_input(
            {k: v for k, v in row.items() if k != "source_state"}, candidate_id=row["smiles"]
        )
