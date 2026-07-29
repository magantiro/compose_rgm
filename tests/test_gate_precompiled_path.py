"""Under --precompiled-corpus the trainer must never fall back to the in-memory record builders.

Those builders (build_corrupted_prior_records / build_cycle_op_records / build_analogue_prior_records)
regenerate a few hundred sources at launch -- the DATA_STARVED_BASELINE -- with no partition discipline
and no contract hashes. A silent fallback would produce a run that trains successfully on the wrong data,
which is far worse than a crash.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

GATE = REPO / "scripts/train_tracelet_cnof_gate.py"
LEGACY_BUILDERS = (
    "build_corrupted_prior_records",
    "build_cycle_op_records",
    "build_analogue_prior_records",
)


def _guarding_conditions(tree, builder):
    """Every `if`/`elif` test that encloses a call to ``builder``."""
    found = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_If(self, node):
            self.stack.append(node.test)
            for child in node.body:
                self.visit(child)
            self.stack.pop()
            # Descending into orelse means the test was FALSE -- an `elif` chain is nested If nodes in
            # orelse, so its negated guard has to be recorded or `elif` reads as unguarded.
            self.stack.append(ast.UnaryOp(op=ast.Not(), operand=node.test))
            for child in node.orelse:
                self.visit(child)
            self.stack.pop()

        def visit_Call(self, node):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == builder:
                found.append(list(self.stack))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


@pytest.mark.parametrize("builder", LEGACY_BUILDERS)
def test_every_legacy_builder_call_is_guarded_against_precompiled_mode(builder):
    """Structural: no call site may be reachable when --precompiled-corpus is set."""
    tree = ast.parse(GATE.read_text())
    call_sites = _guarding_conditions(tree, builder)
    assert call_sites, f"{builder} is no longer called anywhere -- update this test"
    for conditions in call_sites:
        guards = " || ".join(ast.unparse(c) for c in conditions)
        assert "precompiled_corpus" in guards, (
            f"a call to {builder} is reachable without excluding precompiled mode; guards were: {guards}"
        )


def test_precompiled_flag_requires_the_mmp_pool():
    """mmp_analogue carries positive weight, so omitting its source would renormalize the mixture."""
    source = GATE.read_text()
    assert "--precompiled-mmp-pool" in source
    assert "requires --precompiled-mmp-pool" in source


def test_loader_module_does_not_reference_the_legacy_builders():
    """Behavioural surface: the production loader must not import or call them at all."""
    module = (REPO / "src/compose_v4/data/production_edit_corpus.py").read_text()
    for builder in LEGACY_BUILDERS:
        assert builder not in module, f"{builder} leaked into the production corpus loader"


def test_layer_weights_hash_guard_is_present():
    """A drifted weight mapping must abort before the GPU, not train a different mixture."""
    source = GATE.read_text()
    assert "recompute_layer_weights_hash()" in source
    assert "refusing to train" in source


def test_production_loader_works_without_the_legacy_builders_importable(monkeypatch):
    """If the loader secretly depended on a legacy builder, breaking it would break the loader."""
    import compose_v4.experiments.analogue_prior as analogue
    import compose_v4.experiments.corrupted_source_prior as corrupted
    import compose_v4.experiments.cycle_op_prior as cycle

    def _boom(*args, **kwargs):
        raise AssertionError("legacy in-memory builder invoked on the production path")

    monkeypatch.setattr(corrupted, "build_corrupted_prior_records", _boom, raising=False)
    monkeypatch.setattr(cycle, "build_cycle_op_records", _boom, raising=False)
    monkeypatch.setattr(analogue, "build_analogue_prior_records", _boom, raising=False)

    from compose_v4.data.production_edit_corpus import load_mmp_records

    pool = REPO / "tests/fixtures/analogue_trace_pool_sample.jsonl"
    if not pool.exists():
        pytest.skip("local analogue pool sample unavailable")
    records, stats = load_mmp_records(pool, partition="train")
    assert stats["scanned"] > 0
    assert len(records) == stats["kept"]


def test_production_refuses_raw_pool_fallback_for_mmp():
    """The raw-pool path replays every trace (~83 min/partition). Production must never take it."""
    module = (REPO / "src/compose_v4/data/production_edit_corpus.py").read_text()
    assert "require_packed_mmp" in module
    assert "refusing to fall back to the raw pool" in module


def test_mmp_stats_declare_their_source():
    """The launch log must say whether MMP came from the packed store or a replay fallback."""
    module = (REPO / "src/compose_v4/data/production_edit_corpus.py").read_text()
    assert '"source": "packed"' in module
    assert '"raw_pool_replay"' in module
