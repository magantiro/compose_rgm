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


def test_benchmark_contract_refuses_unpacked_layers():
    """BENCHMARK_CONTRACT: every layer must be served from packed storage, or the run aborts."""
    source = GATE.read_text()
    assert "BENCHMARK_CONTRACT violation" in source
    assert "layers not served from packed storage" in source
    assert "refusing to run on a fallback path" in source


def test_startup_reports_storage_for_every_layer():
    """Startup must PROVE per-layer storage, not leave it inferable from an MMP-only stats blob."""
    module = (REPO / "src/compose_v4/data/production_edit_corpus.py").read_text()
    assert '"layer_storage"' in module
    assert '"audit_shard_replay"' in module
    assert '"layer_storage": corpus.provenance["layer_storage"]' in GATE.read_text()


def test_precompiled_corpus_skips_all_denovo_compilation():
    """Both zero-mixture modes must skip de-novo work, not just --scaled-manifest.

    Regression: the de-novo branches were gated on args.scaled_manifest alone, so a --precompiled-corpus
    run compiled 50,000 carbon-tree paths it never uses -- slow, and it crashed with InvalidRewrite before
    the first optimizer step. Every such branch must consult the shared zero_denovo predicate.
    """
    source = GATE.read_text()
    assert "zero_denovo = denovo_weight == 0.0" in source
    for branch in (
        'if args.source_prior == "carbon_tree" and not zero_denovo:',
        "tree_paths_ready = False\n    if zero_denovo:",
        "if empty_partitions and not zero_denovo:",
    ):
        assert branch in source, f"de-novo branch not routed through zero_denovo: {branch!r}"


def test_no_denovo_branch_still_keys_on_scaled_manifest_alone():
    """Any NEW de-novo branch keyed on scaled_manifest alone would reintroduce the same bug."""
    import re

    source = GATE.read_text()
    # the remaining scaled_manifest uses are validation, sizing, logging and the sampler selector --
    # none may guard de-novo compilation
    denovo_markers = ("carbon_tree", "tree_paths_ready", "empty_partitions", "compile de-novo")
    for line_no, line in enumerate(source.splitlines(), 1):
        if "args.scaled_manifest" not in line:
            continue
        if any(marker in line for marker in denovo_markers):
            raise AssertionError(
                f"line {line_no} guards de-novo work on scaled_manifest alone: {line.strip()!r}"
            )
    assert re.search(r"zero_denovo", source)


def test_zero_denovo_comes_from_the_mixture_not_the_storage_mode():
    """Storage mode must never decide scientific mixture.

    A future packed run with a POSITIVE de-novo weight must still compile de-novo paths, so the predicate
    reads the authoritative mixture (PRODUCTION_LAYER_WEIGHTS / locked_mixture) rather than the presence
    of a --packed-* or --precompiled-* flag.
    """
    source = GATE.read_text()
    assert "def _resolve_denovo_weight(args)" in source
    assert "denovo_weight = _resolve_denovo_weight(args)" in source
    assert "zero_denovo = denovo_weight == 0.0" in source
    assert "zero_denovo = bool(args.scaled_manifest or args.precompiled_corpus)" not in source


def test_resolve_denovo_weight_reads_the_locked_mixture(tmp_path, monkeypatch):
    import argparse
    import json as _json
    import importlib

    gate = importlib.import_module("train_tracelet_cnof_gate")

    manifest = tmp_path / "m.json"
    manifest.write_text(_json.dumps({"locked_mixture": {"layer_weights": {"denovo": 0.25, "mmp": 0.75}}}))
    args = argparse.Namespace(precompiled_corpus=None, scaled_manifest=str(manifest))
    assert gate._resolve_denovo_weight(args) == 0.25          # positive -> de-novo must still compile

    manifest.write_text(_json.dumps({"locked_mixture": {"layer_weights": {"mmp": 1.0}}}))
    assert gate._resolve_denovo_weight(args) == 0.0

    # the production mixture has no de-novo layer at all
    args = argparse.Namespace(precompiled_corpus="/artifacts/edit_packed_v1", scaled_manifest=None)
    assert gate._resolve_denovo_weight(args) == 0.0

    # an ordinary run is fully de-novo
    args = argparse.Namespace(precompiled_corpus=None, scaled_manifest=None)
    assert gate._resolve_denovo_weight(args) == 1.0


def test_startup_emits_the_required_phase_markers():
    """A watcher must key off phases, not an empty output directory (which reads as 'still running')."""
    source = GATE.read_text()
    assert '"phase": "mixture_resolved"' in source
    assert '"phase": "PACKED_CORPUS_READY"' in source
    assert '"de_novo_records_built"' in source
    loop = (REPO / "src/compose_v4/experiments/factorized_mark_conditional.py").read_text()
    assert '"phase": "OPTIMIZER_STEP_1"' in loop


def test_zero_denovo_asserts_no_denovo_records_were_built():
    source = GATE.read_text()
    assert "de-novo records were built" in source
    assert "a de-novo branch" in source
