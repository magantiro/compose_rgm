"""The chain verifier must see a pin written in Python, and must scope it.

Two gaps in this verifier let a genuinely stale pin pass as agreement during
this correction round, and both are the same failure in different clothes: the
tool reported ``AGREES`` while
``EXPECTED_T1_CAPACITY_POLICY`` in ``editing_training_gate.py`` addressed a
value no artifact held, which ``tests/test_editing_training_gate.py`` then
rejected with "editing training gate must bind the exact prospective semantic T1
policy". A verifier that cannot fail on a real defect is worse than no verifier,
because it is quoted as evidence.

1. Only JSON was scanned for pointer edges, so a pin expressed as a module-level
   dict constant was invisible.
2. Chain membership was derived from values matching a live or base index, so an
   artifact whose pin resolved to NOTHING -- the worst case, not a benign one --
   fell out of the chain and had its finding downgraded to a warning.

Both fixtures below are absolute: the expected values are literals, not
recomputations of the code under test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "verify_process_v2_hash_chain.py"


@pytest.fixture(scope="module")
def verifier():
    spec = importlib.util.spec_from_file_location("chain_verifier_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves __module__ through sys.modules during class creation.
    sys.modules["chain_verifier_under_test"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("chain_verifier_under_test", None)


_PINNED_CONFIG = "configs/editing_v2_semantic_t1_capacity_policy_v1.json"
# The exact shape that went stale, including the parenthesised continuation
# ruff's line length forces on a 64-character literal.
_MODULE_TEXT = f'''
EXPECTED_T1_CAPACITY_POLICY = {{
    "semantic_t1_capacity_policy": ("{_PINNED_CONFIG}"),
    "semantic_t1_capacity_policy_file_sha256": (
        "5e0d9bc296f512dd462e83297f35d8ccdc5d4ed4c9d97d9f8cd5e7f5c51457c0"
    ),
    "semantic_t1_capacity_policy_sha256": (
        "8526fbc01a12b0fe61a1d754d4850cda5b6ff3dd2092c6c810ebc5da9df76085"
    ),
}}
'''


def test_a_pin_written_as_a_python_dict_constant_is_discovered(verifier) -> None:
    edges = verifier._python_pointer_edges(_ROOT, "src/example.py", _MODULE_TEXT)

    assert {(edge.location, edge.target, edge.value) for edge in edges} == {
        (
            "EXPECTED_T1_CAPACITY_POLICY..semantic_t1_capacity_policy_file_sha256",
            _PINNED_CONFIG,
            "5e0d9bc296f512dd462e83297f35d8ccdc5d4ed4c9d97d9f8cd5e7f5c51457c0",
        ),
        (
            "EXPECTED_T1_CAPACITY_POLICY..semantic_t1_capacity_policy_sha256",
            _PINNED_CONFIG,
            "8526fbc01a12b0fe61a1d754d4850cda5b6ff3dd2092c6c810ebc5da9df76085",
        ),
    }


def test_python_edge_discovery_survives_modules_it_cannot_evaluate(verifier) -> None:
    """A file it cannot parse must yield nothing, never raise: the verifier
    scans every tracked Python file, and one unparseable module must not take
    the whole audit down."""

    assert verifier._python_pointer_edges(_ROOT, "src/broken.py", "def (:\n") == ()
    assert verifier._python_pointer_edges(_ROOT, "src/plain.py", "X = 1\n") == ()
    # A dict naming a path that does not exist is not a pin.
    absent = 'D = {"p": "configs/does_not_exist.json", "p_sha256": "%s"}\n' % ("a" * 64)
    assert verifier._python_pointer_edges(_ROOT, "src/absent.py", absent) == ()


def test_the_real_training_gate_pin_is_discovered_and_resolves(verifier) -> None:
    """The live constant must address the live config, by both roles.

    This is the assertion the round-one re-pin would have failed.
    """

    relative = "src/compose_v4/experiments/editing_training_gate.py"
    text = (_ROOT / relative).read_text(encoding="utf-8")
    edges = verifier._python_pointer_edges(_ROOT, relative, text)
    assert edges, "the training gate's T1 capacity policy pin was not discovered"

    live = verifier._live_value_index(_ROOT, verifier._tracked_paths(_ROOT, verifier.SCAN_GLOBS))
    for edge in edges:
        values = verifier._target_values(_ROOT, edge.target, live)
        assert edge.value in values.values(), (edge.location, edge.value, values)


def test_an_unresolvable_pin_still_places_its_source_in_the_chain(verifier) -> None:
    """Chain membership must follow the edge, not the value.

    A pin whose value matches neither a live nor a base artifact is exactly the
    case that must NOT be scoped out, because a finding outside the chain is
    downgraded from a failure to a warning.
    """

    live = verifier.ValueIndex()
    live.add(_PINNED_CONFIG, "policy_sha256", "b" * 64)
    base = verifier.ValueIndex()
    orphan = verifier.PointerEdge(
        source="src/example.py",
        location="EXPECTED..pin_sha256",
        target=_PINNED_CONFIG,
        value="c" * 64,  # matches nothing at all
    )

    without_edges = verifier._chain_report((), (), live, base, seeds=(_PINNED_CONFIG,))
    with_edges = verifier._chain_report(
        (), (), live, base, seeds=(_PINNED_CONFIG,), edges=(orphan,)
    )

    assert "src/example.py" not in {row["artifact"] for row in without_edges}
    assert "src/example.py" in {row["artifact"] for row in with_edges}
