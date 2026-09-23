"""The exploration floors must be applied in a deterministic order.

WHY THIS IS A STRUCTURAL GUARD AND NOT A BEHAVIOURAL ONE. Both floors MUTATE the weight
vector as they are applied, so the result depends on the order. Iterating a set of strings
makes that order depend on PYTHONHASHSEED, and two identical runs then produced parent
probabilities 3.0e-3 apart -- measured, and the reason a serialization memo that returns
byte-identical SMILES appeared to change the trajectory.

The FIX is verified behaviourally by the determinism controls
(`diagnostics/pmo_determinism_control_v1/`), which compare four real concurrent campaigns
and require exact equality of parent probabilities, option probabilities, selected
endpoints, the charged sequence, observations and RNG state. Those passed at 0.000e+00.

A unit fixture was attempted three times and could not be made discriminating: provoking it
needs the floors to actually BIND, which needs a weight distribution concentrated enough
that most families fall under 0.02, and the fixtures tried never reached that. A test that
passes against the buggy code is worse than no test, so this asserts the property directly
instead of pretending to measure it.
"""
from __future__ import annotations

import ast
import pathlib

SOURCE = pathlib.Path("src/compose_v4/control/pmo_reward_adaptive.py")


def _floor_loops() -> list[ast.For]:
    """The `for` loops inside `intent_policy` that apply a floor by mutating weights."""
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "intent_policy":
            return [n for n in ast.walk(node) if isinstance(n, ast.For)]
    raise AssertionError("intent_policy not found; the guard is addressing nothing")


def test_every_floor_loop_iterates_a_sorted_sequence():
    loops = _floor_loops()
    # Derived from the source, so a new floor cannot be added outside the guard.
    mutating = [
        loop for loop in loops
        if any(isinstance(n, ast.AugAssign) for n in ast.walk(loop))
    ]
    assert mutating, "no weight-mutating loop found; the guard would be vacuous"
    for loop in mutating:
        call = loop.iter
        assert isinstance(call, ast.Call) and getattr(call.func, "id", None) == "sorted", (
            f"floor loop at line {loop.lineno} iterates an unordered sequence; "
            "its result then depends on PYTHONHASHSEED"
        )
