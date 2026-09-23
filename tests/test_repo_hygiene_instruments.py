"""Guard the preservation instruments themselves.

A broken gate does not fail; it passes. Every mechanism in this repository that
was built, tested and then reached by no caller was found after the fact, and a
merge gate that silently stops detecting regressions is the same shape one level
up. So the instruments that decide whether a structural pass lost anything get
their own tests, and each test is written to fail if the corresponding detection
is removed rather than to restate how the code is written.

Covered:

* the pin scanner recognises each pin SHAPE and refuses a lookalike;
* the fingerprint parser accounts for every ``FAILED``/``ERROR`` line, including
  the truncated parametrized ids that an end-anchored pattern silently dropped;
* the fingerprint comparison treats an added test as fine and a VANISHED test as
  disqualifying, which is the direction that matters because a collection error
  removes tests while the run still reports zero failures;
* the capability comparison is a superset check, not an equality check.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_hygiene_{name}", _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---- pin scanner ----


def test_pin_scanner_recognises_key_is_path_and_sibling_shapes(tmp_path):
    scanner = _load("repo_pinned_file_set")
    digest = "a" * 64
    document = {
        "runtime_inputs_sha256": {"src/compose_v4/chem/state.py": digest},
        "capsule": {"path": "tools/t4_launch.py", "payload_sha256": digest},
        # A lookalike: a path key whose value is not a digest, and a mapping with
        # a digest but no path-valued sibling. Neither may be collected.
        "not_a_pin": {"src/compose_v4/gm/rates.py": "not-a-digest"},
        "also_not": {"run_id": digest, "count": 3},
    }
    artifact = tmp_path / "contract.json"
    artifact.write_text(json.dumps(document), encoding="utf-8")

    tracked = {
        "src/compose_v4/chem/state.py",
        "tools/t4_launch.py",
        "src/compose_v4/gm/rates.py",
    }
    sink: dict[str, list[dict]] = {}
    from collections import defaultdict

    sink = defaultdict(list)
    scanner.scan_json(artifact, tracked, sink)

    assert "src/compose_v4/chem/state.py" in sink, "key_is_path shape was not recognised"
    assert "tools/t4_launch.py" in sink, "sibling shape was not recognised"
    assert "src/compose_v4/gm/rates.py" not in sink, (
        "a path key whose value is not a digest must not be collected as a pin"
    )


def test_pin_scanner_ignores_untracked_paths(tmp_path):
    """Over-collection is the safe direction, but only within the repository."""
    scanner = _load("repo_pinned_file_set")
    artifact = tmp_path / "c.json"
    artifact.write_text(json.dumps({"x": {"/opt/dock/qvina02.py": "b" * 64}}), encoding="utf-8")
    from collections import defaultdict

    sink = defaultdict(list)
    scanner.scan_json(artifact, set(), sink)
    assert not sink, "a container-absolute path is not this repository's file to pin"


# ---- fingerprint parser ----


def test_fingerprint_parser_accounts_for_truncated_parametrized_ids():
    fingerprint = _load("repro_test_fingerprint")
    # The second line is what pytest emits when a parametrized id contains spaces
    # and the summary is truncated at terminal width: no closing bracket, and no
    # " - " separator for an end-anchored pattern to rely on.
    raw = (
        "FAILED tests/a.py::test_one - AssertionError: boom\n"
        "FAILED tests/b.py::test_two[selected_step-step, stream\n"
        "ERROR tests/c.py\n"
        "3 failed in 1.0s\n"
    )
    report = fingerprint.parse(raw)
    assert report["failed_count"] == 2, "a truncated parametrized id was dropped"
    assert report["error_count"] == 1
    assert "tests/b.py::test_two[selected_step-step, stream" in report["failed"]


def test_fingerprint_parser_raises_rather_than_under_counting(monkeypatch):
    """The count guard must fire when the pattern stops matching a real line."""
    fingerprint = _load("repro_test_fingerprint")
    import re

    # Simulate a future pattern change that no longer matches parametrized ids.
    monkeypatch.setattr(
        fingerprint, "OUTCOME", re.compile(r"^(?P<kind>FAILED|ERROR)\s+(?P<rest>\S+)$")
    )
    raw = "FAILED tests/a.py::test_one - AssertionError: boom\n"
    with pytest.raises(ValueError, match="accounted for"):
        fingerprint.parse(raw)


# ---- fingerprint comparison ----


def _fingerprint(nodes, failed=()):
    return {
        "collected": len(nodes),
        "collected_nodes": sorted(nodes),
        "failed": {n: "" for n in failed},
        "errored": {},
    }


def test_added_test_is_comparable_but_vanished_test_is_not():
    fingerprint = _load("repro_test_fingerprint")
    base = _fingerprint(["a::1", "a::2", "a::3"])

    added = fingerprint.compare(base, _fingerprint(["a::1", "a::2", "a::3", "a::4"]))
    assert added["verdict"] == "NO_TEST_REGRESSION", "adding a test must not trip the gate"

    vanished = fingerprint.compare(base, _fingerprint(["a::1", "a::2", "a::4"]))
    assert vanished["verdict"] == "NOT_COMPARABLE_TESTS_DISAPPEARED", (
        "a vanished test is the signature of a collection error swallowing tests "
        "while the run still reports zero failures"
    )

    regressed = fingerprint.compare(base, _fingerprint(["a::1", "a::2", "a::3"], failed=["a::2"]))
    assert regressed["verdict"] == "TEST_REGRESSION"
    assert regressed["new_regressions"] == ["a::2"]


# ---- capability comparison ----


def test_capability_comparison_is_a_superset_check_not_an_equality_check():
    capability = _load("repo_capability_baseline")

    def report(modules):
        return {
            "interpreter": "3.12.9",
            "library_versions": {},
            "modules": {
                name: {
                    "ast_symbols": symbols,
                    "import_symbols": symbols,
                    "import_ok": True,
                }
                for name, symbols in modules.items()
            },
        }

    base = report({"m": ["f", "g"]})

    gained = capability.compare(base, report({"m": ["f", "g", "h"]}))
    assert gained["verdict"] == "CAPABILITY_PRESERVED", "a gained symbol is not a loss"

    lost = capability.compare(base, report({"m": ["f"]}))
    assert lost["verdict"] == "CAPABILITY_LOST"
    assert "m.g" in lost["import_symbols"]["lost"]

    dropped_module = capability.compare(base, report({}))
    assert dropped_module["verdict"] == "CAPABILITY_LOST", (
        "a module that stopped importing is a lost capability even if no symbol moved"
    )
