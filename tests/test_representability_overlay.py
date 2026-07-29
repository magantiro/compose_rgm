"""Exclusions must be a FROZEN list, not an open-ended runtime rule.

An open-ended "drop what fails" rule would let a future model or enumerator change silently erode the
corpus while every count still looked healthy and the manifest kept claiming the old identity. The overlay
converts that into a loud failure the first time support changes.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from compose_v4.data.representability_overlay import (  # noqa: E402
    REASON_MULTI_NEIGHBOUR_INSERT,
    REPRESENTABILITY_FILTER,
    RepresentabilityViolation,
    build_overlay,
    check_unlisted,
    excluded_keys,
    filter_implementation_hash,
    load_overlay,
    trace_key,
    unsupported_steps,
)


@dataclass
class _Action:
    neighbors: tuple | None = None


@dataclass
class _Step:
    action: object
    rule_name: str = "atom_insert"


@dataclass
class _Trace:
    steps: tuple


def _trace(*neighbor_sets):
    return _Trace(tuple(_Step(_Action(neighbors=n)) for n in neighbor_sets))


def test_zero_and_one_neighbour_inserts_are_supported():
    assert unsupported_steps(_trace((), ((5, 1),))) == []


def test_two_neighbour_insert_is_unsupported_with_payload():
    """The exact shape that crashed the run: neighbors [[5,1],[16,1]]."""
    problems = unsupported_steps(_trace(((5, 1), (16, 1))))
    assert len(problems) == 1
    assert problems[0]["reason"] == REASON_MULTI_NEIGHBOUR_INSERT
    assert problems[0]["payload"]["neighbors"] == [[5, 1], [16, 1]]


def test_whole_trace_is_excluded_not_just_the_bad_step():
    """A trace is drawn as a unit; keeping one direction would bias family supervision."""
    trace = _trace((), ((5, 1),), ((5, 1), (16, 1)))
    assert unsupported_steps(trace)          # the trace is unusable, not repairable step-wise


def test_listed_exclusion_is_omitted_and_unlisted_one_raises():
    bad = _trace(((5, 1), (16, 1)))
    key = trace_key(bad)
    assert check_unlisted(bad, key, {key}, layer="general_corruption", partition="train") is True
    with pytest.raises(RepresentabilityViolation, match="NOT listed in the frozen overlay"):
        check_unlisted(bad, key, set(), layer="general_corruption", partition="train")


def test_supported_trace_is_never_treated_as_an_exclusion():
    good = _trace((), ((5, 1),))
    assert check_unlisted(good, trace_key(good), set(), layer="mmp_analogue", partition="train") is False


def test_overlay_records_everything_required(tmp_path):
    exclusions = [{
        "layer": "general_corruption", "partition": "train", "shard": "shard_0016.jsonl.gz",
        "trace_key": "t42", "reason": REASON_MULTI_NEIGHBOUR_INSERT,
        "steps": [{"step_index": 1, "family": "atom_insert",
                   "reason": REASON_MULTI_NEIGHBOUR_INSERT,
                   "payload": {"neighbors": [[5, 1], [16, 1]]}}],
        "path_length": 3,
    }]
    counts = {"general_corruption": {"checked": 74949, "accepted": 74948, "excluded": 1}}
    overlay = build_overlay(exclusions, counts=counts, enumerator_hash="abc123",
                            packed_manifest_hashes={"general_corruption/train/shard_0016.jsonl.gz": "d0"})
    for key in ("representability_filter", "filter_implementation_hash", "candidate_enumerator_hash",
                "packed_manifest_hashes", "exclusions", "counts", "effective_corpus_checksum"):
        assert key in overlay
    assert overlay["representability_filter"] == REPRESENTABILITY_FILTER
    assert excluded_keys(overlay, "general_corruption", "train") == {"t42"}
    assert excluded_keys(overlay, "general_corruption", "validation") == set()


def test_overlay_is_rejected_when_the_filter_rule_changes(tmp_path):
    """A changed support rule must invalidate the frozen census, not silently reuse it."""
    overlay = build_overlay([], counts={}, enumerator_hash="abc", packed_manifest_hashes={})
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps(overlay))
    assert load_overlay(path)["filter_implementation_hash"] == filter_implementation_hash()

    overlay["filter_implementation_hash"] = "0000000000000000"
    path.write_text(json.dumps(overlay))
    with pytest.raises(RepresentabilityViolation, match="representability filter changed"):
        load_overlay(path)


def test_effective_corpus_checksum_changes_with_the_excluded_set():
    a = build_overlay([{"layer": "l", "partition": "train", "trace_key": "x"}],
                      counts={"l": {"accepted": 10}}, enumerator_hash="e", packed_manifest_hashes={})
    b = build_overlay([{"layer": "l", "partition": "train", "trace_key": "y"}],
                      counts={"l": {"accepted": 10}}, enumerator_hash="e", packed_manifest_hashes={})
    assert a["effective_corpus_checksum"] != b["effective_corpus_checksum"]


def test_effective_corpus_checksum_changes_with_accepted_counts():
    a = build_overlay([], counts={"l": {"accepted": 10}}, enumerator_hash="e", packed_manifest_hashes={})
    b = build_overlay([], counts={"l": {"accepted": 11}}, enumerator_hash="e", packed_manifest_hashes={})
    assert a["effective_corpus_checksum"] != b["effective_corpus_checksum"]
