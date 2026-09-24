"""Fail-closed checks for the frozen linker development analysis."""

from __future__ import annotations

import pytest

from tools.analyze_fragment_linker_path_binding_dev import (
    _qualified_count,
    _require_frozen_prefix,
)


def test_official_percentage_reconstructs_an_integer_attempt_count() -> None:
    assert _qualified_count({"attempts": 10, "official": {"quality": 30.0}}) == 3
    with pytest.raises(ValueError, match="nonintegral qualified count"):
        _qualified_count({"attempts": 10, "official": {"quality": 35.0}})


def test_census_control_must_replay_the_frozen_shard_prefix() -> None:
    old = {"emitted_samples": ["", "CC", "CO"]}
    frozen = {"emitted_samples": ["", "CC", "CO", "CN"]}
    # The production gate fixes ten attempts. A short synthetic vector still
    # exercises the exact prefix comparison and its mismatch behavior.
    assert _require_frozen_prefix(old, frozen, "fixture") is None
    with pytest.raises(ValueError, match="does not replay frozen"):
        _require_frozen_prefix(old, {"emitted_samples": ["", "CN", "CO"]}, "fixture")
