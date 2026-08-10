"""The durable-path gate must fire on the paths that actually bit us.

A day of work was reaped from a /private/tmp worktree mid-session. The gate
exists so that failure is caught by a check rather than by noticing that
``src/`` has quietly gone from 118 modules to 18.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from compose_v4.data.durable_path import (
    OVERRIDE_ENV,
    ReapablePathError,
    is_reapable,
    require_durable_path,
)


@pytest.mark.parametrize(
    "path",
    [
        "/private/tmp/compose-t1-collated-cache",       # the worktree that vanished
        "/tmp/anything",                                 # the symlink spelling
        "/private/var/folders/xy/T/scratch",
    ],
)
def test_reapable_paths_are_detected(path: str) -> None:
    assert is_reapable(path)


@pytest.mark.parametrize(
    "path",
    ["/Users/someone/compose_v2_work", "/Volumes/data/corpus"],
)
def test_durable_paths_are_allowed(path: str) -> None:
    assert not is_reapable(path)


def test_a_name_that_merely_starts_with_tmp_is_not_reapable() -> None:
    """/tmpfoo is not inside /tmp; prefix matching must respect the boundary."""

    assert not is_reapable("/tmpfoo/corpus")


def test_require_durable_path_raises_with_an_actionable_message() -> None:
    with pytest.raises(ReapablePathError) as caught:
        require_durable_path("/private/tmp/corpus", role="artifact root")
    message = str(caught.value)
    assert "artifact root" in message, "the error must name WHICH root is wrong"
    assert OVERRIDE_ENV in message, "the error must say how to proceed deliberately"


def test_require_durable_path_returns_a_resolved_durable_path(tmp_path_factory) -> None:
    home_like = Path.home()
    assert require_durable_path(home_like, role="repo root") == home_like.resolve()


def test_the_override_is_explicit_and_scoped(monkeypatch) -> None:
    """A throwaway probe may opt out, but only by saying so at the call."""

    monkeypatch.setenv(OVERRIDE_ENV, "1")
    assert require_durable_path("/private/tmp/probe", role="scratch") == Path(
        "/private/tmp/probe"
    ).resolve()
    monkeypatch.delenv(OVERRIDE_ENV)
    with pytest.raises(ReapablePathError):
        require_durable_path("/private/tmp/probe", role="scratch")


def test_override_absent_by_default() -> None:
    """The gate must be on unless someone turned it off for this invocation."""

    assert not os.environ.get(OVERRIDE_ENV)
