"""Refuse to treat an OS-reapable path as a source of truth.

WHY THIS IS A GATE AND NOT A CONVENTION
---------------------------------------
``/private/tmp`` and ``/var/folders`` are periodically reaped by macOS. That is
correct behaviour for scratch and catastrophic for anything that is the only
copy of something. This project learned it three times in one day:

1. the 70,301-entry ``train_65k`` corpus existed only in a scratchpad
   directory, with a partial backup, and was briefly believed not to exist;
2. 21,778 compiled entries existed only on a Modal volume;
3. the git WORKTREE holding a day of work was reaped mid-session -- ``.git``
   vanished and ``src/`` was stripped from 118 modules to 18 while a test run
   was in flight, which first presented as ``ModuleNotFoundError`` and looked
   like a test-config problem.

Nothing was ultimately lost, because worktree objects live in the parent
repository and the corpora had been replicated. But each near-miss was caught
by luck or by noticing an odd symptom, and "remember not to put important
things in /tmp" is exactly the kind of rule that fails at 00:30 after twenty
hours of work.

WHAT THIS DOES NOT CLAIM
------------------------
Durability is not binary and this cannot prove a path survives. It checks the
one property that has actually bitten: residence under a directory the OS
reaps on a schedule. A network mount that disappears, a full disk, or a
deleted parent repository are all still possible, so replication remains the
real protection. This gate only removes the failure mode we have already paid
for three times.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Prefixes the OS reaps on a schedule. ``/tmp`` is a symlink to ``/private/tmp``
#: on macOS, so both spellings are listed: a resolved path shows the latter and
#: an unresolved argument may show the former.
REAPABLE_PREFIXES: tuple[str, ...] = (
    "/private/tmp",
    "/tmp",
    "/private/var/folders",
    "/var/folders",
)

#: Set to allow a reapable path deliberately -- for a throwaway probe whose
#: output nobody will miss. It must be set explicitly per invocation, so the
#: decision is visible in the command that made it rather than inherited from
#: a shell that was configured hours ago.
OVERRIDE_ENV = "COMPOSE_ALLOW_REAPABLE_PATH"


class ReapablePathError(RuntimeError):
    """A source-of-truth path lives somewhere the OS will delete it."""


def is_reapable(path: Path | str) -> bool:
    """True when ``path`` resides under a directory the OS reaps."""

    resolved = str(Path(path).expanduser().resolve())
    return any(
        resolved == prefix or resolved.startswith(prefix.rstrip("/") + "/")
        for prefix in REAPABLE_PREFIXES
    )


def require_durable_path(path: Path | str, *, role: str) -> Path:
    """Return ``path``, or raise if it is somewhere the OS will delete.

    ``role`` names what the path is for, because the error is only actionable
    if it says which of several roots is the problem.
    """

    resolved = Path(path).expanduser().resolve()
    if not is_reapable(resolved) or os.environ.get(OVERRIDE_ENV):
        return resolved
    raise ReapablePathError(
        f"the {role} is {resolved}, which the OS reaps on a schedule.\n"
        "  A day of work was lost from such a path once: the git worktree was "
        "deleted mid-session and src/ went from 118 modules to 18.\n"
        "  Move it under a durable location (a home directory or an explicit "
        "data volume), or set "
        f"{OVERRIDE_ENV}=1 if this output is genuinely disposable."
    )
