"""Fail-closed immutable byte publication across local and Modal filesystems."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
from pathlib import Path


class ImmutableArtifactError(RuntimeError):
    """An immutable path is unsafe, collides, or cannot be verified."""


_HARD_LINK_UNSUPPORTED_ERRNOS = frozenset(
    {
        errno.EPERM,
        errno.EXDEV,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
    }
)


def _existing_artifact_matches(destination: Path, content: bytes) -> bool:
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        raise ImmutableArtifactError(
            f"immutable artifact path is not a regular file: {destination}"
        )
    if metadata.st_size != len(content):
        return False
    offset = 0
    with destination.open("rb") as handle:
        while block := handle.read(1 << 20):
            right = offset + len(block)
            if block != content[offset:right]:
                return False
            offset = right
    return offset == len(content)


def _reuse_or_reject_existing(destination: Path, content: bytes) -> bool:
    if _existing_artifact_matches(destination, content):
        return False
    raise ImmutableArtifactError(
        f"immutable artifact already exists with different bytes: {destination}"
    )


def _exclusive_create_from_bytes(destination: Path, content: bytes) -> bool:
    """Reserve the final path without overwrite when hard links are unavailable.

    ``O_EXCL`` makes final-name reservation atomic. A process crash during the
    subsequent write can leave an incomplete file, but retries reject that file
    rather than accepting or replacing it. Errors observed by the publishing
    process remove only the path that this process exclusively created.
    """

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o644)
    except FileExistsError:
        return _reuse_or_reject_existing(destination, content)

    created = True
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        if not _existing_artifact_matches(destination, content):
            raise ImmutableArtifactError(
                f"exclusive immutable publication failed verification: {destination}"
            )
        created = False
        return True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            destination.unlink(missing_ok=True)


def write_bytes_if_absent(path: str | Path, content: bytes) -> bool:
    """Publish immutable bytes, allowing exact reuse and rejecting collisions.

    A staged hard link gives atomic publication on ordinary filesystems. Modal
    Volumes reject hard links, so known capability errors fall back to an
    exclusive final-name create without weakening no-overwrite semantics.

    Returns ``True`` when this call created the artifact and ``False`` when an
    identical artifact already existed.
    """

    if not isinstance(content, bytes):
        raise TypeError("immutable artifact content must be bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
            return True
        except FileExistsError:
            return _reuse_or_reject_existing(destination, content)
        except OSError as error:
            if error.errno not in _HARD_LINK_UNSUPPORTED_ERRNOS:
                raise
            return _exclusive_create_from_bytes(destination, content)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


__all__ = ["ImmutableArtifactError", "write_bytes_if_absent"]
