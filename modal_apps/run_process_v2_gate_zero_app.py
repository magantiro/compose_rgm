"""Run Process-V2 Gate 0 as ONE bounded streaming job.  Never yet executed.

Gate 0 is not a map/reduce.  It streams one decision index once and publishes
three small artifacts, so it runs in a single container with ``max_containers=1``
and no task namespace at all.  Building a second fan-out system for it would add
a reduction boundary, a partial-generation state and a resume protocol to a job
that has none of those problems.  If a measured execution later shows the stream
is the bottleneck, that measurement is the thing that authorizes parallelism.

WHAT THIS SURFACE REFUSES
-------------------------
* a dirty or unexpected worktree -- the local entrypoint requires the exact
  clean commit before any remote object is constructed;
* a container whose files differ from the ones that were bound -- the image
  revision and the narrow implementation revision are both re-derived remotely
  and required to be equal, so provenance is owner-computed at every boundary;
* an uncommitted chunk-cache generation or rebind run -- both completions are
  the marker their producers write last, so an interrupted upstream is *absent*
  rather than partial;
* an input that grants authority anywhere at any depth;
* a stale Active8 generation -- the index must carry the live Process-V2 process
  identity and the completion the caller named.

WHAT IT PUBLISHES, AND WHAT THAT MEANS
--------------------------------------
Evidence, decision, completion, under a content-addressed run root derived from
the inputs and the bound revisions.  ``FAIL`` publishes exactly as ``PASS`` does.
Neither grants anything: every artifact carries the complete Process-V2 authority
vocabulary explicitly false, and this app never sets one true.

RESTART
-------
There is one unit of work, so resume is exact rather than partial: if the
content-addressed run root already holds a complete artifact set, it is
revalidated against a freshly recomputed snapshot and reused.  A run root holding
some but not all three artifacts is refused rather than completed in place,
because a half-written generation is the one state the immutable publisher must
never be asked to reconcile.

NOTHING RUNS ON IMPORT.  Importing this module builds an image specification and
two entry points and calls nothing.  ``describe`` prints the exact command and
the artifact roots a launch WOULD use and executes nothing; ``main`` is the
launcher, and running it is a separate, unauthorized decision.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - import-time path setup
    sys.path.insert(0, str(ROOT / "src"))

from compose_v4.data.editing_v2_process_v2_launch_binding import (  # noqa: E402
    build_implementation_revision,
    require_tracked,
    validate_implementation_revision,
)
from compose_v4.data.editing_v2_process_v2_schema import (  # noqa: E402
    AUTHORITY_FIELDS,
    authority_false_block,
    canonical_sha256,
    require_no_granted_authority,
)

REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_gate_zero_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_gate_zero"

#: The Gate-0 computation this app owns.  Its closure is derived, never listed.
GATE_ZERO_ENTRY_MODULES: tuple[str, ...] = (
    "compose_v4.experiments.editing_v2_process_v2_gate_zero",
)

#: The Process-V2 Active8 decision index, owned by the Active8 stage.  It is
#: named here and resolved at EXECUTION rather than imported at module scope, so
#: this launcher imports cleanly and refuses loudly with the exact missing name.
#: This is the declared integration seam: if the stage lands under another name,
#: exactly these two constants move.
DECISION_INDEX_MODULE = "compose_v4.data.editing_v2_process_v2_active8_index"
DECISION_INDEX_SYMBOL = "resolve_process_v2_active8_decision_index"

RUN_REQUEST_SCHEMA = "compose.editing_v2.process_v2.gate_zero_run_request"
RUN_REQUEST_SCHEMA_VERSION = 1
RUN_REQUEST_STATUS = "PROCESS_V2_GATE_ZERO_RUN_REQUEST_NO_DOWNSTREAM_AUTHORITY"
REQUEST_FILENAME = "GATE_ZERO_RUN_REQUEST.json"
IMAGE_REVISION_SCHEMA = "compose.editing_v2.process_v2.gate_zero_modal_image_revision"
IMAGE_REVISION_SCHEMA_VERSION = 1

GATE_ZERO_CPU = 4.0
GATE_ZERO_MEMORY_MB = 32768
GATE_ZERO_TIMEOUT_SECONDS = 12 * 3600
#: One streaming job.  Deliberately not a fan-out; see the module docstring.
GATE_ZERO_MAX_CONTAINERS = 1

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
)
for _source_directory in IMAGE_SOURCE_DIRECTORIES:
    image = image.add_local_dir(
        ROOT / _source_directory,
        str(REMOTE_ROOT / _source_directory),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
image = image.add_local_file(
    ROOT / LAUNCHER_SOURCE, str(REMOTE_ROOT / LAUNCHER_SOURCE), copy=True
)

app = modal.App("compose-v4-process-v2-gate-zero")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


# ---- Deterministic identity ----


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE]
    for source_directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / source_directory).rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("the Gate-0 serialized source inventory repeats a path")
    return tuple(paths)


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments), cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot establish Process-V2 Gate-0 Git identity: git {' '.join(arguments)}"
        ) from error


def local_image_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Require the exact clean commit and hash every serialized image input.

    Local only: this is the one place Git is called, because the image carries
    source files and no ``.git``.  The narrow implementation revision is separate
    and deliberately carries no commit, so a commit touching no behaviour cannot
    relocate a content-addressed artifact.
    """

    if not isinstance(expected_commit, str) or _COMMIT_RE.fullmatch(expected_commit) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "Process-V2 Gate 0 requires the exact clean committed worktree; a dirty tree "
            "cannot be bound to a scientific artifact"
        )
    sources = {
        relative: _file_sha256(root / relative) for relative in _serialized_source_paths(root)
    }
    require_tracked(sources, tracked=_git(root, "ls-files").splitlines())
    body = {
        "schema": IMAGE_REVISION_SCHEMA,
        "schema_version": IMAGE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    return {**body, "image_revision_sha256": canonical_sha256(body)}


def validate_remote_image_revision(
    value: Mapping[str, Any], *, remote_root: Path = REMOTE_ROOT
) -> dict[str, Any]:
    """Re-derive the image identity from the files in hand.  No Git is called."""

    body = {key: item for key, item in value.items() if key != "image_revision_sha256"}
    if (
        value.get("schema") != IMAGE_REVISION_SCHEMA
        or value.get("schema_version") != IMAGE_REVISION_SCHEMA_VERSION
        or value.get("worktree_clean") is not True
        or value.get("image_revision_sha256") != canonical_sha256(body)
    ):
        raise RuntimeError("the Process-V2 Gate-0 image revision disagrees")
    sources = value.get("serialized_sources")
    expected = _serialized_source_paths(Path(remote_root))
    if not isinstance(sources, dict) or set(sources) != set(expected):
        raise RuntimeError("the Process-V2 Gate-0 serialized source inventory disagrees")
    for relative, digest in sources.items():
        if _file_sha256(Path(remote_root) / relative) != digest:
            raise RuntimeError(f"a serialized Gate-0 source differs in this container: {relative}")
    return dict(value)


def authority_envelope() -> dict[str, bool]:
    """Every authority field false.  Running Gate 0 grants nothing, ever."""

    return {"training_launched": False, **authority_false_block()}


def published_filenames() -> dict[str, str]:
    """The three artifact names, read from the Gate-0 module rather than restated.

    Imported lazily so this launcher does not pull the model stack into a local
    ``modal run`` invocation, and so the names cannot drift from their owner.
    """

    from compose_v4.experiments.editing_v2_process_v2_gate_zero import (
        COMPLETION_FILENAME,
        DECISION_FILENAME,
        EVIDENCE_FILENAME,
    )

    return {
        "evidence": EVIDENCE_FILENAME,
        "decision": DECISION_FILENAME,
        "completion": COMPLETION_FILENAME,
    }


# ---- Artifact paths ----


def artifact_path(value: str, *, field: str, artifact_root: Path = ARTIFACT_ROOT) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise RuntimeError(f"{field} must be a normalized path below /artifacts")
    root = Path(artifact_root).resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"{field} resolves outside the artifact root")
    return resolved


def artifact_address(path: Path, *, artifact_root: Path = ARTIFACT_ROOT) -> str:
    relative = Path(path).resolve().relative_to(Path(artifact_root).resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def build_run_request(
    *,
    image_revision: Mapping[str, Any],
    gate_zero_revision: Mapping[str, Any],
    decision_index_revision: Mapping[str, Any],
    inputs: Mapping[str, str],
    contract_file_sha256: str,
    contract_sha256: str,
    output_artifact_prefix: str,
) -> dict[str, Any]:
    """Content-address one run from its inputs and every revision it binds."""

    for name, value in inputs.items():
        artifact_path(value, field=name)
    for name, value in (
        ("contract_file_sha256", contract_file_sha256),
        ("contract_sha256", contract_sha256),
    ):
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise RuntimeError(f"{name} must be a lowercase SHA-256")
    artifact_path(f"{output_artifact_prefix}/placeholder", field="output_artifact_prefix")
    body: dict[str, Any] = {
        "schema": RUN_REQUEST_SCHEMA,
        "schema_version": RUN_REQUEST_SCHEMA_VERSION,
        "status": RUN_REQUEST_STATUS,
        **authority_false_block(),
        "inputs": dict(sorted(inputs.items())),
        "image_revision_sha256": str(image_revision["image_revision_sha256"]),
        "gate_zero_implementation_sha256": str(gate_zero_revision["implementation_sha256"]),
        "decision_index_implementation_sha256": str(
            decision_index_revision["implementation_sha256"]
        ),
        "contract_file_sha256": contract_file_sha256,
        "contract_sha256": contract_sha256,
        "output_artifact_prefix": output_artifact_prefix,
        "python_runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }
    request = {**body, "run_identity_sha256": canonical_sha256(body)}
    require_no_granted_authority(request, label="Process-V2 Gate-0 run request")
    return request


def launch_command(
    *,
    entrypoint: str,
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    active8_run_artifact_root: str,
    expected_commit: str,
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
) -> str:
    """The exact command a launch would use.  Printing it executes nothing."""

    return " ".join(
        (
            "modal run --detach",
            f"{LAUNCHER_SOURCE}::{entrypoint}",
            f"--cache-run-artifact-root {cache_run_artifact_root}",
            f"--rebind-run-artifact-root {rebind_run_artifact_root}",
            f"--active8-run-artifact-root {active8_run_artifact_root}",
            f"--expected-commit {expected_commit}",
            f"--output-artifact-prefix {output_artifact_prefix}",
        )
    )


# ---- Remote body ----


def resolve_decision_index_factory(*, remote_root: Path = REMOTE_ROOT):
    """Import the Active8 index stage at execution, or refuse with its exact name."""

    if str(remote_root / "src") not in sys.path:
        sys.path.insert(0, str(remote_root / "src"))
    try:
        module = importlib.import_module(DECISION_INDEX_MODULE)
    except ImportError as error:
        raise RuntimeError(
            f"the Process-V2 Active8 decision index module {DECISION_INDEX_MODULE!r} is not "
            "present in this tree, so Gate 0 has nothing to consume; this launcher is "
            "implemented against the frozen index protocol and is not runnable until that "
            "stage lands"
        ) from error
    factory = getattr(module, DECISION_INDEX_SYMBOL, None)
    if factory is None:
        raise RuntimeError(
            f"{DECISION_INDEX_MODULE!r} declares no {DECISION_INDEX_SYMBOL!r}; the Gate-0 "
            "launcher binds that exact symbol"
        )
    return factory


def require_committed_inputs(
    *,
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    artifact_root: Path,
    remote_root: Path,
) -> dict[str, Any]:
    """Require both upstream generations to be COMMITTED, never merely present."""

    from compose_v4.data.editing_process_v2_admitted_source import (
        resolve_process_v2_admitted_source,
    )
    from compose_v4.data.editing_process_v2_rebind import PLAN_FILENAME
    from compose_v4.data.editing_v2_process_v2_chunk_cache import (
        load_committed_process_v2_chunk_cache_completion,
    )

    cache_completion = load_committed_process_v2_chunk_cache_completion(
        cache_run_artifact_root, artifact_root=artifact_root, repo_root=remote_root
    )
    require_no_granted_authority(cache_completion, label="the chunk-cache completion")
    rebind_root = artifact_path(
        rebind_run_artifact_root, field="rebind_run_artifact_root", artifact_root=artifact_root
    )
    plan_path = rebind_root / PLAN_FILENAME
    if not plan_path.is_file():
        raise RuntimeError(
            f"the Process-V2 rebind generation is not committed; no {PLAN_FILENAME} at "
            f"{rebind_run_artifact_root}"
        )
    plan = json.loads(plan_path.read_bytes())
    # This resolver is the committed-rebind requirement: it demands the exact
    # published plan bytes, the completion marker the reducer writes last, every
    # planned task result, and the LIVE Process-V2 identity.
    admitted = resolve_process_v2_admitted_source(
        plan, artifact_root=artifact_root, repo_root=remote_root
    )
    identity = dict(admitted.identity())
    require_no_granted_authority(identity, label="the admitted-source identity")
    return {
        "cache_completion_sha256": str(cache_completion["completion_sha256"]),
        "rebind_completion_sha256": str(admitted.completion["completion_sha256"]),
        "admitted_source_identity": identity,
    }


def gate_zero_driver(
    *,
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    active8_run_artifact_root: str,
    output_artifact_prefix: str,
    image_revision: Mapping[str, Any],
    gate_zero_revision: Mapping[str, Any],
    decision_index_revision: Mapping[str, Any],
    artifact_root: Path = ARTIFACT_ROOT,
    remote_root: Path = REMOTE_ROOT,
    reload_volume=None,
    commit_volume=None,
) -> dict[str, Any]:
    """One bounded streaming Gate-0 job, resumable by exact revalidation."""

    from compose_v4.experiments.editing_v2_process_v2_gate_zero import (
        load_process_v2_gate_zero_artifacts,
        load_process_v2_gate_zero_contract,
        run_process_v2_gate_zero,
    )

    filenames = published_filenames()

    validate_remote_image_revision(image_revision, remote_root=remote_root)
    validate_implementation_revision(gate_zero_revision, repo_root=remote_root)
    validate_implementation_revision(decision_index_revision, repo_root=remote_root)
    factory = resolve_decision_index_factory(remote_root=remote_root)

    # Boundary 1: before reading anything another container wrote.
    if reload_volume is not None:
        reload_volume()
    upstream = require_committed_inputs(
        cache_run_artifact_root=cache_run_artifact_root,
        rebind_run_artifact_root=rebind_run_artifact_root,
        artifact_root=artifact_root,
        remote_root=remote_root,
    )
    index = factory(
        active8_run_artifact_root,
        artifact_root=artifact_root,
        repo_root=remote_root,
    )
    contract = load_process_v2_gate_zero_contract(repo_root=remote_root)
    if index.process_identity_sha256 != contract.process_identity_sha256:
        raise RuntimeError(
            "the Active8 decision index was computed under process identity "
            f"{index.process_identity_sha256}, not the live Process-V2 identity "
            f"{contract.process_identity_sha256}; a stale generation is not evidence"
        )
    request = build_run_request(
        image_revision=image_revision,
        gate_zero_revision=gate_zero_revision,
        decision_index_revision=decision_index_revision,
        inputs={
            "cache_run_artifact_root": cache_run_artifact_root,
            "rebind_run_artifact_root": rebind_run_artifact_root,
            "active8_run_artifact_root": active8_run_artifact_root,
        },
        contract_file_sha256=contract.file_sha256,
        contract_sha256=contract.sha256,
        output_artifact_prefix=output_artifact_prefix,
    )
    prefix = artifact_path(
        f"{output_artifact_prefix}/placeholder",
        field="output_artifact_prefix",
        artifact_root=artifact_root,
    ).parent
    output_directory = prefix / request["run_identity_sha256"]

    present = [
        name for name in sorted(filenames.values()) if (output_directory / name).is_file()
    ]
    if present and len(present) != len(filenames):
        raise RuntimeError(
            f"the Gate-0 run root {artifact_address(output_directory, artifact_root=artifact_root)}"
            f" holds a partial artifact set {sorted(present)}; a half-written generation is "
            "refused rather than completed in place"
        )
    if present:
        # Exact reuse after an interruption: the persisted bytes are revalidated
        # against a freshly recomputed snapshot, so reuse is a proof, not a guess.
        result = load_process_v2_gate_zero_artifacts(
            output_directory=output_directory, index=index, contract=contract
        )
        reused = True
    else:
        result = run_process_v2_gate_zero(
            index,
            repo_root=remote_root,
            artifact_root=artifact_root,
            output_directory=output_directory,
            contract=contract,
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / REQUEST_FILENAME).write_bytes(
            _canonical_bytes(request, newline=True)
        )
        reused = False
    if commit_volume is not None:
        commit_volume()
    # Boundary 2: after publishing, before reporting what is durably visible.
    if reload_volume is not None:
        reload_volume()
    for name, payload in result.items():
        raw = (output_directory / filenames[name]).read_bytes()
        if raw != _canonical_bytes(payload, newline=True):
            raise RuntimeError(f"the published Gate-0 {name} differs from the returned artifact")
    report = {
        "phase": "process_v2_gate_zero_complete",
        "executed": True,
        "structural_result": result["completion"]["structural_result"],
        "reused_existing_run": reused,
        "run_identity_sha256": request["run_identity_sha256"],
        "run_artifact_root": artifact_address(output_directory, artifact_root=artifact_root),
        "evidence_sha256": result["evidence"]["evidence_sha256"],
        "decision_sha256": result["decision"]["decision_sha256"],
        "completion_sha256": result["completion"]["completion_sha256"],
        "image_revision_sha256": image_revision["image_revision_sha256"],
        "gate_zero_implementation_sha256": gate_zero_revision["implementation_sha256"],
        "decision_index_implementation_sha256": decision_index_revision["implementation_sha256"],
        **upstream,
        **authority_envelope(),
    }
    require_no_granted_authority(report, label="the Gate-0 run report")
    return report


@app.function(
    image=image,
    cpu=GATE_ZERO_CPU,
    memory=GATE_ZERO_MEMORY_MB,
    timeout=GATE_ZERO_TIMEOUT_SECONDS,
    max_containers=GATE_ZERO_MAX_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_gate_zero(
    *,
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    active8_run_artifact_root: str,
    output_artifact_prefix: str,
    image_revision: dict[str, Any],
    gate_zero_revision: dict[str, Any],
    decision_index_revision: dict[str, Any],
) -> dict[str, Any]:
    """The one bounded streaming job.  CPU only; it trains nothing."""

    return gate_zero_driver(
        cache_run_artifact_root=cache_run_artifact_root,
        rebind_run_artifact_root=rebind_run_artifact_root,
        active8_run_artifact_root=active8_run_artifact_root,
        output_artifact_prefix=output_artifact_prefix,
        image_revision=image_revision,
        gate_zero_revision=gate_zero_revision,
        decision_index_revision=decision_index_revision,
        reload_volume=artifact_volume.reload,
        commit_volume=artifact_volume.commit,
    )


# ---- Local entry points ----


@app.local_entrypoint()
def describe(
    cache_run_artifact_root: str = "/artifacts/editing_v2/process_v2_chunk_cache/<generation>",
    rebind_run_artifact_root: str = "/artifacts/editing_v2/process_v2_rebind/<run>",
    active8_run_artifact_root: str = "/artifacts/editing_v2/process_v2_active8/<run>",
    expected_commit: str = "<full-40-hex-commit>",
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
) -> None:
    """Print the exact command and artifact roots a launch WOULD use.

    This entry point executes no remote function, resolves no artifact and
    performs no Git call.  ``executed`` is false in its output because nothing
    ran, and its wording is deliberately conditional so its output can never be
    quoted as evidence that Gate 0 has been executed.
    """

    print(
        json.dumps(
            {
                "phase": "process_v2_gate_zero_launch_description",
                "executed": False,
                "status": "DESCRIPTION_ONLY_NOTHING_WAS_EXECUTED",
                "would_run_command": launch_command(
                    entrypoint="main",
                    cache_run_artifact_root=cache_run_artifact_root,
                    rebind_run_artifact_root=rebind_run_artifact_root,
                    active8_run_artifact_root=active8_run_artifact_root,
                    expected_commit=expected_commit,
                    output_artifact_prefix=output_artifact_prefix,
                ),
                "would_read_artifact_roots": {
                    "chunk_cache": cache_run_artifact_root,
                    "rebind": rebind_run_artifact_root,
                    "active8_decisions": active8_run_artifact_root,
                },
                "would_write_artifact_root": (
                    f"{output_artifact_prefix}/<run_identity_sha256>"
                ),
                "would_publish_filenames": sorted(
                    [*published_filenames().values(), REQUEST_FILENAME]
                ),
                "max_containers": GATE_ZERO_MAX_CONTAINERS,
                "execution_shape": "one_bounded_streaming_job_not_a_map_reduce",
                **authority_envelope(),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def main(
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    active8_run_artifact_root: str,
    expected_commit: str,
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
) -> None:
    """Bind the exact clean commit and both revisions, then run the one job."""

    for field, value in (
        ("cache_run_artifact_root", cache_run_artifact_root),
        ("rebind_run_artifact_root", rebind_run_artifact_root),
        ("active8_run_artifact_root", active8_run_artifact_root),
    ):
        artifact_path(value, field=field)
    image_revision = local_image_revision(expected_commit=expected_commit)
    gate_zero_revision = build_implementation_revision(GATE_ZERO_ENTRY_MODULES, repo_root=ROOT)
    decision_index_revision = build_implementation_revision(
        (DECISION_INDEX_MODULE,), repo_root=ROOT
    )
    report = run_gate_zero.remote(
        cache_run_artifact_root=cache_run_artifact_root,
        rebind_run_artifact_root=rebind_run_artifact_root,
        active8_run_artifact_root=active8_run_artifact_root,
        output_artifact_prefix=output_artifact_prefix,
        image_revision=image_revision,
        gate_zero_revision=gate_zero_revision,
        decision_index_revision=decision_index_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


__all__ = [
    "AUTHORITY_FIELDS",
    "DECISION_INDEX_MODULE",
    "DECISION_INDEX_SYMBOL",
    "GATE_ZERO_ENTRY_MODULES",
    "GATE_ZERO_MAX_CONTAINERS",
    "OUTPUT_ARTIFACT_PREFIX",
    "REQUEST_FILENAME",
    "app",
    "artifact_address",
    "artifact_path",
    "authority_envelope",
    "build_run_request",
    "describe",
    "gate_zero_driver",
    "launch_command",
    "local_image_revision",
    "main",
    "published_filenames",
    "require_committed_inputs",
    "resolve_decision_index_factory",
    "run_gate_zero",
    "validate_remote_image_revision",
]
