#!/usr/bin/env python
"""Content-addressed plan driver for the Process-V2 rebind.

The rebind reuses a completed V1 semantic migration as **immutable chemical
data** and proves each record against the Process-V2 admission authority.  This
script freezes the plan that names that work.  It runs locally, writes at most
the plan document, and grants nothing: no Modal job, no Active8 materialization,
no Gate 0, no T1, no P50, and no training.

Three things it does that a caller must not improvise:

**The pinned identities are read from the payload, not computed live.**  The
historical V1 migration was built under the *superseded* V1 process identity, so
computing `editing_v2_process_identity()` now would produce today's value and
`bind_v1_semantic_payload` would refuse every task.  The driver reads the
identity objects the immutable receipts actually carry, validates them for
self-consistency, and requires the process identity to equal the recorded
superseded value.  `--expected-v1-process-identity` makes that requirement
explicit rather than implicit.

**`entries_per_task` is a data decision, never a fleet decision.**  It is folded
into `run_identity_sha256`, so the published run root moves with it: planning the
same payload for 20 workers and for 1 worker would address two different runs.
It therefore defaults to the library's `DEFAULT_ENTRIES_PER_TASK` and must be set
from the payload, not from how many containers happen to be available.  Worker
count is chosen at map time and is deliberately not an input here.

**The repository must be clean and committed.**  `source_revision` hashes the
implementation boundary, so a dirty tree cannot produce a plan.

**Git runs where `.git` exists, which is never inside the image.**  The Modal
image carries source files but no `.git` metadata, and its `debian_slim` base
carries no `git` binary either, so computing the revision remotely raises
`cannot establish the rebind Git identity: git rev-parse HEAD` and planning
cannot run at all.  `build_plan` therefore accepts an already-computed
`source_revision`: the launcher computes and verifies it locally, passes it in,
and the remote side revalidates it against the image's own files through the
Git-free `validate_process_v2_rebind_source_revision`.  Omitting it keeps the
local behaviour, where `.git` really is present.

Usage::

    .venv/bin/python scripts/plan_process_v2_rebind.py \
        --artifact-root /artifacts \
        --v1-payload-root /artifacts/editing_v2/semantic_v4_migration/<run>/payload \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from compose_v4.data.editing_process_v2_rebind import (  # noqa: E402
    DEFAULT_ENTRIES_PER_TASK,
    DEFAULT_OUTPUT_ARTIFACT_PREFIX,
    PLAN_FILENAME,
    ProcessV2RebindError,
    bind_v1_semantic_payload,
    mounted_process_v2_artifact_path,
    plan_process_v2_rebind,
    repository_process_v2_rebind_source_revision,
    validate_pinned_builder_identity,
    validate_pinned_process_identity,
    validate_process_v2_rebind_source_revision,
    write_process_v2_rebind_plan,
)
from compose_v4.data.semantic_trace_migration_materializer import (  # noqa: E402
    RECEIPT_FILENAME as V1_RECEIPT_FILENAME,
)
from compose_v4.rewrite.editing_v2_process_identity import (  # noqa: E402
    SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
)

TASK_DIRNAME = "tasks"


class PlanDriverError(RuntimeError):
    """The plan cannot be built from the declared inputs."""


def read_pinned_identities(
    payload_root: Path, *, expected_process_identity_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover the identities the immutable V1 payload was built under.

    Every task receipt must carry the same pair; disagreement means the payload
    root mixes runs, which would silently bind a plan to whichever task happened
    to be read first.
    """

    tasks_dir = payload_root / TASK_DIRNAME
    if not tasks_dir.is_dir():
        raise PlanDriverError(f"V1 payload has no task directory: {tasks_dir}")
    receipts = sorted(tasks_dir.glob(f"*/{V1_RECEIPT_FILENAME}"))
    if not receipts:
        raise PlanDriverError(f"V1 payload has no task receipts: {tasks_dir}")

    process_identity: dict[str, Any] | None = None
    builder_identity: dict[str, Any] | None = None
    for path in receipts:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        observed_process = receipt.get("process_identity")
        observed_builder = receipt.get("builder_identity")
        if not isinstance(observed_process, dict) or not isinstance(observed_builder, dict):
            raise PlanDriverError(f"V1 receipt does not carry both identities: {path}")
        if process_identity is None:
            process_identity, builder_identity = observed_process, observed_builder
            continue
        if observed_process != process_identity or observed_builder != builder_identity:
            raise PlanDriverError(
                "V1 payload task receipts disagree on their pinned identities; the "
                f"payload root mixes migrations: {path}"
            )

    assert process_identity is not None and builder_identity is not None
    # Self-consistency only. `validate_frozen_process_identity` deliberately
    # reads no file and proves no currency, which is exactly right here: this
    # identity is expected to be superseded.
    validate_pinned_process_identity(process_identity)
    validate_pinned_builder_identity(builder_identity)

    observed = str(process_identity.get("process_identity_sha256"))
    if observed != expected_process_identity_sha256:
        raise PlanDriverError(
            "V1 payload was built under process identity "
            f"{observed}, not the expected {expected_process_identity_sha256}. "
            "Refusing to plan against an unexpected historical process."
        )
    return process_identity, builder_identity


def resolve_source_revision(
    source_revision: Mapping[str, Any] | None,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate a supplied revision, or compute one from a local Git checkout.

    ``validate_process_v2_rebind_source_revision`` rehashes every serialized
    implementation file against ``repo_root`` and checks the self-hash, so a
    supplied revision is bound to the code that is actually present.  It calls
    no Git, which is what makes it usable inside the image.
    """

    if source_revision is not None:
        return validate_process_v2_rebind_source_revision(source_revision, repo_root=repo_root)
    return repository_process_v2_rebind_source_revision(repo_root=repo_root)


def build_plan(
    *,
    artifact_root: Path,
    v1_payload_root_artifact_path: str,
    expected_process_identity_sha256: str,
    output_artifact_prefix: str,
    entries_per_task: int,
    source_revision: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Freeze the content-addressed plan. Writes nothing."""

    payload_root = mounted_process_v2_artifact_path(
        v1_payload_root_artifact_path,
        artifact_root=artifact_root,
        field="v1_payload_root_artifact_path",
    )
    process_identity, builder_identity = read_pinned_identities(
        payload_root,
        expected_process_identity_sha256=expected_process_identity_sha256,
    )
    revision = resolve_source_revision(source_revision, repo_root=repo_root)
    binding = bind_v1_semantic_payload(
        payload_root_artifact_path=v1_payload_root_artifact_path,
        artifact_root=artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    return plan_process_v2_rebind(
        binding,
        source_revision=revision,
        repo_root=repo_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
        output_artifact_prefix=output_artifact_prefix,
        entries_per_task=entries_per_task,
    )


def plan_envelope(plan: dict[str, Any], *, written: str | None) -> dict[str, Any]:
    """The deterministic report. Every authority field is explicitly false."""

    return {
        "phase": "process_v2_rebind_plan",
        "run_identity_sha256": plan["run_identity_sha256"],
        "run_artifact_root": plan["run_artifact_root"],
        "plan_sha256": plan["plan_sha256"],
        "plan_artifact_path": written,
        "task_inventory_sha256": plan["task_inventory_sha256"],
        "entries_per_task": plan["entries_per_task"],
        "expected_task_count": plan["expected_task_count"],
        "expected_entry_count": plan["expected_entry_count"],
        "pinned_process_identity_sha256": plan["pinned_process_identity"][
            "process_identity_sha256"
        ],
        "process_v2_identity_sha256": plan["process_v2_identity"]["process_identity_sha256"],
        "source_revision_sha256": plan["source_revision"]["source_revision_sha256"],
        "implementation_revision": plan["implementation_revision"],
        # This driver freezes work. It authorizes none of it.
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--v1-payload-root", required=True)
    parser.add_argument(
        "--expected-v1-process-identity",
        default=SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
        help="the superseded V1 process identity the payload must carry",
    )
    parser.add_argument("--output-artifact-prefix", default=DEFAULT_OUTPUT_ARTIFACT_PREFIX)
    parser.add_argument(
        "--entries-per-task",
        type=int,
        default=DEFAULT_ENTRIES_PER_TASK,
        help=(
            "records per range task. Folded into the run identity, so it names "
            "the run; it is a data decision, never a worker-count decision."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="freeze and report the plan identity without publishing the plan",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        plan = build_plan(
            artifact_root=args.artifact_root,
            v1_payload_root_artifact_path=args.v1_payload_root,
            expected_process_identity_sha256=args.expected_v1_process_identity,
            output_artifact_prefix=args.output_artifact_prefix,
            entries_per_task=args.entries_per_task,
        )
    except (PlanDriverError, ProcessV2RebindError) as error:
        print(json.dumps({"phase": "process_v2_rebind_plan_refused", "error": str(error)}))
        return 1

    written: str | None = None
    if not args.dry_run:
        write_process_v2_rebind_plan(
            plan, artifact_root=args.artifact_root, repo_root=REPO_ROOT
        )
        written = str(PurePosixPath(str(plan["run_artifact_root"])) / PLAN_FILENAME)
    print(json.dumps(plan_envelope(plan, written=written), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
