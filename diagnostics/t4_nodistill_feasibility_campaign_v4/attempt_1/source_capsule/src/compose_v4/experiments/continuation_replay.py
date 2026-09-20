"""Replay complete recorded rows, with an explicit stop at unknown transitions.

No molecular enumerator, model, or executor is called. This module does not
certify these records as a cross-revision live kernel cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compose_v4.control.continuation import ContinuationBudgetExceeded, ReferenceRow
from compose_v4.control.sampled_continuation import (
    SampledContinuation,
    sampled_continuation_decision,
)
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file


class RecordedRowMissing(ContinuationBudgetExceeded):
    """An unknown recorded continuation, not zero probability or invalid chemistry."""


class RecordedProfile:
    def __init__(self, directory: Path, inventory: dict):
        self.directory = directory
        self.snapshot_id = inventory["metadata"]["launch"]["run_id"]
        self.rows = {}
        roots = []
        self.lookups, self.empty_row_hits = 0, 0
        self.missing = []
        for relative, expected_hash in sorted(inventory["input"]["sha256"].items()):
            if not relative.startswith("rows/"):
                continue
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe recorded row path: {relative}")
            path = directory / relative_path
            if sha256_file(path) != expected_hash:
                raise ValueError(f"recorded row physical hash mismatch: {path}")
            payload = json.loads(path.read_text())
            if (
                payload["schema_version"] != "continuation_profile_row_v1"
                or payload["snapshot_id"] != self.snapshot_id
            ):
                raise ValueError(f"recorded row schema/snapshot mismatch: {path}")
            if hashlib.sha256(canonical_bytes(payload)).hexdigest() != path.stem:
                raise ValueError(f"recorded row content-address mismatch: {path}")
            row = ReferenceRow(tuple(payload["successors"]), tuple(payload["probabilities"]))
            if len(payload["marks"]) != len(row.successors):
                raise ValueError(f"recorded row marks are not aligned: {path}")
            source = payload["source"]
            key = canonical_bytes(source)
            if key in self.rows:
                raise ValueError(f"duplicate recorded source state: {path}")
            for successor in row.successors:
                if successor["step"] != source["step"] + 1 or any(
                    successor[field] != source[field]
                    for field in ("bundle_id", "option", "horizon", "origin")
                ):
                    raise ValueError(f"recorded successor changed its fixed bundle/program: {path}")
            self.rows[key] = row
            if source["step"] == 0:
                roots.append(source)
        if len(roots) != 1:
            raise ValueError("recorded profile requires exactly one complete root")
        self.root = roots[0]

    def row(self, state: dict) -> ReferenceRow:
        self.lookups += 1
        key = canonical_bytes(state)
        if key not in self.rows:
            self.missing.append(
                {
                    "state_sha256": hashlib.sha256(key).hexdigest(),
                    "step": state["step"],
                    "bundle_id": state["bundle_id"],
                }
            )
            raise RecordedRowMissing("complete row absent from recorded profile")
        row = self.rows[key]
        self.empty_row_hits += int(not row.successors)
        return row


def replay(profile: RecordedProfile, config: dict) -> dict:
    from dataclasses import asdict

    from compose_v4.control.graph_geometry import topology
    from compose_v4.rewrite.trace_shard import decode_state

    def terminal(state):
        origin, final = (
            topology(decode_state(state["origin"])),
            topology(decode_state(state["graph"])),
        )
        return float(
            state["step"] == state["horizon"]
            and final["cycle_rank"] > origin["cycle_rank"]
            and final["n_ring_systems"] > origin["n_ring_systems"]
        )

    root = profile.row(profile.root)
    estimator = SampledContinuation(
        profile.row,
        terminal,
        canonical_bytes,
        snapshot_id=profile.snapshot_id,
        seed=config["seed"],
        samples_per_successor=config["samples_per_successor"],
        alpha=config["alpha"],
        max_expansions=config["max_expansions"],
        max_terminal_evaluations=config["max_terminal_evaluations"],
        max_rollouts=config["max_rollouts"],
    )
    result = sampled_continuation_decision(
        root,
        profile.root["horizon"],
        estimator,
        fallback_values=[1] * len(root.successors),
        kappa=config["kappa"],
        exploration=config["exploration"],
    )
    status = "incomplete_recorded_graph" if profile.missing else result.decision.status
    if profile.missing and (
        result.decision.successor_values is not None or result.estimate is not None
    ):
        raise RuntimeError("incomplete recorded graph produced partial guidance")
    return {
        "status": status,
        "decision": asdict(result),
        "reference_root": root.probabilities,
        "work": asdict(estimator.work),
        "recorded_row_lookups": profile.lookups,
        "known_empty_row_hits": profile.empty_row_hits,
        "missing_rows": profile.missing,
        "new_executor_calls": 0,
        "new_oracle_calls": 0,
    }
