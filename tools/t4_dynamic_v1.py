#!/usr/bin/env python3
"""Zero-oracle reachability gate for the scoped Dynamic COMPOSE v1 study."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import deque
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.chem.molecular_graph import IDX_TO_ELEMENT
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v1 import (
    CarbonylInsertRequest,
    CycleOpenRequest,
    FunctionalizeRequest,
    PendantDeleteRequest,
    RingPathRemodelRequest,
    RingSubstituentAtom,
    RingSystemRestateRequest,
    SubstitutedRingRequest,
    compile_forced_module_sequence,
)
from compose_v4.control.ring_program import RingSpec
from compose_v4.experiments.whole_ring_plan import RingRequest
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sealed(path: Path) -> tuple[dict, dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            raw = handle.read()
    else:
        raw = path.read_text()
    value = json.loads(raw)
    if set(value) != {"payload", "payload_sha256"}:
        raise ValueError(f"{path}: expected one sealed payload envelope")
    if identity(value["payload"]) != value["payload_sha256"]:
        raise ValueError(f"{path}: sealed payload identity does not reproduce")
    return value["payload"], {
        "path": str(path),
        "sha256": _sha256(path),
        "payload_sha256": value["payload_sha256"],
    }


def _external_boundary(source, fragment: tuple[int, ...]) -> set[int]:
    removed = set(fragment)
    return {
        int(neighbor)
        for slot in removed
        for neighbor in np.flatnonzero(source.bonds[slot])
        if int(neighbor) not in removed
    }


def _tree_path(
    adjacency: dict[int, set[int]], start: int, goal: int
) -> tuple[int, ...]:
    queue, parent = deque([start]), {start: None}
    while queue:
        slot = queue.popleft()
        if slot == goal:
            break
        for neighbor in sorted(adjacency.get(slot, ())):
            if neighbor not in parent:
                parent[neighbor] = slot
                queue.append(neighbor)
    if goal not in parent:
        raise ValueError("ring closure endpoints lack a created-atom tree path")
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    return tuple(reversed(path))


def _jak2_requests(candidate: dict):
    source = decode_state(candidate["source_state"])
    actions = candidate["trace"]["actions"]
    prefix = 0
    while prefix < len(actions) and actions[prefix]["executor_rule"] == "atom_delete":
        prefix += 1
    first_close = next(
        index
        for index, record in enumerate(actions[prefix:], start=prefix)
        if record["executor_rule"] == "cycle_close"
    )
    deleted = tuple(record["payload"]["v"] for record in actions[:prefix])
    boundary = _external_boundary(source, deleted)
    if len(boundary) != 1:
        raise ValueError("JAK2 teacher prefix is not one bounded pendant deletion")
    delete_request = PendantDeleteRequest(deleted, next(iter(boundary)))

    insertions = {
        record["payload"]["slot"]: record
        for record in actions[prefix:first_close]
        if record["executor_rule"] == "atom_insert"
    }
    adjacency = {slot: set() for slot in insertions}
    external = []
    for slot, record in insertions.items():
        for neighbor, order in record["payload"]["neighbors"]:
            if neighbor in insertions:
                adjacency[slot].add(neighbor)
                adjacency[neighbor].add(slot)
            else:
                external.append((slot, neighbor, order))
    close = actions[first_close]["payload"]
    path = _tree_path(adjacency, close["a"], close["b"])
    if len(path) not in (5, 6) or len(external) != 1 or external[0][0] != path[0]:
        raise ValueError("teacher ring is not one supported pendant constructed cycle")
    elements = tuple(
        IDX_TO_ELEMENT[insertions[slot]["payload"]["atom_type"]] for slot in path
    )
    pattern = tuple(
        next(
            order
            for neighbor, order in insertions[right]["payload"]["neighbors"]
            if neighbor == left
        )
        for left, right in pairwise(path)
    ) + (close["order"],)
    counts = tuple(elements.count(element) for element in ("C", "N", "O"))
    electronic = "saturated" if set(pattern) == {1} else "nonaromatic"
    ring = RingRequest(
        RingSpec("pendant", len(path), counts, electronic),
        (external[0][1],),
        elements,
        pattern,
    )
    branch_slots = []
    substituents = []
    for slot, record in insertions.items():
        if slot in path:
            continue
        parent, order = record["payload"]["neighbors"][0]
        if parent in path:
            parent_kind, parent_index = "ring", path.index(parent)
        elif parent in branch_slots:
            parent_kind, parent_index = "substituent", branch_slots.index(parent)
        else:
            raise ValueError("teacher substituent is not topologically ordered")
        substituents.append(
            RingSubstituentAtom(
                parent_kind,
                parent_index,
                IDX_TO_ELEMENT[record["payload"]["atom_type"]],
                order,
            )
        )
        branch_slots.append(slot)
    ring_request = SubstitutedRingRequest(ring, tuple(substituents))

    tail = actions[first_close + 1 :]
    if [record["executor_rule"] for record in tail] != ["atom_delete", "cycle_close"]:
        raise ValueError("JAK2 teacher tail is not one ring-path contraction")
    removed = tail[0]["payload"]["v"]
    closing = tail[1]["payload"]
    remodel = RingPathRemodelRequest(
        (closing["a"], removed, closing["b"]),
        (),
        (closing["order"],),
    )
    return source, (delete_request, ring_request, remodel)


def _five_ht_requests(candidate: dict):
    source = decode_state(candidate["source_state"])
    actions = candidate["trace"]["actions"]
    if [record["executor_rule"] for record in actions[:-1]] != ["atom_delete"] * (
        len(actions) - 1
    ) or actions[-1]["executor_rule"] != "atom_insert":
        raise ValueError("5HT1B teacher is not delete-fragment plus functionalization")
    fragment = tuple(record["payload"]["v"] for record in actions[:-1])
    boundary = _external_boundary(source, fragment)
    if len(boundary) != 1:
        raise ValueError("5HT1B teacher deletion has more than one retained boundary")
    inserted = actions[-1]["payload"]
    if len(inserted["neighbors"]) != 1:
        raise ValueError("5HT1B teacher functionalization has unexpected arity")
    anchor, order = inserted["neighbors"][0]
    direct = (
        PendantDeleteRequest(fragment, next(iter(boundary))),
        FunctionalizeRequest(anchor, IDX_TO_ELEMENT[inserted["atom_type"]], order),
    )
    # Dynamic-v0 cannot delete this pendant cycle directly because no boundary
    # edge is initially a bridge.  Opening either boundary edge is a generic
    # v0 module; exact replay determines whether it exposes the same endpoint.
    v0 = None
    for slot in fragment:
        if source.bonds[slot, next(iter(boundary))]:
            edge = tuple(sorted((slot, next(iter(boundary)))))
            attempt = (CycleOpenRequest(*edge), *direct)
            try:
                result = compile_forced_module_sequence(source, attempt)
            except ValueError:
                continue
            if result[3]["endpoint"] == candidate["endpoint"]:
                v0 = attempt
                break
    if v0 is None:
        raise ValueError("no exact three-module Dynamic-v0 teacher factorization found")
    return source, direct, v0


def _braf_requests(candidate: dict):
    actions = candidate["trace"]["actions"]
    if [record["executor_rule"] for record in actions] != [
        "atom_insert",
        "ring_system_restate",
        "ring_system_restate",
    ]:
        raise ValueError("BRAF Dynamic-v0 teacher has an unexpected action structure")
    oxygen = actions[0]["payload"]
    anchor, order = oxygen["neighbors"][0]
    if IDX_TO_ELEMENT[oxygen["atom_type"]] != "O" or order != 2:
        raise ValueError(
            "BRAF Dynamic-v0 teacher does not start with carbonyl insertion"
        )
    requests = [CarbonylInsertRequest(anchor)]
    for record in actions[1:]:
        requests.append(
            RingSystemRestateRequest(
                tuple(
                    (change["a"], change["b"], change["new_order"])
                    for change in record["payload"]["changes"]
                )
            )
        )
    return decode_state(candidate["source_state"]), tuple(requests)


def _check(
    name: str, source, requests, expected: str, *, teacher: str, v0: bool
) -> dict:
    _, program, assignment, trace, metadata = compile_forced_module_sequence(
        source, requests
    )
    exact = trace["endpoint"] == expected
    if not exact:
        raise ValueError(
            f"{name}: forced generic program did not reproduce its teacher endpoint"
        )
    return {
        "cell": name,
        "teacher": teacher,
        "teacher_endpoint_sha256": identity(expected),
        "v0_representable": v0,
        "v1_representable": True,
        "minimum_v1_modules": metadata["module_count"],
        "modules": metadata["modules"],
        "primitive_edits": len(program.marks),
        "blocks": len(program.blocks),
        "assignment_arity": len(assignment),
        "exact_endpoint_replay": exact,
        "intermediate_task_evaluations": metadata["intermediate_task_evaluations"],
        "teacher_parameters_deployed": metadata["teacher_parameters_deployed"],
    }


def reachability(args) -> dict:
    jak_result, jak_input = _read_sealed(args.full_jak2)
    ht_result, ht_result_input = _read_sealed(args.full_5ht1b)
    ht_checkpoint, ht_checkpoint_input = _read_sealed(args.full_5ht1b_checkpoint)
    braf_result, braf_input = _read_sealed(args.dynamic_braf)

    jak_candidate = jak_result["champion"]["candidate"]
    jak_source, jak_requests = _jak2_requests(jak_candidate)
    jak = _check(
        "jak2_1",
        jak_source,
        jak_requests,
        jak_candidate["endpoint"],
        teacher="Full-146 champion complete transformation",
        v0=False,
    )

    ancestry = ht_result["champion"]["candidate"]["provenance"]["metadata"][
        "construction_ancestry"
    ]
    if not ancestry:
        raise ValueError("5HT1B champion lacks its saved construction ancestry")
    precursor = ht_checkpoint["search"]["entries"][ancestry[0]["entry_id"]]
    ht_source, ht_requests, ht_v0_requests = _five_ht_requests(precursor)
    ht = _check(
        "5ht1b_0",
        ht_source,
        ht_requests,
        precursor["endpoint"],
        teacher="Full-146 precursor; later reroute/closure already use v0 families",
        v0=True,
    )
    v0_replay = compile_forced_module_sequence(ht_source, ht_v0_requests)
    ht["minimum_v0_modules"] = v0_replay[4]["module_count"]
    ht["v0_modules"] = v0_replay[4]["modules"]

    braf_candidate = braf_result["champion"]["candidate"]
    braf_source, braf_requests = _braf_requests(braf_candidate)
    braf = _check(
        "braf_1",
        braf_source,
        braf_requests,
        braf_candidate["endpoint"],
        teacher="Dynamic-v0 champion positive-control transformation",
        v0=True,
    )
    braf["minimum_v0_modules"] = braf["minimum_v1_modules"]

    rows = [braf, jak, ht]
    admitted = sum(row["v1_representable"] for row in rows)
    exact = sum(row["exact_endpoint_replay"] for row in rows)
    body = {
        "schema_version": "t4_dynamic_v1_teacher_reachability_v1",
        "scope": {
            "development_cells": ["5ht1b_0", "braf_1", "jak2_1"],
            "delta": 0.4,
            "new_oracle_calls": 0,
            "task_oracle_calls": 0,
            "role": "answer-known teacher-forced support audit; not autonomous recovery",
        },
        "inputs": {
            "full_jak2_result": jak_input,
            "full_5ht1b_result": ht_result_input,
            "full_5ht1b_checkpoint": ht_checkpoint_input,
            "dynamic_v0_braf_result": braf_input,
        },
        "implementation": {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script_sha256": _sha256(Path(__file__)),
            "dynamic_v1_sha256": _sha256(
                ROOT / "src/compose_v4/control/dynamic_program_synthesis_v1.py"
            ),
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "hardware": platform.platform(),
            "accelerator": None,
        },
        "cells": rows,
        "summary": {
            "attempted_teachers": len(rows),
            "v1_representable": admitted,
            "coverage": admitted / len(rows),
            "exact_replays": exact,
            "execution_precision": exact / admitted if admitted else None,
            "v0_representable": sum(row["v0_representable"] for row in rows),
            "v0_coverage": sum(row["v0_representable"] for row in rows) / len(rows),
        },
        "negative_findings": [
            "JAK2 is not representable by Dynamic-v0 under its three-module effective support: the two-nitrogen substituted-ring construction and protected ring contraction are absent.",
            "The BRAF support witness is Dynamic-v0's distinct champion, not an exact factorization of the longer Full-146 champion route.",
            "5HT1B coverage is established for the stored-route precursor; subsequent supported local reroute and closure steps are outside this forced precursor test.",
        ],
        "limitations": [
            "Teacher parameters were inferred retrospectively from answer-known development routes.",
            "Exact forced compilation establishes representation support, not proposal probability or autonomous discovery.",
            "Minimum module counts are with respect to the declared v0/v1 module interfaces, not a proof over every possible program factorization.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return body


def _publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-jak2", type=Path, required=True)
    parser.add_argument("--full-5ht1b", type=Path, required=True)
    parser.add_argument("--full-5ht1b-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamic-braf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reachability(args)
    _publish(args.output, result)
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
