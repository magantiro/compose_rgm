"""Route-free joint constituent and STOP ranking for bounded program support.

This module is a standalone proposal-support diagnostic.  It owns no route,
teacher endpoint, task identity, objective value, or learned parameter.  Every
constituent is enumerated from the current exact molecular state through the
existing generic Dynamic-v0/v1 compilers.  Complete programs are replayed from
their source before they may enter a beam or a candidate lock.

The two arms deliberately share support and work limits:

``independent_marginal``
    Add constituent-local work/displacement costs and use an independent
    depth-two STOP preference.  This is the collapse-prone control.

``coverage_joint_stop``
    Rank the completed joint object by executor/resource feasibility and retain
    round-robin coverage over scale, heavy-atom change, cycle-rank change,
    retained-interface class, and rewrite mode.  The ordering is lexicographic,
    not a fitted or route-frequency-weighted score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import pairwise

import numpy as np

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, MolecularGraph, is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v1 import (
    BoundModuleCandidate,
    CarbonylInsertRequest,
    CycleOpenRequest,
    compile_carbonyl_insert,
    compile_cycle_open,
    enumerate_functionalizations,
    enumerate_pendant_deletions,
    enumerate_ring_path_remodels,
    enumerate_substituted_rings,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
    program_size_profile,
)
from compose_v4.control.graph_geometry import _bridges, atom_environment, topology
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

SCHEMA = "nodistill_joint_constituent_stop_v1"
ARMS = ("independent_marginal", "coverage_joint_stop")


@dataclass(frozen=True)
class JointStopConfig:
    """Frozen route-free support and beam budget."""

    seed: int = 20260919
    min_depth: int = 2
    max_depth: int = 4
    beam_width: int = 24
    lock_width: int = 32
    candidates_per_family: int = 6
    expansions_per_parent: int = 24
    max_primitives: int = 32
    max_blocks: int = 8
    max_active_atoms: int = 40

    def __post_init__(self) -> None:
        integer_fields = (
            "seed",
            "min_depth",
            "max_depth",
            "beam_width",
            "lock_width",
            "candidates_per_family",
            "expansions_per_parent",
            "max_primitives",
            "max_blocks",
            "max_active_atoms",
        )
        if any(type(getattr(self, field)) is not int for field in integer_fields):
            raise ValueError("joint STOP configuration requires integer limits")
        if self.seed < 0 or not 2 <= self.min_depth <= self.max_depth <= 4:
            raise ValueError("joint STOP depth support is exactly a subset of 2..4")
        if (
            min(
                self.beam_width,
                self.lock_width,
                self.candidates_per_family,
                self.expansions_per_parent,
            )
            < 1
        ):
            raise ValueError("joint STOP beam and enumeration limits must be positive")
        if (
            self.max_primitives != 32
            or self.max_blocks != 8
            or self.max_active_atoms != 40
        ):
            raise ValueError(
                "joint STOP may not change 32-primitive/8-block/40-active support"
            )


@dataclass(frozen=True)
class _Constituent:
    product: MolecularGraph
    stage: dict
    family: str
    endpoint: str
    local_primitives: int
    local_delta_heavy: int
    local_delta_cycle: int
    retained_interface: str
    interface_slots: tuple[int, ...]
    rewrite_mode: str


@dataclass(frozen=True)
class _Prefix:
    graph: MolecularGraph
    stages: tuple[dict, ...]
    constituents: tuple[_Constituent, ...]
    record: dict | None = None


def _seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(sha256(text).digest()[:8], "big")


def _real_slots(graph: MolecularGraph) -> tuple[int, ...]:
    return tuple(int(slot) for slot in np.flatnonzero(is_element(graph.atom_types)))


def _interface(stage: dict) -> tuple[str, tuple[int, ...]]:
    parameters = stage["parameters"]
    if "retained_boundaries" in parameters:
        slots = tuple(sorted(int(value) for value in parameters["retained_boundaries"]))
        return "two_boundary", slots
    if "retained_anchor" in parameters:
        slot = int(parameters["retained_anchor"])
        return "one_boundary", (slot,)
    ring = parameters.get("ring")
    if isinstance(ring, dict) and ring.get("anchors"):
        slots = tuple(sorted(int(value) for value in ring["anchors"]))
        return ("one_boundary" if len(slots) == 1 else "two_boundary"), slots
    if "anchor" in parameters:
        slot = int(parameters["anchor"])
        return "one_boundary", (slot,)
    operands = []
    for field in ("a", "b", "u", "v"):
        if field in parameters and type(parameters[field]) is int:
            operands.append(int(parameters[field]))
    if operands:
        slots = tuple(sorted(set(operands)))
        return ("one_boundary" if len(slots) == 1 else "multi_boundary"), slots
    changes = parameters.get("changes", ())
    if changes:
        slots = tuple(sorted({int(value) for row in changes for value in row[:2]}))
        return "multi_boundary", slots
    return "implicit", ()


def _rewrite_mode(family: str, delta_heavy: int, delta_cycle: int) -> str:
    if delta_cycle:
        return "topology"
    if delta_heavy < 0:
        return "shrink"
    if delta_heavy > 0:
        return "grow"
    if family in ("bond_reroute", "ring_system_restate", "heteroatom_substitute"):
        return "restate"
    return "local_rewrite"


def _constituent(
    source: MolecularGraph, candidate: BoundModuleCandidate
) -> _Constituent:
    family = str(candidate.stage["name"])
    before, after = topology(source), topology(candidate.product)
    interface, slots = _interface(candidate.stage)
    delta_heavy = int(after["n_heavy"] - before["n_heavy"])
    delta_cycle = int(after["cycle_rank"] - before["cycle_rank"])
    return _Constituent(
        product=candidate.product,
        stage=candidate.stage,
        family=family,
        endpoint=canonical_state_key(candidate.product),
        local_primitives=len(candidate.stage["actions"]),
        local_delta_heavy=delta_heavy,
        local_delta_cycle=delta_cycle,
        retained_interface=interface,
        interface_slots=slots,
        rewrite_mode=_rewrite_mode(family, delta_heavy, delta_cycle),
    )


def _bound(product: MolecularGraph, stage: dict) -> BoundModuleCandidate:
    return BoundModuleCandidate(product, stage, (canonical_state_key(product),))


def _cycle_open_candidates(
    source: MolecularGraph, *, limit: int
) -> tuple[BoundModuleCandidate, ...]:
    rows = []
    bridges = _bridges(source)
    for left in _real_slots(source):
        for right in _real_slots(source):
            if (
                left >= right
                or not source.bonds[left, right]
                or (left, right) in bridges
            ):
                continue
            try:
                product, stage = compile_cycle_open(
                    source, CycleOpenRequest(left, right)
                )
            except ValueError:
                continue
            rows.append(_bound(product, stage))
            if len(rows) >= limit:
                return tuple(rows)
    return tuple(rows)


def _carbonyl_candidates(source: MolecularGraph) -> tuple[BoundModuleCandidate, ...]:
    rows = []
    for slot in _real_slots(source):
        if (
            int(source.atom_types[slot]) != ELEMENT_TO_IDX["C"]
            or int(source.implicit_h_counts[slot]) < 2
        ):
            continue
        try:
            product, stage = compile_carbonyl_insert(
                source, CarbonylInsertRequest(slot)
            )
        except ValueError:
            continue
        rows.append(_bound(product, stage))
    return tuple(rows)


def enumerate_route_free_constituents(
    source: MolecularGraph,
    *,
    seed: int,
    candidates_per_family: int,
    total_limit: int,
) -> tuple[_Constituent, ...]:
    """Enumerate bounded generic choices from only the supplied exact state."""
    if type(seed) is not int or seed < 0:
        raise ValueError("constituent seed must be a nonnegative integer")
    if any(
        type(value) is not int or value < 1
        for value in (candidates_per_family, total_limit)
    ):
        raise ValueError("constituent enumeration limits must be positive integers")
    state_id = identity(encode_state(source))
    panels: dict[str, tuple[BoundModuleCandidate, ...]] = {
        "substituent_delete": enumerate_pendant_deletions(
            source, max_candidates=candidates_per_family
        ),
        "functionalize": enumerate_functionalizations(
            source, max_candidates=candidates_per_family
        ),
        "ring_path_remodel": enumerate_ring_path_remodels(
            source, max_candidates=candidates_per_family
        ),
        "cycle_open": _cycle_open_candidates(source, limit=candidates_per_family),
        "carbonyl_insert": _carbonyl_candidates(source)[:candidates_per_family],
    }
    ring_rng = np.random.default_rng(_seed(seed, state_id, "substituted_ring"))
    panels["construct_substituted_ring"] = enumerate_substituted_rings(
        source,
        ring_rng,
        max_candidates=candidates_per_family,
        max_trials=max(12, 4 * candidates_per_family),
    )
    ordered = {
        family: tuple(
            sorted(
                (_constituent(source, candidate) for candidate in candidates),
                key=lambda row: (row.endpoint, row.family),
            )
        )
        for family, candidates in panels.items()
    }
    families = tuple(sorted(ordered))
    selected, seen_endpoints = [], set()
    offset = 0
    while len(selected) < total_limit:
        added = False
        for family in families:
            rows = ordered[family]
            if offset >= len(rows):
                continue
            row = rows[offset]
            if row.endpoint not in seen_endpoints:
                selected.append(row)
                seen_endpoints.add(row.endpoint)
                added = True
                if len(selected) >= total_limit:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


def _changed_originals(source: MolecularGraph, product: MolecularGraph) -> int:
    initial = set(_real_slots(source))
    final = set(_real_slots(product))
    return sum(
        atom_environment(source, slot) != atom_environment(product, slot)
        for slot in sorted(initial & final)
    ) + len(initial - final)


def _scale(primitives: int, changed_originals: int) -> str:
    extent = max(primitives, changed_originals)
    if extent <= 4:
        return "small"
    if extent <= 8:
        return "medium"
    return "large"


def _heavy_band(delta: int) -> str:
    if delta <= -4:
        return "major_shrink"
    if delta < 0:
        return "shrink"
    if delta == 0:
        return "stable"
    if delta < 4:
        return "grow"
    return "major_grow"


def _cycle_band(delta: int) -> str:
    return "decrease" if delta < 0 else "increase" if delta > 0 else "stable"


def _interface_class(constituents: tuple[_Constituent, ...]) -> str:
    values = tuple(sorted({row.retained_interface for row in constituents}))
    return values[0] if len(values) == 1 else "mixed"


def _mode_class(constituents: tuple[_Constituent, ...]) -> str:
    values = tuple(sorted({row.rewrite_mode for row in constituents}))
    return values[0] if len(values) == 1 else "mixed"


def _joint_features(
    source: MolecularGraph,
    product: MolecularGraph,
    constituents: tuple[_Constituent, ...],
    *,
    primitives: int,
    blocks: int,
    size_profile: dict,
) -> dict:
    before, after = topology(source), topology(product)
    changed = _changed_originals(source, product)
    continuity = sum(
        bool(set(left.interface_slots) & set(right.interface_slots))
        for left, right in pairwise(constituents)
        if left.interface_slots and right.interface_slots
    )
    delta_heavy = int(after["n_heavy"] - before["n_heavy"])
    delta_cycle = int(after["cycle_rank"] - before["cycle_rank"])
    strata = {
        "scale": _scale(primitives, changed),
        "delta_heavy": _heavy_band(delta_heavy),
        "delta_cycle": _cycle_band(delta_cycle),
        "retained_interfaces": _interface_class(constituents),
        "rewrite_mode": _mode_class(constituents),
    }
    return {
        "depth": len(constituents),
        "primitives": primitives,
        "blocks": blocks,
        "source_heavy": int(before["n_heavy"]),
        "endpoint_heavy": int(after["n_heavy"]),
        "delta_heavy": delta_heavy,
        "delta_cycle": delta_cycle,
        "changed_originals": changed,
        "peak_heavy": int(size_profile["peak_heavy_atoms"]),
        "interface_continuity": int(continuity),
        "distinct_interfaces": len({row.retained_interface for row in constituents}),
        "distinct_rewrite_modes": len({row.rewrite_mode for row in constituents}),
        "strata": strata,
    }


def _rank_record(record: dict, arm: str) -> tuple:
    features = record["features"]
    if arm == "independent_marginal":
        local = record["constituents"]
        return (
            abs(features["depth"] - 2),
            sum(row["local_primitives"] for row in local),
            sum(abs(row["local_delta_heavy"]) for row in local),
            sum(abs(row["local_delta_cycle"]) for row in local),
            tuple(row["family"] for row in local),
            record["endpoint"],
        )
    if arm != "coverage_joint_stop":
        raise ValueError(f"unknown joint STOP arm: {arm!r}")
    return (
        int(features["peak_heavy"] == 40),
        -features["interface_continuity"],
        -features["distinct_rewrite_modes"],
        -features["distinct_interfaces"],
        features["primitives"],
        features["blocks"],
        record["endpoint"],
    )


def _stratum_key(record: dict) -> tuple[str, ...]:
    strata = record["features"]["strata"]
    return tuple(
        strata[field]
        for field in (
            "scale",
            "delta_heavy",
            "delta_cycle",
            "retained_interfaces",
            "rewrite_mode",
        )
    )


def _retain(records: list[dict], arm: str, width: int) -> list[dict]:
    unique: dict[str, dict] = {}
    for record in records:
        endpoint = record["endpoint"]
        if endpoint not in unique or _rank_record(record, arm) < _rank_record(
            unique[endpoint], arm
        ):
            unique[endpoint] = record
    pool = list(unique.values())
    if arm == "independent_marginal":
        return sorted(pool, key=lambda row: _rank_record(row, arm))[:width]
    groups: dict[tuple[str, ...], list[dict]] = {}
    for record in pool:
        groups.setdefault(_stratum_key(record), []).append(record)
    for rows in groups.values():
        rows.sort(key=lambda row: _rank_record(row, arm))
    selected = []
    while groups and len(selected) < width:
        order = sorted(
            groups,
            key=lambda key: (_rank_record(groups[key][0], arm), key),
        )
        for key in order:
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(selected) >= width:
                break
    return selected


def _serialize_constituent(row: _Constituent) -> dict:
    return {
        "family": row.family,
        "local_primitives": row.local_primitives,
        "local_delta_heavy": row.local_delta_heavy,
        "local_delta_cycle": row.local_delta_cycle,
        "retained_interface": row.retained_interface,
        "rewrite_mode": row.rewrite_mode,
    }


def _complete_prefix(
    source: MolecularGraph,
    constituents: tuple[_Constituent, ...],
    config: JointStopConfig,
) -> tuple[MolecularGraph, dict]:
    stages = [row.stage for row in constituents]
    program, assignment = extract_program(source, stages)
    if (
        len(program.marks) > config.max_primitives
        or len(program.blocks) > config.max_blocks
    ):
        raise ValueError("complete program exceeds primitive or block support")
    graph = compile_program_graph(program)
    size_profile = program_size_profile(graph, source.n_real_atoms)
    if size_profile["peak_heavy_atoms"] > config.max_active_atoms:
        raise ValueError("complete program exceeds active-atom support")
    product, trace = execute_program_graph(
        source,
        graph,
        assignment,
        max_primitives=config.max_primitives,
        max_blocks=config.max_blocks,
    )
    expected = canonical_state_key(constituents[-1].product)
    endpoint = canonical_state_key(product)
    if endpoint != expected:
        raise RuntimeError(
            "joint constituent program changed on exact protected replay"
        )
    features = _joint_features(
        source,
        product,
        constituents,
        primitives=len(program.marks),
        blocks=len(program.blocks),
        size_profile=size_profile,
    )
    payload = {
        "endpoint": endpoint,
        "program": program.payload(),
        "assignment": list(assignment),
        "trace_identity": identity(trace),
        "primitive_trace_endpoint": trace["endpoint"],
        "actual_changes": trace["actual_changes"],
        "constituents": [_serialize_constituent(row) for row in constituents],
        "features": features,
        "stop": True,
        "exact_execution": True,
    }
    return product, {**payload, "candidate_id": identity(payload)}


def generate_candidate_lock(
    source: MolecularGraph,
    *,
    arm: str,
    config: JointStopConfig,
) -> dict:
    """Generate a deterministic candidate lock from the graph alone.

    No evaluator, task/cell label, endpoint target, route, teacher action, or
    complete-program template is accepted by this runtime boundary.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown joint STOP arm: {arm!r}")
    if not 1 <= source.n_real_atoms <= config.max_active_atoms:
        raise ValueError("source lies outside frozen active-atom support")
    source_id = identity(encode_state(source))
    beam = [_Prefix(source, (), ())]
    stop_pool: list[dict] = []
    levels = []
    rejected: dict[str, int] = {}
    for depth in range(1, config.max_depth + 1):
        children: list[_Prefix] = []
        enumerated = 0
        for parent in beam:
            options = enumerate_route_free_constituents(
                parent.graph,
                seed=_seed(config.seed, source_id, depth),
                candidates_per_family=config.candidates_per_family,
                total_limit=config.expansions_per_parent,
            )
            enumerated += len(options)
            for option in options:
                constituents = (*parent.constituents, option)
                try:
                    product, record = _complete_prefix(source, constituents, config)
                except ValueError as error:
                    reason = str(error)
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                child = _Prefix(
                    product,
                    (*parent.stages, option.stage),
                    constituents,
                    record,
                )
                children.append(child)
                if depth >= config.min_depth:
                    stop_pool.append(record)
        retained_records = _retain(
            [child.record for child in children if child.record is not None],
            arm,
            config.beam_width,
        )
        retained_ids = {record["candidate_id"] for record in retained_records}
        by_id = {
            child.record["candidate_id"]: child for child in children if child.record
        }
        beam = [by_id[candidate_id] for candidate_id in retained_ids]
        beam.sort(key=lambda row: _rank_record(row.record, arm))
        levels.append(
            {
                "depth": depth,
                "parents": 1 if depth == 1 else levels[-1]["retained"],
                "enumerated": enumerated,
                "completed": len(children),
                "retained": len(beam),
                "retained_strata": len(
                    {_stratum_key(row.record) for row in beam if row.record is not None}
                ),
            }
        )
        if not beam:
            break
    locked = _retain(stop_pool, arm, config.lock_width)
    body = {
        "schema_version": SCHEMA,
        "arm": arm,
        "source_id": source_id,
        "config": asdict(config),
        "runtime_inputs": {
            "source_graph": True,
            "task_or_cell_name": False,
            "objective_or_docking_score": False,
            "route_or_template_identity": False,
            "teacher_action_or_endpoint": False,
            "absolute_route_address": False,
            "learned_weights": False,
        },
        "support": {
            "max_primitives": config.max_primitives,
            "max_blocks": config.max_blocks,
            "max_active_atoms": config.max_active_atoms,
            "depths": [config.min_depth, config.max_depth],
        },
        "levels": levels,
        "rejections": dict(sorted(rejected.items())),
        "candidates": locked,
        "oracle_calls": 0,
        "teacher_loaded_during_generation": False,
    }
    return {**body, "lock_id": identity(body)}


def strata_census(lock: dict) -> dict:
    rows = lock["candidates"]
    axes = (
        "scale",
        "delta_heavy",
        "delta_cycle",
        "retained_interfaces",
        "rewrite_mode",
    )
    return {
        axis: sorted({row["features"]["strata"][axis] for row in rows}) for axis in axes
    } | {"joint_strata": len({_stratum_key(row) for row in rows})}
