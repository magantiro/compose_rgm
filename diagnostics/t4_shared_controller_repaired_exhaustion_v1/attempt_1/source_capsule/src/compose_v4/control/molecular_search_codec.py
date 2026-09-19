"""Exact augmented search-state codec. SMILES are never a replay input."""

from dataclasses import asdict

import numpy as np

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.carbonyl_option import CarbonylProgress
from compose_v4.control.fused_option import FusedProgress
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.control.option_continuation import OptionState
from compose_v4.control.region import Region
from compose_v4.control.region_replacement import ReplacementProgress
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.control.ring_expansion import ExpansionProgress
from compose_v4.control.ring_program import RingProgress
from compose_v4.experiments.continuation_profile import state_payload
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def encode_lineage(lineage: Lineage) -> dict:
    return {
        "slot_of": [[k, v] for k, v in sorted(lineage.slot_of.items())],
        "id_of": [[k, v] for k, v in sorted(lineage.id_of.items())],
        "next_id": lineage.next_id,
    }


def decode_lineage(payload: dict) -> Lineage:
    slot_of, id_of = dict(payload["slot_of"]), dict(payload["id_of"])
    next_id = payload["next_id"]
    if (
        len(slot_of) != len(payload["slot_of"])
        or len(id_of) != len(payload["id_of"])
        or {v: k for k, v in slot_of.items()} != id_of
        or any(type(k) is not int or type(v) is not int or min(k, v) < 0 for k, v in id_of.items())
        or type(next_id) is not int
        or next_id <= max(slot_of, default=-1)
    ):
        raise ValueError("malformed exact slot lineage")
    return Lineage(slot_of, id_of, next_id)


def decode_option(payload: dict) -> OptionState:
    context = dict(payload["context"])
    context.update(
        frozen=frozenset(context["frozen"]),
        locus=frozenset(context["locus"]),
        terminals=tuple(tuple(x) for x in context["terminals"]),
    )
    return OptionState(
        graph=decode_state(payload["graph"]),
        origin=decode_state(payload["origin"]),
        context=RewriteContext(**context),
        lineage=decode_lineage(payload["lineage"]),
        option=payload["option"],
        step=payload["step"],
        horizon=payload["horizon"],
        bundle_id=payload["bundle_id"],
        fused_progress=FusedProgress.from_payload(payload["fused_progress"])
        if "fused_progress" in payload
        else None,
        expansion_progress=ExpansionProgress.from_payload(payload["expansion_progress"])
        if "expansion_progress" in payload
        else None,
        ring_progress=RingProgress.from_payload(payload["ring_progress"])
        if "ring_progress" in payload
        else None,
        carbonyl_progress=CarbonylProgress.from_payload(payload["carbonyl_progress"])
        if "carbonyl_progress" in payload
        else None,
        replacement_progress=ReplacementProgress.from_payload(payload["replacement_progress"])
        if "replacement_progress" in payload
        else None,
    )


def encode_search_state(state: MolecularSearchState) -> dict:
    region = None
    if state.region is not None:
        region = asdict(state.region)
        region["atoms"] = sorted(region["atoms"])
    return {
        "schema_version": "exact_molecular_search_state_v1",
        "graph": encode_state(state.graph),
        "lineage": encode_lineage(state.lineage),
        "budget": state.budget,
        "root_id": state.root_id,
        "stage": state.stage,
        "region": region,
        "active": state_payload(state.active) if state.active else None,
    }


def decode_search_state(payload: dict) -> MolecularSearchState:
    if payload.get("schema_version") != "exact_molecular_search_state_v1":
        raise ValueError("unknown exact molecular search-state schema")
    region = None
    if payload["region"] is not None:
        fields = dict(payload["region"])
        fields.update(
            atoms=frozenset(fields["atoms"]),
            boundary=tuple(tuple(x) for x in fields["boundary"]),
        )
        region = Region(**fields)
    node = MolecularSearchState(
        decode_state(payload["graph"]),
        decode_lineage(payload["lineage"]),
        payload["budget"],
        payload["root_id"],
        payload["stage"],
        region,
        decode_option(payload["active"]) if payload["active"] is not None else None,
    )
    if set(node.lineage.id_of) != set(np.flatnonzero(is_element(node.graph.atom_types))) or (
        node.active is not None and node.active.lineage != node.lineage
    ):
        raise ValueError("search lineage disagrees with exact graph or active option")
    return node
