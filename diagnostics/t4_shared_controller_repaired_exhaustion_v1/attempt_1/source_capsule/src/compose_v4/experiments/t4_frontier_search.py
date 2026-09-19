"""Prepare-only T4 adapter for resumable exact-state frontier search.

This schema is NOT an oracle authorization/candidate lock. Replay auditing and
an explicit scientific launch remain separate. No docking or network I/O here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control import graph_geometry as GG
from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.docking_value import DockingValue
from compose_v4.control.frontier_search import FrontierSearch, payload_hash
from compose_v4.control.molecular_search_codec import (
    decode_lineage,
    decode_search_state,
    encode_lineage,
    encode_search_state,
)
from compose_v4.control.molecular_task_search import (
    MolecularHierarchy,
    MolecularSearchState,
)
from compose_v4.control.option_continuation import (
    EXECUTABLE_PRODUCT_GATE,
    OptionContinuationKernel,
    exact_graph_key,
)
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    encode_action,
    state_payload,
)
from compose_v4.experiments.t4_endpoint_selection import (
    acceptable_endpoint,
    calculate_properties,
    feasible_endpoint,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "t4_resumable_frontier_preparation_v1"


@dataclass(frozen=True)
class FrontierConfig:
    lineages: int = 8
    primitive_budget: int = 16
    planning_transitions: int = 2
    oracle_batch: int = 20
    seed: int = 1000
    guidance: str = "in_loop"

    def __post_init__(self):
        if self.guidance not in ("in_loop", "post_hoc"):
            raise ValueError("frontier guidance must be in_loop or post_hoc")
        if self.guidance == "post_hoc" and self.planning_transitions != 0:
            raise ValueError(
                "post-hoc generation does not perform task-guided planning"
            )
        for field, value in asdict(self).items():
            if field == "guidance":
                continue
            if type(value) is not int or value < (
                0 if field in ("seed", "planning_transitions") else 1
            ):
                raise ValueError(f"invalid frontier configuration {field}={value!r}")
        if self.lineages > 8 or self.primitive_budget > 16 or self.oracle_batch > 20:
            raise ValueError(
                "frontier configuration exceeds the bounded T4 development cell"
            )


def prepare_slice(
    warm: dict,
    *,
    source_sha256: str,
    input_sha256: dict,
    code_revision: str,
    enumerate_law,
    system,
    config: FrontierConfig | None = None,
    checkpoint: dict | None = None,
    events_per_parent: int = 3,
    executor_limit: int | None = None,
    progress=None,
    lineage_indices: tuple[int, ...] | None = None,
) -> dict:
    """Advance retained lineages; a slice/round boundary never deletes an unfinished one.

    input_sha256 binds the frozen reference/executor/codec dependency closure.
    Callers own atomic persistence of progress checkpoints and explicit launches.
    A zero-event slice refreshes task scores without executing molecular edits.
    """
    config = config or FrontierConfig()
    selected_lineages = (
        list(range(config.lineages))
        if lineage_indices is None
        else sorted(lineage_indices)
    )
    if (
        not selected_lineages
        or len(set(selected_lineages)) != len(selected_lineages)
        or any(
            type(i) is not int or not 0 <= i < config.lineages
            for i in selected_lineages
        )
    ):
        raise ValueError(
            "lineage partition must contain distinct in-range integer indices"
        )
    if type(events_per_parent) is not int or events_per_parent < 0:
        raise ValueError("events_per_parent must be a nonnegative scheduling quantum")
    if (
        not input_sha256
        or any(
            len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in input_sha256.values()
        )
        or len(code_revision) != 40
        or any(c not in "0123456789abcdef" for c in code_revision)
    ):
        raise ValueError(
            "frontier preparation requires input SHA-256s and an exact code revision"
        )
    if warm.get("schema_version") != "t4_exact_archive_v1" or not warm.get("archive"):
        raise ValueError("frontier preparation requires an exact-state T4 archive")
    archive, next_round = warm["archive"], warm["round"] + 1
    if warm["oracle_attempts"] != len(archive) - 1:
        raise ValueError("archive does not account for every prior oracle attempt")
    for record in archive:
        if canonical_state_key(decode_state(record["state"])) != record["smiles"]:
            raise ValueError("archive canonical metadata disagrees with exact state")
    model = DockingValue.fit(
        archive, before_round=next_round, source_sha256=source_sha256
    )
    snapshot = model.payload["snapshot_sha256"]
    reference_id = payload_hash(
        {
            "inputs": input_sha256,
            "config": asdict(config),
            "seed_state": archive[0]["state"],
            "product_gate": EXECUTABLE_PRODUCT_GATE,
            "schema": SCHEMA,
        }
    )
    prefix = [payload_hash(row) for row in archive]
    units, prior_calls, previous = [], 0, None
    if checkpoint is not None:
        previous = {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
        if (
            previous.get("schema_version") != SCHEMA
            or previous.get("reference_id") != reference_id
            or payload_hash(previous) != checkpoint.get("checkpoint_sha256")
            or previous.get("partition_lineages", list(range(config.lineages)))
            != selected_lineages
            or previous["round"] > next_round
            or prefix[: len(previous["archive_prefix"])] != previous["archive_prefix"]
            or (
                previous["round"] == next_round and prefix != previous["archive_prefix"]
            )
        ):
            raise ValueError(
                "frontier checkpoint identity/hash or append-only archive mismatch"
            )
        # Detach caller-owned persisted data before extending any lineage.
        units = json.loads(json.dumps(previous["frontier"], allow_nan=False))
        if len(units) != config.lineages or [u["lineage_index"] for u in units] != list(
            range(config.lineages)
        ):
            raise ValueError("frontier checkpoint lost or duplicated a lineage")
        prior_calls = previous["executor_calls"]

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(archive[0]["smiles"]))
    old_keys = {row["smiles"] for row in archive}
    properties, values = {}, {}

    def props(smiles):
        if smiles not in properties:
            properties[smiles] = calculate_properties(
                Chem.MolFromSmiles(smiles),
                seed_fp=seed_fp,
                generator=generator,
                sa_scorer=sascorer.calculateScore,
                delta=0.4,
                qed_min=0.6,
                sa_max=4.0,
            )
        return properties[smiles]

    def endpoint(node):
        # Zero planning alone is NOT a post-hoc baseline: decision() also scores
        # completed successors. Disable that entire generation-time value route.
        if config.guidance == "post_hoc":
            return None
        if node.stage != "where":
            return None
        smiles = canonical_state_key(node.graph)
        if smiles not in values:
            values[smiles] = (
                0.0
                if smiles in old_keys
                else model.desirability(
                    smiles, acceptable_endpoint({"smiles": smiles, **props(smiles)})
                )
            )
        return values[smiles]

    parents = sorted(
        archive,
        key=lambda row: (
            (
                0
                if feasible_endpoint(props(row["smiles"])) and row["ds"] is not None
                else 1
            ),
            row["ds"] if row["ds"] is not None else 0,
            props(row["smiles"])["v"],
        ),
    )[: config.lineages]
    if len(parents) != config.lineages:
        raise ValueError("archive has fewer parents than requested lineages")

    def fresh(index):
        root = MolecularSearchState.start(
            decode_state(parents[index]["state"]),
            budget=config.primitive_budget,
            root_id=f"round{next_round}:lineage{index}",
        )
        seeds = np.random.SeedSequence([config.seed, next_round, index]).spawn(2)
        return {
            "lineage_index": index,
            "started_round": next_round,
            "root": encode_search_state(root),
            "current": encode_search_state(root),
            "planner_seed": int(seeds[0].generate_state(1)[0]),
            "acting_rng": np.random.default_rng(seeds[1]).bit_generator.state,
            "planning_remaining": config.planning_transitions,
            "planner": None,
            "how_witnesses": {},
            "bundle": None,
            "path": [],
            "candidates": [],
            "status": "ready",
        }

    if not units:
        units = [fresh(index) for index in range(config.lineages)]
    retired = (
        [] if previous is None else json.loads(json.dumps(previous["retired_lineages"]))
    )
    for index, unit in enumerate(units):
        if unit["started_round"] < next_round and (
            decode_search_state(unit["current"]).budget == 0
            or unit["status"] == "no_admissible_action"
        ):
            retired.append(unit)
            units[index] = fresh(index)

    pending = None if previous is None else previous.get("pending_slice")
    if pending is not None and previous["round"] == next_round:
        if pending["events_per_parent"] != events_per_parent:
            raise ValueError(
                "resume the unfinished slice with its original events_per_parent"
            )
        targets = list(pending["target_events"])
        if len(targets) != len(units) or any(
            type(t) is not int or t < 0 for t in targets
        ):
            raise ValueError("malformed pending-slice lineage census")
    else:
        targets = [len(unit["path"]) + events_per_parent for unit in units]

    def pending_slice():
        unfinished = any(
            len(unit["path"]) < target
            and unit["current"]["budget"] > 0
            and unit["status"] != "no_admissible_action"
            for unit, target in zip(units, targets, strict=True)
            if unit["lineage_index"] in selected_lineages
        )
        return (
            {"events_per_parent": events_per_parent, "target_events": targets}
            if unfinished
            else None
        )

    started = perf_counter()
    meter = ExecutorMeter(executor_limit)

    def publish():
        body = {
            "schema_version": SCHEMA,
            "reference_id": reference_id,
            "round": next_round,
            "source_sha256": source_sha256,
            "input_sha256": input_sha256,
            "code_revision": code_revision,
            "config": asdict(config),
            "archive_prefix": prefix,
            "value_snapshot": model.payload,
            "frontier": units,
            "partition_lineages": selected_lineages,
            "retired_lineages": retired,
            "pending_slice": pending_slice(),
            "executor_calls": prior_calls + meter.calls,
            "new_executor_calls": meter.calls,
            "new_oracle_calls": 0,
            "prior_oracle_attempts": warm["oracle_attempts"],
            "automatic_docking": False,
            "oracle_authorized": False,
            "software": {"numpy": np.__version__, "rdkit": rdBase.rdkitVersion},
        }
        # A callback must receive a detached, complete checkpoint, not references
        # subsequently mutated by this invocation.
        result = json.loads(json.dumps(body, allow_nan=False))
        result["checkpoint_sha256"] = payload_hash(result)
        if progress:
            progress(result)
        return result

    for unit, target in zip(units, targets, strict=True):
        if unit["lineage_index"] not in selected_lineages:
            continue
        root = decode_search_state(unit["root"])
        node = decode_search_state(unit["current"])
        law_cache = {}

        def cached_law(graph, law_cache=law_cache):
            key = exact_graph_key(graph)
            if key not in law_cache:
                law_cache[key] = enumerate_law(graph)
            return law_cache[key]

        kernel = OptionContinuationKernel(
            cached_law,
            system,
            max_executor_applications=None,
            product_gate=EXECUTABLE_PRODUCT_GATE,
        )
        hierarchy = MolecularHierarchy(kernel)

        def reference(state, hierarchy=hierarchy, kernel=kernel, unit=unit):
            row = hierarchy.row(state)
            if state.stage == "how":
                unit["how_witnesses"][payload_hash(encode_search_state(state))] = [
                    {
                        "mark": encode_action(*mark),
                        "option_product": state_payload(active),
                    }
                    for mark, active in zip(
                        kernel.marks(state.active),
                        kernel.row(state.active).successors,
                        strict=True,
                    )
                ]
            return row

        planner = FrontierSearch(
            reference,
            endpoint,
            lambda state: state.budget == 0,
            MolecularSearchState.key,
            reference_id=reference_id,
            snapshot_id=snapshot,
            seed=unit["planner_seed"],
        )
        if unit["planner"] is not None:
            planner.restore(unit["planner"], decode_search_state)
        rng = np.random.default_rng()
        rng.bit_generator.state = unit["acting_rng"]

        def save(current, unit=unit, planner=planner, rng=rng):
            unit["current"] = encode_search_state(current)
            unit["planner"] = planner.checkpoint(encode_search_state)
            unit["acting_rng"] = rng.bit_generator.state
            return publish()

        try:
            with meter.instrument():
                for _ in range(max(0, target - len(unit["path"]))):
                    if node.budget == 0 or unit["status"] == "no_admissible_action":
                        break
                    before = planner.work.planning_transitions
                    meter.phase = "planning"
                    try:
                        with meter.boundary():
                            planning = planner.advance(node, unit["planning_remaining"])
                    finally:
                        unit["planning_remaining"] -= (
                            planner.work.planning_transitions - before
                        )
                    if planning["status"] == "executor_paused":
                        unit["status"] = "executor_paused"
                        break
                    unit["planning_remaining"] = 0
                    # Completed planning is a durable unit even if constructing
                    # the following committed decision later exceeds a slice cap.
                    save(node)
                    meter.phase = "committed_decision"
                    decision = planner.decision(node)
                    row = planner.row(node)
                    if not row.successors:
                        unit["status"] = "no_admissible_action"
                        break
                    choice = int(
                        rng.choice(len(row.successors), p=decision["probabilities"])
                    )
                    product = row.successors[choice]
                    event = {
                        "index": len(unit["path"]),
                        "round": next_round,
                        "source": encode_search_state(node),
                        "product": encode_search_state(product),
                        "decision": decision,
                        "selected_index": choice,
                        "selected_label": row.labels[choice],
                    }
                    if node.stage == "what":
                        unit["bundle"] = {
                            "bundle_id": product.active.bundle_id,
                            "option": product.active.option,
                            "r_release": node.region.released_fraction,
                            "region_atoms": sorted(node.region.atoms),
                            "interface": node.region.interface,
                            "origin_lineage": encode_lineage(node.lineage),
                        }
                    if node.stage == "how":
                        witness = unit["how_witnesses"][
                            payload_hash(encode_search_state(node))
                        ][choice]
                        event.update(witness)
                        if product.stage == "where":
                            local = GG.structural_displacement(
                                node.active.origin,
                                product.graph,
                                decode_lineage(unit["bundle"]["origin_lineage"]),
                                product.lineage,
                            )
                            unit["candidates"].append(
                                {
                                    **unit["bundle"],
                                    "smiles": canonical_state_key(product.graph),
                                    "state": encode_state(product.graph),
                                    "root_id": root.root_id,
                                    "event": event["index"],
                                    "program_complete": True,
                                    "primitive_depth": root.budget - product.budget,
                                    "r_coherent": local["largest_changed_fraction"],
                                    "r_change": local["changed_fraction"],
                                    "d_cycle_rank": local["d_cycle_rank"],
                                    "d_ring_systems": local["d_ring_systems"],
                                    "d_heavy": local["d_heavy"],
                                    "cumulative_change": GG.structural_displacement(
                                        root.graph,
                                        product.graph,
                                        root.lineage,
                                        product.lineage,
                                    ),
                                }
                            )
                    unit["path"].append(event)
                    node = product
                    unit["planning_remaining"] = config.planning_transitions
                    unit["status"] = (
                        "primitive_budget_complete" if node.budget == 0 else "ready"
                    )
                    save(node)
        except ContinuationBudgetExceeded:
            unit["status"] = "executor_paused"
        save(node)

    # Re-score retained completions using only the current round-frozen predictor.
    pool = {}
    for unit in [*retired, *units]:
        for candidate in unit["candidates"]:
            smiles = candidate["smiles"]
            if smiles in old_keys:
                continue
            if smiles not in pool:
                pool[smiles] = {**candidate, **props(smiles), "origin_bundle_ids": []}
            pool[smiles]["origin_bundle_ids"].append(candidate["bundle_id"])
    candidates = [pool[key] for key in sorted(pool)]
    predictions = model.predict([c["smiles"] for c in candidates]) if candidates else []
    for candidate, prediction in zip(candidates, predictions, strict=True):
        candidate.update(
            predicted_docking=float(prediction),
            oracle_eligible=acceptable_endpoint(candidate),
            origin_bundle_ids=sorted(set(candidate["origin_bundle_ids"])),
        )
    selected, represented = [], set()
    for candidate in sorted(
        candidates, key=lambda c: (c["predicted_docking"], c["smiles"])
    ):
        available = [b for b in candidate["origin_bundle_ids"] if b not in represented]
        if (
            candidate["oracle_eligible"]
            and available
            and len(selected) < config.oracle_batch
        ):
            selected.append({**candidate, "allocated_bundle_id": available[0]})
            represented.add(available[0])
    result = publish()
    return {
        "schema_version": "t4_frontier_slice_result_v1",
        "checkpoint": result,
        "pool": candidates,
        "proposed_for_audit": selected,
        "new_oracle_calls": 0,
        "oracle_authorized": False,
        "seconds": perf_counter() - started,
        "executor_calls": meter.calls,
        "executor_attempts": meter.attempts,
    }
