"""Prepare-only T4 task search; its cross-option lock is intentionally schema v2.

No docking, deployment or automatic continuation occurs in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control import graph_geometry as GG
from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.docking_value import DockingValue, identity
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import OptionContinuationKernel, exact_graph_key
from compose_v4.control.task_search import TaskSearch
from compose_v4.experiments.continuation_profile import ExecutorMeter, encode_action, state_payload
from compose_v4.experiments.t4_endpoint_selection import calculate_properties, feasible_endpoint
from compose_v4.experiments.t4_parent_budget import ParentExecutorShare
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state


@dataclass(frozen=True)
class PreparationConfig:
    lineages: int = 8
    primitive_budget: int = 16
    executor_per_parent: int = 2500
    planning_executor_per_parent: int = 512
    max_rows: int = 128
    max_terminals: int = 256
    max_rollouts: int = 32
    initial_rollouts: int = 8
    rollouts_per_decision: int = 2
    oracle_batch: int = 20
    seed: int = 1000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid preparation field {name}: {value!r}")
        if (
            self.lineages > 8
            or self.primitive_budget > 16
            or self.executor_per_parent > 2500
            or self.planning_executor_per_parent >= self.executor_per_parent
            or self.oracle_batch > 20
        ):
            raise ValueError("preparation exceeds the bounded development allocation")


def prepare(
    warm: dict,
    *,
    source_sha256: str,
    enumerate_law,
    system,
    input_sha256: dict,
    config: PreparationConfig | None = None,
    progress=None,
    parent_cache: dict | None = None,
) -> dict:
    config = config or PreparationConfig()
    if warm.get("schema_version") != "t4_exact_archive_v1" or not warm.get("archive"):
        raise ValueError("hierarchical preparation requires an exact-state T4 archive")
    archive, next_round = warm["archive"], int(warm["round"]) + 1
    if warm["oracle_attempts"] != len(archive) - 1:
        raise ValueError("exact archive does not account for every prior oracle attempt")
    for row in archive:
        graph = decode_state(row["state"])
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None or canonical_state_key(graph) != Chem.MolToSmiles(molecule):
            raise ValueError("archive SMILES metadata disagrees with exact state")
    started = perf_counter()
    model = DockingValue.fit(archive, before_round=next_round, source_sha256=source_sha256)
    snapshot = model.payload["snapshot_sha256"]
    preparation_identity = identity(
        {
            "source": source_sha256,
            "inputs": input_sha256,
            "config": asdict(config),
            "snapshot": snapshot,
        }
    )
    parent_cache = {} if parent_cache is None else parent_cache
    for index, unit in parent_cache.items():
        if (
            type(index) is not int
            or not 0 <= index < config.lineages
            or unit["parent_index"] != index
            or unit["preparation_identity"] != preparation_identity
        ):
            raise ValueError("cached parent identity/configuration mismatch")
    seed = archive[0]["smiles"]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(seed))
    property_cache, value_cache = {}, {}

    def properties(smiles):
        if smiles not in property_cache:
            property_cache[smiles] = calculate_properties(
                Chem.MolFromSmiles(smiles),
                seed_fp=seed_fp,
                generator=generator,
                sa_scorer=sascorer.calculateScore,
                delta=0.4,
                qed_min=0.6,
                sa_max=4.0,
            )
        return property_cache[smiles]

    def terminal(node):
        if node.budget:
            return None
        smiles = canonical_state_key(node.graph)
        if smiles not in value_cache:
            value_cache[smiles] = model.desirability(smiles, feasible_endpoint(properties(smiles)))
        return value_cache[smiles]

    parents = sorted(
        archive,
        key=lambda a: (
            0 if feasible_endpoint(properties(a["smiles"])) and a["ds"] is not None else 1,
            a["ds"] if a["ds"] is not None else 0,
            properties(a["smiles"])["v"],
        ),
    )[: config.lineages]
    if len(parents) < config.lineages:
        raise ValueError("warm archive has fewer parents than the requested lineages")
    old_keys = {canonical_state_key(decode_state(r["state"])) for r in archive}
    pool, units = {}, []
    all_started = 0
    prior_calls = sum(u["parent_budget"]["executor_calls"] for u in parent_cache.values())

    def merge(unit):
        for record in unit["candidates"]:
            smiles = record["smiles"]
            if smiles not in pool:
                pool[smiles] = {**record, "origin_bundle_ids": list(record["origin_bundle_ids"])}
            else:
                pool[smiles]["origin_bundle_ids"] = sorted(
                    set(pool[smiles]["origin_bundle_ids"] + record["origin_bundle_ids"])
                )
        value_cache.update(unit["terminal_values"])

    with ExecutorMeter(
        config.lineages * config.executor_per_parent - prior_calls
    ).instrument() as meter:
        for parent_index, parent in enumerate(parents):
            if parent_index in parent_cache:
                unit = parent_cache[parent_index]
                if unit["parent"] != parent["smiles"]:
                    raise ValueError("cached parent differs from round-frozen parent selection")
                units.append(unit)
                merge(unit)
                all_started += unit["planner"]["rollouts_started"]
                continue
            parent_start = perf_counter()
            attempt_start = len(meter.attempts)
            parent_candidates = []
            if progress:
                progress(
                    {
                        "schema_version": "t4_task_search_parent_started_v2",
                        "parent_index": parent_index,
                        "preparation_identity": preparation_identity,
                    }
                )
            root = MolecularSearchState.start(
                decode_state(parent["state"]),
                budget=config.primitive_budget,
                root_id=f"round{next_round}:lineage{parent_index}",
            )
            law_cache = {}

            def cached_law(graph, law_cache=law_cache, parent_index=parent_index):
                key = exact_graph_key(graph)
                if key not in law_cache:
                    if progress:
                        progress(
                            {
                                "schema_version": "t4_task_search_tick_v2",
                                "parent_index": parent_index,
                                "phase": meter.phase,
                                "executor_calls": prior_calls + meter.calls,
                                "law_enumerations": len(law_cache),
                            }
                        )
                    law_cache[key] = enumerate_law(graph)
                return law_cache[key]

            kernel = OptionContinuationKernel(
                cached_law, system, max_executor_applications=config.executor_per_parent
            )
            hierarchy = MolecularHierarchy(kernel)
            seeds = np.random.SeedSequence([config.seed, next_round, parent_index]).spawn(2)
            planner = TaskSearch(
                hierarchy.row,
                terminal,
                MolecularSearchState.key,
                snapshot_id=snapshot,
                seed=int(seeds[0].generate_state(1)[0]),
                max_rows=config.max_rows,
                max_terminals=config.max_terminals,
                max_rollouts=config.max_rollouts,
                max_path_steps=3 * config.primitive_budget,
            )
            acting_rng = np.random.default_rng(seeds[1])
            share = ParentExecutorShare(config.executor_per_parent)
            planning_share = ParentExecutorShare(config.planning_executor_per_parent)
            node, path, bundle, complete_options = root, [], None, 0
            status = "primitive_budget_complete"
            try:
                with share.instrument():
                    for event_index in range(3 * config.primitive_budget):
                        if node.budget == 0:
                            break
                        if not planning_share.exhausted:
                            meter.phase = "planning"
                            with planning_share.instrument():
                                planner.plan(
                                    node,
                                    config.initial_rollouts
                                    if not path
                                    else config.rollouts_per_decision,
                                )
                        meter.phase = "committed_decision"
                        decision = planner.decision(node)
                        row = planner.row(node)
                        if not row.successors:
                            status = "no_admissible_action"
                            break
                        index = int(
                            acting_rng.choice(len(row.successors), p=decision["probabilities"])
                        )
                        product = row.successors[index]
                        event = {
                            "event": event_index,
                            "stage": node.stage,
                            "budget_before": node.budget,
                            "selected_index": index,
                            "selected_label": row.labels[index],
                            "decision": decision,
                            "source": encode_state(node.graph),
                            "product": encode_state(product.graph),
                        }
                        if node.stage == "where":
                            event.update(
                                region_atoms=sorted(product.region.atoms),
                                r_release=product.region.released_fraction,
                                interface=product.region.interface,
                            )
                        if node.stage == "what":
                            event["option_initial"] = state_payload(product.active)
                            bundle = {
                                "bundle_id": product.active.bundle_id,
                                "option": product.active.option,
                                "region_atoms": sorted(node.region.atoms),
                                "r_release": node.region.released_fraction,
                                "interface": node.region.interface,
                                "kind": node.region.kind,
                                "option_parent": canonical_state_key(node.graph),
                                "option_origin": encode_state(node.graph),
                                "origin_lineage": node.lineage,
                            }
                        if node.stage == "how":
                            active_product = kernel.row(node.active).successors[index]
                            event.update(
                                bundle_id=bundle["bundle_id"],
                                option=node.active.option,
                                primitive_step=node.active.step + 1,
                                mark=encode_action(*kernel.marks(node.active)[index]),
                                option_source=state_payload(node.active),
                                option_product=state_payload(active_product),
                            )
                        path.append(event)
                        if node.stage == "how" and product.stage == "where":
                            complete_options += 1
                            smiles = canonical_state_key(product.graph)
                            local = GG.structural_displacement(
                                node.active.origin,
                                product.graph,
                                bundle["origin_lineage"],
                                product.lineage,
                            )
                            cumulative = GG.structural_displacement(
                                root.graph, product.graph, root.lineage, product.lineage
                            )
                            event["option_completed"] = True
                            if smiles not in old_keys:
                                record = {
                                    **{k: v for k, v in bundle.items() if k != "origin_lineage"},
                                    "smiles": smiles,
                                    "state": encode_state(product.graph),
                                    "parent": parent["smiles"],
                                    "parent_lineage_id": parent_index,
                                    "event": event_index,
                                    "primitive_depth": root.budget - product.budget,
                                    "program_complete": True,
                                    "completed_options": complete_options,
                                    "r_coherent": local["largest_changed_fraction"],
                                    "r_change": local["changed_fraction"],
                                    "d_cycle_rank": local["d_cycle_rank"],
                                    "d_ring_systems": local["d_ring_systems"],
                                    "d_heavy": local["d_heavy"],
                                    "cumulative_change": cumulative,
                                    **properties(smiles),
                                    "predicted_docking": float(model.predict([smiles])[0]),
                                    "origin_bundle_ids": [bundle["bundle_id"]],
                                }
                                parent_candidates.append(record)
                        node = product
            except ContinuationBudgetExceeded as error:
                status = str(error)
            if share.exhausted:
                status = "parent_executor_budget_exhausted"
            unit = {
                "schema_version": "t4_task_search_parent_v2",
                "preparation_identity": preparation_identity,
                "parent_index": parent_index,
                "parent": parent["smiles"],
                "status": status,
                "path": path,
                "completed_options": complete_options,
                "parent_budget": share.receipt(),
                "planning_budget": planning_share.receipt(),
                "planner": planner.receipt(),
                "acting_rng_state": acting_rng.bit_generator.state,
                "law_enumerations": len(law_cache),
                "seconds": perf_counter() - parent_start,
                "candidates": parent_candidates,
                "terminal_values": dict(value_cache),
                "executor_attempts": meter.attempts[attempt_start:],
            }
            units.append(unit)
            merge(unit)
            all_started += planner.work.rollouts_started
            if progress:
                progress(unit)
    candidates = sorted(pool.values(), key=lambda r: r["smiles"])
    eligible = [r for r in candidates if feasible_endpoint(r)]
    # Each emitted completion is one committed bundle outcome. Deduplication can
    # merge origins, never create extra region draws from planning trajectories.
    eligible.sort(key=lambda r: (r["predicted_docking"], r["smiles"]))
    selected, bundles = [], set()
    for candidate in eligible:
        if candidate["bundle_id"] not in bundles and len(selected) < config.oracle_batch:
            selected.append(candidate)
            bundles.add(candidate["bundle_id"])
    return {
        "schema_version": "t4_hierarchical_candidate_lock_v2",
        "controller": "hierarchical_task_v1",
        "round": next_round,
        "config": asdict(config),
        "source_sha256": source_sha256,
        "input_sha256": input_sha256,
        "value_snapshot": model.payload,
        "pool": candidates,
        "take": selected,
        "work": units,
        "new_oracle_calls": 0,
        "prior_oracle_attempts": warm["oracle_attempts"],
        "terminal_value_evaluations": [
            {"smiles": k, "value": value_cache[k]} for k in sorted(value_cache)
        ],
        "planning_rollouts_started": all_started,
        "executor_calls": prior_calls + meter.calls,
        "new_executor_calls": meter.calls,
        "executor_attempts": [a for u in units for a in u["executor_attempts"]],
        "seconds": perf_counter() - started,
        "software": {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
        "config_sha256": identity(asdict(config)),
        "automatic_docking": False,
    }
