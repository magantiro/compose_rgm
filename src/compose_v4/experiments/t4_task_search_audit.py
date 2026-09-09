"""Durable zero-oracle production audit, with exact cross-option replay."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control import graph_geometry as GG
from compose_v4.control.docking_value import DockingValue, identity
from compose_v4.control.fused_option import FusedProgress, completed_fused_cycle
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import OptionState, exact_graph_key
from compose_v4.control.region_rewrite import Lineage, RewriteContext, context_preserved
from compose_v4.control.ring_program import RingProgress, completed_construction, ring_spec
from compose_v4.control.task_search import SearchRow
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    sha256_file,
    state_payload,
    verify_file,
)
from compose_v4.experiments.t4_endpoint_selection import calculate_properties, feasible_endpoint
from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_task_search import PreparationConfig
from compose_v4.gates.med_chem_gate import is_valid
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

CONTRACT_PATH = "configs/t4_task_search_audit.json"
KIND = "t4_task_search_audit"


def decode_option(payload):
    context = dict(payload["context"])
    context.update(
        frozen=frozenset(context["frozen"]),
        locus=frozenset(context["locus"]),
        terminals=tuple(tuple(x) for x in context["terminals"]),
    )
    lineage = payload["lineage"]
    return OptionState(
        decode_state(payload["graph"]),
        decode_state(payload["origin"]),
        RewriteContext(**context),
        Lineage(dict(lineage["slot_of"]), dict(lineage["id_of"]), lineage["next_id"]),
        payload["option"],
        payload["step"],
        payload["horizon"],
        payload["bundle_id"],
        FusedProgress.from_payload(payload["fused_progress"])
        if "fused_progress" in payload
        else None,
        ring_progress=RingProgress.from_payload(payload["ring_progress"])
        if "ring_progress" in payload
        else None,
    )


def verify_lock(lock, warm, system, *, verification_limit=2048):
    """Replay selected edits only. Learned full-row correctness is not re-enumerated."""
    config = PreparationConfig(**lock["config"])
    if (
        lock["schema_version"] != "t4_hierarchical_candidate_lock_v2"
        or lock["round"] != warm["round"] + 1
        or lock["new_oracle_calls"] != 0
        or lock["automatic_docking"] is not False
        or lock["prior_oracle_attempts"] != warm["oracle_attempts"]
        or len(lock["work"]) != config.lineages
        or lock["config_sha256"] != identity(lock["config"])
    ):
        raise ValueError("hierarchical lock schema, round or budget mismatch")
    model = DockingValue.from_payload(lock["value_snapshot"])
    if (
        model.payload
        != DockingValue.fit(
            warm["archive"], before_round=lock["round"], source_sha256=lock["source_sha256"]
        ).payload
    ):
        raise ValueError("value snapshot is not the complete prior-round prefix")
    parents = {r["smiles"]: r for r in warm["archive"]}
    previous = {canonical_state_key(decode_state(r["state"])) for r in warm["archive"]}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(warm["archive"][0]["smiles"]))
    endpoints, replayed = {}, 0
    if sorted(u["parent_index"] for u in lock["work"]) != list(range(config.lineages)):
        raise ValueError("missing or repeated parent unit")
    with ExecutorMeter(verification_limit).instrument() as meter:
        meter.phase = "selected_path_verification"
        for unit in lock["work"]:
            calls = unit["parent_budget"]["executor_calls"]
            if (
                type(calls) is not int
                or calls < 0
                or (config.executor_per_parent is not None and calls > config.executor_per_parent)
                or calls != len(unit["executor_attempts"])
                or unit["parent_budget"]["limit"] != config.executor_per_parent
            ):
                raise ValueError("parent executor accounting mismatch")
            planning_calls = unit["planning_budget"]["executor_calls"]
            if (
                type(planning_calls) is not int
                or not 0 <= planning_calls <= calls
                or (
                    config.planning_executor_per_parent is not None
                    and planning_calls > config.planning_executor_per_parent
                )
                or unit["planning_budget"]["limit"] != config.planning_executor_per_parent
            ):
                raise ValueError("planning exceeded its parent share")
            root = MolecularSearchState.start(
                decode_state(parents[unit["parent"]]["state"]),
                budget=config.primitive_budget,
                root_id=f"round{lock['round']}:lineage{unit['parent_index']}",
            )
            node, active, option_origin_lineage = root, None, None
            hierarchy = MolecularHierarchy(None)
            for index, event in enumerate(unit["path"]):
                if (
                    event["event"] != index
                    or event["stage"] != node.stage
                    or event["source"] != encode_state(node.graph)
                    or event["budget_before"] != node.budget
                ):
                    raise ValueError("cross-option path discontinuity")
                d = event["decision"]
                p, q = np.asarray(d["reference"]), np.asarray(d["probabilities"])
                row = SearchRow(
                    tuple(range(len(p))),
                    tuple(d["labels"]),
                    tuple(d["core"]),
                    tuple(d["floor"]),
                    d["epsilon"],
                )
                if (
                    q.shape != p.shape
                    or not np.isfinite(q).all()
                    or not np.isclose(q.sum(), 1, rtol=0, atol=1e-10)
                    or np.any(q < d["epsilon"] * np.asarray(d["floor"]) - 1e-12)
                    or not np.allclose(p, row.reference, rtol=0, atol=1e-12)
                    or d["snapshot_id"] != model.payload["snapshot_sha256"]
                ):
                    raise ValueError("decision probability/floor/snapshot mismatch")
                live = q > 0
                if np.any(p[live] <= 0):
                    raise ValueError("decision created unsupported mass")
                kl = float(np.sum(q[live] * np.log(q[live] / p[live])))
                selected = event["selected_index"]
                if (
                    not 0 <= selected < len(q)
                    or q[selected] <= 0
                    or kl > 1 + 1e-10
                    or not np.isclose(kl, d["kl"], rtol=0, atol=1e-10)
                    or d["labels"][selected] != event["selected_label"]
                ):
                    raise ValueError("selected decision or KL mismatch")
                if node.stage == "where":
                    regions = hierarchy.row(node)
                    if tuple(d["labels"]) != regions.labels or not np.allclose(
                        p, regions.reference
                    ):
                        raise ValueError("WHERE reference changed")
                    node = regions.successors[selected]
                    if (
                        event["region_atoms"] != sorted(node.region.atoms)
                        or event["r_release"] != node.region.released_fraction
                        or event["interface"] != node.region.interface
                    ):
                        raise ValueError("intended region metadata mismatch")
                elif node.stage == "what":
                    expected = hierarchy.option_state(node, event["selected_label"])
                    if identity(state_payload(expected)) != identity(event["option_initial"]):
                        raise ValueError("option did not start at its exact origin")
                    active, option_origin_lineage = expected, node.lineage
                    node = MolecularSearchState(
                        node.graph,
                        node.lineage,
                        node.budget,
                        node.root_id,
                        "how",
                        node.region,
                        active,
                    )
                else:
                    if identity(state_payload(active)) != identity(event["option_source"]):
                        raise ValueError("option phase or lineage discontinuity")
                    mark = event["mark"]
                    codec = action_codec_v4 if mark["schema_version"] == 4 else action_codec
                    rule, action = codec.decode_action(mark)
                    if event["selected_label"] != f"{rule}:{action!r}":
                        raise ValueError("selected mark differs from decision")
                    product = system.apply(active.graph, rule, action)
                    replayed += 1
                    following = decode_option(event["option_product"])
                    if (
                        exact_graph_key(product) != exact_graph_key(following.graph)
                        or following.lineage != active.lineage.observe(rule, action)
                        or following.step != active.step + 1
                        or following.option != active.option
                        or following.horizon != active.horizon
                        or following.bundle_id != active.bundle_id
                        or exact_graph_key(following.origin) != exact_graph_key(active.origin)
                        or not charge_policy_preserved(active.graph, product)
                        or not context_preserved(
                            active.origin,
                            product,
                            active.context.frozen,
                            active.context.terminal_context_slots,
                        )
                        or not is_valid(canonical_state_key(product))
                    ):
                        raise ValueError("invalid replay, option or context transition")
                    complete = following.remaining == 0
                    if event.get("option_completed", False) != complete:
                        raise ValueError("false option completion")
                    if complete:
                        spec = ring_spec(following.option)
                        if spec and not completed_construction(
                            following.origin,
                            product,
                            following.ring_progress,
                            spec,
                            refined=bool(spec.refine),
                        ):
                            raise ValueError("missing completed ring construction witness")
                        if following.fused_progress and not completed_fused_cycle(
                            following.origin, product, following.fused_progress
                        ):
                            raise ValueError("missing completed fused ring witness")
                        local = GG.structural_displacement(
                            active.origin, product, option_origin_lineage, following.lineage
                        )
                        cumulative = GG.structural_displacement(
                            root.graph, product, root.lineage, following.lineage
                        )
                        endpoints[(unit["parent_index"], index)] = (event, local, cumulative)
                        node = MolecularSearchState(
                            product, following.lineage, node.budget - 1, node.root_id
                        )
                    else:
                        node = MolecularSearchState(
                            product,
                            following.lineage,
                            node.budget - 1,
                            node.root_id,
                            "how",
                            node.region,
                            following,
                        )
                    active = following
                if event["product"] != encode_state(node.graph):
                    raise ValueError("event exact product mismatch")
    pool = lock["pool"]
    if len({r["smiles"] for r in pool}) != len(pool):
        raise ValueError("canonical duplicate in candidate pool")
    for c in pool:
        event, local, cumulative = endpoints[(c["parent_lineage_id"], c["event"])]
        if (
            c["state"] != event["product"]
            or c["bundle_id"] != event["bundle_id"]
            or c["smiles"] in previous
            or not c["program_complete"]
            or canonical_state_key(decode_state(c["state"])) != c["smiles"]
        ):
            raise ValueError("candidate is not a new exact option completion")
        for field, expected in {
            "r_coherent": local["largest_changed_fraction"],
            "r_change": local["changed_fraction"],
            "d_cycle_rank": local["d_cycle_rank"],
            "d_ring_systems": local["d_ring_systems"],
            "d_heavy": local["d_heavy"],
        }.items():
            if not np.isclose(c[field], expected):
                raise ValueError(f"candidate structural metadata mismatch: {field}")
        if c["cumulative_change"] != cumulative:
            raise ValueError("candidate cumulative-change mismatch")
        props = calculate_properties(
            Chem.MolFromSmiles(c["smiles"]),
            seed_fp=seed_fp,
            generator=generator,
            sa_scorer=sascorer.calculateScore,
            delta=0.4,
            qed_min=0.6,
            sa_max=4,
        )
        if any(not np.isclose(c[k], v) for k, v in props.items()):
            raise ValueError("candidate feasibility metadata mismatch")
        if not np.isclose(c["predicted_docking"], model.predict([c["smiles"]])[0]):
            raise ValueError("candidate task score mismatch")
    if (
        len(lock["take"]) > config.oracle_batch
        or len({r["smiles"] for r in lock["take"]}) != len(lock["take"])
        or any(c not in pool or not feasible_endpoint(c) for c in lock["take"])
    ):
        raise ValueError("invalid selected candidate lock")
    if lock["executor_calls"] != sum(u["parent_budget"]["executor_calls"] for u in lock["work"]):
        raise ValueError("total executor accounting mismatch")
    return {
        "status": "verified",
        "replayed_steps": replayed,
        "public_executor_calls": meter.calls,
        "attempts": meter.attempts,
    }


def summarize(lock, verification):
    events = [e for u in lock["work"] for e in u["path"]]
    decisions = {}
    for stage in ("where", "what", "how"):
        rows = [e["decision"] for e in events if e["stage"] == stage]
        tv = [
            float(np.abs(np.asarray(d["probabilities"]) - d["reference"]).sum() / 2) for d in rows
        ]
        decisions[stage] = {
            "total": len(rows),
            "nonreference": sum(v > 1e-8 for v in tv),
            "max_total_variation": max(tv, default=0),
        }
    completed = sum(u["planner"]["rollouts_completed"] for u in lock["work"])
    viable_parents = {c["parent_lineage_id"] for c in lock["take"]}
    passes = (
        completed > 0
        and all(d["nonreference"] > 0 for d in decisions.values())
        and len(viable_parents) >= 2
    )
    return {
        "schema_version": "t4_task_search_audit_v1",
        "new_oracle_calls": 0,
        "decision": "ready_for_matched_comparison" if passes else "diagnose_before_docking",
        "decisions": decisions,
        "rollouts_completed": completed,
        "rollouts_started": lock["planning_rollouts_started"],
        "parents": [
            {
                k: u[k]
                for k in (
                    "parent_index",
                    "status",
                    "completed_options",
                    "parent_budget",
                    "planning_budget",
                    "planner",
                    "seconds",
                )
            }
            for u in lock["work"]
        ],
        "options_selected": dict(
            sorted(Counter(e["selected_label"] for e in events if e["stage"] == "what").items())
        ),
        "unique_candidates": len(lock["pool"]),
        "feasible_candidates": sum(feasible_endpoint(c) for c in lock["pool"]),
        "selected_candidates": len(lock["take"]),
        "selected_parent_lineages": len(viable_parents),
        "ring_changes_in_pool": sum(c["d_cycle_rank"] != 0 for c in lock["pool"]),
        "proposal_seconds": sum(u["seconds"] for u in lock["work"]),
        "executor_calls": lock["executor_calls"],
        "verification": {k: v for k, v in verification.items() if k != "attempts"},
        "automatic_docking": False,
    }


def run_audit(task, warm, prepare, system, output, *, commit=lambda: None, progress=None):
    progress = {} if progress is None else progress
    path = output / "candidate_lock.json"
    if path.exists():
        envelope = unseal(path)
        if envelope["task"] != task:
            raise ValueError("cached audit task mismatch")
        lock = envelope["lock"]
    else:
        cache = {}
        for path_parent in sorted((output / "parents").glob("*.json")):
            unit = unseal(path_parent)
            if unit["task"] != task:
                raise ValueError("cached parent task mismatch")
            cache[unit["parent_index"]] = {k: v for k, v in unit.items() if k != "task"}
        for started in (output / "started").glob("*.json"):
            if int(started.stem) not in cache:
                raise RuntimeError("unfinished started parent; audit cost before replay")

        def checkpoint(unit):
            index = unit["parent_index"]
            if unit["schema_version"] == "t4_task_search_tick_v2":
                progress.update(unit)
                return
            if unit["schema_version"] == "t4_task_search_parent_started_v2":
                seal(output / "started" / f"{index:02d}.json", {**unit, "task": task})
                progress.update(phase="prepare", parent_index=index)
            else:
                seal(output / "parents" / f"{index:02d}.json", {**unit, "task": task})
                progress.update(
                    phase="parent_complete",
                    parent_index=index,
                    completed_rollouts=unit["planner"]["rollouts_completed"],
                    parent_executor_calls=unit["parent_budget"]["executor_calls"],
                )
            commit()

        lock = prepare(task, checkpoint, cache, warm)
        seal(path, {"task": task, "lock": lock})
        commit()
    if (
        lock["input_sha256"] != task["expected_input_sha256"]
        or lock["source_sha256"] != task["warm_archive_file_sha256"]
        or lock["config"] != task["preparation"]
    ):
        raise ValueError("audit input identity mismatch")
    progress.update(phase="verify_selected_paths")
    verification_path = output / "verification.json"
    lock_sha = sha256_file(path)
    if verification_path.exists():
        verification = unseal(verification_path)
        if verification["candidate_lock_sha256"] != lock_sha:
            raise ValueError("verification is not bound to the saved lock")
    else:
        verification = {**verify_lock(lock, warm, system), "candidate_lock_sha256": lock_sha}
        seal(verification_path, verification)
        commit()
    return {**summarize(lock, verification), "candidate_lock_sha256": lock_sha}


def run_remote(
    task,
    repo_root,
    artifact_root,
    volume,
    runtime_factory,
    validate_revision,
    prepare,
    *,
    lazy_probe=False,
    uncapped_probe=False,
):
    if lazy_probe and uncapped_probe:
        raise ValueError("choose one task-search probe")
    kind = (
        "t4_uncapped_lookahead_probe"
        if uncapped_probe
        else "t4_lazy_reference_probe"
        if lazy_probe
        else KIND
    )
    contract_path = f"configs/{kind}.json"
    contract = json.loads((repo_root / contract_path).read_text())
    expected_preparation = (
        PreparationConfig(
            lineages=1,
            planning_policy="lazy_reference",
            compute_policy="metered_uncapped_v1",
            executor_per_parent=None,
            planning_executor_per_parent=None,
            max_rows=None,
        )
        if uncapped_probe
        else PreparationConfig(lineages=1, planning_policy="lazy_reference")
        if lazy_probe
        else PreparationConfig()
    )

    def runner(actual_task, prepare, dock, output, **kwargs):
        if (
            contract["compute"]["oracle_call_limit"] != 0
            or PreparationConfig(**contract["preparation"]) != expected_preparation
            or rdBase.rdkitVersion != contract["required_rdkit"]
        ):
            raise ValueError("audit recipe, oracle limit or RDKit mismatch")
        source = artifact_root / contract["source"]["path"]
        if not source.resolve().is_relative_to(artifact_root.resolve()):
            raise ValueError("audit archive escapes volume root")
        verify_file(source, contract["source"]["sha256"])
        verify_file(repo_root / contract["value_check"]["path"], contract["value_check"]["sha256"])
        check = json.loads((repo_root / contract["value_check"]["path"]).read_text())
        if check["decision"] != "development_signal_present":
            raise ValueError("predictor development gate did not pass")
        warm = unseal(source)
        if warm["round"] != 4 or warm["oracle_attempts"] != 51 or len(warm["archive"]) != 52:
            raise ValueError("audit requires exact complete 51-call archive")
        actual_task.update(
            image_revision=task["image_revision"],
            warm_archive_path=str(source),
            warm_archive_file_sha256=contract["source"]["sha256"],
            warm_start_sha256=identity(warm),
            preparation=contract["preparation"],
        )
        result = run_audit(
            actual_task, warm, prepare, runtime_factory()["system"], output, **kwargs
        )
        if lazy_probe or uncapped_probe:
            # A one-parent cost probe cannot evaluate the two-lineage audit gate.
            # It neither replaces that gate nor authorizes docking.
            result.update(
                decision="review_uncapped_lookahead_probe"
                if uncapped_probe
                else "review_lazy_reference_cost_probe",
                full_audit_routing_applicable=False,
                automatic_docking=False,
            )
        return result

    volume.reload()
    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        runtime_factory,
        validate_revision,
        prepare,
        None,
        contract_path=contract_path,
        run_kind=kind,
        runner=runner,
    )
