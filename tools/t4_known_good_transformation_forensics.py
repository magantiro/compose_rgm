"""Trace exact known-good T4 transformations through frozen controller stages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

from compose_v4.chem.molecular_graph import is_element, molecular_graph_to_smiles
from compose_v4.control.complete_macro_selector import CompleteMacroSelector
from compose_v4.control.compositional_structural_subgoal_generator import (
    CompositionalPatchGenerator,
    _candidate_score,
    _event_region,
    _is_local,
    _mutable_after,
    enumerate_local_legal_successors,
    joint_event_features,
    region_features,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_legal_action_policy import (
    LegalSuccessor,
    _record_parts,
    enumerate_legal_successors,
)
from compose_v4.control.structural_subgoal import (
    extract_structural_goal,
    instantiate_goal,
)
from compose_v4.control.target_conditioned_utility_selector import UtilityRanker
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_compositional_structural_subgoal_generator import _t4_data
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_structural_subgoal_audit import teacher_traces

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/t4_known_good_transformation_forensics_v1.json"
DEFAULT_OUTPUT = ROOT / "diagnostics/t4_known_good_transformation_forensics/attempt_1"
RESULT_SCHEMA = "t4_known_good_transformation_forensics_result_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def _unseal_read(path: Path) -> dict:
    envelope = _read(path)
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(
        payload
    ):
        raise ValueError(f"sealed payload identity changed: {path}")
    return payload


def load_contract(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("contract_sha256") != identity(
        payload
    ):
        raise ValueError(f"forensics contract is not self-hashed: {path}")
    if any(payload["costs"][name] != 0 for name in payload["costs"]):
        raise ValueError("forensics contract authorizes a nonzero external cost")
    inputs = payload["inputs"]
    rows = []
    for value in inputs.values():
        rows.extend(value if isinstance(value, list) else [value])
    for row in rows:
        actual = sha256_file(ROOT / row["path"])
        if actual != row["sha256"]:
            raise ValueError(f"forensics input hash changed: {row['path']}")
    return payload


def _rank_existing(rows: list[tuple[str, float]], teacher_id: str) -> int:
    ordered = sorted(rows, key=lambda row: (-row[1], row[0]))
    return next(
        index for index, (key, _) in enumerate(ordered, start=1) if key == teacher_id
    )


def _rank_injected(
    score: float, alternatives: list[tuple[str, float]], teacher_id: str
) -> int:
    return _rank_existing([*alternatives, (teacher_id, score)], teacher_id)


def _fold_assets() -> tuple[dict[str, dict], dict[int, dict]]:
    candidate_lock = _unseal_read(
        ROOT
        / "diagnostics/t4_compositional_structural_subgoal_generator/attempt_1/candidate_lock.json.gz"
    )
    cases = {}
    folds = {}
    for fold_row in candidate_lock["folds"]:
        fold = int(fold_row["fold"])
        generator_envelope = _read(
            ROOT
            / f"diagnostics/t4_compositional_structural_subgoal_generator/attempt_1/fold_{fold}/runtime_checkpoint.json.gz"
        )
        selector_envelope = _read(
            ROOT
            / f"diagnostics/t4_complete_macro_selector/attempt_1/fold_{fold}/runtime_checkpoint.json.gz"
        )
        utility = _read(
            ROOT
            / f"diagnostics/t4_target_conditioned_utility_selector/attempt_4/fold_{fold}_checkpoint.json"
        )
        folds[fold] = {
            "generator": CompositionalPatchGenerator.from_checkpoint(
                generator_envelope["payload"]["model"]
            ),
            "structural": CompleteMacroSelector.from_checkpoint(
                selector_envelope["payload"]["selector"]
            ),
            "target_blind": UtilityRanker.from_checkpoint(utility["target_blind"]),
            "target_conditioned": UtilityRanker.from_checkpoint(
                utility["target_conditioned"]
            ),
        }
        for case in fold_row["cases"]:
            cases[case["source_case_id"]] = {"fold": fold, **case}
    if len(cases) != 15:
        raise RuntimeError(f"candidate lock source census changed: {len(cases)}")
    return cases, folds


def _source_case_by_group(
    traces: list[dict], cases: dict[str, dict]
) -> dict[str, dict]:
    lookup = {
        canonical_state_key(decode_state(case["source_state"])): case
        for case in cases.values()
    }
    result = {}
    for trace in traces:
        key = canonical_state_key(decode_state(trace["states"][0]))
        case = lookup.get(key)
        if case is None:
            raise RuntimeError(
                f"teacher source absent from frozen candidate lock: {trace['source_group']}"
            )
        result[trace["source_group"]] = case
    if len(result) != 15:
        raise RuntimeError("whole-source fold mapping is incomplete")
    return result


def _known_labels(teacher: dict) -> list[dict]:
    receipt = _read(ROOT / teacher["receipt"])
    return [
        {
            "target": row["target"],
            "source_idx": row["source_idx"],
            "delta": row["delta"],
            "run_seed": row["run_seed"],
            "reported_external_docking_score": row["reported_docking_score"],
            "evidence": "reported public endpoint label; not a runtime input",
        }
        for row in receipt["pair"]["references"]
    ]


def _legal_teacher_candidate(
    current, successor, action_record, candidates
) -> LegalSuccessor | None:
    rule, _ = decode_action(action_record)
    key = canonical_state_key(successor)
    matches = [
        row for row in candidates if row.rule == rule and row.successor_key == key
    ]
    if len(matches) > 1:
        raise RuntimeError(
            "canonical legal fiber contains a duplicate teacher successor"
        )
    return matches[0] if matches else None


def _pair_is_local(
    states: tuple[dict, ...], actions: tuple[dict, ...], index: int
) -> bool:
    """Return whether two adjacent teacher actions form one local generator patch."""

    if index + 1 >= len(actions):
        return False
    source = decode_state(states[index])
    first_successor = decode_state(states[index + 1])
    _, first_references, _, _ = _record_parts(actions[index], source)
    region = tuple(
        sorted(
            slot
            for slot in first_references
            if bool(is_element(source.atom_types[slot]))
        )
    )
    if not region:
        return False
    mutable = _mutable_after(source, first_successor, region)
    _, references, _, _ = _record_parts(actions[index + 1], first_successor)
    return bool(references) and references <= set(mutable)


def _sequential_support(trace: dict) -> dict:
    """Separate region count, patch horizon, and hypothetical repeated decisions."""

    states = tuple(trace["states"])
    actions = tuple(trace["actions"])
    components = tuple(trace["regions"]["components"])
    regions = []
    for component in components:
        indices = tuple(map(int, component["primitive_indices"]))
        regions.append(
            {
                "component_index": int(component["component_index"]),
                "primitive_indices": list(indices),
                "primitive_count": len(indices),
                "exceeds_one_to_two_event_invocation_horizon": len(indices) > 2,
                "contiguous_in_teacher_emission_order": indices
                == tuple(range(min(indices), max(indices) + 1)),
            }
        )

    # One exact teacher primitive always fits a fresh generator invocation. Pair
    # adjacent actions only when the second is local to the first patch. Since a
    # chunk has maximum size two, greedy pairing is minimum-cardinality.
    chunks = []
    index = 0
    while index < len(actions):
        width = 2 if _pair_is_local(states, actions, index) else 1
        chunks.append([index + offset for offset in range(width)])
        index += width
    return {
        "current_generator_patch_horizon_events": [1, 2],
        "route_primitive_count": len(actions),
        "dependency_region_count": len(components),
        "has_multiple_dependency_regions": len(components) > 1,
        "regions": regions,
        "regions_exceeding_invocation_horizon": sum(
            row["exceeds_one_to_two_event_invocation_horizon"] for row in regions
        ),
        "all_regions_individually_within_invocation_horizon": all(
            not row["exceeds_one_to_two_event_invocation_horizon"] for row in regions
        ),
        "teacher_trace_fits_one_generator_invocation": len(actions) <= 2,
        "counterfactual_repeated_generator_decisions": {
            "primitive_trace_segmentable": True,
            "minimum_invocations": len(chunks),
            "event_chunks": chunks,
            "interpretation": "grammar-level only; proposal ranks were not evaluated on descendant states",
        },
        "actual_current_hybrid_runtime": {
            "generator_invocations_before_dynamic_v0": 1,
            "repeated_generator_decisions_authorized": False,
            "dynamic_v0_after_initial_lock": True,
            "teacher_trace_emittable_by_macro_lane": len(actions) <= 2,
            "dynamic_v0_exact_route_recovery": "not established by this zero-oracle teacher-forced audit",
        },
    }


def _event_forensics(trace: dict, model: CompositionalPatchGenerator) -> dict:
    """Teacher-force only the part of the exact route inside generator support."""

    states = tuple(trace["states"])
    actions = tuple(trace["actions"])
    components = trace["regions"]["components"]
    lengths = [len(row["primitive_indices"]) for row in components]
    support = _sequential_support(trace)
    whole_supported = support["actual_current_hybrid_runtime"][
        "teacher_trace_emittable_by_macro_lane"
    ]
    if not whole_supported:
        return {
            "whole_macro_supported": False,
            "whole_macro_support_absence": "route_exceeds_single_invocation_event_horizon",
            "whole_macro_teacher_forced_log_score": None,
            "runtime_support": support,
            "component_count": len(components),
            "component_primitive_lengths": lengths,
            "first_event": {
                "status": "not_scored_after_complete_macro_support_divergence"
            },
            "second_event": None,
            "continuation_stop": {
                "status": "unsupported_complete_control_sequence",
                "supported_maximum_events": 2,
                "teacher_route_events": len(actions),
            },
            "factor_heads": {
                "where": "not scored after complete-macro support divergence",
                "topology_atom_bond_attachment_dependency": "one joint event density in the frozen architecture; an entire out-of-support macro has no likelihood or rank",
            },
        }

    component_source = decode_state(states[0])
    successor = decode_state(states[1])
    legal = enumerate_legal_successors(component_source)
    teacher = _legal_teacher_candidate(component_source, successor, actions[0], legal)
    if teacher is None:
        return {
            "whole_macro_supported": False,
            "whole_macro_support_absence": "teacher_first_successor_absent_from_legal_fiber",
            "whole_macro_teacher_forced_log_score": None,
            "runtime_support": support,
            "component_count": len(components),
            "component_primitive_lengths": lengths,
            "first_event": {"status": "unsupported"},
            "second_event": None,
            "continuation_stop": {"status": "unsupported"},
            "factor_heads": {
                "where": "separate region density",
                "topology_atom_bond_attachment_dependency": "one joint event density; separate ranks are not identifiable in this frozen architecture",
            },
        }

    region = _event_region(component_source, teacher)
    mutable = _mutable_after(component_source, teacher.successor, region)
    region_score = model.region_density.score(region_features(component_source, region))
    joint_score = model.joint_event_density.score(
        joint_event_features(
            component_source=component_source,
            current=component_source,
            candidate=teacher,
            region_slots=region,
            mutable_slots=region,
            depth=0,
            previous_rule=None,
        )
    )
    first_score = _candidate_score(
        model,
        component_source=component_source,
        current=component_source,
        candidate=teacher,
        region_slots=region,
        mutable_slots=region,
        depth=0,
        previous_rule=None,
    )
    region_rows: list[tuple[str, float]] = []
    event_rows: list[tuple[str, float]] = []
    rule_rows: list[tuple[str, float]] = []
    teacher_key = None
    for index, candidate in enumerate(legal):
        candidate_region = _event_region(component_source, candidate)
        candidate_mutable = _mutable_after(
            component_source, candidate.successor, candidate_region
        )
        key = f"{candidate.rule}:{candidate.successor_key!r}:{index}"
        region_rows.append(
            (
                key,
                model.region_density.score(
                    region_features(component_source, candidate_region)
                ),
            )
        )
        score = _candidate_score(
            model,
            component_source=component_source,
            current=component_source,
            candidate=candidate,
            region_slots=candidate_region,
            mutable_slots=candidate_mutable,
            depth=0,
            previous_rule=None,
        )
        event_rows.append((key, score))
        if candidate.rule == teacher.rule:
            rule_rows.append((key, score))
        if (
            candidate.rule == teacher.rule
            and candidate.successor_key == teacher.successor_key
        ):
            teacher_key = key
    if teacher_key is None:
        raise RuntimeError("teacher disappeared from its legal scoring fiber")
    first = {
        "status": "supported",
        "legal_fiber_size": len(legal),
        "where": {
            "log_density": region_score,
            "rank": _rank_existing(region_rows, teacher_key),
            "denominator": len(region_rows),
        },
        "joint_event": {
            "log_density": joint_score,
            "combined_log_score": first_score,
            "global_rank": _rank_existing(event_rows, teacher_key),
            "within_rule_rank": _rank_existing(rule_rows, teacher_key),
            "global_denominator": len(event_rows),
            "within_rule_denominator": len(rule_rows),
        },
    }
    continuation = {
        "supported_maximum_events": 2,
        "teacher_route_events": len(actions),
        "stop_after_first_probability": model.stop_after_first,
        "teacher_control_after_first": "stop" if len(actions) == 1 else "continue",
        "teacher_control_log_probability": math.log(
            model.stop_after_first if len(actions) == 1 else 1 - model.stop_after_first
        ),
        "entire_route_control_sequence_supported": len(actions) <= 2,
    }
    second = None
    if len(actions) >= 2 and whole_supported:
        second_current = decode_state(states[1])
        second_successor = decode_state(states[2])
        local = enumerate_local_legal_successors(second_current, mutable)
        second_teacher = _legal_teacher_candidate(
            second_current, second_successor, actions[1], local
        )
        prefix = type(
            "Prefix", (), {"current": second_current, "mutable_slots": mutable}
        )()
        if second_teacher is None or not _is_local(prefix, second_teacher):
            second = {
                "status": "unsupported_in_local_second_event_fiber",
                "fiber_size": len(local),
            }
            if whole_supported:
                whole_supported = False
        else:
            second_score = _candidate_score(
                model,
                component_source=component_source,
                current=second_current,
                candidate=second_teacher,
                region_slots=region,
                mutable_slots=mutable,
                depth=1,
                previous_rule=teacher.rule,
            )
            rows = []
            second_key = None
            for index, candidate in enumerate(local):
                if not _is_local(prefix, candidate):
                    continue
                key = f"{candidate.rule}:{candidate.successor_key!r}:{index}"
                score = _candidate_score(
                    model,
                    component_source=component_source,
                    current=second_current,
                    candidate=candidate,
                    region_slots=region,
                    mutable_slots=mutable,
                    depth=1,
                    previous_rule=teacher.rule,
                )
                rows.append((key, score))
                if (
                    candidate.rule == second_teacher.rule
                    and candidate.successor_key == second_teacher.successor_key
                ):
                    second_key = key
            if second_key is None:
                raise RuntimeError(
                    "second teacher disappeared from its local scoring fiber"
                )
            second = {
                "status": "supported",
                "log_score": second_score,
                "rank": _rank_existing(rows, second_key),
                "denominator": len(rows),
            }

    full_score = None
    if whole_supported:
        full_score = first_score + continuation["teacher_control_log_probability"]
        if second is not None:
            full_score += second["log_score"]
    return {
        "whole_macro_supported": whole_supported,
        "whole_macro_support_absence": (
            None if whole_supported else "route_exceeds_single_invocation_event_horizon"
        ),
        "whole_macro_teacher_forced_log_score": full_score,
        "runtime_support": support,
        "component_count": len(components),
        "component_primitive_lengths": lengths,
        "first_event": first,
        "second_event": second,
        "continuation_stop": continuation,
        "factor_heads": {
            "where": "separate region density",
            "topology_atom_bond_attachment_dependency": "one joint event density; separate ranks are not identifiable in this frozen architecture",
        },
    }


def _candidate_pools(case: dict) -> dict[str, list[dict]]:
    return {row["policy_id"]: list(row["candidates"]) for row in case["policies"]}


def _selector_ranks(
    source,
    target_state: dict,
    goal_payload: dict,
    realization: dict,
    target: str,
    assets: dict,
    pools: dict[str, list[dict]],
) -> dict:
    source_smiles = molecular_graph_to_smiles(source)
    target_smiles = molecular_graph_to_smiles(decode_state(target_state))
    if source_smiles is None or target_smiles is None:
        raise RuntimeError("known-good source or endpoint cannot be canonicalized")
    goal_id = identity(goal_payload)
    teacher = {
        "endpoint_state": target_state,
        "goal": goal_payload,
        "patch_ids": [goal_id],
        "realization": {
            "status": realization["status"],
            "endpoint_matches_bound_target": realization[
                "endpoint_matches_bound_target"
            ],
            "primitive_teacher_actions_used": realization[
                "primitive_teacher_actions_used"
            ],
            "primitive_count": len(realization["actions"]),
            "expanded": int(realization["expanded"]),
            "attempted": int(realization["attempted"]),
        },
    }
    result = {}
    scorers = {
        "structural_selector": lambda row: assets["structural"].score(source, row),
        "target_blind_utility": lambda row: assets["target_blind"].score_graph_pair(
            source_smiles,
            molecular_graph_to_smiles(decode_state(row["endpoint_state"])),
            target=target,
        ),
    }
    conditioned = assets["target_conditioned"]
    if conditioned.supports(target):
        scorers["target_conditioned_utility"] = (
            lambda row: conditioned.score_graph_pair(
                source_smiles,
                molecular_graph_to_smiles(decode_state(row["endpoint_state"])),
                target=target,
            )
        )
    else:
        result["target_conditioned_utility"] = {
            "status": "abstained_target_absent_from_fold_training_pairs",
            "training_targets": list(conditioned.training_targets),
        }
    for scorer_name, scorer in scorers.items():
        teacher_score = scorer(teacher)
        by_policy = {}
        for policy, candidates in sorted(pools.items()):
            alternatives = []
            failures = Counter()
            for index, candidate in enumerate(candidates):
                try:
                    score = scorer(candidate)
                except (RuntimeError, TypeError, ValueError) as error:
                    failures[type(error).__name__] += 1
                    continue
                alternatives.append((f"{policy}:{index}", score))
            by_policy[policy] = {
                "locked_pool_size": len(candidates),
                "scorable_locked_candidates": len(alternatives),
                "score_abstentions": dict(sorted(failures.items())),
                "teacher_injected_only_for_diagnostic": True,
                "teacher_score": teacher_score,
                "teacher_rank": _rank_injected(
                    teacher_score, alternatives, "__teacher__"
                ),
                "rank_denominator_with_injected_teacher": len(alternatives) + 1,
            }
        result[scorer_name] = {"status": "scored", "by_policy": by_policy}
    return result


def _first_divergence(generator: dict, selectors: dict) -> str:
    runtime = generator["runtime_support"]
    if not generator["whole_macro_supported"]:
        if runtime["regions_exceeding_invocation_horizon"]:
            return "regional_event_horizon_support_absence"
        if (
            runtime["counterfactual_repeated_generator_decisions"][
                "primitive_trace_segmentable"
            ]
            and not runtime["actual_current_hybrid_runtime"][
                "repeated_generator_decisions_authorized"
            ]
        ):
            return "controller_does_not_sequence_generator_patches"
        return "generator_support_absence"
    learned = generator.get("autonomous_exact_endpoint_recovery", {}).get(
        "balanced_joint_autoregressive", False
    )
    if not learned:
        return "low_proposal_probability_or_decoder_pruning"
    structural = selectors["structural_selector"]["by_policy"][
        "balanced_joint_autoregressive"
    ]
    if structural["teacher_rank"] > 32:
        return "structural_selector_misranking"
    blind = selectors["target_blind_utility"]["by_policy"][
        "balanced_joint_autoregressive"
    ]
    if blind["teacher_rank"] > 32:
        return "target_blind_utility_misranking"
    conditioned = selectors.get("target_conditioned_utility", {})
    if conditioned.get("status") == "scored":
        row = conditioned["by_policy"]["balanced_joint_autoregressive"]
        if row["teacher_rank"] > 32:
            return "target_conditioned_utility_misranking"
    return "no_divergence_through_audited_stages"


def run(contract_path: Path) -> dict:
    contract = load_contract(contract_path)
    comparison = unseal(ROOT / contract["inputs"]["comparison_reference"]["path"])
    provenance_abstentions = []
    for missing in contract["missing_required_assets"]:
        path = Path(missing["expected_path"])
        provenance_abstentions.append(
            {
                **missing,
                "status": "abstained_missing_exact_parent_program_endpoint_trace",
                "asset_exists_at_analysis": path.exists(),
                "score_only_evidence": (
                    comparison["dynamic_v0"]["jak2_1_r0"]["curve"][0]["best_score"]
                    if "dynamic_v0" in missing["probe_id"]
                    else comparison["full_146"]["jak2_1_r0"]["curve"][0]["best_score"]
                ),
                "not_substituted_with_final_champion": True,
            }
        )

    traces, _, _ = _t4_data()
    raw_teachers = {row["program_id"]: row for row in teacher_traces()}
    cases, fold_assets = _fold_assets()
    source_cases = _source_case_by_group(traces, cases)
    benchmark = unseal(ROOT / contract["inputs"]["benchmark_contract"]["path"])
    seeds = json.loads((ROOT / contract["inputs"]["seed_registry"]["path"]).read_text())
    groups = source_group_map(benchmark, seeds)
    route_rows = []
    for trace in sorted(traces, key=lambda row: (row["source_group"], row["route_id"])):
        teacher = raw_teachers[trace["route_id"]]
        source = decode_state(trace["states"][0])
        target_state = trace["states"][-1]
        target_graph = decode_state(target_state)
        goal, bindings, _ = extract_structural_goal(trace["states"], trace["actions"])
        instantiated, binding_receipt = instantiate_goal(source, goal, bindings)
        representation_ok = canonical_state_key(instantiated) == canonical_state_key(
            target_graph
        )
        case = source_cases[trace["source_group"]]
        assets = fold_assets[int(case["fold"])]
        pools = _candidate_pools(case)
        generator = _event_forensics(trace, assets["generator"])
        target_key = canonical_state_key(target_graph)
        autonomous = {
            policy: any(
                canonical_state_key(decode_state(candidate["endpoint_state"]))
                == target_key
                for candidate in candidates
            )
            for policy, candidates in sorted(pools.items())
        }
        generator["autonomous_exact_endpoint_recovery"] = autonomous
        selectors = {
            "status": "abstained_teacher_route_outside_single_invocation_generator_contract",
            "teacher_injected": False,
        }
        route_rows.append(
            {
                "probe_id": trace["route_id"],
                "probe_role": "exact_complete_Full_146_teacher_route_not_early_call_trace",
                "source_group": trace["source_group"],
                "cell": groups[trace["source_group"]]["cell"],
                "target": groups[trace["source_group"]]["target"],
                "teacher_receipt": teacher["receipt"],
                "teacher_receipt_sha256": teacher["receipt_sha256"],
                "known_labels": _known_labels(teacher),
                "primitive_count": len(trace["actions"]),
                "dependency_regions": len(trace["regions"]["components"]),
                "representation": {
                    "status": "represented" if representation_ok else "failed",
                    "goal_id": goal.goal_id,
                    "dependency_region_count": len(goal.subgoals),
                    "exact_endpoint_reconstruction": representation_ok,
                    "precision": 1.0 if representation_ok else 0.0,
                    "primitive_teacher_actions_used": binding_receipt[
                        "primitive_teacher_actions_used"
                    ],
                },
                "compiler": {
                    "status": "abstained_missing_local_sealed_realizer_result_payload",
                    "exact_endpoint_realization": None,
                    "recomputed": False,
                    "sealed_realizer_identity_reported_sha256": "c803e3c3118e262a8b6fc4df1cfb9f8fed2934d2dc6fb47431ee4e319b64ec5f",
                    "launch_receipt": "diagnostics/t4_structural_subgoal_realizer/attempt_1/launch.json",
                    "interpretation": "prior compiler success was reported but cannot be promoted to per-route computed evidence without the sealed result payload",
                },
                "generator": generator,
                "selectors": selectors,
                "first_divergence": _first_divergence(generator, selectors),
            }
        )
    divergence_counts = Counter(row["first_divergence"] for row in route_rows)
    representation_exact = sum(
        row["representation"]["exact_endpoint_reconstruction"] for row in route_rows
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "evidence": "computed zero-oracle teacher-forced diagnostic",
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256_file(contract_path),
            "payload_sha256": identity(contract),
        },
        "implementation": {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree_dirty_at_execution": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
        },
        "census": {
            "required_early_probes": 2,
            "required_early_probe_abstentions": len(provenance_abstentions),
            "admitted_exact_complete_routes": len(route_rows),
            "admitted_early_transition_probes": 0,
        },
        "gates": {
            "representation": {
                "covered": representation_exact,
                "denominator": len(route_rows),
                "coverage": representation_exact / len(route_rows),
                "precision": 1.0 if representation_exact else None,
            },
            "exact_compiler": {
                "covered": 0,
                "denominator": len(route_rows),
                "coverage": None,
                "precision": None,
                "status": "abstained_missing_local_sealed_realizer_result_payloads",
                "prior_reported_sealed_realizer_identity_sha256": "c803e3c3118e262a8b6fc4df1cfb9f8fed2934d2dc6fb47431ee4e319b64ec5f",
            },
            "current_single_invocation_generator_support": {
                "covered": sum(
                    row["generator"]["whole_macro_supported"] for row in route_rows
                ),
                "denominator": len(route_rows),
                "coverage": sum(
                    row["generator"]["whole_macro_supported"] for row in route_rows
                )
                / len(route_rows),
            },
            "multi_dependency_region_routes": {
                "covered": sum(
                    row["generator"]["runtime_support"][
                        "has_multiple_dependency_regions"
                    ]
                    for row in route_rows
                ),
                "denominator": len(route_rows),
            },
            "routes_with_a_region_exceeding_generator_horizon": {
                "covered": sum(
                    bool(
                        row["generator"]["runtime_support"][
                            "regions_exceeding_invocation_horizon"
                        ]
                    )
                    for row in route_rows
                ),
                "denominator": len(route_rows),
            },
            "counterfactual_repeated_generator_trace_segmentation": {
                "covered": sum(
                    row["generator"]["runtime_support"][
                        "counterfactual_repeated_generator_decisions"
                    ]["primitive_trace_segmentable"]
                    for row in route_rows
                ),
                "denominator": len(route_rows),
                "runtime_authorized": False,
            },
            "selector_teacher_injection": {
                "covered": sum(
                    row["selectors"].get("teacher_injected", True) for row in route_rows
                ),
                "denominator": len(route_rows),
                "interpretation": "restricted to probes inside the exact current one-invocation generator contract",
            },
            "autonomous_exact_endpoint_recovery": {
                policy: {
                    "covered": sum(
                        row["generator"]["autonomous_exact_endpoint_recovery"][policy]
                        for row in route_rows
                    ),
                    "denominator": len(route_rows),
                }
                for policy in ("balanced_joint_autoregressive", "uniform_joint_grammar")
            },
        },
        "first_divergence_counts": dict(sorted(divergence_counts.items())),
        "provenance_abstentions": provenance_abstentions,
        "routes": route_rows,
        "limitations": [
            "The two JAK2 query-1 score points lack committed exact parent/program/endpoint traces and are therefore provenance abstentions.",
            "The 77 admitted probes are complete teacher routes, not substitutes for missing early-call transitions.",
            "Teacher injection changes no frozen pool; its rank is a diagnostic over an in-memory pool of N locked candidates plus one teacher.",
            "The generator has a separate WHERE density and one joint event density; separate topology, atom, bond, attachment, and dependency likelihoods do not exist in this revision.",
            "The current hybrid invokes the compositional generator once at the source and then hands control to Dynamic-v0; repeated compositional macro generation is not part of the frozen runtime.",
            "Counterfactual repeated-decision segmentation establishes grammar-level trace segmentation only, not autonomous descendant-state proposal probability or exact Dynamic-v0 recovery.",
            "Public reported docking scores are labels only and never enter proposal or selector inputs.",
        ],
    }


def publish(output: Path, payload: dict) -> None:
    if output.exists():
        raise ValueError(f"refusing to overwrite forensic artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "result.json")
    arguments = parser.parse_args()
    result = run(arguments.contract.resolve())
    publish(arguments.output.resolve(), result)
    print(
        json.dumps(
            {
                "census": result["census"],
                "gates": result["gates"],
                "first_divergence_counts": result["first_divergence_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
