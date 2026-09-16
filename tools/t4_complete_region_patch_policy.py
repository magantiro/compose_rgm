"""Split-first support and policy gates for complete T4 region patches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.complete_region_patch_policy import (
    ATOM_TYPES,
    BOND_ORDERS,
    DEGREES,
    FORMAL_CHARGES,
    HYDROGEN_COUNTS,
    ConditionalPatchPolicy,
    PatchPolicyTrainingRow,
    PatchToken,
    SourceRegionContext,
    decode_patch_stream,
    encode_patch_stream,
    fit_conditional_patch_policy,
    patch_stream_support,
    sample_patch_stream,
    token_domain,
)
from compose_v4.control.complete_region_program import CompleteRegionProgram
from compose_v4.control.compositional_structural_subgoal_generator import (
    WeightedDiagonalDensity,
    fit_weighted_density,
    region_features,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import atom_signature, environment
from compose_v4.control.structural_subgoal import extract_structural_goal
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import predeclared_source_folds
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_structural_subgoal_audit import teacher_traces

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_complete_region_patch_policy_v1.json"
SUPPORT = ROOT / "diagnostics/t4_complete_region_runtime/attempt_1/support.json"
BENCHMARK = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
DEFAULT_OUTPUT = ROOT / "diagnostics/t4_complete_region_patch_policy/attempt_1"
SUPPORT_SCHEMA = "t4_complete_region_patch_policy_support_v1"
GATE3_SCHEMA = "t4_complete_region_patch_policy_gate3_v1"
GATE3_AGGREGATE_SCHEMA = "t4_complete_region_patch_policy_gate3_aggregate_v1"
GATE4_BLOCKED_SCHEMA = "t4_complete_region_patch_policy_gate4_blocked_v1"
CHECKPOINT_SCHEMA = "t4_complete_region_patch_policy_runtime_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite complete-region policy artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def load_contract(path: Path = CONTRACT) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("contract_sha256") != identity(payload):
        raise ValueError(f"complete-region policy contract is not self-hashed: {path}")
    if any(value for value in payload["costs"].values()):
        raise ValueError("complete-region policy contract authorizes external cost")
    for row in payload["inputs"].values():
        input_path = ROOT / row["path"]
        if sha256_file(input_path) != row["sha256"]:
            raise ValueError(f"complete-region policy input hash changed: {input_path}")
    return payload


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _require_clean() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("authoritative complete-region policy gate requires clean source")
    return _revision()


def _token_key(token: PatchToken) -> tuple[str, int, str]:
    return token.kind, token.value, token.factor


def _teacher_corpus() -> tuple[list[dict], dict[str, dict]]:
    support = unseal(SUPPORT)
    support_by_program = {
        row["teacher_program_id"]: CompleteRegionProgram.from_payload(row["runtime_program"])
        for row in support["rows"]
    }
    teachers = teacher_traces()
    rows = []
    for teacher in teachers:
        program = support_by_program.get(teacher["program_id"])
        if program is None:
            raise ValueError(
                f"teacher route absent from immutable runtime support: {teacher['program_id']}"
            )
        trace = teacher["trace"]
        goal, bindings, _ = extract_structural_goal(tuple(trace["states"]), tuple(trace["actions"]))
        if tuple(decision.patch for decision in program.decisions) != goal.subgoals:
            raise RuntimeError("immutable runtime program differs from extracted structural goal")
        source = decode_state(trace["states"][0])
        for region_index, (decision, binding) in enumerate(
            zip(program.decisions, bindings, strict=True)
        ):
            rows.append(
                {
                    "source_group": teacher["source_group"],
                    "route_id": teacher["program_id"],
                    "region_index": region_index,
                    "route_region_count": len(program.decisions),
                    "control_after": decision.control_after,
                    "source": source,
                    "source_state": trace["states"][0],
                    "binding": tuple(map(int, binding)),
                    "patch": decision.patch,
                    "context": SourceRegionContext.from_subgoal(decision.patch),
                    "tokens": encode_patch_stream(decision.patch),
                }
            )
    if len(teachers) != 77 or len(rows) != 147:
        raise RuntimeError("complete-region teacher corpus census changed")
    metadata = source_group_map(unseal(BENCHMARK), json.loads(SEEDS.read_text()))
    return rows, metadata


def _balanced_region_weights(rows: list[dict]) -> list[float]:
    by_source: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        by_source.setdefault(row["source_group"], {}).setdefault(row["route_id"], []).append(row)
    weights = []
    for row in rows:
        routes = by_source[row["source_group"]]
        regions = routes[row["route_id"]]
        weights.append(1 / (len(by_source) * len(routes) * len(regions)))
    total = sum(weights)
    return [value / total for value in weights]


def _training_row(row: dict) -> PatchPolicyTrainingRow:
    return PatchPolicyTrainingRow(
        source_group=row["source_group"],
        route_id=row["route_id"],
        region_index=row["region_index"],
        route_region_count=row["route_region_count"],
        context=row["context"],
        tokens=row["tokens"],
        control_after=row["control_after"],
    )


def _context_for_slots(source: MolecularGraph, slots: tuple[int, ...]) -> SourceRegionContext:
    ordered = tuple(
        sorted(
            set(map(int, slots)),
            key=lambda slot: (atom_signature(source, slot), environment(source, slot), slot),
        )
    )
    return SourceRegionContext(
        tuple(atom_signature(source, slot) for slot in ordered),
        tuple(tuple(int(source.bonds[left, right]) for right in ordered) for left in ordered),
        tuple(environment(source, slot) for slot in ordered),
    )


def _where_key(context: SourceRegionContext) -> str:
    return identity(
        {
            "input_atoms": context.input_atoms,
            "input_bonds": context.input_bonds,
            "environments": context.environments,
        }
    )


def _sample_where_fiber(
    source: MolecularGraph,
    policy: ConditionalPatchPolicy,
    *,
    seed: int,
    size: int,
) -> list[tuple[tuple[int, ...], SourceRegionContext]]:
    rng = np.random.default_rng(seed)
    live = tuple(int(value) for value in np.flatnonzero(is_element(source.atom_types)))
    counts = np.asarray(policy.region_size_counts[: len(live)], dtype=float) + 0.25
    counts /= counts.sum()
    rows: dict[str, tuple[tuple[int, ...], SourceRegionContext]] = {}
    attempts = 0
    while len(rows) < size and attempts < size * 128:
        attempts += 1
        count = int(rng.choice(np.arange(1, len(live) + 1), p=counts))
        chosen = {int(rng.choice(live))}
        while len(chosen) < count:
            frontier = sorted(
                {
                    int(neighbor)
                    for slot in chosen
                    for neighbor in np.flatnonzero(source.bonds[slot])
                    if int(neighbor) in live and int(neighbor) not in chosen
                }
            )
            available = frontier or [slot for slot in live if slot not in chosen]
            chosen.add(int(rng.choice(available)))
        slots = tuple(sorted(chosen))
        context = _context_for_slots(source, slots)
        rows.setdefault(_where_key(context), (slots, context))
    return [rows[key] for key in sorted(rows)]


def _softmax_probability_and_rank(
    rows: list[tuple[str, float]], teacher_id: str
) -> tuple[int, float]:
    ordered = sorted(rows, key=lambda row: (-row[1], row[0]))
    rank = next(index for index, row in enumerate(ordered, 1) if row[0] == teacher_id)
    maximum = max(score for _, score in rows)
    weights = {name: math.exp(score - maximum) for name, score in rows}
    probability = weights[teacher_id] / sum(weights.values())
    return rank, probability


def _program_count_probability(policy: ConditionalPatchPolicy, count: int) -> float:
    values = np.asarray(policy.program_count_counts, dtype=float) + policy.smoothing_alpha
    return float(values[count - 1] / values.sum())


def _sample_target_fiber(
    context: SourceRegionContext,
    policy: ConditionalPatchPolicy,
    *,
    seed: int,
    size: int,
) -> tuple[list[tuple], Counter]:
    rng = np.random.default_rng(seed)
    rows = {}
    telemetry = Counter()
    while len(rows) < size and telemetry["attempts"] < size * 512:
        telemetry["attempts"] += 1
        try:
            patch, tokens = sample_patch_stream(policy, context, learned=False, rng=rng)
        except (TypeError, ValueError) as error:
            telemetry[f"abstention:{type(error).__name__}"] += 1
            continue
        rows.setdefault(patch.subgoal_id, (patch, tokens))
    telemetry["unique_complete_patches"] = len(rows)
    telemetry["candidate_shortfall"] = max(0, size - len(rows))
    return [rows[key] for key in sorted(rows)], telemetry


def _teacher_perturbation_fiber(
    context: SourceRegionContext,
    teacher: tuple[PatchToken, ...],
    *,
    seed: int,
    size: int,
) -> list[tuple]:
    """Teacher-conditioned local fiber used only for Gate 3 rank diagnostics."""

    rng = np.random.default_rng(seed)
    teacher_patch = decode_patch_stream(context, teacher)
    rows = {teacher_patch.subgoal_id: (teacher_patch, teacher)}
    mutable = [
        index
        for index, token in enumerate(teacher)
        if token.kind.endswith(("_element", "_formal_charge", "_implicit_hydrogens", "_degree"))
        or token.kind == "bond_order"
    ]
    attempts = 0
    while len(rows) < size and attempts < size * 256 and mutable:
        attempts += 1
        candidate = list(teacher)
        mutations = int(rng.integers(1, min(4, len(mutable) + 1)))
        for index in rng.choice(mutable, size=mutations, replace=False):
            index = int(index)
            token = candidate[index]
            domain = token_domain(
                context,
                tuple(candidate[:index]),
                kind=token.kind,
                factor=token.factor,
            )
            alternatives = [value for value in domain if value != token.value]
            if alternatives:
                candidate[index] = PatchToken(
                    token.kind, int(rng.choice(alternatives)), token.factor
                )
        try:
            patch = decode_patch_stream(context, tuple(candidate))
        except (TypeError, ValueError):
            continue
        rows.setdefault(patch.subgoal_id, (patch, tuple(candidate)))
    return [rows[key] for key in sorted(rows)]


def _size_log_probability(policy: ConditionalPatchPolicy, size: int) -> float:
    counts = np.asarray(policy.region_size_counts, dtype=float) + policy.smoothing_alpha
    return float(math.log(counts[size - 1] / counts.sum()))


def _token_diagnostics(
    policy: ConditionalPatchPolicy,
    context: SourceRegionContext,
    tokens: tuple[PatchToken, ...],
    *,
    learned: bool,
) -> list[dict]:
    """Teacher-forced conditional ranks within the frozen value grammar."""

    prefix: tuple[PatchToken, ...] = ()
    rows = []
    for index, token in enumerate(tokens):
        domain = token_domain(context, prefix, kind=token.kind, factor=token.factor)
        candidates = []
        for value in domain:
            probe = PatchToken(token.kind, value, token.factor)
            candidates.append(
                (
                    value,
                    policy.token_probability(context, prefix, probe, learned=learned),
                )
            )
        ordered = sorted(candidates, key=lambda row: (-row[1], row[0]))
        probability = next(value for candidate, value in candidates if candidate == token.value)
        rank = next(
            rank for rank, (candidate, _) in enumerate(ordered, 1) if candidate == token.value
        )
        rows.append(
            {
                "index": index,
                "factor": token.factor,
                "kind": token.kind,
                "candidate_values": len(domain),
                "teacher_rank": rank,
                "teacher_probability": probability,
                "teacher_nll": -math.log(probability),
            }
        )
        prefix = (*prefix, token)
    return rows


def _checkpoint_for_fold(
    *,
    fold: int,
    policy: ConditionalPatchPolicy,
    where_density: WeightedDiagonalDensity,
    train_rows: list[dict],
    held_rows: list[dict],
    contract: dict,
    revision: str,
) -> dict:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "fold": fold,
        "implementation_revision": revision,
        "contract_payload_sha256": identity(contract),
        "training_identity": policy.training_identity,
        "split_identity": identity(
            {
                "train_source_hashes": sorted(identity(row["source_group"]) for row in train_rows),
                "held_source_hashes": sorted(identity(row["source_group"]) for row in held_rows),
            }
        ),
        "policy": policy.checkpoint(),
        "where_density": where_density.payload(),
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
    }
    serialized = json.dumps(payload, sort_keys=True)
    forbidden = {row["source_group"] for row in [*train_rows, *held_rows]} | {
        row["route_id"] for row in [*train_rows, *held_rows]
    }
    leaked = sorted(value for value in forbidden if value and value in serialized)
    if leaked:
        raise RuntimeError(f"runtime checkpoint leaked training identity: {leaked[:3]}")
    return payload


def _rank_summary(ranks: list[int]) -> dict:
    return {
        "count": len(ranks),
        "mean": float(np.mean(ranks)) if ranks else None,
        "median": float(np.median(ranks)) if ranks else None,
        "top1": sum(value <= 1 for value in ranks),
        "top8": sum(value <= 8 for value in ranks),
        "top32": sum(value <= 32 for value in ranks),
        "top128": sum(value <= 128 for value in ranks),
    }


def gate3_fold(
    output_root: Path,
    *,
    fold_index: int,
    contract_path: Path = CONTRACT,
) -> dict:
    """Fit and evaluate one independent split-first Gate 3 fold."""

    started = time.monotonic()
    contract = load_contract(contract_path)
    revision = _require_clean()
    rows, metadata = _teacher_corpus()
    split = next(
        row for row in predeclared_source_folds(metadata) if int(row["fold"]) == fold_index
    )
    train_sources = set(split["train_sources"])
    held_sources = set(split["test_sources"])
    train = [row for row in rows if row["source_group"] in train_sources]
    held = [row for row in rows if row["source_group"] in held_sources]
    alpha = float(contract["policy"]["smoothing_alpha"])
    budget = int(contract["policy"]["teacher_rank_candidates"])
    seed = int(contract["policy"]["seed"])
    policy = fit_conditional_patch_policy(
        [_training_row(row) for row in train], smoothing_alpha=alpha
    )
    where_density = fit_weighted_density(
        [region_features(row["source"], row["binding"]) for row in train],
        _balanced_region_weights(train),
        variance_floor=0.05,
    )
    checkpoint = _checkpoint_for_fold(
        fold=fold_index,
        policy=policy,
        where_density=where_density,
        train_rows=train,
        held_rows=held,
        contract=contract,
        revision=revision,
    )
    fold_root = output_root / f"fold_{fold_index}"
    _publish(fold_root / "runtime_checkpoint.json", checkpoint)

    evaluated = []
    for row_index, row in enumerate(held):
        row_seed = int(
            identity(
                {
                    "seed": seed,
                    "fold": fold_index,
                    "row": row_index,
                    "subgoal": row["patch"].subgoal_id,
                }
            )[:16],
            16,
        )
        teacher_where = _where_key(row["context"])
        autonomous_where = _sample_where_fiber(
            row["source"], policy, seed=row_seed, size=budget - 1
        )
        autonomous_where_ids = {_where_key(context) for _, context in autonomous_where}
        where_by_id = {_where_key(context): (slots, context) for slots, context in autonomous_where}
        where_by_id.setdefault(teacher_where, (row["binding"], row["context"]))
        learned_where_scores = []
        marginal_where_scores = []
        for candidate_id, (slots, _) in sorted(where_by_id.items()):
            size_logp = _size_log_probability(policy, len(slots))
            learned_where_scores.append(
                (
                    candidate_id,
                    size_logp + where_density.score(region_features(row["source"], slots)),
                )
            )
            marginal_where_scores.append((candidate_id, size_logp))
        learned_where_rank, learned_where_probability = _softmax_probability_and_rank(
            learned_where_scores, teacher_where
        )
        marginal_where_rank, marginal_where_probability = _softmax_probability_and_rank(
            marginal_where_scores, teacher_where
        )

        target_fiber = _teacher_perturbation_fiber(
            row["context"], row["tokens"], seed=row_seed ^ 0x5A17, size=budget
        )
        teacher_target = row["patch"].subgoal_id
        scores_by_arm = {}
        token_rows_by_arm = {}
        for arm, learned in (("learned", True), ("marginal", False)):
            candidate_scores = []
            for patch, tokens in target_fiber:
                score = policy.score_stream(row["context"], tokens, learned=learned)
                candidate_scores.append((patch.subgoal_id, score["log_probability"]))
            target_rank, target_probability = _softmax_probability_and_rank(
                candidate_scores, teacher_target
            )
            teacher_score = policy.score_stream(row["context"], row["tokens"], learned=learned)
            scores_by_arm[arm] = {
                "target_rank": target_rank,
                "target_probability": target_probability,
                "teacher_joint_nll": teacher_score["nll"],
                "teacher_mean_token_nll": teacher_score["mean_nll"],
                "factor_nll": teacher_score["factor_nll"],
                "factor_counts": teacher_score["factor_counts"],
            }
            token_rows_by_arm[arm] = _token_diagnostics(
                policy, row["context"], row["tokens"], learned=learned
            )
        count_probability = _program_count_probability(policy, int(row["route_region_count"]))
        count_values = np.asarray(policy.program_count_counts, dtype=float) + alpha
        count_order = sorted(range(1, 5), key=lambda value: (-count_values[value - 1], value))
        count_rank = count_order.index(int(row["route_region_count"])) + 1
        evaluated.append(
            {
                "teacher_program_id": row["route_id"],
                "source_group": row["source_group"],
                "region_index": row["region_index"],
                "teacher_subgoal_id": teacher_target,
                "token_count": len(row["tokens"]),
                "candidate_fibers": {
                    "where": len(where_by_id),
                    "target_patch": len(target_fiber),
                    "matched_budget": budget,
                    "target_patch_teacher_injected": True,
                    "where_teacher_autonomously_present": teacher_where in autonomous_where_ids,
                },
                "control": {
                    "route_region_count": row["route_region_count"],
                    "continue_or_stop": row["control_after"],
                    "program_count_probability": count_probability,
                    "program_count_rank": count_rank,
                },
                "where": {
                    "learned_rank": learned_where_rank,
                    "learned_probability": learned_where_probability,
                    "marginal_rank": marginal_where_rank,
                    "marginal_probability": marginal_where_probability,
                },
                "arms": scores_by_arm,
                "conditional_token_diagnostics": token_rows_by_arm,
                "support": {
                    "complete_patch_supported": True,
                    "complete_patch_roundtrip_exact": True,
                },
            }
        )

    arm_summary = {}
    for arm in ("learned", "marginal"):
        target_ranks = [row["arms"][arm]["target_rank"] for row in evaluated]
        where_ranks = [row["where"][f"{arm}_rank"] for row in evaluated]
        factor_nll = Counter()
        factor_counts = Counter()
        for row in evaluated:
            factor_nll.update(row["arms"][arm]["factor_nll"])
            factor_counts.update(row["arms"][arm]["factor_counts"])
        arm_summary[arm] = {
            "where_teacher_rank": _rank_summary(where_ranks),
            "complete_patch_teacher_rank": _rank_summary(target_ranks),
            "mean_complete_patch_nll": float(
                np.mean([row["arms"][arm]["teacher_joint_nll"] for row in evaluated])
            ),
            "mean_token_nll": float(
                np.mean([row["arms"][arm]["teacher_mean_token_nll"] for row in evaluated])
            ),
            "mean_teacher_probability_in_diagnostic_fiber": float(
                np.mean([row["arms"][arm]["target_probability"] for row in evaluated])
            ),
            "factor_mean_nll": {
                factor: factor_nll[factor] / factor_counts[factor]
                for factor in sorted(factor_counts)
            },
        }
    payload = {
        "schema_version": GATE3_SCHEMA,
        "evidence": "computed zero-oracle teacher-forced held-source diagnostic",
        "fold": fold_index,
        "implementation_revision": revision,
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256_file(contract_path),
            "payload_sha256": identity(contract),
        },
        "split": {
            "train_source_hashes": sorted(identity(value) for value in train_sources),
            "held_source_hashes": sorted(identity(value) for value in held_sources),
            "train_regions": len(train),
            "held_regions": len(held),
        },
        "candidate_budget": budget,
        "candidate_fiber_note": "WHERE is an autonomous same-budget source-role fiber with teacher added only when absent; target-patch ranks use one identical teacher-conditioned local perturbation fiber for both arms and are not autonomous recovery.",
        "gate": {
            "coverage": {
                "supported": len(evaluated),
                "denominator": len(held),
                "coverage": len(evaluated) / len(held),
            },
            "precision": {
                "roundtrip_exact": len(evaluated),
                "admitted": len(evaluated),
                "precision": 1.0,
            },
        },
        "cross_region_dependency_learning": {
            "status": "abstain_no_positive_created_role_reference_examples",
            "observed_positive_examples": 0,
        },
        "arms": arm_summary,
        "rows": evaluated,
        "compute": {
            "cpu_workers": 1,
            "wall_seconds": time.monotonic() - started,
            "precision": "float64",
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
    }
    _publish(fold_root / "gate3.json", payload)
    return payload


def gate3_aggregate(output_root: Path, output: Path) -> dict:
    """Reduce the three independent fold artifacts in fixed fold order."""

    folds = []
    for fold in range(3):
        path = output_root / f"fold_{fold}" / "gate3.json"
        envelope = json.loads(path.read_text())
        payload = envelope.get("payload")
        if (
            not isinstance(payload, dict)
            or envelope.get("payload_sha256") != identity(payload)
            or payload.get("schema_version") != GATE3_SCHEMA
            or int(payload.get("fold", -1)) != fold
            or any(payload["costs"].values())
        ):
            raise ValueError(f"invalid Gate 3 fold artifact: {path}")
        folds.append((path, payload))
    rows = [row for _, fold in folds for row in fold["rows"]]
    if len(rows) != 147:
        raise RuntimeError(f"Gate 3 aggregate held-region census changed: {len(rows)}")
    arms = {}
    for arm in ("learned", "marginal"):
        target_ranks = [row["arms"][arm]["target_rank"] for row in rows]
        where_ranks = [row["where"][f"{arm}_rank"] for row in rows]
        factor_nll = Counter()
        factor_counts = Counter()
        token_ranks: dict[str, list[int]] = {}
        token_nll: dict[str, list[float]] = {}
        for row in rows:
            factor_nll.update(row["arms"][arm]["factor_nll"])
            factor_counts.update(row["arms"][arm]["factor_counts"])
            for token in row["conditional_token_diagnostics"][arm]:
                token_ranks.setdefault(token["factor"], []).append(token["teacher_rank"])
                token_nll.setdefault(token["factor"], []).append(token["teacher_nll"])
        arms[arm] = {
            "where_teacher_rank": _rank_summary(where_ranks),
            "conditional_complete_patch_teacher_rank": _rank_summary(target_ranks),
            "mean_complete_patch_nll": float(
                np.mean([row["arms"][arm]["teacher_joint_nll"] for row in rows])
            ),
            "mean_token_nll": float(
                np.mean([row["arms"][arm]["teacher_mean_token_nll"] for row in rows])
            ),
            "mean_teacher_probability_in_diagnostic_fiber": float(
                np.mean([row["arms"][arm]["target_probability"] for row in rows])
            ),
            "factor_mean_nll": {
                factor: factor_nll[factor] / factor_counts[factor]
                for factor in sorted(factor_counts)
            },
            "factor_conditional_rank": {
                factor: {
                    **_rank_summary(token_ranks[factor]),
                    "mean_nll": float(np.mean(token_nll[factor])),
                }
                for factor in sorted(token_ranks)
            },
        }
    teacher_where_present = sum(
        row["candidate_fibers"]["where_teacher_autonomously_present"] for row in rows
    )
    payload = {
        "schema_version": GATE3_AGGREGATE_SCHEMA,
        "evidence": "computed deterministic reduction of three independent zero-oracle held-source folds",
        "fold_artifacts": [
            {
                "fold": fold,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "payload_sha256": identity(payload),
            }
            for fold, (path, payload) in enumerate(folds)
        ],
        "census": {
            "folds": 3,
            "held_regions": len(rows),
            "matched_candidate_budget": folds[0][1]["candidate_budget"],
            "where_teacher_autonomously_present": teacher_where_present,
            "target_patch_teacher_injected_diagnostics": len(rows),
        },
        "gate": {
            "coverage": {
                "supported": len(rows),
                "denominator": 147,
                "coverage": len(rows) / 147,
            },
            "precision": {
                "roundtrip_exact": len(rows),
                "admitted": len(rows),
                "precision": 1.0,
            },
            "learned_beats_marginal_at_conditional_patch_top8": arms["learned"][
                "conditional_complete_patch_teacher_rank"
            ]["top8"]
            > arms["marginal"]["conditional_complete_patch_teacher_rank"]["top8"],
            "passed": False,
        },
        "arms": arms,
        "joint_rank_status": {
            "status": "abstain_not_identifiable_from_separate_where_and_teacher_conditioned_patch_fibers",
            "reason": "A joint WHERE-plus-patch candidate fiber was not enumerated; the reported complete-patch rank is conditional on the teacher WHERE. No joint rank is fabricated.",
        },
        "cross_region_dependency_learning": {
            "status": "abstain_no_positive_created_role_reference_examples",
            "observed_positive_examples": 0,
        },
        "decision": "negative_gate3_conditional_policy_does_not_beat_matched_marginal_control",
        "compute": {
            "parallel_cpu_workers": 3,
            "fold_wall_seconds": [fold["compute"]["wall_seconds"] for _, fold in folds],
            "aggregate_wall_seconds_estimate": max(
                fold["compute"]["wall_seconds"] for _, fold in folds
            ),
            "summed_cpu_process_seconds": sum(fold["compute"]["wall_seconds"] for _, fold in folds),
            "precision": "float64",
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
    }
    _publish(output, payload)
    return payload


def gate4_blocked(gate3_path: Path, output: Path) -> dict:
    """Record that autonomous generation is not promoted after failed Gate 3."""

    envelope = json.loads(gate3_path.read_text())
    gate3 = envelope.get("payload")
    if (
        not isinstance(gate3, dict)
        or envelope.get("payload_sha256") != identity(gate3)
        or gate3.get("schema_version") != GATE3_AGGREGATE_SCHEMA
    ):
        raise ValueError(f"invalid Gate 3 aggregate artifact: {gate3_path}")
    if gate3["gate"]["passed"]:
        raise ValueError("Gate 4 cannot be marked blocked after a passing Gate 3")
    payload = {
        "schema_version": GATE4_BLOCKED_SCHEMA,
        "evidence": "blocked zero-oracle milestone outcome; no autonomous generation was run",
        "gate3": {
            "path": str(gate3_path.relative_to(ROOT)),
            "sha256": sha256_file(gate3_path),
            "payload_sha256": identity(gate3),
        },
        "status": "blocked_by_failed_gate3_promotion_rule",
        "reason": "The learned complete-patch policy did not beat the matched source-balanced marginal at conditional patch Top-8, so this checkpoint is not promoted to autonomous Gate 4 generation.",
        "autonomous_programs_generated": 0,
        "autonomous_endpoint_or_transformation_recovery": "not_measured",
        "claims_prohibited": [
            "autonomous complete-program recovery",
            "autonomous endpoint recovery",
            "optimization utility",
        ],
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
    }
    _publish(output, payload)
    return payload


def support_gate(output: Path, *, contract_path: Path = CONTRACT) -> dict:
    contract = load_contract(contract_path)
    revision = _require_clean()
    support = unseal(SUPPORT)
    teacher = teacher_traces()
    by_program = {row["program_id"]: row for row in teacher}
    if len(by_program) != 77:
        raise RuntimeError("teacher route census changed")
    metadata = source_group_map(unseal(BENCHMARK), json.loads(SEEDS.read_text()))
    folds = predeclared_source_folds(metadata)
    rows = []
    for support_row in support["rows"]:
        program_id = support_row["teacher_program_id"]
        teacher_row = by_program.get(program_id)
        if teacher_row is None:
            raise ValueError(f"support route lacks exact teacher provenance: {program_id}")
        program = CompleteRegionProgram.from_payload(support_row["runtime_program"])
        for region_index, decision in enumerate(program.decisions):
            tokens = encode_patch_stream(decision.patch)
            covered, reason = patch_stream_support(decision.patch)
            decoded = (
                decode_patch_stream(SourceRegionContext.from_subgoal(decision.patch), tokens)
                if covered
                else None
            )
            rows.append(
                {
                    "teacher_program_id": program_id,
                    "source_group": teacher_row["source_group"],
                    "region_index": region_index,
                    "subgoal_id": decision.patch.subgoal_id,
                    "token_count": len(tokens),
                    "factor_token_counts": dict(
                        sorted(Counter(token.factor for token in tokens).items())
                    ),
                    "supported": covered,
                    "roundtrip_exact": decoded == decision.patch,
                    "failure_reason": reason,
                    "cross_region_created_role_references": sum(
                        value is not None for value in decision.input_provenance
                    ),
                    "tokens": [token.payload() for token in tokens],
                }
            )
    if len(rows) != 147:
        raise RuntimeError(f"complete-region support census changed: {len(rows)}")
    fold_rows = []
    for split in folds:
        train_sources = set(split["train_sources"])
        held_sources = set(split["test_sources"])
        train = [row for row in rows if row["source_group"] in train_sources]
        held = [row for row in rows if row["source_group"] in held_sources]
        train_tokens = {
            _token_key(PatchToken.from_payload(token)) for row in train for token in row["tokens"]
        }
        held_token_rows = [
            _token_key(PatchToken.from_payload(token)) for row in held for token in row["tokens"]
        ]
        supported = sum(row["supported"] for row in held)
        exact = sum(row["roundtrip_exact"] for row in held)
        fold_rows.append(
            {
                "fold": split["fold"],
                "train_sources": sorted(train_sources),
                "held_sources": sorted(held_sources),
                "train_regions": len(train),
                "held_regions": len(held),
                "held_supported": supported,
                "held_support_coverage": supported / len(held),
                "held_roundtrip_exact": exact,
                "held_roundtrip_precision": exact / max(1, supported),
                "training_observed_token_vocabulary_size": len(train_tokens),
                "held_token_instances": len(held_token_rows),
                "held_token_instances_not_observed_in_train": sum(
                    token not in train_tokens for token in held_token_rows
                ),
                "held_unique_tokens_not_observed_in_train": len(
                    set(held_token_rows) - train_tokens
                ),
                "support_note": "Unseen empirical tokens remain representable only through the frozen platform value grammar; they are not leaked into training-fold learned statistics.",
            }
        )
    supported = sum(row["supported"] for row in rows)
    exact = sum(row["roundtrip_exact"] for row in rows)
    payload = {
        "schema_version": SUPPORT_SCHEMA,
        "evidence": "computed split-first zero-oracle complete-patch grammar support",
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256_file(contract_path),
            "payload_sha256": identity(contract),
        },
        "implementation_revision": revision,
        "fixed_platform_domains": {
            "atom_types": list(ATOM_TYPES),
            "formal_charges": list(FORMAL_CHARGES),
            "implicit_hydrogens": list(HYDROGEN_COUNTS),
            "degrees": list(DEGREES),
            "bond_orders": list(BOND_ORDERS),
        },
        "census": {
            "routes": 77,
            "regions": len(rows),
            "cross_region_created_role_references": sum(
                row["cross_region_created_role_references"] for row in rows
            ),
            "factor_token_counts": dict(
                sorted(Counter(token["factor"] for row in rows for token in row["tokens"]).items())
            ),
        },
        "gate": {
            "teacher_patch_grammar_support": {
                "covered": supported,
                "denominator": len(rows),
                "coverage": supported / len(rows),
            },
            "roundtrip_precision": {
                "exact": exact,
                "admitted": supported,
                "precision": exact / max(1, supported),
            },
            "passed": supported == exact == 147,
        },
        "folds": fold_rows,
        "rows": rows,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
    }
    if not payload["gate"]["passed"]:
        payload["decision"] = "stop_before_fit_missing_complete_patch_grammar_support"
    else:
        payload["decision"] = "complete_patch_grammar_support_gate_passed_fit_authorized"
    _publish(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        choices=("support", "gate3", "gate3-aggregate", "gate4-blocked"),
        default="support",
    )
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "support.json")
    arguments = parser.parse_args()
    if arguments.gate == "support":
        result = support_gate(
            arguments.output.resolve(), contract_path=arguments.contract.resolve()
        )
        summary = {"census": result["census"], "gate": result["gate"]}
    elif arguments.gate == "gate3":
        if arguments.fold is None:
            parser.error("--gate gate3 requires --fold")
        result = gate3_fold(
            arguments.output.resolve(),
            fold_index=arguments.fold,
            contract_path=arguments.contract.resolve(),
        )
        summary = {
            "fold": result["fold"],
            "gate": result["gate"],
            "arms": result["arms"],
            "compute": result["compute"],
        }
    elif arguments.gate == "gate3-aggregate":
        result = gate3_aggregate(DEFAULT_OUTPUT / "gate3", arguments.output.resolve())
        summary = {"census": result["census"], "gate": result["gate"], "arms": result["arms"]}
    else:
        result = gate4_blocked(DEFAULT_OUTPUT / "gate3_aggregate.json", arguments.output.resolve())
        summary = {
            "status": result["status"],
            "autonomous_programs_generated": result["autonomous_programs_generated"],
        }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
