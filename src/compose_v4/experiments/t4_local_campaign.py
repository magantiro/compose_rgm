"""Local T4 program-only loop with immutable query locks and explicit resume.

The caller supplies an identified evaluator and any model assets. The loop does
not download assets, launch jobs, or substitute missing observations. It supports
the three declared proposal lanes, not the separate support-escalation policy.
"""

from __future__ import annotations

import fcntl
import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from numbers import Real
from pathlib import Path

import numpy as np
from rdkit import Chem

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.program_campaign import ProgramQueryLedger
from compose_v4.control.program_task import ProgramTask
from compose_v4.control.reference_guidance import GuidanceConfig
from compose_v4.control.reference_programs import (
    FrozenProgramReference,
    ProgramPanelGuidance,
    t4_program_input,
)
from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert
from compose_v4.control.structural_subgoal_realizer import RealizerConfig
from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.experiments.t4_fiber_campaign import (
    COMPOSE_VALID,
    PROPOSAL_SLOTS,
    REPRESENTABLE_HEAVY_ATOMS,
    Fiber,
    expand,
)
from compose_v4.experiments.t4_integrated_route_fiber import (
    EXPERTS,
    attach_features,
    merge_expert_pools,
    select_batch,
)
from compose_v4.experiments.t4_route_proposals import RouteProposalConfig, expand_route
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state


@dataclass(frozen=True)
class T4RunConfig:
    lead: str
    delta: float
    seed: int
    budget: int = 250
    batch: int = 8
    max_rounds: int = 250
    parents: int = 3
    parent_exploration: float = 0.3
    draws_per_parent: int = 8
    horizon: int = 3
    compiler_expansions: int = 4000
    expert_floor_rounds: int = 2
    route_scale_floor_rounds: int = 2
    endpoint_support: str = COMPOSE_VALID
    lanes: tuple[str, ...] = EXPERTS
    route: RouteProposalConfig = field(default_factory=RouteProposalConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.guidance, GuidanceConfig) or not isinstance(
            self.route, RouteProposalConfig
        ):
            raise TypeError("guidance and route require their typed configuration objects")
        if self.delta not in (0.4, 0.6):
            raise ValueError("T4 delta must be 0.4 or 0.6")
        for name in (
            "budget",
            "batch",
            "max_rounds",
            "parents",
            "draws_per_parent",
            "horizon",
            "compiler_expansions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "expert_floor_rounds", "route_scale_floor_rounds"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.horizon > 3:
            raise ValueError("horizon must be at most 3")
        if not math.isfinite(self.parent_exploration) or not 0 <= self.parent_exploration <= 1:
            raise ValueError("parent_exploration must be in [0, 1]")
        if (
            not isinstance(self.lanes, tuple)
            or not self.lanes
            or len(set(self.lanes)) != len(self.lanes)
            or set(self.lanes) - set(EXPERTS)
        ):
            raise ValueError(f"lanes must be an explicit distinct tuple from {EXPERTS}")
        Fiber(self.lead, self.delta, support=self.endpoint_support)
        source = smiles_to_molecular_graph(self.lead)
        if source.n_atoms > REPRESENTABLE_HEAVY_ATOMS:
            raise ValueError("initial lead exceeds the declared molecular representation")
        pad_molecular_graph(source, PROPOSAL_SLOTS)


def _save(path: Path, payload: dict) -> str:
    digest = identity(payload)
    envelope = {"payload": payload, "payload_sha256": digest}
    write_bytes_if_absent(
        path, json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return digest


def _load(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"payload", "payload_sha256"}
        or identity(envelope["payload"]) != envelope["payload_sha256"]
    ):
        raise ValueError(f"invalid immutable run record: {path}")
    return envelope["payload"]


@contextmanager
def _exclusive_writer(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another writer holds this run directory: {output}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def generate_panel(
    parents: list[str],
    state: SearchState,
    fiber: Fiber,
    rng: np.random.Generator,
    config: T4RunConfig,
    expert: RouteDistilledGoalExpert | None,
) -> tuple[list[dict], dict]:
    """Construct the declared lane union without evaluating the objective."""
    pools: dict[str, list[dict]] = {lane: [] for lane in config.lanes}
    telemetry = []
    for parent in parents:
        for lane in config.lanes:
            if lane == "route_complete_region":
                if expert is None:
                    raise ValueError("route_complete_region requires a route expert")
                rows, work = expand_route(
                    parent,
                    state.archive[parent],
                    fiber,
                    expert,
                    config=config.route,
                    include_realized_actions=True,
                )
                telemetry.append({"parent": parent, "lane": lane, **work})
            else:
                rows = expand(
                    parent,
                    state.archive[parent],
                    fiber,
                    rng,
                    draws=config.draws_per_parent,
                    horizon=config.horizon,
                    proposal_lane=lane,
                    include_realized_actions=True,
                    realizer_config=RealizerConfig(maximum_expansions=config.compiler_expansions),
                )
                telemetry.append({"parent": parent, "lane": lane, "eligible": len(rows)})
            pools[lane].extend(rows)
    merged = merge_expert_pools(pools)
    fresh = [row for row in merged if row["smiles"] not in state.archive]
    return fresh, {
        "lanes": telemetry,
        "unique": len(merged),
        "already_scored": len(merged) - len(fresh),
    }


def _serializable_row(row: dict) -> dict:
    return {
        **row,
        "features": np.asarray(row["features"]).tolist(),
        "fingerprint": sorted(row["fingerprint"]),
    }


def score_blind_parent_probabilities(
    archive: dict[str, float], exhaustion: dict[str, int], exploration: float
) -> tuple[list[str], np.ndarray]:
    """Sample measured parents without using their docking scores.

    Repeatedly exhausted parents lose proposal mass, but retain the uniform
    exploration floor. This is a proposal rule, not an objective-value model.
    """
    parents = sorted(archive)
    if not parents:
        return [], np.empty(0, dtype=float)
    counts = np.asarray([exhaustion.get(parent, 0) for parent in parents], dtype=float)
    if np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("parent exhaustion counts must be finite and nonnegative")
    weights = 1.0 / (1.0 + counts)
    weights /= weights.sum()
    probabilities = exploration / len(parents) + (1.0 - exploration) * weights
    return parents, probabilities


def _after_parent_exhaustion(
    before: dict[str, int], parents: list[str], candidates: list[dict]
) -> dict[str, int]:
    available = {row["parent"] for row in candidates}
    after = dict(before)
    for parent in parents:
        after[parent] = 0 if parent in available else after.get(parent, 0) + 1
    return after


def executable_panel(candidates: list[dict], state: SearchState) -> tuple[list[dict], list[dict]]:
    """Separate construction abstentions before reference scoring or selection.

    All admitted candidates replay from their exact measured parent. No neural
    scoring support is required at this boundary. Missing or zero-step programs
    remain in the construction record, not the expensive-evaluation pool.
    Malformed existing traces are errors, not ordinary construction abstentions.
    """
    eligible, abstentions = [], []
    for row in candidates:
        actions = row.get("realized_actions")
        if not actions:
            abstentions.append(
                {
                    "reason": "missing_primitive_program"
                    if actions is None
                    else "zero_primitive_program",
                    "proposal": row,
                }
            )
            continue
        if (
            row["parent"] not in state.archive
            or row["parent_score"] != state.archive[row["parent"]]
        ):
            raise ValueError("candidate parent is not the recorded measured archive state")
        if canonical_state_key(decode_state(row["source_state"])) != row["parent"]:
            raise ValueError("candidate program source differs from its measured parent")
        t4_program_input(row, candidate_id=row["smiles"])
        eligible.append(row)
    return eligible, abstentions


def run_t4(
    *,
    output: Path,
    config: T4RunConfig,
    evaluate: Callable[[str], float],
    evaluator_identity: dict,
    implementation_identity: dict,
    reference: FrozenProgramReference | None = None,
    route_expert: RouteDistilledGoalExpert | None = None,
    asset_identity: dict | None = None,
    resume: bool = False,
) -> dict:
    """Run or resume one lead under a fixed budget, with no implicit downloads.

    All scores, including the starting lead, are charged. The lead may fail the
    endpoint gate but still supplies the measured parent score. Only eligible
    molecules contribute to the reported best feasible score. Exceptions leave a
    reservation and block retry. A locked round with resolved receipts resumes
    from those receipts without generating or scoring the panel again. Only
    completed rounds update the archive, not a program-value model. ``evaluate`` must use exactly the
    supplied evaluator identity. The CLI builds that adapter from verified assets.
    """
    if config.guidance.mode != "off" and reference is None:
        raise ValueError("shadow/active selection requires a frozen reference")
    if config.guidance.mode == "off" and reference is not None:
        raise ValueError("off mode must not receive a loaded reference")
    if ("route_complete_region" in config.lanes) != (route_expert is not None):
        raise ValueError("route expert must be supplied exactly when the route lane is enabled")
    if not evaluator_identity or not implementation_identity:
        raise ValueError("evaluator and implementation identities are required")
    output = Path(output)
    manifest = {
        "schema": "compose.t4.local_run.v1",
        "configuration": asdict(config),
        "evaluator": evaluator_identity,
        "implementation": implementation_identity,
        "assets": asset_identity or {},
        "construction_gate": "nonempty_exact_primitive_replay_from_measured_parent_v1",
        "parent_policy": "score_blind_exhaustion_tempered_v1",
        "program_value_model": "none",
        "reference": None if reference is None else reference.identity(),
        "route_expert": None if route_expert is None else identity(route_expert.checkpoint()),
    }
    # JSON form is also the resume contract, including tuples and scalar types.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    with _exclusive_writer(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"run already exists, use explicit resume: {output}")
            if _load(manifest_path) != manifest:
                raise ValueError(
                    "run configuration, code, environment, evaluator or asset changed on resume"
                )
        elif resume:
            raise FileNotFoundError(f"no run manifest to resume: {manifest_path}")
        else:
            unexpected = [path.name for path in output.iterdir() if path.name != ".writer.lock"]
            if unexpected:
                raise FileExistsError(f"new run directory is not empty: {unexpected}")
            _save(manifest_path, manifest)
        return _run(output, config, evaluate, manifest, reference, route_expert)


def _run(output, config, evaluate, manifest, reference, expert):
    task = ProgramTask("t4_local", identity(manifest["evaluator"]), "t4", config.lead, config.delta)
    fiber = Fiber(config.lead, config.delta, support=config.endpoint_support)
    source_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(config.lead))

    def checked_evaluate(smiles):
        if smiles != source_smiles and fiber.check(smiles) is None:
            raise ValueError("locked endpoint fails the declared eligibility gate")
        score = evaluate(smiles)
        if (
            isinstance(score, (bool, np.bool_))
            or not isinstance(score, Real)
            or not math.isfinite(score)
        ):
            raise ValueError(
                "evaluator must return a finite real score, not a missing value or Boolean"
            )
        return float(score)

    ledger = ProgramQueryLedger(output / "oracle", task, checked_evaluate, budget=config.budget)
    root = fiber.check(config.lead) or {
        "smiles": source_smiles,
        "endpoint_eligible": False,
    }
    if "endpoint_eligible" not in root:
        root = {**root, "endpoint_eligible": True}
    root_lock = _save(
        output / "initialization.json", {"manifest_id": identity(manifest), "lead": root}
    )
    root_answer = ledger.query(root["smiles"], lock_id=root_lock, role="initialization")
    if root_answer["lock_id"] != root_lock or root_answer["role"] != "initialization":
        raise ValueError("initialization receipt belongs to a different query lock")
    state = SearchState(archive={root["smiles"]: root_answer["score"]}, budget=config.budget - 1)
    rng = np.random.default_rng(config.seed)
    parent_exhaustion: dict[str, int] = {}
    value = ProgramValue()
    status = "round_limit"
    accounted = {root_answer["receipt_id"]}
    for index in range(1, config.max_rounds + 1):
        if state.budget == 0:
            status = "budget_exhausted"
            break
        folder = output / f"round_{index:06d}"
        lock_path = folder / "lock.json"
        before = identity(
            {
                "archive": state.archive,
                "rng": rng.bit_generator.state,
                "parent_exhaustion": parent_exhaustion,
            }
        )
        if lock_path.exists():
            lock = _load(lock_path)
            if lock["before"] != before or lock["manifest_id"] != identity(manifest):
                raise ValueError(f"round state or manifest mismatch: {lock_path}")
        else:
            parent_names, probabilities = score_blind_parent_probabilities(
                state.archive, parent_exhaustion, config.parent_exploration
            )
            chosen_indices = rng.choice(
                len(parent_names),
                size=min(config.parents, len(parent_names)),
                replace=False,
                p=probabilities,
            )
            parents = [parent_names[int(position)] for position in chosen_indices]
            candidates, telemetry = generate_panel(parents, state, fiber, rng, config, expert)
            candidates, abstentions = executable_panel(candidates, state)
            rows = attach_features(candidates, state, fiber)
            if len(rows) != len(candidates):
                raise RuntimeError("feature preparation changed the eligible proposal panel")
            programs = tuple(t4_program_input(row, candidate_id=row["smiles"]) for row in rows)
            guide = ProgramPanelGuidance(
                programs, reference, config.guidance, identity_field="smiles"
            )
            receipts = []
            chosen = select_batch(
                rows,
                value,
                state,
                rng,
                round_index=index,
                batch=min(config.batch, state.budget),
                exploration=min(config.batch, state.budget),
                expert_floor_rounds=config.expert_floor_rounds,
                route_scale_floor_rounds=config.route_scale_floor_rounds,
                reference_guide=guide,
                reference_receipts=receipts,
            )
            lock = {
                "manifest_id": identity(manifest),
                "before": before,
                "round": index,
                "parents": parents,
                "parent_probabilities": dict(
                    zip(parent_names, probabilities.tolist(), strict=True)
                ),
                "parent_exhaustion_after": _after_parent_exhaustion(
                    parent_exhaustion, parents, rows
                ),
                "telemetry": telemetry,
                "reference_selection": receipts,
                "construction_abstentions": abstentions,
                "candidates": [_serializable_row(row) for row in rows],
                "selected": [_serializable_row(row) for row in chosen],
                "rng_after": rng.bit_generator.state,
            }
            _save(lock_path, lock)
        if lock["parent_exhaustion_after"] != _after_parent_exhaustion(
            parent_exhaustion, lock["parents"], lock["candidates"]
        ):
            raise ValueError(f"parent exhaustion accounting changed: {lock_path}")
        parent_exhaustion = lock["parent_exhaustion_after"]
        chosen = lock["selected"]
        if len(chosen) > state.budget:
            raise ValueError(f"locked batch exceeds remaining budget: {lock_path}")
        rng.bit_generator.state = lock["rng_after"]
        if not chosen:
            status = "candidate_exhausted"
            break
        prior_best = state.incumbent
        answers = []
        for row in chosen:
            checked, abstentions = executable_panel([row], state)
            if not checked or abstentions:
                raise ValueError("locked query does not contain an executable nonempty program")
            answer = ledger.query(row["smiles"], lock_id=identity(lock), role="candidate")
            if answer["lock_id"] != identity(lock) or answer["receipt_id"] in accounted:
                raise ValueError(
                    "query receipt belongs to another lock or duplicates a charged endpoint"
                )
            accounted.add(answer["receipt_id"])
            answers.append(answer["receipt_id"])
            state.archive[row["smiles"]] = answer["score"]
        state.budget -= len(chosen)
        state.rounds = index
        state.history.append({"round": index, "improved": state.incumbent < prior_best})
        _save(folder / "complete.json", {"lock_id": identity(lock), "receipts": answers})
    if state.budget == 0:
        status = "budget_exhausted"
    if len(accounted) != len(ledger.rows):
        raise ValueError("query ledger contains observations outside the run's locked batches")
    feasible_archive = {
        smiles: score for smiles, score in state.archive.items() if fiber.check(smiles) is not None
    }
    best_feasible = min(feasible_archive, key=feasible_archive.get) if feasible_archive else None
    result = {
        "schema": "compose.t4.local_result.v1",
        "manifest_id": identity(manifest),
        "status": status,
        "charged_calls": len(ledger.rows),
        "budget": config.budget,
        "rounds": state.rounds,
        "archive": state.archive,
        "feasible_archive_count": len(feasible_archive),
        "best_smiles": best_feasible,
        "best_score": feasible_archive[best_feasible] if best_feasible is not None else None,
        "rng": rng.bit_generator.state,
        "history": state.history,
    }
    _save(output / "result.json", result)
    return result
