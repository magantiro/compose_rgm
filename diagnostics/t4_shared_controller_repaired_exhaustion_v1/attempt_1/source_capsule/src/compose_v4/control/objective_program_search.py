"""Objective-driven Dynamic portfolio, sharing exact execution and one archive.

The unchanged v0 engine owns mutation/recombination/program composition. A
separate persistent RNG supplies progressive structured proposals. Candidate
competition happens after exact execution, final eligibility and deduplication.
No pretrained teacher route, endpoint, or target-specific selection rule is used.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from time import perf_counter

import numpy as np

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import DynamicProgramOptimizer
from compose_v4.control.progressive_structured_sampler import synthesize_progressive_program
from compose_v4.rewrite.trace_shard import decode_state

LANES = ("shallow", "structured")


@dataclass(frozen=True)
class ObjectiveSearchConfig:
    parent_allocation: str = "niche_score"
    structured_attempts: int = 32
    structured_candidates: int = 8
    structured_wall_seconds: float = 45.0
    query_batch_size: int = 8
    channel_exploration: float = 0.5
    improvement_scale: float = 1.0

    def __post_init__(self):
        if self.parent_allocation not in {"score_blind", "score_rank", "niche_score"}:
            raise ValueError("unsupported objective parent allocation")
        for name in ("structured_attempts", "structured_candidates", "query_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "query_batch_size" else 0):
                raise ValueError(f"{name} must be a valid bounded integer")
        for name in ("structured_wall_seconds", "channel_exploration", "improvement_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


def v0_search_config(*, seed=20260913, parent_allocation="score_blind"):
    """Historical v0 mechanics, explicit rather than inherited changing defaults."""
    return replace(
        ProgramSearchConfig.two_program_recipe(seed=seed),
        max_composed_programs=3,
        current_state_edit_probability=0.0,
        proposal_cache_entries=128,
        wall_seconds=45.0,
        parent_allocation=parent_allocation,
    )


class ObjectiveProgramSearch:
    def __init__(
        self,
        config: ProgramSearchConfig,
        policy: ObjectiveSearchConfig,
        *,
        source_group: str,
        oracle_protocol: str,
    ):
        if config.score_direction != "minimize":
            raise ValueError("this T4 revision supports minimize objectives only")
        self.policy = policy
        self.base = DynamicProgramOptimizer(
            replace(config, parent_allocation=policy.parent_allocation),
            source_group=source_group,
            oracle_protocol=oracle_protocol,
        )
        self.structured_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 260916, 2])
        )
        self.allocation_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 260916, 3])
        )
        self.channel_stats = {
            lane: {"charged": 0, "scored": 0, "reward_sum": 0.0, "incumbent_improvements": 0}
            for lane in LANES
        }

    def add_measured_program(self, candidate, *, receipt_id, score):
        incumbent = min((r["score"] for r in self.base.observations.values()), default=math.inf)
        already_recorded = receipt_id in self.base.observations
        key = self.base.add_measured_program(
            candidate, receipt_id=receipt_id, score=score, static_score=score
        )
        if not already_recorded:
            stats = self.channel_stats[self.lane(candidate)]
            stats["charged"] += 1
            stats["scored"] += 1
            if math.isfinite(incumbent) and score < incumbent:
                stats["incumbent_improvements"] += 1
                stats["reward_sum"] += min(1.0, (incumbent - score) / self.policy.improvement_scale)
        return key

    def _structured_pool(self, eligibility, seen):
        began = perf_counter()
        candidates, attempts = [], []
        keys, weights = self.base.selection()
        for index in range(self.policy.structured_attempts):
            if (
                len(candidates) >= self.policy.structured_candidates
                or perf_counter() - began >= self.policy.structured_wall_seconds
            ):
                break
            position = int(self.structured_rng.choice(len(keys), p=weights))
            entry = self.base.entries[keys[position]]
            parent_scores = [
                r["score"]
                for r in self.base.observations.values()
                if r["endpoint"] == entry["endpoint"]
            ]
            parent = {
                "entry_id": entry["entry_id"],
                "parent_probability": float(weights[position]),
                "parent_measured_score": float(np.mean(parent_scores)),
            }
            try:
                source = decode_state(entry["trace"]["states"][-1])
                _, program, binding, trace, detail = synthesize_progressive_program(
                    source,
                    self.structured_rng,
                    max_primitives=self.base.config.max_primitives,
                    max_blocks=self.base.config.max_blocks,
                )
            except ValueError as error:
                attempts.append(
                    {
                        "attempt": index,
                        **parent,
                        "planner_channel": "structured",
                        "status": "execution_rejected",
                        "reason": str(error),
                    }
                )
                continue
            endpoint = trace["endpoint"]
            properties = eligibility({"smiles": endpoint})
            if type(properties.get("oracle_eligible")) is not bool:
                raise ValueError("endpoint evaluator must return boolean eligibility")
            status = (
                "duplicate"
                if endpoint in seen
                else ("eligible" if properties["oracle_eligible"] else "ineligible")
            )
            record = {
                "attempt": index,
                **parent,
                "planner_channel": "structured",
                "channel": "progressive_structured",
                "endpoint": endpoint,
                "status": status,
                "properties": properties,
                "actual_changes": trace["actual_changes"],
                "metadata": {
                    "progressive_structured": detail,
                    **self.base._continuation_lineage(entry),
                },
            }
            attempts.append(record)
            if status != "eligible":
                continue
            seen.add(endpoint)
            candidate = {
                "source_group": self.base.source_group,
                "oracle_protocol": self.base.oracle_protocol,
                "source_state": trace["states"][0],
                "program": program.payload(),
                "assignment": list(binding),
                "trace": trace,
                "endpoint": endpoint,
                "provenance": record,
                "inherited_static_score": entry["static_score"],
            }
            candidate.update(
                {
                    k: v
                    for k, v in self.base._continuation_lineage(entry).items()
                    if k != "accounting"
                }
            )
            candidates.append({**candidate, "candidate_id": identity(candidate)})
        return candidates, attempts

    @staticmethod
    def lane(candidate):
        return (
            "structured"
            if candidate["provenance"].get("planner_channel") == "structured"
            else "shallow"
        )

    def _select(self, candidates, limit):
        pools = {lane: [c for c in candidates if self.lane(c) == lane] for lane in LANES}
        positions = defaultdict(int)
        chosen = []
        # One floor per available lane, only if enough charged slots remain.
        if limit >= sum(bool(p) for p in pools.values()):
            for lane in LANES:
                if pools[lane]:
                    chosen.append(pools[lane][0])
                    positions[lane] += 1
        total = sum(s["charged"] for s in self.channel_stats.values())
        while len(chosen) < min(limit, len(candidates)):
            active = [lane for lane in LANES if positions[lane] < len(pools[lane])]
            scores = {}
            for lane in active:
                s = self.channel_stats[lane]
                scores[lane] = s["reward_sum"] / max(
                    1, s["charged"]
                ) + self.policy.channel_exploration * math.sqrt(
                    math.log(total + len(chosen) + 2) / (s["charged"] + positions[lane] + 1)
                )
            maximum = max(scores.values())
            tied = sorted(lane for lane in active if math.isclose(scores[lane], maximum))
            lane = tied[int(self.allocation_rng.integers(len(tied)))]
            chosen.append(pools[lane][positions[lane]])
            positions[lane] += 1
        return chosen

    def propose_batch(self, eligibility, *, limit=None):
        """Both pools use one measured archive; no oracle is called here."""
        if limit is None:
            limit = self.policy.query_batch_size
        if type(limit) is not int or not 1 <= limit <= self.policy.query_batch_size:
            raise ValueError("query limit must fit the declared batch budget")
        began = perf_counter()
        shallow = self.base.propose_batch(eligibility)
        seen = {e["endpoint"] for e in self.base.entries.values()} | self.base.failed_endpoints
        seen.update(c["endpoint"] for c in shallow["candidates"])
        rich, attempts = self._structured_pool(eligibility, seen)
        pool = [*shallow["candidates"], *rich]
        chosen = self._select(pool, limit)
        body = {k: v for k, v in self.base.pending.items() if k != "batch_id"}
        body.update(
            candidates=chosen,
            attempts=[*body["attempts"], *attempts],
            proposal_pool={"shallow_batch_id": shallow["batch_id"], "candidates": pool},
            selection={
                "policy": "bounded_incumbent_gain_ucb_v1",
                "selected_ids": [c["candidate_id"] for c in chosen],
                "channel_stats_before": json.loads(json.dumps(self.channel_stats)),
            },
        )
        self.base.pending = {**body, "batch_id": identity(body)}
        return {
            **self.base.pending,
            "proposal_seconds": perf_counter() - began,
            "new_oracle_calls": 0,
        }

    def observe_batch(self, batch_id, outcomes):
        if self.base.pending is None or self.base.pending["batch_id"] != batch_id:
            raise ValueError("objective outcome has no matching pending lock")
        pending = list(self.base.pending["candidates"])
        incumbent = min((r["score"] for r in self.base.observations.values()), default=math.inf)
        before = json.loads(json.dumps(self.channel_stats))
        # Validate/commit through the unchanged strict receipt boundary first.
        self.base.observe_batch(batch_id, outcomes)
        lookup = {r["candidate_id"]: r for r in outcomes}
        for candidate in pending:
            result = lookup[candidate["candidate_id"]]
            stats = before[self.lane(candidate)]
            stats["charged"] += 1
            if result["score"] is not None:
                score = float(result["score"])
                stats["scored"] += 1
                if math.isfinite(incumbent) and score < incumbent:
                    stats["incumbent_improvements"] += 1
                    stats["reward_sum"] += min(
                        1.0, (incumbent - score) / self.policy.improvement_scale
                    )
                incumbent = min(incumbent, score)
        self.channel_stats = before

    def snapshot(self, *, include_history=True):
        body = {
            "schema_version": "objective_program_search_v1",
            "policy": asdict(self.policy),
            "base": self.base.snapshot(include_history=include_history),
            "structured_rng": self.structured_rng.bit_generator.state,
            "allocation_rng": self.allocation_rng.bit_generator.state,
            "channel_stats": self.channel_stats,
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(cls, snapshot):
        body = {k: v for k, v in snapshot.items() if k != "snapshot_id"}
        if snapshot.get("schema_version") != "objective_program_search_v1" or identity(
            body
        ) != snapshot.get("snapshot_id"):
            raise ValueError("corrupt or incompatible objective-search snapshot")
        base = DynamicProgramOptimizer.restore(snapshot["base"])
        result = cls(
            base.config,
            ObjectiveSearchConfig(**snapshot["policy"]),
            source_group=base.source_group,
            oracle_protocol=base.oracle_protocol,
        )
        result.base = base
        result.structured_rng.bit_generator.state = snapshot["structured_rng"]
        result.allocation_rng.bit_generator.state = snapshot["allocation_rng"]
        result.channel_stats = json.loads(json.dumps(snapshot["channel_stats"]))
        return result
