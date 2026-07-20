#!/usr/bin/env python3
"""Audit global and local event structure in a saved ancestral rollout set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from compose_v4.chem.molecular_graph import is_element


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {key: 0.0 for key in ("mean", "q05", "q25", "q50", "q75", "q95")}
    return {
        "mean": float(array.mean()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def audit_rollouts(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or not isinstance(payload.get("rollouts"), tuple):
        raise ValueError("expected a merged COMPOSE rollout payload")
    rollouts = payload["rollouts"]
    if not rollouts:
        raise ValueError("rollout payload is empty")

    total_rules: Counter[str] = Counter()
    trajectories_with_rule: Counter[str] = Counter()
    trajectory_rule_counts: list[Counter[str]] = []
    event_times: dict[str, list[float]] = defaultdict(list)
    event_positions: dict[str, list[float]] = defaultdict(list)
    first_rules: Counter[str] = Counter()
    last_rules: Counter[str] = Counter()
    event_counts = []
    initial_atoms = []
    final_atoms = []
    final_bonds = []
    final_cycle_rank = []
    net_size_changes = []
    graft_fractions = []
    ring_last_decile = 0
    ring_events = 0
    first_ring_positions = []
    exhausted = 0

    for rollout in rollouts:
        rules = tuple(str(rule) for rule in rollout.event_rules)
        times = tuple(float(value) for value in rollout.event_times)
        if len(rules) != len(times):
            raise ValueError("rollout event rules and times are misaligned")
        counts = Counter(rules)
        trajectory_rule_counts.append(counts)
        total_rules.update(counts)
        trajectories_with_rule.update(counts.keys())
        if rules:
            first_rules[rules[0]] += 1
            last_rules[rules[-1]] += 1
        event_counts.append(len(rules))
        denominator = max(len(rules) - 1, 1)
        for index, (rule, time) in enumerate(zip(rules, times)):
            position = index / denominator
            event_times[rule].append(time)
            event_positions[rule].append(position)
            if rule == "ring_system_grow":
                ring_events += 1
                ring_last_decile += int(position >= 0.9)
        ring_indices = [index for index, rule in enumerate(rules) if rule == "ring_system_grow"]
        if ring_indices:
            first_ring_positions.append(ring_indices[0] / denominator)

        state = rollout.final_state
        atoms = int(np.count_nonzero(is_element(state.atom_types)))
        bonds = int(np.count_nonzero(np.triu(state.bonds != 0, k=1)))
        final_atoms.append(atoms)
        final_bonds.append(bonds)
        final_cycle_rank.append(max(bonds - atoms + int(atoms > 0), 0))
        size_change = int(counts["atom_insert"] - counts["atom_delete"])
        net_size_changes.append(size_change)
        initial_atoms.append(atoms - size_change)
        graft_fractions.append(counts["bond_reroute"] / max(len(rules), 1))
        exhausted += int(bool(rollout.exhausted_event_budget))

    samples = len(rollouts)
    rules = tuple(sorted(total_rules))
    total_events = sum(total_rules.values())
    return {
        "format": "compose_v4_pancake_rollout_audit_v1",
        "samples": samples,
        "total_events": total_events,
        "event_count": _quantiles(event_counts),
        "event_rule_counts": dict(total_rules),
        "event_rule_fractions": {
            rule: total_rules[rule] / max(total_events, 1) for rule in rules
        },
        "trajectories_with_rule_fraction": {
            rule: trajectories_with_rule[rule] / samples for rule in rules
        },
        "events_per_trajectory": {
            rule: _quantiles(counts.get(rule, 0) for counts in trajectory_rule_counts)
            for rule in rules
        },
        "event_time": {rule: _quantiles(event_times[rule]) for rule in rules},
        "normalized_event_position": {
            rule: _quantiles(event_positions[rule]) for rule in rules
        },
        "first_rule_fraction": {
            rule: first_rules[rule] / samples for rule in sorted(first_rules)
        },
        "last_rule_fraction": {
            rule: last_rules[rule] / samples for rule in sorted(last_rules)
        },
        "first_ring_position": _quantiles(first_ring_positions),
        "ring_events_in_last_decile_fraction": ring_last_decile / max(ring_events, 1),
        "initial_atoms": _quantiles(initial_atoms),
        "final_atoms": _quantiles(final_atoms),
        "net_size_change": _quantiles(net_size_changes),
        "final_bonds": _quantiles(final_bonds),
        "final_cycle_rank": _quantiles(final_cycle_rank),
        "graft_fraction_per_trajectory": _quantiles(graft_fractions),
        "graft_majority_trajectory_fraction": float(
            np.mean(np.asarray(graft_fractions) > 0.5)
        ),
        "graft_three_quarters_trajectory_fraction": float(
            np.mean(np.asarray(graft_fractions) > 0.75)
        ),
        "event_budget_exhaustion_fraction": exhausted / samples,
        "limitations": {
            "canonical_self_events": (
                "not recoverable from compact 2,000-sample payload; use full-state "
                "trajectory replays"
            ),
            "immediate_reversals": (
                "not recoverable from rule names without intermediate states/actions"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.rollouts, map_location="cpu", weights_only=False)
    report = audit_rollouts(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
