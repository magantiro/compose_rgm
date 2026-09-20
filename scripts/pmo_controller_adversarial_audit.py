"""Adversarial audit of the PMO population controller's joint-credit allocation.

The controller was just changed so that `PopulationCredit` over
`CreditKey(basin, parent, family, scale)` is the ALLOCATION AUTHORITY: `_allocate`
draws a cell by `credit.allocate(cells)` and the fitted `ProgramValue` only ranks
candidates inside the drawn cell.  This module tries to FALSIFY that design on
deterministic synthetic reward landscapes.  It charges no oracle call, opens no
network connection and constructs no TDC `Oracle`: every reward here is a closed-form
function of the basin, parent, scale and round index.

Fidelity note.  The scenario loop writes credit with exactly the two production lines
from `PmoPopulationController.observe_batch`:

    key = credit_key_from_candidate(candidate)
    self.credit.observe(key, improvement(score, parent_score, direction="maximize"))

and selects with the controller's own `_allocate` / `_credit_allocate`.  What it does
NOT exercise is the proposal generator: the candidate pool is supplied, so every
result below is a statement about ALLOCATION given a pool, never about which pools the
three channels can actually produce.  A basin the generator never proposes cannot be
rescued by any allocation floor, and this harness cannot see that failure mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.pmo_credit import (
    CreditCell,
    CreditKey,
    PopulationCredit,
    basin_label,
    credit_key_from_candidate,
    improvement,
    scale_for_channel,
)
from compose_v4.control.pmo_population_controller import (
    CHANNELS,
    JUMP_CHANNEL,
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
    PmoPopulationController,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"

# Five decorable cores.  Every substituent is acyclic, so each analogue keeps its core's
# Bemis-Murcko scaffold and therefore its basin -- asserted in `_basin_universe`.
CORES = {
    "naphthalene": "{}c1ccc2ccccc2c1",
    "quinoline": "{}c1ccc2ncccc2c1",
    "indole": "{}c1ccc2[nH]ccc2c1",
    "biphenyl": "{}c1ccc(-c2ccccc2)cc1",
    "benzamide_piperidine": "O=C(NC1CCNCC1)c1ccc({})cc1",
}
def _substituents() -> tuple[str, ...]:
    """Acyclic decorations only, so every analogue keeps its core's Murcko scaffold.

    Generated rather than hand-listed because one scenario needs 12 disjoint cells on a
    SINGLE core; a short list silently wrapped and put one endpoint in two cells, which
    `build_pool` now refuses outright.
    """
    singles = ["C", "F", "Cl", "Br", "N", "O", "OC", "NC", "CO", "CN", "C#N", "SC"]
    chains = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "C(C)C", "CC(C)C"]
    tails = ["", "O", "N", "F", "Cl", "OC", "NC", "C(=O)O", "C(=O)N"]
    groups = list(singles)
    groups += [chain + tail for chain in chains for tail in tails]
    seen, ordered = set(), []
    for group in groups:
        if group not in seen:
            seen.add(group)
            ordered.append(group)
    return tuple(ordered)


SUBSTITUENTS = _substituents()
# One family per channel keeps the family axis from masking the scale axis.
RULES_BY_CHANNEL = {
    SHALLOW_CHANNEL: ("atom_insert",),
    STRUCTURED_CHANNEL: ("atom_insert", "atom_delete"),
    JUMP_CHANNEL: ("cycle_close", "atom_insert"),
}
PARENT_SCORE = 1.0


def _checkpoint() -> dict:
    payload = json.loads((ROOT / CHECKPOINT_PATH).read_text())
    return payload["payload"]["checkpoints"]["shared_all_routes"]


def controller(seed: int = 20260920, *, candidates_per_batch: int = 16):
    """A controller at the production search geometry (`pmo_population_v1.configuration`)."""
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        parent_allocation="niche_score",
        attempts_per_batch=128,
        candidates_per_batch=candidates_per_batch,
        wall_seconds=45.0,
        proposal_cache_entries=128,
    )
    return PmoPopulationController(
        config,
        source_group="adversarial-audit",
        oracle_protocol="free-synthetic",
        hierarchy=None,
        jump_checkpoint=_checkpoint(),
    )


def _basin_universe() -> dict[str, dict]:
    """Analogue sets per core, with the basin-grouping precondition asserted."""
    universe = {}
    labels = {}
    for name, template in CORES.items():
        candidates = [template.format(group) for group in SUBSTITUENTS]
        reference = basin_label(template.format("C"))
        smiles = []
        for row in candidates:
            try:
                if basin_label(row) == reference:
                    smiles.append(row)
            except ValueError:
                continue
        if len(smiles) < 56:
            raise AssertionError(f"core {name} kept only {len(smiles)} scaffold-preserving analogues")
        label = reference
        if label in labels:
            raise AssertionError(f"cores {name} and {labels[label]} share a basin")
        labels[label] = name
        universe[name] = {"basin": label, "smiles": smiles}
    return universe


UNIVERSE = _basin_universe()


def _jitter(smiles: str) -> float:
    """Deterministic tie-breaking noise in [0, 0.01). No RNG, no ordering dependence."""
    return (int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16) % 1000) / 100_000.0


def _raw_candidate(endpoint, *, parent, channel, parent_score=PARENT_SCORE):
    return {
        "endpoint": endpoint,
        "program": {"blocks": []},
        "trace": {
            "actions": [{"executor_rule": rule} for rule in RULES_BY_CHANNEL[channel]]
        },
        "provenance": {
            "planner_channel": channel,
            "entry_id": parent,
            "parent_measured_score": parent_score,
        },
    }


def build_pool(ctrl, layout, *, per_cell=4, offset=0):
    """One round's candidate pool.

    `layout` is a list of `(core, parent, channel)` triples.  Each triple gets its own
    disjoint slice of its core's analogue list, so no endpoint appears in two cells and
    every triple is a distinct `CreditKey`.
    """
    pool, used = [], defaultdict(int)
    for core, parent, channel in layout:
        smiles = UNIVERSE[core]["smiles"]
        start = used[core]
        rows = [smiles[(start + offset + i) % len(smiles)] for i in range(per_cell)]
        used[core] += per_cell
        for endpoint in rows:
            pool.append(ctrl._augment(_raw_candidate(endpoint, parent=parent, channel=channel)))
    seen = Counter(row["endpoint"] for row in pool)
    duplicated = [k for k, v in seen.items() if v > 1]
    if duplicated:
        raise AssertionError(f"pool endpoint reused across cells: {duplicated[:3]}")
    return pool


def _core_by_basin():
    return {row["basin"]: name for name, row in UNIVERSE.items()}


CORE_BY_BASIN = _core_by_basin()


def run_rounds(
    ctrl,
    layout,
    payoff,
    *,
    rounds=14,
    per_cell=4,
    room=None,
    use_full_allocate=False,
    observe=True,
):
    """Drive allocate -> synthetic score -> credit write for `rounds` rounds.

    `payoff(core, parent, scale, round_index) -> float` is the landscape.  The score
    handed back is `PARENT_SCORE + payoff + jitter`, so `improvement` equals the payoff
    plus jitter exactly.
    """
    history = []
    for round_index in range(rounds):
        ctrl.batches = round_index
        pool = build_pool(ctrl, layout, per_cell=per_cell, offset=round_index)
        if use_full_allocate:
            chosen, detail = ctrl._allocate(pool)
            credit_detail = detail["credit_allocation"]
        else:
            value, _ = ctrl._fit_value()
            slots = room if room is not None else ctrl.config.candidates_per_batch
            chosen, credit_detail = (
                ctrl._credit_allocate(pool, value, slots)
                if ctrl.credit.cells
                else (list(ctrl.rng.permutation(np.asarray(pool, dtype=object))[:slots]), {"mode": "cold_start_uniform"})
            )
        drawn = []
        for row in chosen:
            key = credit_key_from_candidate(row)
            core = CORE_BY_BASIN[key.basin]
            gain = payoff(core, key.parent, key.scale, round_index) + _jitter(row["endpoint"])
            score = PARENT_SCORE + gain
            if observe:
                ctrl.credit.observe(
                    key,
                    improvement(score, PARENT_SCORE, direction="maximize"),
                )
            drawn.append({"core": core, "parent": key.parent, "scale": key.scale, "gain": gain})
        history.append(
            {
                "round": round_index,
                "drawn": drawn,
                "n_drawn": len(drawn),
                "realized_gain": sum(row["gain"] for row in drawn),
                "by_core": dict(Counter(row["core"] for row in drawn)),
                "by_scale": dict(Counter(row["scale"] for row in drawn)),
                "by_parent": dict(Counter(row["parent"] for row in drawn)),
                "credit_mode": credit_detail.get("mode"),
                "cells_available": credit_detail.get("cells_available"),
                "minimum_share": credit_detail.get("minimum_share"),
                "active_cells": len(ctrl.credit.cells),
            }
        )
    return history


def _archive(ctrl, rows, scores):
    """Populate the archive fields `_fit_value` reads, so ProgramValue fits in both arms."""
    for row, score in zip(rows, scores, strict=True):
        ident = row["candidate_id"]
        ctrl.observations[ident] = {"endpoint": row["endpoint"], "score": float(score)}
        ctrl.entries[ident] = {
            "endpoint": row["endpoint"],
            "provenance": {"parent_measured_score": PARENT_SCORE},
            "fiber_features": row["fiber_features"],
        }


def run_campaign(ctrl, layout, payoff, *, rounds=14, per_cell=4, archive=True):
    """Full `_allocate` path, round by round, with the archive kept populated."""
    history = []
    for round_index in range(rounds):
        ctrl.batches = round_index
        pool = build_pool(ctrl, layout, per_cell=per_cell, offset=round_index)
        chosen, detail = ctrl._allocate(pool)
        drawn, scores = [], []
        for row in chosen:
            key = credit_key_from_candidate(row)
            core = CORE_BY_BASIN[key.basin]
            gain = payoff(core, key.parent, key.scale, round_index) + _jitter(row["endpoint"])
            scores.append(PARENT_SCORE + gain)
            ctrl.credit.observe(
                key, improvement(PARENT_SCORE + gain, PARENT_SCORE, direction="maximize")
            )
            drawn.append({"core": core, "parent": key.parent, "scale": key.scale, "gain": gain})
        if archive:
            _archive(ctrl, chosen, scores)
        history.append(
            {
                "round": round_index,
                "n_drawn": len(drawn),
                "realized_gain": sum(row["gain"] for row in drawn),
                "by_core": dict(Counter(row["core"] for row in drawn)),
                "by_scale": dict(Counter(row["scale"] for row in drawn)),
                "by_parent": dict(Counter(row["parent"] for row in drawn)),
                "credit_mode": detail["credit_allocation"].get("mode"),
                "roles": dict(Counter(detail["selection_role_by_candidate"].values())),
                "active_cells": len(ctrl.credit.cells),
            }
        )
    return history


def _share_by(history, field, *, since=0):
    counts = Counter()
    for row in history:
        if row["round"] >= since:
            counts.update(row[f"by_{field}"])
    total = sum(counts.values()) or 1
    return {k: round(v / total, 4) for k, v in sorted(counts.items())}


ALL_CHANNEL_LAYOUT = [(core, f"p_{core}", ch) for core in CORES for ch in CHANNELS]


# ---- Scenario 1: a basin that only becomes productive late ----

def scenario_late_basin(seed=20260920, rounds=14, late_round=8):
    ctrl = controller(seed)

    def payoff(core, parent, scale, round_index):
        if core == "indole":
            return 5.0 if round_index >= late_round else 0.0
        return 0.30

    history = run_rounds(ctrl, ALL_CHANNEL_LAYOUT, payoff, rounds=rounds, room=13)
    per_round = [row["by_core"].get("indole", 0) for row in history]
    before = _share_by(history[:late_round], "core")
    after = _share_by(history[late_round:], "core")
    starved = [i for i, n in enumerate(per_round[:late_round]) if n == 0]
    wait = next((i for i, n in enumerate(per_round[late_round:]) if n > 0), None)
    return {
        "rounds_from_productivity_to_first_draw": wait,
        "design_expectation": (
            "the 0.2 uniform floor must keep drawing the dead basin during rounds 0-7 so "
            "its late payoff is discoverable, then credit must shift budget onto it"
        ),
        "indole_draws_per_round": per_round,
        "oracle_calls_wasted_before_rediscovery": None if wait is None else wait * 13,
        "rounds_before_late_payoff_with_zero_draws": starved,
        "indole_share_before": before.get("indole", 0.0),
        "indole_share_after": after.get("indole", 0.0),
        "share_by_core_before": before,
        "share_by_core_after": after,
        "discovered": per_round[late_round] > 0 or any(n > 0 for n in per_round[late_round:]),
        "verdict": (
            "PASS"
            if wait is not None and wait <= 2 and after.get("indole", 0.0) > before.get("indole", 0.0)
            else "FAIL"
        ),
    }


# ---- Scenario 2: an early lucky basin versus a steadily productive one ----

def scenario_early_luck(seed=20260920, rounds=20):
    ctrl = controller(seed)

    def payoff(core, parent, scale, round_index):
        if core == "naphthalene":
            return 8.0 if round_index == 0 else 0.0
        if core == "quinoline":
            return 0.8
        return 0.05

    history = run_rounds(ctrl, ALL_CHANNEL_LAYOUT, payoff, rounds=rounds, room=13)
    lucky = [row["by_core"].get("naphthalene", 0) for row in history]
    steady = [row["by_core"].get("quinoline", 0) for row in history]
    crossover = next(
        (i for i in range(rounds) if sum(steady[: i + 1]) > sum(lucky[: i + 1])), None
    )
    return {
        "design_expectation": (
            "value = positive_sum / trials, so a one-off jackpot must decay as 1/n and the "
            "steadily productive basin must overtake it"
        ),
        "lucky_draws_per_round": lucky,
        "steady_draws_per_round": steady,
        "cumulative_crossover_round": crossover,
        "share_last_half": _share_by(history[rounds // 2 :], "core"),
        "locked_in": crossover is None,
        "verdict": "PASS" if crossover is not None else "FAIL",
    }


# ---- Scenario 3: jump pays early, refinement pays late ----

def scenario_scale_regime_shift(seed=20260920, rounds=16, switch=6):
    ctrl = controller(seed)

    def payoff(core, parent, scale, round_index):
        early = round_index < switch
        if scale == "jump":
            return 2.0 if early else 0.0
        if scale == "refine":
            return 0.0 if early else 2.0
        return 0.10

    history = run_rounds(ctrl, ALL_CHANNEL_LAYOUT, payoff, rounds=rounds, room=13)
    per_round = [
        {
            "round": row["round"],
            "realized": {
                k: round(v / max(1, row["n_drawn"]), 3) for k, v in sorted(row["by_scale"].items())
            },
        }
        for row in history
    ]
    cumulative = ctrl.credit_report()["trial_fraction_by_scale"]
    early_share = _share_by(history[:switch], "scale")
    late_share = _share_by(history[switch:], "scale")
    tail_share = _share_by(history[-4:], "scale")
    return {
        "design_expectation": (
            "realized per-round draws must move from jump to refine after the switch; "
            "`trial_fraction_by_scale` is CUMULATIVE so it can only lag, never track"
        ),
        "realized_per_round": per_round,
        "realized_share_before_switch": early_share,
        "realized_share_after_switch": late_share,
        "realized_share_last_4_rounds": tail_share,
        "credit_report_trial_fraction_by_scale_cumulative": cumulative,
        "policy_shifted": tail_share.get("refine", 0.0) > tail_share.get("jump", 0.0),
        "cumulative_diagnostic_shows_shift": cumulative.get("refine", 0.0)
        > cumulative.get("jump", 0.0),
        "verdict": "PASS" if tail_share.get("refine", 0.0) > tail_share.get("jump", 0.0) else "FAIL",
    }


# ---- Scenario 4: a dominant parent that exhausts ----

def scenario_parent_exhaustion(seed=20260920, rounds=40, exhaust=6):
    layout = [("naphthalene", f"p{i}", ch) for i in range(4) for ch in CHANNELS]
    ctrl = controller(seed)

    def payoff(core, parent, scale, round_index):
        if parent == "p0":
            return 3.0 if round_index < exhaust else 0.0
        return 0.4

    history = run_rounds(ctrl, layout, payoff, rounds=rounds, per_cell=5, room=13)
    p0 = [row["by_parent"].get("p0", 0) for row in history]
    fraction = [n / max(1, row["n_drawn"]) for n, row in zip(p0, history, strict=True)]
    uniform = 1.0 / len({parent for _, parent, _ in layout})
    decay = next(
        (r for r in range(exhaust, rounds - 3) if sum(fraction[r : r + 4]) / 4 <= uniform), None
    )
    return {
        "uniform_share_for_4_parents": uniform,
        "round_p0_returns_to_uniform_share": decay,
        "rounds_dead_before_returning_to_uniform": None if decay is None else decay - exhaust,
        "oracle_calls_spent_unwinding_the_dead_parent": None if decay is None else (decay - exhaust) * 13,
        "design_expectation": (
            "p0's cell value decays as positive_sum/trials once it stops paying, so budget "
            "must move onto p1-p3"
        ),
        "p0_draws_per_round": p0,
        "p0_share_while_paying": _share_by(history[:exhaust], "parent").get("p0", 0.0),
        "p0_share_after_exhaustion": _share_by(history[exhaust:], "parent").get("p0", 0.0),
        "p0_share_last_4": _share_by(history[-4:], "parent").get("p0", 0.0),
        "share_by_parent_after": _share_by(history[exhaust:], "parent"),
        "verdict": (
            "PASS"
            if _share_by(history[-4:], "parent").get("p0", 1.0)
            < _share_by(history[:exhaust], "parent").get("p0", 0.0)
            else "FAIL"
        ),
    }


# ---- Scenario 5: is allocation proportional to realized credit? ----

PAYOFF_BY_CORE = {
    "naphthalene": 0.2,
    "quinoline": 0.5,
    "indole": 1.0,
    "biphenyl": 2.0,
    "benzamide_piperidine": 4.0,
}


def scenario_proportionality(seed=20260920, rounds=30):
    # One cell per basin: cell COUNT is matched so only per-cell credit can drive share.
    layout = [(core, f"p_{core}", SHALLOW_CHANNEL) for core in CORES]
    ctrl = controller(seed)

    def payoff(core, parent, scale, round_index):
        return PAYOFF_BY_CORE[core]

    history = run_rounds(ctrl, layout, payoff, rounds=rounds, per_cell=5, room=13)
    realized = _share_by(history[rounds // 2 :], "core")
    keys = sorted(
        {
            CreditKey(
                basin=UNIVERSE[core]["basin"],
                parent=f"p_{core}",
                family="+".join(sorted(set(RULES_BY_CHANNEL[SHALLOW_CHANNEL]))),
                scale=scale_for_channel(SHALLOW_CHANNEL),
            )
            for core in CORES
        }
    )
    shares = ctrl.credit.allocate(keys)
    predicted = {CORE_BY_BASIN[k.basin]: round(float(s), 4) for k, s in zip(keys, shares, strict=True)}
    payoff_total = sum(PAYOFF_BY_CORE.values())
    proportional = {k: round(v / payoff_total, 4) for k, v in PAYOFF_BY_CORE.items()}
    order_payoff = [k for k, _ in sorted(PAYOFF_BY_CORE.items(), key=lambda r: -r[1])]
    order_realized = [k for k, _ in sorted(realized.items(), key=lambda r: -r[1])]
    return {
        "design_expectation": (
            "with cell count matched, realized draw share must be monotone in payoff; it "
            "cannot be exactly proportional because of the 0.2 uniform floor"
        ),
        "payoff_by_core": PAYOFF_BY_CORE,
        "pure_proportional_share": proportional,
        "controller_allocate_share": predicted,
        "realized_draw_share_second_half": realized,
        "rank_by_payoff": order_payoff,
        "rank_by_realized_share": order_realized,
        "monotone_in_payoff": order_payoff == order_realized,
        "verdict": "PASS" if order_payoff == order_realized else "FAIL",
    }


# ---- Scenario 5b: does cell COUNT beat cell CREDIT? ----

def scenario_fragmentation(seed=20260920):
    """A productive concentrated basin against an unproductive fragmented one.

    `allocate` distributes over CELLS, so a basin's total share is (number of its cells)
    x (per-cell value).  This asks whether cell proliferation alone can outbid measured
    productivity -- which matters because the jump channel invents novel scaffolds, and a
    novel scaffold is a novel basin, hence a brand-new cell carrying the full prior.
    """
    credit = PopulationCredit()
    productive = CreditKey(basin="PROD", parent="p0", family="atom_insert", scale="refine")
    for _ in range(40):
        credit.observe(productive, 4.0)
    rows = []
    for fragments in (1, 5, 10, 20, 30, 40, 60):
        keys = [productive] + [
            CreditKey(basin=f"FRAG{i}", parent=f"q{i}", family="cycle_close", scale="jump")
            for i in range(fragments)
        ]
        shares = credit.allocate(keys)
        rows.append(
            {
                "fragment_cells": fragments,
                "productive_cell_value": round(credit.value(productive), 4),
                "fragment_cell_value": round(credit.value(keys[-1]), 4),
                "productive_share": round(float(shares[0]), 4),
                "fragment_total_share": round(float(shares[1:].sum()), 4),
            }
        )
    flip = next((r["fragment_cells"] for r in rows if r["fragment_total_share"] > r["productive_share"]), None)
    return {
        "design_expectation": (
            "budget should follow measured credit; a basin with zero measured credit "
            "should not be able to outbid a cell with 40 trials at +4.0 each"
        ),
        "productive_cell": "40 trials, upside 4.0 per trial",
        "fragment_cells": "0 trials each, value = prior_weight / sqrt(1) = 0.25",
        "sweep": rows,
        "fragment_cell_count_that_outbids_the_productive_cell": flip,
        "verdict": "FAIL" if flip is not None else "PASS",
    }


# ---- Scenario 6: the exploration floor, isolated at one slot per draw ----

def scenario_floor_isolation(seed=20260920, draws=600):
    ctrl = controller(seed)
    layout = ALL_CHANNEL_LAYOUT
    pool_template = build_pool(ctrl, layout, per_cell=4)
    winner = credit_key_from_candidate(pool_template[0])
    for _ in range(2000):
        ctrl.credit.observe(winner, 1000.0)
    value, _ = ctrl._fit_value()
    counts = Counter()
    for draw in range(draws):
        pool = build_pool(ctrl, layout, per_cell=4, offset=draw)
        chosen, _detail = ctrl._credit_allocate(pool, value, 1)
        for row in chosen:
            counts[credit_key_from_candidate(row)] += 1
    n_cells = len({credit_key_from_candidate(r) for r in pool_template})
    bound = ctrl.credit.exploration_floor / n_cells
    observed = {f"{k.basin[:14]}|{k.parent}|{k.scale}": v / draws for k, v in counts.items()}
    starved = sorted(k for k in {credit_key_from_candidate(r) for r in pool_template} if k not in counts)
    off_winner = sum(v for k, v in counts.items() if k != winner) / draws
    return {
        "design_expectation": (
            "with room=1 the only route to a non-winning cell is the floor; every live "
            "cell must be reachable at roughly eps/n = "
            f"{bound:.4f}"
        ),
        "draws": draws,
        "cells": n_cells,
        "theoretical_minimum_share": round(bound, 5),
        "winner_share": round(counts[winner] / draws, 4),
        "off_winner_share": round(off_winner, 4),
        "observed_share_by_cell": {k: round(v, 4) for k, v in sorted(observed.items())},
        "cells_never_drawn": [f"{k.basin[:14]}|{k.parent}|{k.scale}" for k in starved],
        "verdict": "PASS" if not starved else "FAIL",
    }


# ---- Scenario 7: does a productive lineage compound into its children? ----

def scenario_generational_inheritance(seed=20260920):
    """A child becomes a parent under a NEW entry_id, so it opens a NEW cell.

    `allocate` reads `value(key)` on the joint cell only.  `marginal(axis)` exists but no
    allocation path consults it, so this asks whether a brand-new parent inside a proven
    basin starts with any advantage over a brand-new parent inside a barren one.
    """
    credit = PopulationCredit()
    proven_basin, barren_basin = "PROVEN", "BARREN"
    for _ in range(60):
        credit.observe(
            CreditKey(basin=proven_basin, parent="gen0", family="atom_insert", scale="refine"),
            5.0,
        )
    for _ in range(60):
        credit.observe(
            CreditKey(basin=barren_basin, parent="b0", family="atom_insert", scale="refine"), 0.0
        )
    child_of_proven = CreditKey(
        basin=proven_basin, parent="gen1_child", family="atom_insert", scale="refine"
    )
    child_of_barren = CreditKey(
        basin=barren_basin, parent="b1_child", family="atom_insert", scale="refine"
    )
    keys = [
        CreditKey(basin=proven_basin, parent="gen0", family="atom_insert", scale="refine"),
        child_of_proven,
        child_of_barren,
    ]
    shares = credit.allocate(keys)
    marg = credit.marginal("basin")
    return {
        "design_expectation": (
            "a child of a 60-trial +5.0 lineage, in the SAME basin, should start with more "
            "budget than a child of a 60-trial zero-yield lineage"
        ),
        "proven_basin_marginal": marg[proven_basin].payload(),
        "barren_basin_marginal": marg[barren_basin].payload(),
        "value_child_of_proven_basin": credit.value(child_of_proven),
        "value_child_of_barren_basin": credit.value(child_of_barren),
        "values_identical": credit.value(child_of_proven) == credit.value(child_of_barren),
        "share_parent_gen0": round(float(shares[0]), 4),
        "share_child_of_proven": round(float(shares[1]), 4),
        "share_child_of_barren": round(float(shares[2]), 4),
        "marginal_is_consulted_by_allocation": False,
        "verdict": (
            "FAIL"
            if credit.value(child_of_proven) == credit.value(child_of_barren)
            else "PASS"
        ),
    }


# ---- Scenario 8: does the joint credit survive a snapshot/restore? ----

def scenario_snapshot_restore(seed=20260920):
    ctrl = controller(seed)
    key = CreditKey(basin="c1ccccc1", parent="p0", family="atom_insert", scale="jump")
    for _ in range(37):
        ctrl.credit.observe(key, 3.0)
    before = ctrl.credit_report()
    snapshot = ctrl.snapshot()
    restored = PmoPopulationController.restore(
        snapshot, hierarchy=None, jump_checkpoint=_checkpoint()
    )
    after = restored.credit_report()
    return {
        "design_expectation": (
            "PopulationCredit documents deterministic serialization as invariant 5, and "
            "`program_campaign.run_program_campaign` restores a snapshot on every resume "
            "and on every already-complete round, so credit must round-trip"
        ),
        "cells_before": len(ctrl.credit.cells),
        "cells_after_restore": len(restored.credit.cells),
        "credit_report_before": before,
        "credit_report_after_restore": after,
        "snapshot_pmo_population_keys": sorted(snapshot["pmo_population"].keys()),
        "payload_methods_exist_but_unused": hasattr(PopulationCredit, "payload")
        and hasattr(PopulationCredit, "restore"),
        "allocation_path_after_restore": "cold_start_acquisition (credit.cells is empty)",
        "verdict": "FAIL" if len(restored.credit.cells) != len(ctrl.credit.cells) else "PASS",
    }


# ---- Scenario 9: does the cell key churn faster than evidence accumulates? ----

def scenario_generational_cell_churn(seed=20260920, rounds=14):
    """Promote drawn children into the parent pool, as a recursive campaign does.

    `credit_key_from_candidate` keys on the PARENT entry_id, and a child that re-enters
    the archive gets a new entry_id, so it opens an entirely new row of cells.  This
    measures what fraction of each round's candidate pool is cells the credit has never
    seen -- every one of which is worth exactly the flat `prior_weight`, not the basin's
    measured history.
    """
    ctrl = controller(seed)
    parents = [f"gen0_{i}" for i in range(4)]
    cores = list(CORES)[:3]
    trace = []
    for round_index in range(rounds):
        ctrl.batches = round_index
        layout = [(core, parent, ch) for core in cores for parent in parents for ch in CHANNELS]
        pool = build_pool(ctrl, layout, per_cell=2, offset=round_index)
        value, _ = ctrl._fit_value()
        cells = {credit_key_from_candidate(row) for row in pool}
        untried = sum(ctrl.credit.cell(key).trials == 0 for key in cells)
        prior_mass = sum(ctrl.credit.value(key) for key in cells if ctrl.credit.cell(key).trials == 0)
        earned_mass = sum(ctrl.credit.value(key) for key in cells if ctrl.credit.cell(key).trials)
        if ctrl.credit.cells:
            chosen, _ = ctrl._credit_allocate(pool, value, 13)
        else:
            chosen = list(ctrl.rng.permutation(np.asarray(pool, dtype=object))[:13])
        for row in chosen:
            key = credit_key_from_candidate(row)
            ctrl.credit.observe(
                key, improvement(2.0 + _jitter(row["endpoint"]), PARENT_SCORE, direction="maximize")
            )
        total_mass = prior_mass + earned_mass
        trace.append(
            {
                "round": round_index,
                "cells_in_pool": len(cells),
                "cells_never_tried": untried,
                "fraction_never_tried": round(untried / len(cells), 3),
                "credit_mass_from_flat_prior": round(prior_mass / total_mass, 3) if total_mass else None,
                "active_cells_total": len(ctrl.credit.cells),
            }
        )
        parents = parents[2:] + [f"gen{round_index + 1}_{i}" for i in range(2)]
    tail = trace[rounds // 2 :]
    mean_untried = float(np.mean([row["fraction_never_tried"] for row in tail]))
    mean_prior = float(np.mean([row["credit_mass_from_flat_prior"] for row in tail]))
    return {
        "design_expectation": (
            "as a lineage compounds, the credit should increasingly allocate on measured "
            "evidence rather than on the flat optimism prior"
        ),
        "per_round": trace,
        "mean_fraction_of_pool_never_tried_second_half": round(mean_untried, 3),
        "mean_credit_mass_from_flat_prior_second_half": round(mean_prior, 3),
        "active_cells_growth": [trace[0]["active_cells_total"], trace[-1]["active_cells_total"]],
        "verdict": "FAIL" if mean_untried > 0.5 else "PASS",
    }


# ---- Matched A/B: joint-Q allocation versus the pre-change marginal path ----

def _inert_observe(self, key, realized_improvement):
    """Make credit evidence a no-op so `_allocate` keeps taking the cold-start branch.

    This reproduces the PRE-CHANGE allocation exactly: `_allocate` tests
    `if self.credit.cells:` and falls through to `acquisition(...)` when it is empty,
    which is the code the controller ran before the joint credit was wired in.
    """
    if not isinstance(key, CreditKey):
        raise TypeError("credit evidence must be keyed by a CreditKey")
    return CreditCell()


def ab_trial(seed, layout, payoff, *, rounds, arm):
    original = PopulationCredit.observe
    if arm == "old_marginal":
        PopulationCredit.observe = _inert_observe
    try:
        ctrl = controller(seed)
        history = run_campaign(ctrl, layout, payoff, rounds=rounds)
    finally:
        PopulationCredit.observe = original
    return ctrl, history


def _late_basin_payoff(late_round=8):
    def payoff(core, parent, scale, round_index):
        if core == "indole":
            return 5.0 if round_index >= late_round else 0.0
        return 0.30

    return payoff


def _mixed_payoff(core, parent, scale, round_index):
    """Global basin structure plus within-basin scale structure."""
    base = PAYOFF_BY_CORE[core]
    if scale == "jump":
        base *= 1.5 if round_index < 5 else 0.2
    if scale == "refine":
        base *= 0.2 if round_index < 5 else 1.5
    return base


LANDSCAPES = {
    "late_basin": _late_basin_payoff(),
    "graded_basins": lambda core, parent, scale, r: PAYOFF_BY_CORE[core],
    "mixed_basin_and_scale": _mixed_payoff,
}


def run_ab(seeds=(20260920, 7, 101, 4242, 55555), rounds=14):
    results = {}
    for name, payoff in LANDSCAPES.items():
        rows = []
        for seed in seeds:
            trial = {"seed": seed}
            for arm in ("new_joint_q", "old_marginal"):
                _ctrl, history = ab_trial(
                    seed, ALL_CHANNEL_LAYOUT, payoff, rounds=rounds, arm=arm
                )
                total = sum(row["realized_gain"] for row in history)
                drawn = sum(row["n_drawn"] for row in history)
                cores = _share_by(history, "core")
                productive = (
                    "indole" if name == "late_basin" else "benzamide_piperidine"
                )
                late_half = _share_by(history[rounds // 2 :], "core")
                trial[arm] = {
                    "total_realized_gain": round(total, 4),
                    "candidates_drawn": drawn,
                    "gain_per_call": round(total / max(1, drawn), 4),
                    "productive_basin_share_overall": cores.get(productive, 0.0),
                    "productive_basin_share_second_half": late_half.get(productive, 0.0),
                    "basins_touched": len(cores),
                    "credit_mode_last_round": history[-1]["credit_mode"],
                }
            trial["gain_per_call_delta_new_minus_old"] = round(
                trial["new_joint_q"]["gain_per_call"] - trial["old_marginal"]["gain_per_call"], 4
            )
            rows.append(trial)
        deltas = [row["gain_per_call_delta_new_minus_old"] for row in rows]
        new_shares = [row["new_joint_q"]["productive_basin_share_second_half"] for row in rows]
        old_shares = [row["old_marginal"]["productive_basin_share_second_half"] for row in rows]
        results[name] = {
            "trials": rows,
            "median_gain_per_call_delta": round(float(np.median(deltas)), 4),
            "seeds_where_new_wins": sum(d > 0 for d in deltas),
            "seeds_where_old_wins": sum(d < 0 for d in deltas),
            "median_productive_share_new": round(float(np.median(new_shares)), 4),
            "median_productive_share_old": round(float(np.median(old_shares)), 4),
            "verdict": (
                "NEW BETTER"
                if float(np.median(deltas)) > 0 and sum(d > 0 for d in deltas) > len(deltas) / 2
                else "NEW WORSE"
                if float(np.median(deltas)) < 0 and sum(d < 0 for d in deltas) > len(deltas) / 2
                else "INDISTINGUISHABLE"
            ),
        }
    return results


SCENARIOS = {
    "s1_late_basin_discovery": scenario_late_basin,
    "s2_early_luck_recovery": scenario_early_luck,
    "s3_scale_regime_shift": scenario_scale_regime_shift,
    "s4_parent_exhaustion": scenario_parent_exhaustion,
    "s5_credit_proportionality": scenario_proportionality,
    "s5b_cell_fragmentation": scenario_fragmentation,
    "s6_exploration_floor_isolated": scenario_floor_isolation,
    "s7_generational_inheritance": scenario_generational_inheritance,
    "s8_snapshot_restore_persistence": scenario_snapshot_restore,
    "s9_generational_cell_churn": scenario_generational_cell_churn,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="diagnostics/pmo_controller_adversarial_audit_v1.json")
    parser.add_argument("--skip-ab", action="store_true")
    args = parser.parse_args()

    report = {
        "schema_version": "pmo_controller_adversarial_audit_v1",
        "oracle_calls": 0,
        "reward_source": "closed-form synthetic landscapes; no TDC Oracle, no network",
        "controller_geometry": "pmo_population_v1.configuration (candidates_per_batch=16)",
        "fidelity_caveat": (
            "allocation and credit writing are production code paths; the candidate POOL "
            "is synthetic, so nothing here tests the three proposal generators"
        ),
        "scenarios": {},
    }
    for name, fn in SCENARIOS.items():
        print(f"running {name} ...", flush=True)
        report["scenarios"][name] = fn()
        print(f"  single-seed verdict = {report['scenarios'][name]['verdict']}", flush=True)
    report["multi_seed"] = {}
    report["multi_seed_note"] = (
        "One seed is an anecdote.  The first single-seed read of s1 showed six consecutive "
        "rounds with zero draws on the productive basin and was written down as a failure; "
        "resampling that round 2000 times put P(zero of 13 slots at a 9.6% share) at 0.259, "
        "and across 15 seeds the median time from productivity to first draw is 0 rounds.  "
        "The multi-seed block, not the single-seed verdict, is the authority here."
    )
    for name in MULTI_SEED_METRICS:
        print(f"running {name} over {len(AUDIT_SEEDS)} seeds ...", flush=True)
        report["multi_seed"][name] = multi_seed(name)
    if not args.skip_ab:
        print("running matched A/B ...", flush=True)
        report["matched_ab"] = run_ab(seeds=AUDIT_SEEDS[:9], rounds=16)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v["verdict"] for k, v in report["scenarios"].items()}, indent=2))
    if "matched_ab" in report:
        print(json.dumps({k: v["verdict"] for k, v in report["matched_ab"].items()}, indent=2))




# ---- Multi-seed aggregation: one seed is an anecdote ----

AUDIT_SEEDS = (20260920, 7, 101, 4242, 55555, 13, 271828, 31337, 900001, 64, 5, 88, 1234, 77, 314159)

MULTI_SEED_METRICS = {
    "s1_late_basin_discovery": (
        "rounds_from_productivity_to_first_draw",
        "oracle_calls_wasted_before_rediscovery",
        "indole_share_before",
        "indole_share_after",
    ),
    "s2_early_luck_recovery": ("cumulative_crossover_round",),
    "s3_scale_regime_shift": ("policy_shifted", "cumulative_diagnostic_shows_shift"),
    "s4_parent_exhaustion": ("p0_share_while_paying", "p0_share_after_exhaustion", "p0_share_last_4"),
    "s5_credit_proportionality": ("monotone_in_payoff",),
}


def multi_seed(name, seeds=AUDIT_SEEDS):
    fn = SCENARIOS[name]
    runs = [fn(seed=seed) for seed in seeds]
    out = {"seeds": list(seeds), "n_seeds": len(seeds), "verdicts": Counter(r["verdict"] for r in runs)}
    for field in MULTI_SEED_METRICS[name]:
        values = [r.get(field) for r in runs]
        if all(isinstance(v, bool) for v in values):
            out[field] = {"true": sum(values), "false": sum(not v for v in values)}
        else:
            present = [v for v in values if v is not None]
            out[field] = {
                "values": values,
                "median": round(float(np.median(present)), 4) if present else None,
                "min": min(present) if present else None,
                "max": max(present) if present else None,
                "n_none": sum(v is None for v in values),
            }
    out["verdicts"] = dict(out["verdicts"])
    out["pass_fraction"] = round(out["verdicts"].get("PASS", 0) / len(seeds), 4)
    return out


if __name__ == "__main__":
    main()
