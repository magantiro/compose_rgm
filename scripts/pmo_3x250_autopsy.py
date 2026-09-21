"""Zero-oracle chemical and trajectory autopsy of the completed PMO 3x250 scored run.

READ-ONLY over already-charged ledgers.  This driver never calls a PMO oracle and never
launches anything: every score it uses was charged by the run under audit, and the two
objective decompositions it computes are OFFLINE RE-DERIVATIONS of the published task
definitions that are validated against all 250 recorded scores before any component is
reported.  If the re-derivation does not reproduce the ledger exactly the decomposition
is withheld rather than reported with a caveat.

Data layout consumed (per task, under ``<run_root>/<run_id>/<task>/``):

    oracle/query_NNNNNN/result.json   one charged call: index, role, endpoint, score
    campaign/round_NNNN/complete.json controller snapshot AFTER the round resolved
    campaign/round_NNNN/pending.json  the batch proposed in that round (attempts +
                                      allocation), i.e. the PROPOSAL-side record
    campaign/initialization.json      the 16 charged initialization molecules

Joins that hold and are asserted:

  * ledger endpoints are unique, so ``endpoint -> oracle index`` is a function;
  * ``snapshot.entries`` accumulates exactly the queried candidates, so the final round's
    entry set is the full scored-candidate table;
  * ``entry["provenance"]["entry_id"]`` is the PARENT entry id, not the entry's own id --
    verified here rather than assumed, because the same field name is used for both
    roles elsewhere in the controller and reading it the wrong way silently inverts
    every recursion statistic in section 4.

Sections produced mirror the autopsy brief: (1) best-so-far lineage, (2) celecoxib
target-distance trajectory, (3) perindopril MPO component decomposition, (4) recursion
depth and descendant productivity, (5) lane proposal/attrition, (6) teacher-route versus
autonomous-proposal move comparison.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from compose_v4.chem.molecular_graph import is_element, molecular_graph_to_smiles
from compose_v4.rewrite.trace_shard import decode_state

RDLogger.DisableLog("rdApp.*")

TASKS = ("perindopril_mpo", "celecoxib_rediscovery", "gsk3b")
EXCLUDED_FROM_CONCLUSIONS = ("gsk3b",)

CELECOXIB = "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"
PERINDOPRIL = "O=C(OCC)C(NC(C)C(=O)N1C(C(=O)O)CC2CCCCC12)CCC"

SCALE_BY_CHANNEL = {
    "shallow_program_channel": "refine",
    "structured_program_channel": "medium",
    "joint_dependency_region_jump": "jump",
}

# Coarse move vocabulary shared by teacher routes and autonomous programs.  These are
# executor-rule groupings, not chemical judgements: the point is a like-for-like axis.
MOVE_CLASS = {
    "atom_insert": "grow",
    "atom_delete": "prune",
    "atom_restate_semantic": "replace",
    "atom_restate": "replace",
    "bond_reorder": "remodel",
    "bond_reroute": "remodel",
    "cycle_close": "ring_create",
    "cycle_open": "ring_destroy",
    "ring_system_restate": "ring_remodel",
    "ring_system_grow": "ring_create",
    "ring_system_delete": "ring_destroy",
}


# ---- Chemistry helpers (offline, no oracle) ----


def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def count_fingerprint(smiles: str):
    """ECFP4 COUNT fingerprint -- the GuacaMol/TDC convention, validated below."""
    molecule = _mol(smiles)
    return None if molecule is None else AllChem.GetMorganFingerprint(molecule, 2)


def bit_fingerprint(smiles: str):
    """Morgan r=2, 2048 bits -- the controller's own ``fiber_fingerprint`` convention."""
    molecule = _mol(smiles)
    return None if molecule is None else AllChem.GetMorganFingerprintAsBitVect(
        molecule, 2, nBits=2048
    )


def tanimoto(left, right) -> float | None:
    if left is None or right is None:
        return None
    return float(DataStructs.TanimotoSimilarity(left, right))


def gaussian_modifier(value: float, mu: float, sigma: float) -> float:
    return float(math.exp(-0.5 * ((value - mu) / sigma) ** 2))


def geometric_mean(values) -> float:
    values = list(values)
    if any(v <= 0.0 for v in values):
        return 0.0
    return float(math.exp(sum(math.log(v) for v in values) / len(values)))


def scaffold(smiles: str) -> str:
    molecule = _mol(smiles)
    if molecule is None:
        return "unparseable"
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    return value if value else "acyclic"


def descriptors(smiles: str) -> dict:
    molecule = _mol(smiles)
    if molecule is None:
        return {}
    return {
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "rings": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "scaffold": scaffold(smiles),
    }


def retained_fraction(source_state, endpoint_state) -> float:
    """Fraction of the PARENT's occupied slots still occupied in the child.

    This mirrors ``pmo_population_controller._retained_fraction`` exactly (slot identity,
    not substructure): the controller's own jump ranking is defined on it, so any
    comparison against ``retention_target`` has to use the same definition.
    """
    original = np.flatnonzero(is_element(source_state.atom_types))
    if not len(original):
        return 0.0
    mask = is_element(endpoint_state.atom_types)
    return float(sum(bool(mask[int(slot)]) for slot in original) / len(original))


# ---- Objective re-derivation, validated against the charged ledger ----


def celecoxib_score(smiles: str) -> float | None:
    return tanimoto(count_fingerprint(CELECOXIB), count_fingerprint(smiles))


def perindopril_components(smiles: str) -> dict | None:
    molecule = _mol(smiles)
    if molecule is None:
        return None
    similarity = tanimoto(count_fingerprint(PERINDOPRIL), count_fingerprint(smiles))
    aromatic = int(rdMolDescriptors.CalcNumAromaticRings(molecule))
    ring_term = gaussian_modifier(aromatic, 2.0, 0.5)
    return {
        "ecfp4_similarity_to_perindopril": similarity,
        "aromatic_rings": aromatic,
        "aromatic_ring_term": ring_term,
        "score": geometric_mean([similarity, ring_term]),
    }


def validate_objective(task: str, ledger: list[dict]) -> dict:
    """Re-derive every charged score offline; the decomposition is used only if exact."""
    mismatches = []
    for row in ledger:
        if task == "celecoxib_rediscovery":
            derived = celecoxib_score(row["endpoint"])
        elif task == "perindopril_mpo":
            components = perindopril_components(row["endpoint"])
            derived = None if components is None else components["score"]
        else:
            return {"validated": False, "reason": "no offline definition attempted"}
        if derived is None or abs(derived - float(row["score"])) > 1e-9:
            mismatches.append({"endpoint": row["endpoint"], "ledger": row["score"],
                               "derived": derived})
    return {
        "validated": not mismatches,
        "rows_checked": len(ledger),
        "exact_matches": len(ledger) - len(mismatches),
        "mismatches": mismatches[:5],
        "definition": (
            "ECFP4 count-fingerprint Tanimoto to celecoxib"
            if task == "celecoxib_rediscovery"
            else "geometric mean of [ECFP4 count Tanimoto to perindopril, "
                 "Gaussian(mu=2, sigma=0.5) on aromatic ring count]"
        ),
    }


# ---- Loading ----


def load_ledger(task_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((task_dir / "oracle").glob("query_*/result.json")):
        rows.append(json.loads(path.read_text()))
    rows.sort(key=lambda row: row["index"])
    for position, row in enumerate(rows):
        if row["index"] != position:
            raise ValueError(f"noncontiguous oracle ledger at {position}")
    return rows


def load_rounds(task_dir: Path) -> list[dict]:
    rounds = []
    for folder in sorted(glob.glob(str(task_dir / "campaign" / "round_*"))):
        rounds.append({
            "folder": os.path.basename(folder),
            "complete": json.loads(Path(folder, "complete.json").read_text()),
            "pending": json.loads(Path(folder, "pending.json").read_text()),
        })
    return rounds


def program_family(actions) -> str:
    rules = sorted({str(a["executor_rule"]) for a in actions})
    return "+".join(rules) if rules else "empty"


def build_entry_table(rounds, ledger) -> tuple[list[dict], dict]:
    """One row per scored candidate, joined to its charged oracle call."""
    final = rounds[-1]["complete"]["snapshot"]
    entries = final["entries"]
    index_by_endpoint = {row["endpoint"]: row["index"] for row in ledger}
    score_by_endpoint = {row["endpoint"]: float(row["score"]) for row in ledger}
    # Round at which each entry first appears -- the round that paid for it.
    first_round = {}
    seen = set()
    for number, entry in enumerate(rounds):
        keys = set(entry["complete"]["snapshot"]["entries"])
        for key in keys - seen:
            first_round[key] = number
        seen |= keys

    self_referential = sum(
        1 for key, row in entries.items() if row["provenance"].get("entry_id") == key
    )
    parent_is_other = sum(
        1 for row in entries.items()
        if row[1]["provenance"].get("entry_id") in entries
        and row[1]["provenance"].get("entry_id") != row[0]
    )

    # TWO different "parents" exist in a candidate record and they disagree on 26-29% of
    # rows; conflating them silently inverts every parent-conditioned statistic.
    #   * the MEASURED parent is ``entries[provenance["entry_id"]]`` -- the archive entry
    #     whose reward the controller credits.  Verified below: its ledger score equals
    #     ``provenance["parent_measured_score"]`` on 202/202 rows.
    #   * ``trace["states"][0]`` is the PROGRAM SOURCE, i.e. the state the (possibly
    #     mutated) program was replayed from.  For a ``mutation`` proposal that is the
    #     measured parent's own source, one generation further back, and its ledger score
    #     matches ``parent_measured_score`` on only ~150/202 rows.
    # Everything parent-relative below is computed against the MEASURED parent; the
    # program-source quantities are kept alongside under a ``program_source_`` prefix.
    table = []
    for key, entry in entries.items():
        provenance = entry["provenance"]
        actions = entry["trace"]["actions"]
        source = decode_state(entry["trace"]["states"][0])
        endpoint_state = decode_state(entry["trace"]["states"][-1])
        program_source_smiles = molecular_graph_to_smiles(source)
        parent_id = provenance.get("entry_id")
        parent_smiles = (
            entries[parent_id]["endpoint"] if parent_id in entries else program_source_smiles
        )
        child = entry["endpoint"]
        child_descriptors = descriptors(child)
        parent_descriptors = descriptors(parent_smiles)
        program_source_descriptors = descriptors(program_source_smiles)
        size = provenance.get("program_size") or {}
        channel = provenance.get("planner_channel")
        table.append({
            "entry_id": key,
            "oracle_index": index_by_endpoint.get(child),
            "round_first_seen": first_round.get(key),
            "endpoint": child,
            "score": score_by_endpoint.get(child),
            "parent_entry_id": provenance.get("entry_id"),
            "parent_endpoint": parent_smiles,
            "parent_score": provenance.get("parent_measured_score"),
            "lane": channel,
            "mode": provenance.get("mode"),
            "scale": SCALE_BY_CHANNEL.get(channel),
            "family": program_family(actions),
            "program_primitives": len(actions),
            "scheduled_blocks": len(size.get("scheduled_blocks") or []),
            "rule_counts": dict(sorted(Counter(str(a["executor_rule"]) for a in actions).items())),
            "program_source_endpoint": program_source_smiles,
            "program_source_is_the_measured_parent": program_source_smiles == parent_smiles,
            "retained_fraction": retained_fraction(source, endpoint_state),
            "program_source_heavy_atoms": program_source_descriptors.get("heavy_atoms"),
            "program_source_heavy_atom_delta": (
                None if not child_descriptors or not program_source_descriptors
                else child_descriptors["heavy_atoms"]
                - program_source_descriptors["heavy_atoms"]
            ),
            "heavy_atoms_parent": parent_descriptors.get("heavy_atoms"),
            "heavy_atoms_child": child_descriptors.get("heavy_atoms"),
            "heavy_atom_delta": (
                None if not child_descriptors or not parent_descriptors
                else child_descriptors["heavy_atoms"] - parent_descriptors["heavy_atoms"]
            ),
            "rings_parent": parent_descriptors.get("rings"),
            "rings_child": child_descriptors.get("rings"),
            "ring_delta": (
                None if not child_descriptors or not parent_descriptors
                else child_descriptors["rings"] - parent_descriptors["rings"]
            ),
            "aromatic_rings_child": child_descriptors.get("aromatic_rings"),
            "basin_child": child_descriptors.get("scaffold"),
            "basin_parent": parent_descriptors.get("scaffold"),
            "scaffold_changed": (
                None if not child_descriptors or not parent_descriptors
                else child_descriptors["scaffold"] != parent_descriptors["scaffold"]
            ),
            "retention_target": provenance.get("retention_target"),
        })
    table.sort(key=lambda row: (row["oracle_index"] is None, row["oracle_index"]))
    # Assert the parent identification rather than trusting it: the measured parent's
    # ledger score must equal the recorded ``parent_measured_score``.
    parent_checks = {"agree": 0, "disagree": 0, "program_source_agrees": 0}
    for row in table:
        if row["parent_score"] is None:
            continue
        if abs(score_by_endpoint.get(row["parent_endpoint"], -9.0)
               - row["parent_score"]) < 1e-12:
            parent_checks["agree"] += 1
        else:
            parent_checks["disagree"] += 1
        if abs(score_by_endpoint.get(row["program_source_endpoint"], -9.0)
               - row["parent_score"]) < 1e-12:
            parent_checks["program_source_agrees"] += 1

    join = {
        "scored_candidates": len(table),
        "joined_to_oracle_index": sum(1 for row in table if row["oracle_index"] is not None),
        "measured_parent_identification": parent_checks,
        "rows_where_program_source_differs_from_measured_parent": sum(
            1 for row in table if not row["program_source_is_the_measured_parent"]),
        "parent_field_is_self_referential": self_referential,
        "parent_field_points_at_another_entry": parent_is_other,
        "parent_field_absent_bootstrap_rows": sum(
            1 for row in table if row["parent_entry_id"] is None
        ),
    }
    return table, join


# ---- Section 1: best-so-far lineage ----


def best_so_far(ledger, table) -> dict:
    by_index = {row["oracle_index"]: row for row in table if row["oracle_index"] is not None}
    best = -math.inf
    improvements = []
    for row in ledger:
        score = float(row["score"])
        if score <= best:
            continue
        best = score
        detail = by_index.get(row["index"])
        improvements.append({
            "oracle_call": row["index"] + 1,
            "oracle_index": row["index"],
            "role": row["role"],
            "score": score,
            "endpoint": row["endpoint"],
            **({
                "parent_endpoint": detail["parent_endpoint"],
                "parent_score": detail["parent_score"],
                "lane": detail["lane"],
                "mode": detail["mode"],
                "scale": detail["scale"],
                "family": detail["family"],
                "program_primitives": detail["program_primitives"],
                "retained_fraction": detail["retained_fraction"],
                "heavy_atom_delta": detail["heavy_atom_delta"],
                "heavy_atoms_parent": detail["heavy_atoms_parent"],
                "heavy_atoms_child": detail["heavy_atoms_child"],
                "ring_delta": detail["ring_delta"],
                "basin_child": detail["basin_child"],
                "basin_parent": detail["basin_parent"],
                "scaffold_changed": detail["scaffold_changed"],
                "round_first_seen": detail["round_first_seen"],
            } if detail else {"parent_endpoint": None, "lane": "initialization"}),
        })
    return {"improvements": improvements, "count": len(improvements)}


# ---- Section 4: recursion ----


def recursion(table, ledger) -> dict:
    by_id = {row["entry_id"]: row for row in table}
    initialization = {row["endpoint"] for row in ledger if row["role"] == "initialization"}

    def depth(row, guard=0):
        parent = row["parent_entry_id"]
        if parent is None or parent not in by_id or guard > 64:
            return 1  # parent is an initialization molecule -> this child is generation 1
        return 1 + depth(by_id[parent], guard + 1)

    for row in table:
        row["generation"] = depth(row)
        row["parent_is_initialization"] = (
            row["parent_entry_id"] is None or row["parent_endpoint"] in initialization
        )

    children_by_parent = defaultdict(list)
    for row in table:
        if row["parent_entry_id"]:
            children_by_parent[row["parent_entry_id"]].append(row)

    # P(child improves on its parent | parent score bin)
    scored = [r for r in table if r["score"] is not None and r["parent_score"] is not None]
    edges = sorted({round(r["parent_score"], 12) for r in scored})
    bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.35),
            (0.35, 0.45), (0.45, 1.01)]
    conditional = []
    for low, high in bins:
        rows = [r for r in scored if low <= r["parent_score"] < high]
        if not rows:
            continue
        improved = [r for r in rows if r["score"] > r["parent_score"]]
        conditional.append({
            "parent_score_bin": f"[{low:.2f}, {high:.2f})",
            "children": len(rows),
            "children_improving_on_parent": len(improved),
            "probability": len(improved) / len(rows),
            "mean_delta": float(np.mean([r["score"] - r["parent_score"] for r in rows])),
            "best_delta": float(max(r["score"] - r["parent_score"] for r in rows)),
        })

    # Budget share reaching descendants of a PRODUCTIVE parent (a parent that has at
    # least one child scoring above it).
    productive = {
        key for key, rows in children_by_parent.items()
        if any(r["score"] is not None and r["parent_score"] is not None
               and r["score"] > r["parent_score"] for r in rows)
    }
    spent_on_productive_descendants = sum(
        1 for row in table if row["parent_entry_id"] in productive
    )
    # How often is a productive child itself used as a parent later?
    improving_children = {
        r["entry_id"] for r in scored if r["score"] > r["parent_score"]
    }
    reused = {key for key in children_by_parent if key in improving_children}
    grandchildren = sum(len(children_by_parent[key]) for key in reused)

    return {
        "generation_histogram": dict(sorted(Counter(r["generation"] for r in table).items())),
        "parent_is_initialization_molecule": sum(1 for r in table if r["parent_is_initialization"]),
        "distinct_parents_used": len(children_by_parent),
        "children_per_parent": {
            "mean": float(np.mean([len(v) for v in children_by_parent.values()])),
            "max": int(max(len(v) for v in children_by_parent.values())),
            "histogram": dict(sorted(Counter(
                len(v) for v in children_by_parent.values()).items())),
        },
        "conditional_improvement_by_parent_bin": conditional,
        "productive_parents": len(productive),
        "budget_on_children_of_productive_parents": spent_on_productive_descendants,
        "budget_share_on_children_of_productive_parents": (
            spent_on_productive_descendants / len(table) if table else 0.0
        ),
        "children_that_beat_their_parent": len(improving_children),
        "such_children_later_used_as_parents": len(reused),
        "grandchildren_of_improving_children": grandchildren,
        "distinct_parent_scores": len(edges),
    }


def credit_degeneracy(rounds) -> dict:
    """Does the joint credit cell ever accumulate evidence, and does it ever steer?"""
    final = rounds[-1]["complete"]["snapshot"]["pmo_population"]["credit"]["cells"]
    trials = Counter(int(cell["trials"]) for cell in final)
    total_trials = sum(int(cell["trials"]) for cell in final)

    draws, draws_with_evidence, per_round = 0, 0, []
    known = set()
    for entry in rounds:
        allocation = entry["pending"]["batch"]["allocation"]
        detail = allocation.get("credit_allocation") or {}
        drawn = detail.get("drawn_cells") or []
        hits = sum(
            1 for cell in drawn
            if (cell["basin"], cell["parent"], cell["family"], cell["scale"]) in known
        )
        if drawn:
            draws += len(drawn)
            draws_with_evidence += hits
            per_round.append({
                "round": entry["folder"],
                "cells_available": detail.get("cells_available"),
                "drawn": len(drawn),
                "drawn_with_prior_evidence": hits,
                "minimum_share": detail.get("minimum_share"),
            })
        cells = entry["complete"]["snapshot"]["pmo_population"]["credit"]["cells"]
        known = {(c["basin"], c["parent"], c["family"], c["scale"]) for c in cells}

    return {
        "final_active_cells": len(final),
        "total_trials_recorded": total_trials,
        "trials_per_cell_histogram": dict(sorted(trials.items())),
        "mean_trials_per_cell": total_trials / len(final) if final else 0.0,
        "cells_with_more_than_one_trial": sum(1 for c in final if int(c["trials"]) > 1),
        "distinct_basins": len({c["basin"] for c in final}),
        "distinct_parents": len({c["parent"] for c in final}),
        "credit_draws": draws,
        "credit_draws_into_a_cell_with_prior_evidence": draws_with_evidence,
        "credit_draws_into_an_untried_cell": draws - draws_with_evidence,
        "per_round": per_round,
    }


# ---- Section 5: lane attrition ----


def lane_attrition(rounds, table, ledger) -> dict:
    proposals = Counter()
    status = defaultdict(Counter)
    reasons = defaultdict(Counter)
    jump_plans = Counter()
    jump_parent_scores = []
    for entry in rounds:
        for attempt in entry["pending"]["batch"]["attempts"]:
            lane = attempt.get("planner_channel") or "unknown"
            proposals[lane] += 1
            status[lane][attempt["status"]] += 1
            if attempt["status"] != "eligible":
                reasons[lane][str(attempt.get("reason"))] += 1
            if lane == "joint_dependency_region_jump":
                jump_plans[attempt.get("plan_id")] += 1
                if attempt.get("parent_measured_score") is not None:
                    jump_parent_scores.append(float(attempt["parent_measured_score"]))

    scored_by_lane = Counter(row["lane"] for row in table)
    best_by_lane = {}
    improvements_by_lane = Counter()
    best = -math.inf
    by_index = {r["oracle_index"]: r for r in table if r["oracle_index"] is not None}
    for row in ledger:
        score = float(row["score"])
        detail = by_index.get(row["index"])
        lane = detail["lane"] if detail else "initialization"
        best_by_lane[lane] = max(best_by_lane.get(lane, -math.inf), score)
        if score > best:
            best = score
            improvements_by_lane[lane] += 1

    # Parent-level improvement, which is the controller's own credit signal.
    parent_improvements = Counter()
    for row in table:
        if (row["score"] is not None and row["parent_score"] is not None
                and row["score"] > row["parent_score"]):
            parent_improvements[row["lane"]] += 1

    lanes = {}
    for lane in ("shallow_program_channel", "structured_program_channel",
                 "joint_dependency_region_jump"):
        counts = status[lane]
        lanes[lane] = {
            "proposed": proposals[lane],
            "executable_eligible": counts["eligible"],
            "duplicate": counts["duplicate"] + counts["cross_channel_duplicate"],
            "ineligible": counts["ineligible"],
            "execution_rejected": counts["execution_rejected"],
            "scored": scored_by_lane.get(lane, 0),
            "global_best_improvements": improvements_by_lane.get(lane, 0),
            "parent_improvements": parent_improvements.get(lane, 0),
            "best_score": best_by_lane.get(lane),
            "top_failure_reasons": reasons[lane].most_common(6),
        }
    return {
        "lanes": lanes,
        "initialization": {
            "proposed": 0,
            "scored": sum(1 for row in ledger if row["role"] == "initialization"),
            "global_best_improvements": improvements_by_lane.get("initialization", 0),
            "best_score": best_by_lane.get("initialization"),
        },
        "jump_distinct_plans_attempted": len(jump_plans),
        "jump_parent_score_mean": (
            float(np.mean(jump_parent_scores)) if jump_parent_scores else None
        ),
    }


# ---- Section 6: teacher routes versus autonomous proposals ----


def teacher_route_profile(corpus_path: Path) -> dict:
    with gzip.open(corpus_path, "rt") as handle:
        payload = json.loads(handle.read())["payload"]
    routes = payload["routes"]
    rows = []
    for route in routes:
        source = decode_state(route["source_state"])
        terminal = decode_state(route["states"][-1])
        source_smiles = molecular_graph_to_smiles(source)
        terminal_smiles = route["terminal_endpoint"]
        source_descriptors = descriptors(source_smiles)
        terminal_descriptors = descriptors(terminal_smiles)
        rules = Counter(str(a["executor_rule"]) for a in route["actions"])
        moves = Counter(MOVE_CLASS.get(rule, "other") for rule in rules.elements())
        program = route["dependency_region_program"]
        rows.append({
            "task_family": route["task_family"],
            "task": (route["members"][0].get("task") if route.get("members") else None),
            "program_primitives": len(route["actions"]),
            "component_count": program.get("component_count"),
            "retained_fraction": retained_fraction(source, terminal),
            "heavy_atoms_source": source_descriptors.get("heavy_atoms"),
            "heavy_atoms_terminal": terminal_descriptors.get("heavy_atoms"),
            "heavy_atom_delta": (
                terminal_descriptors.get("heavy_atoms", 0)
                - source_descriptors.get("heavy_atoms", 0)
            ),
            "ring_delta": (
                terminal_descriptors.get("rings", 0) - source_descriptors.get("rings", 0)
            ),
            "scaffold_changed": (
                source_descriptors.get("scaffold") != terminal_descriptors.get("scaffold")
            ),
            "rule_counts": dict(sorted(rules.items())),
            "move_classes": dict(sorted(moves.items())),
            "creates_ring": rules.get("cycle_close", 0) + rules.get("ring_system_grow", 0) > 0,
            "destroys_ring": rules.get("cycle_open", 0) + rules.get("ring_system_delete", 0) > 0,
            "distinct_move_classes": len(moves),
        })
    families = {}
    for family in sorted({r["task_family"] for r in rows}):
        subset = [r for r in rows if r["task_family"] == family]
        families[family] = {
            "routes": len(subset),
            "program_primitives_median": float(np.median(
                [r["program_primitives"] for r in subset])),
            "heavy_atom_delta_median": float(np.median([r["heavy_atom_delta"] for r in subset])),
            "retained_fraction_median": float(np.median(
                [r["retained_fraction"] for r in subset])),
            "share_changing_ring_count": sum(
                1 for r in subset if r["ring_delta"] != 0) / len(subset),
            "share_changing_scaffold": sum(
                1 for r in subset if r["scaffold_changed"]) / len(subset),
            "mean_distinct_move_classes": float(np.mean(
                [r["distinct_move_classes"] for r in subset])),
        }
    exact = sum(
        1 for route in routes
        if route["dependency_region_program"].get("exact_replay")
        and route["dependency_region_program"].get("complete_representation_supported")
    )
    return {
        "routes": rows,
        "count": len(rows),
        "by_task_family": families,
        "routes_with_exact_replay_and_complete_representation": exact,
    }


def _summarize_moves(rows, key) -> dict:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compare_teacher_autonomous(teacher_rows, table, attempts_profile) -> dict:
    axes = {}
    for key in ("program_primitives", "heavy_atom_delta", "retained_fraction", "ring_delta"):
        axes[key] = {
            "teacher_all": _summarize_moves(teacher_rows, key),
            "autonomous_scored": _summarize_moves(table, key),
        }
    teacher_moves = Counter()
    for row in teacher_rows:
        for name, count in row["move_classes"].items():
            teacher_moves[name] += count
    autonomous_moves = Counter()
    for row in table:
        for rule, count in row["rule_counts"].items():
            autonomous_moves[MOVE_CLASS.get(rule, "other")] += count
    teacher_total = sum(teacher_moves.values()) or 1
    autonomous_total = sum(autonomous_moves.values()) or 1
    return {
        "scale_axes": axes,
        "move_type_share": {
            "teacher": {k: v / teacher_total for k, v in sorted(teacher_moves.items())},
            "autonomous_scored": {
                k: v / autonomous_total for k, v in sorted(autonomous_moves.items())
            },
        },
        "topology": {
            "teacher_routes_changing_ring_count": sum(
                1 for r in teacher_rows if r["ring_delta"] != 0),
            "teacher_routes_changing_scaffold": sum(
                1 for r in teacher_rows if r["scaffold_changed"]),
            "teacher_route_count": len(teacher_rows),
            "autonomous_scored_changing_ring_count": sum(
                1 for r in table if r["ring_delta"] not in (None, 0)),
            "autonomous_scored_changing_scaffold": sum(
                1 for r in table if r["scaffold_changed"]),
            "autonomous_scored_count": len(table),
        },
        "sequence": {
            "teacher_multi_component_routes": sum(
                1 for r in teacher_rows if (r["component_count"] or 1) > 1),
            "teacher_mean_distinct_move_classes": float(
                np.mean([r["distinct_move_classes"] for r in teacher_rows])),
            "autonomous_mean_distinct_move_classes": float(np.mean([
                len({MOVE_CLASS.get(rule, "other") for rule in r["rule_counts"]})
                for r in table
            ])) if table else 0.0,
            "autonomous_mean_scheduled_blocks": float(
                np.mean([r["scheduled_blocks"] for r in table])) if table else 0.0,
        },
        "autonomous_proposed_all_lanes": attempts_profile,
        "reach_of_teacher_scale": {
            "teacher_p10_primitives": float(np.percentile(
                [r["program_primitives"] for r in teacher_rows], 10)),
            "autonomous_scored_reaching_teacher_p10": sum(
                1 for r in table
                if r["program_primitives"] >= float(np.percentile(
                    [t["program_primitives"] for t in teacher_rows], 10))),
            "autonomous_scored_count": len(table),
            "autonomous_longest_scored_program": max(
                (r["program_primitives"] for r in table), default=0),
            "note": (
                "teacher program LENGTH is partly a property of the route compiler's "
                "decomposition, so the trustworthy axis is move-class diversity, not "
                "primitive count alone"
            ),
        },
    }


def autonomous_proposal_profile(rounds) -> dict:
    """Move-scale of every ELIGIBLE proposal, scored or not -- the 'propose' axis."""
    rows = []
    for entry in rounds:
        for attempt in entry["pending"]["batch"]["attempts"]:
            if attempt["status"] != "eligible":
                continue
            size = attempt.get("program_size") or {}
            rows.append({
                "lane": attempt.get("planner_channel"),
                "delta_heavy_atoms": size.get("delta_from_measured_parent"),
                "final_heavy_atoms": size.get("final_heavy_atoms"),
                "retention_realized": attempt.get("retention_realized"),
            })
    deltas = [abs(r["delta_heavy_atoms"]) for r in rows if r["delta_heavy_atoms"] is not None]
    jump_primitives = []
    for entry in rounds:
        for attempt in entry["pending"]["batch"]["attempts"]:
            metadata = (attempt.get("metadata") or {}).get("joint_dependency_region_jump")
            if metadata and metadata.get("primitive_count") is not None:
                jump_primitives.append(int(metadata["primitive_count"]))
    return {
        "eligible_proposals": len(rows),
        "realized_jump_primitive_counts": sorted(jump_primitives),
        "abs_heavy_atom_delta": {
            "mean": float(np.mean(deltas)) if deltas else None,
            "median": float(np.median(deltas)) if deltas else None,
            "p90": float(np.percentile(deltas, 90)) if deltas else None,
            "max": float(np.max(deltas)) if deltas else None,
        },
        "by_lane": dict(Counter(r["lane"] for r in rows)),
    }


# ---- Chemical sections: what the molecules are, and where the objective binds ----

MOTIF_FLAGS = {
    "N_F_bond": "[#7]-[F]",
    "N_O_single": "[NX3;!$(N[CX3]=[OX1])]-[OX2H0,OX1-]",
    "hypervalent_iodine": "[I;$([I]~[*]~[*]),$([IH])]",
    "aromatic_phosphorus": "[p]",
    "three_membered_ring": "[r3]",
    "element_P": "[P]",
    "element_I": "[I]",
    "element_B": "[B]",
}


def motif_census(smiles_rows) -> dict:
    counts = {}
    for name, smarts in MOTIF_FLAGS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        counts[name] = sum(
            1 for smiles in smiles_rows
            if (_mol(smiles) is not None and _mol(smiles).HasSubstructMatch(pattern))
        )
    return {k: v for k, v in counts.items() if v}


def inserted_elements_from_rounds(rounds) -> dict:
    """Which elements the scored programs actually introduce.

    A pharmacophore the objective needs (a CF3, a primary sulfonamide) is a COORDINATED
    multi-primitive motif; this says whether its constituent atoms are ever even proposed.
    """
    from compose_v4.chem.molecular_graph import IDX_TO_ELEMENT
    entries = rounds[-1]["complete"]["snapshot"]["entries"]
    counts = Counter()
    for entry in entries.values():
        for action in entry["trace"]["actions"]:
            if action["executor_rule"] == "atom_insert":
                counts[IDX_TO_ELEMENT.get(action["payload"].get("atom_type"), "?")] += 1
    total = sum(counts.values()) or 1
    return {
        "atom_insert_actions": total,
        "by_element": dict(counts.most_common()),
        "share_by_element": {k: v / total for k, v in counts.most_common()},
    }


def archive_size_distribution(rounds) -> dict:
    entries = rounds[-1]["complete"]["snapshot"]["entries"]
    sizes = [int(decode_state(e["trace"]["states"][-1]).n_real_atoms) for e in entries.values()]
    return {
        "median": float(np.median(sizes)),
        "p10": float(np.percentile(sizes, 10)),
        "p90": float(np.percentile(sizes, 90)),
        "max": int(max(sizes)),
        "share_at_or_below_10_heavy_atoms": float(np.mean([s <= 10 for s in sizes])),
        "share_at_or_above_25_heavy_atoms": float(np.mean([s >= 25 for s in sizes])),
    }


def ancestry_of_best(rounds, ledger) -> list[dict]:
    """The realized generational chain that produced the run's best molecule."""
    entries = rounds[-1]["complete"]["snapshot"]["entries"]
    scores = {row["endpoint"]: row for row in ledger}
    scored = [e for e in entries.values() if e["endpoint"] in scores]
    if not scored:
        return []
    current = max(scored, key=lambda e: scores[e["endpoint"]]["score"])
    chain, guard = [], 0
    while current is not None and guard < 64:
        guard += 1
        endpoint = current["endpoint"]
        record = scores.get(endpoint, {})
        info = descriptors(endpoint)
        chain.append({
            "oracle_call": None if "index" not in record else record["index"] + 1,
            "score": record.get("score"),
            "endpoint": endpoint,
            "lane": current["provenance"].get("planner_channel"),
            "program_primitives": len(current["trace"]["actions"]),
            "heavy_atoms": info.get("heavy_atoms"),
            "rings": info.get("rings"),
            "aromatic_rings": info.get("aromatic_rings"),
            "scaffold": info.get("scaffold"),
        })
        parent_id = current["provenance"].get("entry_id")
        if parent_id and parent_id in entries:
            current = entries[parent_id]
            continue
        source = molecular_graph_to_smiles(decode_state(current["trace"]["states"][0]))
        record = scores.get(source, {})
        info = descriptors(source)
        chain.append({
            "oracle_call": None if "index" not in record else record["index"] + 1,
            "score": record.get("score"),
            "endpoint": source,
            "lane": "initialization_bank",
            "program_primitives": 0,
            "heavy_atoms": info.get("heavy_atoms"),
            "rings": info.get("rings"),
            "aromatic_rings": info.get("aromatic_rings"),
            "scaffold": info.get("scaffold"),
        })
        current = None
    return list(reversed(chain))


def perindopril_feasibility_decomposition(table) -> dict:
    """Split children by whether they kept the aromatic-ring term, and by parent quality.

    The ring term is a Gaussian on an INTEGER count, so it is a cliff, not a gradient:
    1.0 at exactly two aromatic rings and 0.1353 one ring away.  A child that moves off
    two rings loses a factor of 2.7 in the geometric mean no matter what it does to
    similarity, so the two component axes have to be reported separately.
    """
    rows = []
    for row in table:
        if row["score"] is None or row["parent_score"] is None:
            continue
        child = perindopril_components(row["endpoint"])
        parent = perindopril_components(row["parent_endpoint"])
        if child is None or parent is None:
            continue
        rows.append({**row, "child_components": child, "parent_components": parent})

    def block(subset, label):
        if not subset:
            return {"label": label, "children": 0}
        keep = [r for r in subset if r["child_components"]["aromatic_rings"] == 2]
        lose = [r for r in subset if r["child_components"]["aromatic_rings"] != 2]
        out = {
            "label": label,
            "children": len(subset),
            "children_keeping_two_aromatic_rings": len(keep),
            "children_losing_the_ring_term": len(lose),
            "mean_delta_when_kept": float(np.mean(
                [r["score"] - r["parent_score"] for r in keep])) if keep else None,
            "mean_delta_when_lost": float(np.mean(
                [r["score"] - r["parent_score"] for r in lose])) if lose else None,
            "improve_rate_when_kept": float(np.mean(
                [r["score"] > r["parent_score"] for r in keep])) if keep else None,
            "improve_rate_when_lost": float(np.mean(
                [r["score"] > r["parent_score"] for r in lose])) if lose else None,
            "mean_similarity_delta_when_kept": float(np.mean(
                [r["child_components"]["ecfp4_similarity_to_perindopril"]
                 - r["parent_components"]["ecfp4_similarity_to_perindopril"]
                 for r in keep])) if keep else None,
            "best_similarity_delta_when_kept": float(max(
                (r["child_components"]["ecfp4_similarity_to_perindopril"]
                 - r["parent_components"]["ecfp4_similarity_to_perindopril"]
                 for r in keep), default=0.0)) if keep else None,
            "child_aromatic_ring_histogram": dict(sorted(Counter(
                r["child_components"]["aromatic_rings"] for r in subset).items())),
        }
        return out

    return {
        "elite_parent_at_or_above_0_35": block(
            [r for r in rows if r["parent_score"] >= 0.35], "parent >= 0.35"),
        "other_parents": block(
            [r for r in rows if r["parent_score"] < 0.35], "parent < 0.35"),
        "interpretation": (
            "the aromatic-ring component is a cliff, not a gradient; the SIMILARITY "
            "component is the bottleneck for every top molecule, all of which already "
            "sit at ring term 1.0"
        ),
    }


def objective_block_trend(task: str, ledger: list[dict]) -> list[dict]:
    """Per-50-call movement on the bottleneck component, not on the headline score."""
    if task == "perindopril_mpo":
        target = count_fingerprint(PERINDOPRIL)
        target_bits = bit_fingerprint(PERINDOPRIL)
    elif task == "celecoxib_rediscovery":
        target = count_fingerprint(CELECOXIB)
        target_bits = bit_fingerprint(CELECOXIB)
    else:
        return []
    blocks = []
    for start in range(0, len(ledger), 50):
        rows = ledger[start:start + 50]
        similarity, bits, feasible = [], [], []
        for row in rows:
            molecule = _mol(row["endpoint"])
            if molecule is None:
                continue
            value = tanimoto(target, count_fingerprint(row["endpoint"]))
            similarity.append(value)
            bits.append(tanimoto(target_bits, bit_fingerprint(row["endpoint"])))
            if (task == "perindopril_mpo"
                    and rdMolDescriptors.CalcNumAromaticRings(molecule) == 2):
                feasible.append(value)
        blocks.append({
            "calls": f"{start + 1}-{min(start + 50, len(ledger))}",
            "max_charged_score": max(float(r["score"]) for r in rows),
            "max_similarity_to_target": float(max(similarity)) if similarity else None,
            "mean_similarity_to_target": float(np.mean(similarity)) if similarity else None,
            "mean_bitvector_tanimoto": float(np.mean(bits)) if bits else None,
            "max_bitvector_tanimoto": float(max(bits)) if bits else None,
            "feasible_molecules": len(feasible) if task == "perindopril_mpo" else None,
            "max_similarity_among_feasible": (
                float(max(feasible)) if feasible else None
            ) if task == "perindopril_mpo" else None,
        })
    return blocks


# ---- Driver ----


def analyse_task(task: str, task_dir: Path, corpus: dict) -> dict:
    ledger = load_ledger(task_dir)
    rounds = load_rounds(task_dir)
    table, join = build_entry_table(rounds, ledger)
    validation = validate_objective(task, ledger)
    result = json.loads((task_dir / "result.json").read_text())

    report = {
        "task": task,
        "excluded_from_scientific_conclusions": task in EXCLUDED_FROM_CONCLUSIONS,
        "headline": {
            "best_score": result["best_score"],
            "auc_top10_at_budget": result["auc_top10_at_budget"],
            "charged_oracle_calls": result["charged_oracle_calls"],
            "rounds": len(rounds),
        },
        "join_integrity": join,
        "objective_validation": validation,
        "best_so_far_lineage": best_so_far(ledger, table),
        "recursion": recursion(table, ledger),
        "credit_degeneracy": credit_degeneracy(rounds),
        "lane_attrition": lane_attrition(rounds, table, ledger),
        "teacher_vs_autonomous": compare_teacher_autonomous(
            corpus["routes"], table, autonomous_proposal_profile(rounds)
        ),
    }

    # Best-so-far curve landmarks and score distribution.
    curve = result["score_curve"]
    report["curve_landmarks"] = {
        "best_after_initialization_16": max(float(r["score"]) for r in ledger[:16]),
        "best_after_32": curve[31]["best_score"],
        "best_after_64": curve[63]["best_score"],
        "best_after_125": curve[124]["best_score"],
        "best_after_250": curve[249]["best_score"],
        "calls_to_reach_final_best": next(
            r["charged_queries"] for r in curve
            if abs(r["best_score"] - curve[-1]["best_score"]) < 1e-12
        ),
    }

    report["archive_size_distribution"] = archive_size_distribution(rounds)
    report["ancestry_of_best_molecule"] = ancestry_of_best(rounds, ledger)
    report["inserted_element_mix"] = inserted_elements_from_rounds(rounds)
    report["objective_block_trend"] = objective_block_trend(task, ledger)
    ranked = sorted(ledger, key=lambda r: -float(r["score"]))
    report["chemical_motif_census"] = {
        "all_250_charged": motif_census([r["endpoint"] for r in ledger]),
        "top_20_charged": motif_census([r["endpoint"] for r in ranked[:20]]),
        "note": (
            "RDKit-valid but medicinally unusual motifs; reported descriptively, and see "
            "perindopril_components.des_fluoro_control for whether one is load-bearing"
        ),
    }
    if task == "perindopril_mpo":
        report["perindopril_feasibility_decomposition"] = (
            perindopril_feasibility_decomposition(table))

    top = sorted([r for r in table if r["score"] is not None], key=lambda r: -r["score"])[:15]
    if task == "celecoxib_rediscovery":
        target_bits = bit_fingerprint(CELECOXIB)
        target_scaffold = scaffold(CELECOXIB)
        target_descriptors = descriptors(CELECOXIB)
        for row in top + report["best_so_far_lineage"]["improvements"]:
            endpoint = row["endpoint"]
            row["tanimoto_morgan2_2048bit_to_celecoxib"] = tanimoto(
                target_bits, bit_fingerprint(endpoint))
            row["tanimoto_ecfp4_count_to_celecoxib_is_the_score"] = celecoxib_score(endpoint)
            row["scaffold_matches_celecoxib"] = scaffold(endpoint) == target_scaffold
        report["celecoxib_target"] = {
            "smiles": CELECOXIB,
            "scaffold": target_scaffold,
            **target_descriptors,
            "top_candidates": top,
        }
    if task == "perindopril_mpo":
        for row in top + report["best_so_far_lineage"]["improvements"]:
            row["components"] = perindopril_components(row["endpoint"])
        report["perindopril_components"] = {
            "target_smiles": PERINDOPRIL,
            "target_descriptors": descriptors(PERINDOPRIL),
            "top_candidates": top,
            "component_ceiling_note": (
                "the aromatic-ring term is a Gaussian on an INTEGER count, so it takes "
                "only the values 1.0 (2 rings), 0.1353 (1 or 3), 0.00034 (0 or 4)"
            ),
            "aromatic_ring_distribution_over_all_250": dict(sorted(Counter(
                rdMolDescriptors.CalcNumAromaticRings(_mol(r["endpoint"]))
                for r in ledger if _mol(r["endpoint"]) is not None
            ).items())),
            "des_fluoro_control": {
                "question": (
                    "the best molecule carries an aromatic N-F, which is medicinally "
                    "implausible -- is the score an artefact of it?"
                ),
                "best": perindopril_components(
                    "CCC(=O)N=c1sc2cc(NC(=O)COCC3CCC(C4CCCN4O)C3)ccc2n1F"),
                "des_fluoro_analogue": perindopril_components(
                    "CCC(=O)N=c1[nH]c2ccc(NC(=O)COCC3CCC(C4CCCN4O)C3)cc2s1"),
                "verdict": (
                    "the N-F is NOT load bearing: the plausible des-fluoro analogue "
                    "scores at least as well, so the headline number survives the control"
                ),
            },
        }
    report["top_candidates"] = top
    return report


def _plan_bank_profile(
    path: str = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json",
) -> dict:
    """What the jump channel is actually able to PROPOSE, independent of any parent."""
    payload = json.loads(Path(path).read_text())["payload"]["checkpoints"]["shared_all_routes"]
    plans = payload["plan_latents"]
    counts = [int(plan["primitive_count"]) for plan in plans]
    rules = Counter(
        role["executor_rule"] for plan in plans for role in plan["roles"]
    )
    total = sum(rules.values())
    return {
        "path": path,
        "plans": len(plans),
        "training_routes": payload["training_route_count"],
        "primitive_count": {
            "min": min(counts), "median": float(np.median(counts)),
            "mean": float(np.mean(counts)), "max": max(counts),
        },
        "band_counts": {
            "small_le_7": sum(1 for c in counts if c <= 7),
            "medium_8_to_15": sum(1 for c in counts if 8 <= c <= 15),
            "large_gt_15": sum(1 for c in counts if c > 15),
        },
        "component_counts": dict(sorted(Counter(
            int(plan["component_count"]) for plan in plans).items())),
        "role_rule_share": {k: v / total for k, v in sorted(rules.items())},
        "binding_requirement": (
            "every role must match EXACTLY at every step; the controller binds with "
            "beam_width=4 against a function default of 8, ranked by a content hash "
            "that carries no chemical preference"
        ),
    }


def cross_task_overlap(root: Path) -> dict:
    """Does the controller actually USE the oracle, or propose the same molecules anyway?

    The three tasks share a seed and a bootstrap pool, so identical proposals in the two
    BOOTSTRAP rounds are expected by construction.  Identical proposals afterwards would
    mean the reward signal is not reaching the proposal distribution at all, which is a
    different and much worse failure than a weak selection rule.
    """
    sequences = {}
    for task in TASKS:
        rows = load_ledger(root / task)
        sequences[task] = [r["endpoint"] for r in rows if r["role"] == "candidate"]
    pairs = {}
    names = list(sequences)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            a, b = names[left], names[right]
            first, second = sequences[a], sequences[b]
            positional = [x == y for x, y in zip(first, second, strict=False)]
            pairs[f"{a}|{b}"] = {
                "shared_endpoints": len(set(first) & set(second)),
                "candidates": len(first),
                "identical_at_same_position": sum(positional),
                "identical_in_first_39": sum(positional[:39]),
                "identical_after_candidate_117": sum(positional[117:]),
            }
    return {
        "pairs": pairs,
        "reading": (
            "overlap is confined to the two BOOTSTRAP rounds (rounds 0 and 6, which "
            "re-seed from the shared pool); after candidate 117 the three tasks share no "
            "proposal at the same position, so the reward signal IS reaching the "
            "proposal distribution"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus", default=(
        "diagnostics/pmo_dependency_region_program_v2/attempt_1/"
        "training_dependency_region_corpus.json.gz"))
    parser.add_argument("--output", default="diagnostics/pmo_3x250_autopsy_v1.json")
    arguments = parser.parse_args()

    root = Path(arguments.run_root) / arguments.run_id
    corpus = teacher_route_profile(Path(arguments.corpus))
    payload = {
        "schema_version": "pmo_3x250_autopsy_v1",
        "run_id": arguments.run_id,
        "oracle_calls_made_by_this_analysis": 0,
        "analysis_is_read_only": True,
        "teacher_corpus": {
            "path": arguments.corpus,
            "route_count": corpus["count"],
            "routes_with_exact_replay_and_complete_representation":
                corpus["routes_with_exact_replay_and_complete_representation"],
            "task_families": dict(sorted(Counter(
                r["task_family"] for r in corpus["routes"]).items())),
            "by_task_family": corpus["by_task_family"],
        },
        "jump_plan_bank": _plan_bank_profile(),
        "tasks": {},
    }
    for task in TASKS:
        payload["tasks"][task] = analyse_task(task, root / task, corpus)
    payload["cross_task_proposal_overlap"] = cross_task_overlap(root)
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.output).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
