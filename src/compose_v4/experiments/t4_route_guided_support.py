"""Zero-oracle support probe for a route-guided constructive proposal expert.

The route corpus supplies a target-held-out structural prior over constructive
``(WHERE, HOW)`` decisions.  The production compiler still proposes and exact-executes
every concrete stage.  This module measures whether using that prior *inside* program
generation changes the completed molecular endpoints, compared with uniform selection
from the same bounded stage-candidate mechanism.

No target name enters the proposal model.  Target is used only to remove that target's
rows before fitting and, after generation, to select the appropriate frozen endpoint
gate and diagnostic motif audit.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED

from compose_v4.control.constructive_features import (
    MODE_FEATURE_NAMES,
    SITE_FEATURE_NAMES,
    mode_matrix,
)
from compose_v4.control.constructive_guided_composition import compose_guided
from compose_v4.control.constructive_policy import PolicyShape, fit_additive_shared
from compose_v4.experiments.t4_constructive_prior import encode, mode_vocabulary
from compose_v4.experiments.t4_fiber_campaign import (
    COMPOSE_VALID,
    QED_MIN,
    REPRESENTABLE_HEAVY_ATOMS,
    SA_MAX,
    Fiber,
    sascorer,
)
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

SCHEMA_VERSION = "t4_route_guided_support_v1"

_ESTER = Chem.MolFromSmarts("[CX4][OX2][CX3](=O)[CH2]")
_AMIDE = Chem.MolFromSmarts("[NX3][CX3](=O)[CH2]")
_NCN_RING = Chem.MolFromSmarts("[NX3;R][CX4;R][NX3;R]")
_NCCN_RING = Chem.MolFromSmarts("[NX3;R][CX4;R][CX4;R][NX3;R]")
_UREA = Chem.MolFromSmarts("[NX3;R][CX3](=O)[NX3]")


def load_decisions(path: Path) -> list[dict]:
    """Load the immutable constructive corpus without changing row identity."""
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def fit_held_target_prior(rows: list[dict], held_target: str) -> tuple[dict, list[dict], dict]:
    """Fit the selected additive proposal law after excluding one complete target.

    The mode vocabulary is also derived after the split.  An unseen runtime mode remains
    in generator support but receives no learned score and is reachable through the
    explicit exploration lane.
    """
    train = [row for row in rows if row.get("target") != held_target]
    if not train:
        raise ValueError(f"holding out {held_target!r} left no constructive decisions")
    if any(row.get("target") == held_target for row in train):
        raise RuntimeError(f"target leakage in {held_target!r} prior")
    vocabulary = mode_vocabulary(train)
    examples = encode(train, vocabulary)
    shape = PolicyShape(len(SITE_FEATURE_NAMES), len(MODE_FEATURE_NAMES))
    penalties = {"site": 1e-2, "mode": 1e-2, "interaction": 1e7}
    model = fit_additive_shared(
        examples,
        shape,
        mode_matrix(vocabulary),
        penalties=penalties,
    )
    audit = {
        "held_target": held_target,
        "corpus_rows": len(rows),
        "training_rows": len(train),
        "encoded_training_rows": len(examples),
        "excluded_held_target_rows": len(rows) - len(train),
        "training_targets": sorted({row["target"] for row in train}),
        "held_target_absent_from_training": held_target not in {row["target"] for row in train},
        "mode_vocabulary_size": len(vocabulary),
        "penalties": penalties,
        "model_converged": model["converged"],
        "model_objective": model["objective"],
        "model_iterations": model["iterations"],
        "weighting": (
            "one unit per corpus-deduplicated construction decision; no docking score "
            "enters q_route"
        ),
    }
    return model, vocabulary, audit


def jak2_motif_flags(smiles: str) -> dict[str, bool]:
    """The predeclared structural funnel from the measured JAK2 basin autopsy."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {
            "ester_broken": False,
            "amide": False,
            "ncn_ring": False,
            "nccn_ring": False,
            "diamine_ring": False,
            "urea": False,
            "basin": False,
        }
    amide = mol.HasSubstructMatch(_AMIDE)
    ncn = mol.HasSubstructMatch(_NCN_RING)
    nccn = mol.HasSubstructMatch(_NCCN_RING)
    ring = ncn or nccn
    return {
        "ester_broken": not mol.HasSubstructMatch(_ESTER),
        "amide": amide,
        "ncn_ring": ncn,
        "nccn_ring": nccn,
        "diamine_ring": ring,
        "urea": mol.HasSubstructMatch(_UREA),
        "basin": amide and ring,
    }


def endpoint_audit(smiles: str, fiber: Fiber) -> dict:
    """Descriptive gate factors plus the authoritative all-factor Fiber decision."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or "." in smiles:
        return {
            "canonical_smiles": None,
            "parseable": False,
            "official_compose_valid": False,
        }
    canonical = Chem.MolToSmiles(mol)
    heavy = mol.GetNumHeavyAtoms()
    similarity = DataStructs.TanimotoSimilarity(fiber.seed, fiber.generator.GetFingerprint(mol))
    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol)
    official = fiber.check(canonical)
    return {
        "canonical_smiles": canonical,
        "parseable": True,
        "heavy": heavy,
        "representable": heavy <= REPRESENTABLE_HEAVY_ATOMS,
        "similarity": similarity,
        "similarity_pass": similarity >= fiber.delta,
        "qed": qed,
        "qed_pass": qed >= QED_MIN,
        "sa": sa,
        "sa_pass": sa <= SA_MAX,
        "structurally_valid": structurally_valid(canonical),
        "official_compose_valid": official is not None,
        "official_properties": official,
    }


def _rates(counts: Counter, denominator: int) -> dict[str, dict]:
    return {
        key: {"count": int(value), "rate": value / denominator if denominator else 0.0}
        for key, value in sorted(counts.items())
    }


def run_arm(
    source,
    original_seed: str,
    target: str,
    model,
    vocabulary,
    *,
    arm: str,
    attempts: int,
    seed: int,
    delta: float,
    depths: tuple[int, ...],
    candidates_per_step: int,
    route_exploration: float,
    temperature: float,
    progress=None,
) -> dict:
    """Run one equal-budget zero-oracle arm and retain a complete endpoint ledger."""
    if arm not in ("uniform_same_pool", "route_prior"):
        raise ValueError(f"unknown support-probe arm {arm!r}")
    exploration = 1.0 if arm == "uniform_same_pool" else route_exploration
    rng = np.random.default_rng(np.random.SeedSequence([seed]))
    fiber = Fiber(original_seed, delta, support=COMPOSE_VALID)
    seed_mol = Chem.MolFromSmiles(original_seed)
    seed_heavy = seed_mol.GetNumHeavyAtoms()
    began = perf_counter()
    failures = Counter()
    raw_flags, feasible_flags = Counter(), Counter()
    rows = []
    unique_raw, unique_feasible = set(), set()
    for attempt in range(attempts):
        if progress is not None:
            progress(attempt, attempts, len(unique_feasible))
        depth = depths[int(rng.integers(len(depths)))]
        try:
            trace, guidance = compose_guided(
                source,
                rng,
                depth,
                model,
                vocabulary,
                candidates_per_step=candidates_per_step,
                exploration=exploration,
                temperature=temperature,
            )
        except (ValueError, RuntimeError) as exc:
            failures[type(exc).__name__] += 1
            rows.append(
                {
                    "attempt": attempt,
                    "status": "generation_failed",
                    "reason": type(exc).__name__,
                }
            )
            continue

        audit = endpoint_audit(trace["endpoint"], fiber)
        canonical = audit["canonical_smiles"]
        motif = jak2_motif_flags(canonical) if target == "jak2" else {}
        factors = {
            "complete": True,
            "unique_raw": canonical not in unique_raw,
            "qed_pass": bool(audit.get("qed_pass")),
            "similarity_pass": bool(audit.get("similarity_pass")),
            "sa_pass": bool(audit.get("sa_pass")),
            "structurally_valid": bool(audit.get("structurally_valid")),
            "official_compose_valid": bool(audit["official_compose_valid"]),
            "heavy_reduction_1plus": bool(audit.get("heavy", seed_heavy) <= seed_heavy - 1),
            "heavy_reduction_7plus": bool(audit.get("heavy", seed_heavy) <= seed_heavy - 7),
            **motif,
        }
        for key, value in factors.items():
            raw_flags[key] += int(value)
        if canonical is not None:
            unique_raw.add(canonical)
        if audit["official_compose_valid"]:
            if canonical is not None:
                unique_feasible.add(canonical)
            for key, value in factors.items():
                feasible_flags[key] += int(value)

        rows.append(
            {
                "attempt": attempt,
                "status": "complete",
                "endpoint": canonical,
                "requested_depth": depth,
                "primitive_count": len(trace["actions"]),
                "block_count": len(trace["blocks"]),
                "changed_originals": len(trace["actual_changes"]["changed_original_slots"]),
                "surviving_new_atoms": trace["actual_changes"]["surviving_new_atoms"],
                "audit": audit,
                "motif": motif,
                "guidance": guidance,
            }
        )
    completed = sum(row["status"] == "complete" for row in rows)
    feasible = int(raw_flags["official_compose_valid"])
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "attempt_budget": attempts,
        "attempted": len(rows),
        "completed": completed,
        "generation_failures": dict(sorted(failures.items())),
        "unique_raw_endpoints": len(unique_raw),
        "feasible_programs": feasible,
        "unique_feasible_endpoints": len(unique_feasible),
        "raw_factor_rates": _rates(raw_flags, completed),
        "feasible_factor_rates": _rates(feasible_flags, feasible),
        "elapsed_seconds": perf_counter() - began,
        "new_oracle_calls": 0,
        "ledger": rows,
    }


def aggregate_attempt_results(results: list[dict], *, arm: str, attempt_budget: int) -> dict:
    """Deterministically reduce independent one-attempt workers into one arm result."""
    if len(results) != attempt_budget:
        raise ValueError(
            f"expected {attempt_budget} attempt results for {arm}, received {len(results)}"
        )
    ordered = sorted(results, key=lambda result: result["worker_seed"])
    if len({result["worker_seed"] for result in ordered}) != attempt_budget:
        raise ValueError("support-probe worker seeds are not unique")
    if any(result["arm"] != arm or result["attempted"] != 1 for result in ordered):
        raise ValueError("support-probe worker result does not match its arm contract")
    if any(result["new_oracle_calls"] != 0 for result in ordered):
        raise ValueError("a zero-oracle support worker reported an oracle call")

    completed = sum(result["completed"] for result in ordered)
    feasible = sum(result["feasible_programs"] for result in ordered)
    raw_counts, feasible_counts, failures = Counter(), Counter(), Counter()
    ledger = []
    elapsed = []
    for attempt, result in enumerate(ordered):
        for key, value in result["raw_factor_rates"].items():
            raw_counts[key] += value["count"]
        for key, value in result["feasible_factor_rates"].items():
            feasible_counts[key] += value["count"]
        failures.update(result["generation_failures"])
        elapsed.append(result["elapsed_seconds"])
        row = dict(result["ledger"][0])
        row["attempt"] = attempt
        row["worker_seed"] = result["worker_seed"]
        ledger.append(row)
    unique_raw = {
        row["endpoint"]
        for row in ledger
        if row["status"] == "complete" and row.get("endpoint") is not None
    }
    unique_feasible = {
        row["endpoint"]
        for row in ledger
        if row["status"] == "complete"
        and row.get("endpoint") is not None
        and row["audit"]["official_compose_valid"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "attempt_budget": attempt_budget,
        "attempted": len(ordered),
        "completed": completed,
        "generation_failures": dict(sorted(failures.items())),
        "unique_raw_endpoints": len(unique_raw),
        "feasible_programs": feasible,
        "unique_feasible_endpoints": len(unique_feasible),
        "raw_factor_rates": _rates(raw_counts, completed),
        "feasible_factor_rates": _rates(feasible_counts, feasible),
        "worker_seconds_sum": sum(elapsed),
        "worker_seconds_max": max(elapsed, default=0.0),
        "new_oracle_calls": 0,
        "ledger": ledger,
    }
