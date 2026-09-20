"""Zero-oracle stage audit of the T4 held-target proposal funnel.

WHY THIS EXISTS
---------------
Five of the fifteen T4 held-target cells terminate at ``candidate_exhaustion`` after
one charged call: the root is docked, round one proposes nothing the task fiber
admits, and ``select_batch`` returns an empty batch.  Within a protein the held-target
prior is byte-identical across its three cells, so the difference is carried by the
SOURCE molecule, not by the weights.  This driver reproduces round one of those cells
offline -- no docking, no Modal, no oracle -- and records where support is lost.

WHAT IS REPRODUCED, EXACTLY
---------------------------
``modal_apps/t4_integrated_route_fiber_parp1_app.py`` (the app every held-target
launcher wraps) builds round one as: one parent (the root, because the archive holds
only the docked root), three experts, and per-expert proposal seed

    controller_seed + 1_000_003 * round_index + 10_007 * parent_index + 101 * expert_index

with ``round_index = 1`` and ``parent_index = 0``.  This driver calls the SAME library
functions with the SAME arguments -- ``t4_fiber_campaign.expand`` for the ``shallow``
and ``anchored_replacement`` lanes, ``route_distilled_goal_expert``
``propose_route_expert_candidates`` plus ``Fiber.check`` for ``route_complete_region``
-- then the production ``merge_expert_pools`` / ``attach_features`` / ``select_batch``.
Nothing is transcribed; the production modules are imported and run.

HOW THE FUNNEL IS INSTRUMENTED WITHOUT TOUCHING PRODUCTION CODE
---------------------------------------------------------------
``expand`` returns only endpoints that already passed the fiber, so the interior of the
funnel is invisible from its return value.  This driver installs MEASUREMENT-ONLY
wrappers on the module globals ``t4_fiber_campaign`` resolves at call time, inside this
process only; no file on disk is modified and every wrapper delegates the decision to
the original callable.  ``Fiber.check`` is wrapped so that the ORIGINAL method decides
accept/reject and the wrapper independently recomputes the per-gate decomposition; the
two verdicts are compared on every call and any disagreement is reported as
``gate_reconstruction_disagreements`` (expected: zero).

MARGINS, NOT BINARY COUNTS
--------------------------
Every endpoint that reaches the fiber records ``delta_sim = similarity - delta``,
``delta_qed = QED - 0.6`` and ``delta_sa = 4.0 - SA`` whether or not it passed, so a
near miss at similarity 0.58 is distinguishable from genuinely absent support.

SLOT SEMANTICS
--------------
``expand`` and the route lane both build their source with
``pad_molecular_graph(smiles_to_molecular_graph(parent), 48)``, so atom birth is
expressible.  The driver asserts that padding is present on every source before doing
any work (see ``slot_semantics_preflight``); a tight graph would silently delete the
whole insertion family from the legal support.

OUTPUT
------
One shard per cell under ``--out``; ``--reduce`` merges shards into
``diagnostics/t4_support_stage_audit_v1.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.route_distilled_goal_expert import (
    RouteDistilledGoalExpert,
    propose_route_expert_candidates,
)
from compose_v4.experiments import t4_fiber_campaign as campaign
from compose_v4.experiments.t4_fiber_campaign import (
    BENCHMARK_ONLY,
    LEGACY_SCREENED,
    QED_MIN,
    REPRESENTABLE_HEAVY_ATOMS,
    SA_MAX,
    Fiber,
    expand,
)
from compose_v4.experiments.t4_integrated_route_fiber import (
    EXPERTS,
    attach_features,
    expert_census,
    merge_expert_pools,
    select_batch,
)
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

SCHEMA_VERSION = "t4_support_stage_audit_v1"

ROOT = Path(__file__).resolve().parents[1]

#: The observed terminal status of each cell in the live scored held-target campaign,
#: supplied by the run owner.  It is a LABEL for grouping only; nothing in this audit
#: reads a docking score, and no arm of the measurement depends on it.
OBSERVED_STATUS = {
    "braf_0": "candidate_exhaustion",
    "braf_1": "candidate_exhaustion",
    "braf_2": "searching",
    "fa7_0": "candidate_exhaustion",
    "fa7_1": "searching",
    "fa7_2": "candidate_exhaustion",
    "5ht1b_0": "searching",
    "5ht1b_1": "searching",
    "5ht1b_2": "candidate_exhaustion",
    "parp1_0": "searching",
    "parp1_1": "searching",
    "parp1_2": "searching",
    "jak2_0": "searching",
    "jak2_1": "searching",
    "jak2_2": "searching",
}

#: The slot capacity every T4 proposal lane pads its source to.  It is larger than the
#: fiber's 40-heavy-atom representability ceiling on purpose: the program may transit
#: through wider intermediates, only the ENDPOINT is capped.
T4_PROPOSAL_SLOTS = 48


# ---- Contract access ----


def contract_path(protein: str) -> Path:
    return ROOT / f"configs/t4_held_target_distilled_{protein}_d06_250.json"


def checkpoint_path(protein: str) -> Path:
    return ROOT / f"diagnostics/t4_held_target_distillation_quality_v1/{protein}_checkpoint.json"


def load_contract(protein: str) -> dict:
    envelope = json.loads(contract_path(protein).read_text())
    payload = envelope["payload"]
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"{protein} contract payload hash does not verify")
    return payload


def verify_runtime_inputs(contract: dict) -> dict[str, str]:
    """Refuse to audit a tree whose pinned runtime inputs are not the sealed ones."""

    from compose_v4.experiments.continuation_profile import sha256_file

    mismatched = {}
    for relative, expected in contract["runtime_inputs_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            mismatched[relative] = actual
    if mismatched:
        raise ValueError(f"runtime input mismatch: {mismatched}")
    return dict(contract["runtime_inputs_sha256"])


def load_expert(protein: str) -> RouteDistilledGoalExpert:
    envelope = json.loads(checkpoint_path(protein).read_text())
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError("route expert checkpoint payload hash mismatch")
    return RouteDistilledGoalExpert.from_checkpoint(envelope["payload"]["expert"])


# ---- Preflight ----


def slot_semantics_preflight(smiles: str) -> dict:
    """Assert the production proposal source carries free slots for atom birth.

    ``assert_production_state_semantics`` compares a realized legal-family census
    against the census the Process-V2 compiler recorded for the SAME source state.  The
    T4 seeds are not Process-V2 corpus states and no compiler census exists for them,
    and the factorized successor model those helpers need is not part of this
    controller's runtime at all -- so that exact check cannot be run here without
    inventing a reference.  What IS checkable, and is the half of the preflight that the
    known failure mode turns on, is the slot semantics: the tight graph is built and
    compared against the padded graph the production lanes actually use.
    """

    tight = smiles_to_molecular_graph(smiles)
    padded = pad_molecular_graph(smiles_to_molecular_graph(smiles), T4_PROPOSAL_SLOTS)
    tight_slots = int(tight.atom_types.shape[0])
    padded_slots = int(padded.atom_types.shape[0])
    heavy = int(Chem.MolFromSmiles(smiles).GetNumHeavyAtoms())
    evidence = {
        "smiles": smiles,
        "heavy_atoms": heavy,
        "tight_graph_slots": tight_slots,
        "padded_graph_slots": padded_slots,
        "free_slots_in_production_source": padded_slots - heavy,
        "tight_graph_would_forbid_atom_birth": tight_slots <= heavy,
        "passed": padded_slots == T4_PROPOSAL_SLOTS and padded_slots > heavy,
    }
    if not evidence["passed"]:
        raise RuntimeError(f"slot-semantics preflight failed for {smiles!r}: {evidence}")
    return evidence


# ---- Endpoint description ----


class SourceContext:
    """Cheap, lane-independent descriptors of one endpoint relative to its source."""

    def __init__(self, fiber: Fiber, source_smiles: str):
        self.fiber = fiber
        self.source_smiles = source_smiles
        molecule = Chem.MolFromSmiles(source_smiles)
        self.source_mol = molecule
        self.heavy = molecule.GetNumHeavyAtoms()
        self.rings = molecule.GetRingInfo().NumRings()
        self.bits = set(fiber.generator.GetFingerprint(molecule).GetOnBits())
        self.scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)

    def describe(self, molecule) -> dict:
        bits = set(self.fiber.generator.GetFingerprint(molecule).GetOnBits())
        retained = len(self.bits & bits) / max(len(self.bits), 1)
        heavy = molecule.GetNumHeavyAtoms()
        rings = molecule.GetRingInfo().NumRings()
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        except (ValueError, RuntimeError):
            scaffold = None
        return {
            "heavy": heavy,
            "heavy_delta": heavy - self.heavy,
            "ring_delta": rings - self.rings,
            "retained_source_bit_fraction": round(retained, 5),
            "added_bit_fraction": round(len(bits - self.bits) / max(len(bits), 1), 5),
            "scaffold_preserved": bool(scaffold is not None and scaffold == self.scaffold),
            "direction": (
                "grow" if heavy > self.heavy else "prune" if heavy < self.heavy else "replace"
            ),
        }


def gate_decomposition(fiber: Fiber, smiles: str) -> dict:
    """Recompute Fiber.check stage by stage; the original method stays authoritative."""

    row: dict = {"smiles_in": smiles}
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    row["parses"] = molecule is not None
    row["fragmented"] = "." in (smiles or "")
    if molecule is None or row["fragmented"]:
        row["stage"] = "chemically_valid"
        row["verdict"] = False
        return row
    heavy = molecule.GetNumHeavyAtoms()
    row["heavy"] = heavy
    row["capacity_ok"] = heavy <= REPRESENTABLE_HEAVY_ATOMS
    if not row["capacity_ok"]:
        row["stage"] = "capacity_valid"
        row["verdict"] = False
        return row
    similarity = DataStructs.TanimotoSimilarity(
        fiber.seed, fiber.generator.GetFingerprint(molecule)
    )
    quality = QED.qed(molecule)
    access = sascorer.calculateScore(molecule)
    row["similarity"] = similarity
    row["qed"] = quality
    row["sa"] = access
    row["delta_sim"] = similarity - fiber.delta
    row["delta_qed"] = quality - QED_MIN
    row["delta_sa"] = SA_MAX - access
    row["sim_ok"] = similarity >= fiber.delta
    row["qed_ok"] = quality >= QED_MIN
    row["sa_ok"] = access <= SA_MAX
    if not (row["sim_ok"] and row["qed_ok"] and row["sa_ok"]):
        row["stage"] = (
            "similarity"
            if not row["sim_ok"]
            else "qed"
            if not row["qed_ok"]
            else "sa"
        )
        row["verdict"] = False
        return row
    if fiber.support != BENCHMARK_ONLY:
        row["med_chem_ok"] = bool(structurally_valid(smiles))
        if not row["med_chem_ok"]:
            row["stage"] = "med_chem_valid"
            row["verdict"] = False
            return row
    if fiber.support == LEGACY_SCREENED:
        raise ValueError("legacy_screened support is not instrumented by this audit")
    row["stage"] = "eligible"
    row["verdict"] = True
    return row


# ---- Measurement-only instrumentation ----


class FunnelRecorder:
    """Counts and per-endpoint rows for one (cell, lane) pass."""

    def __init__(self, fiber: Fiber, context: SourceContext, lane: str):
        self.fiber = fiber
        self.context = context
        self.lane = lane
        self.counts: dict[str, int] = {}
        self.rows: dict[str, dict] = {}
        self.disagreements: list[dict] = []
        self.goal_context: dict | None = None

    def bump(self, key: str, amount: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + amount

    def observe(self, smiles: str, verdict: bool) -> None:
        self.bump("gate_calls")
        decomposition = gate_decomposition(self.fiber, smiles)
        if decomposition["verdict"] != verdict:
            self.disagreements.append({"smiles": smiles, "recomputed": decomposition["verdict"]})
        self.bump(f"gate_stage_{decomposition['stage']}")
        key = smiles
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is not None and not decomposition["fragmented"]:
            key = Chem.MolToSmiles(molecule)
        existing = self.rows.get(key)
        if existing is not None:
            existing["occurrences"] += 1
            return
        row = {
            "lane": self.lane,
            "smiles": key,
            "occurrences": 1,
            "stage": decomposition["stage"],
            "eligible": decomposition["verdict"],
        }
        for field in (
            "heavy",
            "similarity",
            "qed",
            "sa",
            "delta_sim",
            "delta_qed",
            "delta_sa",
            "capacity_ok",
            "sim_ok",
            "qed_ok",
            "sa_ok",
            "med_chem_ok",
        ):
            if field in decomposition:
                value = decomposition[field]
                row[field] = round(value, 5) if isinstance(value, float) else value
        if molecule is not None and not decomposition["fragmented"]:
            row.update(self.context.describe(molecule))
        if self.goal_context is not None:
            row.update(self.goal_context)
        self.rows[key] = row


def _goal_attributes(goal) -> dict:
    subgoals = tuple(goal.subgoals)
    created = sum(len(sub.output_atoms) for sub in subgoals)
    deleted = sum(1 for sub in subgoals for atom in sub.target_atoms if atom is None)
    retained = sum(1 for sub in subgoals for atom in sub.target_atoms if atom is not None)
    return {
        "goal_regions": len(subgoals),
        "goal_created_atoms": created,
        "goal_deleted_atoms": deleted,
        "goal_retained_atoms": retained,
        "goal_program_scale": created + deleted,
    }


@contextlib.contextmanager
def instrumented(fiber: Fiber, recorder: FunnelRecorder):
    """Wrap the module globals ``expand`` resolves, in this process only."""

    original_check = Fiber.check
    original_bindings = campaign.attachment_bindings
    original_instantiate = campaign.instantiate_goal
    original_to_smiles = campaign.molecular_graph_to_smiles
    original_extract = campaign.extract_structural_goal
    original_dynamic = campaign.synthesize_dynamic_program
    original_anchored = campaign.synthesize_anchored_replacement_program

    def check(self, smiles):
        result = original_check(self, smiles)
        if self is fiber:
            recorder.observe(smiles, result is not None)
        return result

    def attachment_bindings(subgoal, source):
        census = original_bindings(subgoal, source)
        recorder.bump("binding_queries")
        if not census.assignments:
            recorder.bump("binding_unbindable")
        return census

    def instantiate_goal(source, goal, bindings):
        recorder.bump("programs_bound")
        recorder.goal_context = _goal_attributes(goal)
        try:
            return original_instantiate(source, goal, bindings)
        except (ValueError, KeyError, IndexError, TypeError):
            recorder.bump("instantiation_refused")
            raise

    def molecular_graph_to_smiles(graph):
        try:
            result = original_to_smiles(graph)
        except (ValueError, KeyError, IndexError, TypeError):
            recorder.bump("endpoint_decode_refused")
            raise
        recorder.bump("programs_executed")
        return result

    def extract_structural_goal(states, actions):
        try:
            result = original_extract(states, actions)
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            recorder.bump("goal_extraction_refused")
            raise
        recorder.bump("goals_extracted")
        return result

    def synthesize_dynamic_program(source, rng, **kwargs):
        recorder.bump("synthesis_attempts")
        try:
            result = original_dynamic(source, rng, **kwargs)
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            recorder.bump("synthesis_refused")
            raise
        recorder.bump("synthesis_succeeded")
        return result

    def synthesize_anchored_replacement_program(source, rng, **kwargs):
        recorder.bump("synthesis_attempts")
        try:
            result = original_anchored(source, rng, **kwargs)
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            recorder.bump("synthesis_refused")
            raise
        recorder.bump("synthesis_succeeded")
        return result

    Fiber.check = check
    campaign.attachment_bindings = attachment_bindings
    campaign.instantiate_goal = instantiate_goal
    campaign.molecular_graph_to_smiles = molecular_graph_to_smiles
    campaign.extract_structural_goal = extract_structural_goal
    campaign.synthesize_dynamic_program = synthesize_dynamic_program
    campaign.synthesize_anchored_replacement_program = synthesize_anchored_replacement_program
    try:
        yield
    finally:
        Fiber.check = original_check
        campaign.attachment_bindings = original_bindings
        campaign.instantiate_goal = original_instantiate
        campaign.molecular_graph_to_smiles = original_to_smiles
        campaign.extract_structural_goal = original_extract
        campaign.synthesize_dynamic_program = original_dynamic
        campaign.synthesize_anchored_replacement_program = original_anchored


# ---- One cell ----


def audit_cell(protein: str, cell_name: str, *, draw_scale: float = 1.0) -> dict:
    contract = load_contract(protein)
    runtime_inputs = verify_runtime_inputs(contract)
    cell = next(row for row in contract["cells"] if row["cell"] == cell_name)
    source_smiles = cell["smiles"]
    preflight = slot_semantics_preflight(source_smiles)

    fiber = Fiber(source_smiles, contract["delta"], support=contract["support"])
    context = SourceContext(fiber, source_smiles)
    started = time.time()

    lanes: dict[str, dict] = {}
    pools: dict[str, list[dict]] = {expert: [] for expert in EXPERTS}
    all_rows: list[dict] = []
    disagreements: list[dict] = []

    for expert_index, expert in enumerate(EXPERTS):
        proposal_seed = int(
            cell["controller_seed"] + 1_000_003 * 1 + 10_007 * 0 + 101 * expert_index
        )
        recorder = FunnelRecorder(fiber, context, expert)
        lane_started = time.time()
        telemetry: dict = {}
        if expert in {"shallow", "anchored_replacement"}:
            settings = contract["proposal"][expert]
            draws = max(1, round(settings["draws"] * draw_scale))
            with instrumented(fiber, recorder):
                records = expand(
                    source_smiles,
                    0.0,
                    fiber,
                    np.random.default_rng(proposal_seed),
                    draws=draws,
                    multi_region=True,
                    horizon=settings["horizon"],
                    proposal_lane=expert,
                )
            telemetry = {"raw_draws": draws, "contract_draws": settings["draws"]}
        else:
            settings = contract["proposal"][expert]
            expert_model = load_expert(protein)
            padded = pad_molecular_graph(
                smiles_to_molecular_graph(source_smiles), T4_PROPOSAL_SLOTS
            )
            proposed, telemetry = propose_route_expert_candidates(
                padded,
                expert_model,
                pool_size=settings["pool_size"],
                realization_limit=settings["realization_limit"],
                beam_width=settings["beam_width"],
                expansion_width=settings["expansion_width"],
                max_bindings_per_template=settings["max_bindings_per_template"],
                maximum_expansions=settings["maximum_expansions"],
            )
            records = []
            with instrumented(fiber, recorder):
                for row in proposed:
                    recorder.goal_context = {
                        "route_prior_score": row.get("route_prior_score"),
                        "route_proposal_rank": row.get("route_proposal_rank"),
                        "realized_primitives": row.get("realized_primitives"),
                        "realized_primitive_band": row.get("realized_primitive_band"),
                        "rewrite_events": row.get("rewrite_events"),
                        "rewrite_scale": row.get("rewrite_scale"),
                        "goal_regions": row.get("regions"),
                        "goal_created_atoms": row.get("created"),
                        "goal_deleted_atoms": row.get("deleted"),
                        "goal_program_scale": (row.get("created") or 0)
                        + (row.get("deleted") or 0),
                    }
                    properties = fiber.check(row["smiles"])
                    if properties is None or properties["smiles"] == source_smiles:
                        continue
                    records.append(
                        {
                            **row,
                            **properties,
                            "parent": source_smiles,
                            "parent_score": 0.0,
                            "delta": contract["delta"],
                        }
                    )
            telemetry = {
                key: value
                for key, value in telemetry.items()
                if not isinstance(value, (dict, list)) or key.endswith("counts")
            }

        pools[expert] = records
        rows = sorted(recorder.rows.values(), key=lambda row: row["smiles"])
        all_rows.extend(rows)
        disagreements.extend(
            {**row, "lane": expert} for row in recorder.disagreements
        )
        lanes[expert] = {
            "proposal_seed": proposal_seed,
            "counts": dict(sorted(recorder.counts.items())),
            "telemetry": telemetry,
            "distinct_endpoints_reaching_fiber": len(rows),
            "eligible_records_returned": len(records),
            "elapsed_seconds": round(time.time() - lane_started, 2),
        }

    # ---- The production round-one decision, verbatim ----
    state = SearchState(archive={source_smiles: 0.0}, budget=contract["charged_calls_per_cell"] - 1)
    rng = np.random.default_rng(cell["controller_seed"])
    parents = state.parents(limit=contract["parents"], rng=rng, explore=contract["parent_explore"])
    merged = merge_expert_pools(pools)
    fresh = [row for row in merged if row["smiles"] not in state.archive]
    candidates = attach_features(fresh, state, fiber)
    room = min(contract["batch"], state.budget)
    selected = select_batch(
        candidates,
        ProgramValue(penalty=contract["value_penalty"]),
        state,
        rng,
        round_index=1,
        batch=room,
        exploration=min(contract["exploration"], room),
        expert_floor_rounds=contract["expert_floor_rounds"],
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "protein": protein,
        "cell": cell_name,
        "observed_status": OBSERVED_STATUS.get(cell_name),
        "source_smiles": source_smiles,
        "controller_seed": cell["controller_seed"],
        "contract_payload_sha256": identity(contract),
        "route_expert_training_identity": load_expert(protein).training_identity,
        "runtime_inputs_sha256": runtime_inputs,
        "slot_semantics_preflight": preflight,
        "source": {
            "heavy": context.heavy,
            "rings": context.rings,
            "qed": round(QED.qed(context.source_mol), 5),
            "sa": round(sascorer.calculateScore(context.source_mol), 5),
            "formal_charge": Chem.GetFormalCharge(context.source_mol),
            "headroom_heavy_atoms": REPRESENTABLE_HEAVY_ATOMS - context.heavy,
            "source_qed_margin": round(QED.qed(context.source_mol) - QED_MIN, 5),
            "source_sa_margin": round(SA_MAX - sascorer.calculateScore(context.source_mol), 5),
        },
        "parents_round_one": parents,
        "app_ignores_contract_scale_balanced": bool(
            contract["proposal"]["route_complete_region"].get("scale_balanced")
        ),
        "lanes": lanes,
        "round_one_decision": {
            "merged_pool": len(merged),
            "fresh": len(fresh),
            "featured_candidates": len(candidates),
            "selected": len(selected),
            "pool_census": expert_census(candidates),
            "selected_census": expert_census(selected),
            "reproduces_candidate_exhaustion": not selected,
        },
        "gate_reconstruction_disagreements": disagreements,
        "endpoints": all_rows,
        "draw_scale": draw_scale,
        "elapsed_seconds": round(time.time() - started, 2),
    }


# ---- Reduction ----


def _quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 5),
        "p10": round(float(np.percentile(array, 10)), 5),
        "median": round(float(np.median(array)), 5),
        "p90": round(float(np.percentile(array, 90)), 5),
        "max": round(float(array.max()), 5),
        "mean": round(float(array.mean()), 5),
    }


PAIRS = (
    ("braf", ("braf_0", "braf_1"), ("braf_2",)),
    ("fa7", ("fa7_0", "fa7_2"), ("fa7_1",)),
    ("5ht1b", ("5ht1b_2",), ("5ht1b_0", "5ht1b_1")),
    ("parp1", (), ("parp1_0", "parp1_1", "parp1_2")),
    ("jak2", (), ("jak2_0", "jak2_1", "jak2_2")),
)


def _histogram(values, edges) -> dict:
    counts = {}
    for low, high in itertools.pairwise(edges):
        counts[f"[{low:g},{high:g})"] = sum(1 for v in values if low <= v < high)
    counts[f"<{edges[0]:g}"] = sum(1 for v in values if v < edges[0])
    counts[f">={edges[-1]:g}"] = sum(1 for v in values if v >= edges[-1])
    return counts


def _pareto(rows: list[dict], limit: int = 20) -> list[dict]:
    """Non-dominated endpoints in (delta_sim, delta_qed); both are maximised."""

    front = []
    for row in rows:
        dominated = any(
            other["delta_sim"] >= row["delta_sim"]
            and other["delta_qed"] >= row["delta_qed"]
            and (
                other["delta_sim"] > row["delta_sim"]
                or other["delta_qed"] > row["delta_qed"]
            )
            for other in rows
        )
        if not dominated:
            front.append(row)
    front.sort(key=lambda row: -row["delta_qed"])
    return front[:limit]


def _compact(row: dict) -> dict:
    keep = (
        "lane",
        "smiles",
        "stage",
        "delta_sim",
        "delta_qed",
        "delta_sa",
        "heavy_delta",
        "ring_delta",
        "retained_source_bit_fraction",
        "direction",
        "goal_program_scale",
        "goal_regions",
        "scaffold_preserved",
        "realized_primitive_band",
        "route_proposal_rank",
    )
    return {key: row[key] for key in keep if key in row}


def _qed_properties(smiles: str) -> dict:
    """The eight descriptors QED is a weighted geometric mean of desirabilities over."""

    properties = QED.properties(Chem.MolFromSmiles(smiles))
    return {
        "MW": round(properties.MW, 2),
        "ALOGP": round(properties.ALOGP, 3),
        "HBA": properties.HBA,
        "HBD": properties.HBD,
        "PSA": round(properties.PSA, 2),
        "ROTB": properties.ROTB,
        "AROM": properties.AROM,
        "ALERTS": properties.ALERTS,
    }


def _relaxation_probe(reached: list[dict]) -> dict:
    """How many endpoints would pass if exactly ONE threshold were loosened.

    This separates "barely missing a boundary" from "genuinely absent support".  The
    med-chem structural gate is NOT applied here, because ``gate_decomposition`` short
    circuits before it for anything that already failed a property threshold; measured on
    a control cell it refuses 3 of 3,496 endpoints, so its omission cannot carry a
    conclusion.
    """

    def count(**tolerance) -> int:
        sim = tolerance.get("sim", 0.0)
        qed = tolerance.get("qed", 0.0)
        sa = tolerance.get("sa", 0.0)
        return sum(
            1
            for row in reached
            if row["delta_sim"] >= -sim and row["delta_qed"] >= -qed and row["delta_sa"] >= -sa
        )

    return {
        "no_relaxation": count(),
        "similarity_minus_0.05": count(sim=0.05),
        "similarity_minus_0.10": count(sim=0.10),
        "similarity_minus_0.20": count(sim=0.20),
        "qed_minus_0.05": count(qed=0.05),
        "qed_minus_0.10": count(qed=0.10),
        "qed_minus_0.20": count(qed=0.20),
        "sa_plus_0.5": count(sa=0.5),
        "sa_plus_1.0": count(sa=1.0),
        "all_thresholds_off_by_0.10_and_1.0": count(sim=0.10, qed=0.10, sa=1.0),
    }


def summarize(shard: dict) -> dict:
    """Per-cell funnel, margins, conditional feasibility and program attributes."""

    rows = shard["endpoints"]
    stages = (
        "chemically_valid",
        "capacity_valid",
        "similarity",
        "qed",
        "sa",
        "med_chem_valid",
        "eligible",
    )
    funnel = {stage: sum(1 for row in rows if row["stage"] == stage) for stage in stages}
    reached = [row for row in rows if "delta_sim" in row]
    sim_pass = [row for row in reached if row["sim_ok"]]
    qed_pass = [row for row in reached if row["qed_ok"]]
    sa_pass = [row for row in reached if row["sa_ok"]]
    sim_qed = [row for row in sim_pass if row["qed_ok"]]
    eligible_rows = [row for row in rows if row["eligible"]]

    return {
        "cell": shard["cell"],
        "observed_status": shard["observed_status"],
        "source": shard["source"],
        "source_qed_properties": _qed_properties(shard["source_smiles"]),
        "best_qed_endpoint_properties": (
            _qed_properties(max(reached, key=lambda row: row["delta_qed"])["smiles"])
            if reached
            else None
        ),
        "funnel": {
            "proposal_draws": sum(
                info["counts"].get("synthesis_attempts", 0)
                for info in shard["lanes"].values()
            ),
            "programs_bound": sum(
                info["counts"].get("programs_bound", 0) for info in shard["lanes"].values()
            ),
            "programs_exactly_executed": sum(
                info["counts"].get("programs_executed", 0)
                for info in shard["lanes"].values()
            )
            + shard["lanes"]["route_complete_region"]["counts"].get("gate_calls", 0),
            "endpoint_gate_calls_with_duplicates": sum(
                info["counts"].get("gate_calls", 0) for info in shard["lanes"].values()
            ),
            "distinct_lane_endpoint_pairs": len(rows),
            "distinct_endpoints": len({row["smiles"] for row in rows}),
            "chemically_valid": len(rows) - funnel["chemically_valid"],
            "capacity_valid": len(reached),
            "similarity_pass": len(sim_pass),
            "qed_pass": len(qed_pass),
            "sa_pass": len(sa_pass),
            "similarity_and_qed_pass": len(sim_qed),
            "all_three_pass": sum(1 for row in sim_qed if row["sa_ok"]),
            "fully_eligible": len(eligible_rows),
            "first_failing_stage_census": funnel,
        },
        "round_one_decision": shard["round_one_decision"],
        "margins_all_reaching_gate": {
            "delta_sim": _quantiles([row["delta_sim"] for row in reached]),
            "delta_qed": _quantiles([row["delta_qed"] for row in reached]),
            "delta_sa": _quantiles([row["delta_sa"] for row in reached]),
        },
        "single_gate_relaxation": _relaxation_probe(reached),
        "conditional_margins": {
            "delta_qed_given_similarity_pass": _quantiles(
                [row["delta_qed"] for row in sim_pass]
            ),
            "delta_sim_given_qed_pass": _quantiles([row["delta_sim"] for row in qed_pass]),
            "delta_sa_given_sim_and_qed_pass": _quantiles(
                [row["delta_sa"] for row in sim_qed]
            ),
        },
        "histograms": {
            "delta_sim": _histogram(
                [row["delta_sim"] for row in reached], [-0.3, -0.2, -0.1, -0.05, 0.0, 0.1, 0.3]
            ),
            "delta_qed": _histogram(
                [row["delta_qed"] for row in reached], [-0.5, -0.3, -0.15, -0.05, 0.0, 0.1, 0.3]
            ),
            "delta_sa": _histogram(
                [row["delta_sa"] for row in reached], [-1.0, -0.5, -0.1, 0.0, 0.5, 1.0, 2.0]
            ),
        },
        "attributes": {
            "heavy_delta": _quantiles([row["heavy_delta"] for row in reached]),
            "ring_delta": _quantiles([row["ring_delta"] for row in reached]),
            "retained_source_bit_fraction": _quantiles(
                [row["retained_source_bit_fraction"] for row in reached]
            ),
            "goal_program_scale": _quantiles(
                [
                    row["goal_program_scale"]
                    for row in rows
                    if row.get("goal_program_scale") is not None
                ]
            ),
            "direction_census": {
                name: sum(1 for row in reached if row.get("direction") == name)
                for name in ("grow", "prune", "replace")
            },
            "direction_census_similarity_pass": {
                name: sum(1 for row in sim_pass if row.get("direction") == name)
                for name in ("grow", "prune", "replace")
            },
            "scaffold_preserved": sum(1 for row in reached if row.get("scaffold_preserved")),
            "fragmented_endpoints": funnel["chemically_valid"],
        },
        "per_lane": {
            lane: {
                "distinct_endpoints": info["distinct_endpoints_reaching_fiber"],
                "eligible": info["eligible_records_returned"],
                "counts": info["counts"],
                "elapsed_seconds": info["elapsed_seconds"],
            }
            for lane, info in shard["lanes"].items()
        },
        "pareto_front_sim_vs_qed": [_compact(row) for row in _pareto(reached)],
        "best_qed_among_similarity_passing": [
            _compact(row)
            for row in sorted(sim_pass, key=lambda row: -row["delta_qed"])[:10]
        ],
        "best_similarity_among_qed_passing": [
            _compact(row)
            for row in sorted(qed_pass, key=lambda row: -row["delta_sim"])[:10]
        ],
        "eligible_endpoints": [_compact(row) for row in eligible_rows[:25]],
        "gate_reconstruction_disagreements": len(shard["gate_reconstruction_disagreements"]),
        "slot_semantics_preflight": shard["slot_semantics_preflight"],
    }


def reduce_shards(shard_dir: Path, destination: Path, headroom: dict | None = None) -> dict:
    cells = {}
    provenance = {}
    for path in sorted(shard_dir.glob("*.json")):
        shard = json.loads(path.read_text())
        cells[shard["cell"]] = summarize(shard)
        provenance[shard["cell"]] = {
            "contract_payload_sha256": shard["contract_payload_sha256"],
            "route_expert_training_identity": shard["route_expert_training_identity"],
            "controller_seed": shard["controller_seed"],
            "source_smiles": shard["source_smiles"],
            "app_ignores_contract_scale_balanced": shard.get(
                "app_ignores_contract_scale_balanced"
            ),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "notes": {
            "parent_score": (
                "The docked root score is unavailable offline, so every reproduction sets "
                "parent_score = 0.0. It provably cannot change the round-one candidate or "
                "selected count: Fiber.check never sees it, ProgramValue is unfitted at "
                "round one (weights is None) so acquisition takes no model picks, and both "
                "the expert floor and the exploration quota choose uniformly at random."
            ),
            "scale_balanced": (
                "The fa7 and 5ht1b contracts declare proposal.route_complete_region."
                "scale_balanced = true, but the pinned app never forwards it to "
                "propose_route_expert_candidates, so the live runs and this audit both use "
                "the default scale_balanced = False."
            ),
            "instrumentation": (
                "Fiber.check's original method decides accept/reject; the per-gate "
                "decomposition is recomputed independently and every disagreement is "
                "counted. No production file was modified."
            ),
        },
        "gate": {
            "delta": 0.6,
            "qed_min": QED_MIN,
            "sa_max": SA_MAX,
            "representable_heavy_atoms": REPRESENTABLE_HEAVY_ATOMS,
        },
        "provenance": provenance,
        "cells": cells,
    }
    if headroom is not None:
        payload["fiber_geometry_control"] = headroom
    payload["fail_versus_control"] = _fail_versus_control(cells, headroom)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return payload


def _fail_versus_control(cells: dict, headroom: dict | None) -> dict:
    """Side-by-side of each failing cell against its same-protein sibling controls."""

    def view(name: str) -> dict | None:
        row = cells.get(name)
        if row is None:
            return None
        conditional = row["conditional_margins"]
        control = (headroom or {}).get(name) or {}
        return {
            "cell": name,
            "observed_status": row["observed_status"],
            "source_heavy": row["source"]["heavy"],
            "source_qed": row["source"]["qed"],
            "source_sa": row["source"]["sa"],
            "heavy_headroom_to_representability_ceiling": row["source"][
                "headroom_heavy_atoms"
            ],
            "distinct_endpoints": row["funnel"]["distinct_endpoints"],
            "similarity_pass": row["funnel"]["similarity_pass"],
            "qed_pass": row["funnel"]["qed_pass"],
            "sa_pass": row["funnel"]["sa_pass"],
            "similarity_and_qed_pass": row["funnel"]["similarity_and_qed_pass"],
            "fully_eligible": row["funnel"]["fully_eligible"],
            "selected": row["round_one_decision"]["selected"],
            "best_delta_qed_among_similarity_passing": (
                conditional["delta_qed_given_similarity_pass"] or {}
            ).get("max"),
            "best_delta_sim_among_qed_passing": (
                conditional["delta_sim_given_qed_pass"] or {}
            ).get("max"),
            "best_delta_qed_overall": row["margins_all_reaching_gate"]["delta_qed"]["max"]
            if row["margins_all_reaching_gate"]["delta_qed"]
            else None,
            "direction_census": row["attributes"]["direction_census"],
            "heavy_delta_median": (row["attributes"]["heavy_delta"] or {}).get("median"),
            "retained_bit_fraction_median": (
                row["attributes"]["retained_source_bit_fraction"] or {}
            ).get("median"),
            "goal_program_scale_median": (
                row["attributes"]["goal_program_scale"] or {}
            ).get("median"),
            "generic_edit_control_fiber_non_empty": (
                control.get("generic_single_edit_beam") or {}
            ).get("fiber_proved_non_empty"),
            "generic_edit_control_best_qed": (
                (control.get("generic_single_edit_beam") or {}).get("best_qed_inside_fiber")
                or {}
            ).get("qed"),
            "truncation_control_fiber_non_empty": (
                control.get("acyclic_truncation_beam") or {}
            ).get("fiber_proved_non_empty"),
            "truncation_control_witness": (
                control.get("acyclic_truncation_beam") or {}
            ).get("eligible_witness"),
        }

    result = {}
    for protein, fails, controls in PAIRS:
        rows = {
            "fail": [view(name) for name in fails if view(name) is not None],
            "control": [view(name) for name in controls if view(name) is not None],
        }
        if rows["fail"] or rows["control"]:
            result[protein] = rows
    return result


# ---- Lane-independent fiber-geometry control ----

_SUBSTITUTIONS = (6, 7, 8, 9, 16, 17)
_ADDITIONS = (6, 7, 8, 9)


def _canonical(molecule) -> str | None:
    try:
        Chem.SanitizeMol(molecule)
    except (ValueError, RuntimeError):
        return None
    smiles = Chem.MolToSmiles(molecule)
    if not smiles or "." in smiles:
        return None
    return smiles


def generic_neighbours(smiles: str) -> list[str]:
    """Single generic medicinal-chemistry edits, independent of any COMPOSE program.

    This deliberately does NOT use the controller's program synthesiser.  Its purpose is
    to answer a different question: whether the delta=0.6 fiber around this source
    CONTAINS a QED>=0.6 molecule at all, regardless of which programs the prior likes.
    """

    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        return []
    out: set[str] = set()
    for atom in parent.GetAtoms():
        index = atom.GetIdx()
        if atom.GetDegree() == 1 and not atom.IsInRing():
            editable = Chem.RWMol(parent)
            editable.RemoveAtom(index)
            result = _canonical(editable)
            if result:
                out.add(result)
        for number in _SUBSTITUTIONS:
            if number == atom.GetAtomicNum():
                continue
            editable = Chem.RWMol(parent)
            target = editable.GetAtomWithIdx(index)
            target.SetAtomicNum(number)
            target.SetNumExplicitHs(0)
            target.SetNoImplicit(False)
            target.SetFormalCharge(0)
            result = _canonical(editable)
            if result:
                out.add(result)
        if atom.GetTotalNumHs() > 0:
            for number in _ADDITIONS:
                editable = Chem.RWMol(parent)
                added = editable.AddAtom(Chem.Atom(number))
                editable.AddBond(index, added, Chem.BondType.SINGLE)
                result = _canonical(editable)
                if result:
                    out.add(result)
    for bond in parent.GetBonds():
        if bond.GetIsAromatic() or bond.IsInRing():
            continue
        for order in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE):
            if bond.GetBondType() == order:
                continue
            editable = Chem.RWMol(parent)
            editable.GetBondBetweenAtoms(
                bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ).SetBondType(order)
            result = _canonical(editable)
            if result:
                out.add(result)
    out.discard(Chem.MolToSmiles(parent))
    return sorted(out)


def truncation_neighbours(smiles: str) -> list[str]:
    """Every fragment obtained by cutting one acyclic single bond, both sides kept.

    The winning programs in this audit are large PRUNES (the eligible braf_2 endpoints
    delete 13-14 heavy atoms), and a single-atom-edit beam cannot reach them at any
    practical depth.  This enumerates the move class that actually wins, independent of
    any COMPOSE program or learned prior.
    """

    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        return []
    out: set[str] = set()
    for bond in parent.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        editable = Chem.RWMol(parent)
        editable.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        try:
            pieces = Chem.GetMolFrags(editable.GetMol(), asMols=True, sanitizeFrags=True)
        except (ValueError, RuntimeError):
            continue
        for piece in pieces:
            if piece.GetNumHeavyAtoms() < 5:
                continue
            result = _canonical(Chem.RWMol(piece))
            if result:
                out.add(result)
    out.discard(Chem.MolToSmiles(parent))
    return sorted(out)


def fiber_truncation_control(
    smiles: str, delta: float, *, depth: int = 3, beam: int = 60
) -> dict:
    """Beam over acyclic-bond truncations: does a large PRUNE land inside the fiber?"""

    fiber = Fiber(smiles, delta, support="compose_valid")
    frontier = [smiles]
    seen = {smiles}
    best_row = None
    found = None
    visited = 0
    for level in range(1, depth + 1):
        scored = []
        for parent in frontier:
            for candidate in truncation_neighbours(parent):
                if candidate in seen:
                    continue
                seen.add(candidate)
                visited += 1
                molecule = Chem.MolFromSmiles(candidate)
                if molecule is None:
                    continue
                similarity = DataStructs.TanimotoSimilarity(
                    fiber.seed, fiber.generator.GetFingerprint(molecule)
                )
                quality = QED.qed(molecule)
                access = sascorer.calculateScore(molecule)
                scored.append((quality, candidate, similarity, level))
                if similarity >= delta and (
                    best_row is None or quality > best_row["qed"]
                ):
                    best_row = {
                        "smiles": candidate,
                        "similarity": round(similarity, 5),
                        "qed": round(quality, 5),
                        "sa": round(access, 5),
                        "heavy_delta": molecule.GetNumHeavyAtoms()
                        - Chem.MolFromSmiles(smiles).GetNumHeavyAtoms(),
                        "depth": level,
                    }
                if (
                    found is None
                    and similarity >= delta
                    and quality >= QED_MIN
                    and access <= SA_MAX
                    and structurally_valid(candidate)
                ):
                    found = {
                        "smiles": candidate,
                        "similarity": round(similarity, 5),
                        "qed": round(quality, 5),
                        "sa": round(access, 5),
                        "heavy_delta": molecule.GetNumHeavyAtoms()
                        - Chem.MolFromSmiles(smiles).GetNumHeavyAtoms(),
                        "depth": level,
                    }
        if not scored:
            break
        scored.sort(key=lambda row: (-(row[2] >= delta), -row[0]))
        frontier = [row[1] for row in scored[:beam]]
    return {
        "states_visited": visited,
        "best_qed_inside_fiber": best_row,
        "eligible_witness": found,
        "fiber_proved_non_empty": found is not None,
    }


def fiber_geometry_control(
    smiles: str,
    delta: float,
    *,
    depth: int = 5,
    beam: int = 40,
    slack: float = 0.15,
) -> dict:
    """Greedy beam over generic edits: a LOWER BOUND on QED reachable inside the fiber.

    Reported numbers are existence evidence in ONE direction only.  Reaching QED>=0.6 at
    similarity>=delta proves the delta-fiber around this source is non-empty; failing to
    reach it proves nothing about emptiness, only that this bounded search did not find
    a witness.  Treat a negative as uninformative -- it was measured to be negative on a
    source where the production controller does find eligible candidates.

    The beam ranks in-fiber states strictly above out-of-fiber ones and only then by QED.
    A purely QED-greedy ranking was measured to walk the whole beam out of the similarity
    ball at the first level (QED rises by shrinking the molecule) and never return.
    """

    fiber = Fiber(smiles, delta, support="compose_valid")
    frontier = [smiles]
    seen = {smiles}
    best_qed = -1.0
    best_row = None
    found = None
    visited = 0
    for level in range(1, depth + 1):
        scored = []
        for parent in frontier:
            for candidate in generic_neighbours(parent):
                if candidate in seen:
                    continue
                seen.add(candidate)
                visited += 1
                molecule = Chem.MolFromSmiles(candidate)
                if molecule is None or molecule.GetNumHeavyAtoms() > REPRESENTABLE_HEAVY_ATOMS:
                    continue
                similarity = DataStructs.TanimotoSimilarity(
                    fiber.seed, fiber.generator.GetFingerprint(molecule)
                )
                if similarity < delta - slack:
                    continue
                quality = QED.qed(molecule)
                access = sascorer.calculateScore(molecule)
                scored.append((quality, candidate, similarity, access, level))
                if similarity >= delta and quality > best_qed:
                    best_qed = quality
                    best_row = {
                        "smiles": candidate,
                        "similarity": round(similarity, 5),
                        "qed": round(quality, 5),
                        "sa": round(access, 5),
                        "depth": level,
                    }
                if (
                    found is None
                    and similarity >= delta
                    and quality >= QED_MIN
                    and access <= SA_MAX
                    and structurally_valid(candidate)
                ):
                    found = {
                        "smiles": candidate,
                        "similarity": round(similarity, 5),
                        "qed": round(quality, 5),
                        "sa": round(access, 5),
                        "depth": level,
                    }
        if not scored:
            break
        scored.sort(key=lambda row: (-(row[2] >= delta), -row[0]))
        frontier = [row[1] for row in scored[:beam]]
    return {
        "smiles": smiles,
        "delta": delta,
        "depth": depth,
        "beam": beam,
        "slack": slack,
        "states_visited": visited,
        "best_qed_inside_fiber": best_row,
        "eligible_witness": found,
        "fiber_proved_non_empty": found is not None,
    }



def report(payload: dict) -> str:
    """Human-readable funnel and FAIL-vs-CONTROL tables from a reduced payload."""

    cells = payload["cells"]
    lines = ["FUNNEL (distinct endpoints, round one, all three experts)", ""]
    header = (
        f"{'cell':9s} {'st':4s} {'hvy':>3s} {'srcQED':>6s} {'srcSA':>5s} "
        f"{'prop':>6s} {'chem':>5s} {'cap':>5s} {'sim':>5s} {'qed':>5s} {'sa':>5s} "
        f"{'s&q':>4s} {'all3':>4s} {'elig':>4s} {'sel':>3s}"
    )
    lines.append(header)
    for protein, fails, controls in PAIRS:
        for cell_name in (*fails, *controls):
            row = cells.get(cell_name)
            if row is None:
                continue
            funnel = row["funnel"]
            lines.append(
                f"{cell_name:9s} "
                f"{'FAIL' if cell_name in fails else 'ctrl':4s} "
                f"{row['source']['heavy']:3d} "
                f"{row['source']['qed']:6.3f} {row['source']['sa']:5.2f} "
                f"{funnel['distinct_endpoints']:6d} {funnel['chemically_valid']:5d} "
                f"{funnel['capacity_valid']:5d} {funnel['similarity_pass']:5d} "
                f"{funnel['qed_pass']:5d} {funnel['sa_pass']:5d} "
                f"{funnel['similarity_and_qed_pass']:4d} {funnel['all_three_pass']:4d} "
                f"{funnel['fully_eligible']:4d} "
                f"{row['round_one_decision']['selected']:3d}"
            )
        lines.append("")
    lines.append("MARGINS (all endpoints reaching the property gates)")
    lines.append(
        f"{'cell':9s} {'st':4s} {'dSIM max':>8s} {'dSIM med':>8s} "
        f"{'dQED max':>8s} {'dQED med':>8s} {'dSA max':>8s} "
        f"{'dQED|sim max':>12s} {'dSIM|qed max':>12s}"
    )
    for protein, fails, controls in PAIRS:
        for cell_name in (*fails, *controls):
            row = cells.get(cell_name)
            if row is None:
                continue
            margins = row["margins_all_reaching_gate"]
            conditional = row["conditional_margins"]

            def value(block, key):
                return "     n/a" if block is None else f"{block[key]:8.4f}"

            given_qed = conditional["delta_qed_given_similarity_pass"]
            given_sim = conditional["delta_sim_given_qed_pass"]
            lines.append(
                f"{cell_name:9s} {'FAIL' if cell_name in fails else 'ctrl':4s} "
                f"{value(margins['delta_sim'], 'max')} {value(margins['delta_sim'], 'median')} "
                f"{value(margins['delta_qed'], 'max')} {value(margins['delta_qed'], 'median')} "
                f"{value(margins['delta_sa'], 'max')} "
                f"{value(given_qed, 'max'):>12s} {value(given_sim, 'max'):>12s}"
            )
    lines.append("")
    lines.append("SINGLE-GATE RELAXATION (endpoints passing all three thresholds)")
    keys = (
        "no_relaxation",
        "similarity_minus_0.05",
        "similarity_minus_0.10",
        "similarity_minus_0.20",
        "qed_minus_0.05",
        "qed_minus_0.10",
        "qed_minus_0.20",
        "sa_plus_0.5",
        "sa_plus_1.0",
    )
    labels = ("none", "sim-.05", "sim-.10", "sim-.20", "qed-.05", "qed-.10",
              "qed-.20", "sa+0.5", "sa+1.0")
    lines.append(
        f"{'cell':9s} {'st':4s} " + " ".join(f"{label:>8s}" for label in labels)
    )
    for protein, fails, controls in PAIRS:
        for cell_name in (*fails, *controls):
            row = cells.get(cell_name)
            if row is None:
                continue
            probe = row["single_gate_relaxation"]
            lines.append(
                f"{cell_name:9s} {'FAIL' if cell_name in fails else 'ctrl':4s} "
                + " ".join(f"{probe[key]:8d}" for key in keys)
            )
    lines.append("")
    lines.append("LANE ATTRIBUTION (eligible endpoints per proposal expert)")
    lines.append(
        f"{'cell':9s} {'st':4s} {'shallow':>9s} {'anchored':>9s} {'route':>9s} "
        f"{'grow':>6s} {'prune':>6s} {'repl':>6s} {'hvyDmed':>8s} {'retmed':>7s}"
    )
    for protein, fails, controls in PAIRS:
        for cell_name in (*fails, *controls):
            row = cells.get(cell_name)
            if row is None:
                continue
            per_lane = row["per_lane"]
            direction = row["attributes"]["direction_census"]
            heavy = row["attributes"]["heavy_delta"] or {}
            retained = row["attributes"]["retained_source_bit_fraction"] or {}
            lines.append(
                f"{cell_name:9s} {'FAIL' if cell_name in fails else 'ctrl':4s} "
                f"{per_lane['shallow']['eligible']:9d} "
                f"{per_lane['anchored_replacement']['eligible']:9d} "
                f"{per_lane['route_complete_region']['eligible']:9d} "
                f"{direction['grow']:6d} {direction['prune']:6d} {direction['replace']:6d} "
                f"{heavy.get('median', float('nan')):8.1f} "
                f"{retained.get('median', float('nan')):7.3f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein")
    parser.add_argument("--cell")
    parser.add_argument("--draw-scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reduce", action="store_true")
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--headroom", type=Path)
    parser.add_argument("--headroom-only", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT / "diagnostics/t4_support_stage_audit_v1.json",
    )
    args = parser.parse_args()

    if args.report:
        print(report(json.loads(args.destination.read_text())))
        return

    if args.headroom_only:
        control = {}
        for protein in ("braf", "fa7", "5ht1b", "parp1", "jak2"):
            contract = load_contract(protein)
            for cell in contract["cells"]:
                control[cell["cell"]] = {
                    "smiles": cell["smiles"],
                    "delta": contract["delta"],
                    "generic_single_edit_beam": fiber_geometry_control(
                        cell["smiles"], contract["delta"]
                    ),
                    "acyclic_truncation_beam": fiber_truncation_control(
                        cell["smiles"], contract["delta"]
                    ),
                }
                print(
                    json.dumps(
                        {
                            "cell": cell["cell"],
                            "generic_edit_witness": control[cell["cell"]][
                                "generic_single_edit_beam"
                            ]["fiber_proved_non_empty"],
                            "truncation_witness": control[cell["cell"]][
                                "acyclic_truncation_beam"
                            ]["fiber_proved_non_empty"],
                        }
                    ),
                    flush=True,
                )
        destination = args.headroom or (ROOT / "diagnostics/t4_fiber_geometry_control.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(control, indent=1, sort_keys=True) + "\n")
        return

    if args.reduce:
        control = json.loads(args.headroom.read_text()) if args.headroom else None
        payload = reduce_shards(args.shard_dir, args.destination, headroom=control)
        print(json.dumps({"cells": sorted(payload["cells"])}, indent=1))
        return

    shard = audit_cell(args.protein, args.cell, draw_scale=args.draw_scale)
    destination = args.out or (ROOT / f"scratch_{args.cell}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(shard, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "cell": shard["cell"],
                "selected": shard["round_one_decision"]["selected"],
                "eligible_by_lane": {
                    lane: info["eligible_records_returned"]
                    for lane, info in shard["lanes"].items()
                },
                "disagreements": len(shard["gate_reconstruction_disagreements"]),
                "seconds": shard["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
