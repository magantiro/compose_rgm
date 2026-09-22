"""Matched-arm harness for the declared-macro-option protection mechanism.

ZERO charged benchmark oracle calls. The scorer is `valley_similarity_v1`, a declared
synthetic function of the molecule alone. It is not a PMO oracle, it carries no
benchmark answer, and no number produced under it is a PMO result.

The landscape, and why it is shaped this way
-------------------------------------------
The MEASURED motivation is that 6 of 11 teacher routes and 17 of 44 planned transports
pass through a midpoint scoring BELOW their own source, at a median relative depth of
0.659. The declared cause is prune-then-install: the path passes through a molecule
smaller than both endpoints. `valley_similarity_v1` reproduces exactly that, and only
that -- a molecule at least `VALLEY_DROP` heavy atoms below the run's own starting
molecule is penalised by `VALLEY_DEPTH`.

The valley is declared STRUCTURALLY and independently of any option, so whether the
declared bridges actually fall into it is a MEASUREMENT, not a construction. The
harness reports `bridges_in_valley` and the predeclaration VOIDS the run if it is zero:
a landscape the mechanism is never tested on cannot reject or support it.

What this harness can and cannot establish
------------------------------------------
It can REJECT the mechanism, and it can show the mechanism is not inert. It CANNOT show
that macro options improve PMO, because the landscape was built to require them. That
scoping is in the predeclaration and is repeated here so a reader of either file sees it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_macro_option_controller import MacroOptionController
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask

RDLogger.DisableLog("rdApp.*")

SCHEMA = "pmo_macro_option_arms_v1"

# Declared in diagnostics/pmo_macro_option_v1/falsifier_predeclaration_v1.json BEFORE any
# measurement. Changing either number invalidates that predeclaration.
VALLEY_DEPTH = 0.66
VALLEY_DROP = 2

# A fixed synthetic reference. It is NOT a benchmark answer: no PMO task declares it, and
# it is never used to select anything except through the counted synthetic score.
SYNTHETIC_REFERENCE = "O=C(Nc1ccc(F)cc1)C1CCN(Cc2ccccc2)CC1"

INITIALIZATION = "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
JUMP_CHECKPOINT = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else _GENERATOR.GetFingerprint(molecule)


class ValleySimilarityScorer:
    """Similarity to a fixed synthetic reference, with a declared structural valley.

    The valley predicate reads ONLY the molecule's heavy-atom count against the run's
    starting size, so it cannot be tuned per option and does not know an option exists.
    """

    protocol = "synthetic:valley_similarity_v1"

    def __init__(self, *, origin_heavy_atoms: int, depth=VALLEY_DEPTH, drop=VALLEY_DROP):
        if not 0.0 <= depth < 1.0 or drop < 1:
            raise ValueError("valley depth must be in [0, 1) and the drop at least one atom")
        self.origin_heavy_atoms = int(origin_heavy_atoms)
        self.depth = float(depth)
        self.drop = int(drop)
        self.reference = _fingerprint(SYNTHETIC_REFERENCE)
        self.calls = 0
        self.in_valley = 0

    def in_the_valley(self, smiles: str) -> bool:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        return molecule.GetNumHeavyAtoms() <= self.origin_heavy_atoms - self.drop

    def __call__(self, smiles: str) -> float:
        self.calls += 1
        fingerprint = _fingerprint(smiles)
        if fingerprint is None:
            return 0.0
        base = float(DataStructs.TanimotoSimilarity(self.reference, fingerprint))
        if self.in_the_valley(smiles):
            self.in_valley += 1
            return base * (1.0 - self.depth)
        return base

    def payload(self) -> dict[str, Any]:
        return {
            "name": "valley_similarity_v1",
            "reference": SYNTHETIC_REFERENCE,
            "origin_heavy_atoms": self.origin_heavy_atoms,
            "valley_depth": self.depth,
            "valley_drop": self.drop,
            "benchmark_oracle": False,
        }


def window_decomposition(folder: Path) -> dict[str, Any]:
    """Why each declared option ended where it did, from the rounds already published.

    The outcome metric -- destination reached -- cannot separate a floor that was applied
    and still lost from a floor that was never large enough to matter. The mechanism has
    three consecutive requirements per crossing and each fails alone:

      1. the window OPENS    -- the bridge is charged, so `note_charged` opens a window
      2. the bridge is DRAWN -- the next leg is offered only from a parent the round's
                                schedule drew, which is what makes the floor load-bearing
      3. the leg is CHARGED  -- the reserved slot survives into the locked batch

    Computed here rather than in a post-hoc reducer because the matched-arm driver deletes
    each seed's campaign directory once its artifact lands, so a later pass has nothing to
    read. Zero additional chemistry and zero oracle calls.
    """
    from compose_v4.control.pmo_macro_option_controller import MACRO_OPTION_CHANNEL_TAG

    rounds = sorted((folder / "campaign").glob("round_*/complete.json"))
    if not rounds:
        return {}
    offered: dict[str, int] = {}
    locked: dict[str, int] = {}
    reservation = {"reserved_added": 0, "reserved_already_chosen": 0, "displaced": 0}
    for path in sorted((folder / "campaign").glob("round_*/pending.json")):
        batch = json.loads(path.read_text())["batch"]
        # `proposal_pool` is the full generated pool; `candidates` is the LOCKED subset
        # the ledger charges. Offered-but-not-charged is a different failure from never
        # offered at all, so both are counted.
        for row in batch.get("proposal_pool", batch)["candidates"]:
            if row["provenance"].get("entry_channel") == MACRO_OPTION_CHANNEL_TAG:
                key = row["provenance"]["macro_option_id"]
                offered[key] = offered.get(key, 0) + 1
        for row in batch["candidates"]:
            if row["provenance"].get("entry_channel") == MACRO_OPTION_CHANNEL_TAG:
                key = row["provenance"]["macro_option_id"]
                locked[key] = locked.get(key, 0) + 1
        detail = (batch.get("allocation") or {}).get("macro_option_reservation") or {}
        for key in reservation:
            reservation[key] += int(detail.get(key, 0) or 0)

    final = json.loads(rounds[-1].read_text())["snapshot"]["macro_options"]
    options = {}
    for record in final["registry"]["records"]:
        option_id = record["option"]["option_id"]
        status, frontier = record["status"], record["frontier"]
        if status == "reached":
            failure = None
        elif status == "declared":
            failure = "bridge_never_charged"
        elif offered.get(option_id, 0) > locked.get(option_id, 0):
            failure = "offered_but_not_charged"
        else:
            # The window opened and the next leg was never offered, which means the
            # bridge was not in the round's drawn parent schedule: the floor was applied
            # and was not enough to get it drawn.
            failure = "bridge_not_drawn_in_window"
        options[option_id] = {
            "status": status,
            "legs_charged": frontier + 1,
            "legs_required": len(record["option"]["stages"]),
            "legs_offered": offered.get(option_id, 0),
            "legs_locked": locked.get(option_id, 0),
            "failure": failure,
        }
    return {"options": options, "reservation": reservation}


def configuration(seed: int) -> ProgramSearchConfig:
    """The PMO-v1 search geometry, with only the seed varying across replicates."""
    return replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        parent_allocation="niche_evidence",
        attempts_per_batch=48,
        candidates_per_batch=8,
        wall_seconds=20.0,
        proposal_cache_entries=128,
    )


def assert_production_batch_geometry(config: ProgramSearchConfig, queries_per_round: int) -> None:
    """Refuse any geometry in which `_allocate` is not the binding discard.

    MEASURED, and the reason this is a guard rather than a convention. Production PMO
    sets `candidates_per_batch = QUERIES_PER_ROUND` (pmo_population_v1.py:57, :226), so
    `prepare_query_batch` takes its `len(candidates) <= count` branch, `lock_query_subset`
    discards ZERO, and the controller's `_allocate` is the only gate that drops a
    candidate before it is charged. That is where the macro-option reservation sits.

    Give the campaign fewer queries than the controller allocates and a SECOND discard
    appears -- `select_parent_edits` in parent_edit_search.py picks `queries_per_round` of
    the allocated pool, and the reservation does not reach it. MEASURED on this harness at
    8 allocated against 4 charged: the reserved stage-0 leg was generated, reserved and
    chosen by `_allocate`, and then silently dropped, so no bridge was ever charged and no
    protection window ever opened across a whole campaign. Nothing raised; the run simply
    measured a mechanism that never fired.

    That is a different system from production, so the harness refuses it outright rather
    than reporting a null obtained under it.
    """
    if int(queries_per_round) != int(config.candidates_per_batch):
        raise ValueError(
            "matched-arm geometry must charge exactly what the controller allocates: "
            f"queries_per_round={queries_per_round} against "
            f"candidates_per_batch={config.candidates_per_batch}. Production PMO sets "
            "them equal, and when they differ `prepare_query_batch` adds a second "
            "discard the macro-option reservation does not reach -- the mechanism under "
            "test would never bind and the measurement would be of a different system"
        )


def load_initialization(root: Path, *, count: int, seed: int) -> dict[str, Any]:
    """A bounded slice of the frozen task-independent initialization bank."""
    stored = json.loads((root / INITIALIZATION).read_text())
    rows = stored["candidates"][:count]
    body = {
        "candidates": rows,
        "count": len(rows),
        "task_independent": True,
        "provenance": f"{INITIALIZATION}#first_{count}",
        "seed": seed,
    }
    return {**body, "lock_sha256": identity(body)}


def load_jump_checkpoint(root: Path) -> dict[str, Any]:
    envelope = json.loads((root / JUMP_CHECKPOINT).read_text())
    return envelope["payload"]["checkpoints"]["shared_all_routes"]


def run_arm(
    *,
    root: Path,
    output: Path,
    seed: int,
    protection: bool,
    budget: int,
    rounds: int,
    queries_per_round: int,
    initialization_count: int = 4,
    macro_option_settings: dict[str, Any] | None = None,
    enable_macro_options: bool = True,
) -> dict[str, Any]:
    """One matched arm. Charges only the synthetic scorer; no benchmark oracle exists."""
    assert_production_batch_geometry(configuration(seed), queries_per_round)
    initialized = load_initialization(root, count=initialization_count, seed=seed)
    origin = Chem.MolFromSmiles(initialized["candidates"][0]["endpoint"])
    scorer = ValleySimilarityScorer(origin_heavy_atoms=origin.GetNumHeavyAtoms())
    task = ProgramTask("synthetic_valley_similarity", scorer.protocol, "pmo")
    ledger = ProgramQueryLedger(output / "oracle", task, scorer, budget=budget)
    optimizer_kwargs = {
        "jump_checkpoint": load_jump_checkpoint(root),
        "enable_online_memory": False,
        # The arm selector rides in optimizer_kwargs, so `run_program_campaign` folds it
        # into `optimizer_kwargs_sha256` and the two arms cannot share a run identity.
        "enable_macro_options": bool(enable_macro_options),
        "macro_option_protection": bool(protection),
        "macro_option_settings": dict(macro_option_settings or {}),
    }
    campaign = run_program_campaign(
        output=output / "campaign",
        task=task,
        config=configuration(seed),
        initialization=initialized,
        library=(),
        ledger=ledger,
        rounds=rounds,
        queries_per_round=queries_per_round,
        hierarchy=None,
        fit_model=None,
        stagnation_rounds=None,
        bootstrap_rounds=1,
        initialization_mode="all_scored_pool",
        initial_parent_fraction=0.2,
        optimizer_type=MacroOptionController,
        optimizer_kwargs=optimizer_kwargs,
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )
    report = campaign["snapshot"]["macro_options"]
    registry = report["registry"]
    charged = {row["endpoint"]: float(row["score"]) for row in ledger.rows}
    options = []
    for record in registry["records"]:
        option = record["option"]
        bridges = [stage["endpoint"] for stage in option["stages"][:-1]]
        options.append(
            {
                "option_id": option["option_id"],
                "origin_endpoint": option["origin_endpoint"],
                "bridge_endpoints": bridges,
                "destination_endpoint": option["stages"][-1]["endpoint"],
                "total_primitives": option["total_primitives"],
                "stages": len(option["stages"]),
                "status": record["status"],
                "destination_reached": record["status"] == "reached",
                "bridges_charged": sum(endpoint in charged for endpoint in bridges),
                "bridges_in_valley": sum(scorer.in_the_valley(e) for e in bridges),
                "origin_score": charged.get(option["origin_endpoint"]),
                "bridge_scores": [charged.get(endpoint) for endpoint in bridges],
                "destination_score": charged.get(option["stages"][-1]["endpoint"]),
            }
        )
    # MEASURED on this harness: the jump lane spends its full `wall_seconds` every round
    # (20.1s against a 20.0s budget) while the structured lane finishes inside it. So the
    # size of the pool a declared leg competes against is LOAD-DEPENDENT, and on a busy
    # machine the two arms of a pair can face different competition. The lane is kept --
    # removing the main competitor would make the mechanism look better for free, and
    # competition for allocation slots is exactly what "allocation discard" means -- so
    # the confound is recorded per arm instead, as a load-independent work counter beside
    # the seconds. A reader can check the two arms of a pair faced comparable work.
    proposal_work = {"attempts": 0, "pool_candidates": 0, "seconds": 0.0, "rounds": 0}
    for path in sorted((output / "campaign").glob("round_*/pending.json")):
        published = json.loads(path.read_text())["batch"]
        proposal_work["rounds"] += 1
        proposal_work["attempts"] += len(published.get("attempts", []))
        proposal_work["pool_candidates"] += len(
            published.get("proposal_pool", published)["candidates"]
        )
        proposal_work["seconds"] += float(published.get("proposal_seconds") or 0.0)
    proposal_work["seconds"] = round(proposal_work["seconds"], 2)

    values = sorted((row["score"] for row in ledger.rows), reverse=True)
    return {
        "proposal_work": proposal_work,
        "window_decomposition": window_decomposition(output),
        "schema_version": SCHEMA,
        "seed": seed,
        "arm": "protected" if protection else "declared_unprotected",
        "protection_enabled": bool(protection),
        "macro_options_enabled": bool(enable_macro_options),
        "scorer": scorer.payload(),
        "benchmark_oracle_calls": 0,
        "synthetic_scorer_calls": scorer.calls,
        "charged_calls": len(ledger.rows),
        "budget": budget,
        "best": values[0] if values else None,
        "top10": sum(values[:10]) / len(values[:10]) if values else None,
        "rounds_completed": len(campaign["history"]),
        "options": options,
        "declaration": report["diagnostics"]["declaration"],
        "run_identity": identity(optimizer_kwargs),
    }


__all__ = [
    "INITIALIZATION",
    "JUMP_CHECKPOINT",
    "SCHEMA",
    "SYNTHETIC_REFERENCE",
    "VALLEY_DEPTH",
    "VALLEY_DROP",
    "ValleySimilarityScorer",
    "assert_production_batch_geometry",
    "configuration",
    "load_initialization",
    "load_jump_checkpoint",
    "run_arm",
    "window_decomposition",
]
