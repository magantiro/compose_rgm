"""TEST C: can BLIND search enter a region the frozen local controller can exploit?

Tests A and B removed two explanations for the PMO gap.  Test A showed that,
given a destination, transport CONSTRUCTS and TRANSFERS it (102/102 exact
recovery from neutral sources), so "COMPOSE cannot build these molecules" is
false.  Test B showed that, started AT a teacher region with 64 calls, the
frozen local controller fills a strong top ten, so local exploitation is not
the gap either.  What remains is DISCOVERY: started from the task-independent
bank with no atlas, does the optimizer ever reach a region of that quality?

Operational definition
----------------------
A region is PRODUCTIVE when the frozen local controller, started there with a
fixed small budget, repeatedly produces high-value DISTINCT candidates.  That
is exactly the quantity Test B measured at three positions on every teacher
spine, so those runs are the calibrated ladder this test reads against.
Structural proximity to a teacher route is a DIAGNOSTIC only: two nearby
molecules can behave differently and two distant scaffolds can both be
excellent, so productivity is always verified with the objective itself.

Information regime
------------------
Everything here is ``DEVELOPMENT_INFORMED_DIAGNOSTIC``.  The atlas is visible
only to the EVALUATOR.  The optimizer under test receives the production
task-independent initialization and nothing else, and
:func:`assert_optimizer_blind` refuses to let a run start otherwise.  Every
objective evaluation is a counted diagnostic call.

Data structures and invariants
------------------------------
* :class:`TrajectoryRow` is one charged call of a blind run, carrying the
  ledger's order, the score, and the proposal provenance (parent, channel,
  program size) recovered from the campaign's own round records.  Rows are in
  ledger order, so ``index`` IS the charged-call number at which the molecule
  was first seen.
* The 48-slot endpoint STATE is carried through from the campaign record, never
  rebuilt by re-parsing SMILES: a SMILES round trip yields a tight graph, and
  the PMO path hard-refuses anything but an exact 48-slot source.
* :func:`nearest_approach` reports the RUNNING MAXIMUM similarity as a function
  of charged calls, so "how close did it get, and when" is answered by one
  monotone curve rather than by a single summary number.
* :func:`classify_rung` names the highest Test-B rung a measured productivity
  matches, and refuses to name a rung the measurement does not reach.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compose_v4.experiments.pmo_atlas_routes import (
    DEVELOPMENT_INFORMED_LABEL,
    AtlasDossier,
    file_sha256,
)

# ---- Blind initialization ----

#: The production task-independent initialization, pinned exactly as
#: ``configs/pmo_population_controller_v1.json`` pins it.  The optimizer under
#: test starts from this and from nothing else.
BLIND_INITIALIZATION_PATH = "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
BLIND_INITIALIZATION_SHA256 = "a1f7840df1f75567565de633c4df3d013c3165df3e43cbf2ac87b156d5f206ff"
BLIND_INITIALIZATION_SOURCE_SHA256 = (
    "b679b4dc54006acf570fefb80f49d83a46c18c979c9e376fdd4d37b2cec5f860"
)
BLIND_INITIALIZATION_COUNT = 16


class OptimizerBlindnessError(RuntimeError):
    """Raised when the optimizer under test could see answer-known information."""


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule)


def load_blind_initialization(repo_root: Path | str) -> dict[str, Any]:
    """Load the production task-independent initialization, or refuse.

    The file hash is checked against the value the production contract pins, so
    a locally edited bank cannot enter a blind run wearing the production name.
    """

    from compose_v4.control.docking_value import identity

    path = Path(repo_root) / BLIND_INITIALIZATION_PATH
    digest = file_sha256(path)
    if digest != BLIND_INITIALIZATION_SHA256:
        raise OptimizerBlindnessError(
            f"{BLIND_INITIALIZATION_PATH} moved: expected {BLIND_INITIALIZATION_SHA256}, "
            f"observed {digest}"
        )
    payload = json.loads(path.read_text())
    body = {key: value for key, value in payload.items() if key != "lock_sha256"}
    if identity(body) != payload.get("lock_sha256"):
        raise OptimizerBlindnessError("blind initialization lock does not match its content")
    if payload.get("source_sha256") != BLIND_INITIALIZATION_SOURCE_SHA256:
        raise OptimizerBlindnessError("blind initialization source bank changed")
    if payload.get("count") != BLIND_INITIALIZATION_COUNT or len(
        payload.get("candidates", ())
    ) != BLIND_INITIALIZATION_COUNT:
        raise OptimizerBlindnessError("blind initialization count changed")
    if any("score" in row or "task" in row for row in payload["candidates"]):
        raise OptimizerBlindnessError("blind initialization carries task information")
    return payload


def atlas_molecules(dossier: AtlasDossier, *, task: str | None = None) -> frozenset[str]:
    """Every molecule the atlas knows, as canonical SMILES.

    Sources, recorded endpoints, declared destinations and every recorded
    INTERMEDIATE state are included: a run seeded with a teacher midpoint is
    just as informed as one seeded with the answer.
    """

    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.trace_shard import decode_state

    found: set[str] = set()
    for route in dossier.routes:
        if task is not None and route.task != task:
            continue
        for smiles in (route.source_smiles, route.destination_smiles, route.recorded_endpoint_smiles):
            if smiles:
                canonical = _canonical(smiles)
                if canonical:
                    found.add(canonical)
        for state in route.recorded_states:
            canonical = _canonical(canonical_state_key(decode_state(state)))
            if canonical:
                found.add(canonical)
    return frozenset(found)


def assert_optimizer_blind(
    *,
    initialization: dict[str, Any],
    optimizer_kwargs: dict[str, Any],
    library: Sequence[Any],
    atlas: frozenset[str],
) -> dict[str, Any]:
    """Refuse to start a blind run whose inputs carry answer-known molecules.

    Two independent checks, deliberately not sharing a sink: the initialization
    must be the pinned production bank by hash (enforced at load), and no
    molecule reachable from the optimizer's own inputs may appear in the atlas.
    """

    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.trace_shard import decode_state

    if library:
        raise OptimizerBlindnessError(
            "a blind run takes no program library; a distilled library is answer-known"
        )
    seen: list[str] = []
    for candidate in initialization.get("candidates", ()):
        endpoint = _canonical(str(candidate["endpoint"]))
        if endpoint:
            seen.append(endpoint)
        state = candidate.get("state")
        if state is not None:
            decoded = _canonical(canonical_state_key(decode_state(state)))
            if decoded:
                seen.append(decoded)
    for key in ("destination", "target", "teacher", "atlas", "route", "anchor"):
        if key in optimizer_kwargs:
            raise OptimizerBlindnessError(
                f"optimizer_kwargs carries an answer-known key: {key!r}"
            )
    leaked = sorted(set(seen) & atlas)
    if leaked:
        raise OptimizerBlindnessError(
            f"{len(leaked)} optimizer input molecules are atlas molecules: {leaked[:3]}"
        )
    return {
        "initialization_molecules": len(set(seen)),
        "atlas_molecules_compared": len(atlas),
        "leaked": [],
        "library_size": len(library),
        "information_regime": DEVELOPMENT_INFORMED_LABEL,
    }


# ---- Blind trajectory ----


@dataclass(frozen=True)
class TrajectoryRow:
    """One charged call of a blind run, in ledger order."""

    index: int
    endpoint: str
    score: float
    role: str
    round_index: int | None = None
    parent_endpoint: str | None = None
    parent_score: float | None = None
    parent_label_disagrees: bool = False
    planner_channel: str | None = None
    mode: str | None = None
    program_primitives: int | None = None
    heavy_atoms: int | None = None
    state: dict[str, Any] | None = field(default=None, repr=False)


def _provenance_index(campaign: Path) -> dict[str, dict[str, Any]]:
    """Map canonical endpoint -> the proposal record that produced it.

    Provenance is read from the campaign's own round records rather than
    reconstructed, so a molecule whose proposal record is missing is reported
    with ``None`` fields instead of an invented parent.
    """

    index: dict[str, dict[str, Any]] = {}
    for pending in sorted(campaign.glob("round_*/pending.json")):
        try:
            payload = json.loads(pending.read_text())
        except json.JSONDecodeError:
            continue
        round_index = int(pending.parent.name.split("_")[-1])
        batch = payload.get("batch") or {}
        for candidate in batch.get("candidates", ()):
            endpoint = _canonical(str(candidate.get("endpoint", "")))
            if not endpoint or endpoint in index:
                continue
            provenance = candidate.get("provenance") or {}
            trace = candidate.get("trace") or {}
            states = trace.get("states") or ()
            size = candidate.get("program_size") or provenance.get("program_size") or {}
            # TWO independent routes to the parent: the label the controller wrote,
            # and the FIRST STATE of the trace the executor actually replayed.  The
            # trace is authoritative -- a label can drift, an executed state cannot --
            # and a disagreement is recorded rather than silently resolved.
            labelled = (
                _canonical(str(provenance["parent_endpoint"]))
                if provenance.get("parent_endpoint")
                else None
            )
            executed = _state_smiles(states[0]) if states else None
            index[endpoint] = {
                "round_index": round_index,
                "parent_endpoint": executed or labelled,
                "parent_endpoint_labelled": labelled,
                "parent_endpoint_executed": executed,
                "parent_label_disagrees": bool(
                    labelled and executed and labelled != executed
                ),
                "parent_score": provenance.get("parent_measured_score"),
                "planner_channel": provenance.get("planner_channel"),
                "mode": provenance.get("mode"),
                "program_primitives": size.get("primitive_count")
                or provenance.get("primitive_count")
                or (len(trace["actions"]) if trace.get("actions") is not None else None),
                "state": states[-1] if states else None,
            }
    return index


def read_blind_trajectory(run_dir: Path | str) -> tuple[TrajectoryRow, ...]:
    """Reconstruct a blind run's charged trajectory from its durable artifacts.

    The ORDER is the ledger's, which is the charged-call order; provenance is
    joined in by canonical endpoint.  A molecule the ledger charged but no
    round record claims is an INITIALIZATION molecule, and is labelled so.
    """

    run = Path(run_dir)
    rows: list[TrajectoryRow] = []
    provenance = _provenance_index(run / "campaign")
    initialization: set[str] = set()
    init_states: dict[str, Any] = {}
    init_path = run / "campaign" / "initialization.json"
    if init_path.exists():
        payload = json.loads(init_path.read_text())
        for candidate in payload.get("candidates", ()):
            key = _canonical(str(candidate.get("endpoint", "")))
            if key:
                initialization.add(key)
                init_states[key] = candidate.get("state")
    for result in sorted((run / "oracle").glob("query_*/result.json")):
        record = json.loads(result.read_text())
        if record.get("status") != "complete":
            raise RuntimeError(f"incomplete charged query in a blind run: {result}")
        endpoint = _canonical(str(record["endpoint"])) or str(record["endpoint"])
        joined = provenance.get(endpoint, {})
        state = joined.get("state") or init_states.get(endpoint)
        rows.append(
            TrajectoryRow(
                index=int(record["index"]) + 1,
                endpoint=endpoint,
                score=float(record["score"]),
                role="initialization" if endpoint in initialization else "candidate",
                round_index=joined.get("round_index"),
                parent_endpoint=joined.get("parent_endpoint"),
                parent_score=joined.get("parent_score"),
                parent_label_disagrees=bool(joined.get("parent_label_disagrees")),
                planner_channel=joined.get("planner_channel"),
                mode=joined.get("mode"),
                program_primitives=joined.get("program_primitives"),
                heavy_atoms=_heavy_atoms(endpoint),
                state=state,
            )
        )
    rows.sort(key=lambda row: row.index)
    return tuple(rows)


def _state_smiles(state: dict[str, Any] | None) -> str | None:
    """Canonical SMILES of an encoded 48-slot state, without a SMILES round trip."""

    if state is None:
        return None
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.trace_shard import decode_state

    return _canonical(canonical_state_key(decode_state(state)))


def _heavy_atoms(smiles: str) -> int | None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else molecule.GetNumHeavyAtoms()


# ---- Structural diagnostic ----


def _fingerprint(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)


def nearest_approach(
    trajectory: Sequence[TrajectoryRow],
    references: Iterable[str],
) -> dict[str, Any]:
    """Running maximum Tanimoto from a blind trajectory to a reference set.

    Structural only: this locates candidates worth verifying, and never decides
    by itself whether a region is productive.
    """

    from rdkit import DataStructs

    reference_prints = []
    reference_smiles = []
    for smiles in references:
        print_ = _fingerprint(smiles)
        if print_ is not None:
            reference_prints.append(print_)
            reference_smiles.append(smiles)
    if not reference_prints:
        raise ValueError("an empty reference set cannot measure an approach")

    best = 0.0
    best_row: TrajectoryRow | None = None
    best_reference: str | None = None
    curve: list[dict[str, Any]] = []
    per_row: list[float] = []
    for row in trajectory:
        print_ = _fingerprint(row.endpoint)
        if print_ is None:
            per_row.append(0.0)
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(print_, reference_prints)
        value = max(similarities)
        per_row.append(value)
        if value > best:
            best = value
            best_row = row
            best_reference = reference_smiles[similarities.index(value)]
            curve.append({"charged_calls": row.index, "max_similarity": round(value, 4)})
    return {
        "references": len(reference_prints),
        "max_similarity": round(best, 4),
        "at_charged_call": best_row.index if best_row else None,
        "nearest_blind_molecule": best_row.endpoint if best_row else None,
        "nearest_reference": best_reference,
        "running_max_curve": curve,
        "per_row_similarity": [round(value, 4) for value in per_row],
    }


# ---- Productivity ladder ----


@dataclass(frozen=True)
class LadderRung:
    task: str
    label: str
    seed_score: float
    top_ten_new_mean: float


def load_ladder(test_b_payload: dict[str, Any]) -> dict[str, tuple[LadderRung, ...]]:
    """The Test-B calibration: what a teacher region is worth, per task."""

    ladder: dict[str, list[LadderRung]] = {}
    for run in test_b_payload["runs"]:
        ladder.setdefault(run["task"], []).append(
            LadderRung(
                task=run["task"],
                label=run["checkpoint_label"],
                seed_score=float(run["seed_score"]),
                top_ten_new_mean=float(run["top_ten_new_mean"]),
            )
        )
    return {
        task: tuple(sorted(rungs, key=lambda rung: rung.top_ten_new_mean))
        for task, rungs in ladder.items()
    }


def classify_rung(
    measured_top_ten_new_mean: float,
    rungs: Sequence[LadderRung],
) -> dict[str, Any]:
    """Name the highest teacher rung a measured productivity reaches.

    A measurement below the lowest rung is reported as ``below_early`` rather
    than being rounded up to it: the ladder is what a teacher region is worth,
    and a run that does not reach the bottom of it has not entered one.
    """

    if not rungs:
        raise ValueError("an empty ladder cannot classify a measurement")
    ordered = sorted(rungs, key=lambda rung: rung.top_ten_new_mean)
    reached = [rung for rung in ordered if measured_top_ten_new_mean >= rung.top_ten_new_mean]
    label = reached[-1].label if reached else "below_" + ordered[0].label
    top = ordered[-1]
    return {
        "rung_reached": label,
        "rungs_reached": len(reached),
        "rungs_total": len(ordered),
        "measured": round(measured_top_ten_new_mean, 4),
        "anchor_productivity": round(top.top_ten_new_mean, 4),
        "fraction_of_anchor": round(measured_top_ten_new_mean / top.top_ten_new_mean, 4)
        if top.top_ten_new_mean
        else None,
        "ladder": [
            {"label": rung.label, "top_ten_new_mean": round(rung.top_ten_new_mean, 4)}
            for rung in ordered
        ],
    }


# ---- Verified productivity ----

#: The campaign geometry a productivity probe shares with Test B.  Test C is
#: only comparable to the teacher ladder if the probe runs the SAME controller
#: with the SAME budget from a different seed molecule, so these kwargs are
#: asserted equal to the Test-B driver's by ``tests/test_pmo_atlas_discovery.py``
#: rather than transcribed and hoped for.
def local_lift_kwargs(
    *,
    repo_root: Path,
    config,
    rounds: int,
    queries_per_round: int,
    optimizer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    from compose_v4.control.dynamic_program_synthesis_v21 import (
        initial_dynamic_program_batch_v21,
    )
    from compose_v4.control.pmo_population_controller import PmoPopulationController

    del repo_root
    return {
        "config": config,
        "library": (),
        "rounds": rounds,
        "queries_per_round": queries_per_round,
        "hierarchy": None,
        "fit_model": None,
        "stagnation_rounds": None,
        "bootstrap_rounds": 1,
        "initialization_mode": "all_scored_pool",
        "initial_parent_fraction": 0.2,
        "optimizer_type": PmoPopulationController,
        "optimizer_kwargs": optimizer_kwargs,
        "initial_batch_fn": initial_dynamic_program_batch_v21,
    }


def single_seed_initialization(smiles: str, state: dict[str, Any], origin: str) -> dict[str, Any]:
    """One measured molecule as the whole starting population."""

    from compose_v4.control.docking_value import identity

    body = {
        "candidates": [{"endpoint": smiles, "source_id": origin, "state": state}],
        "count": 1,
        "task_independent": False,
        "provenance": origin,
        "information_regime": DEVELOPMENT_INFORMED_LABEL,
    }
    return {**body, "lock_sha256": identity(body)}


def summarize_lift(
    rows: Sequence[dict[str, Any]],
    seed_smiles: str,
) -> dict[str, Any]:
    """Test B's reduction: a run that only preserves its seed is not a lift."""

    scores = {row["endpoint"]: float(row["score"]) for row in rows}
    seed_score = scores.get(seed_smiles)
    new = {smiles: value for smiles, value in scores.items() if smiles != seed_smiles}
    ranked = sorted(new.values(), reverse=True)
    top_ten_new = ranked[:10]
    above = [v for v in new.values() if seed_score is not None and v > seed_score]
    return {
        "seed_score": seed_score,
        "charged_calls": len(rows),
        "distinct_new_molecules": len(new),
        "best_overall": max(scores.values()) if scores else None,
        "best_new": ranked[0] if ranked else None,
        "top_ten_new_mean": (sum(top_ten_new) / len(top_ten_new)) if top_ten_new else None,
        "top_ten_new_count": len(top_ten_new),
        "distinct_new_above_seed": len(above),
        "lift_best_new_minus_seed": (
            ranked[0] - seed_score if ranked and seed_score is not None else None
        ),
    }
