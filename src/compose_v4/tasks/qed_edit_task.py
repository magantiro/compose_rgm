"""Dedicated constrained-QED editing task (Jin ICLR-2019 panel), not a T4 label.

WHY THIS EXISTS
---------------
Routing the constrained-QED benchmark through ``ProgramTask(kind="t4")`` imports
three restrictions the benchmark does not have, and inverts the objective:

1. ``ProgramTask.endpoint_evaluator`` hardcodes ``qed_min=0.6`` and ``sa_max=4.0``
   (``src/compose_v4/control/program_task.py:76-77``) and then screens the result
   through ``t4_endpoint_selection.acceptable_endpoint``, which additionally
   applies ``compose_v4.gates.med_chem_gate.is_valid``.  A candidate at QED 0.95 /
   similarity 0.5 / SA 4.3 *solves* this benchmark and is refused by that gate.
   Because the 0.6 floor is a literal in the function body, a contract field
   ``qed_target: 0.90`` pointed at that path would be decorative.
2. ``parent_edit_search.prepare_query_batch`` (line 23) forces
   ``score_direction == "minimize" if task.kind == "t4" else "maximize"``, and
   ``ProgramTask.utility`` negates for ``t4``.  QED is a MAXIMIZATION objective,
   so the T4 pairing ranks the archive by *lowest* QED.
3. ``PRODUCTION_MAX_ATOMS = 40`` is an ACTIVE heavy-atom ceiling, not a padding
   width.  ``whole_ring_plan.execute_program`` requires
   ``graph.n_atoms == 48 and 1 <= graph.n_real_atoms <= 40`` -- two different
   numbers meaning two different things.  Padding to 40 slots fails closed with
   "expected an exact supported 48-slot source".

None of the frozen modules is edited.  This task is a drop-in duck-typed peer of
``ProgramTask`` (``name``/``oracle_protocol``/``kind``/``top_k``/``task_id``/
``utility``/``endpoint_evaluator``) that supplies the correct semantics instead.

FEASIBILITY LIVES IN THE SUPPORT, NOT IN THE REWARD
---------------------------------------------------
``AdaptiveProgramSearch.propose_batch`` applies the task's endpoint evaluator to
every executed endpoint and drops anything whose ``oracle_eligible`` is False
before it can become a candidate (``adaptive_program_optimizer.py``:
``if outcome != "eligible": continue``).  That is the support seam.  This task
puts the similarity floor there and lets the reward be QED itself.  The earlier
wiring carried the similarity indicator inside the reward
(``score = QED * 1[sim >= 0.4]``), which admitted infeasible endpoints and spent
budget on zero-scored molecules.

OUTPUT FEASIBILITY IS SEPARATE FROM INTERMEDIATE PROGRAM VALIDITY
------------------------------------------------------------------
``qed_target`` is a SUCCESS criterion on returned endpoints (``is_success``).  It
is deliberately NOT an admission gate: requiring every proposed endpoint to
already satisfy QED >= 0.90 would leave the search nothing to climb, and requiring
it of every intermediate primitive state would refuse valid programs whose last
block supplies the gain.

THRESHOLDS ARE CONTRACT-GOVERNED, NOT MODULE CONSTANTS
-------------------------------------------------------
``qed_target`` and ``similarity_floor`` have NO defaults on the dataclass.  A
contract that omits them is a ``TypeError``, never a silent fallback, and both
values enter ``task_id`` through ``identity(asdict(self))`` -- so a changed
threshold is a different task by construction.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdFingerprintGenerator

from compose_v4.control.docking_value import identity

# ---- Slot semantics: two numbers, two meanings ----

# The exact padded array width whole_ring_plan.execute_program requires.
ALLOCATED_GRAPH_SLOTS = 48
# The active heavy-atom ceiling. This -- not the padding width -- is what
# editing_v2_evaluation_semantics.PRODUCTION_MAX_ATOMS = 40 actually bounds.
MAX_ACTIVE_HEAVY_ATOMS = 40

# Frozen dispatch vocabulary. ``kind`` is a routing token consumed by frozen
# control code (parent_edit_search direction guard, program_campaign
# initialization-mode guard), NOT the identity of the task. It is resolved and
# asserted at construction by resolve_score_direction, never assumed.
DISPATCH_KIND = "pmo"

_CONTRACT_SCHEMA = "qed_edit_task_contract_v1"


# ---- Runtime direction resolution ----


class _DirectionProbeSearch:
    """Minimal stand-in used only to interrogate the frozen direction guard."""

    def __init__(self, oracle_protocol: str, direction: str):
        self.oracle_protocol = oracle_protocol
        self.config = type("_Config", (), {"score_direction": direction})()


def resolve_score_direction(task) -> str:
    """Ask the frozen guard which direction it demands for ``task``.

    The rule lives in ``parent_edit_search.prepare_query_batch`` line 23.  It is
    resolved here by CALLING that function and observing which direction it
    refuses, rather than by restating the expression -- a restatement would agree
    with itself even if the frozen rule changed underneath it.
    """
    from compose_v4.control.parent_edit_search import prepare_query_batch

    accepted = []
    for direction in ("maximize", "minimize"):
        probe = _DirectionProbeSearch(task.oracle_protocol, direction)
        try:
            # count=0 is refused AFTER the direction guard, so reaching that
            # refusal proves the direction guard passed.
            prepare_query_batch(probe, task, count=0, seed=0)
        except ValueError as error:
            if "direction disagrees" in str(error):
                continue
            accepted.append(direction)
            continue
        accepted.append(direction)
    if len(accepted) != 1:
        raise RuntimeError(
            f"frozen direction guard did not resolve to exactly one direction: {accepted}"
        )
    return accepted[0]


# ---- The task ----


@dataclass(frozen=True)
class QedEditTask:
    """Constrained-QED editing task. Duck-typed peer of ProgramTask.

    Fields carry no scientific defaults: every threshold must arrive from a
    contract.  ``asdict`` must stay JSON-serialisable because
    ``ProgramQueryLedger`` writes it into its manifest.
    """

    name: str
    oracle_protocol: str
    original_source: str
    qed_target: float
    similarity_floor: float
    fingerprint_radius: int
    fingerprint_bits: int
    archive_top_k: int
    kind: str = DISPATCH_KIND

    def __post_init__(self):
        if not self.name or not self.oracle_protocol:
            raise ValueError("task needs a name and an oracle protocol")
        if not self.original_source or Chem.MolFromSmiles(self.original_source) is None:
            raise ValueError("QED editing task requires a parseable original source molecule")
        for field, value in (
            ("qed_target", self.qed_target),
            ("similarity_floor", self.similarity_floor),
        ):
            if type(value) not in (int, float):
                raise ValueError(f"{field} must be a number supplied by the contract")
            if not math.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field} must be a finite value in [0, 1]")
        if type(self.fingerprint_radius) is not int or self.fingerprint_radius < 0:
            raise ValueError("fingerprint radius must be a nonnegative integer")
        if type(self.fingerprint_bits) is not int or self.fingerprint_bits < 1:
            raise ValueError("fingerprint bit count must be a positive integer")
        if type(self.archive_top_k) is not int or self.archive_top_k < 1:
            raise ValueError("archive top-k must be a positive integer")
        direction = resolve_score_direction(self)
        if direction != "maximize":
            raise ValueError(
                "QED is a maximization objective but the frozen direction guard "
                f"demands {direction!r} for kind={self.kind!r}; this pairing would "
                "invert the objective"
            )

    # -- ProgramTask-compatible surface --

    @property
    def top_k(self) -> int:
        return self.archive_top_k

    @property
    def task_id(self) -> str:
        return identity(asdict(self))

    def utility(self, score):
        """Maximization. No negation, no SA term, no blended scalar."""
        if isinstance(score, bool) or not math.isfinite(score):
            raise ValueError("task observations must be finite numbers")
        if not 0 <= score <= 1:
            raise ValueError("QED reward outside declared [0,1] range")
        return float(score)

    # -- Chemistry --

    def _generator(self):
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fingerprint_radius, fpSize=self.fingerprint_bits
        )

    def measure(self, smiles: str) -> dict:
        """QED and Tanimoto-to-original for one endpoint. Local RDKit; free."""
        generator = self._generator()
        source_fp = generator.GetFingerprint(Chem.MolFromSmiles(self.original_source))
        return self._measure_with(smiles, generator, source_fp)

    def _measure_with(self, smiles, generator, source_fp) -> dict:
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
        if molecule is None:
            return {"valid": False, "qed": 0.0, "sim": 0.0}
        return {
            "valid": True,
            "qed": float(QED.qed(molecule)),
            "sim": float(
                DataStructs.TanimotoSimilarity(source_fp, generator.GetFingerprint(molecule))
            ),
        }

    def endpoint_evaluator(self):
        """Support gate for propose_batch: validity AND the similarity floor.

        Deliberately absent: the SA ceiling, the 0.6 QED floor, and the inherited
        medicinal-chemistry screen.  Deliberately absent as well: ``qed_target``
        -- output feasibility is not an admission condition.
        """
        generator = self._generator()
        source_fp = generator.GetFingerprint(Chem.MolFromSmiles(self.original_source))
        floor = float(self.similarity_floor)

        def evaluate(row):
            measured = self._measure_with(row.get("smiles"), generator, source_fp)
            eligible = bool(measured["valid"] and measured["sim"] >= floor)
            return {
                **row,
                **measured,
                "oracle_eligible": eligible,
                "reason": None if eligible else ("invalid_molecule" if not measured["valid"]
                                                 else "below_similarity_floor"),
            }

        return evaluate

    def reward(self):
        """Ledger scorer: QED itself, on the admitted support. No indicator."""
        generator = self._generator()
        source_fp = generator.GetFingerprint(Chem.MolFromSmiles(self.original_source))
        seen: dict[str, dict] = {}

        def score(smiles: str) -> float:
            measured = self._measure_with(smiles, generator, source_fp)
            seen[smiles] = measured
            return float(measured["qed"])

        score.observed = seen  # type: ignore[attr-defined]
        return score

    def is_success(self, measured: dict) -> bool:
        """Output feasibility: the benchmark success event on a RETURNED endpoint."""
        return bool(
            measured.get("valid")
            and float(measured["qed"]) >= float(self.qed_target)
            and float(measured["sim"]) >= float(self.similarity_floor)
        )


# ---- Contract loading ----


def load_qed_edit_contract(path) -> dict:
    contract = json.loads(Path(path).read_text())
    if contract.get("schema_version") != _CONTRACT_SCHEMA:
        raise ValueError(f"expected {_CONTRACT_SCHEMA}, found {contract.get('schema_version')!r}")
    for section, keys in (
        ("region", ("qed_target", "similarity_floor")),
        ("similarity", ("radius", "bits")),
        ("slots", ("allocated_graph_slots", "max_active_heavy_atoms")),
    ):
        missing = [k for k in keys if k not in contract.get(section, {})]
        if missing:
            raise ValueError(f"contract section {section!r} is missing {missing}")
    if "archive_top_k" not in contract:
        raise ValueError("contract is missing archive_top_k")
    slots = contract["slots"]
    if slots["allocated_graph_slots"] != ALLOCATED_GRAPH_SLOTS:
        raise ValueError(
            "contract allocated_graph_slots disagrees with the width the program "
            f"executor requires ({ALLOCATED_GRAPH_SLOTS})"
        )
    if slots["max_active_heavy_atoms"] != MAX_ACTIVE_HEAVY_ATOMS:
        raise ValueError(
            "contract max_active_heavy_atoms disagrees with the executor ceiling "
            f"({MAX_ACTIVE_HEAVY_ATOMS})"
        )
    return contract


def task_from_contract(contract: dict, *, name: str, oracle_protocol: str, source: str):
    """Build the task with every operative threshold taken from the contract."""
    return QedEditTask(
        name=name,
        oracle_protocol=oracle_protocol,
        original_source=source,
        qed_target=float(contract["region"]["qed_target"]),
        similarity_floor=float(contract["region"]["similarity_floor"]),
        fingerprint_radius=int(contract["similarity"]["radius"]),
        fingerprint_bits=int(contract["similarity"]["bits"]),
        archive_top_k=int(contract["archive_top_k"]),
    )


def production_source_state(smiles: str):
    """Padded source state with the two slot numbers kept distinct.

    ``pad`` to ALLOCATED_GRAPH_SLOTS (the array width the executor requires);
    verify ``n_real_atoms`` against MAX_ACTIVE_HEAVY_ATOMS (the chemistry limit).
    A TIGHT graph from ``smiles_to_molecular_graph`` silently removes the whole
    ``atom_insert`` family from the legal support, so this must never be skipped.
    """
    from compose_v4.experiments.editing_v2_evaluation_semantics import (
        production_state_from_smiles,
    )

    state = production_state_from_smiles(smiles, ALLOCATED_GRAPH_SLOTS)
    if state.n_atoms != ALLOCATED_GRAPH_SLOTS:
        raise ValueError(f"expected {ALLOCATED_GRAPH_SLOTS} allocated slots, got {state.n_atoms}")
    if not 1 <= state.n_real_atoms <= MAX_ACTIVE_HEAVY_ATOMS:
        raise ValueError(
            f"source has {state.n_real_atoms} active heavy atoms, outside "
            f"1..{MAX_ACTIVE_HEAVY_ATOMS}"
        )
    return state
