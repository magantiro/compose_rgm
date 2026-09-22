"""Does the SCORED entry point reach the donor cut law? Proven by execution, offline.

WHY THIS IS NOT THE TEST THAT ALREADY EXISTS
--------------------------------------------
``tests/test_pmo_donor_channel_wiring.py`` proves the CONTROLLER consumes its law: it
hands a raising law to a hand-built :class:`PmoPopulationController` and requires
``propose_batch`` to reach it.  That is necessary and it is not sufficient, because every
inert mechanism this repository has shipped was inert at a hop ABOVE the object that was
tested.  ``bridge_region_law`` was measured, tested and merged with a working
``region_law=`` keyword; what was missing was a production caller passing it.  A rescue
launched on that commit would have spent up to 1,241 oracle calls reproducing the same
failure it was built to fix.

So this gate starts where a scored run starts.  It drives the REAL
:func:`compose_v4.experiments.pmo_population_v1.execute_task` -- not a transcription of
its argument handling -- captures the ``optimizer_kwargs`` it actually builds, constructs
the controller from exactly those kwargs the way
:func:`~compose_v4.control.program_campaign.run_program_campaign` does, and then requires
the production ``propose_batch`` to reach a law that raises.

THE THREE HOPS IT CLOSES, AND WHY EACH ONE CAN FAIL ALONE
----------------------------------------------------------
1. ``execute_task`` -> ``optimizer_kwargs``.  A flag that stops here leaves the lane
   unreachable from every scored launch while every controller-level test stays green.
2. ``optimizer_kwargs`` -> the CONSTRUCTOR.  ``run_program_campaign`` splats the dict, so
   a key the constructor does not accept is a ``TypeError`` at round zero.
3. ``optimizer_kwargs`` -> ``restore``.  The campaign splats the SAME dict into
   ``restore`` on every resumed round, so a flag accepted by ``__init__`` alone makes the
   fresh path work and the RESUME path raise -- which is exactly how this surfaced once
   before, on the first 1k extension, after the ledger had already been widened.

ZERO ORACLE CALLS, STRUCTURALLY
-------------------------------
The evaluator handed to ``execute_task`` RAISES.  Scoring is not merely absent by
intention; a single charged call would abort the gate with that exception.  The campaign
itself is replaced, so nothing runs a round.  The runtime contract is built exactly as
:mod:`modal_apps.pmo_population_v1_app` builds it -- ``dict(contract)`` with the two
authorization bits set -- which is a documented runtime construction, not an authorization:
no launcher is invoked, no payload is re-sealed, and no molecule is scored.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_donor_channel import (
    DEFAULT_CUT_LAW,
    DONOR_CHANNEL,
    DonorLawProbe,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.experiments import pmo_population_v1
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)

#: The donor keys the scored path must carry when the lane is on, and must NOT carry when
#: it is off. Absence is the only byte-identical off at this level too: the campaign hashes
#: `optimizer_kwargs` into `optimizer_kwargs_sha256`, writes it into the manifest, and
#: refuses a resume whose recipe moved -- so an unconditional key would break every
#: in-flight arm A/B/C run at its next preemption retry while changing nothing it computes.
DONOR_KWARGS = ("enable_donor_channel", "donor_cut_law")


class _NoScoring(Exception):
    """Raised by the gate's evaluator. Reaching it means a charged call was attempted."""


def _contract_payload(root: Path) -> dict:
    """The sealed contract payload, envelope hash checked.

    Deliberately NOT ``pmo_population_v1.load_contract``. That preflight additionally
    verifies ``implementation_sha256`` over twelve source files, and this branch MEASURED
    that pin to be stale on ``pmo_population_controller.py`` and ``pmo_population_v1.py``
    ALREADY AT THE BRANCH BASE -- arms B and C moved both files without re-pinning it.
    Routing the gate through it would make every donor run report a failure that belongs
    to a different, pre-existing, owner-level decision. ``execute_task`` is the scored
    entry point and it takes a contract DICT; the verifications that gate IT --
    the initialization lock and the joint-checkpoint hash -- still run for real inside it.
    """
    envelope = json.loads((root / pmo_population_v1.CONTRACT).read_text())
    payload = envelope["payload"]
    if envelope.get("payload_sha256") != identity(payload):
        raise ValueError("PMO-v1 contract envelope hash changed")
    if payload.get("schema_version") != pmo_population_v1.SCHEMA:
        raise ValueError("unexpected PMO-v1 contract schema")
    return payload


def _runtime_contract(root: Path) -> dict:
    """The contract the worker runs, built the way the worker builds it."""
    runtime = dict(_contract_payload(root))
    runtime["scored_launch_authorized"] = True
    runtime["modal_launch_authorized"] = True
    return runtime


def capture_scored_optimizer_kwargs(
    root: Path,
    folder: Path,
    *,
    task_name: str = "celecoxib_rediscovery",
    **arm: Any,
) -> dict:
    """Run the REAL ``execute_task`` with the campaign stubbed; return what it passed.

    Everything before ``run_program_campaign`` executes for real -- the budget check, the
    task lock, the authorization check, the initialization lock, the checkpoint hash and
    the config. Only the campaign is replaced, and only so the gate does not need a round.
    """
    captured: dict[str, Any] = {}
    original = pmo_population_v1.run_program_campaign

    def _stub(**kwargs):
        captured["optimizer_kwargs"] = dict(kwargs.get("optimizer_kwargs") or {})
        captured["optimizer_type"] = kwargs.get("optimizer_type")
        captured["config"] = kwargs.get("config")
        captured["task"] = kwargs.get("task")
        return {"stubbed": True, "history": []}

    pmo_population_v1.run_program_campaign = _stub
    try:
        pmo_population_v1.execute_task(
            _runtime_contract(root),
            root,
            folder,
            task_name,
            evaluate=_evaluate_must_not_be_called,
            charged_calls_per_task=250,
            **arm,
        )
    finally:
        pmo_population_v1.run_program_campaign = original
    if "optimizer_kwargs" not in captured:
        raise AssertionError("execute_task never reached run_program_campaign")
    return captured


def _evaluate_must_not_be_called(smiles: str) -> float:
    raise _NoScoring(f"the gate attempted a charged oracle call on {smiles!r}")


def _seeded_controller(kwargs: dict, source_smiles: str, *, seed: int) -> PmoPopulationController:
    """Build the controller the way ``run_program_campaign`` does, then give it parents.

    The construction line is deliberately the campaign's own shape -- positional config,
    ``source_group``/``oracle_protocol``/``hierarchy`` keywords, then ``**optimizer_kwargs``
    -- because the hop being proven is that THOSE kwargs reach THIS constructor.
    """
    source = production_state_from_smiles(source_smiles, max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=6,
        candidates_per_batch=4,
        wall_seconds=20,
        parent_allocation="niche_score",
    )
    batch = initial_dynamic_program_batch_v21(
        source,
        (),
        config,
        source_group="donor-scored-gate",
        oracle_protocol="free-no-oracle",
        eligibility=_eligibility,
    )
    controller = PmoPopulationController(
        config,
        source_group="donor-scored-gate",
        oracle_protocol="free-no-oracle",
        hierarchy=None,
        **kwargs,
    )
    for index, candidate in enumerate(batch["candidates"][:4]):
        controller.add_measured_program(candidate, receipt_id=f"gate{index}", score=0.5)
    return controller


def _eligibility(row: dict) -> dict:
    from rdkit import Chem

    return {"oracle_eligible": Chem.MolFromSmiles(row["smiles"]) is not None}


def gate(
    root: Path,
    folder: Path,
    *,
    source_smiles: str | None = None,
    seeds: int = 16,
) -> dict:
    """The whole chain, executed. Raises on any broken hop; returns the evidence.

    ``seeds`` is FIXED and plural on purpose. One draw can legitimately miss the law for
    an unrelated reason -- every region refused, no donor available -- so a single random
    seed would make a wiring defect and an unlucky draw indistinguishable. With fixed
    seeds the verdict is deterministic: for a given code state this always passes or
    always fails.
    """
    off = capture_scored_optimizer_kwargs(root, folder / "off")
    on = capture_scored_optimizer_kwargs(root, folder / "on", enable_donor_channel=True)

    present = [key for key in DONOR_KWARGS if key in off["optimizer_kwargs"]]
    if present:
        raise AssertionError(
            f"the OFF arm carried donor keys {present}; absence is the only "
            "byte-identical off, because the campaign hashes these kwargs into the "
            "manifest recipe and refuses a resume whose recipe moved"
        )
    missing = [key for key in DONOR_KWARGS if key not in on["optimizer_kwargs"]]
    if missing:
        raise AssertionError(
            f"the scored entry point never passed {missing} -- the donor lane is "
            "unreachable from execute_task, exactly as the region law was unreachable "
            "from its production caller"
        )
    if on["optimizer_type"] is not PmoPopulationController:
        raise AssertionError("the scored entry point no longer runs the population controller")

    source = source_smiles or _initialization_source(root)
    controller = _seeded_controller(on["optimizer_kwargs"], source, seed=11)
    if DONOR_CHANNEL not in controller.channels:
        raise AssertionError(
            "the scored kwargs built a controller whose live channel set has no donor "
            "lane -- the flag reached the constructor and was dropped inside it"
        )

    consumed = _consume(on["optimizer_kwargs"], source, seeds=seeds)
    restored = _restore_accepts(controller, on["optimizer_kwargs"])
    return {
        "schema_version": "pmo_donor_scored_gate_v1",
        "oracle_calls": 0,
        "entry_point": "compose_v4.experiments.pmo_population_v1.execute_task",
        "source_smiles": source,
        "off_arm_optimizer_kwargs": sorted(off["optimizer_kwargs"]),
        "on_arm_optimizer_kwargs": sorted(on["optimizer_kwargs"]),
        "donor_cut_law": on["optimizer_kwargs"]["donor_cut_law"],
        "default_cut_law": DEFAULT_CUT_LAW,
        "live_channels": list(controller.channels),
        "law_consumed": consumed,
        "restore_accepts_scored_kwargs": restored,
        "verdict": "SCORED_ENTRY_POINT_REACHES_THE_DONOR_CUT_LAW",
    }


def _consume(kwargs: dict, source: str, *, seeds: int) -> dict:
    """Install a raising law where the controller resolves it; require it to be reached.

    The probe is a ``BaseException`` because the donor lane catches
    ``(ValueError, RuntimeError)`` per attempt exactly as the jump lane does; a probe
    raising either would be swallowed by the very code being observed.
    """
    for seed in range(seeds):
        controller = _seeded_controller(kwargs, source, seed=11)
        controller.donor_rng = np.random.default_rng([20260921, seed])

        def _probe(_graph):
            raise DonorLawProbe("the donor cut law was reached from the scored kwargs")

        controller.donor_law = _probe
        try:
            controller.propose_batch(_eligibility)
        except DonorLawProbe:
            return {"consumed": True, "seed": seed, "seeds_tried": seed + 1}
    raise AssertionError(
        f"the donor cut law was NEVER consulted across {seeds} fixed seeds from the "
        "scored entry point's own optimizer_kwargs -- the lane is wired and inert"
    )


def _restore_accepts(controller: PmoPopulationController, kwargs: dict) -> bool:
    """`run_program_campaign` splats the same dict into `restore` on every resumed round."""
    snapshot = json.loads(json.dumps(controller.snapshot(include_history=True)))
    restored = PmoPopulationController.restore(snapshot, hierarchy=None, **kwargs)
    if DONOR_CHANNEL not in restored.channels:
        raise AssertionError("restore rebuilt the donor arm as an unflagged controller")
    if restored.donor_cut_law != controller.donor_cut_law:
        raise AssertionError("restore rebuilt the donor lane under a different cut law")
    return True


def _initialization_source(root: Path) -> str:
    """A molecule from the scored run's OWN initialization, so the gate runs on the real
    substrate rather than a fixture the wiring might happen to suit."""
    contract = _contract_payload(root)
    path = root / contract["initialization"]["path"]
    return json.loads(path.read_text())["candidates"][0]["endpoint"]
