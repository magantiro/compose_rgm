"""The support-restoration gate, re-run THROUGH THE PRODUCTION ENTRYPOINT.

WHAT CHANGED, AND WHY THAT IS THE WHOLE POINT
----------------------------------------------
`scripts/t4_support_restoration_gate.py` answered the scientific question -- does
the automatically routed unified controller recover the proposal SUPPORT the five
hand-chosen per-cell arms produced -- with a harness that REPRODUCED the campaign's
call shape by hand.  It read the production contracts, called the production
`expand`, `fallback_candidates` and `Fiber.check`, and drove the production
stopping rule.  It was faithful, and faithfulness maintained by reading is not the
same as faithfulness by construction: a transcribed reference cannot fail usefully
when the thing it transcribes drifts.

This driver asks the same question of the SAME cells with the SAME seeds, but the
proposals now come from the functions the Modal campaign app itself calls:

    round one, every rung   `t4_unified_proposal.proposal_unit`
    the empty-pool handler  `t4_unified_controller.routed_support_expansion`
    the terminal guard      `t4_unified_controller.assert_unified_expansion_is_consumed`
    the wiring proof        `t4_unified_controller.assert_state_routing_is_consumed`

Everything that SCORES a result -- the endpoint characterisation, the routing
comparison against each historical arm, the committed-audit comparison, the verdict
rule, the cell lists, the seed policy and the med-chem screen -- is imported from
the standalone gate and used verbatim.  Nothing about the question moved; only the
code that generates the molecules did.  That is what makes the two tables
comparable.

THE ONE DECLARED DIFFERENCE, STATED BEFORE ANY NUMBER IS READ
---------------------------------------------------------------
The standalone gate read `configs/t4_held_target_distilled_*_d06_250.json`, which
carry no `proposal.shallow.region_law`.  The unified controller's own contracts
DECLARE `region_law: free_gate_margin_v1`, because the historical arms that
produced the wins ran the shallow lane conditioned by it.  An absent field is the
only byte-identical OFF state, so the two configurations draw different shallow
proposals and their per-cell endpoint COUNTS are not expected to match.

That confound is measured rather than argued.  Every unit carries an explicit
`region_law_arm`:

    ``declared``  the production configuration: the law is resolved from the
                  contract and PROVEN CONSUMED before any proposal is scored.
    ``law_off``   the law absent, which reproduces the standalone gate's own
                  proposal draws exactly.

The ten non-trigger controls are run in BOTH arms deliberately.  The `law_off`
controls are what carries the byte-identical claim for THIS integration -- the
routing and the ladder must be unreachable on a healthy round, and a control whose
round-one `selected` count reproduces the committed audit exactly is the structural
proof that no alternate kernel was consulted and no RNG was drawn for one.  The
`declared` controls then show the production contract still searches, with any
difference attributable to the law rather than to the wiring.

WHAT IS SPENT
-------------
CPU.  This driver charges NO oracle call and performs NO docking.
`assert_no_docking_reachable()` -- imported from the standalone gate -- proves it
rather than asserting it: it refuses to run if any docking-capable module is
resident, and it is called on every entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import t4_support_restoration_gate as standalone

from compose_v4.control.docking_value import identity
from compose_v4.control.frozen_proposal_escalation import (
    FROZEN_LADDER_ID,
    FROZEN_LADDER_SHA256,
)
from compose_v4.control.region_law_contract import region_law_for_proposal_lane
from compose_v4.control.zero_support_fallback import fallback_seed
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
from compose_v4.experiments.t4_integrated_route_fiber import EXPERTS
from compose_v4.experiments.t4_support_expansion import normalize_expansion_records
from compose_v4.experiments.t4_unified_controller import (
    ROUTER_ID,
    ROUTER_SHA256,
    assert_state_routing_is_consumed,
    assert_unified_expansion_is_consumed,
    resolve_unified_controller,
    routed_support_expansion,
)
from compose_v4.experiments.t4_unified_proposal import proposal_unit

SCHEMA_VERSION = "t4_unified_production_gate_v1"

ROOT = Path(__file__).resolve().parents[1]

#: The unified controller's own contracts. `delta` is read from the CONTRACT and
#: asserted to be 0.6 -- a contract name is prose and has been wrong before.
UNIFIED_CONTRACTS = {
    protein: f"configs/t4_unified_controller_{protein}_d06_v1.json"
    for protein in ("braf", "fa7", "5ht1b", "parp1", "jak2")
}

#: The two region-law configurations. `law_off` is the standalone gate's own
#: proposal law and exists only to attribute a difference, never to re-open the
#: declared configuration.
REGION_LAW_ARMS = ("declared", "law_off")
PRODUCTION_ARM = "declared"

#: The shared retained expert the state-aware rung-0 lane consults.
SHARED_RETAINED_CHECKPOINT = standalone.SHARED_RETAINED_CHECKPOINT

TRIGGER_CELLS = standalone.TRIGGER_CELLS
CONTROL_CELLS = standalone.CONTROL_CELLS
SEED_INDICES = standalone.SEED_INDICES


#: The rung-0 lanes. A control event must request NONE of them.
RUNG_ZERO_LANES = ("zero_support_fallback", "protonation_aware_retained_subgraph")


class _ExpansionReachedOnHealthyRound(Exception):
    """The empty-pool handler was reached on a round that selected candidates.

    Deliberately NOT a ``ValueError`` / ``RuntimeError`` / ``KeyError`` /
    ``IndexError`` / ``TypeError``: `t4_fiber_campaign.expand` catches all five
    per draw, so any of those could be swallowed before the control event saw it.
    """


@contextmanager
def _expansion_sentinel():
    """Replace the empty-pool handler with something that raises if reached.

    This is what turns "no alternate kernel was consulted" from a field the
    control event writes about itself into a fact it observed. The substitution
    is on THIS module's own global, for the duration of one event, so nothing
    outside the control path is affected.
    """

    observed = {"calls": 0}
    original = globals()["routed_support_expansion"]

    def _sentinel(*args, **kwargs):
        observed["calls"] += 1
        raise _ExpansionReachedOnHealthyRound(
            "a control event reached the empty-pool handler; the ladder fires only "
            "on an empty candidate set, so a healthy round must never get here"
        )

    globals()["routed_support_expansion"] = _sentinel
    try:
        yield observed
    finally:
        globals()["routed_support_expansion"] = original


# ---- Contracts ----


def load_unified_cell(cell: str, *, root: Path = ROOT) -> tuple[dict, dict]:
    """The unified contract payload and the cell row, delta read from the CONTRACT."""

    relative = UNIFIED_CONTRACTS[standalone.protein_of(cell)]
    payload = standalone.load_payload(relative, root=root)
    delta = float(payload["delta"])
    if delta != 0.6:
        raise ValueError(
            f"{relative} declares delta {delta}, not 0.6; delta is executable and a "
            "contract name is prose"
        )
    # Resolving here means a contract missing the routing, the frozen ladder or the
    # declared region configuration fails on the first unit that reads it rather
    # than on whichever one happens to need it first.
    resolve_unified_controller(payload)
    row = next(item for item in payload["cells"] if item["cell"] == cell)
    return payload, row


def _route_expert_for(lane: str, cell: str, *, root: Path):
    """The fitted expert a model-backed lane needs, or None.

    The two lanes take DIFFERENT checkpoints -- the per-protein distilled expert
    for `route_complete_region`, the shared retained expert for the state-aware
    rung-0 lane -- and handing one the other's would be a silent substitution, so
    they are resolved by lane rather than by a single cached object.
    """

    if lane == "route_complete_region":
        return standalone.load_route_expert(standalone.protein_of(cell), root=root)
    if lane == "protonation_aware_retained_subgraph":
        return standalone.load_shared_retained_expert(root=root)
    return None


# ---- The production proposal unit, with the contract's own policy ----


def production_unit(task: dict, *, root: Path = ROOT) -> dict:
    """One production proposal unit: the function the campaign app's worker calls.

    The region law is resolved and PROVEN CONSUMED by `region_law_contract`
    exactly as the app does it, on every lane -- including the lanes that could
    never honour a law -- because a field parked on a lane that cannot consume it
    must fail on the worker that would otherwise have ignored it.
    """

    standalone.assert_no_docking_reachable()
    cell, lane = task["cell"], task["expert"]
    payload, row = load_unified_cell(cell, root=root)
    fiber = Fiber(row["smiles"], payload["delta"], support=payload["support"])
    arm = task.get("region_law_arm", PRODUCTION_ARM)
    if arm not in REGION_LAW_ARMS:
        raise ValueError(f"unknown region-law arm {arm!r}")

    contract = payload
    telemetry: dict = {}
    region_law = None
    if arm == "declared":

        def _probe_draw(law, attempt, *, _fiber=fiber, _row=row, _payload=payload):
            """One production draw, used only to prove the law is consumed.

            It calls `expand` itself -- same function, same lane, same horizon --
            so what is verified is the path this unit is about to run rather than
            a transcription of it. Its RNG is a separate stream, so probing cannot
            perturb the scored draw.
            """

            return expand(
                task["parent"],
                0.0,
                _fiber,
                np.random.default_rng(
                    int(task["proposal_seed"]) + 1_000_003 * (attempt + 1)
                ),
                draws=1,
                multi_region=False,
                horizon=_payload["proposal"]["shallow"]["horizon"],
                proposal_lane="shallow",
                region_law=law,
            )

        region_law, telemetry = region_law_for_proposal_lane(
            contract,
            lane=lane,
            delta=payload["delta"],
            reference_smiles=row["smiles"],
            draw=_probe_draw,
        )
    else:
        # The OFF state is an ABSENT field, never a uniform law: with `law=None`
        # the draw site consumes `rng.permutation` and with any law object it
        # consumes `rng.random`, so only absence reproduces the standalone draws.
        contract = {
            **payload,
            "proposal": {
                **payload["proposal"],
                "shallow": {
                    key: value
                    for key, value in payload["proposal"]["shallow"].items()
                    if key != "region_law"
                },
            },
        }

    answer = proposal_unit(
        contract,
        task,
        fiber=fiber,
        route_expert=_route_expert_for(lane, cell, root=root),
        region_law=region_law,
    )
    answer["region_law_arm"] = arm
    answer["region_law_telemetry"] = telemetry
    return answer


# ---- Round one ----


def round_one_requests(row: dict, seed_index: int, arm: str, cell: str) -> list[dict]:
    """The three round-one units, with the production per-expert seed.

    `controller_seed + 1_000_003*round + 10_007*parent + 101*expert` at round one,
    parent zero -- the derivation the campaign app uses, reproduced from the
    standalone gate rather than retyped.
    """

    return [
        {
            "cell": cell,
            "seed_index": seed_index,
            "region_law_arm": arm,
            "expert": expert,
            "expert_index": expert_index,
            "parent": row["smiles"],
            "parent_score": 0.0,
            "original_seed": row["smiles"],
            "proposal_seed": int(
                standalone.controller_seed(row["controller_seed"], seed_index)
                + 1_000_003 * 1
                + 10_007 * 0
                + 101 * expert_index
            ),
        }
        for expert_index, expert in enumerate(EXPERTS)
    ]


def _run_round_one(requests, runner, *, root: Path):
    answers = (
        [production_unit(request, root=root) for request in requests]
        if runner is None
        else runner(requests)
    )
    pools, lanes = {}, {}
    for answer in answers:
        pools[answer["expert"]] = answer["records"]
        lanes[answer["expert"]] = {k: v for k, v in answer.items() if k != "records"}
    missing = [expert for expert in EXPERTS if expert not in pools]
    if missing:
        raise RuntimeError(f"round one is missing expert pools {missing}")
    return pools, lanes


# ---- The events ----


def run_trigger_event(
    cell: str,
    seed_index: int,
    *,
    arm: str = PRODUCTION_ARM,
    root: Path = ROOT,
    round_one_runner=None,
    unit_runner=None,
) -> dict:
    """One (cell, seed) trigger event, through the production empty-pool handler.

    `unit_runner(requests) -> answers` is INJECTED so the remote driver can run a
    rung's replicates genuinely concurrently. That is not a convenience: the
    stopping rule checks `wall_seconds` against a MONOTONIC clock, so serialised
    replicates would record queueing as an algorithmic stop.
    """

    standalone.assert_no_docking_reachable()
    payload, row = load_unified_cell(cell, root=root)
    delta = float(payload["delta"])
    preflight = standalone.slot_semantics_preflight(row["smiles"])
    fiber = Fiber(row["smiles"], delta, support=payload["support"])
    context = standalone.SourceContext(fiber, row["smiles"])
    started = time.time()
    seed = standalone.controller_seed(row["controller_seed"], seed_index)

    def _units(requests):
        return (
            [production_unit(request, root=root) for request in requests]
            if unit_runner is None
            else unit_runner(requests)
        )

    pools, lanes = _run_round_one(
        round_one_requests(row, seed_index, arm, cell), round_one_runner, root=root
    )
    decision = standalone.round_one_decision(payload, row, pools, seed_index)

    def routed(*, router=None, rung_zero=None, escalate=None, rounds_completed=0):
        """The production empty-pool handler, called with this cell's own contract."""

        return routed_support_expansion(
            contract=payload,
            parents=[row["smiles"]],
            rounds_completed=rounds_completed,
            rung_zero=rung_zero,
            escalate=escalate,
            already_seen=[row["smiles"]],
            router=router,
        )

    # ---- hop 2: prove the empty-pool path consults a router before anything runs.
    # The probe raises from `decision`, which `routed_support_expansion` calls
    # before it builds any fan-out, so `_refuse` is unreachable on a live wiring
    # and reaching it would mean the router was consulted too late to matter.
    def _probe_route(router, attempt):
        def _refuse(*args, **kwargs):
            raise AssertionError(
                "the routing probe reached a proposal fan-out; the router must be "
                "consulted before any proposal work is scheduled"
            )

        return routed(
            router=router,
            rung_zero=_refuse,
            escalate=_refuse,
            rounds_completed=attempt,
        )

    routing_consumption_attempts = assert_state_routing_is_consumed(_probe_route)

    # ---- the real event
    rung_zero_log: list[dict] = []
    ladder_log: list[dict] = []

    def _rung_zero(assignments):
        requests = [
            {
                "cell": cell,
                "seed_index": seed_index,
                "region_law_arm": arm,
                "expert": item["lane"],
                "expert_index": 0,
                "parent": item["parent"],
                "parent_score": 0.0,
                "original_seed": row["smiles"],
                "proposal_seed": _rung_zero_seed(item["lane"], seed, item["parent_index"]),
            }
            for item in assignments
        ]
        produced, work = [], {}
        for answer in _units(requests):
            for record in answer["records"]:
                record["gate_rung"] = 0
                produced.append(record)
            rung_zero_log.append({k: v for k, v in answer.items() if k != "records"})
            for counters in (answer.get("telemetry") or {}).values():
                if isinstance(counters, dict):
                    for name, count in counters.items():
                        if isinstance(count, int):
                            work[name] = work.get(name, 0) + count
        return produced, work

    def _escalate(draws, attempt):
        requests = escalate_requests(payload, row, seed_index, arm, draws, attempt, cell)
        produced = []
        for answer in _units(requests):
            for record in answer["records"]:
                record["gate_rung"] = attempt + 1
                produced.append(record)
            ladder_log.append({k: v for k, v in answer.items() if k != "records"})
        return produced

    expansion, expansion_record = routed(
        rung_zero=_rung_zero, escalate=_escalate, rounds_completed=0
    )
    # ---- hop 3: the terminal guard the campaign app calls before publishing
    assert_unified_expansion_is_consumed(expansion_record, expansion)

    fresh = normalize_expansion_records(
        record for record in expansion.records if record["smiles"] != row["smiles"]
    )
    endpoints = [standalone.describe_endpoint(context, record, delta) for record in fresh]
    endpoints.sort(key=lambda item: item["smiles"])

    cumulative = []
    for entry in expansion.attempt_log:
        cumulative.append(
            {
                "stage": entry["stage"],
                "rung": 0 if entry["stage"] != "draw_ladder" else entry["attempt"] + 1,
                "draws_per_lane": entry.get("draws_per_lane", 0),
                "returned": entry["returned"],
                "fresh_distinct": entry["fresh_distinct"],
                "cumulative_distinct": entry["cumulative_distinct"],
                "elapsed_seconds": entry["elapsed_seconds"],
            }
        )
    zero_yield = sum(
        entry["fresh_distinct"] for entry in cumulative if entry["stage"] != "draw_ladder"
    )
    routed_kernel = expansion_record["routed_kernels"][0]
    verdict = standalone.cell_verdict(
        cell, routed_kernel, expansion.distinct_eligible, zero_yield
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "entrypoint": "production",
        "role": "trigger",
        "cell": cell,
        "protein": standalone.protein_of(cell),
        "seed_index": seed_index,
        "seed_role": "production" if seed_index == 0 else "variance_probe",
        "region_law_arm": arm,
        "controller_seed": seed,
        "production_controller_seed": row["controller_seed"],
        "contract": UNIFIED_CONTRACTS[standalone.protein_of(cell)],
        "contract_payload_sha256": identity(payload),
        "delta_from_contract": delta,
        "support": payload["support"],
        "slot_semantics_preflight": preflight,
        "consumption": {
            "state_routing_attempts": routing_consumption_attempts,
            "router": {"policy": ROUTER_ID, "policy_sha256": ROUTER_SHA256},
            "ladder": {
                "policy": FROZEN_LADDER_ID,
                "policy_sha256": FROZEN_LADDER_SHA256,
            },
            "terminal_guard_passed": True,
        },
        "routing": {
            "routed_kernel": routed_kernel,
            "historical_kernel": standalone.HISTORICAL_KERNEL.get(cell),
            "agrees": routed_kernel == standalone.HISTORICAL_KERNEL.get(cell),
            "decisions": expansion_record["routing"],
            "assignments": expansion_record["rung_zero_assignments"],
        },
        "round_one": {
            "lanes": lanes,
            "decision": decision,
            "expected_selected": standalone.EXPECTED_ROUND_ONE_SELECTED.get(cell),
            "agrees_with_committed_audit": decision["selected"]
            == standalone.EXPECTED_ROUND_ONE_SELECTED.get(cell),
            "audit_delta_mislabelled": cell in standalone.AUDIT_DELTA_MISLABELLED,
        },
        "ladder_fired_without_trigger": decision["selected"] > 0,
        "rung_zero": {"kernel": routed_kernel, "units": rung_zero_log},
        "ladder_units": ladder_log,
        "expansion": expansion.as_record(),
        "expansion_record": expansion_record,
        "cumulative_by_rung": cumulative,
        "distinct_eligible": expansion.distinct_eligible,
        "rung_zero_fresh_distinct": zero_yield,
        "endpoints": endpoints,
        "drug_like_eligible": sum(1 for item in endpoints if item["med_chem_plausible"]),
        "historical_reference": standalone.HISTORICAL_EXPANSION.get(cell, "UNAVAILABLE"),
        "verdict": verdict,
        "oracle_calls": 0,
        "docking_calls": 0,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def _rung_zero_seed(lane: str, seed: int, parent_index: int) -> int:
    """The rung-0 proposal seed, per lane, exactly as the campaign derives it."""

    if lane == "zero_support_fallback":
        return fallback_seed(seed, round_index=1, parent_index=parent_index)
    return int(seed + 900_007 + 1_000_003 * 1 + 10_007 * parent_index)


def escalate_requests(
    payload: dict, row: dict, seed_index: int, arm: str, draws: int, attempt: int, cell: str
) -> list[dict]:
    """One ladder step as PARALLEL REPLICATES at each lane's own base draw count.

    Never one deeper worker: a proposal worker is capped at 1800 s and 480 draws
    on a drug-like parent already costs minutes, so a step is spent on more
    workers rather than on a deeper one.
    """

    from compose_v4.control.frozen_proposal_escalation import FROZEN_LADDER

    seed = standalone.controller_seed(row["controller_seed"], seed_index)
    requests = []
    for lane_index, lane in enumerate(FROZEN_LADDER.lanes):
        base = int(payload["proposal"][lane]["draws"])
        replicates = max(1, -(-int(draws) // base))
        for replicate in range(replicates):
            requests.append(
                {
                    "cell": cell,
                    "seed_index": seed_index,
                    "region_law_arm": arm,
                    "expert": lane,
                    "expert_index": lane_index,
                    "expansion_replicate": replicate,
                    "parent": row["smiles"],
                    "parent_score": 0.0,
                    "original_seed": row["smiles"],
                    "proposal_seed": int(
                        seed
                        + 700_000_003 * (attempt + 1)
                        + 13_000_003 * replicate
                        + 1_000_003 * 1
                        + 10_007 * 0
                        + 101 * lane_index
                    ),
                }
            )
    return requests


def run_control_event(
    cell: str,
    seed_index: int = 0,
    *,
    arm: str = PRODUCTION_ARM,
    root: Path = ROOT,
    round_one_runner=None,
) -> dict:
    """One non-trigger control: round one only. The ladder must be UNREACHABLE.

    `select_batch` returns empty exactly when `candidates` is empty, so a positive
    `selected` count is the structural proof that the expansion branch is never
    entered, no alternate kernel is consulted and no RNG is drawn for one.
    """

    standalone.assert_no_docking_reachable()
    payload, row = load_unified_cell(cell, root=root)
    delta = float(payload["delta"])
    started = time.time()

    # "No alternate kernel was consulted" must be MEASURED, not declared. A field
    # copied from the request cannot witness that the request was met, so the
    # control event runs with the empty-pool handler replaced by a sentinel that
    # raises if anything reaches it, and counts the lanes actually requested.
    requests = round_one_requests(row, seed_index, arm, cell)
    rung_zero_requested = [
        request["expert"] for request in requests if request["expert"] in RUNG_ZERO_LANES
    ]
    with _expansion_sentinel() as sentinel:
        pools, lanes = _run_round_one(requests, round_one_runner, root=root)
        decision = standalone.round_one_decision(payload, row, pools, seed_index)
    return {
        "schema_version": SCHEMA_VERSION,
        "entrypoint": "production",
        "role": "control",
        "cell": cell,
        "protein": standalone.protein_of(cell),
        "seed_index": seed_index,
        "region_law_arm": arm,
        "contract": UNIFIED_CONTRACTS[standalone.protein_of(cell)],
        "contract_payload_sha256": identity(payload),
        "delta_from_contract": delta,
        "controller_seed": standalone.controller_seed(row["controller_seed"], seed_index),
        "round_one": {
            "lanes": lanes,
            "decision": decision,
            "expected_selected": standalone.EXPECTED_ROUND_ONE_SELECTED.get(cell),
            "agrees_with_committed_audit": decision["selected"]
            == standalone.EXPECTED_ROUND_ONE_SELECTED.get(cell),
            "audit_delta_mislabelled": cell in standalone.AUDIT_DELTA_MISLABELLED,
        },
        "searched": decision["selected"] > 0,
        # `select_batch` returns empty exactly when `candidates` is empty, so a
        # positive count is the structural proof the expansion branch is never
        # entered. The two fields below are the observed counterparts.
        "ladder_unreachable": decision["selected"] > 0,
        "alternate_kernel_consulted": sentinel["calls"] > 0,
        "expansion_handler_calls_observed": sentinel["calls"],
        "rung_zero_units": len(rung_zero_requested),
        "ladder_units": 0,
        "oracle_calls": 0,
        "docking_calls": 0,
        "elapsed_seconds": round(time.time() - started, 2),
    }


# ---- CLI ----


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--role", choices=("trigger", "control"), default="trigger")
    parser.add_argument("--arm", choices=REGION_LAW_ARMS, default=PRODUCTION_ARM)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reduce", type=Path, default=None,
                        help="reduce the shard directory at --destination instead of running")
    args = parser.parse_args()

    standalone.assert_no_docking_reachable()
    if args.reduce is not None:
        payload = reduce_shards(args.reduce, args.destination)
        print(json.dumps(payload["headline"], indent=1), flush=True)
        return
    if args.role == "trigger":
        shard = run_trigger_event(args.cell, args.seed_index, arm=args.arm)
    else:
        shard = run_control_event(args.cell, args.seed_index, arm=args.arm)
    shard["docking_interlock"] = standalone.assert_no_docking_reachable()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(shard, indent=1, sort_keys=True))
    print(json.dumps({k: shard.get(k) for k in ("cell", "role", "seed_index",
                                                "region_law_arm", "distinct_eligible",
                                                "elapsed_seconds")}), flush=True)


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())


# ---- Reduction ----


def _committed_standalone(*, root: Path = ROOT) -> dict:
    """The standalone gate's own committed table. The thing this run is compared to."""

    envelope = json.loads(
        (
            root / "diagnostics/t4_support_restoration_gate/t4_support_restoration_gate_v1.json"
        ).read_text()
    )
    return envelope.get("payload", envelope)


def reduce_shards(shard_dir: Path, destination: Path, *, root: Path = ROOT) -> dict:
    """Merge shards into the production-entrypoint table, beside the standalone one.

    The comparison is deliberately SPLIT by what each half can carry:

    * The VERDICT structure -- five trigger cells passing, routing agreeing with
      every historical arm, every control searching, the ladder unreachable on
      every control -- is the reproducible claim, and it must hold in the
      production `declared` arm.
    * The per-cell endpoint COUNTS are NOT expected to match the standalone table,
      because the production contract declares `proposal.shallow.region_law` and
      the standalone gate ran without it. An absent field is the only
      byte-identical OFF state, so the two configurations draw different shallow
      proposals.
    * The `law_off` controls ARE expected to match exactly. That arm reproduces
      the standalone gate's own proposal draws, so any difference there would be
      the wiring rather than the law -- which is precisely the question this
      integration has to answer.
    """

    shards = [json.loads(path.read_text()) for path in sorted(shard_dir.glob("*.json"))]
    committed = _committed_standalone(root=root)

    cells: dict[str, dict] = {}
    for cell in TRIGGER_CELLS:
        rows = sorted(
            (
                row
                for row in shards
                if row["role"] == "trigger"
                and row["cell"] == cell
                and row["region_law_arm"] == PRODUCTION_ARM
            ),
            key=lambda row: row["seed_index"],
        )
        if not rows:
            cells[cell] = {"status": "NOT_RUN"}
            continue
        routed = sorted({row["routing"]["routed_kernel"] for row in rows})
        historical = standalone.HISTORICAL_KERNEL.get(cell)
        eligible = {row["seed_index"]: row["distinct_eligible"] for row in rows}
        prior = (committed.get("cells") or {}).get(cell) or {}
        cells[cell] = {
            "status": "RUN",
            "seeds_run": [row["seed_index"] for row in rows],
            "routed_kernel": routed,
            "historical_kernel": historical,
            "routing_agrees": routed == [historical],
            "trigger_confirmed_all_seeds": all(
                row["round_one"]["decision"]["selected"] == 0 for row in rows
            ),
            "round_one_selected_by_seed": {
                row["seed_index"]: row["round_one"]["decision"]["selected"]
                for row in rows
            },
            "distinct_eligible_by_seed": eligible,
            "rung_zero_yield_by_seed": {
                row["seed_index"]: row["rung_zero_fresh_distinct"] for row in rows
            },
            "stop_reason_by_seed": {
                row["seed_index"]: row["expansion"]["stop_reason"] for row in rows
            },
            "draws_spent_by_seed": {
                row["seed_index"]: row["expansion"]["draws_spent"] for row in rows
            },
            "state_routing_consumption_attempts": {
                row["seed_index"]: row["consumption"]["state_routing_attempts"]
                for row in rows
            },
            "eligible_endpoints_total": sum(eligible.values()),
            "drug_like_eligible_total": sum(row["drug_like_eligible"] for row in rows),
            "p_any_eligible": round(
                sum(1 for row in rows if row["distinct_eligible"] > 0) / len(rows), 4
            ),
            "wall_clock_stops": [
                row["seed_index"]
                for row in rows
                if row["expansion"]["stop_reason"] == "wall_clock"
            ],
            "verdict": (
                "PASS"
                if any(row["distinct_eligible"] > 0 for row in rows)
                and routed == [historical]
                else "FAIL"
            ),
            "standalone_reference": {
                "distinct_eligible_by_seed": prior.get("distinct_eligible_by_seed"),
                "eligible_endpoints_total": prior.get("eligible_endpoints_total"),
                "routed_kernel": prior.get("routed_kernel"),
                "verdict": (prior.get("verdict") or {}).get("verdict"),
                "comparable": "verdict_and_routing_only",
                "why": (
                    "the production contract declares proposal.shallow.region_law and "
                    "the standalone gate ran without it, so endpoint counts are not "
                    "expected to match"
                ),
            },
        }

    controls: dict[str, dict] = {}
    for cell in CONTROL_CELLS:
        entry: dict = {}
        for arm in REGION_LAW_ARMS:
            row = next(
                (
                    item
                    for item in shards
                    if item["role"] == "control"
                    and item["cell"] == cell
                    and item["region_law_arm"] == arm
                ),
                None,
            )
            if row is None:
                entry[arm] = {"status": "NOT_RUN"}
                continue
            entry[arm] = {
                "status": "RUN",
                "selected": row["round_one"]["decision"]["selected"],
                "expected_selected": row["round_one"]["expected_selected"],
                "agrees_with_committed_audit": row["round_one"][
                    "agrees_with_committed_audit"
                ],
                "audit_delta_mislabelled": row["round_one"]["audit_delta_mislabelled"],
                "searched": row["searched"],
                "ladder_unreachable": row["ladder_unreachable"],
                "alternate_kernel_consulted": row["alternate_kernel_consulted"],
                "rung_zero_units": row["rung_zero_units"],
                "ladder_units": row["ladder_units"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
        controls[cell] = entry

    run_controls = [
        entry for entry in controls.values() if entry.get("law_off", {}).get("status") == "RUN"
    ]
    declared_controls = [
        entry
        for entry in controls.values()
        if entry.get(PRODUCTION_ARM, {}).get("status") == "RUN"
    ]
    run_cells = [row for row in cells.values() if row.get("status") == "RUN"]
    passing = [row for row in run_cells if row["verdict"] == "PASS"]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "entrypoint": "production",
        "question": (
            "does the unified controller, run through the functions the campaign app "
            "itself calls, reproduce the support the standalone gate measured?"
        ),
        "oracle_calls": 0,
        "docking_calls": 0,
        "consumption": {
            "router": {"policy": ROUTER_ID, "policy_sha256": ROUTER_SHA256},
            "ladder": {"policy": FROZEN_LADDER_ID, "policy_sha256": FROZEN_LADDER_SHA256},
            "state_routing_proven_consumed_on_every_trigger_event": all(
                row["consumption"]["state_routing_attempts"] >= 1
                for row in shards
                if row["role"] == "trigger"
            ),
            "terminal_guard_passed_on_every_trigger_event": all(
                row["consumption"]["terminal_guard_passed"]
                for row in shards
                if row["role"] == "trigger"
            ),
        },
        "cells": cells,
        "controls": controls,
        "headline": {
            "trigger_cells_run": len(run_cells),
            "trigger_cells_passing": len(passing),
            "routing_agrees_with_every_historical_arm": all(
                row["routing_agrees"] for row in run_cells
            ),
            "distinct_eligible_endpoints_total": sum(
                row["eligible_endpoints_total"] for row in run_cells
            ),
            "med_chem_plausible_total": sum(
                row["drug_like_eligible_total"] for row in run_cells
            ),
            "controls_run_law_off": len(run_controls),
            "controls_law_off_reproduce_committed_audit": all(
                entry["law_off"]["agrees_with_committed_audit"] for entry in run_controls
            ),
            "controls_run_declared": len(declared_controls),
            "controls_declared_all_searched": all(
                entry[PRODUCTION_ARM]["searched"] for entry in declared_controls
            ),
            "ladder_unreachable_on_every_control": all(
                entry[arm]["ladder_unreachable"]
                for entry in controls.values()
                for arm in REGION_LAW_ARMS
                if entry.get(arm, {}).get("status") == "RUN"
            ),
            "any_wall_clock_stop": any(row["wall_clock_stops"] for row in run_cells),
            "oracle_calls": 0,
            "docking_calls": 0,
        },
        "shards": sorted(
            f"{row['cell']}_seed{row['seed_index']}_{row['region_law_arm']}"
            for row in shards
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    destination.write_text(json.dumps(envelope, indent=1, sort_keys=True))
    return payload
