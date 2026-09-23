"""Build and re-seal the five unified T4 controller contracts.

WHAT IT PRODUCES
----------------
`configs/t4_unified_controller_<protein>_d06_v1.json` for parp1, jak2, braf, fa7
and 5ht1b.  Each is the corresponding held-target delta=0.6 250-call contract --
its cells, its docking box and evaluator, its baselines, its budget -- with the
unified controller's three declarations added and every runtime hash recomputed
against THIS tree.

WHY A TOOL AND NOT FIVE HAND-EDITED FILES
------------------------------------------
The five contracts must differ in the protein and in NOTHING ELSE that the
controller reads, because the claim is one controller over the panel rather than
a family of five.  A builder makes that checkable: the shared blocks are
literally the same expressions, and `assert_declared_region_configuration`
refuses a payload in which any cell row carries its own proposal configuration.

It is re-runnable and idempotent: the output is a pure function of the source
contracts and the on-disk bytes of the pinned runtime inputs, serialized
canonically, so running it twice writes identical files.

WHAT IT DELIBERATELY DOES NOT DO
---------------------------------
It does not authorize a launch.  Every contract is written with
`status = FROZEN_PENDING_OWNER_LAUNCH_AUTHORIZATION` and both
`scored_launch_authorized` and `modal_launch_authorized` false, and
`t4_unified_controller_app._local_task` REFUSES to spawn unless the owner has
flipped them against a payload hash they name.  A scored panel is the owner's
decision; a sealer that could grant it would be manufacturing consent.

It also DROPS `resume_predecessor` and `legacy_resume` from the source payloads.
Those name checkpoints and app hashes belonging to a DIFFERENT controller in a
DIFFERENT output namespace; carrying them across would let `_resume_state`
restore state that this controller never produced.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.control.docking_value import identity
from compose_v4.control.frozen_proposal_escalation import (
    FROZEN_LADDER_ID,
    FROZEN_LADDER_SHA256,
    frozen_escalation_block,
)
from compose_v4.control.region_law_contract import FREE_GATE_MARGIN_V1
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_unified_controller import (
    REGION_CONFIGURATION_FIELD,
    REGION_CONFIGURATION_LANE,
    ROUTER_ID,
    ROUTER_SHA256,
    assert_declared_region_configuration,
    resolve_unified_controller,
    state_routing_block,
)
from compose_v4.experiments.t4_unified_proposal import STATE_AWARE_LANE

SCHEMA_VERSION = "t4_unified_controller_contract_v1"

#: The panel. `jak2` reads the `_true_` source DELIBERATELY: the plain
#: `t4_held_target_distilled_jak2_d06_250.json` carries `delta: 0.4` despite its
#: name, inherited from a partial copy of a delta=0.4 controller contract while
#: only `charged_calls_per_cell` was updated. `delta` is EXECUTABLE
#: (`Fiber(cell["smiles"], contract["delta"], ...)`) and `claim_boundary` is
#: prose, so the two must be diffed rather than trusted -- which is why every
#: emitted payload is asserted at `delta == 0.6` below.
SOURCES = {
    "parp1": "configs/t4_held_target_distilled_parp1_d06_250.json",
    "jak2": "configs/t4_held_target_distilled_jak2_true_d06_250.json",
    "braf": "configs/t4_held_target_distilled_braf_d06_250.json",
    "fa7": "configs/t4_held_target_distilled_fa7_d06_250.json",
    "5ht1b": "configs/t4_held_target_distilled_5ht1b_d06_250.json",
}

#: The state-aware rung-0 lane's settings are COPIED VERBATIM from the historical
#: arm that measured them, never invented here.
PROTONATION_SOURCE = "configs/t4_5ht1b2_protonation_rescue_d06_v1.json"

DELTA = 0.6

#: Runtime inputs every arm pins, independent of protein. This is the scoring
#: path: the engine, the two mechanism modules, the contract surfaces that make
#: them reachable, the shared proposal unit, the chemistry the proposal lanes
#: run, and the shared retained-rewrite expert the state-aware lane reads.
#:
#: A module on the scoring path that is not pinned is free to drift, so every
#: file this app BAKES is pinned here except the contract itself -- which is
#: covered by the strictly stronger `contract_payload_sha256` identity check.
SHARED_RUNTIME_INPUTS = (
    "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json",
    "docs/GENMOL_T4_SEEDS.json",
    "modal_apps/t4_unified_controller_app.py",
    "src/compose_v4/control/bridge_region_law.py",
    "src/compose_v4/control/complete_region_program.py",
    "src/compose_v4/control/dynamic_program_synthesis.py",
    "src/compose_v4/control/fiber_control.py",
    "src/compose_v4/control/frozen_proposal_escalation.py",
    "src/compose_v4/control/progressive_structured_sampler.py",
    "src/compose_v4/control/protonation_aware_proposal.py",
    "src/compose_v4/control/protonation_restate_program.py",
    "src/compose_v4/control/region_law_contract.py",
    "src/compose_v4/control/route_distilled_goal_expert.py",
    "src/compose_v4/control/structural_subgoal_policy.py",
    "src/compose_v4/control/structural_subgoal_realizer.py",
    "src/compose_v4/control/t4_unified_routing.py",
    "src/compose_v4/control/zero_support_fallback.py",
    "src/compose_v4/experiments/t4_docking_adapter.py",
    "src/compose_v4/experiments/t4_fiber_campaign.py",
    "src/compose_v4/experiments/t4_integrated_route_fiber.py",
    "src/compose_v4/experiments/t4_support_expansion.py",
    "src/compose_v4/experiments/t4_unified_controller.py",
    "src/compose_v4/experiments/t4_unified_proposal.py",
    # The endpoint gate and the executor. `Fiber.check` decides eligibility through
    # `med_chem_gate.is_valid`, and `kernel` / `operators` / `action_codec_v5` are
    # the exact execution every lane's endpoint is produced by, so a drift in any
    # of them changes which molecules the panel is allowed to return. The
    # protonation-rescue arm that contributed the state-aware lane pins the same
    # four; this contract inherits that pin along with the lane.
    "src/compose_v4/gates/med_chem_gate.py",
    "src/compose_v4/rewrite/action_codec_v5.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/operators.py",
)

#: Fields that name a DIFFERENT controller's checkpoints, app hashes or output
#: namespace. Carried across they would let `_resume_state` restore state this
#: controller never produced.
DROPPED_SOURCE_FIELDS = ("resume_predecessor", "legacy_resume")


def contract_path(protein: str) -> Path:
    return ROOT / f"configs/t4_unified_controller_{protein}_d06_v1.json"


def wrapper_path(protein: str) -> str:
    return f"modal_apps/t4_unified_controller_{protein}_app.py"


def route_checkpoint_path(protein: str) -> str:
    return f"diagnostics/t4_held_target_distillation_quality_v1/{protein}_checkpoint.json"


def runtime_inputs(protein: str) -> dict:
    """Every pinned path, hashed against THIS tree. Refuses a path that is absent."""

    relatives = sorted(
        {*SHARED_RUNTIME_INPUTS, wrapper_path(protein), route_checkpoint_path(protein)}
    )
    pinned = {}
    for relative in relatives:
        absolute = ROOT / relative
        if not absolute.is_file():
            raise FileNotFoundError(
                f"{relative} is pinned by the unified {protein} contract but does "
                "not exist in this tree; a pin that addresses nothing reads as "
                "verified and is the worst case, not a benign one"
            )
        pinned[relative] = sha256_file(absolute)
    return pinned


def protonation_lane_settings() -> dict:
    """The state-aware lane's settings, verbatim from the arm that measured them."""

    payload = unseal(ROOT / PROTONATION_SOURCE)
    block = payload["proposal"][STATE_AWARE_LANE]
    return copy.deepcopy(block)


def build_payload(protein: str) -> dict:
    source_relative = SOURCES[protein]
    source = unseal(ROOT / source_relative)
    source_identity = identity(source)
    payload = copy.deepcopy(source)

    # `delta` is executable and `claim_boundary` is prose. Assert the executable
    # one rather than reading the name of the file it came from.
    if payload.get("delta") != DELTA:
        raise ValueError(
            f"{source_relative} carries delta={payload.get('delta')!r}, not {DELTA}; "
            "the unified panel is the delta=0.6 arm and a contract whose prose and "
            "whose executable field disagree must not be propagated"
        )

    for field in DROPPED_SOURCE_FIELDS:
        payload.pop(field, None)

    payload["schema_version"] = SCHEMA_VERSION
    payload["state_routing"] = state_routing_block()
    payload["support_expansion"] = frozen_escalation_block()

    proposal = copy.deepcopy(payload["proposal"])
    # The declared production region configuration. It applies for ALL applicable
    # molecular states and NEVER for named cells, which is why no cell row may
    # carry its own `proposal` / `region_law` / `state_routing` key.
    proposal[REGION_CONFIGURATION_LANE][REGION_CONFIGURATION_FIELD] = FREE_GATE_MARGIN_V1
    proposal[STATE_AWARE_LANE] = protonation_lane_settings()
    payload["proposal"] = proposal

    # JSON int/float typing, not a controller choice: parp1 stored `1` and the other
    # four `1.0`, which `ProgramValue(penalty=...)` cannot tell apart. Normalized so
    # the panel-uniformity statement reports only divergences that are real.
    payload["value_penalty"] = float(payload["value_penalty"])

    payload["unified_controller"] = {
        "state_routing_policy": ROUTER_ID,
        "state_routing_policy_sha256": ROUTER_SHA256,
        "escalation_policy": FROZEN_LADDER_ID,
        "escalation_policy_sha256": FROZEN_LADDER_SHA256,
        "declared_region_law": FREE_GATE_MARGIN_V1,
        "declared_for_all_applicable_states": True,
        "per_cell_overrides": [],
        "rung_zero_route_expert": "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json",
        "round_one_route_expert": route_checkpoint_path(protein),
    }
    payload["frozen_from"] = {
        "source_contract": source_relative,
        "source_contract_payload_sha256": source_identity,
        "protonation_lane_settings_from": PROTONATION_SOURCE,
        "protonation_lane_settings_payload_sha256": identity(
            unseal(ROOT / PROTONATION_SOURCE)
        ),
        "carried_unchanged": [
            "cells",
            "delta",
            "support",
            "docking_box",
            "docking_seed",
            "evaluator_sha256",
            "charged_calls_per_cell",
            "total_charged_call_ceiling",
            "batch",
            "parents",
            "parent_explore",
            "exploration",
            "expert_floor_rounds",
            "value_penalty",
            "primary_baselines",
        ],
        "added_by_the_unified_controller": [
            "state_routing",
            "support_expansion",
            f"proposal.{REGION_CONFIGURATION_LANE}.{REGION_CONFIGURATION_FIELD}",
            f"proposal.{STATE_AWARE_LANE}",
            "unified_controller",
        ],
        "dropped_from_source": list(DROPPED_SOURCE_FIELDS),
    }
    payload["claim_boundary"] = (
        f"unified T4 controller on the held-target {protein} panel at delta 0.6 with "
        f"{payload['charged_calls_per_cell']} charged calls per cell: state routing "
        f"({ROUTER_ID}) selects the rung-0 proposal kernel from the molecular state, "
        f"and an exhausted round climbs the frozen target-agnostic ladder "
        f"({FROZEN_LADDER_ID}) before any cell may publish candidate exhaustion. This "
        "is a NAMED controller distinct from the frozen historical panel; its numbers "
        "must never be spliced into an unchanged-v1 table."
    )
    # The source prose still read "fixed 49-call ceiling per cell" while the
    # executable `charged_calls_per_cell` is 250. A contract whose prose and whose
    # executable field disagree is the exact shape of the jak2 delta defect, so the
    # prose is rewritten FROM the executable fields rather than carried across.
    payload["experimental_setting"] = (
        f"prospective three-cell {protein.upper()} delta={payload['delta']} panel under the "
        f"unified T4 controller; {payload['charged_calls_per_cell']} charged calls per cell, "
        f"{payload['total_charged_call_ceiling']} per arm; an exhausted round runs the routed "
        "frozen support expansion, which spends CPU and never an oracle call, before "
        "candidate exhaustion may be published. Candidate exhaustion and abstention remain "
        "valid outcomes."
    )
    payload["status"] = "FROZEN_PENDING_OWNER_LAUNCH_AUTHORIZATION"
    payload["scored_launch_authorized"] = False
    payload["modal_launch_authorized"] = False
    payload["launch_authority"] = (
        "The five-protein unified panel is an owner decision, not an agent one. "
        "`t4_unified_controller_app._local_task` refuses to spawn while either "
        "`scored_launch_authorized` or `modal_launch_authorized` is false, so the "
        "refusal is executable rather than a status string."
    )
    payload["runtime_inputs_sha256"] = runtime_inputs(protein)

    # Fail closed on the contract this tree will actually read back.
    resolve_unified_controller(payload)
    assert_declared_region_configuration(payload)
    return payload


#: The fields the ENGINE reads to decide what to propose and how to select. The
#: claim is one controller over the panel, so a divergence here is a divergence in
#: the controller itself -- as opposed to `cells`, `docking_box`, `evaluator_sha256`
#: or a baseline table, which are benchmark INPUTS and are expected to differ.
CONTROLLER_PARAMETERS = (
    "batch",
    "charged_calls_per_cell",
    "delta",
    "expert_floor_rounds",
    "exploration",
    "parent_explore",
    "parents",
    "proposal.anchored_replacement",
    "proposal.protonation_aware_retained_subgraph",
    "proposal.route_complete_region",
    "proposal.shallow",
    "state_routing",
    "support",
    "support_expansion",
    "value_penalty",
)


#: The `route_complete_region` keys `t4_unified_proposal.proposal_unit` passes
#: straight into `propose_route_expert_candidates`. Everything else in that block
#: (`training_split`, `training_scope`, `scale_balanced`) is a per-arm PROVENANCE
#: string that the runtime never reads, and grouping on the whole block would hide
#: the executable split behind five distinct provenance strings.
ROUTE_LANE_EXECUTABLE_KEYS = (
    "beam_width",
    "expansion_width",
    "max_bindings_per_template",
    "maximum_expansions",
    "pool_size",
    "realization_limit",
)


def _at(payload: dict, dotted: str):
    node = payload
    for part in dotted.split("."):
        node = node[part]
    return node


def panel_uniformity(payloads: dict) -> dict:
    """A DECLARED, hashed statement of where the five arms agree and where they do not.

    MEASURED, and it is the finding worth carrying: the five source contracts do
    NOT agree on `proposal.route_complete_region`. parp1 and braf declare
    beam_width 32 / expansion_width 24 / realization_limit 32 while jak2, fa7 and
    5ht1b declare 48 / 48 / 64, and `t4_unified_proposal.proposal_unit` passes
    every one of those straight into `propose_route_expert_candidates` -- so they
    are executable, not descriptive.

    That is inherited from the historical arms, not introduced here, and it is
    left UNRESOLVED on purpose: picking one of the two settings changes what the
    route lane searches on three proteins or on two, which is a scientific call
    for the owner rather than a sealing detail. Recording it in every payload is
    what stops it from being rediscovered after a scored run.
    """

    proteins = sorted(payloads)
    divergent = {}
    for dotted in CONTROLLER_PARAMETERS:
        values = {
            protein: json.dumps(_at(payloads[protein], dotted), sort_keys=True)
            for protein in proteins
        }
        if len(set(values.values())) == 1:
            continue
        groups = {}
        for protein, value in values.items():
            groups.setdefault(value, []).append(protein)
        divergent[dotted] = [
            {"arms": sorted(arms), "value": json.loads(value)}
            for value, arms in sorted(groups.items())
        ]
    uniform = [key for key in CONTROLLER_PARAMETERS if key not in divergent]
    route_groups: dict = {}
    for protein in proteins:
        lane = _at(payloads[protein], "proposal.route_complete_region")
        executable = json.dumps(
            {key: lane[key] for key in ROUTE_LANE_EXECUTABLE_KEYS}, sort_keys=True
        )
        route_groups.setdefault(executable, []).append(protein)
    return {
        "checked_controller_parameters": list(CONTROLLER_PARAMETERS),
        "uniform_across_the_panel": uniform,
        "divergent_across_the_panel": divergent,
        "one_controller": not divergent,
        "route_lane_executable_settings": [
            {"arms": sorted(arms), "settings": json.loads(value)}
            for value, arms in sorted(route_groups.items())
        ],
        "route_lane_executable_keys": list(ROUTE_LANE_EXECUTABLE_KEYS),
        "note": (
            "Fields NOT listed here -- cells, docking_box, docking_seed, "
            "evaluator_sha256, runtime_inputs_sha256, baseline tables and prose -- "
            "are benchmark inputs or per-arm identity and are EXPECTED to differ. "
            "Only the listed fields decide what the controller proposes and selects."
        ),
    }


def seal_panel() -> list[tuple[Path, str, int, str]]:
    """Build all five, compute the uniformity statement once, then write."""

    payloads = {protein: build_payload(protein) for protein in SOURCES}
    # Computed on the payloads BEFORE injection and injected identically into all
    # five, so the block cannot itself create the divergence it reports.
    uniformity = panel_uniformity(payloads)
    written = []
    for protein, payload in payloads.items():
        payload["panel_uniformity"] = copy.deepcopy(uniformity)
        resolve_unified_controller(payload)
        assert_declared_region_configuration(payload)
        envelope = {"payload": payload, "payload_sha256": identity(payload)}
        text = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
        destination = contract_path(protein)
        destination.parent.mkdir(parents=True, exist_ok=True)
        written.append(
            (
                destination,
                envelope["payload_sha256"],
                len(payload["runtime_inputs_sha256"]),
                text,
            )
        )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and report drift without writing anything",
    )
    arguments = parser.parse_args()

    drifted = []
    for destination, payload_sha256, pins, text in seal_panel():
        if arguments.check:
            actual = destination.read_text() if destination.exists() else ""
            status = "OK" if actual == text else "DRIFT"
            if status == "DRIFT":
                drifted.append(destination.name)
        else:
            destination.write_text(text)
            status = "SEALED"
        print(
            f"{status:6s} {destination.relative_to(ROOT)} "
            f"payload_sha256={payload_sha256} runtime_inputs={pins}"
        )
    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
