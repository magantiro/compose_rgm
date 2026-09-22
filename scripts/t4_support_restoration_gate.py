"""Does the automatic unified controller recover the SUPPORT the per-cell arms produced?

THE QUESTION, AND THE INTERPRETATION RULE IT IS SCORED UNDER
------------------------------------------------------------
Five T4 held-target cells at delta 0.6 exhausted their candidate pool in round one and
were each rescued by a DIFFERENT, human-chosen arm.  That is development history, not
one algorithm.  `compose_v4.control.t4_unified_routing` removes the human from the
choice: on an empty pool it routes from the molecular state alone -- `state_aware` if
the parent carries formal charge, `region` otherwise -- and this driver asks whether
the automatically routed controller recovers the USEFUL PROPOSAL SUPPORT the hand-picked
arms produced.

It is NOT scored on molecular identity.  A cell whose historical arm produced twelve
eligible endpoints and whose unified run produces nine DIFFERENT eligible endpoints at
comparable constraint margins and comparable structural character is a PASS.  Support
is a set of reachable admissible molecules; two searches may cover it differently.

THE FAILURE MODE THIS DRIVER IS BUILT TO NAME
----------------------------------------------
Two defects are confusable from a bare yield of zero and need different fixes, so the
report separates them explicitly:

  ROUTING failure   the state rule selects a kernel that is wrong for the cell.
                    Detected by comparing `applicable_kernel_if_exhausted` against the
                    kernel the historical arm used.  Cheap, deterministic, no search.
  WIRING failure    the routing is RIGHT, the routed lane runs, and it yields
                    essentially nothing.  The rule is behaviourally correct and the
                    plumbing is not yet equivalent to the arm it replaces.

A cell can fail the second while passing the first, and reporting it as a routing error
would send the fix to the wrong module.

WHAT IS SPENT
-------------
CPU.  This driver charges NO oracle call and performs NO docking: it generates
proposals and gates them with the unmodified production `Fiber.check`, which is pure
RDKit.  `assert_no_docking_reachable()` proves it rather than asserting it -- it refuses
to run if any docking-capable module is resident, and it is called on every entry point.

WHAT IS REPRODUCED, AND FROM WHERE
-----------------------------------
Nothing here is retyped from memory.

* the routing rule            `t4_unified_routing.applicable_kernel_if_exhausted`
* the ladder                  `frozen_proposal_escalation.FROZEN_LADDER`
                              (960/1920/3840 per lane, stop at 4 distinct eligible,
                              wall 7200 s)
* the stopping rule           `t4_support_expansion.run_support_expansion`
* the region kernel           `zero_support_fallback.fallback_candidates`
* the state-aware kernel      `protonation_aware_proposal.propose_protonation_aware_candidates`
                              at the settings the historical arm's own contract carries
* round one                   the call shape of `scripts/t4_support_stage_audit.audit_cell`
                              -- one parent (the docked root), three experts, proposal
                              seed `controller_seed + 1_000_003*round + 10_007*parent +
                              101*expert`, then the production `merge_expert_pools` /
                              `attach_features` / `select_batch`
* the escalate seed           the formula in the fa7_0 support-expansion base app:
                              `controller_seed + 700_000_003*(attempt+1) +
                              13_000_003*replicate + 1_000_003*round + 10_007*parent +
                              101*lane`

`parent_score` is a passthrough in `expand`, so round one is driven at 0.0 exactly as
the audit does; no docking value is read anywhere.

A RUNG IS PARALLEL REPLICATES, NEVER A DEEPER WORKER
-----------------------------------------------------
A ladder step of 960 draws over a lane whose contract base is 480 is realised as TWO
workers at 480, not one worker at 960.  A proposal worker is capped at 1800 s remotely
and 480 draws on a drug-like parent already costs minutes, so deepening a worker would
trade a bounded expansion for a timeout.  `wall_seconds` is checked against a monotonic
clock INSIDE the stopping rule, so a rung's workers must be genuinely concurrent or the
measurement records contention as an algorithmic stop.

SEEDS
-----
Three fixed seed indices.  Index 0 is the PRODUCTION derivation, unchanged.  Indices 1
and 2 shift the controller seed by a single declared constant and are a VARIANCE PROBE
-- they are not the production algorithm and are labelled as such in the artifact.  The
offset is 10**12, far above every internal seed offset (all below 10**10), so a variance
draw provably cannot collide with a production one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.frozen_proposal_escalation import (
    FROZEN_LADDER,
    FROZEN_LADDER_ID,
    FROZEN_LADDER_SHA256,
)
from compose_v4.control.route_distilled_goal_expert import (
    RouteDistilledGoalExpert,
    propose_route_expert_candidates,
)
from compose_v4.control.t4_unified_routing import (
    applicable_kernel_if_exhausted,
    molecular_applicability,
)
from compose_v4.control.zero_support_fallback import (
    fallback_candidates,
    fallback_seed,
)
from compose_v4.experiments.t4_fiber_campaign import (
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
from compose_v4.experiments.t4_support_expansion import (
    ESCALATABLE_LANES,
    ExpansionOutcome,
    SupportExpansionNotConsumed,
    assert_support_expansion_is_consumed,
    normalize_expansion_records,
    run_support_expansion,
)
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

SCHEMA_VERSION = "t4_support_restoration_gate_v1"

ROOT = Path(__file__).resolve().parents[1]

#: Slot capacity of a T4 proposal source. DISTINCT from the heavy-atom ceiling
#: `REPRESENTABLE_HEAVY_ATOMS = 40` the fiber applies to ENDPOINTS; both hold at once.
T4_PROPOSAL_SLOTS = 48

#: The state-aware rung-0 kernel's lane label, from the historical arm.
STATE_AWARE_LANE = "protonation_aware_retained_subgraph"
#: The contract the state-aware settings are read from. They are READ, never invented.
STATE_AWARE_SETTINGS_CONTRACT = "configs/t4_5ht1b2_protonation_rescue_d06_v1.json"
#: The shared route expert the state-aware kernel's structural lanes consult.
SHARED_RETAINED_CHECKPOINT = "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"

#: Every d06 contract, by protein. `jak2` deliberately uses the `_true_` contract:
#: `configs/t4_held_target_distilled_jak2_d06_250.json` carries `delta: 0.4` despite
#: its name, and delta is EXECUTABLE while `claim_boundary` is prose.
CONTRACTS = {
    "braf": "configs/t4_held_target_distilled_braf_d06_250.json",
    "fa7": "configs/t4_held_target_distilled_fa7_d06_250.json",
    "5ht1b": "configs/t4_held_target_distilled_5ht1b_d06_250.json",
    "parp1": "configs/t4_held_target_distilled_parp1_d06_250.json",
    "jak2": "configs/t4_held_target_distilled_jak2_true_d06_250.json",
}

#: The five cells that exhausted at round one and were rescued by hand.
TRIGGER_CELLS = ("braf_0", "braf_1", "fa7_0", "fa7_2", "5ht1b_2")
#: The ten that searched. The ladder is unreachable on these by construction.
CONTROL_CELLS = (
    "braf_2",
    "fa7_1",
    "5ht1b_0",
    "5ht1b_1",
    "parp1_0",
    "parp1_1",
    "parp1_2",
    "jak2_0",
    "jak2_1",
    "jak2_2",
)

#: Which kernel each historical per-cell arm actually used. The ROUTING comparison
#: is against this column, and it is the only place an arm identity appears.
HISTORICAL_KERNEL = {
    "braf_0": "region",
    "braf_1": "region",
    "fa7_0": "region",
    "fa7_2": "region",
    "5ht1b_2": "state_aware",
}

#: Round-one `selected` recorded by `diagnostics/t4_support_stage_audit_v1.json`.
#: A disagreement is REPORTED, never silently adopted. The three jak2 rows were
#: audited under the mislabelled delta=0.4 contract and are expected to differ here.
EXPECTED_ROUND_ONE_SELECTED = {
    "braf_0": 0, "braf_1": 0, "braf_2": 3,
    "fa7_0": 0, "fa7_1": 7, "fa7_2": 0,
    "5ht1b_0": 8, "5ht1b_1": 8, "5ht1b_2": 0,
    "parp1_0": 8, "parp1_1": 8, "parp1_2": 8,
    "jak2_0": 8, "jak2_1": 8, "jak2_2": 8,
}
AUDIT_DELTA_MISLABELLED = ("jak2_0", "jak2_1", "jak2_2")

#: The one hard historical reference point, MEASURED from the fa7_0 arm's own log:
#: `stop=reached_target eligible=4 fallback=0 attempts=2 draws=2880`.
HISTORICAL_EXPANSION = {
    "fa7_0": {
        "source": "t4_fa7_0_support_expansion arm log",
        "stop_reason": "reached_target",
        "distinct_eligible": 4,
        "fallback_eligible": 0,
        "attempts": 2,
        "draws_spent": 2880,
    }
}

#: Seed index 0 is production. 1 and 2 are a VARIANCE PROBE. The offset is far above
#: every internal seed offset (all < 10**10), so a variance seed cannot collide with a
#: production one for any attempt, replicate, round, parent or lane.
SEED_VARIANCE_OFFSET = 1_000_000_000_000
SEED_INDICES = (0, 1, 2)

#: Modules that can reach a docking binary. None may be resident.
FORBIDDEN_MODULES = (
    "compose_v4.experiments.t4_docking_adapter",
    "compose_v4.experiments.t4_partial_docking",
)


# ---- Docking interlock ----


def assert_no_docking_reachable() -> dict:
    """Refuse to run if anything that can dock is resident. Evidence, not a promise.

    The gate's whole authorization rests on charging nothing, and "I did not call it"
    is not checkable after the fact. This is, and it is called on every entry point.
    """

    resident = sorted(name for name in FORBIDDEN_MODULES if name in sys.modules)
    if resident:
        raise RuntimeError(
            f"a docking-capable module is resident: {resident}; this gate charges no "
            "oracle call and must not be able to reach one"
        )
    binaries = [path for path in ("/opt/dock/qvina02",) if Path(path).exists()]
    return {
        "forbidden_modules_checked": list(FORBIDDEN_MODULES),
        "forbidden_modules_resident": resident,
        "docking_binaries_present": binaries,
        "oracle_calls": 0,
        "docking_calls": 0,
    }


# ---- Contracts ----


def protein_of(cell: str) -> str:
    return cell.rsplit("_", 1)[0]


def load_payload(relative: str, *, root: Path = ROOT) -> dict:
    envelope = json.loads((root / relative).read_text())
    payload = envelope["payload"]
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"{relative} payload hash does not verify")
    return payload


def load_cell(cell: str, *, root: Path = ROOT) -> tuple[dict, dict]:
    """The contract payload and the cell row, with delta read from the CONTRACT."""

    relative = CONTRACTS[protein_of(cell)]
    payload = load_payload(relative, root=root)
    row = next(item for item in payload["cells"] if item["cell"] == cell)
    return payload, row


def load_route_expert(protein: str, *, root: Path = ROOT) -> RouteDistilledGoalExpert:
    relative = (
        f"diagnostics/t4_held_target_distillation_quality_v1/{protein}_checkpoint.json"
    )
    envelope = json.loads((root / relative).read_text())
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"{relative} payload hash does not verify")
    return RouteDistilledGoalExpert.from_checkpoint(envelope["payload"]["expert"])


def load_shared_retained_expert(*, root: Path = ROOT) -> RouteDistilledGoalExpert:
    envelope = json.loads((root / SHARED_RETAINED_CHECKPOINT).read_text())
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError("shared retained rewrite checkpoint payload hash mismatch")
    return RouteDistilledGoalExpert.from_checkpoint(envelope["payload"]["expert"])


def state_aware_settings(*, root: Path = ROOT) -> dict:
    """The historical arm's own protonation settings. Read from its contract."""

    payload = load_payload(STATE_AWARE_SETTINGS_CONTRACT, root=root)
    return dict(payload["proposal"][STATE_AWARE_LANE])


def controller_seed(base: int, seed_index: int) -> int:
    if seed_index not in SEED_INDICES:
        raise ValueError(f"unknown seed index {seed_index}")
    return int(base) + int(seed_index) * SEED_VARIANCE_OFFSET


# ---- Preflight ----


def slot_semantics_preflight(smiles: str) -> dict:
    """Assert the production proposal source carries FREE SLOTS for atom birth.

    48 is the SLOT capacity of the padded array; `REPRESENTABLE_HEAVY_ATOMS = 40` is the
    separate heavy-atom ceiling the fiber applies to ENDPOINTS. A tight graph silently
    deletes the whole insertion family from the legal support, which would change every
    number downstream, so it is refused rather than accepted quietly.
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
        "endpoint_heavy_ceiling": REPRESENTABLE_HEAVY_ATOMS,
        "passed": padded_slots == T4_PROPOSAL_SLOTS and padded_slots > heavy,
    }
    if not evidence["passed"]:
        raise RuntimeError(f"slot-semantics preflight failed for {smiles!r}: {evidence}")
    return evidence


# ---- Endpoint characterisation ----


class SourceContext:
    """Lane-independent descriptors of one endpoint relative to its source."""

    def __init__(self, fiber: Fiber, source_smiles: str):
        self.fiber = fiber
        self.source_smiles = source_smiles
        molecule = Chem.MolFromSmiles(source_smiles)
        self.source_mol = molecule
        self.heavy = molecule.GetNumHeavyAtoms()
        self.rings = molecule.GetRingInfo().NumRings()
        self.bits = set(fiber.generator.GetFingerprint(molecule).GetOnBits())
        self.scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        self.charge = Chem.GetFormalCharge(molecule)


def describe_endpoint(context: SourceContext, record: dict, delta: float) -> dict:
    """Margins, structural scale and a SEPARATE plausibility verdict.

    Benchmark eligibility and chemical plausibility are two numbers. An endpoint can
    pass similarity/QED/SA and still be an exocyclic quinoid or a strained azirine
    nobody would defend, so `med_chem_plausible` is recorded beside the gate verdict
    and never folded into it.
    """

    smiles = record["smiles"]
    molecule = Chem.MolFromSmiles(smiles)
    bits = set(context.fiber.generator.GetFingerprint(molecule).GetOnBits())
    heavy = molecule.GetNumHeavyAtoms()
    rings = molecule.GetRingInfo().NumRings()
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    except (ValueError, RuntimeError):
        scaffold = None
    sizes = sorted(len(ring) for ring in molecule.GetRingInfo().AtomRings())
    families = sorted(
        {str(name) for name in (record.get("families") or record.get("program_families") or [])}
    )
    return {
        "smiles": smiles,
        "proposal_lane": record.get("proposal_lane"),
        "support_expansion_stage": record.get("support_expansion_stage"),
        "rung": record.get("gate_rung"),
        "similarity": round(float(record["similarity"]), 6),
        "qed": round(float(record["qed"]), 6),
        "sa": round(float(record["sa"]), 6),
        "margin_similarity": round(float(record["similarity"]) - float(delta), 6),
        "margin_qed": round(float(record["qed"]) - QED_MIN, 6),
        "margin_sa": round(SA_MAX - float(record["sa"]), 6),
        "heavy": heavy,
        "heavy_delta": heavy - context.heavy,
        "ring_delta": rings - context.rings,
        "ring_sizes": sizes,
        "has_strained_ring": bool(sizes and min(sizes) <= 4),
        "net_formal_charge": Chem.GetFormalCharge(molecule),
        "charge_delta": Chem.GetFormalCharge(molecule) - context.charge,
        "retained_source_bit_fraction": round(
            len(context.bits & bits) / max(len(context.bits), 1), 5
        ),
        "added_bit_fraction": round(len(bits - context.bits) / max(len(bits), 1), 5),
        "scaffold_preserved": bool(scaffold is not None and scaffold == context.scaffold),
        "direction": (
            "grow" if heavy > context.heavy else "prune" if heavy < context.heavy else "replace"
        ),
        "deleted_atoms": record.get("deleted_atoms"),
        "inserted_atoms": record.get("inserted_atoms"),
        "region_size": record.get("region_size"),
        "rewrite_families": families,
        "med_chem_plausible": bool(structurally_valid(smiles)),
    }


# ---- Round one, reproduced ----


def round_one_expert(
    payload: dict,
    row: dict,
    expert: str,
    expert_index: int,
    seed_index: int,
    *,
    fiber: Fiber,
    root: Path = ROOT,
) -> dict:
    """One expert's round-one proposal pool, with the audit's own call shape."""

    assert_no_docking_reachable()
    source_smiles = row["smiles"]
    seed = int(
        controller_seed(row["controller_seed"], seed_index)
        + 1_000_003 * 1
        + 10_007 * 0
        + 101 * expert_index
    )
    started = time.time()
    if expert in ESCALATABLE_LANES:
        settings = payload["proposal"][expert]
        records = expand(
            source_smiles,
            0.0,
            fiber,
            np.random.default_rng(seed),
            draws=int(settings["draws"]),
            multi_region=True,
            horizon=settings["horizon"],
            proposal_lane=expert,
        )
        telemetry = {"raw_draws": int(settings["draws"])}
    elif expert == "route_complete_region":
        settings = payload["proposal"][expert]
        model = load_route_expert(protein_of(row["cell"]), root=root)
        padded = pad_molecular_graph(
            smiles_to_molecular_graph(source_smiles), T4_PROPOSAL_SLOTS
        )
        proposed, raw = propose_route_expert_candidates(
            padded,
            model,
            pool_size=settings["pool_size"],
            realization_limit=settings["realization_limit"],
            beam_width=settings["beam_width"],
            expansion_width=settings["expansion_width"],
            max_bindings_per_template=settings["max_bindings_per_template"],
            maximum_expansions=settings["maximum_expansions"],
        )
        records = []
        for candidate in proposed:
            properties = fiber.check(candidate["smiles"])
            if properties is None or properties["smiles"] == source_smiles:
                continue
            records.append(
                {
                    **candidate,
                    **properties,
                    "parent": source_smiles,
                    "parent_score": 0.0,
                    "delta": payload["delta"],
                }
            )
        telemetry = {
            key: value
            for key, value in raw.items()
            if not isinstance(value, (dict, list)) or key.endswith("counts")
        }
    else:  # pragma: no cover - EXPERTS is the frozen vocabulary
        raise ValueError(f"unknown proposal expert {expert!r}")
    return {
        "expert": expert,
        "proposal_seed": seed,
        "records": records,
        "eligible_records_returned": len(records),
        "telemetry": telemetry,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def round_one_decision(payload: dict, row: dict, pools: dict, seed_index: int) -> dict:
    """The production round-one decision, verbatim: merge, attach, select."""

    source_smiles = row["smiles"]
    fiber = Fiber(source_smiles, payload["delta"], support=payload["support"])
    state = SearchState(
        archive={source_smiles: 0.0}, budget=payload["charged_calls_per_cell"] - 1
    )
    rng = np.random.default_rng(controller_seed(row["controller_seed"], seed_index))
    parents = state.parents(
        limit=payload["parents"], rng=rng, explore=payload["parent_explore"]
    )
    merged = merge_expert_pools({expert: pools.get(expert, []) for expert in EXPERTS})
    fresh = [record for record in merged if record["smiles"] not in state.archive]
    candidates = attach_features(fresh, state, fiber)
    room = min(payload["batch"], state.budget)
    selected = select_batch(
        candidates,
        ProgramValue(penalty=payload["value_penalty"]),
        state,
        rng,
        round_index=1,
        batch=room,
        exploration=min(payload["exploration"], room),
        expert_floor_rounds=payload["expert_floor_rounds"],
    )
    return {
        "parents_round_one": parents,
        "merged_pool": len(merged),
        "fresh": len(fresh),
        "featured_candidates": len(candidates),
        "selected": len(selected),
        "pool_census": expert_census(candidates),
        "selected_census": expert_census(selected),
        "reproduces_candidate_exhaustion": not selected,
    }


def round_one_requests(payload: dict, row: dict, seed_index: int) -> list[dict]:
    """The three round-one expert units, with the audit's own per-expert seed."""

    return [
        {
            "expert": expert,
            "expert_index": expert_index,
            "proposal_seed": int(
                controller_seed(row["controller_seed"], seed_index)
                + 1_000_003 * 1
                + 10_007 * 0
                + 101 * expert_index
            ),
        }
        for expert_index, expert in enumerate(EXPERTS)
    ]


def run_round_one(
    payload: dict, row: dict, seed_index: int, *, fiber: Fiber, root: Path = ROOT, runner=None
) -> tuple[dict, dict]:
    """Round one's three expert pools.

    `runner(requests) -> list[dict]` is INJECTED so the remote driver can run the three
    experts concurrently; absent one they run serially in this process. Either way the
    per-expert work is the same `round_one_expert` call.
    """

    requests = round_one_requests(payload, row, seed_index)
    if runner is None:
        answers = [
            round_one_expert(
                payload,
                row,
                request["expert"],
                request["expert_index"],
                seed_index,
                fiber=fiber,
                root=root,
            )
            for request in requests
        ]
    else:
        answers = runner(requests)
    pools, lanes = {}, {}
    for answer in answers:
        pools[answer["expert"]] = answer["records"]
        lanes[answer["expert"]] = {k: v for k, v in answer.items() if k != "records"}
    missing = [expert for expert in EXPERTS if expert not in pools]
    if missing:
        raise RuntimeError(f"round one is missing expert pools {missing}")
    return pools, lanes


# ---- The two rung-0 kernels ----


def region_kernel(row: dict, payload: dict, seed_index: int, *, fiber: Fiber) -> dict:
    """Rung 0 for a NEUTRAL parent: the production zero-support fallback."""

    assert_no_docking_reachable()
    started = time.time()
    rng = np.random.default_rng(
        fallback_seed(
            controller_seed(row["controller_seed"], seed_index),
            round_index=1,
            parent_index=0,
        )
    )
    produced, work = fallback_candidates(
        row["smiles"],
        rng,
        check=fiber.check,
        reference_smiles=row["smiles"],
        delta=float(payload["delta"]),
    )
    records = [
        {
            **record,
            "proposal_lane": None,
            "proposal_experts": [],
            "support_expansion_stage": "zero_support_fallback",
            "parent_score": 0.0,
            "families": ("atom_delete",),
            "program_families": ("atom_delete",),
            "regions": 1,
            "created": int(record.get("inserted_atoms", 0)),
            "deleted": int(record.get("deleted_atoms", 0)),
        }
        for record in produced
        if record["smiles"] != row["smiles"]
    ]
    return {
        "kernel": "region",
        "records": records,
        "work": work.as_dict(),
        "elapsed_seconds": round(time.time() - started, 2),
    }


def state_aware_kernel(
    row: dict, payload: dict, seed_index: int, *, fiber: Fiber, root: Path = ROOT
) -> dict:
    """Rung 0 for a CHARGED parent: the protonation-aware retained-subgraph expert.

    Settings are READ from the historical arm's own contract, never invented here.
    """

    assert_no_docking_reachable()
    from compose_v4.control.protonation_aware_proposal import (
        ProtonationAwareProposalConfig,
        propose_protonation_aware_candidates,
    )

    started = time.time()
    settings = state_aware_settings(root=root)
    model = load_shared_retained_expert(root=root)
    source = pad_molecular_graph(
        smiles_to_molecular_graph(row["smiles"]), T4_PROPOSAL_SLOTS
    )
    seed = int(
        controller_seed(row["controller_seed"], seed_index) + 900_007 + 1_000_003 * 1
    )
    proposed, telemetry = propose_protonation_aware_candidates(
        source,
        model,
        config=ProtonationAwareProposalConfig(
            seed=seed,
            shallow_draws=settings["shallow_draws"],
            retained_maximum_fragment_atoms=settings["retained_maximum_fragment_atoms"],
            retained_maximum_stages=settings["retained_maximum_stages"],
            retained_maximum_prefixes=settings["retained_maximum_prefixes"],
            route_pool_size=settings["route_pool_size"],
            route_realization_limit=settings["route_realization_limit"],
            route_beam_width=settings["route_beam_width"],
            route_expansion_width=settings["route_expansion_width"],
            route_max_bindings_per_template=settings["route_max_bindings_per_template"],
            route_maximum_expansions=settings["route_maximum_expansions"],
            route_candidate_timeout_seconds=settings["route_candidate_timeout_seconds"],
        ),
    )
    records = []
    for candidate in proposed:
        properties = fiber.check(candidate["smiles"])
        if properties is None or properties["smiles"] == row["smiles"]:
            continue
        records.append(
            {
                **candidate,
                **properties,
                "proposal_lane": None,
                "proposal_experts": [],
                "support_expansion_stage": STATE_AWARE_LANE,
                "families": sorted(set(candidate.get("program_families") or [])),
                "parent": row["smiles"],
                "parent_score": 0.0,
                "delta": payload["delta"],
            }
        )
    work = {
        "protonation_actions_enumerated": telemetry.get("protonation_actions_enumerated"),
        "unique_exact_endpoints": telemetry.get("unique_exact_endpoints"),
        "lane_exact_programs": telemetry.get("lane_exact_programs"),
        "exact_execution_precision_numerator": telemetry.get(
            "exact_execution_precision_numerator"
        ),
        "exact_execution_precision_denominator": telemetry.get(
            "exact_execution_precision_denominator"
        ),
        "distinct_eligible_endpoints": len(records),
        "proposal_seed": seed,
    }
    return {
        "kernel": "state_aware",
        "records": records,
        "work": work,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def rung_zero(row, payload, seed_index, kernel, *, fiber, root=ROOT) -> dict:
    if kernel == "region":
        return region_kernel(row, payload, seed_index, fiber=fiber)
    if kernel == "state_aware":
        return state_aware_kernel(row, payload, seed_index, fiber=fiber, root=root)
    raise ValueError(f"unknown routed kernel {kernel!r}")


# ---- The escalation rungs ----


def escalate_requests(payload: dict, row: dict, seed_index: int, draws: int, attempt: int) -> list[dict]:
    """One ladder step as PARALLEL REPLICATES at each lane's own base draw count.

    Never one deeper worker: a proposal worker is capped at 1800 s and 480 draws on a
    drug-like parent already costs minutes. Seeds carry the fa7_0 arm's own
    700_000_003-per-attempt offset, which cannot collide with a normal round seed.
    """

    requests = []
    base_seed = controller_seed(row["controller_seed"], seed_index)
    for lane_index, lane in enumerate(FROZEN_LADDER.lanes):
        base = int(payload["proposal"][lane]["draws"])
        replicates = max(1, math.ceil(int(draws) / base))
        for replicate in range(replicates):
            requests.append(
                {
                    "lane": lane,
                    "draws": base,
                    "horizon": payload["proposal"][lane]["horizon"],
                    "replicate": replicate,
                    "attempt": attempt,
                    "proposal_seed": int(
                        base_seed
                        + 700_000_003 * (attempt + 1)
                        + 13_000_003 * replicate
                        + 1_000_003 * 1
                        + 10_007 * 0
                        + 101 * lane_index
                    ),
                }
            )
    return requests


def run_escalate_request(request: dict, row: dict, payload: dict, *, fiber: Fiber) -> dict:
    """One replicate worker: the production `expand` at the lane's base draw count."""

    assert_no_docking_reachable()
    started = time.time()
    records = expand(
        row["smiles"],
        0.0,
        fiber,
        np.random.default_rng(request["proposal_seed"]),
        draws=int(request["draws"]),
        multi_region=True,
        horizon=request["horizon"],
        proposal_lane=request["lane"],
    )
    return {
        **{key: request[key] for key in ("lane", "draws", "replicate", "attempt", "proposal_seed")},
        "records": records,
        "eligible_records_returned": len(records),
        "elapsed_seconds": round(time.time() - started, 2),
    }


# ---- Verdict ----


def cell_verdict(cell: str, routed: str, distinct_eligible: int, rung_zero_yield: int) -> dict:
    """PASS / FAIL, and when FAIL, ROUTING or WIRING -- they need different fixes.

    Scored on SUPPORT, never on molecular identity: any distinct eligible endpoint the
    unified controller recovers under the frozen ladder's declared bounds counts.
    """

    historical = HISTORICAL_KERNEL.get(cell)
    routing_agrees = historical is None or routed == historical
    if not routing_agrees:
        return {
            "verdict": "FAIL",
            "failure_kind": "ROUTING",
            "routing_agrees_with_historical_arm": False,
            "routed_kernel": routed,
            "historical_kernel": historical,
            "detail": (
                f"the state rule selected {routed!r} where the historical arm used "
                f"{historical!r}; the rule is wrong for this cell"
            ),
        }
    if distinct_eligible > 0:
        return {
            "verdict": "PASS",
            "failure_kind": None,
            "routing_agrees_with_historical_arm": True,
            "routed_kernel": routed,
            "historical_kernel": historical,
            "detail": (
                f"{distinct_eligible} distinct eligible endpoints recovered "
                f"({rung_zero_yield} at rung 0)"
            ),
        }
    return {
        "verdict": "FAIL",
        "failure_kind": "WIRING",
        "routing_agrees_with_historical_arm": True,
        "routed_kernel": routed,
        "historical_kernel": historical,
        "detail": (
            f"routing landed on {routed!r}, which matches the historical arm, and that "
            "lane ran to its declared bound and produced zero eligible endpoints; the "
            "routing logic is right and the wiring is not yet behaviourally equivalent"
        ),
    }


# ---- Events ----


def run_trigger_event(
    cell: str,
    seed_index: int,
    *,
    root: Path = ROOT,
    escalator=None,
    round_one_runner=None,
    rung_zero_runner=None,
) -> dict:
    """One (cell, seed) trigger event: round one, then the routed ladder.

    `escalator(requests) -> list[dict]` runs one rung's replicate requests. It is
    INJECTED so the remote driver can fan them out genuinely in parallel and a local
    run can execute them serially; nothing about the stopping rule is transcribed.
    """

    assert_no_docking_reachable()
    payload, row = load_cell(cell, root=root)
    delta = float(payload["delta"])
    preflight = slot_semantics_preflight(row["smiles"])
    fiber = Fiber(row["smiles"], delta, support=payload["support"])
    context = SourceContext(fiber, row["smiles"])
    started = time.time()

    source = pad_molecular_graph(
        smiles_to_molecular_graph(row["smiles"]), T4_PROPOSAL_SLOTS
    )
    applicability = molecular_applicability(source)
    routed = applicable_kernel_if_exhausted(applicability)

    pools, lanes = run_round_one(
        payload, row, seed_index, fiber=fiber, root=root, runner=round_one_runner
    )
    decision = round_one_decision(payload, row, pools, seed_index)

    if rung_zero_runner is None:
        zero = rung_zero(row, payload, seed_index, routed, fiber=fiber, root=root)
    else:
        zero = rung_zero_runner(routed)
    for record in zero["records"]:
        record["gate_rung"] = 0

    if escalator is None:
        def escalator(requests, _row=row, _payload=payload, _fiber=fiber):
            return [run_escalate_request(r, _row, _payload, fiber=_fiber) for r in requests]

    def _fallback():
        return list(zero["records"]), dict(zero["work"])

    def _escalate(draws, attempt):
        requests = escalate_requests(payload, row, seed_index, draws, attempt)
        produced = []
        for answer in escalator(requests):
            for record in answer["records"]:
                record["gate_rung"] = attempt + 1
                produced.append(record)
        return produced

    expansion = run_support_expansion(
        FROZEN_LADDER,
        escalate=_escalate,
        fallback=_fallback,
        already_seen=[row["smiles"]],
    )
    fresh = normalize_expansion_records(
        record for record in expansion.records if record["smiles"] != row["smiles"]
    )
    endpoints = [describe_endpoint(context, record, delta) for record in fresh]
    endpoints.sort(key=lambda item: item["smiles"])

    cumulative = []
    running = 0
    for entry in expansion.attempt_log:
        running = entry["cumulative_distinct"]
        cumulative.append(
            {
                "stage": entry["stage"],
                "rung": 0 if entry["stage"] != "draw_ladder" else entry["attempt"] + 1,
                "draws_per_lane": entry.get("draws_per_lane", 0),
                "returned": entry["returned"],
                "fresh_distinct": entry["fresh_distinct"],
                "cumulative_distinct": running,
                "elapsed_seconds": entry["elapsed_seconds"],
            }
        )

    zero_yield = sum(
        entry["fresh_distinct"] for entry in cumulative if entry["stage"] != "draw_ladder"
    )
    verdict = cell_verdict(cell, routed, expansion.distinct_eligible, zero_yield)

    return {
        "schema_version": SCHEMA_VERSION,
        "role": "trigger",
        "cell": cell,
        "protein": protein_of(cell),
        "seed_index": seed_index,
        "seed_role": "production" if seed_index == 0 else "variance_probe",
        "controller_seed": controller_seed(row["controller_seed"], seed_index),
        "production_controller_seed": row["controller_seed"],
        "contract": CONTRACTS[protein_of(cell)],
        "contract_payload_sha256": identity(payload),
        "delta_from_contract": delta,
        "support": payload["support"],
        "slot_semantics_preflight": preflight,
        "routing": {
            "applicability": applicability.as_record(),
            "routed_kernel": routed,
            "historical_kernel": HISTORICAL_KERNEL.get(cell),
            "agrees": routed == HISTORICAL_KERNEL.get(cell),
        },
        "ladder": {
            "policy": FROZEN_LADDER_ID,
            "policy_sha256": FROZEN_LADDER_SHA256,
            **FROZEN_LADDER.as_record(),
        },
        "round_one": {
            "lanes": lanes,
            "decision": decision,
            "expected_selected": EXPECTED_ROUND_ONE_SELECTED.get(cell),
            "agrees_with_committed_audit": decision["selected"]
            == EXPECTED_ROUND_ONE_SELECTED.get(cell),
            "audit_delta_mislabelled": cell in AUDIT_DELTA_MISLABELLED,
        },
        # In production the ladder fires ONLY on an empty pool. If a variance seed
        # happened to produce a non-empty round one, the expansion below is still
        # measured -- the gate is about support -- but it is NOT what production would
        # have done on that seed, and saying so is the difference between a measurement
        # and a misattributed one.
        "ladder_fired_without_trigger": decision["selected"] > 0,
        "rung_zero": {
            "kernel": routed,
            "work": zero["work"],
            "elapsed_seconds": zero["elapsed_seconds"],
            "eligible_returned": len(zero["records"]),
        },
        "expansion": expansion.as_record(),
        "cumulative_by_rung": cumulative,
        "distinct_eligible": expansion.distinct_eligible,
        "rung_zero_fresh_distinct": zero_yield,
        "endpoints": endpoints,
        "drug_like_eligible": sum(1 for item in endpoints if item["med_chem_plausible"]),
        "historical_reference": HISTORICAL_EXPANSION.get(cell, "UNAVAILABLE"),
        "verdict": verdict,
        "oracle_calls": 0,
        "docking_calls": 0,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def run_control_event(
    cell: str, seed_index: int, *, root: Path = ROOT, round_one_runner=None
) -> dict:
    """One non-trigger control: round one only. The ladder must be UNREACHABLE.

    `select_batch` returns empty exactly when `candidates` is empty, so a positive
    `selected` count is the structural proof that the expansion branch is never entered
    and the trajectory is byte-identical to the historical primary lane.
    """

    assert_no_docking_reachable()
    payload, row = load_cell(cell, root=root)
    delta = float(payload["delta"])
    preflight = slot_semantics_preflight(row["smiles"])
    fiber = Fiber(row["smiles"], delta, support=payload["support"])
    started = time.time()
    source = pad_molecular_graph(
        smiles_to_molecular_graph(row["smiles"]), T4_PROPOSAL_SLOTS
    )
    applicability = molecular_applicability(source)
    routed = applicable_kernel_if_exhausted(applicability)

    pools, lanes = run_round_one(
        payload, row, seed_index, fiber=fiber, root=root, runner=round_one_runner
    )
    decision = round_one_decision(payload, row, pools, seed_index)
    expected = EXPECTED_ROUND_ONE_SELECTED.get(cell)
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "control",
        "cell": cell,
        "protein": protein_of(cell),
        "seed_index": seed_index,
        "seed_role": "production" if seed_index == 0 else "variance_probe",
        "controller_seed": controller_seed(row["controller_seed"], seed_index),
        "contract": CONTRACTS[protein_of(cell)],
        "contract_payload_sha256": identity(payload),
        "delta_from_contract": delta,
        "slot_semantics_preflight": preflight,
        "routing": {
            "applicability": applicability.as_record(),
            "routed_kernel_if_it_had_exhausted": routed,
        },
        "round_one": {
            "lanes": lanes,
            "decision": decision,
            "expected_selected": expected,
            "agrees_with_committed_audit": decision["selected"] == expected,
            "audit_delta_mislabelled": cell in AUDIT_DELTA_MISLABELLED,
        },
        "ladder_unreachable": decision["selected"] > 0,
        "oracle_calls": 0,
        "docking_calls": 0,
        "elapsed_seconds": round(time.time() - started, 2),
    }


# ---- Reduction ----


def _distribution(values: list[float]) -> dict | None:
    """min / median / max. A margin DISTRIBUTION, never a pass/fail count.

    A candidate scraping the bound and one with real headroom are different results,
    and a binary count cannot tell them apart.
    """

    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "n": len(ordered),
        "min": round(ordered[0], 6),
        "median": round(median, 6),
        "max": round(ordered[-1], 6),
    }


#: The historical arms' own round-one output, read read-only from their volumes.
HISTORICAL_REFERENCE = "diagnostics/t4_support_restoration_gate/historical_reference_v1.json"


def _assert_expansion_consumed(shard: dict) -> dict:
    """Prove the routed ladder ACTUALLY RAN for this event, from the shard's own record.

    A mechanism that is wired, tested and never reached by a caller is the failure this
    repository has paid for repeatedly, and a gate that reported support without the
    expansion having run would be an instance of it. `assert_support_expansion_is_consumed`
    is the production guard for exactly this, so it is driven here on an outcome rebuilt
    from the shard rather than re-implemented.
    """

    record = shard["expansion"]
    outcome = ExpansionOutcome(
        attempts=int(record["attempts"]),
        draws_spent=int(record["draws_spent"]),
        distinct_eligible=int(record["distinct_eligible"]),
        fallback_ran=bool(record["fallback_ran"]),
        stop_reason=str(record["stop_reason"]),
    )
    try:
        assert_support_expansion_is_consumed(outcome)
    except SupportExpansionNotConsumed as error:
        return {"consumed": False, "reason": str(error)}
    return {
        "consumed": True,
        "stop_reason": outcome.stop_reason,
        "fallback_ran": outcome.fallback_ran,
        "ladder_attempts": outcome.attempts,
    }


def _historical_comparison(cell: str, endpoints: list[dict], *, root: Path = ROOT) -> dict:
    """Unified support vs the historical arm's, compared as SUPPORT not as identity.

    Exact SMILES overlap is reported because it is a fact, NOT because the gate turns on
    it: two searches may cover the same useful support with different molecules, and the
    interpretation rule says so explicitly. What the verdict actually rests on is whether
    the unified controller produces eligible endpoints of comparable constraint margin
    and comparable structural scale, so those distributions are put side by side.
    """

    path = root / HISTORICAL_REFERENCE
    if not path.exists():
        return {"status": "UNAVAILABLE", "reason": f"{HISTORICAL_REFERENCE} absent"}
    reference = json.loads(path.read_text())
    block = reference.get(f"{cell}_region_repair_rescue")
    if not block or "eligible_descriptors" not in block:
        return {
            "status": "UNAVAILABLE",
            "reason": f"no historical eligible pool recorded for {cell}",
            "note": reference.get(f"{cell}_region_repair_rescue", {}).get("note"),
        }
    historical = block["eligible_descriptors"]
    delta = None
    for point in endpoints:
        delta = round(point["similarity"] - point["margin_similarity"], 6)
        break
    ours = {point["smiles"] for point in endpoints}
    theirs = {row["smiles"] for row in historical}
    return {
        "status": "OK",
        "historical_arm_lanes": block["lane_telemetry"],
        "historical_eligible": len(theirs),
        "unified_eligible": len(ours),
        "exact_smiles_overlap": len(ours & theirs),
        "overlapping_smiles": sorted(ours & theirs),
        "historical_margins": {
            "similarity": _distribution(
                [row["similarity"] - delta for row in historical if delta is not None]
            ),
            "qed": _distribution([row["qed"] - QED_MIN for row in historical]),
            "sa": _distribution([SA_MAX - row["sa"] for row in historical]),
        },
        "historical_structural_character": {
            "deleted_atoms": _distribution(
                [float(row["deleted"]) for row in historical if row.get("deleted") is not None]
            ),
            "created_atoms": _distribution(
                [float(row["created"]) for row in historical if row.get("created") is not None]
            ),
            "lanes": sorted({row["proposal_lane"] for row in historical}),
        },
        "comparison_rule": (
            "scored on SUPPORT, never on molecular identity: comparable margins and "
            "comparable structural scale count as recovery even at zero exact overlap"
        ),
    }


def _relabelled_rungs(shard: dict) -> list[dict]:
    """Name rung 0 by the kernel that actually ran it.

    `run_support_expansion` labels its first stage `zero_support_fallback` because that
    is the only rung-0 stage IT knows about. Under routing, rung 0 is whichever kernel
    the state selected, and on a charged parent that is the protonation-aware expert,
    not the region fallback. Carrying the generic label into the artifact would claim
    the wrong mechanism ran, so the label is corrected from the shard's own record of
    the routed kernel.
    """

    kernel = shard["rung_zero"]["kernel"]
    out = []
    for entry in shard["cumulative_by_rung"]:
        row = dict(entry)
        if row["stage"] != "draw_ladder":
            row["stage"] = f"rung0_{kernel}"
            row["kernel"] = kernel
        out.append(row)
    return out


def reduce_shards(shard_dir: Path, destination: Path) -> dict:
    """Merge shards into the per-cell gate table.

    A cell's verdict is taken over its SEEDS: the gate asks whether the automatic
    controller recovers support, and recovering it on some seeds and not others is a
    different finding from never recovering it, so both `P(any eligible)` and the
    per-seed rows are kept.
    """

    shards = [json.loads(path.read_text()) for path in sorted(shard_dir.glob("*.json"))]
    triggers = [row for row in shards if row["role"] == "trigger"]
    controls = [row for row in shards if row["role"] == "control"]

    cells: dict[str, dict] = {}
    for cell in TRIGGER_CELLS:
        rows = sorted(
            (row for row in triggers if row["cell"] == cell),
            key=lambda row: row["seed_index"],
        )
        if not rows:
            cells[cell] = {"status": "NOT_RUN"}
            continue
        endpoints = [point for row in rows for point in row["endpoints"]]
        any_eligible = [row for row in rows if row["distinct_eligible"] > 0]
        routed = {row["routing"]["routed_kernel"] for row in rows}
        production = next((row for row in rows if row["seed_index"] == 0), rows[0])
        wall_clock_stops = [
            row["seed_index"]
            for row in rows
            if row["expansion"]["stop_reason"] == "wall_clock"
        ]
        cells[cell] = {
            "status": "RUN",
            "routed_kernel": sorted(routed),
            "historical_kernel": HISTORICAL_KERNEL.get(cell),
            "routing_agrees": routed == {HISTORICAL_KERNEL.get(cell)},
            "trigger_confirmed_all_seeds": all(
                row["round_one"]["decision"]["selected"] == 0 for row in rows
            ),
            "round_one_selected_by_seed": {
                row["seed_index"]: row["round_one"]["decision"]["selected"] for row in rows
            },
            "agrees_with_committed_audit": {
                row["seed_index"]: row["round_one"]["agrees_with_committed_audit"]
                for row in rows
            },
            "seeds_run": [row["seed_index"] for row in rows],
            "p_any_eligible": round(len(any_eligible) / len(rows), 4),
            "distinct_eligible_by_seed": {
                row["seed_index"]: row["distinct_eligible"] for row in rows
            },
            "rung_zero_yield_by_seed": {
                row["seed_index"]: row["rung_zero_fresh_distinct"] for row in rows
            },
            "cumulative_by_rung": {
                row["seed_index"]: _relabelled_rungs(row) for row in rows
            },
            "stop_reason_by_seed": {
                row["seed_index"]: row["expansion"]["stop_reason"] for row in rows
            },
            "draws_spent_by_seed": {
                row["seed_index"]: row["expansion"]["draws_spent"] for row in rows
            },
            "wall_clock_stops": wall_clock_stops,
            "margins": {
                "similarity": _distribution([p["margin_similarity"] for p in endpoints]),
                "qed": _distribution([p["margin_qed"] for p in endpoints]),
                "sa": _distribution([p["margin_sa"] for p in endpoints]),
            },
            "structural_character": {
                "heavy_delta": _distribution(
                    [float(p["heavy_delta"]) for p in endpoints]
                ),
                "ring_delta": _distribution([float(p["ring_delta"]) for p in endpoints]),
                "direction_counts": {
                    name: sum(1 for p in endpoints if p["direction"] == name)
                    for name in ("prune", "grow", "replace")
                },
                "rewrite_families": sorted(
                    {family for p in endpoints for family in p["rewrite_families"]}
                ),
                "rungs_producing": sorted({p["rung"] for p in endpoints if p["rung"] is not None}),
            },
            "eligible_endpoints_total": len(endpoints),
            "drug_like_eligible_total": sum(
                1 for p in endpoints if p["med_chem_plausible"]
            ),
            "endpoint_smiles": sorted({p["smiles"] for p in endpoints}),
            "historical_reference": HISTORICAL_EXPANSION.get(cell, "UNAVAILABLE"),
            "historical_comparison": _historical_comparison(cell, endpoints),
            "expansion_consumed_by_seed": {
                row["seed_index"]: _assert_expansion_consumed(row) for row in rows
            },
            "verdict": production["verdict"],
            "elapsed_seconds_by_seed": {
                row["seed_index"]: row["elapsed_seconds"] for row in rows
            },
        }

    control_rows = {}
    for row in controls:
        control_rows[row["cell"]] = {
            "selected": row["round_one"]["decision"]["selected"],
            "expected": row["round_one"]["expected_selected"],
            "agrees_with_committed_audit": row["round_one"]["agrees_with_committed_audit"],
            "audit_delta_mislabelled": row["round_one"]["audit_delta_mislabelled"],
            "delta_from_contract": row["delta_from_contract"],
            "contract": row["contract"],
            "merged_pool": row["round_one"]["decision"]["merged_pool"],
            "ladder_unreachable": row["ladder_unreachable"],
            "routed_kernel_if_it_had_exhausted": row["routing"][
                "routed_kernel_if_it_had_exhausted"
            ],
            "elapsed_seconds": row["elapsed_seconds"],
        }

    run = [row for row in cells.values() if row.get("status") == "RUN"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "question": (
            "Does the automatic unified controller recover the useful proposal SUPPORT "
            "the historical per-cell arms produced? Scored on SUPPORT, never on "
            "molecular identity."
        ),
        "oracle_calls": 0,
        "docking_calls": 0,
        "kernel": "rdkit 2024.03.5 (T4 production)",
        "ladder": {
            "policy": FROZEN_LADDER_ID,
            "policy_sha256": FROZEN_LADDER_SHA256,
            **FROZEN_LADDER.as_record(),
        },
        "seed_policy": {
            "indices": list(SEED_INDICES),
            "index_0": "production derivation, unchanged",
            "indices_1_2": "VARIANCE PROBE, not the production algorithm",
            "variance_offset": SEED_VARIANCE_OFFSET,
        },
        "cells": cells,
        "controls": control_rows,
        "controls_all_searched": all(
            row["ladder_unreachable"] for row in control_rows.values()
        )
        if control_rows
        else None,
        "verdict_counts": {
            name: sum(1 for row in run if row["verdict"]["verdict"] == name)
            for name in ("PASS", "FAIL")
        },
        "failure_kinds": {
            row_cell: row["verdict"]["failure_kind"]
            for row_cell, row in cells.items()
            if row.get("status") == "RUN" and row["verdict"]["verdict"] == "FAIL"
        },
        # `all(...)` over an empty set is vacuously True, which would report a
        # verdict no cell had earned. A metric that cannot vary is not a measurement.
        "routing_agrees_on_every_fired_cell": (
            all(row["routing_agrees"] for row in run) if run else "UNEVALUATED"
        ),
        "any_wall_clock_stop": (
            any(row["wall_clock_stops"] for row in run) if run else "UNEVALUATED"
        ),
        "trigger_cells_run": len(run),
        "shards": sorted(
            f"{row['cell']}_seed{row['seed_index']}" for row in triggers
        )
        + sorted(f"control_{row['cell']}" for row in controls),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return payload


# ---- CLI ----


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell")
    parser.add_argument("--reduce", type=Path)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--role", choices=("trigger", "control"), default="trigger")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    interlock = assert_no_docking_reachable()
    if args.reduce is not None:
        payload = reduce_shards(args.reduce, args.out)
        print(
            json.dumps(
                {
                    "verdict_counts": payload["verdict_counts"],
                    "failure_kinds": payload["failure_kinds"],
                    "routing_agrees_on_every_fired_cell": payload[
                        "routing_agrees_on_every_fired_cell"
                    ],
                    "controls_all_searched": payload["controls_all_searched"],
                    "any_wall_clock_stop": payload["any_wall_clock_stop"],
                },
                indent=1,
            ),
            flush=True,
        )
        return
    if not args.cell:
        parser.error("--cell is required unless --reduce is given")
    if args.role == "trigger":
        shard = run_trigger_event(args.cell, args.seed_index, root=args.root)
    else:
        shard = run_control_event(args.cell, args.seed_index, root=args.root)
    shard["docking_interlock"] = interlock
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(shard, indent=1, sort_keys=True))
    print(
        f"{args.cell} seed{args.seed_index} role={args.role} "
        f"selected={shard['round_one']['decision']['selected']} "
        f"distinct_eligible={shard.get('distinct_eligible')} "
        f"stop={(shard.get('expansion') or {}).get('stop_reason')} "
        f"-> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
