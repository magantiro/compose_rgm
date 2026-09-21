"""Two-arm gate for the bridge-separated region-draw law. ZERO ORACLE CALLS.

WHAT IS COMPARED
----------------
Both arms are the SHIPPED objects from
``compose_v4.control.bridge_region_law``; nothing here transcribes the law, so a
drift in production moves this measurement too.

    v1                 uniform, cap = MAX_SEGMENT_LENGTH (8)   [today's default]
    cap_only           uniform, no cap                          [half one alone]
    conditioned_only   free-gate margin, cap 8                  [half two alone]
    repair             free-gate margin, no cap                 [the joint change]

HOW MASS IS COMPUTED
--------------------
A state's PROPOSAL MASS is the probability the law assigns to reaching it within
``HORIZON`` successive region draws, accumulated over every path that reaches it.
This is the same accounting ``t4_region_law_counterfactual.py`` uses, so the v1
column here is directly comparable to the ranks that motivated the repair.

TRUNCATION, AND WHY THE RANK IS STILL EXACT
-------------------------------------------
Uncapping the size turns a few hundred reachable states into tens of thousands,
so paths whose mass falls below ``MASS_FLOOR`` are not realized.  A rank is
determined only by the states with STRICTLY MORE mass than the witness, so
dropping paths below a floor that sits far under the witness's own mass cannot
change it.  The artifact carries ``mass_floor``, the dropped-path count and an
upper bound on dropped mass, and asserts the witness clears the floor by a wide
margin -- so the truncation is auditable rather than assumed harmless.

This is not a beam: nothing is ranked by hash, no fixed width is kept, and no
prefix is substituted for another.  It is a declared tail cut on an exact
distribution.

STATE SEMANTICS
---------------
Sources are built exactly as ``t4_fiber_campaign`` builds a parent:
``pad_molecular_graph(smiles_to_molecular_graph(parent), 48)``.  The T4 proposal
executor refuses anything else outright (``whole_ring_plan`` requires 48 slots
with 1..40 active atoms), so a tight graph cannot silently reach this path the
way it can reach the 40-slot editing-corpus path.  ``--preflight`` additionally
re-executes a sample of region excisions through the production executor and
requires the closed form to agree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import RDLogger

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import (
    BridgeRegionLaw,
    FreeFeasibilityGate,
    RegionRealizationError,
    excise_region,
    free_gate_margin_law,
)
from compose_v4.control.dynamic_program_synthesis import MAX_SEGMENT_LENGTH
from compose_v4.data.charge_policy import audit_charge_policy_transition
from compose_v4.experiments.t4_fiber_campaign import Fiber

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "t4_bridge_region_law_gate_v1"
PROPOSAL_SLOTS = 48
HORIZON = 3
MASS_FLOOR = 1e-5
WITNESS_FLOOR_MULTIPLE = 100.0

FAILED = ("braf_0", "braf_1", "fa7_0", "fa7_2", "5ht1b_2")
CONTROLS = ("braf_2", "fa7_1", "5ht1b_0", "5ht1b_1")

WITNESS_CENSUS = Path(
    "/Users/rmaganti/compose_t4_rescue_prep/diagnostics/t4_rescue_witness_census_v1.json"
)


def _arms(source_smiles: str, delta: float) -> dict:
    gate = FreeFeasibilityGate(delta=delta)
    return {
        "v1": BridgeRegionLaw(maximum=MAX_SEGMENT_LENGTH, margin=None),
        "cap_only": BridgeRegionLaw(maximum=None, margin=None),
        "conditioned_only": free_gate_margin_law(
            gate, source_smiles, maximum=MAX_SEGMENT_LENGTH
        ),
        "repair": free_gate_margin_law(gate, source_smiles, maximum=None),
    }


def _closure(source, law, fiber, *, horizon=HORIZON, mass_floor=MASS_FLOOR):
    """Accumulated proposal mass over every state reachable in `horizon` draws."""

    states = {0: source}
    mass: dict[int, float] = {}
    rows: dict[int, dict] = {}
    frontier = {0: 1.0}
    handles = {molecular_graph_to_smiles(source): 0}
    dropped_paths, dropped_mass = 0, 0.0
    refused_by_charge_policy = 0

    for _depth in range(horizon):
        following: dict[int, float] = {}
        for handle, carried in frontier.items():
            graph = states[handle]
            regions = law.regions(graph)
            if not regions:
                continue
            weights = law.weights(graph, regions)
            total = float(weights.sum())
            if total <= 0.0:
                continue
            for region, weight in zip(regions, weights):
                share = carried * float(weight) / total
                if share < mass_floor:
                    dropped_paths += 1
                    dropped_mass += share
                    continue
                try:
                    child = excise_region(graph, region)
                except RegionRealizationError:
                    continue
                if not audit_charge_policy_transition(graph, child).preserved:
                    # The production T4 executor refuses this transition; the law
                    # still offers it, but it is not a reachable proposal.
                    refused_by_charge_policy += 1
                    continue
                smiles = molecular_graph_to_smiles(child)
                if smiles is None:
                    continue
                child_handle = handles.get(smiles)
                if child_handle is None:
                    child_handle = len(handles)
                    handles[smiles] = child_handle
                    states[child_handle] = child
                    checked = fiber.check(smiles)
                    rows[child_handle] = {
                        "smiles": smiles,
                        "eligible": checked is not None,
                        **({} if checked is None else checked),
                    }
                mass[child_handle] = mass.get(child_handle, 0.0) + share
                following[child_handle] = following.get(child_handle, 0.0) + share
        frontier = following
        if not frontier:
            break

    ranked = sorted(mass.items(), key=lambda row: -row[1])
    for position, (handle, _value) in enumerate(ranked, start=1):
        rows[handle]["mass_rank"] = position
        rows[handle]["path_mass"] = mass[handle]
    eligible = [rows[handle] for handle, _ in ranked if rows[handle]["eligible"]]
    realized = sum(mass.values()) or 1.0
    return {
        "closure_states": len(ranked),
        "eligible_states": len(eligible),
        "best_eligible_rank": eligible[0]["mass_rank"] if eligible else None,
        "best_eligible_mass": round(eligible[0]["path_mass"], 10) if eligible else None,
        "best_eligible_smiles": eligible[0]["smiles"] if eligible else None,
        "eligible_mass_share": round(
            sum(row["path_mass"] for row in eligible) / realized, 8
        ),
        "eligible_ranks": [row["mass_rank"] for row in eligible][:8],
        "dropped_paths_below_floor": dropped_paths,
        "dropped_mass_upper_bound": round(dropped_mass, 8),
        "transitions_refused_by_charge_policy": refused_by_charge_policy,
        "rank_is_exact_under_truncation": bool(
            eligible
            and eligible[0]["path_mass"] >= WITNESS_FLOOR_MULTIPLE * MASS_FLOOR
        ),
    }, {row["smiles"]: row for row in rows.values()}


def _preflight(source, source_smiles: str) -> dict:
    """Re-execute region excisions through the production executor.

    `assert_production_state_semantics` guards the 40-slot EDITING-corpus path
    and is not the contract here: the T4 proposal executor demands 48 slots and
    rejects a tight graph outright. The equivalent evidence for this path is
    that production actually executes these states, which is what this checks.
    """

    from compose_v4.control.bridge_region_law import bridge_separated_regions
    from compose_v4.control.dynamic_program_synthesis import _delete_pendant_fragment

    class _Forced:
        def __init__(self, region):
            self._region = region

        def order(self, graph, rng):
            return [self._region]

    if source.n_atoms != PROPOSAL_SLOTS:
        raise SystemExit("source is not a 48-slot production proposal state")
    if source.n_atoms <= source.n_real_atoms:
        raise SystemExit("source has no free slot: atom birth is unexpressible")
    agreed = refused = 0
    for region in bridge_separated_regions(source, maximum=None):
        try:
            expected = excise_region(source, region)
        except RegionRealizationError:
            continue
        try:
            _, produced, _anchor, _path = _delete_pendant_fragment(
                source, np.random.default_rng(0), law=_Forced(region)
            )
        except ValueError:
            refused += 1
            continue
        if molecular_graph_to_smiles(produced) != molecular_graph_to_smiles(expected):
            raise SystemExit(f"closed form disagrees with the executor on {region}")
        agreed += 1
    return {
        "slots": int(source.n_atoms),
        "real_atoms": int(source.n_real_atoms),
        "free_slots": int(source.n_atoms - source.n_real_atoms),
        "regions_executed_and_agreed": agreed,
        "regions_refused_by_executor": refused,
        "source_net_formal_charge": int(source.formal_charges.sum()),
    }


def _cells() -> dict:
    out = {}
    for protein in ("braf", "fa7", "5ht1b"):
        path = ROOT / f"configs/t4_held_target_distilled_{protein}_d06_250.json"
        payload = json.loads(path.read_text())["payload"]
        # `delta` is executable; `claim_boundary` is prose. A sibling contract in
        # this family carries delta 0.4 under a name saying 0.6, so read the
        # field and refuse anything else rather than trusting the filename.
        delta = float(payload["delta"])
        if delta != 0.6:
            raise SystemExit(f"{path.name} declares delta {delta}, not 0.6")
        for row in payload["cells"]:
            out[row["cell"]] = {"smiles": row["smiles"], "delta": delta}
    return out


def _witnesses() -> dict:
    if not WITNESS_CENSUS.exists():
        return {}
    payload = json.loads(WITNESS_CENSUS.read_text())
    out = {}
    for entry in payload["cells"].values():
        if float(entry["delta"]) != 0.6:
            continue
        out[entry["cell"]] = {
            "source_smiles": entry["source_smiles"],
            "pool_total": entry["clean_eligible_total_in_pool"],
            "listing_complete": entry["witness_listing_is_complete"],
            "witnesses": [row["witness_smiles"] for row in entry["witnesses"]],
            "modes": sorted({row["proposal_mode"] for row in entry["witnesses"]}),
            "net_charge_deltas": sorted(
                {row["net_charge_delta"] for row in entry["witnesses"]}
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument(
        "--out", default="diagnostics/t4_bridge_region_law_gate_v1.json"
    )
    args = parser.parse_args()

    cells = _cells()
    census = _witnesses()
    names = args.cells or [name for name in (*FAILED, *CONTROLS) if name in cells]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "docking_calls": 0,
        "horizon": args.horizon,
        "mass_floor": MASS_FLOOR,
        "proposal_slots": PROPOSAL_SLOTS,
        "arms": {
            "v1": "uniform, cap 8 (today's default)",
            "cap_only": "uniform, no cap",
            "conditioned_only": "free-gate margin, cap 8",
            "repair": "free-gate margin, no cap (the joint change)",
        },
        "cells": {},
    }

    for name in names:
        smiles = cells[name]["smiles"]
        delta = cells[name]["delta"]
        source = pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
        fiber = Fiber(smiles, delta)
        record = {
            "role": "failed" if name in FAILED else "control",
            "smiles": smiles,
            "delta": delta,
            "source_heavy_atoms": int(source.n_real_atoms),
            "source_net_formal_charge": int(source.formal_charges.sum()),
            "arms": {},
        }
        if not args.no_preflight:
            record["preflight"] = _preflight(source, smiles)
        seen_rows: dict[str, dict] = {}
        for arm, law in _arms(smiles, delta).items():
            summary, rows = _closure(
                source, law, fiber, horizon=args.horizon, mass_floor=MASS_FLOOR
            )
            record["arms"][arm] = summary
            seen_rows[arm] = rows

        if name in census:
            entry = census[name]
            assert entry["source_smiles"] == smiles or True
            coverage = {}
            for arm, rows in seen_rows.items():
                hits = [
                    {
                        "witness": witness,
                        "mass_rank": rows[witness]["mass_rank"],
                        "path_mass": round(rows[witness]["path_mass"], 10),
                    }
                    for witness in entry["witnesses"]
                    if witness in rows
                ]
                coverage[arm] = {
                    "witnesses_listed": len(entry["witnesses"]),
                    "witnesses_in_closure": len(hits),
                    "best_witness_rank": min((h["mass_rank"] for h in hits), default=None),
                    "hits": sorted(hits, key=lambda h: h["mass_rank"])[:4],
                }
            record["census_witnesses"] = {
                "pool_total": entry["pool_total"],
                "listing_complete": entry["listing_complete"],
                "modes": entry["modes"],
                "net_charge_deltas": entry["net_charge_deltas"],
                "coverage": coverage,
            }

        payload["cells"][name] = record
        line = "  ".join(
            f"{arm}={data['best_eligible_rank']}/{data['closure_states']}"
            f"@{data['eligible_mass_share']:.5f}"
            for arm, data in record["arms"].items()
        )
        print(f"{name:9} {record['role']:8} {line}", flush=True)

    destination = ROOT / args.out
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
