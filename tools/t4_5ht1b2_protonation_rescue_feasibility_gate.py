#!/usr/bin/env python3
"""Minimal decisive launch-feasibility gate for the 5HT1B seed-2 protonation rescue arms.

This gate answers one question at BOTH thresholds: does the protonation-aware proposal
expert, driven on the PRODUCTION path with the arm contract's own settings, yield several
clean eligible endpoints that survive the unmodified production endpoint gate?

Three properties separate it from the earlier zero-oracle proposal gate
(``tools/t4_protonation_aware_proposal_gate.py``), each deliberate:

  * That gate TRANSCRIBED the endpoint thresholds
    (``sim >= delta, qed >= 0.6, sa <= 4.0`` at its line 129).  A transcription omits
    ``structurally_valid`` -- the med-chem validity gate that removes radicals,
    hypervalent sulfur and iodine, and cumulenes -- and the heavy-atom and multi-fragment
    refusals.  This gate calls ``Fiber.check`` itself, unmodified and unwrapped, so the
    number it reports is the number the campaign would see.  Per the repository's own
    rule: instrument a gate by driving it, never by restating it.
  * It drives ``t4_shared_controller_scored_runtime.protonation_aware_records``, which is
    the function the scored round loop calls, rather than the raw proposer underneath it.
  * It runs at delta 0.6 AND delta 0.4, because delta is a contract-level field and the
    two rescue arms are two contracts.

Generality is asserted structurally, not asserted in prose: the admitted-site rule is
``(charge, implicit_h) in {(+1,1), (0,0)}`` on a real nitrogen carrying exactly three
single bonds to real atoms.  It reads state, never identity.  The predeclared contrast
roots are carried through unchanged -- a neutral tertiary amine exercising the REVERSE
transition, and a root whose protonation fiber is empty, which must abstain.

Zero oracle calls.  Nothing here docks, launches, or writes to any Modal volume.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert
from compose_v4.experiments.t4_fiber_campaign import QED_MIN, SA_MAX
from compose_v4.experiments.t4_shared_controller_scored_runtime import (
    protonation_aware_records,
)
from compose_v4.rewrite.operators import enumerate_atom_protonation_restates

SCHEMA_VERSION = "t4_5ht1b2_protonation_rescue_feasibility_gate_v1"
SETTINGS_KEY = "protonation_aware_retained_subgraph"
ROUTE_CHECKPOINT = Path("diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json")
SEED_REGISTRY = Path("docs/GENMOL_T4_SEEDS.json")

#: Predeclared in configs/t4_protonation_aware_proposal_gate_v1.json and carried
#: unchanged.  Index 8 is the arm's cell; 0 and 13 are the generality contrasts.
PRIMARY_INDEX = 8
CONTRAST_REVERSE_INDEX = 0
CONTRAST_ABSTENTION_INDEX = 13


def _read_envelope(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(payload):
        raise ValueError(f"input is not a valid self-hashed envelope: {path}")
    return payload


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _admitted_sites(smiles: str) -> list[dict]:
    """Report the narrow protonation fiber on a padded production source state."""

    source = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
    rows = []
    for action in enumerate_atom_protonation_restates(source):
        vertex = int(action.v)
        rows.append(
            {
                "vertex": vertex,
                "formal_charge": int(source.formal_charges[vertex]),
                "implicit_h": int(source.implicit_h_counts[vertex]),
                "target_state": str(action.target_state),
            }
        )
    return rows


def _chemistry_sanity(smiles: str) -> dict:
    """Disclose what the endpoint actually is, beyond pass/fail."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"parses": False}
    elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
    return {
        "parses": True,
        "elements": elements,
        "net_formal_charge": int(Chem.GetFormalCharge(mol)),
        "radical_electrons": int(
            sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
        ),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "rings": int(mol.GetRingInfo().NumRings()),
        "fragments": len(Chem.GetMolFrags(mol)),
        "exotic_elements": [e for e in elements if e not in ("C", "N", "O", "F", "S", "Cl", "Br", "I", "P")],
    }


def _evaluate_root(
    *,
    label: str,
    smiles: str,
    delta: float,
    support: str,
    settings: dict,
    seed_value: int,
    route_expert: RouteDistilledGoalExpert,
) -> dict:
    """Drive the PRODUCTION adapter; every admission decision is Fiber.check's."""

    sites = _admitted_sites(smiles)
    records, telemetry = protonation_aware_records(
        parent=smiles,
        parent_score=0.0,  # metadata only; never reaches generation. No oracle value used.
        original_seed=smiles,
        delta=float(delta),
        support=support,
        proposal_seed_value=int(seed_value),
        route_expert=route_expert,
        settings=settings,
    )
    endpoints = []
    for row in records:
        endpoints.append(
            {
                "smiles": row["smiles"],
                "similarity": round(float(row["similarity"]), 4),
                "qed": round(float(row["qed"]), 4),
                "sa": round(float(row["sa"]), 4),
                "heavy": int(row["heavy"]),
                "similarity_margin": round(float(row["similarity"]) - float(delta), 4),
                "qed_margin": round(float(row["qed"]) - QED_MIN, 4),
                "sa_margin": round(SA_MAX - float(row["sa"]), 4),
                "realized_primitives": int(row["realized_primitives"]),
                "chemistry": _chemistry_sanity(row["smiles"]),
            }
        )
    endpoints.sort(key=lambda e: (-e["similarity_margin"], e["smiles"]))
    unique = {e["smiles"] for e in endpoints}
    tight = [
        e["smiles"]
        for e in endpoints
        if min(e["similarity_margin"], e["qed_margin"], e["sa_margin"]) < 0.02
    ]
    return {
        "label": label,
        "delta": float(delta),
        "source_smiles": smiles,
        "admitted_protonation_sites": sites,
        "admitted_protonation_site_count": len(sites),
        "fiber_eligible_endpoints": len(endpoints),
        "fiber_eligible_unique": len(unique),
        "endpoints": endpoints,
        "endpoints_within_0_02_of_a_threshold": tight,
        "proposal_telemetry": {
            k: telemetry[k]
            for k in ("fiber_admitted_unique", "proposal_expert")
            if k in telemetry
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True, action="append",
                        help="rescue contract(s); repeat for each threshold arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = json.loads(SEED_REGISTRY.read_text())
    checkpoint = _read_envelope(ROUTE_CHECKPOINT)
    route_expert = RouteDistilledGoalExpert.from_checkpoint(checkpoint["expert"])

    arms = []
    consumption = []
    for contract_path in args.contract:
        envelope = json.loads(contract_path.read_text())
        payload = envelope["payload"]
        settings = dict(payload["proposal"][SETTINGS_KEY])
        delta = float(payload["delta"])
        support = str(payload["support"])
        cell = payload["cells"][0]
        seed_value = int(cell["controller_seed"])

        # -- consumption: a contract field that nothing reads is a documentary field. --
        # Re-run the same root with one setting perturbed; if the result is identical the
        # contract is not being consumed and the arm would be a no-op.
        probe_settings = dict(settings)
        probe_settings["shallow_draws"] = max(1, int(settings["shallow_draws"]) // 8)
        base = _evaluate_root(
            label="consumption_base", smiles=cell["smiles"], delta=delta, support=support,
            settings=settings, seed_value=seed_value, route_expert=route_expert,
        )
        probe = _evaluate_root(
            label="consumption_probe", smiles=cell["smiles"], delta=delta, support=support,
            settings=probe_settings, seed_value=seed_value, route_expert=route_expert,
        )
        consumption.append(
            {
                "contract": str(contract_path),
                "delta": delta,
                "setting_perturbed": f"proposal.{SETTINGS_KEY}.shallow_draws",
                "contract_value": int(settings["shallow_draws"]),
                "probe_value": int(probe_settings["shallow_draws"]),
                "endpoints_at_contract_value": base["fiber_eligible_unique"],
                "endpoints_at_probe_value": probe["fiber_eligible_unique"],
                "settings_are_consumed": base["fiber_eligible_unique"]
                != probe["fiber_eligible_unique"],
            }
        )

        results = [dict(base, label="primary_strict_support_failure")]
        for index, label in (
            (CONTRAST_REVERSE_INDEX, "neutral_tertiary_reverse_transition_contrast"),
            (CONTRAST_ABSTENTION_INDEX, "no_admitted_site_abstention_contrast"),
        ):
            results.append(
                _evaluate_root(
                    label=label,
                    smiles=registry[index]["smiles"],
                    delta=delta,
                    support=support,
                    settings=settings,
                    seed_value=seed_value,
                    route_expert=route_expert,
                )
            )
        arms.append(
            {
                "contract": str(contract_path),
                "contract_payload_sha256": envelope["payload_sha256"],
                "delta": delta,
                "support": support,
                "cell": cell["cell"],
                "roots": results,
            }
        )

    primary = {
        arm["delta"]: next(
            r for r in arm["roots"] if r["label"] == "primary_strict_support_failure"
        )
        for arm in arms
    }
    reverse = {
        arm["delta"]: next(
            r for r in arm["roots"]
            if r["label"] == "neutral_tertiary_reverse_transition_contrast"
        )
        for arm in arms
    }
    abstention = {
        arm["delta"]: next(
            r for r in arm["roots"]
            if r["label"] == "no_admitted_site_abstention_contrast"
        )
        for arm in arms
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence": "deterministic zero-oracle production-path launch feasibility",
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "endpoint_gate_authority": (
            "compose_v4.experiments.t4_fiber_campaign.Fiber.check, called unmodified "
            "through t4_shared_controller_scored_runtime.protonation_aware_records; "
            "thresholds are NOT restated by this gate"
        ),
        "generality_rule": {
            "admitted_site": (
                "real nitrogen with exactly three single bonds to real atoms whose "
                "(formal_charge, implicit_h) is (+1,1) or (0,0); the action swaps it to "
                "the other"
            ),
            "implementation": "compose_v4.rewrite.operators.enumerate_atom_protonation_restates",
            "reads_target_identity": False,
            "reads_similarity_qed_sa_or_reward": False,
            "note": (
                "generation receives (padded source graph, route expert, frozen config) "
                "only; delta and the seed SMILES enter afterwards, at endpoint admission"
            ),
        },
        "contract_consumption": consumption,
        "arms": arms,
        "aggregate": {
            "primary_eligible_unique_by_delta": {
                str(d): r["fiber_eligible_unique"] for d, r in primary.items()
            },
            "reverse_contrast_sites_by_delta": {
                str(d): r["admitted_protonation_site_count"] for d, r in reverse.items()
            },
            "abstention_contrast_sites_by_delta": {
                str(d): r["admitted_protonation_site_count"] for d, r in abstention.items()
            },
            "abstention_contrast_endpoints_by_delta": {
                str(d): r["fiber_eligible_unique"] for d, r in abstention.items()
            },
            "settings_consumed_every_arm": all(
                row["settings_are_consumed"] for row in consumption
            ),
        },
        "claim_boundary": (
            "Autonomous proposal support and free endpoint feasibility only, measured "
            "through the production endpoint gate. No docking reward, no FiberControl "
            "selection and no T4 benchmark performance is measured or implied."
        ),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "environment_caveat": (
                "the production image pins rdkit 2024.3.5 / numpy 1.26.4 / python 3.11; "
                "this gate reports the interpreter it actually ran on and endpoint "
                "margins are published so threshold-hugging cases are visible"
            ),
        },
        "code_revision": _git_revision(),
    }
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps(payload["aggregate"], indent=2))
    print("consumption:", json.dumps(consumption, indent=2))
    print("written:", args.output)


if __name__ == "__main__":
    main()
