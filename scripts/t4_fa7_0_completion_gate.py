"""The offline gate for the replace-completion law: fa7_0 AND cells that already work.

REQUIREMENT THIS EXISTS TO MEET
-------------------------------
A law that rescues one cell and degrades a working one is rejected.  That is not
hypothetical here: an earlier attempt to widen the region size cap alone
demoted a healthy control from rank 2 to rank 22, and only a control arm caught
it.  So this gate runs the TARGET and three cells that search today, in BOTH
arms, and reports eligible-per-draw for every one of them.

The two arms differ in exactly one contract field,
``proposal.shallow.completion_law``.  ABSENT is the OFF arm and is
byte-identical to the historical construction; the ON arm names the registered
law.  The region law is held at the SAME setting in both arms of a given cell,
so nothing but the completion law moves.

AMIDINE PRESERVATION
--------------------
For this target the objective is not merely an eligible molecule: a
pharmacophore-deleted endpoint can clear every free threshold while being
useless, which is exactly what the unconditioned path produced.  So eligible
endpoints are additionally reported as pharmacophore-preserving or not, by
SMARTS match against the groups the SOURCE carries.  That is a generic
comparison -- source substructure retained or lost -- not an FA7 rule, and it is
reported as a SECOND number beside eligibility rather than folded into it.

ZERO ORACLE CALLS.  Nothing docks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from compose_v4.control.completion_law_contract import (
    CONTRACT_FIELD as COMPLETION_FIELD,
)
from compose_v4.control.completion_law_contract import (
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    assert_completion_law_is_consumed,
    resolve_completion_law,
)
from compose_v4.control.region_law_contract import resolve_region_law
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
from compose_v4.experiments.t4_matched_pilot import unseal

SCHEMA_VERSION = "t4_fa7_0_completion_gate_v1"

#: Pharmacophore groups checked for retention. These are read off the SOURCE --
#: a group is only checked when the source has it -- so the rule is "did the
#: edit keep what the lead carried", which applies to any cell.
PHARMACOPHORE_SMARTS = {
    "amidine": "[CX3](=[NX2])[NX3]",
    "carboxamide": "[CX3](=O)[NX3]",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
    "carboxylate": "[CX3](=O)[OX1H0-,OX2H1]",
}


def _source_groups(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    out = {}
    for name, smarts in PHARMACOPHORE_SMARTS.items():
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            out[name] = smarts
    return out


def _preserves(smiles: str, groups: dict) -> bool:
    if not groups:
        return True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return all(
        mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in groups.values()
    )


def _arm(payload, cell, *, completion_on, draws, seed_offset) -> dict:
    shaped = json.loads(json.dumps(payload))
    lane = shaped["proposal"][CONTRACT_LANE]
    if completion_on:
        lane[COMPLETION_FIELD] = FREE_GATE_MARGIN_V1
    else:
        lane.pop(COMPLETION_FIELD, None)

    region_law = resolve_region_law(
        shaped, delta=shaped["delta"], reference_smiles=cell["smiles"]
    )
    completion_law = resolve_completion_law(
        shaped, delta=shaped["delta"], reference_smiles=cell["smiles"]
    )
    if completion_on and completion_law is None:
        raise ValueError("completion law requested but did not resolve")
    if not completion_on and completion_law is not None:
        raise ValueError("an absent completion field must resolve to no law")

    consumption = None
    if completion_law is not None:
        def _probe(law, attempt, _cell=cell, _payload=shaped):
            fiber = Fiber(_cell["smiles"], _payload["delta"], support=_payload["support"])
            return expand(
                _cell["smiles"],
                0.0,
                fiber,
                np.random.default_rng(int(_cell["controller_seed"]) + 7_919 * (attempt + 1)),
                draws=1,
                multi_region=False,
                horizon=_payload["proposal"][CONTRACT_LANE]["horizon"],
                proposal_lane=CONTRACT_LANE,
                region_law=region_law,
                completion_law=law,
            )

        consumption = assert_completion_law_is_consumed(_probe)

    fiber = Fiber(cell["smiles"], payload["delta"], support=payload["support"])
    groups = _source_groups(cell["smiles"])
    source_heavy = Chem.MolFromSmiles(cell["smiles"]).GetNumHeavyAtoms()
    seed = int(cell["controller_seed"]) + seed_offset
    started = time.time()
    records = expand(
        cell["smiles"],
        0.0,
        fiber,
        np.random.default_rng(seed),
        draws=draws,
        multi_region=True,
        horizon=lane["horizon"],
        proposal_lane=CONTRACT_LANE,
        region_law=region_law,
        completion_law=completion_law,
    )
    preserved = [r for r in records if _preserves(r["smiles"], groups)]
    return {
        "completion_law": FREE_GATE_MARGIN_V1 if completion_law is not None else None,
        "completion_law_candidates": (
            int(completion_law.candidates) if completion_law is not None else None
        ),
        "completion_law_consumption_attempts": consumption,
        "region_law": "free_gate_margin_v1" if region_law is not None else None,
        "draws": draws,
        "seed": seed,
        "eligible": len(records),
        "eligible_per_draw": len(records) / draws if draws else 0.0,
        "pharmacophore_preserved": len(preserved),
        "pharmacophore_groups_required": sorted(groups),
        "endpoints": [
            {
                "smiles": r["smiles"],
                "similarity": round(r["similarity"], 6),
                "qed": round(r["qed"], 6),
                "sa": round(r["sa"], 4),
                "heavy": r["heavy"],
                "heavy_delta": r["heavy"] - source_heavy,
                "pharmacophore_preserved": _preserves(r["smiles"], groups),
            }
            for r in sorted(records, key=lambda r: -r["qed"])
        ],
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", required=True, type=Path)
    p.add_argument("--cell", required=True)
    p.add_argument("--draws", type=int, default=480)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    payload = unseal(a.contract)
    cell = next(c for c in payload["cells"] if c["cell"] == a.cell)
    off = _arm(payload, cell, completion_on=False, draws=a.draws, seed_offset=a.seed_offset)
    on = _arm(payload, cell, completion_on=True, draws=a.draws, seed_offset=a.seed_offset)
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": str(a.contract),
        "cell": a.cell,
        "delta": payload["delta"],
        "support": payload["support"],
        "source_smiles": cell["smiles"],
        "off": off,
        "on": on,
        "delta_eligible_per_draw": on["eligible_per_draw"] - off["eligible_per_draw"],
        "oracle_calls": 0,
        "docking_calls": 0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(
        f"{a.cell:>9}  OFF elig={off['eligible']:<3} ({off['eligible_per_draw']:.5f}/draw) "
        f"pharm={off['pharmacophore_preserved']:<3} | "
        f"ON elig={on['eligible']:<3} ({on['eligible_per_draw']:.5f}/draw) "
        f"pharm={on['pharmacophore_preserved']:<3} K={on['completion_law_candidates']} "
        f"consumed@{on['completion_law_consumption_attempts']}  "
        f"[{off['seconds']}s/{on['seconds']}s]",
        flush=True,
    )
    for e in on["endpoints"][:8]:
        flag = "PHARM" if e["pharmacophore_preserved"] else "lost "
        print(
            f"    {flag} sim={e['similarity']:.4f} qed={e['qed']:.4f} sa={e['sa']:.3f} "
            f"hd={e['heavy_delta']:<4} {e['smiles']}"
        )


if __name__ == "__main__":
    main()
