"""Does `expand` gate the molecule the completion law optimized, or a different one?

THE QUESTION
------------
The completion law conditions the endpoint of the ``segment_replace`` MODULE.
But ``t4_fiber_campaign.expand`` does not gate that molecule.  It takes the
synthesized program, abstracts it with ``extract_structural_goal``, expands
``_variants`` over the subgoals, re-binds each subgoal to the source with
``attachment_bindings(...).assignments[0]``, and gates whatever
``instantiate_goal`` builds.

If that round trip does not reproduce the module endpoint, then conditioning the
module endpoint is partly wasted: the law improves a molecule the campaign never
scores.  That would explain a measured per-completion eligible fraction that
does not show up as eligible endpoints end to end.

WHAT IT MEASURES, on the production path and with zero oracle calls
------------------------------------------------------------------
For each draw, with the completion law installed exactly as the contract says:

  program_endpoint   the last state of the synthesized program's trace -- the
                     molecule the completion law actually ranked
  gated_endpoints    what ``expand`` offers to ``Fiber.check`` for the same draw

and reports how often ``program_endpoint`` is among ``gated_endpoints``, plus
the eligibility of each set.  ``recovered_fraction`` near 1 means the
indirection is faithful and the law's effect should propagate; near 0 means the
campaign scores something else.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.completion_law_contract import (
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    resolve_completion_law,
)
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.region_law_contract import resolve_region_law
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
from compose_v4.experiments.t4_matched_pilot import unseal

SCHEMA_VERSION = "t4_fa7_0_goal_indirection_probe_v1"
PROPOSAL_SLOTS = 48


def _canon(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def run(contract: Path, *, cell_name: str, draws: int, law_on: bool, seed_offset: int) -> dict:
    payload = unseal(contract)
    cell = next(c for c in payload["cells"] if c["cell"] == cell_name)
    shaped = json.loads(json.dumps(payload))
    if law_on:
        shaped["proposal"][CONTRACT_LANE]["completion_law"] = FREE_GATE_MARGIN_V1
    else:
        shaped["proposal"][CONTRACT_LANE].pop("completion_law", None)
    region_law = resolve_region_law(
        shaped, delta=shaped["delta"], reference_smiles=cell["smiles"]
    )
    completion_law = resolve_completion_law(
        shaped, delta=shaped["delta"], reference_smiles=cell["smiles"]
    )
    fiber = Fiber(cell["smiles"], payload["delta"], support=payload["support"])
    source = pad_molecular_graph(smiles_to_molecular_graph(cell["smiles"]), PROPOSAL_SLOTS)
    horizon = shaped["proposal"][CONTRACT_LANE]["horizon"]
    base_seed = int(cell["controller_seed"]) + seed_offset

    rows = []
    program_eligible = 0
    gated_eligible = 0
    recovered = 0
    considered = 0
    program_hits = []
    started = time.time()
    for index in range(draws):
        # Same seed drives both, so the program below is the program `expand`
        # synthesizes for this draw.
        try:
            _, _, _, trace, _meta = synthesize_dynamic_program(
                source,
                np.random.default_rng(base_seed + index),
                max_modules=horizon,
                region_law=region_law,
                completion_law=completion_law,
            )
            # `trace["states"]` holds ENCODED state dicts, not graphs; the
            # program's final molecule is carried directly as `trace["endpoint"]`.
            endpoint = trace["endpoint"]
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            continue
        program_key = _canon(endpoint)
        if program_key is None:
            continue
        considered += 1
        program_gate = fiber.check(program_key)
        if program_gate is not None:
            program_eligible += 1
            program_hits.append(
                {
                    "smiles": program_gate["smiles"],
                    "similarity": round(program_gate["similarity"], 6),
                    "qed": round(program_gate["qed"], 6),
                    "sa": round(program_gate["sa"], 4),
                    "heavy": program_gate["heavy"],
                }
            )
        records = expand(
            cell["smiles"],
            0.0,
            fiber,
            np.random.default_rng(base_seed + index),
            draws=1,
            multi_region=True,
            horizon=horizon,
            proposal_lane=CONTRACT_LANE,
            region_law=region_law,
            completion_law=completion_law,
        )
        gated = {r["smiles"] for r in records}
        gated_eligible += len(gated)
        if program_key in gated:
            recovered += 1
        rows.append(
            {
                "draw": index,
                "program_endpoint_eligible": program_gate is not None,
                "gated_eligible_count": len(gated),
                "program_endpoint_recovered_in_gated": program_key in gated,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": str(contract),
        "cell": cell_name,
        "completion_law": FREE_GATE_MARGIN_V1 if completion_law is not None else None,
        "region_law": "free_gate_margin_v1" if region_law is not None else None,
        "draws": draws,
        "programs_considered": considered,
        "program_endpoints_eligible": program_eligible,
        "program_endpoint_eligible_rate": program_eligible / considered if considered else 0.0,
        "gated_eligible_total": gated_eligible,
        # THE DISCRIMINATOR.
        "recovered_fraction": recovered / considered if considered else 0.0,
        "eligible_program_endpoints": program_hits[:20],
        "seconds": round(time.time() - started, 1),
        "oracle_calls": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", required=True, type=Path)
    p.add_argument("--cell", default="fa7_0")
    p.add_argument("--draws", type=int, default=120)
    p.add_argument("--law", choices=("on", "off"), default="on")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    r = run(a.contract, cell_name=a.cell, draws=a.draws, law_on=a.law == "on", seed_offset=a.seed_offset)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1, sort_keys=True))
    print(
        f"{r['cell']} law={r['completion_law']} programs={r['programs_considered']} "
        f"PROGRAM-endpoint eligible={r['program_endpoints_eligible']} "
        f"({r['program_endpoint_eligible_rate']:.4f}) "
        f"GATED eligible={r['gated_eligible_total']} "
        f"recovered_fraction={r['recovered_fraction']:.4f} {r['seconds']}s"
    )
    for e in r["eligible_program_endpoints"][:6]:
        print(f"   PROG-ELIG sim={e['similarity']:.4f} qed={e['qed']:.4f} sa={e['sa']:.3f} {e['smiles']}")


if __name__ == "__main__":
    main()
