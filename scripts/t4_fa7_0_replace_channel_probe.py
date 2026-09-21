"""Isolate the REPLACE half: conditional on the module firing, is an eligible endpoint drawn?

The reachability probe measures the whole proposal path.  This driver measures
the one channel the failure has been localised to -- `segment_replace`, which is
`select region -> delete -> construct a replacement at the retained interface`.

It drives the LIVE `compile_generic_module` with the production region law, so
both halves are the production ones, and puts every resulting endpoint through
the UNMODIFIED production `Fiber.check`.

WHAT IT SEPARATES
-----------------
`delete_ok` ...... the region draw produced a legal contracted state
`grow_ok` ........ a replacement was constructed and the module executed
`eligible` ....... the resulting endpoint passed the production gate

and, over the executed modules, the joint distribution of
`(inserted_atoms, elements)` -- the two quantities `segment_replace` currently
draws BLIND (`rng.integers(1, capacity+1)` and a uniform draw over C/N/O per
atom).  If eligible endpoints exist in this channel but at a vanishing rate, the
defect is that the completion is unconditioned, not that it is inexpressible.

ZERO ORACLE CALLS.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import compile_generic_module
from compose_v4.control.region_law_contract import (
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    resolve_region_law,
)
from compose_v4.experiments.t4_fiber_campaign import Fiber
from compose_v4.experiments.t4_matched_pilot import unseal

SCHEMA_VERSION = "t4_fa7_0_replace_channel_probe_v1"
PROPOSAL_SLOTS = 48


def run(contract: Path, *, cell_name: str, family: str, draws: int, law_on: bool, seed_offset: int) -> dict:
    payload = unseal(contract)
    cell = next(c for c in payload["cells"] if c["cell"] == cell_name)
    shaped = json.loads(json.dumps(payload))
    if law_on:
        shaped["proposal"][CONTRACT_LANE]["region_law"] = FREE_GATE_MARGIN_V1
    else:
        shaped["proposal"][CONTRACT_LANE].pop("region_law", None)
    law = resolve_region_law(shaped, delta=shaped["delta"], reference_smiles=cell["smiles"])

    fiber = Fiber(cell["smiles"], payload["delta"], support=payload["support"])
    source = pad_molecular_graph(smiles_to_molecular_graph(cell["smiles"]), PROPOSAL_SLOTS)
    source_heavy = int(source.n_real_atoms)
    rng = np.random.default_rng(int(cell["controller_seed"]) + seed_offset)

    executed = 0
    refused = 0
    eligible: dict[str, dict] = {}
    inserted_hist: Counter = Counter()
    element_hist: Counter = Counter()
    deleted_hist: Counter = Counter()
    near: list[dict] = []
    started = time.time()
    for _ in range(draws):
        try:
            state, meta = compile_generic_module(source, rng, family, region_law=law)
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            refused += 1
            continue
        executed += 1
        # `compile_generic_module` returns (state, meta); the module's own draw is
        # recorded under meta["parameters"]. Unpacking this wrongly is silent under
        # the broad except above, so the shape is asserted rather than assumed.
        if not isinstance(meta, dict) or "parameters" not in meta:
            raise ValueError(f"unexpected module meta shape: {type(meta)}")
        detail = meta["parameters"]
        if isinstance(detail, dict):
            if "inserted_atoms" in detail:
                inserted_hist[int(detail["inserted_atoms"])] += 1
            if detail.get("elements"):
                element_hist["".join(detail["elements"])] += 1
            if "deleted_atoms" in detail:
                deleted_hist[int(detail["deleted_atoms"])] += 1
        try:
            endpoint = molecular_graph_to_smiles(state)
        except (ValueError, KeyError, IndexError, TypeError, RuntimeError):
            continue
        if not endpoint:
            continue
        gate = fiber.check(endpoint)
        if gate is not None and gate["smiles"] not in eligible:
            eligible[gate["smiles"]] = {
                "smiles": gate["smiles"],
                "similarity": round(gate["similarity"], 6),
                "qed": round(gate["qed"], 6),
                "sa": round(gate["sa"], 4),
                "heavy": gate["heavy"],
                "heavy_delta": gate["heavy"] - source_heavy,
                "inserted_atoms": detail.get("inserted_atoms") if isinstance(detail, dict) else None,
                "elements": detail.get("elements") if isinstance(detail, dict) else None,
                "deleted_atoms": detail.get("deleted_atoms") if isinstance(detail, dict) else None,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": str(contract),
        "cell": cell_name,
        "family": family,
        "delta": payload["delta"],
        "support": payload["support"],
        "source_heavy_atoms": source_heavy,
        "region_law": FREE_GATE_MARGIN_V1 if law is not None else None,
        "draws": draws,
        "seed": int(cell["controller_seed"]) + seed_offset,
        "module_executed": executed,
        "module_refused": refused,
        "eligible_count": len(eligible),
        "eligible_per_executed_module": len(eligible) / executed if executed else 0.0,
        "eligible_endpoints": list(eligible.values()),
        "inserted_atoms_histogram": dict(sorted(inserted_hist.items())),
        "deleted_atoms_histogram": dict(sorted(deleted_hist.items())),
        "top_element_strings": dict(element_hist.most_common(12)),
        "seconds": round(time.time() - started, 1),
        "oracle_calls": 0,
        "docking_calls": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", required=True, type=Path)
    p.add_argument("--cell", default="fa7_0")
    p.add_argument("--family", default="segment_replace")
    p.add_argument("--draws", type=int, default=400)
    p.add_argument("--law", choices=("on", "off"), default="on")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    r = run(a.contract, cell_name=a.cell, family=a.family, draws=a.draws, law_on=a.law == "on", seed_offset=a.seed_offset)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1, sort_keys=True))
    print(
        f"{r['cell']} {r['family']} law={r['region_law']} draws={r['draws']} "
        f"executed={r['module_executed']} refused={r['module_refused']} "
        f"ELIGIBLE={r['eligible_count']} rate={r['eligible_per_executed_module']:.5f} {r['seconds']}s"
    )
    print(" inserted:", r["inserted_atoms_histogram"])
    print(" deleted :", r["deleted_atoms_histogram"])
    for e in r["eligible_endpoints"][:10]:
        print(f"   ELIG sim={e['similarity']:.4f} qed={e['qed']:.4f} sa={e['sa']:.3f} hd={e['heavy_delta']} {e['smiles']}")


if __name__ == "__main__":
    main()
