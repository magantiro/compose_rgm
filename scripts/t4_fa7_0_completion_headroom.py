"""Conditional on a region the PRODUCTION law draws, how many completions are eligible?

The replace channel is `select region -> delete -> construct replacement at the
retained interface`.  The region half is conditioned (the free-gate-margin law)
and the completion half is not: `segment_replace` draws
`rng.integers(1, capacity+1)` atoms and then a uniform C/N/O chain.

This driver holds the production region draw fixed and resamples ONLY the
completion, using the production `_grow_actions` and the production executor, and
puts every endpoint through the UNMODIFIED production `Fiber.check`.

WHAT IT DECIDES
---------------
For each drawn region it reports how many of `--completions` blind completions
are eligible.  That fraction `p` is the headroom of a completion law that draws
K candidates and re-ranks them: such a law finds an eligible completion with
probability about `1 - (1-p)^K` per firing, so

  * p == 0 on every region ....... conditioning the completion cannot help; the
                                   eligible endpoint is not a completion of any
                                   region this law draws, and the repair must be
                                   somewhere else;
  * p > 0 but small .............. the completion is expressible and merely
                                   improbable -- re-ranking a bounded candidate
                                   set is the smallest change that closes it.

Nothing here is FA7-specific: the region law, the growth primitive, the executor
and the gate are all the production ones, and the only quantity measured is the
gate margin the benchmark itself declares.

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
from compose_v4.control.dynamic_program_synthesis import (
    MAX_SEGMENT_LENGTH,
    _delete_pendant_fragment,
    _grow_actions,
)
from compose_v4.control.region_law_contract import (
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    resolve_region_law,
)
from compose_v4.experiments.t4_fiber_campaign import Fiber
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.whole_ring_plan import execute_program

SCHEMA_VERSION = "t4_fa7_0_completion_headroom_v1"
PROPOSAL_SLOTS = 48


def run(
    contract: Path,
    *,
    cell_name: str,
    regions: int,
    completions: int,
    law_on: bool,
    seed_offset: int,
) -> dict:
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

    rows = []
    eligible_total: dict[str, dict] = {}
    length_hist: Counter = Counter()
    started = time.time()
    for _ in range(regions):
        try:
            delete_actions, contracted, anchor, path = _delete_pendant_fragment(
                source, rng, law=law
            )
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            continue
        capacity = min(MAX_SEGMENT_LENGTH, 40 - int(contracted.n_real_atoms))
        if capacity < 1:
            continue
        hits = 0
        tried = 0
        best = None
        for _ in range(completions):
            growth = int(rng.integers(1, capacity + 1))
            try:
                grow_actions, _, elements = _grow_actions(
                    contracted, rng, length=growth, elements=("C", "N", "O"), anchor=anchor
                )
                state, _ = execute_program(source, [*delete_actions, *grow_actions])
                endpoint = molecular_graph_to_smiles(state)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            tried += 1
            if not endpoint:
                continue
            gate = fiber.check(endpoint)
            if gate is None:
                continue
            hits += 1
            length_hist[growth] += 1
            record = {
                "smiles": gate["smiles"],
                "similarity": round(gate["similarity"], 6),
                "qed": round(gate["qed"], 6),
                "sa": round(gate["sa"], 4),
                "heavy": gate["heavy"],
                "heavy_delta": gate["heavy"] - source_heavy,
                "deleted_atoms": len(path),
                "inserted_atoms": growth,
                "elements": list(elements),
            }
            eligible_total.setdefault(gate["smiles"], record)
            if best is None or gate["qed"] > best["qed"]:
                best = record
        rows.append(
            {
                "deleted_atoms": len(path),
                "capacity": capacity,
                "completions_executed": tried,
                "eligible": hits,
                "eligible_fraction": hits / tried if tried else 0.0,
                "best": best,
            }
        )

    productive = [r for r in rows if r["eligible"] > 0]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": str(contract),
        "cell": cell_name,
        "delta": payload["delta"],
        "support": payload["support"],
        "source_heavy_atoms": source_heavy,
        "region_law": FREE_GATE_MARGIN_V1 if law is not None else None,
        "regions_drawn": len(rows),
        "completions_per_region": completions,
        "seed": int(cell["controller_seed"]) + seed_offset,
        "regions_with_an_eligible_completion": len(productive),
        "distinct_eligible_endpoints": len(eligible_total),
        "pooled_eligible_fraction": (
            sum(r["eligible"] for r in rows) / max(1, sum(r["completions_executed"] for r in rows))
        ),
        "eligible_inserted_atoms_histogram": dict(sorted(length_hist.items())),
        "eligible_endpoints": sorted(
            eligible_total.values(), key=lambda r: -r["qed"]
        )[:25],
        "productive_regions": sorted(
            productive, key=lambda r: -r["eligible_fraction"]
        )[:15],
        "seconds": round(time.time() - started, 1),
        "oracle_calls": 0,
        "docking_calls": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", required=True, type=Path)
    p.add_argument("--cell", default="fa7_0")
    p.add_argument("--regions", type=int, default=60)
    p.add_argument("--completions", type=int, default=64)
    p.add_argument("--law", choices=("on", "off"), default="on")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    r = run(
        a.contract,
        cell_name=a.cell,
        regions=a.regions,
        completions=a.completions,
        law_on=a.law == "on",
        seed_offset=a.seed_offset,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1, sort_keys=True))
    print(
        f"{r['cell']} law={r['region_law']} regions={r['regions_drawn']} "
        f"x{r['completions_per_region']} :: regions_with_eligible="
        f"{r['regions_with_an_eligible_completion']} distinct_eligible="
        f"{r['distinct_eligible_endpoints']} pooled_p={r['pooled_eligible_fraction']:.5f} "
        f"{r['seconds']}s"
    )
    print(" eligible inserted-atom histogram:", r["eligible_inserted_atoms_histogram"])
    for e in r["eligible_endpoints"][:10]:
        print(
            f"   ELIG sim={e['similarity']:.4f} qed={e['qed']:.4f} sa={e['sa']:.3f} "
            f"hd={e['heavy_delta']} del={e['deleted_atoms']} ins={e['inserted_atoms']}"
            f"{''.join(e['elements'])} {e['smiles']}"
        )


if __name__ == "__main__":
    main()
