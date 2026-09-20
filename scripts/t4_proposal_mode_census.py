"""Realized MODE distribution of v1's shallow proposal. DIAGNOSIS ONLY, zero oracle calls.

`t4_prune_closure_probe.py` gives the region-conditional mass: given that a program is
a sequence of bounded pendant deletions, how much of its law lands inside the feasible
region. This module supplies the other factor -- how often v1 draws such a program at
all -- by sampling `synthesize_dynamic_program` exactly as `t4_fiber_campaign.expand`
does (horizon 3, the contract's value) and recording the realized module families.

The product of the two factors is the per-draw probability that the shallow lane emits
the known-feasible witness, which is what turns "the pool did not contain it" into a
number.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "t4_proposal_mode_census_v1"
PROPOSAL_SLOTS = 48
HORIZON = 3

#: Families whose net effect is pure heavy-atom removal, i.e. the only families that
#: can realize a deletion-only witness. `segment_replace` deletes then regrows, so it
#: cannot; `functionalize`/`append_ring`/`fuse_ring`/`segment_grow` add atoms.
PRUNE_FAMILIES = frozenset({"substituent_delete", "segment_shrink"})


def census(smiles: str, seed: int, draws: int) -> dict:
    source = pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
    rng = np.random.default_rng(seed)
    family_counts: Counter[str] = Counter()
    first_module: Counter[str] = Counter()
    module_counts: Counter[int] = Counter()
    all_prune = 0
    all_prune_by_depth: Counter[int] = Counter()
    refused = 0
    for _ in range(draws):
        try:
            _, _, _, _, metadata = synthesize_dynamic_program(
                source, rng, max_modules=HORIZON
            )
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            refused += 1
            continue
        families = [module["family"] for module in metadata["modules"]]
        if not families:
            continue
        family_counts.update(families)
        first_module[families[0]] += 1
        module_counts[len(families)] += 1
        if all(family in PRUNE_FAMILIES for family in families):
            all_prune += 1
            all_prune_by_depth[len(families)] += 1
    executed = draws - refused
    return {
        "smiles": smiles,
        "seed": seed,
        "draws": draws,
        "refused": refused,
        "module_count_distribution": {
            str(k): round(v / executed, 5) for k, v in sorted(module_counts.items())
        },
        "family_share_over_modules": {
            k: round(v / sum(family_counts.values()), 5)
            for k, v in family_counts.most_common()
        },
        "first_module_share": {
            k: round(v / executed, 5) for k, v in first_module.most_common()
        },
        "all_prune_programs": all_prune,
        "all_prune_share": round(all_prune / executed, 5) if executed else None,
        "all_prune_share_by_depth": {
            str(k): round(v / executed, 5) for k, v in sorted(all_prune_by_depth.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--draws", type=int, default=250)
    arguments = parser.parse_args()

    cells = {}
    for protein in ("braf", "fa7", "5ht1b", "jak2", "parp1"):
        path = ROOT / f"configs/t4_held_target_distilled_{protein}_d06_250.json"
        for row in json.loads(path.read_text())["payload"]["cells"]:
            cells[row["cell"]] = row

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "horizon": HORIZON,
        "prune_families": sorted(PRUNE_FAMILIES),
        "cells": {},
    }
    for name in arguments.cells:
        row = cells[name]
        # The audit's shallow-lane seed derivation, verbatim:
        #   controller_seed + 1_000_003 * round_index + 10_007 * 0 + 101 * expert_index
        # with round_index 1 and expert_index 0 for the shallow lane.
        seed = int(row["controller_seed"] + 1_000_003 * 1 + 101 * 0)
        result = census(row["smiles"], seed, arguments.draws)
        payload["cells"][name] = result
        print(
            f"{name:10} seed={seed} all_prune={result['all_prune_share']} "
            f"by_depth={result['all_prune_share_by_depth']} "
            f"modules={result['module_count_distribution']}",
            flush=True,
        )
        print(f"           first_module={dict(list(result['first_module_share'].items())[:6])}", flush=True)

    destination = ROOT / "diagnostics/t4_proposal_mode_census_v1.json"
    existing = {}
    if destination.exists():
        existing = json.loads(destination.read_text()).get("cells", {})
    payload["cells"] = {**existing, **payload["cells"]}
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
