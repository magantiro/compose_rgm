"""Can the enabled lanes express the transformation fa7_0 needs? Measured, not inferred.

THE QUESTION
------------
`fa7_0`'s clean eligible witnesses are large coherent EXCISIONS of 7-15 heavy atoms,
and `segment_replace`'s completion half draws a length on 1..`MAX_SEGMENT_LENGTH`(=8)
and builds a LINEAR SINGLE-BONDED chain over ("C","N","O") off one anchor -- it can
install neither a ring nor a branch nor a non-CNO element.  That invites the
inference that the lane structurally cannot match, and that a zero yield is
incapability rather than rarity.

WHY THE INFERENCE DOES NOT TRANSFER, AND WHY THIS PROBE EXISTS
---------------------------------------------------------------
`expand` does NOT gate the molecule the module builds.  It abstracts the synthesized
program with `extract_structural_goal`, expands `_variants`, re-binds each subgoal
and gates whatever `instantiate_goal` CONSTRUCTS -- measured on this cell, the
program's own endpoint reaches the gate with frequency 0.0000 over 150 draws.  So a
statement about `_grow_actions`' content is a statement about an object the campaign
never scores, and the reachable support has to be measured at the gate.

WHAT IT MEASURES
----------------
It WRAPS the production `Fiber.check` rather than transcribing it, so every endpoint
the lane offers is recorded with the real verdict beside it, and the decomposition
is checked against that verdict on every call.  For each offered endpoint it reports
the heavy-atom delta against the seed (the excision scale), the ring-count delta,
whether the endpoint retains the amidine, and whether it is eligible.

The question it answers is EXPRESSIBILITY, not yield: does the reachable support
contain the required move class at all, and does it contain ring-bearing content?

ZERO oracle calls.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from compose_v4.control.completion_law_contract import completion_law_for_proposal_lane
from compose_v4.control.region_law_contract import region_law_for_proposal_lane
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand

RDLogger.DisableLog("rdApp.*")

AMIDINE = Chem.MolFromSmarts("[CX3](=[NX2])[NX3]")


def _probe_factory(seed_smiles, fiber, horizon, kind):
    def _draw(law, attempt):
        kwargs = {"region_law": law} if kind == "region" else {"completion_law": law}
        return expand(
            seed_smiles,
            0.0,
            fiber,
            np.random.default_rng((90_000 if kind == "region" else 91_000) + attempt),
            draws=1,
            multi_region=False,
            horizon=horizon,
            proposal_lane="shallow",
            **kwargs,
        )

    return _draw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--cell", default="fa7_0")
    parser.add_argument("--lane", default="shallow", choices=["shallow", "anchored_replacement"])
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    contract = json.loads(pathlib.Path(args.contract).read_text())["payload"]
    cell = next(row for row in contract["cells"] if row["cell"] == args.cell)
    seed_smiles = cell["smiles"]
    horizon = contract["proposal"][args.lane]["horizon"]
    fiber = Fiber(seed_smiles, contract["delta"], support=contract["support"])
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    seed_heavy = seed_mol.GetNumHeavyAtoms()
    seed_rings = rdMolDescriptors.CalcNumRings(seed_mol)

    region_law, _ = region_law_for_proposal_lane(
        contract,
        lane=args.lane,
        delta=contract["delta"],
        reference_smiles=seed_smiles,
        draw=_probe_factory(seed_smiles, fiber, horizon, "region"),
    )
    completion_law, _ = completion_law_for_proposal_lane(
        contract,
        lane=args.lane,
        delta=contract["delta"],
        reference_smiles=seed_smiles,
        draw=_probe_factory(seed_smiles, fiber, horizon, "completion"),
    )

    # WRAP the production gate. `expand` returns only survivors, so the support it
    # OFFERS is invisible from its return value; the wrapper records every offer
    # with the real verdict, and the real method still decides.
    offered: dict[str, dict] = {}
    original_check = fiber.check

    def _wrapped(smiles: str):
        verdict = original_check(smiles)
        if smiles not in offered:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is not None:
                offered[smiles] = {
                    "heavy_delta": mol.GetNumHeavyAtoms() - seed_heavy,
                    "ring_delta": rdMolDescriptors.CalcNumRings(mol) - seed_rings,
                    "retains_amidine": bool(mol.HasSubstructMatch(AMIDINE)),
                    "eligible": verdict is not None,
                    "similarity": round(float(verdict["similarity"]), 4) if verdict else None,
                    "qed": round(float(verdict["qed"]), 4) if verdict else None,
                    "sa": round(float(verdict["sa"]), 4) if verdict else None,
                }
        return verdict

    fiber.check = _wrapped
    started = time.time()
    records = expand(
        seed_smiles,
        -7.5,
        fiber,
        np.random.default_rng(int(cell["controller_seed"] + 1_000_003 * (args.seed_index + 1))),
        draws=args.draws,
        multi_region=True,
        horizon=horizon,
        proposal_lane=args.lane,
        region_law=region_law,
        completion_law=completion_law,
    )
    fiber.check = original_check

    # The decomposition must predict the real verdict, or the attribution is void.
    eligible_from_wrapper = {s for s, row in offered.items() if row["eligible"]}
    returned = {row["smiles"] for row in records}
    disagreements = len(returned - {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in eligible_from_wrapper})

    deltas = [row["heavy_delta"] for row in offered.values()]
    excisions = [d for d in deltas if d < 0]
    big = {row_key: row for row_key, row in offered.items() if row["heavy_delta"] <= -7}
    ring_bearing = [row for row in offered.values() if row["ring_delta"] != 0]
    payload = {
        "schema_version": "t4_fa7_0_expressibility_v1",
        "cell": args.cell,
        "lane": args.lane,
        "draws": args.draws,
        "seed_index": args.seed_index,
        "oracle_calls": 0,
        "seed_heavy_atoms": seed_heavy,
        "seed_rings": seed_rings,
        "elapsed_seconds": round(time.time() - started, 1),
        "gate_disagreements": disagreements,
        "distinct_offered": len(offered),
        "distinct_eligible": len(eligible_from_wrapper),
        "excisions_offered": len(excisions),
        "max_excision_heavy_atoms": -min(deltas) if deltas else 0,
        "offered_with_excision_ge_7": len(big),
        "offered_with_excision_ge_7_eligible": sum(1 for r in big.values() if r["eligible"]),
        "offered_with_ring_count_change": len(ring_bearing),
        "offered_with_ring_change_and_excision_ge_7": sum(
            1 for r in big.values() if r["ring_delta"] != 0
        ),
        "eligible_rows": [
            {"smiles": s, **row} for s, row in offered.items() if row["eligible"]
        ],
        "largest_excisions": [
            {"smiles": s, **row}
            for s, row in sorted(offered.items(), key=lambda kv: kv[1]["heavy_delta"])[:15]
        ],
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(args.out).open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    for key in (
        "distinct_offered",
        "distinct_eligible",
        "gate_disagreements",
        "max_excision_heavy_atoms",
        "offered_with_excision_ge_7",
        "offered_with_excision_ge_7_eligible",
        "offered_with_ring_count_change",
        "offered_with_ring_change_and_excision_ge_7",
        "elapsed_seconds",
    ):
        print(f"  {key}: {payload[key]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
