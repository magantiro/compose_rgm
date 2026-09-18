"""Factor the teacher-program probability at the JAK2 root, one conditional at a time.

The basin autopsy established that the strong delta=0.6 architecture -- the root's methyl
ester replaced by an amide to a 1,2-diamine ring bearing a urea -- appears in none of 365
feasible endpoints drawn from 2,400 free programs. By the rule of three that bounds the
joint rate at roughly 3/2400 = 0.00125, so it is a proposal-MASS problem rather than a
proof of absent support, and the useful question is which factor of

    q0(Z|G) = q(R|G) q(A|G,R) q(C|G,R,A) q(z|G,R,A,C)

carries the smallness. A whole-program frequency cannot say.

The factors are estimated by CONDITIONING on sampled draws rather than by forcing the
sampler, which yields the same conditional rates while leaving the proposal law untouched.
Forcing is only necessary where a conditioning stage starves, and the report says when
that happens.

Root slots, read off the padded graph rather than assumed:
    slot 0  methyl carbon    (3 H, bonded only to slot 1)
    slot 1  ESTER OXYGEN     -- the atom the strong basin retypes to nitrogen
    slot 2  carbonyl carbon
    slot 3  carbonyl oxygen
Costs no oracle calls.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.structural_subgoal import (
    attachment_bindings,
    extract_structural_goal,
    instantiate_goal,
)
from compose_v4.experiments.t4_fiber_campaign import Fiber, _variants

JAK2_ROOT = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"
ESTER_OXYGEN, METHYL_CARBON = 1, 0
NITROGEN, OXYGEN = 3, 4

PATTERNS = {
    "ester": "[CX4][OX2][CX3](=O)[CH2]",
    "amide": "[NX3][CX3](=O)[CH2]",
    "ncn": "[NX3;R][CX4;R][NX3;R]",
    "nccn": "[NX3;R][CX4;R][CX4;R][NX3;R]",
    "urea": "[NX3;R][CX3](=O)[NX3]",
}
QUERIES = {name: Chem.MolFromSmarts(smarts) for name, smarts in PATTERNS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    fiber = Fiber(JAK2_ROOT, 0.6)
    rng = np.random.default_rng(options.seed)
    source = pad_molecular_graph(smiles_to_molecular_graph(JAK2_ROOT), 48)

    tally = Counter()
    families_at_ester = Counter()
    for _ in range(options.draws):
        tally["programs"] += 1
        try:
            _, _, _, trace, _ = synthesize_dynamic_program(source, rng, max_modules=3)
            goal, _, _ = extract_structural_goal(tuple(trace["states"]), tuple(trace["actions"]))
        except (ValueError, KeyError, IndexError, TypeError):
            tally["synthesis_failed"] += 1
            continue

        # ---- factor 1: WHERE. Which source slots does this program bind? ----
        touched: set[int] = set()
        bindings = []
        ok = True
        for subgoal in goal.subgoals:
            census = attachment_bindings(subgoal, source)
            if not census.assignments:
                ok = False
                break
            assignment = census.assignments[0]
            bindings.append(assignment)
            touched.update(int(slot) for slot in assignment)
        if not ok:
            tally["unbindable"] += 1
            continue

        at_ester = ESTER_OXYGEN in touched
        tally["bound"] += 1
        tally["where_ester_oxygen"] += at_ester
        tally["where_methyl_carbon"] += METHYL_CARBON in touched
        tally["where_ester_group"] += bool(touched & {0, 1, 2, 3})

        # ---- factors 2-4: for every variant this program admits, what is realized ----
        for index, subgoal in enumerate(goal.subgoals):
            for family, candidate in _variants(subgoal):
                subs = list(goal.subgoals)
                subs[index] = candidate
                merged = replace(goal, subgoals=tuple(subs))
                try:
                    built, _ = instantiate_goal(source, merged, tuple(bindings))
                    endpoint = molecular_graph_to_smiles(built)
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                if not endpoint:
                    continue
                tally["variants"] += 1
                gate = fiber.check(endpoint)
                tally["feasible"] += gate is not None

                # factor 2: does the role bound to the ester oxygen become NITROGEN?
                retyped = False
                if ESTER_OXYGEN in [int(s) for s in bindings[index]]:
                    role = [int(s) for s in bindings[index]].index(ESTER_OXYGEN)
                    if role < len(candidate.target_atoms):
                        signature = candidate.target_atoms[role]
                        if signature is None:
                            tally["ester_oxygen_deleted"] += 1
                        elif int(signature[0]) == NITROGEN:
                            retyped = True
                if retyped:
                    tally["ester_oxygen_to_nitrogen"] += 1
                    families_at_ester[family] += 1
                    if gate is not None:
                        tally["ester_oxygen_to_nitrogen_feasible"] += 1

                if gate is None:
                    continue
                molecule = Chem.MolFromSmiles(gate["smiles"])
                if molecule is None:
                    continue
                has = {n: molecule.HasSubstructMatch(q) for n, q in QUERIES.items()}
                ring = has["ncn"] or has["nccn"]
                tally["feasible_ester_broken"] += not has["ester"]
                tally["feasible_amide"] += has["amide"]
                tally["feasible_diamine_ring"] += ring
                tally["feasible_urea"] += has["urea"]
                tally["feasible_amide_and_ring"] += has["amide"] and ring
                tally["feasible_basin"] += has["amide"] and ring and has["urea"]
                if at_ester:
                    tally["at_ester_feasible"] += 1
                    tally["at_ester_amide"] += has["amide"]
                    tally["at_ester_ring"] += ring
                    tally["at_ester_amide_and_ring"] += has["amide"] and ring

    Path(options.out).write_text(json.dumps(
        {"tally": dict(tally), "families_retyping_ester_oxygen": dict(families_at_ester),
         "draws": options.draws, "seed": options.seed}, indent=1))


if __name__ == "__main__":
    main()
