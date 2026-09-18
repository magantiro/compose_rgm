"""End-to-end FiberControl campaign on one T4 cell. The JAK2 route is never read.

Each round: expand a diverse frontier into a large FREE pool of complete programs, keep
only endpoints passing the exact task gate, choose a small batch by reward-guided
acquisition, dock it, update, replan. Only the first oracle decision of each round is
executed; the tree is rebuilt from what was actually measured.

Nothing target-specific enters except counted docking outcomes.
"""

from __future__ import annotations

import itertools
import os
import sys
from dataclasses import replace

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, DataStructs, RDConfig, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.fiber_control import program_features
from compose_v4.control.intervention_closure import (
    intervene_bond_order,
    intervene_element,
    intervene_retained_deletion,
    intervene_retained_element,
    intervene_scale,
    validate,
)
from compose_v4.control.progressive_structured_sampler import synthesize_progressive_program
from compose_v4.control.queryable_fiber import instability
from compose_v4.control.structural_subgoal import (
    attachment_bindings,
    extract_structural_goal,
    instantiate_goal,
)
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

SCHEMA_VERSION = "t4_fiber_campaign_v1"


#: What the benchmark itself requires of a returned molecule, verbatim from the task
#: contract in `modal_apps/genmol_t4_opt_app.py`: "RETURNED only QED>=0.6, SA<=4,
#: sim>=delta; among those, best docking", with QED_MIN, SA_MAX = 0.6, 4.0.
QED_MIN, SA_MAX = 0.6, 4.0

#: Heavy-atom ceiling. This is a REPRESENTATION limit of the padded state, not a
#: benchmark criterion, and is recorded separately for exactly that reason.
REPRESENTABLE_HEAVY_ATOMS = 40

#: Three nested supports, named for what they actually are. The distinction matters for
#: any claim: a method may search a STRICTER subset and still beat a published number --
#: arguably more convincingly -- but it may not call that subset "the benchmark".
#:
#:     Q_legacy_screened  subset of  Q_compose_valid  subset of  Q_benchmark
#:
#: BENCHMARK_ONLY is the task contract verbatim and nothing else.
#: COMPOSE_VALID adds structural validity, because COMPOSE will not spend an oracle call
#:   on a species that is not a molecule. Accepting all 25 published winners shows this
#:   gate does not remove what IVG found; it does NOT show the gate equals the task.
#: LEGACY_SCREENED further adds a medicinal-chemistry preference list that refuses 7.0%
#:   of the benchmark's own leads, and exists only to reproduce runs already made on it.
BENCHMARK_ONLY, COMPOSE_VALID, LEGACY_SCREENED = (
    "benchmark_only",
    "compose_valid",
    "legacy_screened",
)
_SUPPORTS = (BENCHMARK_ONLY, COMPOSE_VALID, LEGACY_SCREENED)


class Fiber:
    """The task gate. Free to evaluate; the oracle is never asked about it.

    `support` decides WHOSE criterion this is, and the distinction is load-bearing for
    any comparison against a published number.

    `COMPOSE_VALID` is the task thresholds plus a structural validity gate. It is a
    STRICT SUBSET of the benchmark support, not a restatement of it, and that is the
    honest description: a method is allowed to search a smaller set and still beat a
    published number.

    The validity gate is `med_chem_gate.is_valid`. Calibration shows it accepts all 15
    T4 seeds and all 25 published InVirtuoGen winners, which establishes that it does not
    remove what IVG found -- it does NOT establish that the gate equals the task
    criterion, and the two claims must not be confused. What it removes is species that
    are not molecules: radicals, hypervalent sulfur and iodine, cumulenes, strained N-N
    rings.

    This is here because removing it was tried and failed loudly. A first version of the
    benchmark support tested only similarity, QED and SA, on the reasoning that anything
    else is our own addition. Two rounds later the leading endpoint was
    `COC(=O)CC1Nc2cc(C3(C[NH])CCCCC3)ccc2-...`, an aminyl RADICAL at -9.70, with two
    carbon radicals and one `[SH3]` in the same batch. The radical check had lived inside
    the `instability` bundle that was removed, so stripping the bundle silently deleted
    it. A radical is not a candidate the benchmark or anyone else would score.

    It also means our own intervention layer still emits radicals. `_rebalance` exists to
    prevent exactly that, and `retained_deletion` and `retained_element` are still
    producing them -- which the old bundled gate was concealing rather than fixing.

    `LEGACY_SCREENED` additionally applies `queryable_fiber.instability`, which bundles a
    medicinal-chemistry preference list and a net-charge rule into the SEARCH support.
    Measured, that bundle refuses 7.0% of the benchmark's own Jin QED lead set and 13.8%
    of held-out GuacaMol, chiefly through a plain-enone rule firing on 9.0% of GuacaMol.
    A run under it is searching a strictly smaller fiber than the task defines, so its
    score is not comparable to a number obtained on the full one. It is retained only to
    reproduce runs already made under it.

    Three ways the screened form was tighter than the task, all now separated:
      * it required QED strictly above 0.6 and SA strictly below 4.0, where the contract
        says `>=0.6` and `<=4`. Strong solutions hug active constraints -- the delta=0.6
        leaders carry a median similarity margin of +0.019 and several sit at exactly
        0.600 -- so a boundary-exclusive comparison is not a small difference;
      * it imposed an 18-heavy-atom floor that appears nowhere in the task;
      * it folded the med-chem bundle into the support.

    Medicinal-chemistry judgement belongs to the RETURNED molecule, where the repo
    already puts it (`t4_endpoint_selection.acceptable_endpoint`, documented as "a
    heuristic, not an additional benchmark threshold"). Report it as a second number.
    """

    def __init__(self, seed_smiles: str, delta: float, *, support: str = COMPOSE_VALID):
        if support not in _SUPPORTS:
            raise ValueError(f"unknown fiber support {support!r}; expected one of {_SUPPORTS}")
        self.delta = delta
        self.support = support
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.seed = self.generator.GetFingerprint(Chem.MolFromSmiles(seed_smiles))

    def check(self, smiles: str) -> dict | None:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or "." in smiles:
            return None
        heavy = mol.GetNumHeavyAtoms()
        if heavy > REPRESENTABLE_HEAVY_ATOMS:
            return None
        similarity = DataStructs.TanimotoSimilarity(self.seed, self.generator.GetFingerprint(mol))
        quality = QED.qed(mol)
        access = sascorer.calculateScore(mol)
        if similarity < self.delta or quality < QED_MIN or access > SA_MAX:
            return None
        if self.support != BENCHMARK_ONLY and not structurally_valid(smiles):
            return None
        if self.support == LEGACY_SCREENED and instability(smiles):
            return None
        return {
            "smiles": Chem.MolToSmiles(mol),
            "similarity": similarity,
            "qed": quality,
            "sa": access,
            "heavy": heavy,
        }


def _variants(subgoal):
    out = [("base", subgoal)]
    created = len(subgoal.output_atoms)
    if created:
        for count in (created - 1, created + 1):
            if count >= 1:
                try:
                    out.append(("scale", intervene_scale(subgoal, output_count=count)[0]))
                except (ValueError, IndexError):
                    pass
        for element in (3, 4):
            try:
                out.append(
                    ("element", intervene_element(subgoal, output_index=0, element=element)[0])
                )
            except (ValueError, IndexError):
                pass
    for role in range(len(subgoal.target_atoms)):
        for element in (2, 3, 4):
            try:
                out.append(
                    (
                        "retained_element",
                        intervene_retained_element(subgoal, input_index=role, element=element)[0],
                    )
                )
            except (ValueError, IndexError):
                pass
        try:
            out.append(
                ("retained_deletion", intervene_retained_deletion(subgoal, input_index=role)[0])
            )
        except (ValueError, IndexError):
            pass
    bonds = subgoal.target_bonds
    for left in range(len(bonds)):
        for right in range(left + 1, len(bonds)):
            if bonds[left][right]:
                for order in (1, 2):
                    if order == bonds[left][right]:
                        continue
                    try:
                        out.append(
                            (
                                "bond_order",
                                intervene_bond_order(subgoal, left=left, right=right, order=order)[
                                    0
                                ],
                            )
                        )
                    except (ValueError, IndexError):
                        pass
                break
        else:
            continue
        break
    keep = [("base", subgoal)]
    for name, candidate in out[1:]:
        report = validate(subgoal, candidate)
        if report["legal"] and report["round_trips"] and report["complement_preserved"]:
            keep.append((name, candidate))
    return keep


def expand(
    parent: str,
    parent_score: float,
    fiber: Fiber,
    rng,
    *,
    draws: int,
    multi_region: bool = True,
    horizon: int = 3,
    proposal_lane: str = "shallow",
) -> list[dict]:
    """Free complete programs from one parent; only queryable endpoints are returned.

    `horizon` is the receding-horizon depth: the most modules a single program may carry
    before the controller replans against a measured outcome. It is a parameter rather
    than a constant so that one-step control is the same code path as depth-three
    control, and the comparison between them is not a comparison of two programs.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one module")
    if proposal_lane not in ("shallow", "structured"):
        raise ValueError("proposal_lane must be 'shallow' or 'structured'")
    try:
        source = pad_molecular_graph(smiles_to_molecular_graph(parent), 48)
    except (ValueError, KeyError):
        return []
    found: dict[str, dict] = {}
    for _ in range(draws):
        try:
            if proposal_lane == "shallow":
                _, _, _, trace, metadata = synthesize_dynamic_program(
                    source, rng, max_modules=horizon
                )
            else:
                _, _, _, trace, metadata = synthesize_progressive_program(source, rng)
            goal, _, _ = extract_structural_goal(tuple(trace["states"]), tuple(trace["actions"]))
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            continue
        program_families = [
            module.get("family")
            for module in metadata.get("modules", [])
            if module.get("family")
        ]
        variants = {i: _variants(sg) for i, sg in enumerate(goal.subgoals)}
        edits = [{i: c} for i, vs in variants.items() for _, c in vs]
        labels = [[n] for i, vs in variants.items() for n, _ in vs]
        if multi_region and len(goal.subgoals) >= 2:
            for i, j in itertools.combinations(sorted(variants), 2):
                for (ni, ci), (nj, cj) in itertools.product(variants[i][:4], variants[j][:4]):
                    if ni == "base" and nj == "base":
                        continue
                    edits.append({i: ci, j: cj})
                    labels.append([ni, nj])
        for edit, families in zip(edits, labels):
            subs = list(goal.subgoals)
            for index, candidate in edit.items():
                subs[index] = candidate
            merged = replace(goal, subgoals=tuple(subs))
            bindings, ok = [], True
            for subgoal in merged.subgoals:
                census = attachment_bindings(subgoal, source)
                if not census.assignments:
                    ok = False
                    break
                bindings.append(census.assignments[0])
            if not ok:
                continue
            try:
                built, _ = instantiate_goal(source, merged, tuple(bindings))
                endpoint = molecular_graph_to_smiles(built)
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            gate = fiber.check(endpoint)
            if gate is None or gate["smiles"] in found or gate["smiles"] == parent:
                continue
            created = sum(len(c.output_atoms) for c in edit.values())
            found[gate["smiles"]] = {
                **gate,
                "parent": parent,
                "parent_score": parent_score,
                "families": families,
                "program_families": program_families,
                "proposal_lane": proposal_lane,
                "regions": len(edit),
                "created": created,
                "deleted": sum(1 for c in edit.values() for t in c.target_atoms if t is None),
                "delta": fiber.delta,
            }
    return list(found.values())


def prepare(records, state, fiber: Fiber) -> list[dict]:
    """Attach the decision features and a structural key used for batch diversity."""
    prepared = []
    for record in records:
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is None:
            continue
        bits = fiber.generator.GetFingerprint(mol).GetOnBits()
        prepared.append(
            {**record, "features": program_features(record, state), "fingerprint": set(bits)}
        )
    return prepared
