"""T4 proposal expansion and endpoint eligibility, without docking side effects.

Expand a parent into structural programs, apply supported goal interventions,
and retain endpoints accepted by the configured gate. Optional exact program
realization supplies traces for frozen-reference selection. The caller owns
objective evaluation, budget accounting, locking and campaign persistence."""

from __future__ import annotations

import itertools
import math
import os
import sys
from dataclasses import asdict, replace

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
from compose_v4.control.progressive_structured_sampler import (
    synthesize_anchored_replacement_program,
    synthesize_progressive_program,
)
from compose_v4.control.queryable_fiber import instability
from compose_v4.control.structural_subgoal import (
    attachment_bindings,
    extract_structural_goal,
    instantiate_goal,
)
from compose_v4.control.structural_subgoal_realizer import (
    RealizerConfig,
    realize_structural_goal,
)
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA_VERSION = "t4_fiber_campaign_v1"


#: What the benchmark itself requires of a returned molecule, verbatim from the task
#: contract in `modal_apps/genmol_t4_opt_app.py`. Only eligible endpoints are docked.
QED_MIN, SA_MAX = 0.6, 4.0

#: Heavy-atom ceiling. This is a REPRESENTATION limit of the padded state, not a
#: benchmark criterion, and is recorded separately for exactly that reason.
REPRESENTABLE_HEAVY_ATOMS = 40
PROPOSAL_SLOTS = 48

#: Three nested supports, named for what they actually are. The distinction matters for
#: any claim: a method may search a STRICTER subset and still beat a published number --
#: arguably more convincingly -- but it may not call that subset "the benchmark".
#:
#:     Q_legacy_screened  subset of  Q_compose_valid  subset of  Q_benchmark
#:
#: BENCHMARK_ONLY is the task contract verbatim and nothing else.
#: COMPOSE_VALID adds structural validity, because COMPOSE will not spend an oracle call
#:   on a species that is not a molecule. Accepting all 25 published winners shows this
#:   gate does not remove what IVG found. It does not show the gate equals the task.
#: LEGACY_SCREENED further adds a medicinal-chemistry preference list that refuses 7.0%
#:   of the benchmark's own leads, and exists only to reproduce runs already made on it.
BENCHMARK_ONLY, COMPOSE_VALID, LEGACY_SCREENED = (
    "benchmark_only",
    "compose_valid",
    "legacy_screened",
)
_SUPPORTS = (BENCHMARK_ONLY, COMPOSE_VALID, LEGACY_SCREENED)


class Fiber:
    """Check source-relative endpoint eligibility without a docking call.

    All modes enforce QED >= 0.6, SA <= 4, configured Morgan similarity and the
    40-heavy-atom representation limit. BENCHMARK_ONLY adds no structural screening.
    COMPOSE_VALID also applies med_chem_gate.is_valid. LEGACY_SCREENED additionally
    applies queryable_fiber.instability. These are distinct search supports and must
    remain identifiable in run configuration.

    The extra structural screen is a computational heuristic, not a guarantee of
    synthesizability or biological suitability. Constraints apply to endpoints.
    Intermediates remain subject to the primitive executor. Similarity is always
    measured against the supplied starting lead, not the current parent."""

    def __init__(self, seed_smiles: str, delta: float, *, support: str = COMPOSE_VALID):
        if support not in _SUPPORTS:
            raise ValueError(f"unknown fiber support {support!r}; expected one of {_SUPPORTS}")
        if isinstance(delta, bool) or not math.isfinite(delta) or not 0 <= delta <= 1:
            raise ValueError("fiber delta must be finite and in [0, 1]")
        seed_molecule = Chem.MolFromSmiles(seed_smiles) if seed_smiles else None
        if (
            seed_molecule is None
            or seed_molecule.GetNumHeavyAtoms() == 0
            or len(Chem.GetMolFrags(seed_molecule)) != 1
        ):
            raise ValueError(f"fiber lead must be a nonempty connected molecule: {seed_smiles!r}")
        self.delta = delta
        self.support = support
        #: The declared similarity reference, retained as written. The region-draw
        #: law is built against THIS string, never the current parent, so a law
        #: cannot drift its own target as the campaign walks away from the root.
        self.seed_smiles = seed_smiles
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.seed = self.generator.GetFingerprint(seed_molecule)

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
    region_law=None,
    include_realized_actions: bool = False,
    realizer_config: RealizerConfig | None = None,
) -> list[dict]:
    """Generate complete programs from one parent and return queryable endpoints.

    `horizon` is the receding-horizon depth: the most modules a single program may carry
    before the controller replans against a measured outcome. It is a parameter rather
    than a constant so that one-step control is the same code path as depth-three
    control, and the comparison between them is not a comparison of two programs.

    `region_law` is the bridge region-draw law selected by the contract field
    `proposal.shallow.region_law`, resolved by
    `compose_v4.control.region_law_contract`. `None` is the historical behaviour and
    is byte-identical: `synthesize_dynamic_program` then draws under v1's uniform
    bounded law, consuming the same RNG stream as before. It is accepted only on the
    `shallow` lane because that is the only lane that reaches a synthesizer able to
    thread it. Requesting it on another lane raises rather than being ignored.

    ``include_realized_actions`` compiles each accepted structural goal through
    the executor, preserving its exact source and realized actions for reference
    scoring. This is a new realization of the generated goal, not the original
    pre-intervention recipe. Compilation abstentions retain the candidate as
    explicitly unscored. They do not change the proposal pool or its RNG stream.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one module")
    if type(include_realized_actions) is not bool:
        raise ValueError("include_realized_actions must be a Boolean")
    if realizer_config is not None and not include_realized_actions:
        raise ValueError("realizer_config requires include_realized_actions")
    if proposal_lane not in ("shallow", "structured", "anchored_replacement"):
        raise ValueError("proposal_lane must be 'shallow', 'structured' or 'anchored_replacement'")
    if region_law is not None and proposal_lane != "shallow":
        # Fail closed rather than drop it. A law accepted here and never consulted
        # is precisely the "declared but not consumed" defect the contract check
        # exists to prevent, and it would be invisible in the run artifact.
        raise ValueError(
            f"region_law is only consumable on the 'shallow' lane, not {proposal_lane!r}"
        )
    try:
        source = pad_molecular_graph(smiles_to_molecular_graph(parent), PROPOSAL_SLOTS)
    except (ValueError, KeyError):
        return []
    found: dict[str, dict] = {}
    for _ in range(draws):
        try:
            if proposal_lane == "shallow":
                _, _, _, trace, metadata = synthesize_dynamic_program(
                    source, rng, max_modules=horizon, region_law=region_law
                )
            elif proposal_lane == "structured":
                _, _, _, trace, metadata = synthesize_progressive_program(source, rng)
            else:
                _, _, _, trace, metadata = synthesize_anchored_replacement_program(source, rng)
            goal, _, _ = extract_structural_goal(tuple(trace["states"]), tuple(trace["actions"]))
        except (ValueError, RuntimeError, KeyError, IndexError):
            continue
        program_families = [
            module.get("family") for module in metadata.get("modules", []) if module.get("family")
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
            if include_realized_actions:
                config = RealizerConfig() if realizer_config is None else realizer_config
                realized = realize_structural_goal(source, merged, tuple(bindings), config=config)
                row = found[gate["smiles"]]
                row["source_state"] = encode_state(source)
                row["trace_origin"] = "compiled_structural_goal"
                row["trace_compilation"] = {
                    "status": realized["status"],
                    "configuration": asdict(config),
                    "expanded": realized["expanded"],
                    "attempted": realized["attempted"],
                }
                if realized["status"] == "realized":
                    actual_endpoint = canonical_state_key(decode_state(realized["states"][-1]))
                    if actual_endpoint != row["smiles"]:
                        raise RuntimeError("T4 realized program changed the proposed endpoint")
                    row["realized_actions"] = realized["actions"]
                    row["realized_endpoint_key"] = actual_endpoint
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
