"""Complete COMPOSE region programs under explicit fragment boundary constraints.

No QED, SA, target drug, reward or pretrained-head flag enters this module.
Structural content is proposed before execution. Shared COMPOSE compilers
produce exact traces; the frozen mark model ranks whole legal programs. Its
length-normalized score is a panel preference, NOT an exact conditional CTMC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from rdkit import Chem

from compose_v4.benchmark.fragment_conditioned_sampler import PromptContext, RegionLock
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    _fragment_spec,
    check_fragment_constraint,
)
from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.control.edit_program import atom_signature, environment, extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    StructuralSubgoal,
    instantiate_goal,
)
from compose_v4.control.structural_subgoal_realizer import (
    RealizerConfig,
    realize_target_without_search,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    LEGACY_ATOM_RESTATE_ACTION_SEMANTICS,
    LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS,
    LEGACY_CYCLE_OPEN_ACTION_SEMANTICS,
    LEGACY_EDITING_PROCESS_SEMANTICS,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import (
    AtomRestate,
    BondDelete,
    BondInsert,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.trace_shard import decode_state
from compose_v4.rewrite.tracelets import RingSystemRestate


@dataclass(frozen=True)
class ProgramConstraint:
    locked_slots: tuple[int, ...]
    interfaces: tuple[int, ...]
    requirements: tuple[tuple[int, int], ...]

    @classmethod
    def from_context(cls, context: PromptContext) -> ProgramConstraint:
        if context.prompt.task not in (
            FragmentTask.MOTIF_EXTENSION,
            FragmentTask.SCAFFOLD_DECORATION,
        ):
            raise ValueError("official linker/morphing semantics are not admitted by this adapter")
        if context.attachment is None or not context.attachment.interfaces:
            raise ValueError("complete attachment programs require declared interfaces")
        return cls(
            context.locked_slots,
            context.attachment.interfaces,
            context.attachment.requirements,
        )

    def lock(self, source: MolecularGraph) -> RegionLock:
        return RegionLock(
            source,
            self.locked_slots,
            preserve_effective_chemistry=True,
            allowed_external_slots=self.interfaces,
        )

    def complete(self, graph: MolecularGraph) -> bool:
        locked = frozenset(self.locked_slots)
        return all(
            sum(
                int(graph.bonds[site, other] > 0)
                for other in range(graph.n_atoms)
                if other not in locked
            )
            >= required
            for site, required in self.requirements
        )


@dataclass(frozen=True)
class CompleteProgram:
    endpoint: MolecularGraph
    trace: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def smiles(self) -> str:
        return canonical_state_key(self.endpoint)


def _capped_region(rooted_smiles: str):
    """Decode labelled boundary roles without limiting ring or element content."""
    mol = Chem.MolFromSmiles(rooted_smiles)
    if mol is None:
        raise ValueError("unparseable proposed region")
    edit = Chem.RWMol(mol)
    roots = {}
    for atom in edit.GetAtoms():
        if atom.GetAtomicNum():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    for atom in edit.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        label = atom.GetIsotope()
        if label < 1 or label in roots or atom.GetDegree() != 1:
            raise ValueError("boundary dummies require unique positive isotope labels")
        neighbor = atom.GetNeighbors()[0]
        bond = edit.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        if bond.GetBondType() != Chem.BondType.SINGLE:
            raise ValueError("region boundary requires a single bond")
        roots[label] = neighbor.GetAtomMapNum()
        atom.SetAtomicNum(1)
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    if not 1 <= len(roots) <= 2:
        raise ValueError("region needs one or two declared boundaries")
    capped = Chem.RemoveHs(edit.GetMol())
    text = Chem.MolToSmiles(capped, canonical=False)
    decoded = Chem.MolFromSmiles(text)
    if decoded is None or len(Chem.GetMolFrags(decoded)) != 1:
        raise ValueError("proposed region is not one connected component")
    positions = {a.GetAtomMapNum(): a.GetIdx() for a in decoded.GetAtoms()}
    graph = smiles_to_molecular_graph(text)
    return graph, tuple(positions[roots[label]] for label in sorted(roots))


def region_goal(source: MolecularGraph, rooted_smiles: str, anchors: tuple[int, ...]):
    """Bind complete proposed content to retained roles via the shared T4 IR."""
    region, roots = _capped_region(rooted_smiles)
    if len(anchors) != len(roots) or len(set(anchors)) != len(anchors):
        raise ValueError("region boundary arity/anchor identity mismatch")
    n = len(anchors)
    inputs = tuple(atom_signature(source, anchor) for anchor in anchors)
    targets = []
    for signature in inputs:
        element, charge, hydrogen, degree = signature
        if hydrogen < 1:
            raise ValueError("retained boundary has no available hydrogen")
        targets.append((element, charge, hydrogen - 1, degree + 1))
    output = [list(atom_signature(region, i)) for i in range(region.n_real_atoms)]
    bonds = np.zeros((n + len(output), n + len(output)), dtype=int)
    bonds[:n, :n] = source.bonds[np.ix_(anchors, anchors)]
    bonds[n:, n:] = region.bonds
    for index, root in enumerate(roots):
        if output[root][2] < 1:
            raise ValueError("proposed region boundary cannot release its cap")
        output[root][2] -= 1
        output[root][3] += 1
        bonds[index, n + root] = bonds[n + root, index] = 1
    subgoal = StructuralSubgoal(
        input_atoms=inputs,
        input_bonds=tuple(tuple(int(source.bonds[a, b]) for b in anchors) for a in anchors),
        environments=tuple(environment(source, anchor) for anchor in anchors),
        target_atoms=tuple(targets),
        output_atoms=tuple(tuple(row) for row in output),
        target_bonds=tuple(tuple(int(v) for v in row) for row in bonds),
    )
    return StructuralGoal((subgoal,)), (anchors,)


def compile_region(source: MolecularGraph, rooted_smiles: str, anchors: tuple[int, ...]):
    """Compile a complete bound region with the existing no-search exact scheduler."""
    goal, binding = region_goal(source, rooted_smiles, anchors)
    target, target_receipt = instantiate_goal(source, goal, binding)
    result = realize_target_without_search(
        source, target, config=RealizerConfig(maximum_expansions=32)
    )
    if result["status"] != "realized":
        raise ValueError(f"complete region compiler abstained: {result['status']}")
    return result, {
        "lane": "shared_structural_region",
        "region": rooted_smiles,
        "boundary_arity": len(anchors),
        "bound_target": target_receipt,
        "compiler_strategy": result["compiler_strategy"],
    }


def admit_complete_program(
    source: MolecularGraph,
    trace: dict,
    constraint: ProgramConstraint,
    *,
    provenance: dict,
    require_complete: bool = True,
) -> CompleteProgram:
    """Replay a full proposal; chemistry/preservation are pathwise, task completion final."""
    actions = trace["actions"]
    if not 1 <= len(actions) <= 32:
        raise ValueError("complete program exceeds 1..32 primitive support")
    stage = {
        "name": "constrained_region_program",
        "actions": actions,
        "states": trace["states"],
        "endpoint": canonical_state_key(decode_state(trace["states"][-1])),
    }
    return admit_program_stages(
        source, [stage], constraint, provenance=provenance, require_complete=require_complete
    )


def admit_program_stages(source, stages, constraint, *, provenance, require_complete=True):
    """Keep complete constructor blocks and their cross-block created handles."""
    program, assignment = extract_program(source, stages)
    graph = compile_program_graph(program)
    endpoint, replay = execute_program_graph(
        source, graph, assignment, max_primitives=32, max_blocks=8
    )
    if canonical_state_key(endpoint) != stages[-1]["endpoint"]:
        raise RuntimeError("compiled program changed its endpoint on shared exact replay")
    lock = constraint.lock(source)
    if any(not lock.permits(decode_state(state)) for state in replay["states"]):
        raise ValueError("complete program violates the immutable core or external interfaces")
    if require_complete and not constraint.complete(endpoint):
        raise ValueError("complete program leaves required interfaces unsatisfied")
    return CompleteProgram(
        endpoint,
        replay,
        {
            **provenance,
            "program_block_lengths": [len(stages[i]["actions"]) for i in replay["block_order"]],
            "program_block_labels": [stages[i]["name"] for i in replay["block_order"]],
            "dependencies": list(graph.dependencies),
            "conflicts": list(graph.conflicts),
            "serialization_edges": list(graph.serialization_edges),
        },
    )


def candidate_stages(candidate):
    """Recover exact scheduled blocks without flattening coordinated programs."""
    lengths = candidate.provenance["program_block_lengths"]
    labels = candidate.provenance["program_block_labels"]
    if sum(lengths) != len(candidate.trace["actions"]) or any(n < 1 for n in lengths):
        raise ValueError("complete candidate block lengths do not cover its primitive stream")
    stages, start = [], 0
    for label, length in zip(labels, lengths, strict=True):
        stop = start + length
        stages.append(
            {
                "name": label,
                "actions": candidate.trace["actions"][start:stop],
                "states": candidate.trace["states"][start : stop + 1],
                "endpoint": canonical_state_key(decode_state(candidate.trace["states"][stop])),
            }
        )
        start = stop
    return stages


def verify_prompt_endpoint(context: PromptContext, candidate: CompleteProgram) -> None:
    # This check is never a QED/SA filter and does not replace the pathwise lock.
    specs = tuple(_fragment_spec(text) for text in context.prompt.fragments)
    mol = Chem.MolFromSmiles(candidate.smiles)
    if mol is None or not all(mol.HasSubstructMatch(spec.core) for spec in specs):
        raise ValueError("completed program lost the benchmark-visible fragment")
    if not check_fragment_constraint(context.prompt, candidate.smiles).satisfied:
        raise ValueError("completed program fails the benchmark boundary condition")


def sample_boundary_content(context_keys, entries, room, rng):
    """Sample observed content while reserving the actual minimum at later sites.

    This is a feasibility-conditioned sequential law, not uniform sampling over
    complete combinations. It uses training size/context only, never QED/SA.
    """
    pools = [tuple(e for e in entries if e["contexts"] == [key]) for key in context_keys]
    if any(not pool for pool in pools):
        raise ValueError("no observed region has a required boundary context")
    minimum_sizes = [min(e["heavy_atoms"] for e in pool) for pool in pools]
    if sum(minimum_sizes) > room:
        raise ValueError("observed boundary minima exceed remaining capacity")
    chosen = []
    for index, pool in enumerate(pools):
        reserved = sum(minimum_sizes[index + 1 :])
        compatible = [e for e in pool if e["heavy_atoms"] <= room - reserved]
        if not compatible:
            raise RuntimeError("feasible boundary reservation lost all compatible regions")
        weights = np.sqrt([entry["occurrences"] for entry in compatible])
        weights /= weights.sum()
        selected = compatible[int(rng.choice(len(compatible), p=weights))]
        chosen.append(selected)
        room -= selected["heavy_atoms"]
    return chosen


def propose_region_completion(context: PromptContext, entries: tuple[dict, ...], rng):
    """Sample all boundary content, then compile one complete multi-region program.

    All choices are graph/context and source-frequency based. No completion is
    committed until the full exact program satisfies the entire prompt.
    """
    from compose_v4.benchmark.training_attachment_fragments import atom_context

    constraint = ProgramConstraint.from_context(context)
    source = context.start_state
    core = Chem.MolFromSmiles(context.start_smiles)
    slots = [site for site, count in constraint.requirements for _ in range(count)]
    # Avoid always letting the first address consume all capacity.
    slots = [slots[int(i)] for i in rng.permutation(len(slots))]
    content = sample_boundary_content(
        [atom_context(core.GetAtomWithIdx(site)) for site in slots],
        entries,
        40 - source.n_real_atoms,
        rng,
    )
    chosen = zip(slots, content, strict=True)
    current, actions, stages, regions = source, [], [], []
    for site, selected in chosen:
        trace, provenance = compile_region(current, selected["rooted_smiles"], (site,))
        actions.extend(trace["actions"])
        stages.append(
            {
                "name": "observed_training_region",
                "actions": trace["actions"],
                "states": trace["states"],
                "endpoint": canonical_state_key(decode_state(trace["states"][-1])),
            }
        )
        if len(actions) > 32:
            raise ValueError("complete multi-boundary program exceeds 32 primitives")
        current = decode_state(trace["states"][-1])
        regions.append(
            {
                **provenance,
                "source_rows": selected["source_rows"],
                "training_occurrences": selected["occurrences"],
            }
        )
    candidate = admit_program_stages(
        source,
        stages,
        constraint,
        provenance={
            "lane": "shared_structural_region",
            "regions": regions,
            "module_count": len(regions),
            "learned_head_flags_changed": False,
        },
    )
    verify_prompt_endpoint(context, candidate)
    return candidate


def native_scoring_mark(model, predecessor, successor, rule, action):
    """Translate records only for the same slot-mapped molecular successor.

    This is a codec/semantics bridge, not a learned-head or support expansion.
    The original exact program stays unchanged. Raw Kekule phase may differ only
    when canonical identity, every atom channel and every mapped perceived bond
    agree. No atom permutation or different chemical transition is accepted.
    """
    if (
        isinstance(action, CycleCloseEdge)
        and model.cycle_close_action_semantics == LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS
    ):
        rule, action = "bond_insert", BondInsert(action.a, action.b, action.order)
    elif (
        isinstance(action, CycleOpenEdge)
        and model.cycle_open_action_semantics == LEGACY_CYCLE_OPEN_ACTION_SEMANTICS
    ):
        rule, action = "bond_delete", BondDelete(action.a, action.b)
    elif (
        isinstance(action, SemanticAtomRestate)
        and model.atom_restate_action_semantics == LEGACY_ATOM_RESTATE_ACTION_SEMANTICS
    ):
        slot = action.v
        rule, action = (
            "atom_restate",
            AtomRestate(
                slot,
                int(successor.atom_types[slot]),
                int(successor.formal_charges[slot]),
                int(successor.implicit_h_counts[slot]),
            ),
        )
    if model.editing_process_semantics == LEGACY_EDITING_PROCESS_SEMANTICS:
        native_successor = de_novo_rewrite_system().apply(predecessor, rule, action)
        if exact_graph_key(native_successor) != exact_graph_key(successor):
            same_atoms = all(
                np.array_equal(getattr(native_successor, field), getattr(successor, field))
                for field in ("atom_types", "formal_charges", "implicit_h_counts")
            )
            same_chemistry = same_atoms and canonical_state_key(
                native_successor
            ) == canonical_state_key(successor)
            if same_chemistry:
                same_chemistry = np.array_equal(
                    resonance_invariant_bond_classes(native_successor),
                    resonance_invariant_bond_classes(successor),
                )
            if not same_chemistry:
                raise ValueError(
                    "native scoring action does not reproduce the exact program successor chemistry"
                )
    return rule, action


def validate_native_compound_membership(batch):
    """Proposal-side admission for the native teacher scorer's finite compound fiber."""
    for index, action in enumerate(batch.teacher_actions):
        if (
            isinstance(action, RingSystemRestate)
            and action not in batch.ring_restate_actions[index]
        ):
            raise ValueError("compiled ring restate is outside the frozen model action fiber")


def learned_program_scores(model, candidates: tuple[CompleteProgram, ...], *, batch_size=16):
    """Mean native log-mark probabilities. No unsupported program gets an invented score."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = []
    lengths = []
    for candidate in candidates:
        actions = candidate.trace["actions"]
        states = candidate.trace["states"]
        if len(states) != len(actions) + 1:
            raise ValueError("program trace lacks exact predecessor states")
        lengths.append(len(actions))
        for index, record in enumerate(actions):
            rule, action = decode_action(record)
            predecessor, successor = decode_state(states[index]), decode_state(states[index + 1])
            rule, action = native_scoring_mark(model, predecessor, successor, rule, action)
            rows.append((predecessor, rule, action, index))
    scores = []
    # Use the checkpoint's actual semantics and enabled heads, never guessed defaults.
    keys = (
        "editing_process_semantics",
        "atom_restate_action_semantics",
        "ring_restate_scorer_mode",
        "cycle_close_action_semantics",
        "cycle_open_action_semantics",
        "atom_delete_action_semantics",
    )
    options = {key: getattr(model, key) for key in keys}
    options.update(
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=model.enable_ring_grow_macro,
        compute_ring_restates=model.enable_ring_restates,
        compute_cyclic_graft=model.enable_cyclic_graft,
        compute_ring_opening=model.enable_ring_opening,
        compute_ring_system_delete=model.enable_ring_system_delete,
    )
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start : start + batch_size]
            batch = prepare_factorized_mark_batch(
                tuple(row[0] for row in part),
                tuple(0.0 for _ in part),
                tuple(row[2] for row in part),
                tuple(row[1] for row in part),
                tuple(0.0 for _ in part),
                **options,
            )
            # The model's teacher scorer assumes its compound action is in the
            # native finite fiber and raises RuntimeError otherwise. A proposal
            # panel is not a teacher corpus: check membership explicitly, keeping
            # genuine implementation RuntimeErrors visible.
            validate_native_compound_membership(batch)
            scores.extend(
                model.forward_mark_batch(batch).selected_mark_log_probability.cpu().tolist()
            )
    result, offset = [], 0
    for length in lengths:
        values = np.asarray(scores[offset : offset + length], dtype=float)
        result.append(float(values.mean()) if np.isfinite(values).all() else float("-inf"))
        offset += length
    return tuple(result)


def select_learned_program(candidates, scores, rng):
    """Sample a canonical-endpoint panel with actual learned score influence."""
    if len(candidates) != len(scores):
        raise ValueError("program/score counts disagree")
    seen, kept = set(), []
    for index, candidate in enumerate(candidates):
        if candidate.smiles not in seen and np.isfinite(scores[index]):
            seen.add(candidate.smiles)
            kept.append(index)
    if not kept:
        raise ValueError("no model-supported complete program")
    log_weights = np.asarray([scores[i] for i in kept], dtype=float)
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    local = int(rng.choice(len(kept), p=weights))
    chosen = kept[local]
    return candidates[chosen], {
        "selected_index": chosen,
        "unique_model_supported_endpoints": len(kept),
        "selected_probability": float(weights[local]),
        "mean_native_log_mark_probability": float(scores[chosen]),
        "model_used": True,
        "score_semantics": "mean_log_mark_at_time_zero; finite_panel_softmax; not_path_probability",
    }
