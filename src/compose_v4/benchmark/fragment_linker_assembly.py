"""Bridge-free, connected-state assembly using the existing COMPOSE compiler.

This is a content compiler and fidelity check, not a learned linker sampler.
Two disconnected cores are conditioning, never an executable graph state.
The initial core is locked throughout; the second core is realized exactly
at completion. No supplied core is claimed present before it is constructed.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations, pairwise, product

from rdkit import Chem

from compose_v4.benchmark.fragment_conditioned_sampler import RegionLock
from compose_v4.benchmark.fragment_constrained import (
    FragmentPrompt,
    FragmentTask,
    _fragment_spec,
)
from compose_v4.benchmark.fragment_constrained_runner import (
    FragmentProposalAbstention,
    ProposalLimits,
    _append_molecule,
    _capped_core,
    _compile_groups,
    _Group,
    _mapped_graph,
    _open_core,
    _sanitize_candidate,
    _source_graph,
)
from compose_v4.benchmark.fragment_program_adapter import CompleteProgram
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.control.edit_program_graph import execute_program_graph
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

LINKER_TASKS = (FragmentTask.LINKER_DESIGN, FragmentTask.SCAFFOLD_MORPHING)


def _specs(prompt: FragmentPrompt):
    if prompt.task not in LINKER_TASKS or len(prompt.fragments) != 2:
        raise ValueError("linker assembly requires two supplied cores")
    for text in prompt.fragments:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(f"invalid linker core: {text}")
        if any(
            bond.GetBondType() != Chem.BondType.SINGLE
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() == 0
            for bond in atom.GetBonds()
        ):
            raise ValueError("linker core interfaces require supplied single bonds")
    specs = tuple(_fragment_spec(text) for text in prompt.fragments)
    if any(
        len(s.attachment_requirements) != 1 or s.attachment_requirements[0][1] != 1 for s in specs
    ):
        raise ValueError("each linker core must declare exactly one attachment")
    return specs


def _mapped_fidelity(molecule, specs, maps):
    groups = tuple(frozenset(m) for m in maps)
    if not groups[0].isdisjoint(groups[1]):
        return {"satisfied": False, "reason": "overlapping_cores"}
    sites = tuple(m[s.attachment_requirements[0][0]] for s, m in zip(specs, maps, strict=True))
    for spec, mapping, group, site in zip(specs, maps, groups, sites, strict=True):
        for i, mapped in enumerate(mapping):
            a, b = spec.core.GetAtomWithIdx(i), molecule.GetAtomWithIdx(mapped)
            if (a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic()) != (
                b.GetAtomicNum(),
                b.GetFormalCharge(),
                b.GetIsAromatic(),
            ):
                return {"satisfied": False, "reason": "core_atom_changed"}
        for i, j in combinations(range(len(mapping)), 2):
            a = spec.core.GetBondBetweenAtoms(i, j)
            b = molecule.GetBondBetweenAtoms(mapping[i], mapping[j])
            if (None if a is None else a.GetBondType()) != (None if b is None else b.GetBondType()):
                return {"satisfied": False, "reason": "core_internal_bond_changed"}
        contacts = [
            (a, n.GetIdx())
            for a in group
            for n in molecule.GetAtomWithIdx(a).GetNeighbors()
            if n.GetIdx() not in group
        ]
        if len(contacts) != 1 or contacts[0][0] != site:
            return {"satisfied": False, "reason": "wrong_or_extra_core_contact"}
    direct = molecule.GetBondBetweenAtoms(*sites) is not None
    blocked = groups[0] | groups[1]
    frontier = deque([(sites[0], (sites[0],))])
    visited = {sites[0]}
    path = ()
    while frontier:
        atom, prefix = frontier.popleft()
        for neighbor in sorted(n.GetIdx() for n in molecule.GetAtomWithIdx(atom).GetNeighbors()):
            if neighbor == sites[1]:
                path = (*prefix, neighbor)
                break
            if neighbor not in visited and neighbor not in blocked:
                visited.add(neighbor)
                frontier.append((neighbor, (*prefix, neighbor)))
        if path:
            break
    return {
        "satisfied": bool(path) and not direct,
        "reason": "direct_core_shortcut" if direct else None if path else "no_boundary_path",
        "core_maps": maps,
        "attachment_atoms": sites,
        "linker_path": path,
        "linker_internal_atoms": len(path) - 2 if path else None,
        "direct_core_shortcut": direct,
        "exact_core_contacts": True,
    }


def linker_fidelity(prompt: FragmentPrompt, smiles: str, *, max_pairs: int = 10000) -> dict:
    """Stricter path/interface diagnostic, separate from official validity.

    Match by explicit atom roles, allowing automorphisms but not overlapping
    cores. Exhausting the stated matching budget is reported, not silently
    treated as a chemical or generative-support failure. Stereo is outside
    the existing graph representation's guarantee.
    """
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    specs = _specs(prompt)
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return {"satisfied": False, "reason": "unparseable"}
    if len(Chem.GetMolFrags(molecule)) != 1:
        return {"satisfied": False, "reason": "disconnected"}
    matches = [
        molecule.GetSubstructMatches(
            s.core, uniquify=False, useChirality=False, maxMatches=max_pairs + 1
        )
        for s in specs
    ]
    if any(not m for m in matches):
        return {"satisfied": False, "reason": "missing_core"}
    result = {"satisfied": False, "reason": "no_disjoint_core_mapping"}
    for index, maps in enumerate(product(*matches)):
        if index >= max_pairs:
            return {"satisfied": False, "reason": "mapping_budget_exhausted"}
        current = _mapped_fidelity(molecule, specs, maps)
        if current["satisfied"]:
            return current
        # Keep a fully matched boundary witness (e.g. a direct shortcut) rather
        # than overwrite it with a later, wrong-site automorphism's refusal.
        if not result.get("exact_core_contacts"):
            result = current
    return result


def _connector(rooted_smiles: str):
    molecule = Chem.MolFromSmiles(rooted_smiles)
    if molecule is None:
        raise ValueError("connector is unparseable")
    dummies = [a for a in molecule.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2 or {a.GetIsotope() for a in dummies} != {1, 2}:
        raise ValueError("connector must have exactly the two boundary labels [1*] and [2*]")
    retained = [a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() != 0]
    if not retained:
        raise ValueError("connector must have at least one internal atom")
    index_map = {old: new for new, old in enumerate(retained)}
    roots = {}
    for dummy in dummies:
        if dummy.GetDegree() != 1 or dummy.GetNeighbors()[0].GetAtomicNum() == 0:
            raise ValueError("connector boundary must touch one internal atom")
        neighbor = dummy.GetNeighbors()[0].GetIdx()
        if (
            molecule.GetBondBetweenAtoms(dummy.GetIdx(), neighbor).GetBondType()
            != Chem.BondType.SINGLE
        ):
            raise ValueError("connector boundaries require single bonds")
        roots[dummy.GetIsotope()] = index_map[neighbor]
    opened = _fragment_spec(rooted_smiles).core
    if len(Chem.GetMolFrags(opened)) != 1:
        raise ValueError("connector internal graph must be connected")
    return opened, (roots[1], roots[2])


def assemble_linker_program(
    prompt: FragmentPrompt, rooted_connector: str, *, limits: ProposalLimits | None = None
) -> CompleteProgram:
    """Compile supplied two-ended content, with no artificial starting bridge."""
    specs = _specs(prompt)
    limits = limits or ProposalLimits()
    if limits.max_primitives > 32 or limits.max_blocks > 8:
        raise ValueError("linker assembly may not increase 32-primitive/eight-block support")
    opened = tuple(_open_core(text) for text in prompt.fragments)
    capped = tuple(_capped_core(text) for text in prompt.fragments)
    source_index = min(range(2), key=lambda i: (-capped[i].GetNumAtoms(), i))
    other = 1 - source_index
    connector, roots = _connector(rooted_connector)
    edit = Chem.RWMol(opened[source_index].molecule)
    source_atoms = tuple(range(edit.GetNumAtoms()))
    region_atoms = _append_molecule(edit, connector)
    other_atoms = _append_molecule(edit, opened[other].molecule)
    edit.AddBond(
        opened[source_index].attachment_sites[0],
        region_atoms[roots[source_index]],
        Chem.BondType.SINGLE,
    )
    edit.AddBond(
        region_atoms[roots[other]],
        other_atoms[opened[other].attachment_sites[0]],
        Chem.BondType.SINGLE,
    )
    molecule = _sanitize_candidate(edit)
    core_maps = [None, None]
    core_maps[source_index], core_maps[other] = source_atoms, other_atoms
    mapped_fidelity = _mapped_fidelity(molecule, specs, tuple(core_maps))
    if not mapped_fidelity["satisfied"]:
        raise ValueError(f"assembled mapped cores changed: {mapped_fidelity['reason']}")
    if molecule.GetNumHeavyAtoms() > limits.max_active_atoms:
        raise FragmentProposalAbstention("active_atom_budget", str(molecule.GetNumHeavyAtoms()))
    target, mapping = _mapped_graph(molecule)
    if target.n_real_atoms > limits.max_active_atoms:
        raise FragmentProposalAbstention("active_atom_budget", str(target.n_real_atoms))
    source = _source_graph(capped[source_index], source_atoms, mapping)
    groups = (
        _Group("proposed_two_boundary_region", region_atoms),
        _Group("supplied_second_core_assembly", other_atoms),
    )
    graph, assignment = _compile_groups(source, target, groups, mapping)
    if (
        len(graph.program.marks) > limits.max_primitives
        or len(graph.program.blocks) > limits.max_blocks
    ):
        raise FragmentProposalAbstention(
            "program_budget", "complete core/connector assembly exceeds support"
        )
    endpoint, trace = execute_program_graph(
        source,
        graph,
        assignment,
        max_primitives=limits.max_primitives,
        max_blocks=limits.max_blocks,
    )
    if canonical_state_key(endpoint) != canonical_state_key(target):
        raise RuntimeError("exact connector assembly endpoint differs from proposed content")
    lock = RegionLock(
        source,
        tuple(mapping[i] for i in source_atoms),
        preserve_effective_chemistry=True,
        allowed_external_slots=(mapping[opened[source_index].attachment_sites[0]],),
    )
    states = [decode_state(state) for state in trace["states"]]
    if not all(is_valid_state(s) and is_connected_or_null(s) for s in states):
        raise RuntimeError("connector compiler violated chemical/connected-state invariants")
    if not all(lock.permits(s) for s in states):
        raise ValueError("connector program changed the retained initial core")
    fidelity = linker_fidelity(prompt, canonical_state_key(endpoint))
    if not fidelity["satisfied"]:
        raise ValueError(f"connector endpoint violates boundary semantics: {fidelity['reason']}")
    stops = [0, *(block["stop"] for block in trace["blocks"])]
    return CompleteProgram(
        endpoint,
        trace,
        {
            "lane": "shared_linker_assembly",
            "connector": rooted_connector,
            "source_fragment_index": source_index,
            "source_core_locked_all_states": True,
            "second_core_preservation": "exact mapped endpoint; not present before assembly",
            "initial_bridge_atoms": 0,
            "initial_state_contains_only_one_core": True,
            "exact_mapped_core_identity_checked": True,
            "core_slot_maps": [tuple(mapping[i] for i in group) for group in core_maps],
            "fidelity": fidelity,
            "dependencies": list(graph.dependencies),
            "conflicts": list(graph.conflicts),
            "serialization_edges": list(graph.serialization_edges),
            "program_block_lengths": [right - left for left, right in pairwise(stops)],
            "program_block_labels": [block["label"] for block in trace["blocks"]],
            "reference_used_for_proposal": False,
        },
    )
