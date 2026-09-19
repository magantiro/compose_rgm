"""Zero-oracle COMPOSE proposals for retained-fragment prompts.

The runner conditions only on the released task label and fragment strings. It
never reads the benchmark's original drug while constructing a candidate.  A
proposal is a dependency-aware :class:`EditProgram` executed by the production
Editing-V2 executor, so every committed state remains a complete connected
molecule.  Linker prompts preserve one retained fragment as the source and
construct the bridge plus the second retained fragment inside one protected
program; disconnected fragment states are never committed.

This is a deliberately small proposal-support gate, not an optimization or
benchmark result.  It calls no objective, quality metric, docking function, or
learned reference law.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
from rdkit import Chem

from compose_v4.benchmark.fragment_constrained import (
    FragmentPrompt,
    FragmentTask,
    _fragment_spec,
    check_fragment_constraint,
)
from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    MAX_H_COUNT,
    MolecularGraph,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (
    empty_molecular_graph,
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.control.edit_program import (
    EditProgram,
    ProgramBlock,
    _json,
    _slots,
    atom_signature,
    environment,
)
from compose_v4.control.edit_program_graph import (
    ProgramGraph,
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import AtomInsert, CycleCloseEdge
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "compose_fragment_prompt_proposal_v1"
SMOKE_SCHEMA = "compose_fragment_prompt_smoke_v1"
SLOTS = 48


@dataclass(frozen=True)
class ProposalLimits:
    max_active_atoms: int = 40
    max_primitives: int = 32
    max_blocks: int = 8

    def __post_init__(self) -> None:
        if not 1 <= self.max_active_atoms <= 40:
            raise ValueError("max_active_atoms must be inside the frozen 1..40 support")
        if self.max_primitives < 1 or self.max_blocks < 1:
            raise ValueError("proposal primitive and block limits must be positive")


class FragmentProposalAbstention(ValueError):
    """The prompt is outside this bounded generic proposal construction."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class _OpenCore:
    molecule: Chem.Mol
    attachment_sites: tuple[int, ...]


@dataclass(frozen=True)
class _Group:
    label: str
    assembly_atoms: tuple[int, ...]


@dataclass(frozen=True)
class _CandidatePlan:
    target: Chem.Mol
    source_core: Chem.Mol
    source_assembly_atoms: tuple[int, ...]
    groups: tuple[_Group, ...]
    fragment_lock_groups: tuple[int, ...]
    source_fragment_index: int
    construction: str


@dataclass(frozen=True)
class _CompiledPlan:
    source: MolecularGraph
    target: MolecularGraph
    program_graph: ProgramGraph
    assignment: tuple[int, ...]
    fragment_lock_groups: tuple[int, ...]
    construction: str
    source_fragment_index: int


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def prompt_id(prompt: FragmentPrompt) -> str:
    """Address-free prompt identity; the original benchmark drug is excluded."""

    payload = {"task": prompt.task.value, "fragments": list(prompt.fragments)}
    return hashlib.sha256(_stable_json(payload).encode()).hexdigest()


def _open_core(fragment: str) -> _OpenCore:
    spec = _fragment_spec(fragment)
    sites = tuple(
        site for site, count in spec.attachment_requirements for _ in range(count)
    )
    return _OpenCore(Chem.Mol(spec.core), sites)


def _capped_core(fragment: str) -> Chem.Mol:
    """Replace each released dummy by H, then remove explicit H for a valid source."""

    molecule = Chem.MolFromSmiles(fragment)
    if molecule is None:
        raise FragmentProposalAbstention("invalid_fragment", fragment)
    editable = Chem.RWMol(molecule)
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0:
            editable.ReplaceAtom(atom.GetIdx(), Chem.Atom(1))
    capped = editable.GetMol()
    try:
        Chem.SanitizeMol(capped)
        capped = Chem.RemoveHs(capped)
        Chem.SanitizeMol(capped)
    except Exception as error:
        raise FragmentProposalAbstention(
            "unsupported_capped_fragment", str(error)
        ) from error
    return capped


def _append_molecule(editable: Chem.RWMol, molecule: Chem.Mol) -> tuple[int, ...]:
    offset = editable.GetNumAtoms()
    for atom in molecule.GetAtoms():
        copied = Chem.Atom(atom)
        copied.SetAtomMapNum(0)
        editable.AddAtom(copied)
    for bond in molecule.GetBonds():
        editable.AddBond(
            offset + bond.GetBeginAtomIdx(),
            offset + bond.GetEndAtomIdx(),
            bond.GetBondType(),
        )
    return tuple(range(offset, offset + molecule.GetNumAtoms()))


def _sanitize_candidate(editable: Chem.RWMol) -> Chem.Mol:
    candidate = editable.GetMol()
    try:
        Chem.SanitizeMol(candidate)
    except Exception as error:
        raise FragmentProposalAbstention(
            "candidate_chemistry_unsupported", str(error)
        ) from error
    if len(Chem.GetMolFrags(candidate)) != 1:
        raise FragmentProposalAbstention(
            "disconnected_candidate", "the generic construction did not join every core"
        )
    return candidate


def _linker_plan(prompt: FragmentPrompt, variant_index: int) -> _CandidatePlan:
    if len(prompt.fragments) != 2:
        raise FragmentProposalAbstention(
            "malformed_linker_prompt", "exactly two retained fragments are required"
        )
    opens = tuple(_open_core(fragment) for fragment in prompt.fragments)
    if any(len(core.attachment_sites) != 1 for core in opens):
        raise FragmentProposalAbstention(
            "unsupported_linker_attachment_arity",
            "each retained linker fragment must declare exactly one open site",
        )
    capped = tuple(_capped_core(fragment) for fragment in prompt.fragments)
    # Preserve the larger retained component and construct the smaller one. This
    # is a task-independent work reduction, not a lookup of the reference drug.
    source_index = min(
        range(2), key=lambda index: (-capped[index].GetNumAtoms(), index)
    )
    other_index = 1 - source_index
    editable = Chem.RWMol(opens[source_index].molecule)
    source_atoms = tuple(range(editable.GetNumAtoms()))

    chain_length = 1 + variant_index % 3
    chain = tuple(editable.AddAtom(Chem.Atom(6)) for _ in range(chain_length))
    editable.AddBond(
        opens[source_index].attachment_sites[0], chain[0], Chem.BondType.SINGLE
    )
    for left, right in pairwise(chain):
        editable.AddBond(left, right, Chem.BondType.SINGLE)

    other_atoms = _append_molecule(editable, opens[other_index].molecule)
    editable.AddBond(
        chain[-1],
        other_atoms[opens[other_index].attachment_sites[0]],
        Chem.BondType.SINGLE,
    )
    groups = [_Group("generic_linker_path", chain)]
    groups.append(_Group("retained_fragment_construction", other_atoms))
    # The source fragment's declared site is complete after the chain block;
    # the constructed fragment locks after its own block.
    locks_by_fragment = [0, 0]
    locks_by_fragment[source_index] = 0
    locks_by_fragment[other_index] = 1
    return _CandidatePlan(
        _sanitize_candidate(editable),
        capped[source_index],
        source_atoms,
        tuple(groups),
        tuple(locks_by_fragment),
        source_index,
        "preserve_one_core_then_build_generic_linker_and_second_core",
    )


def _first_extendable_atom(molecule: Chem.Mol) -> int:
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() > 1 and atom.GetTotalNumHs() > 0:
            return atom.GetIdx()
    raise FragmentProposalAbstention(
        "no_generic_extension_site", "the retained core has no hydrogen-bearing atom"
    )


def _single_fragment_plan(prompt: FragmentPrompt, variant_index: int) -> _CandidatePlan:
    if len(prompt.fragments) != 1:
        raise FragmentProposalAbstention(
            "malformed_single_fragment_prompt",
            "exactly one retained fragment is required",
        )
    opened = _open_core(prompt.fragments[0])
    capped = _capped_core(prompt.fragments[0])
    editable = Chem.RWMol(opened.molecule)
    source_atoms = tuple(range(editable.GetNumAtoms()))
    groups: list[_Group] = []

    if prompt.task is FragmentTask.SUPERSTRUCTURE_GENERATION:
        if opened.attachment_sites:
            raise FragmentProposalAbstention(
                "unexpected_superstructure_dummy",
                "the frozen superstructure input must not invent a fixed site",
            )
        site = _first_extendable_atom(capped)
        chain_length = 1 + variant_index % 3
        chain = tuple(editable.AddAtom(Chem.Atom(6)) for _ in range(chain_length))
        editable.AddBond(site, chain[0], Chem.BondType.SINGLE)
        for left, right in pairwise(chain):
            editable.AddBond(left, right, Chem.BondType.SINGLE)
        groups.append(_Group("generic_superstructure_extension", chain))
        lock_group = -1  # Core containment holds in the source and every successor.
        construction = "preserve_core_then_extend_generic_hydrogen_site"
    else:
        if not opened.attachment_sites:
            raise FragmentProposalAbstention(
                "missing_declared_attachment",
                f"{prompt.task.value} requires an open site",
            )
        chain_length = (
            1 + variant_index % 3 if prompt.task is FragmentTask.MOTIF_EXTENSION else 1
        )
        first_site, *remaining_sites = opened.attachment_sites
        chain = tuple(editable.AddAtom(Chem.Atom(6)) for _ in range(chain_length))
        editable.AddBond(first_site, chain[0], Chem.BondType.SINGLE)
        for left, right in pairwise(chain):
            editable.AddBond(left, right, Chem.BondType.SINGLE)
        groups.append(_Group("generic_declared_site_extension", chain))
        for index, site in enumerate(remaining_sites):
            atom = editable.AddAtom(Chem.Atom(6))
            editable.AddBond(site, atom, Chem.BondType.SINGLE)
            groups.append(_Group(f"generic_decoration_site_{index + 1}", (atom,)))
        lock_group = len(groups) - 1
        construction = "preserve_retained_core_then_complete_declared_sites"

    return _CandidatePlan(
        _sanitize_candidate(editable),
        capped,
        source_atoms,
        tuple(groups),
        (lock_group,),
        0,
        construction,
    )


def _candidate_plan(prompt: FragmentPrompt, variant_index: int) -> _CandidatePlan:
    if type(variant_index) is not int or variant_index < 0:
        raise ValueError("variant_index must be a nonnegative integer")
    if prompt.task in {FragmentTask.LINKER_DESIGN, FragmentTask.SCAFFOLD_MORPHING}:
        return _linker_plan(prompt, variant_index)
    return _single_fragment_plan(prompt, variant_index)


def _mapped_graph(molecule: Chem.Mol) -> tuple[MolecularGraph, dict[int, int]]:
    mapped = Chem.Mol(molecule)
    for index, atom in enumerate(mapped.GetAtoms()):
        atom.SetAtomMapNum(index + 1)
    mapped_smiles = Chem.MolToSmiles(mapped, canonical=False, isomericSmiles=False)
    reparsed = Chem.MolFromSmiles(mapped_smiles)
    if reparsed is None:
        raise FragmentProposalAbstention(
            "mapped_serialization_failed", "RDKit rejected its own mapped serialization"
        )
    assembly_to_slot = {
        atom.GetAtomMapNum() - 1: atom.GetIdx() for atom in reparsed.GetAtoms()
    }
    if set(assembly_to_slot) != set(range(mapped.GetNumAtoms())):
        raise FragmentProposalAbstention(
            "mapped_serialization_drift", "atom-map lineage was not bijective"
        )
    graph = pad_molecular_graph(smiles_to_molecular_graph(mapped_smiles), SLOTS)
    return graph, assembly_to_slot


def _source_graph(
    source_core: Chem.Mol,
    source_assembly_atoms: tuple[int, ...],
    assembly_to_target_slot: dict[int, int],
) -> MolecularGraph:
    if source_core.GetNumAtoms() != len(source_assembly_atoms):
        raise FragmentProposalAbstention(
            "source_mapping_mismatch", "capped and open retained cores changed size"
        )
    mapped = Chem.Mol(source_core)
    for local, atom in enumerate(mapped.GetAtoms()):
        atom.SetAtomMapNum(source_assembly_atoms[local] + 1)
    smiles = Chem.MolToSmiles(mapped, canonical=False, isomericSmiles=False)
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise FragmentProposalAbstention(
            "source_serialization_failed", "RDKit rejected the capped retained source"
        )
    local_graph = smiles_to_molecular_graph(smiles)
    local_to_target = {
        atom.GetIdx(): assembly_to_target_slot[atom.GetAtomMapNum() - 1]
        for atom in parsed.GetAtoms()
    }
    blank = empty_molecular_graph(SLOTS)
    atom_types = blank.atom_types.copy()
    charges = blank.formal_charges.copy()
    hydrogens = blank.implicit_h_counts.copy()
    bonds = blank.bonds.copy()
    for local, target in local_to_target.items():
        atom_types[target] = local_graph.atom_types[local]
        charges[target] = local_graph.formal_charges[local]
        hydrogens[target] = local_graph.implicit_h_counts[local]
    for left, target_left in local_to_target.items():
        for right, target_right in local_to_target.items():
            bonds[target_left, target_right] = local_graph.bonds[left, right]
    source = MolecularGraph(atom_types, charges, hydrogens, bonds)
    if not is_valid_state(source) or not is_connected_or_null(source):
        raise FragmentProposalAbstention(
            "source_outside_runtime_support",
            "the capped retained core is not valid and connected",
        )
    return source


def _edge(left: int, right: int) -> tuple[int, int]:
    return tuple(sorted((left, right)))


def _compile_groups(
    source: MolecularGraph,
    target: MolecularGraph,
    groups: tuple[_Group, ...],
    assembly_to_slot: dict[int, int],
) -> tuple[ProgramGraph, tuple[int, ...]]:
    built = {int(slot) for slot in np.flatnonzero(is_element(source.atom_types))}
    committed = {
        _edge(left, right)
        for left in built
        for right in built
        if left < right and int(source.bonds[left, right]) != 0
    }
    roots = sorted(
        {
            int(neighbor)
            for group in groups
            for assembly in group.assembly_atoms
            for neighbor in np.flatnonzero(
                target.bonds[assembly_to_slot[assembly]] != 0
            )
            if int(neighbor) in built
        }
    )
    if not roots:
        raise FragmentProposalAbstention(
            "no_source_boundary",
            "no generated component attaches to the retained source",
        )
    handles = {slot: {"input": index} for index, slot in enumerate(roots)}
    created = 0
    marks: list[str] = []
    blocks: list[ProgramBlock] = []

    for group in groups:
        vertices = {assembly_to_slot[index] for index in group.assembly_atoms}
        boundaries = sorted(
            (vertex, int(neighbor))
            for vertex in vertices
            for neighbor in np.flatnonzero(target.bonds[vertex] != 0)
            if int(neighbor) in built
        )
        if not boundaries:
            raise FragmentProposalAbstention(
                "group_dependency_unbound",
                f"{group.label} has no earlier connected anchor",
            )
        root, root_parent = boundaries[0]
        parents = {root: root_parent}
        order: list[int] = []
        stack = [root]
        while stack:
            vertex = stack.pop()
            order.append(vertex)
            neighbors = sorted(
                (
                    int(other)
                    for other in np.flatnonzero(target.bonds[vertex] != 0)
                    if int(other) in vertices and int(other) not in parents
                ),
                reverse=True,
            )
            for other in neighbors:
                parents[other] = vertex
                stack.append(other)
        if set(order) != vertices:
            raise FragmentProposalAbstention(
                "disconnected_program_group",
                f"{group.label} is not internally connected",
            )

        for vertex in order:
            parent = parents[vertex]
            bond_order = int(target.bonds[vertex, parent])
            missing_load = sum(
                BOND_CLASS_TO_H_CHANGE[int(target.bonds[vertex, other])]
                for other in np.flatnonzero(target.bonds[vertex] != 0)
                if int(other) != parent
            )
            insertion_h = int(target.implicit_h_counts[vertex]) + int(missing_load)
            if not 0 <= insertion_h <= MAX_H_COUNT:
                raise FragmentProposalAbstention(
                    "transient_hydrogen_outside_support",
                    f"{group.label} slot {vertex} needs transient H={insertion_h}",
                )
            handles[vertex] = {"created": created}
            created += 1
            action = AtomInsert(
                slot=vertex,
                atom_type=int(target.atom_types[vertex]),
                formal_charge=int(target.formal_charges[vertex]),
                implicit_h_count=insertion_h,
                neighbors=((parent, bond_order),),
            )
            record = encode_action("atom_insert", action)
            marks.append(_json(_slots(record, lambda slot: handles[slot])))
            committed.add(_edge(vertex, parent))

        available = built | vertices
        closures = sorted(
            _edge(left, right)
            for left in vertices
            for right in available
            if left < right
            and int(target.bonds[left, right]) != 0
            and _edge(left, right) not in committed
        )
        for left, right in closures:
            action = CycleCloseEdge(left, right, int(target.bonds[left, right]))
            record = encode_action("cycle_close", action)
            marks.append(_json(_slots(record, lambda slot: handles[slot])))
            committed.add((left, right))
        built.update(vertices)
        blocks.append(ProgramBlock(group.label, len(marks)))

    if built != {int(slot) for slot in np.flatnonzero(is_element(target.atom_types))}:
        raise FragmentProposalAbstention(
            "incomplete_program", "the program did not account for every target atom"
        )
    program = EditProgram(
        tuple(atom_signature(source, slot) for slot in roots),
        tuple(tuple(int(source.bonds[a, b]) for b in roots) for a in roots),
        tuple(environment(source, slot) for slot in roots),
        tuple(marks),
        tuple(blocks),
    )
    return compile_program_graph(program), tuple(roots)


def compile_prompt_program(
    prompt: FragmentPrompt,
    *,
    variant_index: int = 0,
    limits: ProposalLimits | None = None,
) -> _CompiledPlan:
    """Compile one prompt without reading its original/reference molecule."""

    limits = limits or ProposalLimits()
    plan = _candidate_plan(prompt, variant_index)
    target, assembly_to_slot = _mapped_graph(plan.target)
    if target.n_real_atoms > limits.max_active_atoms:
        raise FragmentProposalAbstention(
            "active_atom_budget",
            f"candidate has {target.n_real_atoms} atoms, limit {limits.max_active_atoms}",
        )
    source = _source_graph(
        plan.source_core, plan.source_assembly_atoms, assembly_to_slot
    )
    graph, assignment = _compile_groups(source, target, plan.groups, assembly_to_slot)
    if len(graph.program.marks) > limits.max_primitives:
        raise FragmentProposalAbstention(
            "primitive_budget",
            f"program has {len(graph.program.marks)} primitives, limit {limits.max_primitives}",
        )
    if len(graph.program.blocks) > limits.max_blocks:
        raise FragmentProposalAbstention(
            "block_budget",
            f"program has {len(graph.program.blocks)} blocks, limit {limits.max_blocks}",
        )
    return _CompiledPlan(
        source,
        target,
        graph,
        assignment,
        plan.fragment_lock_groups,
        plan.construction,
        plan.source_fragment_index,
    )


def _fragment_lock_audit(
    prompt: FragmentPrompt,
    receipt: dict,
    lock_groups: tuple[int, ...],
) -> list[dict]:
    blocks = receipt["blocks"]
    rows = []
    for index, (fragment, group_index) in enumerate(
        zip(prompt.fragments, lock_groups, strict=True)
    ):
        start = 0 if group_index < 0 else int(blocks[group_index]["stop"])
        one_fragment = FragmentPrompt(
            drug_name=prompt.drug_name,
            original_smiles=prompt.original_smiles,
            task=FragmentTask.MOTIF_EXTENSION,
            fragments=(fragment,),
        )
        checks = [
            check_fragment_constraint(
                one_fragment, canonical_state_key(decode_state(state))
            ).satisfied
            for state in receipt["states"][start:]
        ]
        rows.append(
            {
                "fragment_index": index,
                "lock_after_primitive": start,
                "checked_committed_states": len(checks),
                "all_locked_states_satisfied": bool(checks and all(checks)),
            }
        )
    return rows


def propose_prompt(
    prompt: FragmentPrompt,
    *,
    variant_index: int = 0,
    limits: ProposalLimits | None = None,
) -> dict:
    """Execute one complete prompt-conditioned proposal, or report abstention."""

    limits = limits or ProposalLimits()
    identity = prompt_id(prompt)
    base = {
        "schema_version": SCHEMA,
        "prompt_id": identity,
        "drug_name": prompt.drug_name,
        "task": prompt.task.value,
        "variant_index": variant_index,
        "conditioning_fields": ["task", "fragments", "variant_index"],
        "reference_original_used_for_proposal": False,
        "oracle_calls": 0,
        "scoring_calls": 0,
        "limits": asdict(limits),
    }
    try:
        compiled = compile_prompt_program(
            prompt, variant_index=variant_index, limits=limits
        )
        product, receipt = execute_program_graph(
            compiled.source,
            compiled.program_graph,
            compiled.assignment,
            max_primitives=limits.max_primitives,
            max_blocks=limits.max_blocks,
        )
        if canonical_state_key(product) != canonical_state_key(compiled.target):
            raise RuntimeError("executed endpoint differs from the compiled target")
        committed = tuple(decode_state(state) for state in receipt["states"])
        if not all(is_valid_state(state) for state in committed):
            raise RuntimeError("a committed program state is chemically invalid")
        if not all(is_connected_or_null(state) for state in committed):
            raise RuntimeError("a committed program state is disconnected")
        constraint = check_fragment_constraint(prompt, receipt["endpoint"])
        locks = _fragment_lock_audit(prompt, receipt, compiled.fragment_lock_groups)
        if not constraint.satisfied or not all(
            row["all_locked_states_satisfied"] for row in locks
        ):
            raise RuntimeError(
                "exact fragment constraint or post-lock invariant failed"
            )
        return {
            **base,
            "status": "complete",
            "construction": compiled.construction,
            "source_fragment_index": compiled.source_fragment_index,
            "source": {
                "canonical_smiles": canonical_state_key(compiled.source),
                "active_atoms": compiled.source.n_real_atoms,
            },
            "endpoint": receipt["endpoint"],
            "endpoint_active_atoms": product.n_real_atoms,
            "program": {
                "program_id": receipt["program_id"],
                "graph_id": receipt["graph_id"],
                "primitive_edits": receipt["primitive_edits"],
                "blocks": receipt["blocks"],
                "dependencies": compiled.program_graph.dependencies,
                "serialization_edges": compiled.program_graph.serialization_edges,
            },
            "validation": {
                "chemical_valid": True,
                "connected_or_null_all_committed_states": True,
                "fragment_constraint_satisfied": True,
                "constraint_reason": constraint.reason,
                "fragment_locks": locks,
                "committed_states": len(committed),
            },
            "receipt": receipt,
        }
    except FragmentProposalAbstention as error:
        return {
            **base,
            "status": "abstained",
            "reason_code": error.reason_code,
            "detail": error.detail,
        }


def smoke_prompts(prompts: tuple[FragmentPrompt, ...]) -> tuple[FragmentPrompt, ...]:
    """Freeze the first manifest prompt for each task before observing outcomes."""

    selected = []
    for task in FragmentTask:
        matches = [prompt for prompt in prompts if prompt.task is task]
        if not matches:
            raise ValueError(f"frozen prompt manifest is missing task {task.value}")
        selected.append(matches[0])
    return tuple(selected)


def run_smoke_panel(
    prompts: tuple[FragmentPrompt, ...],
    *,
    variant_index: int = 0,
    limits: ProposalLimits | None = None,
) -> dict:
    """Run exactly five zero-oracle proposals, one per frozen task label."""

    selected = smoke_prompts(prompts)
    rows = [
        propose_prompt(prompt, variant_index=variant_index, limits=limits)
        for prompt in selected
    ]
    return {
        "schema_version": SMOKE_SCHEMA,
        "selection": "first manifest prompt per task before proposal execution",
        "frozen_manifest_prompt_count": len(prompts),
        "attempted_prompts": len(rows),
        "completed": sum(row["status"] == "complete" for row in rows),
        "abstained": sum(row["status"] == "abstained" for row in rows),
        "oracle_calls": 0,
        "scoring_calls": 0,
        "proposals": rows,
    }


__all__ = [
    "SCHEMA",
    "SMOKE_SCHEMA",
    "FragmentProposalAbstention",
    "ProposalLimits",
    "compile_prompt_program",
    "prompt_id",
    "propose_prompt",
    "run_smoke_panel",
    "smoke_prompts",
]
