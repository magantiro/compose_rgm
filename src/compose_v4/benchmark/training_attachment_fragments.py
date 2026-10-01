"""Train-only rooted substituents compiled through exact COMPOSE atom insertions.

This is an opt-in *proposal lane*, not a chemical-validity repair or a change to
the frozen fragment sampler.  Only the benchmark-declared attachment interface
is used at inference.  QED, SA and the original drug never enter this module.
The first development version deliberately supports neutral acyclic C/N/O/F
fragments with one single-bond attachment; unsupported source fragments are
counted, not silently coerced into this representation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import BRICS

from compose_v4.benchmark.fragment_conditioned_sampler import (
    PromptContext,
    RegionLock,
)
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    _fragment_spec,
    check_fragment_constraint,
)
from compose_v4.chem.molecular_graph import (
    ALLOWED_VALENCES,
    ELEMENT_TO_IDX,
    MolecularGraph,
    molecular_graph_to_smiles,
)
from compose_v4.experiments.whole_ring_plan import fresh_slot
from compose_v4.rewrite.operators import AtomInsert

CATALOG_SCHEMA = "training_attachment_fragments_v1"
SUPPORTED_ELEMENTS = frozenset({"C", "N", "O", "F"})
SUPPORTED_TASKS = frozenset({FragmentTask.MOTIF_EXTENSION, FragmentTask.SCAFFOLD_DECORATION})


def physical_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atom_context(atom: Chem.Atom) -> str:
    """A coarse, task-blind proximal interface class available on both sides."""

    return f"{atom.GetSymbol()}:{int(atom.GetIsAromatic())}:{int(atom.IsInRing())}"


def _component_across_cut(mol: Chem.Mol, root: int, near: int) -> tuple[int, ...] | None:
    seen = {int(root)}
    stack = [int(root)]
    while stack:
        current = stack.pop()
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            other = int(neighbor.GetIdx())
            if {current, other} == {root, near}:
                continue
            if other not in seen:
                seen.add(other)
                stack.append(other)
    if near in seen:
        return None
    return tuple(sorted(seen))


def _rooted_fragment_smiles(
    mol: Chem.Mol, root: int, atoms: tuple[int, ...]
) -> tuple[str | None, str | None]:
    if not atoms:
        return None, "empty"
    if any(
        mol.GetAtomWithIdx(index).GetSymbol() not in SUPPORTED_ELEMENTS
        or mol.GetAtomWithIdx(index).GetFormalCharge() != 0
        or mol.GetAtomWithIdx(index).GetIsAromatic()
        or mol.GetAtomWithIdx(index).IsInRing()
        or mol.GetAtomWithIdx(index).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for index in atoms
    ):
        return None, "atom_outside_declared_support"
    old_to_new = {old: new for new, old in enumerate(atoms)}
    fragment = Chem.RWMol()
    for old in atoms:
        fragment.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old).GetSymbol()))
    edge_count = 0
    for bond in mol.GetBonds():
        left, right = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        if left not in old_to_new or right not in old_to_new:
            continue
        if bond.GetIsAromatic() or bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return None, "bond_outside_declared_support"
        if bond.GetBondType() not in (
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ):
            return None, "bond_outside_declared_support"
        fragment.AddBond(old_to_new[left], old_to_new[right], bond.GetBondType())
        edge_count += 1
    if edge_count != len(atoms) - 1:
        return None, "not_a_tree"
    dummy = fragment.AddAtom(Chem.Atom(0))
    fragment.AddBond(dummy, old_to_new[root], Chem.BondType.SINGLE)
    result = fragment.GetMol()
    try:
        Chem.SanitizeMol(result)
    except (ValueError, RuntimeError):
        return None, "fragment_does_not_sanitize"
    return Chem.MolToSmiles(result, canonical=True), None


@dataclass(frozen=True)
class FragmentEntry:
    context: str
    rooted_smiles: str
    heavy_atoms: int
    occurrences: int
    source_rows: tuple[int, ...]


@dataclass(frozen=True)
class AttachmentCatalog:
    source_sha256: str
    source_path: str
    source_rows: int
    unique_source_molecules: int
    duplicate_source_rows: int
    excluded_reference_rows: tuple[int, ...]
    excluded: dict[str, int]
    entries: tuple[FragmentEntry, ...]
    rdkit_version: str
    max_fragment_atoms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CATALOG_SCHEMA,
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "source_rows": self.source_rows,
            "unique_source_molecules": self.unique_source_molecules,
            "duplicate_source_rows": self.duplicate_source_rows,
            "excluded_reference_rows": list(self.excluded_reference_rows),
            "excluded": dict(sorted(self.excluded.items())),
            "entries": [asdict(entry) for entry in self.entries],
            "rdkit_version": self.rdkit_version,
            "max_fragment_atoms": self.max_fragment_atoms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttachmentCatalog:
        if data.get("schema") != CATALOG_SCHEMA:
            raise ValueError(f"unsupported attachment catalog schema: {data.get('schema')!r}")
        entries = tuple(
            FragmentEntry(
                context=str(row["context"]),
                rooted_smiles=str(row["rooted_smiles"]),
                heavy_atoms=int(row["heavy_atoms"]),
                occurrences=int(row["occurrences"]),
                source_rows=tuple(int(value) for value in row["source_rows"]),
            )
            for row in data["entries"]
        )
        if tuple(sorted(entries, key=lambda row: (row.context, row.rooted_smiles))) != entries:
            raise ValueError("attachment catalog entries are not deterministically ordered")
        return cls(
            source_sha256=str(data["source_sha256"]),
            source_path=str(data["source_path"]),
            source_rows=int(data["source_rows"]),
            unique_source_molecules=int(data["unique_source_molecules"]),
            duplicate_source_rows=int(data["duplicate_source_rows"]),
            excluded_reference_rows=tuple(int(value) for value in data["excluded_reference_rows"]),
            excluded=dict(data["excluded"]),
            entries=entries,
            rdkit_version=str(data["rdkit_version"]),
            max_fragment_atoms=int(data["max_fragment_atoms"]),
        )


def build_attachment_catalog(
    source: Path,
    *,
    expected_sha256: str,
    max_fragment_atoms: int = 6,
    excluded_canonical: frozenset[str] = frozenset(),
) -> AttachmentCatalog:
    """Count rooted BRICS-side trees from a hash-bound training-only corpus."""

    if not 1 <= max_fragment_atoms <= 6:
        raise ValueError("max_fragment_atoms must be in [1, 6] for this pilot support")
    observed_sha256 = physical_sha256(source)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"training source {source} SHA-256 {observed_sha256} != {expected_sha256}")
    occurrences: dict[tuple[str, str], list[int]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    seen_source: set[str] = set()
    source_rows = 0
    duplicate_source_rows = 0
    excluded_reference_rows: list[int] = []
    with source.open(encoding="utf-8") as handle:
        for source_rows, line in enumerate(handle, 1):
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                excluded["unparseable_source_row"] += 1
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical in excluded_canonical:
                excluded["benchmark_reference_overlap"] += 1
                excluded_reference_rows.append(source_rows)
                continue
            if canonical in seen_source:
                duplicate_source_rows += 1
                continue
            seen_source.add(canonical)
            for (left, right), _labels in BRICS.FindBRICSBonds(mol):
                bond = mol.GetBondBetweenAtoms(int(left), int(right))
                if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                    excluded["cut_not_single_bond"] += 2
                    continue
                for near, root in ((int(left), int(right)), (int(right), int(left))):
                    atoms = _component_across_cut(mol, root, near)
                    if atoms is None:
                        excluded["cut_not_bridge"] += 1
                        continue
                    if len(atoms) > max_fragment_atoms:
                        excluded["too_many_fragment_atoms"] += 1
                        continue
                    rooted_smiles, reason = _rooted_fragment_smiles(mol, root, atoms)
                    if rooted_smiles is None:
                        excluded[str(reason)] += 1
                        continue
                    context = atom_context(mol.GetAtomWithIdx(near))
                    occurrences[(context, rooted_smiles)].append(source_rows)
    entries = tuple(
        FragmentEntry(
            context=context,
            rooted_smiles=rooted_smiles,
            heavy_atoms=Chem.MolFromSmiles(rooted_smiles).GetNumAtoms() - 1,
            occurrences=len(rows),
            source_rows=tuple(rows),
        )
        for (context, rooted_smiles), rows in sorted(occurrences.items())
    )
    return AttachmentCatalog(
        source_sha256=observed_sha256,
        source_path=str(source),
        source_rows=source_rows,
        unique_source_molecules=len(seen_source),
        duplicate_source_rows=duplicate_source_rows,
        excluded_reference_rows=tuple(excluded_reference_rows),
        excluded=dict(sorted(excluded.items())),
        entries=entries,
        rdkit_version=rdBase.rdkitVersion,
        max_fragment_atoms=max_fragment_atoms,
    )


def catalog_bytes(catalog: AttachmentCatalog) -> bytes:
    return (json.dumps(catalog.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()


@dataclass(frozen=True)
class CatalogRuntime:
    """One read-only index shared by all attempts; never rebuilt in the hot path."""

    catalog: AttachmentCatalog
    by_context: dict[str, tuple[FragmentEntry, ...]]


def prepare_catalog(catalog: AttachmentCatalog) -> CatalogRuntime:
    grouped: dict[str, list[FragmentEntry]] = defaultdict(list)
    for entry in catalog.entries:
        grouped[entry.context].append(entry)
    return CatalogRuntime(catalog, {key: tuple(rows) for key, rows in grouped.items()})


def _fragment_order(fragment: Chem.Mol) -> tuple[int, ...]:
    dummy = [atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy) != 1 or fragment.GetAtomWithIdx(dummy[0]).GetDegree() != 1:
        raise ValueError("rooted fragment must have exactly one terminal dummy")
    root = fragment.GetAtomWithIdx(dummy[0]).GetNeighbors()[0].GetIdx()
    order: list[int] = []
    queue = [int(root)]
    seen = {int(dummy[0]), int(root)}
    while queue:
        current = queue.pop(0)
        order.append(current)
        for neighbor in sorted(
            fragment.GetAtomWithIdx(current).GetNeighbors(), key=lambda a: a.GetIdx()
        ):
            index = int(neighbor.GetIdx())
            if index not in seen:
                seen.add(index)
                queue.append(index)
    if len(order) != fragment.GetNumAtoms() - 1:
        raise ValueError("rooted fragment is not one connected tree")
    return tuple(order)


def graft_observed_fragment(
    state: MolecularGraph,
    *,
    site: int,
    rooted_smiles: str,
    system: Any,
    lock: RegionLock,
) -> tuple[MolecularGraph, tuple[AtomInsert, ...]]:
    """Privately compile and execute one full fragment through exact rewrites."""

    fragment = Chem.MolFromSmiles(rooted_smiles)
    if fragment is None:
        raise ValueError(f"invalid rooted fragment {rooted_smiles!r}")
    order = _fragment_order(fragment)
    if state.n_real_atoms + len(order) > 40:
        raise ValueError("fragment exceeds unchanged 40-active-atom support")
    old_to_slot: dict[int, int] = {}
    actions: list[AtomInsert] = []
    current = state
    for index in order:
        atom = fragment.GetAtomWithIdx(index)
        # A rooted tree has exactly one predecessor: the dummy for the root,
        # or an already created atom for every later vertex.
        predecessors = [
            bond
            for bond in atom.GetBonds()
            if bond.GetOtherAtomIdx(index) in old_to_slot
            or fragment.GetAtomWithIdx(bond.GetOtherAtomIdx(index)).GetAtomicNum() == 0
        ]
        if len(predecessors) != 1:
            raise ValueError("fragment ordering did not expose one predecessor")
        predecessor = predecessors[0]
        parent = int(predecessor.GetOtherAtomIdx(index))
        bond_order = int(predecessor.GetBondTypeAsDouble())
        if bond_order not in (1, 2, 3):
            raise ValueError("fragment bond order outside declared support")
        symbol = atom.GetSymbol()
        if symbol not in SUPPORTED_ELEMENTS or atom.GetFormalCharge() != 0:
            raise ValueError("fragment atom outside declared neutral C/N/O/F support")
        allowed = ALLOWED_VALENCES[symbol]
        if len(allowed) != 1:
            raise ValueError(f"ambiguous neutral valence for {symbol}")
        action = AtomInsert(
            slot=fresh_slot(current),
            atom_type=ELEMENT_TO_IDX[symbol],
            formal_charge=0,
            implicit_h_count=allowed[0] - bond_order,
            neighbors=((old_to_slot.get(parent, site), bond_order),),
        )
        successor = system.apply(current, "atom_insert", action)
        if not lock.permits(successor):
            raise ValueError("fragment insertion would violate the retained-core lock")
        current = successor
        old_to_slot[index] = action.slot
        actions.append(action)
    return current, tuple(actions)


@dataclass
class CatalogCompletionReceipt:
    attempted_fragments: list[dict[str, Any]] = field(default_factory=list)
    accepted_actions: list[dict[str, Any]] = field(default_factory=list)
    fallback_count: int = 0
    failure_reason: str | None = None
    committed_smiles: str | None = None


def sample_catalog_completion(
    context: PromptContext,
    catalog: AttachmentCatalog | CatalogRuntime,
    system: Any,
    rng: np.random.Generator,
) -> tuple[str | None, CatalogCompletionReceipt]:
    """One constructive attempt for motif/decorate; no score-based rejection."""

    if context.prompt.task not in SUPPORTED_TASKS:
        raise ValueError(f"catalog lane does not support {context.prompt.task.value}")
    spec = context.attachment
    if spec is None or not spec.requirements:
        raise ValueError("catalog lane requires declared attachment sites")
    original = Chem.MolFromSmiles(context.start_smiles)
    if original is None:
        raise ValueError("invalid prompt start SMILES")
    lock = RegionLock(
        context.start_state,
        context.locked_slots,
        preserve_effective_chemistry=True,
        allowed_external_slots=spec.interfaces,
        lock_groups=spec.lock_groups,
        fragment_queries=tuple(
            _fragment_spec(fragment).core for fragment in context.prompt.fragments
        ),
    )
    runtime = catalog if isinstance(catalog, CatalogRuntime) else prepare_catalog(catalog)
    current = context.start_state
    receipt = CatalogCompletionReceipt()
    for site, required_count in spec.requirements:
        context_key = atom_context(original.GetAtomWithIdx(int(site)))
        for _ in range(int(required_count)):
            room = 40 - current.n_real_atoms
            choices = [
                entry
                for entry in runtime.by_context.get(context_key, ())
                if entry.heavy_atoms <= room
            ]
            chosen: FragmentEntry | None = None
            if choices:
                weights = np.sqrt(np.asarray([entry.occurrences for entry in choices], dtype=float))
                weights /= weights.sum()
                chosen = choices[int(rng.choice(len(choices), p=weights))]
            rooted_smiles = chosen.rooted_smiles if chosen is not None else "*C"
            attempt = {
                "site": int(site),
                "context": context_key,
                "fragment": rooted_smiles,
                "training_occurrences": chosen.occurrences if chosen is not None else 0,
            }
            try:
                successor, actions = graft_observed_fragment(
                    current,
                    site=int(site),
                    rooted_smiles=rooted_smiles,
                    system=system,
                    lock=lock,
                )
            except (ValueError, RuntimeError) as exc:
                attempt["refusal"] = str(exc)
                receipt.attempted_fragments.append(attempt)
                if rooted_smiles == "*C":
                    receipt.failure_reason = "minimal_completion_refused"
                    return None, receipt
                receipt.fallback_count += 1
                try:
                    successor, actions = graft_observed_fragment(
                        current,
                        site=int(site),
                        rooted_smiles="*C",
                        system=system,
                        lock=lock,
                    )
                except (ValueError, RuntimeError) as fallback_error:
                    receipt.failure_reason = f"minimal_completion_refused:{fallback_error}"
                    return None, receipt
                receipt.attempted_fragments.append(
                    {
                        "site": int(site),
                        "context": context_key,
                        "fragment": "*C",
                        "training_occurrences": 0,
                        "minimal_fallback": True,
                    }
                )
            else:
                receipt.attempted_fragments.append(attempt)
            receipt.accepted_actions.extend(
                {"rule": "atom_insert", "payload": asdict(action)} for action in actions
            )
            current = successor
    smiles = molecular_graph_to_smiles(current)
    if not smiles:
        receipt.failure_reason = "no_endpoint_smiles"
        return None, receipt
    receipt.committed_smiles = smiles
    result = check_fragment_constraint(context.prompt, smiles)
    if not result.satisfied or result.canonical_smiles is None:
        receipt.failure_reason = f"constraint_failure:{result.reason}"
        return None, receipt
    return result.canonical_smiles, receipt
