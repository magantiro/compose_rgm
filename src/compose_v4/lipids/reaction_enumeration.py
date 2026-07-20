"""Apply a qualified atom-mapped reaction transform to role-typed building blocks.

This module is the executable ``implementation.enumerator`` referenced by
``qualified_for_enumeration`` reaction records.  It refuses any reaction that is
not qualified (fail-closed via :meth:`ReactionSpec.require_qualified`), validates
that every supplied building block carries the reactive handle its role requires,
and emits deduplicated, sanitized, single-component products with a route
certificate.  Canonicalization preserves charge and tautomer, matching the frozen
``compose_lipid`` canonicalization contract (RDKit isomeric SMILES, explicit Hs
removed, no neutralization, no salt stripping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem

from .reaction_registry import ReactionSpec
from .streaming_enumeration import CandidateProduct, canonicalize_product, product_bins


@dataclass(frozen=True)
class BuildingBlock:
    """One role-typed reactant with provenance and architecture annotations."""

    block_id: str
    role: str
    smiles: str
    architecture_tags: Mapping[str, str]

    def molecule(self) -> Chem.Mol:
        molecule = Chem.MolFromSmiles(self.smiles)
        if molecule is None:
            raise ValueError(f"building block {self.block_id!r} has unparsable SMILES: {self.smiles!r}")
        return molecule


class EnumerationError(ValueError):
    """Raised when an enumeration request is malformed or uses an unqualified route."""


def _role_names(spec: ReactionSpec) -> tuple[str, ...]:
    names = tuple(str(role.get("name", "")) for role in spec.reactant_roles)
    if any(not name for name in names):
        raise EnumerationError(f"{spec.versioned_id}: every reactant role must be named")
    if len(set(names)) != len(names):
        raise EnumerationError(f"{spec.versioned_id}: reactant role names must be unique")
    return names


class ReactionEnumerator:
    """Deterministic executor for a single qualified reaction transform.

    Reactant-template order in the atom-mapped SMARTS must match the declared
    ``reactant_roles`` order; building blocks are supplied per role and combined
    in that order.  Every distinct canonical product is emitted once per
    substrate tuple, so multi-handle substrates (e.g. multiple reactive amines)
    yield each regiochemical product with the same route certificate.
    """

    def __init__(self, spec: ReactionSpec) -> None:
        spec.require_qualified()
        self.spec = spec
        self.role_order = _role_names(spec)
        self.reaction = AllChem.ReactionFromSmarts(str(spec.atom_mapped_reaction_smarts))
        if self.reaction is None:
            raise EnumerationError(f"{spec.versioned_id}: atom-mapped reaction SMARTS did not parse")
        if self.reaction.GetNumReactantTemplates() != len(self.role_order):
            raise EnumerationError(
                f"{spec.versioned_id}: reaction has {self.reaction.GetNumReactantTemplates()} "
                f"reactant templates but {len(self.role_order)} declared roles"
            )
        self._handles: dict[str, Chem.Mol] = {}
        for role in spec.reactant_roles:
            name = str(role["name"])
            smarts = str(role.get("required_handle_smarts", ""))
            pattern = Chem.MolFromSmarts(smarts) if smarts else None
            if pattern is None:
                raise EnumerationError(
                    f"{spec.versioned_id}: role {name!r} has no parsable required_handle_smarts"
                )
            self._handles[name] = pattern

    def block_has_required_handle(self, block: BuildingBlock) -> bool:
        pattern = self._handles.get(block.role)
        if pattern is None:
            raise EnumerationError(f"{self.spec.versioned_id}: unknown role {block.role!r}")
        return block.molecule().HasSubstructMatch(pattern)

    def _combine_tags(self, blocks: Sequence[BuildingBlock]) -> dict[str, str]:
        tags: dict[str, str] = {}
        for block in blocks:
            for key, value in block.architecture_tags.items():
                tags[key] = str(value)
        return tags

    def react(self, blocks: Sequence[BuildingBlock]) -> Iterator[CandidateProduct]:
        """Apply the transform to one ordered building-block tuple.

        ``blocks`` must be in ``role_order``.  Blocks lacking the required handle
        for their role yield nothing (fail-closed chemoselectivity).
        """

        if tuple(block.role for block in blocks) != self.role_order:
            raise EnumerationError(
                f"{self.spec.versioned_id}: blocks must be supplied in role order {self.role_order}"
            )
        for block in blocks:
            if not self.block_has_required_handle(block):
                return
        reactant_mols = tuple(block.molecule() for block in blocks)
        tags = self._combine_tags(blocks)
        reactant_ids = tuple(block.block_id for block in blocks)
        seen: set[str] = set()
        for product_set in self.reaction.RunReactants(reactant_mols):
            for raw_product in product_set:
                result = canonicalize_product(raw_product)
                if result is None:
                    continue
                canonical, parsed = result
                if canonical in seen:
                    continue
                seen.add(canonical)
                yield CandidateProduct(
                    canonical_smiles=canonical,
                    reaction_id=self.spec.reaction_id,
                    reactant_ids=reactant_ids,
                    reactant_roles=self.role_order,
                    architecture_tags=tags,
                    product_bins=product_bins(parsed),
                )

    def enumerate(
        self, blocks_by_role: Mapping[str, Iterable[BuildingBlock]]
    ) -> Iterator[CandidateProduct]:
        """Stream products over the Cartesian product of role-typed blocks.

        Deduplication here is per-substrate-tuple only; global deduplication of
        identical products arising from different substrate tuples must be done
        downstream (e.g. a disk-backed unique index before the reservoir).
        """

        missing = [role for role in self.role_order if role not in blocks_by_role]
        if missing:
            raise EnumerationError(f"{self.spec.versioned_id}: no building blocks for roles {missing}")
        materialized = {role: list(blocks_by_role[role]) for role in self.role_order}
        for role, blocks in materialized.items():
            for block in blocks:
                if block.role != role:
                    raise EnumerationError(
                        f"{self.spec.versioned_id}: block {block.block_id!r} filed under role "
                        f"{role!r} but tagged {block.role!r}"
                    )
        yield from self._enumerate_recursive(materialized, [], 0)

    def _enumerate_recursive(
        self,
        materialized: Mapping[str, list[BuildingBlock]],
        chosen: list[BuildingBlock],
        depth: int,
    ) -> Iterator[CandidateProduct]:
        if depth == len(self.role_order):
            yield from self.react(chosen)
            return
        role = self.role_order[depth]
        for block in materialized[role]:
            chosen.append(block)
            yield from self._enumerate_recursive(materialized, chosen, depth + 1)
            chosen.pop()
