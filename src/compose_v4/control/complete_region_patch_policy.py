"""Canonical complete-region graph-delta tokens and split-first policy support.

One decoded object is a whole :class:`StructuralSubgoal`.  The token stream is
an internal graph-construction language for a semantic target patch; it never
contains or predicts executor primitive actions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.control.structural_subgoal import StructuralSubgoal

TOKEN_SCHEMA = "complete_region_patch_token_v1"
STREAM_SCHEMA = "complete_region_patch_stream_v1"

ATOM_TYPES = tuple(sorted(set(ORGANIC_VOCABULARY.element_index)))
FORMAL_CHARGES = tuple(range(-2, 3))
HYDROGEN_COUNTS = tuple(range(5))
DEGREES = tuple(range(7))
BOND_ORDERS = tuple(range(1, 5))
MAXIMUM_ROLES = 40


@dataclass(frozen=True)
class PatchToken:
    """One typed decision in a canonical complete-patch graph stream."""

    kind: str
    value: int
    factor: str

    def payload(self) -> dict:
        return {
            "schema_version": TOKEN_SCHEMA,
            "kind": self.kind,
            "value": self.value,
            "factor": self.factor,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> PatchToken:
        if payload.get("schema_version") != TOKEN_SCHEMA or set(payload) != {
            "schema_version",
            "kind",
            "value",
            "factor",
        }:
            raise ValueError("complete-region patch token schema mismatch")
        return cls(str(payload["kind"]), int(payload["value"]), str(payload["factor"]))


@dataclass(frozen=True)
class SourceRegionContext:
    """Address-free current-state roles supplied by the WHERE decision."""

    input_atoms: tuple[tuple[int, int, int, int], ...]
    input_bonds: tuple[tuple[int, ...], ...]
    environments: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.input_atoms)
        if (
            not 1 <= size <= MAXIMUM_ROLES
            or len(self.environments) != size
            or len(self.input_bonds) != size
            or any(len(row) != size for row in self.input_bonds)
        ):
            raise ValueError("complete-region source context is malformed")

    @classmethod
    def from_subgoal(cls, subgoal: StructuralSubgoal) -> SourceRegionContext:
        return cls(subgoal.input_atoms, subgoal.input_bonds, subgoal.environments)


def _atom_tokens(prefix: str, atom: tuple[int, int, int, int]) -> tuple[PatchToken, ...]:
    element, charge, hydrogens, degree = map(int, atom)
    return (
        PatchToken(f"{prefix}_element", element, "atom_attributes"),
        PatchToken(f"{prefix}_formal_charge", charge, "atom_attributes"),
        PatchToken(f"{prefix}_implicit_hydrogens", hydrogens, "atom_attributes"),
        PatchToken(f"{prefix}_degree", degree, "atom_attributes"),
    )


def _dependency_parent(subgoal: StructuralSubgoal, output: int) -> int:
    role = len(subgoal.input_atoms) + output
    neighbors = [
        other
        for other, order in enumerate(subgoal.target_bonds[role])
        if int(order) and other != role
    ]
    return min(neighbors) if neighbors else -1


def encode_patch_stream(subgoal: StructuralSubgoal) -> tuple[PatchToken, ...]:
    """Losslessly encode one whole target patch without executor actions."""

    tokens: list[PatchToken] = [PatchToken("output_count", len(subgoal.output_atoms), "control")]
    live: list[bool] = []
    for target in subgoal.target_atoms:
        survives = target is not None
        tokens.append(PatchToken("input_survival", int(survives), "atom_attributes"))
        live.append(survives)
        if target is not None:
            tokens.extend(_atom_tokens("input_target", target))
    for atom in subgoal.output_atoms:
        live.append(True)
        tokens.extend(_atom_tokens("output", atom))
    for left in range(len(live)):
        if not live[left]:
            continue
        for right in range(left + 1, len(live)):
            if not live[right]:
                continue
            order = int(subgoal.target_bonds[left][right])
            attachment = left < len(subgoal.input_atoms) <= right
            factor = "attachments" if attachment else "target_topology"
            tokens.append(PatchToken("edge_presence", int(bool(order)), factor))
            if order:
                tokens.append(
                    PatchToken(
                        "bond_order",
                        order,
                        "attachments" if attachment else "bond_attributes",
                    )
                )
    for output in range(len(subgoal.output_atoms)):
        tokens.append(
            PatchToken(
                "output_dependency_parent",
                _dependency_parent(subgoal, output),
                "dependencies",
            )
        )
    tokens.append(PatchToken("patch_stop", 1, "control"))
    return tuple(tokens)


class _Reader:
    def __init__(self, tokens: Iterable[PatchToken]) -> None:
        self.tokens = tuple(tokens)
        self.index = 0

    def take(self, kind: str, factor: str, allowed: tuple[int, ...]) -> int:
        if self.index >= len(self.tokens):
            raise ValueError(f"missing complete-patch token: {kind}")
        token = self.tokens[self.index]
        self.index += 1
        if token.kind != kind or token.factor != factor or token.value not in allowed:
            raise ValueError(f"unsupported complete-patch token at {self.index - 1}: {token}")
        return token.value

    def finish(self) -> None:
        if self.index != len(self.tokens):
            raise ValueError("complete-patch stream has trailing tokens")


def _read_atom(reader: _Reader, prefix: str) -> tuple[int, int, int, int]:
    return (
        reader.take(f"{prefix}_element", "atom_attributes", ATOM_TYPES),
        reader.take(f"{prefix}_formal_charge", "atom_attributes", FORMAL_CHARGES),
        reader.take(f"{prefix}_implicit_hydrogens", "atom_attributes", HYDROGEN_COUNTS),
        reader.take(f"{prefix}_degree", "atom_attributes", DEGREES),
    )


def decode_patch_stream(
    context: SourceRegionContext, tokens: Iterable[PatchToken]
) -> StructuralSubgoal:
    """Decode one canonical graph stream into a complete structural patch."""

    reader = _Reader(tokens)
    output_count = reader.take("output_count", "control", tuple(range(MAXIMUM_ROLES + 1)))
    if len(context.input_atoms) + output_count > MAXIMUM_ROLES:
        raise ValueError("complete target patch exceeds active-role support")
    target_atoms = []
    live = []
    for _ in context.input_atoms:
        survives = bool(reader.take("input_survival", "atom_attributes", (0, 1)))
        live.append(survives)
        target_atoms.append(_read_atom(reader, "input_target") if survives else None)
    output_atoms = tuple(_read_atom(reader, "output") for _ in range(output_count))
    live.extend([True] * output_count)
    target_bonds = [[0 for _ in live] for _ in live]
    for left in range(len(live)):
        if not live[left]:
            continue
        for right in range(left + 1, len(live)):
            if not live[right]:
                continue
            attachment = left < len(context.input_atoms) <= right
            factor = "attachments" if attachment else "target_topology"
            present = reader.take("edge_presence", factor, (0, 1))
            order = (
                reader.take(
                    "bond_order",
                    "attachments" if attachment else "bond_attributes",
                    BOND_ORDERS,
                )
                if present
                else 0
            )
            target_bonds[left][right] = target_bonds[right][left] = order
    dependencies = [
        reader.take(
            "output_dependency_parent",
            "dependencies",
            tuple(range(-1, len(live))),
        )
        for _ in range(output_count)
    ]
    reader.take("patch_stop", "control", (1,))
    reader.finish()
    subgoal = StructuralSubgoal(
        input_atoms=context.input_atoms,
        input_bonds=context.input_bonds,
        environments=context.environments,
        target_atoms=tuple(target_atoms),
        output_atoms=output_atoms,
        target_bonds=tuple(tuple(row) for row in target_bonds),
    )
    expected = [_dependency_parent(subgoal, output) for output in range(output_count)]
    if dependencies != expected:
        raise ValueError("dependency-role tokens disagree with the decoded target graph")
    return subgoal


def patch_stream_support(subgoal: StructuralSubgoal) -> tuple[bool, str | None]:
    """Check declared grammar support and exact canonical round-trip."""

    try:
        context = SourceRegionContext.from_subgoal(subgoal)
        decoded = decode_patch_stream(context, encode_patch_stream(subgoal))
    except (TypeError, ValueError) as error:
        return False, f"{type(error).__name__}:{error}"
    if decoded != subgoal:
        return False, "roundtrip_mismatch"
    return True, None


__all__ = [
    "ATOM_TYPES",
    "BOND_ORDERS",
    "DEGREES",
    "FORMAL_CHARGES",
    "HYDROGEN_COUNTS",
    "MAXIMUM_ROLES",
    "PatchToken",
    "SourceRegionContext",
    "decode_patch_stream",
    "encode_patch_stream",
    "patch_stream_support",
]
