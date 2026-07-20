"""Fail-closed loader for the lipid reaction registry.

The registry intentionally supports incomplete ``literature_spec`` records so
that source extraction can be versioned before executable chemistry exists.
Only ``qualified_for_enumeration`` records may be passed to an enumerator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


QUALIFIED_STATUS = "qualified_for_enumeration"
ALLOWED_STATUSES = {"literature_spec", "validated_transform", QUALIFIED_STATUS}


class RegistryError(ValueError):
    """Raised when a registry is malformed or an unqualified route is requested."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class ReactionSpec:
    """One versioned reaction record with an explicit qualification gate."""

    reaction_id: str
    reaction_version: int
    status: str
    architecture: str
    sources: tuple[Mapping[str, Any], ...]
    reactant_roles: tuple[Mapping[str, Any], ...]
    atom_mapped_reaction_smarts: str | None
    selectivity_policy: str
    stereochemistry_policy: str
    protonation_and_salt_policy: str
    known_positive_examples: tuple[Mapping[str, Any], ...]
    known_negative_examples: tuple[Mapping[str, Any], ...]
    implementation: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ReactionSpec":
        required = {
            "reaction_id",
            "reaction_version",
            "status",
            "architecture",
            "sources",
            "reactant_roles",
            "selectivity_policy",
            "known_positive_examples",
            "known_negative_examples",
            "implementation",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise RegistryError(f"reaction record lacks required fields: {missing}")
        reaction_id = raw["reaction_id"]
        status = raw["status"]
        if not _nonempty_string(reaction_id):
            raise RegistryError("reaction_id must be a nonempty string")
        if status not in ALLOWED_STATUSES:
            raise RegistryError(f"{reaction_id}: unsupported status {status!r}")
        if not isinstance(raw["reaction_version"], int) or raw["reaction_version"] < 1:
            raise RegistryError(f"{reaction_id}: reaction_version must be a positive integer")
        sources = raw["sources"]
        roles = raw["reactant_roles"]
        if not isinstance(sources, list) or not sources:
            raise RegistryError(f"{reaction_id}: at least one primary source is required")
        if not isinstance(roles, list) or len(roles) < 2:
            raise RegistryError(f"{reaction_id}: at least two reactant roles are required")
        return cls(
            reaction_id=reaction_id,
            reaction_version=raw["reaction_version"],
            status=status,
            architecture=str(raw["architecture"]),
            sources=tuple(sources),
            reactant_roles=tuple(roles),
            atom_mapped_reaction_smarts=raw.get("atom_mapped_reaction_smarts"),
            selectivity_policy=str(raw["selectivity_policy"]),
            stereochemistry_policy=str(raw.get("stereochemistry_policy", "")),
            protonation_and_salt_policy=str(raw.get("protonation_and_salt_policy", "")),
            known_positive_examples=tuple(raw["known_positive_examples"]),
            known_negative_examples=tuple(raw["known_negative_examples"]),
            implementation=dict(raw["implementation"]),
        )

    @property
    def versioned_id(self) -> str:
        return f"{self.reaction_id}@{self.reaction_version}"

    def qualification_blockers(self) -> tuple[str, ...]:
        """Return every fail-closed reason preventing production enumeration."""

        blockers: list[str] = []
        if self.status != QUALIFIED_STATUS:
            blockers.append(f"status is {self.status!r}, not {QUALIFIED_STATUS!r}")
        if not _nonempty_string(self.atom_mapped_reaction_smarts):
            blockers.append("atom-mapped reaction SMARTS is absent")
        for role in self.reactant_roles:
            name = str(role.get("name", "<unnamed>"))
            if not _nonempty_string(role.get("required_handle_smarts")):
                blockers.append(f"role {name!r} lacks a required-handle SMARTS")
            mapped_atoms = role.get("mapped_reactive_atoms")
            if not isinstance(mapped_atoms, list) or not mapped_atoms:
                blockers.append(f"role {name!r} lacks mapped reactive atoms")
        if not self.known_positive_examples:
            blockers.append("no reproduced positive example")
        if not self.known_negative_examples:
            blockers.append("no rejected negative example")
        for key in ("enumerator", "test_manifest", "artifact_hash"):
            if not _nonempty_string(self.implementation.get(key)):
                blockers.append(f"implementation.{key} is absent")
        return tuple(blockers)

    def require_qualified(self) -> None:
        blockers = self.qualification_blockers()
        if blockers:
            joined = "; ".join(blockers)
            raise RegistryError(f"{self.versioned_id} is not enumerable: {joined}")


@dataclass(frozen=True)
class ReactionRegistry:
    """Immutable loaded registry plus artifact provenance."""

    registry_version: str
    reactions: tuple[ReactionSpec, ...]
    path: Path
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "ReactionRegistry":
        registry_path = Path(path).resolve()
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RegistryError("registry root must be an object")
        version = raw.get("registry_version")
        reactions = raw.get("reactions")
        if not _nonempty_string(version):
            raise RegistryError("registry_version must be a nonempty string")
        if not isinstance(reactions, list):
            raise RegistryError("reactions must be a list")
        parsed = tuple(ReactionSpec.from_mapping(item) for item in reactions)
        versioned_ids = [reaction.versioned_id for reaction in parsed]
        if len(versioned_ids) != len(set(versioned_ids)):
            raise RegistryError("registry contains duplicate reaction_id/version pairs")
        return cls(str(version), parsed, registry_path, _sha256(registry_path))

    def by_id(self, reaction_id: str) -> ReactionSpec:
        matches = [reaction for reaction in self.reactions if reaction.reaction_id == reaction_id]
        if not matches:
            raise RegistryError(f"reaction {reaction_id!r} is absent")
        if len(matches) > 1:
            raise RegistryError(
                f"reaction {reaction_id!r} has multiple versions; request an explicit version"
            )
        return matches[0]

    def qualified(self) -> tuple[ReactionSpec, ...]:
        return tuple(reaction for reaction in self.reactions if not reaction.qualification_blockers())

    def preflight(self) -> dict[str, Any]:
        return {
            "registry_version": self.registry_version,
            "registry_path": str(self.path),
            "registry_sha256": self.sha256,
            "reaction_count": len(self.reactions),
            "qualified_count": len(self.qualified()),
            "reactions": [
                {
                    "versioned_id": reaction.versioned_id,
                    "status": reaction.status,
                    "qualification_blockers": list(reaction.qualification_blockers()),
                }
                for reaction in self.reactions
            ],
        }
