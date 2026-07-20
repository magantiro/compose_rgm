"""Load the role-annotated building-block pool and serve blocks by reaction role."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .reaction_enumeration import BuildingBlock


@dataclass(frozen=True)
class BuildingBlockPool:
    """Immutable pool of reactive substrates indexed by the roles they can fill."""

    blocks: tuple[dict, ...]

    def blocks_for_roles(self, roles: tuple[str, ...]) -> dict[str, list[BuildingBlock]]:
        out: dict[str, list[BuildingBlock]] = {role: [] for role in roles}
        for block in self.blocks:
            block_roles = set(block.get("reaction_roles", ()))
            for role in roles:
                if role in block_roles:
                    out[role].append(self._to_block(block, role))
        return out

    @staticmethod
    def _to_block(block: Mapping, role: str) -> BuildingBlock:
        provenance = block.get("provenance", {})
        descriptors = block.get("descriptors", {})
        tags = {"source": str(provenance.get("source", "")), "handle": str(block.get("handle", ""))}
        if "chain_length" in descriptors:
            tags["chain_length"] = str(descriptors["chain_length"])
            tags["unsaturation"] = str(descriptors.get("unsaturation", 0))
        return BuildingBlock(
            block_id=block["block_id"],
            role=role,
            smiles=block["canonical_smiles"],
            architecture_tags=tags,
        )


def load_pool(path: str | Path) -> BuildingBlockPool:
    data = json.loads(Path(path).read_text())
    return BuildingBlockPool(tuple(data["blocks"]))
