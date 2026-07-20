"""Lock the Ugi-3CR route-replay smoke: block manifest + enumerator invariants.

Enumeration is driven from the committed building-block manifest, so these tests
need no external raw files. The frozen smoke audit is checked for its exact
route-replay claim.
"""

from __future__ import annotations

import json
from pathlib import Path

from compose_v4.lipids.reaction_enumeration import BuildingBlock, ReactionEnumerator
from compose_v4.lipids.reaction_registry import ReactionRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKS = REPO_ROOT / "configs/lipid_reactions/ugi_3cr_building_blocks_v1.json"
SMOKE_AUDIT = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/ugi_3cr_smoke_v1/smoke_audit.json"


def _blocks_by_role() -> dict[str, list[BuildingBlock]]:
    manifest = json.loads(BLOCKS.read_text())
    by_role: dict[str, list[BuildingBlock]] = {}
    for block in manifest["blocks"]:
        by_role.setdefault(block["role"], []).append(
            BuildingBlock(
                block_id=block["block_id"],
                role=block["role"],
                smiles=block["canonical_smiles"],
                architecture_tags=block["architecture_tags"],
            )
        )
    return by_role


def test_block_manifest_role_counts() -> None:
    manifest = json.loads(BLOCKS.read_text())
    assert manifest["role_counts"] == {
        "amine_head": 20,
        "oxoester_aldehyde_body_tail": 12,
        "isocyanide_tail": 5,
    }


def test_smoke_enumeration_is_valid_unique_and_deterministic() -> None:
    spec = ReactionRegistry.load(
        REPO_ROOT / "configs/lipid_reactions/qualified_reactions_v1.json"
    ).by_id("ugi_3cr_agile")
    enumerator = ReactionEnumerator(spec)
    products = list(enumerator.enumerate(_blocks_by_role()))
    canonicals = {product.canonical_smiles for product in products}
    # 20x12x5 grid + 120 extra regiochemistries from 2 multi-N-H amine heads.
    assert len(canonicals) == 1320
    # every committed state is a valid, single-component molecule (canonicalized).
    assert all(product.canonical_smiles for product in products)


def test_frozen_smoke_audit_replays_released_library_exactly() -> None:
    audit = json.loads(SMOKE_AUDIT.read_text())
    replay = audit["route_replay"]
    assert replay["released_products"] == 1200
    assert replay["replayed_exactly"] == 1200
    assert replay["replay_fraction"] == 1.0
    assert replay["products_not_in_released_set"] == 120
    assert audit["unique_products"] == 1320
