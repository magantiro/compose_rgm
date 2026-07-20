#!/usr/bin/env python3
"""Preflight or stream a route-certified COMPOSE-Lipid enumeration pilot.

The script is fail-closed: every selected reaction must be
``qualified_for_enumeration`` and every reactant role must point to a frozen CSV
component manifest. Candidate products are generated lazily, canonicalized,
deduplicated in SQLite, and selected with deterministic stratified reservoirs.
The committed pilot configuration is expected to fail enumeration until a
primary-source reaction is actually qualified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from compose_v4.lipids.reaction_registry import ReactionRegistry, ReactionSpec, RegistryError
from compose_v4.lipids.streaming_enumeration import (
    CandidateProduct,
    StratifiedReservoir,
    canonicalize_product,
    product_bins,
    stable_priority,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/lipid_reactions/pilot_enumeration_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/datasets/compose_lipid_r1_pilot_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class BuildingBlock:
    block_id: str
    canonical_smiles: str
    molecule: Chem.Mol
    architecture_class: str


def load_blocks(
    role_name: str,
    role_spec: Mapping[str, Any],
    source_spec: Mapping[str, Any],
) -> tuple[list[BuildingBlock], dict[str, Any]]:
    path_value = source_spec.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise RegistryError(f"role {role_name!r} lacks a frozen component-manifest path")
    path = resolve_repo_path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    id_field = str(source_spec.get("id_field", "building_block_id"))
    smiles_field = str(source_spec.get("smiles_field", "canonical_isomeric_smiles"))
    class_field = str(source_spec.get("class_field", "architecture_class"))
    handle_smarts = str(role_spec["required_handle_smarts"])
    handle = Chem.MolFromSmarts(handle_smarts)
    if handle is None:
        raise RegistryError(f"role {role_name!r} has invalid required-handle SMARTS")
    blocks: dict[str, BuildingBlock] = {}
    rejected = 0
    with path.open(newline="", encoding="utf-8-sig") as handle_csv:
        reader = csv.DictReader(handle_csv)
        required = {id_field, smiles_field, class_field}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RegistryError(f"{path} lacks component fields {sorted(required)}")
        for row in reader:
            block_id = str(row[id_field]).strip()
            raw_smiles = str(row[smiles_field]).strip()
            architecture_class = str(row[class_field]).strip()
            molecule = Chem.MolFromSmiles(raw_smiles)
            if not block_id or molecule is None or not molecule.HasSubstructMatch(handle):
                rejected += 1
                continue
            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            blocks[block_id] = BuildingBlock(
                block_id=block_id,
                canonical_smiles=canonical,
                molecule=molecule,
                architecture_class=architecture_class,
            )
    ordered = [blocks[key] for key in sorted(blocks)]
    if not ordered:
        raise RegistryError(f"role {role_name!r} has no qualified building blocks")
    return ordered, {
        "path": str(path),
        "sha256": sha256_file(path),
        "accepted_count": len(ordered),
        "rejected_or_duplicate_count": rejected,
    }


def expanded_reactant_roles(
    reaction: ReactionSpec,
    reaction_config: Mapping[str, Any],
) -> tuple[list[str], list[list[BuildingBlock]], dict[str, Any], dict[str, str]]:
    role_sources = reaction_config.get("role_sources")
    if not isinstance(role_sources, dict):
        raise RegistryError(f"{reaction.reaction_id}: role_sources must be an object")
    architecture_tags = reaction_config.get("architecture_tags")
    if not isinstance(architecture_tags, dict):
        raise RegistryError(f"{reaction.reaction_id}: architecture_tags must be an object")
    names: list[str] = []
    block_lists: list[list[BuildingBlock]] = []
    manifests: dict[str, Any] = {}
    for role in reaction.reactant_roles:
        name = str(role["name"])
        source = role_sources.get(name)
        if not isinstance(source, dict):
            raise RegistryError(f"{reaction.reaction_id}: no component source for role {name!r}")
        blocks, manifest = load_blocks(name, role, source)
        manifests[name] = manifest
        for occurrence in range(int(role["count"])):
            names.append(name if int(role["count"]) == 1 else f"{name}[{occurrence}]")
            block_lists.append(blocks)
    return names, block_lists, manifests, {str(k): str(v) for k, v in architecture_tags.items()}


class StructureIndex:
    """Disk-backed exact product deduplication and route multiplicity counter."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                structure_sha256 TEXT PRIMARY KEY,
                canonical_smiles TEXT NOT NULL,
                reaction_id TEXT NOT NULL,
                first_route_json TEXT NOT NULL,
                route_count INTEGER NOT NULL
            )
            """
        )

    def add(self, candidate: CandidateProduct) -> bool:
        route = json.dumps(
            {
                "reactant_ids": candidate.reactant_ids,
                "reactant_roles": candidate.reactant_roles,
            },
            sort_keys=True,
        )
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO products VALUES (?, ?, ?, ?, 1)",
            (candidate.identity, candidate.canonical_smiles, candidate.reaction_id, route),
        )
        inserted = cursor.rowcount == 1
        if not inserted:
            self.connection.execute(
                "UPDATE products SET route_count = route_count + 1 WHERE structure_sha256 = ?",
                (candidate.identity,),
            )
        return inserted

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM products").fetchone()[0])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def stream_candidates(
    reaction: ReactionSpec,
    role_names: list[str],
    block_lists: list[list[BuildingBlock]],
    architecture_tags: Mapping[str, str],
) -> Iterable[CandidateProduct]:
    reaction.require_qualified()
    rd_reaction = rdChemReactions.ReactionFromSmarts(
        str(reaction.atom_mapped_reaction_smarts), useSmiles=False
    )
    if rd_reaction is None:
        raise RegistryError(f"{reaction.reaction_id}: reaction SMARTS did not compile")
    if rd_reaction.GetNumReactantTemplates() != len(block_lists):
        raise RegistryError(
            f"{reaction.reaction_id}: SMARTS expects {rd_reaction.GetNumReactantTemplates()} "
            f"reactants but expanded roles provide {len(block_lists)}"
        )
    for blocks in itertools.product(*block_lists):
        outcomes = rd_reaction.RunReactants(tuple(block.molecule for block in blocks))
        for outcome in outcomes:
            if len(outcome) != 1:
                continue
            canonicalized = canonicalize_product(outcome[0])
            if canonicalized is None:
                continue
            canonical, molecule = canonicalized
            role_classes: dict[str, set[str]] = {}
            for role_name, block in zip(role_names, blocks, strict=True):
                base_name = role_name.split("[", 1)[0]
                role_classes.setdefault(base_name, set()).add(block.architecture_class)
            tags = dict(architecture_tags)
            for role_name, classes in role_classes.items():
                source = role_name.lower()
                if "head" in source:
                    tag = "head_class"
                elif "tail" in source or "chain" in source or "isocyanide" in source:
                    tag = "tail_class"
                else:
                    tag = f"{source}_class"
                tags[tag] = "+".join(sorted(classes))
            tags.setdefault("head_class", "not_applicable")
            tags.setdefault("tail_class", "not_applicable")
            yield CandidateProduct(
                canonical_smiles=canonical,
                reaction_id=reaction.reaction_id,
                reactant_ids=tuple(block.block_id for block in blocks),
                reactant_roles=tuple(role_names),
                architecture_tags=tags,
                product_bins=product_bins(molecule),
            )


def round_robin_limit(
    reservoir: StratifiedReservoir, target: int
) -> list[CandidateProduct]:
    by_stratum: dict[tuple[str, ...], list[CandidateProduct]] = {}
    for candidate in reservoir.selected():
        by_stratum.setdefault(candidate.stratum(reservoir.axes), []).append(candidate)
    selected: list[CandidateProduct] = []
    round_index = 0
    while len(selected) < target:
        progressed = False
        for stratum in sorted(by_stratum):
            values = by_stratum[stratum]
            if round_index < len(values):
                selected.append(values[round_index])
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
        round_index += 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", default="chemistry_smoke")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-combinations", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    registry = ReactionRegistry.load(resolve_repo_path(config["registry_path"]))
    stages = {stage["name"]: stage for stage in config["stages"]}
    if args.stage not in stages:
        raise ValueError(f"unknown stage {args.stage!r}; choose from {sorted(stages)}")
    target = int(stages[args.stage]["target_unique_products"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "release_id": config["release_id"],
        "stage": args.stage,
        "target_unique_products": target,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "registry": registry.preflight(),
        "enumeration_started": False,
        "enumerated_new_molecule_count": 0,
    }
    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    reaction_configs = {item["reaction_id"]: item for item in config["reactions"]}
    selected_specs: list[
        tuple[ReactionSpec, list[str], list[list[BuildingBlock]], dict[str, Any], dict[str, str]]
    ] = []
    total_combinations = 0
    component_manifests: dict[str, Any] = {}
    for reaction_id, reaction_config in sorted(reaction_configs.items()):
        reaction = registry.by_id(reaction_id)
        reaction.require_qualified()
        role_names, block_lists, manifests, architecture_tags = expanded_reactant_roles(
            reaction, reaction_config
        )
        combination_count = 1
        for blocks in block_lists:
            combination_count *= len(blocks)
        total_combinations += combination_count
        component_manifests[reaction_id] = manifests
        selected_specs.append(
            (reaction, role_names, block_lists, manifests, architecture_tags)
        )
    if total_combinations > args.max_combinations:
        raise RuntimeError(
            f"preflight predicts {total_combinations:,} reactant combinations, above "
            f"--max-combinations={args.max_combinations:,}; shard explicitly before scaling"
        )

    stratification = config["stratification"]
    reservoir = StratifiedReservoir(
        axes=stratification["axes"],
        quota_per_stratum=int(stratification["quota_per_observed_stratum"]),
        seed=str(config["seed"]),
    )
    index = StructureIndex(output_dir / "candidate_index.sqlite")
    generated_outcomes = 0
    try:
        for reaction, role_names, block_lists, _, architecture_tags in selected_specs:
            for candidate in stream_candidates(
                reaction, role_names, block_lists, architecture_tags
            ):
                generated_outcomes += 1
                if index.add(candidate):
                    reservoir.consider(candidate)
        unique_count = index.count()
    finally:
        index.close()

    final = round_robin_limit(reservoir, target)
    output_csv = output_dir / "selected_products.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "structure_sha256",
            "canonical_isomeric_smiles",
            "reaction_id",
            "reactant_ids_json",
            "reactant_roles_json",
            "architecture_tags_json",
            "product_bins_json",
            "selection_priority",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in final:
            writer.writerow(
                {
                    "structure_sha256": candidate.identity,
                    "canonical_isomeric_smiles": candidate.canonical_smiles,
                    "reaction_id": candidate.reaction_id,
                    "reactant_ids_json": json.dumps(candidate.reactant_ids),
                    "reactant_roles_json": json.dumps(candidate.reactant_roles),
                    "architecture_tags_json": json.dumps(
                        candidate.architecture_tags, sort_keys=True
                    ),
                    "product_bins_json": json.dumps(candidate.product_bins, sort_keys=True),
                    "selection_priority": stable_priority(
                        str(config["seed"]), candidate.identity
                    ),
                }
            )
    manifest = {
        **preflight,
        "enumeration_started": True,
        "predicted_reactant_combinations": total_combinations,
        "generated_product_outcomes": generated_outcomes,
        "enumerated_new_molecule_count": unique_count,
        "selected_unique_product_count": len(final),
        "component_manifests": component_manifests,
        "stratum_counts": {
            "|".join(key): value for key, value in reservoir.stratum_counts.items()
        },
        "selected_artifact": {
            "path": str(output_csv),
            "sha256": sha256_file(output_csv),
        },
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
