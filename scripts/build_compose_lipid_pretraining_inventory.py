#!/usr/bin/env python3
"""Build the audited, structure-only COMPOSE-Lipid source inventory.

This script deliberately does *not* enumerate molecules.  It canonicalizes only
full structures already present in frozen source artifacts, retains provenance,
and quantifies within- and cross-source duplication.  Formulation/assay rows are
collapsed to source-specific molecular identities for structural pretraining;
their original occurrence counts remain recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_ROOT = Path(
    "/Users/rmaganti/Desktop/thesis_projects_ML/diffusion_project/lipid_diffusion"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1"
MARKUSH_COMPONENT = re.compile(r"(R[1-4])\(\d+\):([^,\s)]+)")


@dataclass(frozen=True)
class SourceInput:
    source_id: str
    path: Path
    smiles_field: str
    record_field: str | None
    structure_status: str
    reaction_family: str
    license_id: str
    metadata_fields: tuple[str, ...] = ()
    region_fields: tuple[str, ...] = ()


@dataclass
class CanonicalAggregate:
    source_id: str
    canonical_smiles: str
    representative_raw_smiles: str
    structure_status: str
    reaction_family: str
    license_id: str
    occurrence_count: int = 0
    record_ids: set[str] = field(default_factory=set)
    metadata_values: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    region_values: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize(value: str) -> tuple[str, Chem.Mol] | None:
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NAN", "NONE"}:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    molecule = Chem.RemoveHs(molecule)
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=False,
    )
    return canonical, molecule


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"NA", "N/A", "NAN", "NONE"} else text


def _sorted_join(values: Iterable[str], *, limit: int | None = None) -> str:
    ordered = sorted({_clean(value) for value in values if _clean(value)})
    if limit is not None and len(ordered) > limit:
        return "|".join(ordered[:limit]) + f"|...(+{len(ordered) - limit})"
    return "|".join(ordered)


def _lumi_regions(row: Mapping[str, str]) -> dict[str, str]:
    code = _clean(row.get("Markush code", ""))
    return {name: value for name, value in MARKUSH_COMPONENT.findall(code)}


def load_source(source: SourceInput) -> tuple[list[CanonicalAggregate], dict[str, Any]]:
    aggregates: dict[str, CanonicalAggregate] = {}
    parse_failures: list[dict[str, str]] = []
    raw_strings: set[str] = set()
    total_rows = 0
    nonempty_rows = 0

    with source.path.open(newline="", encoding="utf-8-sig", errors="strict") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or source.smiles_field not in reader.fieldnames:
            raise ValueError(
                f"{source.path} lacks required structure field {source.smiles_field!r}"
            )
        for row_index, row in enumerate(reader, start=2):
            total_rows += 1
            raw_smiles = _clean(row.get(source.smiles_field, ""))
            if not raw_smiles:
                continue
            nonempty_rows += 1
            raw_strings.add(raw_smiles)
            parsed = canonicalize(raw_smiles)
            if parsed is None:
                parse_failures.append(
                    {"row": str(row_index), "raw_smiles": raw_smiles, "reason": "RDKit parse"}
                )
                continue
            canonical, _ = parsed
            aggregate = aggregates.get(canonical)
            if aggregate is None:
                aggregate = CanonicalAggregate(
                    source_id=source.source_id,
                    canonical_smiles=canonical,
                    representative_raw_smiles=raw_smiles,
                    structure_status=source.structure_status,
                    reaction_family=source.reaction_family,
                    license_id=source.license_id,
                )
                aggregates[canonical] = aggregate
            aggregate.occurrence_count += 1
            if source.record_field:
                record_id = _clean(row.get(source.record_field, ""))
                if record_id:
                    aggregate.record_ids.add(record_id)
            for field_name in source.metadata_fields:
                value = _clean(row.get(field_name, ""))
                if value:
                    aggregate.metadata_values[field_name].add(value)
            for field_name in source.region_fields:
                value = _clean(row.get(field_name, ""))
                if value:
                    aggregate.region_values[field_name].add(value)
            if source.source_id == "lumi_4cr1920":
                for region_name, value in _lumi_regions(row).items():
                    aggregate.region_values[region_name].add(value)

    summary = {
        "source_id": source.source_id,
        "path": str(source.path),
        "file_size_bytes": source.path.stat().st_size,
        "sha256": sha256_file(source.path),
        "total_rows": total_rows,
        "nonempty_structure_rows": nonempty_rows,
        "unique_raw_structure_strings": len(raw_strings),
        "parse_failure_count": len(parse_failures),
        "parse_failures": parse_failures,
        "unique_canonical_structures": len(aggregates),
        "canonical_collisions": len(raw_strings) - len(aggregates),
    }
    return list(aggregates.values()), summary


@lru_cache(maxsize=None)
def molecular_profile(canonical_smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:  # pragma: no cover - canonical SMILES was already validated
        raise ValueError(f"canonical structure no longer parses: {canonical_smiles}")
    elements = sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})
    chiral_centers = Chem.FindMolChiralCenters(
        molecule, includeUnassigned=True, includeCIP=True, useLegacyImplementation=False
    )
    return {
        "structure_sha256": hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest(),
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())),
        "fragment_count": int(len(Chem.GetMolFrags(molecule))),
        "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "stereogenic_atom_count": int(len(chiral_centers)),
        "molecular_weight": round(float(Descriptors.MolWt(molecule)), 6),
        "elements": "|".join(elements),
    }


def write_source_rows(path: Path, rows: list[CanonicalAggregate]) -> None:
    fieldnames = [
        "source_id",
        "canonical_isomeric_smiles",
        "structure_sha256",
        "representative_raw_smiles",
        "structure_status",
        "reaction_family",
        "license_id",
        "source_occurrence_count",
        "source_record_ids",
        "source_metadata_json",
        "region_annotation_json",
        "heavy_atoms",
        "formal_charge",
        "fragment_count",
        "ring_count",
        "aromatic_ring_count",
        "rotatable_bonds",
        "stereogenic_atom_count",
        "molecular_weight",
        "elements",
        "eligible_single_component",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for aggregate in sorted(rows, key=lambda item: (item.source_id, item.canonical_smiles)):
            profile = molecular_profile(aggregate.canonical_smiles)
            metadata = {
                key: sorted(values) for key, values in sorted(aggregate.metadata_values.items())
            }
            regions = {
                key: sorted(values) for key, values in sorted(aggregate.region_values.items())
            }
            writer.writerow(
                {
                    "source_id": aggregate.source_id,
                    "canonical_isomeric_smiles": aggregate.canonical_smiles,
                    **profile,
                    "representative_raw_smiles": aggregate.representative_raw_smiles,
                    "structure_status": aggregate.structure_status,
                    "reaction_family": aggregate.reaction_family,
                    "license_id": aggregate.license_id,
                    "source_occurrence_count": aggregate.occurrence_count,
                    "source_record_ids": _sorted_join(aggregate.record_ids, limit=100),
                    "source_metadata_json": json.dumps(metadata, sort_keys=True),
                    "region_annotation_json": json.dumps(regions, sort_keys=True),
                    "eligible_single_component": profile["fragment_count"] == 1,
                }
            )


def write_union_rows(path: Path, rows: list[CanonicalAggregate]) -> dict[str, int]:
    by_structure: dict[str, list[CanonicalAggregate]] = defaultdict(list)
    for row in rows:
        by_structure[row.canonical_smiles].append(row)
    fields = [
        "canonical_isomeric_smiles",
        "structure_sha256",
        "source_ids",
        "source_count",
        "total_source_occurrences",
        "structure_statuses",
        "reaction_families",
        "license_ids",
        "heavy_atoms",
        "formal_charge",
        "fragment_count",
        "ring_count",
        "aromatic_ring_count",
        "rotatable_bonds",
        "stereogenic_atom_count",
        "molecular_weight",
        "elements",
        "eligible_single_component",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for canonical, group in sorted(by_structure.items()):
            profile = molecular_profile(canonical)
            writer.writerow(
                {
                    "canonical_isomeric_smiles": canonical,
                    **profile,
                    "source_ids": _sorted_join(item.source_id for item in group),
                    "source_count": len(group),
                    "total_source_occurrences": sum(item.occurrence_count for item in group),
                    "structure_statuses": _sorted_join(item.structure_status for item in group),
                    "reaction_families": _sorted_join(item.reaction_family for item in group),
                    "license_ids": _sorted_join(item.license_id for item in group),
                    "eligible_single_component": profile["fragment_count"] == 1,
                }
            )
    return {
        "unique_canonical_union": len(by_structure),
        "unique_single_component_union": sum(
            molecular_profile(canonical)["fragment_count"] == 1 for canonical in by_structure
        ),
    }


def _percentile(values: list[int], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def chemistry_profile(rows: Iterable[CanonicalAggregate]) -> dict[str, Any]:
    profiles = [molecular_profile(row.canonical_smiles) for row in rows]
    heavy_atoms = [int(profile["heavy_atoms"]) for profile in profiles]
    charges: dict[str, int] = defaultdict(int)
    element_counts: dict[str, int] = defaultdict(int)
    for profile in profiles:
        charges[str(profile["formal_charge"])] += 1
        for element in str(profile["elements"]).split("|"):
            element_counts[element] += 1
    return {
        "unique_canonical_structures": len(profiles),
        "heavy_atoms": {
            "min": min(heavy_atoms),
            "median": statistics.median(heavy_atoms),
            "mean": statistics.fmean(heavy_atoms),
            "p95": _percentile(heavy_atoms, 0.95),
            "max": max(heavy_atoms),
        },
        "fraction_ring_bearing": sum(profile["ring_count"] > 0 for profile in profiles)
        / len(profiles),
        "fraction_aromatic_ring_bearing": sum(
            profile["aromatic_ring_count"] > 0 for profile in profiles
        )
        / len(profiles),
        "fraction_formally_charged": sum(profile["formal_charge"] != 0 for profile in profiles)
        / len(profiles),
        "fraction_with_stereogenic_atom": sum(
            profile["stereogenic_atom_count"] > 0 for profile in profiles
        )
        / len(profiles),
        "formal_charge_histogram": dict(sorted(charges.items(), key=lambda item: int(item[0]))),
        "element_presence_counts": dict(sorted(element_counts.items())),
        "max_atom_kernel_eligibility": {
            "40": sum(value <= 40 for value in heavy_atoms),
            "64": sum(value <= 64 for value in heavy_atoms),
            "96": sum(value <= 96 for value in heavy_atoms),
            "128": sum(value <= 128 for value in heavy_atoms),
        },
    }


def write_r0_release(path: Path, rows: list[CanonicalAggregate]) -> dict[str, Any]:
    """Write one row per measured/observed structure with frozen leakage groups."""

    by_structure: dict[str, list[CanonicalAggregate]] = defaultdict(list)
    for row in rows:
        if row.structure_status != "virtual_candidate":
            by_structure[row.canonical_smiles].append(row)
    all_memberships: dict[str, list[CanonicalAggregate]] = defaultdict(list)
    for row in rows:
        all_memberships[row.canonical_smiles].append(row)

    fields = [
        "r0_structure_id",
        "canonical_isomeric_smiles",
        "structure_sha256",
        "observed_source_ids",
        "all_available_source_ids",
        "observed_source_count",
        "observed_occurrence_count",
        "provenance_json",
        "region_annotations_json",
        "leakage_group_id",
        "study_split_groups_json",
        "component_holdout_groups_json",
        "reaction_family_holdout_groups",
        "prospective_lock_status",
        "biological_label_policy",
        "heavy_atoms",
        "formal_charge",
        "fragment_count",
        "ring_count",
        "aromatic_ring_count",
        "rotatable_bonds",
        "stereogenic_atom_count",
        "molecular_weight",
        "elements",
        "r0_pretraining_eligible",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (canonical, group) in enumerate(sorted(by_structure.items()), start=1):
            profile = molecular_profile(canonical)
            structure_hash = str(profile["structure_sha256"])
            provenance: dict[str, Any] = {}
            study_groups: dict[str, list[str]] = {}
            component_groups: dict[str, dict[str, list[str]]] = {}
            for item in sorted(group, key=lambda candidate: candidate.source_id):
                provenance[item.source_id] = {
                    "source_occurrence_count": item.occurrence_count,
                    "source_record_ids": sorted(item.record_ids),
                    "metadata": {
                        key: sorted(values)
                        for key, values in sorted(item.metadata_values.items())
                    },
                    "reaction_family": item.reaction_family,
                }
                study_tokens: set[str] = set()
                for key in ("Experiment_ID", "Publication_PMID", "Library_ID"):
                    study_tokens.update(item.metadata_values.get(key, set()))
                if study_tokens:
                    study_groups[item.source_id] = sorted(study_tokens)
                if item.region_values:
                    component_groups[item.source_id] = {
                        key: sorted(values) for key, values in sorted(item.region_values.items())
                    }
            all_group = all_memberships[canonical]
            writer.writerow(
                {
                    "r0_structure_id": f"R0-{index:06d}",
                    "canonical_isomeric_smiles": canonical,
                    **profile,
                    "observed_source_ids": _sorted_join(item.source_id for item in group),
                    "all_available_source_ids": _sorted_join(
                        item.source_id for item in all_group
                    ),
                    "observed_source_count": len(group),
                    "observed_occurrence_count": sum(item.occurrence_count for item in group),
                    "provenance_json": json.dumps(provenance, sort_keys=True),
                    "region_annotations_json": json.dumps(component_groups, sort_keys=True),
                    "leakage_group_id": f"mol-{structure_hash}",
                    "study_split_groups_json": json.dumps(study_groups, sort_keys=True),
                    "component_holdout_groups_json": json.dumps(
                        component_groups, sort_keys=True
                    ),
                    "reaction_family_holdout_groups": _sorted_join(
                        item.reaction_family for item in group
                    ),
                    "prospective_lock_status": "retrospective_pretraining_only",
                    "biological_label_policy": "no_label_inheritance_in_structural_pretraining",
                    "r0_pretraining_eligible": profile["fragment_count"] == 1,
                }
            )
    return {
        "release_id": "compose_lipid_r0_observed_v1",
        "row_count": len(by_structure),
        "eligible_row_count": sum(
            molecular_profile(canonical)["fragment_count"] == 1 for canonical in by_structure
        ),
        "excluded_virtual_only_structures": len(
            {row.canonical_smiles for row in rows if row.structure_status == "virtual_candidate"}
            - set(by_structure)
        ),
        "policy": {
            "unit": "canonical isomeric molecular structure",
            "measured_occurrences": "collapsed within source; occurrence counts retained",
            "split": "canonical structures never cross splits; study/component/reaction groups frozen",
            "labels": "no formulation or delivery label is inherited by structural pretraining rows",
            "prospective": "all retrospective structures are excluded from prospective novelty claims",
        },
    }


def overlap_report(rows: list[CanonicalAggregate]) -> dict[str, Any]:
    source_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_sets[row.source_id].add(row.canonical_smiles)
    source_ids = sorted(source_sets)
    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(source_ids):
        for right in source_ids[left_index + 1 :]:
            intersection = source_sets[left] & source_sets[right]
            union = source_sets[left] | source_sets[right]
            pairwise.append(
                {
                    "source_a": left,
                    "source_b": right,
                    "source_a_count": len(source_sets[left]),
                    "source_b_count": len(source_sets[right]),
                    "intersection_count": len(intersection),
                    "fraction_of_a": len(intersection) / len(source_sets[left]),
                    "fraction_of_b": len(intersection) / len(source_sets[right]),
                    "jaccard": len(intersection) / len(union),
                }
            )
    membership_histogram: dict[str, int] = defaultdict(int)
    all_structures = set().union(*source_sets.values())
    for structure in all_structures:
        membership = sum(structure in values for values in source_sets.values())
        membership_histogram[str(membership)] += 1
    return {
        "source_counts": {source: len(values) for source, values in sorted(source_sets.items())},
        "pairwise": pairwise,
        "union_count": len(all_structures),
        "source_membership_histogram": dict(sorted(membership_histogram.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_root = args.external_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = (
        SourceInput(
            source_id="lnpdb_v1",
            path=external_root / "external/LNPDB/LNPDB.csv",
            smiles_field="IL_SMILES",
            record_field="LNP_ID",
            structure_status="measured_or_commercial",
            reaction_family="mixed_literature",
            license_id="MIT",
            metadata_fields=("Experiment_ID", "Publication_PMID", "IL_name"),
            region_fields=(
                "IL_head_SMILES",
                "IL_linker_SMILES",
                "IL_tail1_SMILES",
                "IL_tail2_SMILES",
                "IL_tail3_SMILES",
                "IL_tail4_SMILES",
            ),
        ),
        SourceInput(
            source_id="agile_virtual12k",
            path=external_root / "data/agile_virtual/agile_virtual.csv",
            smiles_field="smiles",
            record_field=None,
            structure_status="virtual_candidate",
            reaction_family="ugi_3cr_fixed_core",
            license_id="MIT",
        ),
        SourceInput(
            source_id="agile_measured1200",
            path=external_root / "external/AGILE/AGILE_smiles_with_value_group.csv",
            smiles_field="combined_mol_SMILES",
            record_field="id",
            structure_status="measured",
            reaction_family="ugi_3cr",
            license_id="MIT",
            metadata_fields=("label",),
            region_fields=("A_smiles", "B_smiles", "C_smiles"),
        ),
        SourceInput(
            source_id="lion_repository_all",
            path=external_root / "external/LNP_ML/data/all_data.csv",
            smiles_field="smiles",
            record_field="Formulation_ID",
            structure_status="measured_formulation_occurrence",
            reaction_family="mixed_literature",
            license_id="MIT",
            metadata_fields=("Experiment_ID", "Library_ID", "Lipid_name"),
            region_fields=(
                "Amine_SMILES",
                "Amine",
                "Tail",
                "Ketone",
                "Isocyanide",
                "Aldehyde",
                "Carboxylic_acid",
                "Linker",
            ),
        ),
        SourceInput(
            source_id="lumi_4cr1920",
            path=REPO_ROOT / "tmp/lipid_data/lumi_lab/4CR-1920.csv",
            smiles_field="mol",
            record_field="Markush code",
            structure_status="measured",
            reaction_family="ugi_4cr",
            license_id="CC-BY-4.0",
            metadata_fields=("Markush code",),
        ),
    )
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(f"missing frozen source artifact: {source.path}")

    all_rows: list[CanonicalAggregate] = []
    summaries: list[dict[str, Any]] = []
    for source in sources:
        rows, summary = load_source(source)
        all_rows.extend(rows)
        summaries.append(summary)

    write_source_rows(output_dir / "available_source_structures.csv", all_rows)
    union_summary = write_union_rows(output_dir / "canonical_structure_union.csv", all_rows)
    r0_path = output_dir / "r0_observed_real_structures.csv"
    r0_summary = write_r0_release(r0_path, all_rows)
    overlaps = overlap_report(all_rows)
    source_profiles = {
        source.source_id: chemistry_profile(
            row for row in all_rows if row.source_id == source.source_id
        )
        for source in sources
    }
    real_rows = [row for row in all_rows if row.structure_status != "virtual_candidate"]
    real_by_structure: dict[str, CanonicalAggregate] = {}
    for row in real_rows:
        real_by_structure.setdefault(row.canonical_smiles, row)
    audit = {
        "inventory_version": "compose_lipid_pretraining_v1",
        "canonicalization_policy": {
            "parser": f"RDKit {rdBase.rdkitVersion}",
            "canonical": True,
            "isomeric_smiles": True,
            "explicit_hydrogens_removed": True,
            "tautomer_normalization": False,
            "charge_neutralization": False,
            "salt_or_fragment_stripping": False,
            "eligibility": "parseable and single connected component; originals retained",
        },
        "source_summaries": summaries,
        "source_chemistry_profiles": source_profiles,
        "r0_chemistry_profile": chemistry_profile(real_by_structure.values()),
        "overlap": overlaps,
        **union_summary,
        "r0_release": r0_summary,
        "enumerated_new_molecule_count": 0,
    }
    (output_dir / "canonicalization_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    r0_manifest = {
        **r0_summary,
        "artifact": {
            "path": str(r0_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(r0_path),
            "size_bytes": r0_path.stat().st_size,
        },
        "canonicalization_audit": {
            "path": str(
                (output_dir / "canonicalization_audit.json").relative_to(REPO_ROOT)
            ),
            "sha256": sha256_file(output_dir / "canonicalization_audit.json"),
        },
        "enumerated_new_molecule_count": 0,
    }
    (output_dir / "r0_release_manifest.json").write_text(
        json.dumps(r0_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
