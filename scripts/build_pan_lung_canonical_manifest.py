#!/usr/bin/env python3
"""Materialize the canonical row-level pan-lung manifest.

Joins LNPDB-lung, LUMI, and LuT into one typed-row table at the
(measurement x typed-head) grain, preserving formulation / cargo / dose / route /
species / tissue / cell / assay / readout / timepoint / study / batch / lipid
identity / provenance.  Heads stay semantically typed: in-vitro transfection,
functional in-vivo expression, biodistribution, and selectivity are distinct
readout types and are never pooled into one scalar.

Missingness is explicit (``missing_fields``); study-constant covariates are
recorded from the frozen source configs, never invented.  This is the stated
blocker for the R3/R4/R5 oracle-matrix cells.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/build_pan_lung_canonical_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

# LNPDB typed heads: (experiment_id, model_type) -> (typed_head, readout_type)
LNPDB_HEADS = {
    ("JW_2024", "A549"): ("airway_a549_in_vitro_expression", "in_vitro_transfection"),
    ("JW_2024", "HBEC_ALI"): ("airway_hbec_ali_in_vitro_expression", "in_vitro_transfection"),
    ("BL_2023", "Mouse_B6"): ("local_intratracheal_functional_expression", "functional_in_vivo_expression"),
    ("LX_2024", "Mouse_B6"): ("systemic_iv_barcoded_lung_uptake", "biodistribution_barcode_uptake"),
}

CANONICAL_FIELDS = [
    "canonical_row_id", "source_id", "typed_head", "readout_type",
    "publication_id", "experiment_id", "formulation_id", "batch_id", "replicate_id",
    "raw_lipid_id", "raw_smiles", "canonical_lipid_id", "has_full_structure",
    "head_smiles", "linker_smiles", "tail_smiles",
    "helper_lipids", "component_ratios", "np_or_charge_ratio",
    "cargo", "cargo_type", "dose", "dose_unit", "route", "species",
    "tissue", "cell_type", "assay", "raw_value", "raw_unit", "timepoint",
    "pooling_type", "barcode_id", "structure_family", "reaction_family",
    "study_group", "chemical_group", "source_file", "source_row", "license",
    "missing_fields",
]


def canonical_smiles(value: str) -> str | None:
    molecule = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule else None


def _row_id(source_id: str, source_row: str, typed_head: str) -> str:
    return hashlib.sha256(f"{source_id}\0{source_row}\0{typed_head}".encode()).hexdigest()[:16]


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _pipe(values) -> str:
    return "|".join(v for v in (_clean(x) for x in values) if v)


def build_lnpdb_rows(path: Path, license_id: str) -> list[dict]:
    frame = pd.read_csv(path, low_memory=False)
    lung = frame[frame["Model_target"].astype(str).str.contains("lung", case=False, na=False)].copy()
    if len(lung) != 1975:
        raise SystemExit(f"expected frozen 1,975-row LNPDB lung slice, found {len(lung)}")
    rows: list[dict] = []
    for idx, record in lung.iterrows():
        key = (_clean(record["Experiment_ID"]), _clean(record["Model_type"]))
        if key not in LNPDB_HEADS:
            raise SystemExit(f"unexpected LNPDB lung head {key} at row {idx}")
        typed_head, readout = LNPDB_HEADS[key]
        canon = canonical_smiles(record["IL_SMILES"])
        model_type = _clean(record["Model_type"])
        species = "human_cell_line" if model_type in {"A549", "HBEC_ALI"} else "mouse_C57BL6"
        helpers = _pipe([record.get("HL_name"), record.get("CHL_name"),
                         record.get("PEG_name"), record.get("fifthcomponent_name")])
        ratios = _pipe([record.get("IL_molratio"), record.get("HL_molratio"),
                        record.get("CHL_molratio"), record.get("PEG_molratio")])
        missing = [f for f, v in {"cargo": record.get("Cargo"),
                                  "dose": record.get("Dose_ug_nucleicacid"),
                                  "canonical_lipid_id": canon}.items() if not _clean(v)]
        rows.append({
            "canonical_row_id": _row_id("lnpdb_2026", str(record["Index"]), typed_head),
            "source_id": "lnpdb_2026", "typed_head": typed_head, "readout_type": readout,
            "publication_id": _clean(record.get("Publication_PMID")),
            "experiment_id": _clean(record.get("Experiment_ID")),
            "formulation_id": _clean(record.get("Formulation_ID")),
            "batch_id": _clean(record.get("Experiment_batching")),
            "replicate_id": "",
            "raw_lipid_id": _clean(record.get("IL_name")),
            "raw_smiles": _clean(record.get("IL_SMILES")),
            "canonical_lipid_id": canon or "",
            "has_full_structure": bool(canon),
            "head_smiles": _clean(record.get("IL_head_SMILES")),
            "linker_smiles": _clean(record.get("IL_linker_SMILES")),
            "tail_smiles": _pipe([record.get("IL_tail1_SMILES"), record.get("IL_tail2_SMILES"),
                                  record.get("IL_tail3_SMILES"), record.get("IL_tail4_SMILES")]),
            "helper_lipids": helpers, "component_ratios": ratios,
            "np_or_charge_ratio": _clean(record.get("IL_to_nucleicacid_chargeratio")),
            "cargo": _clean(record.get("Cargo")), "cargo_type": _clean(record.get("Cargo_type")),
            "dose": _clean(record.get("Dose_ug_nucleicacid")), "dose_unit": "ug_nucleic_acid",
            "route": _clean(record.get("Route_of_administration")), "species": species,
            "tissue": "lung", "cell_type": model_type,
            "assay": _clean(record.get("Experiment_method")),
            "raw_value": _clean(record.get("Experiment_value")), "raw_unit": "lnpdb_experiment_value",
            "timepoint": "",
            "pooling_type": "barcode_pool" if readout == "biodistribution_barcode_uptake" else "individual",
            "barcode_id": "", "structure_family": "", "reaction_family": "",
            "study_group": f"lnpdb_2026:{_clean(record.get('Experiment_ID'))}",
            "chemical_group": canon or f"lnpdb_row:{record['Index']}",
            "source_file": path.name, "source_row": str(record["Index"]), "license": license_id,
            "missing_fields": "|".join(missing),
        })
    return rows


def build_lumi_rows(path: Path, license_id: str) -> list[dict]:
    frame = pd.read_csv(path)
    rows: list[dict] = []
    for idx, record in frame.iterrows():
        canon = canonical_smiles(record["mol"])
        rows.append({
            "canonical_row_id": _row_id("lumi_lab_4cr1920", str(idx), "airway_hbe_in_vitro_expression"),
            "source_id": "lumi_lab_4cr1920",
            "typed_head": "airway_hbe_in_vitro_expression", "readout_type": "in_vitro_transfection",
            "publication_id": "10.1016/j.cell.2026.01.012", "experiment_id": "LUMI_4CR1920",
            "formulation_id": "", "batch_id": "", "replicate_id": "",
            "raw_lipid_id": _clean(record.get("Markush code")),
            "raw_smiles": _clean(record.get("mol")), "canonical_lipid_id": canon or "",
            "has_full_structure": bool(canon),
            "head_smiles": "", "linker_smiles": "", "tail_smiles": "",
            "helper_lipids": "", "component_ratios": "", "np_or_charge_ratio": "",
            "cargo": "", "cargo_type": "", "dose": "", "dose_unit": "",
            "route": "in_vitro", "species": "human_bronchial_epithelial",
            "tissue": "lung_airway", "cell_type": "HBE",
            "assay": "relative_luminescence", "raw_value": _clean(record.get("RLU (log2)")),
            "raw_unit": "log2_RLU", "timepoint": "",
            "pooling_type": "individual", "barcode_id": "",
            "structure_family": "ugi_4cr", "reaction_family": "ugi_4cr",
            "study_group": "lumi_lab_4cr1920:LUMI_4CR1920",
            "chemical_group": canon or f"lumi_row:{idx}",
            "source_file": path.name, "source_row": str(idx), "license": license_id,
            "missing_fields": "cargo|dose|formulation_ratios",
        })
    return rows


def build_lut_rows(path: Path, context: dict, license_id: str) -> list[dict]:
    frame = pd.read_csv(path)
    endpoints = [
        ("systemic_iv_lung_expression", "functional_in_vivo_expression",
         "lung_expression_log10_photons_per_second", "log10_photons_per_second"),
        ("systemic_iv_lung_selectivity", "selectivity",
         "lung_selectivity_fraction", "fraction_of_lung_liver_spleen"),
    ]
    rows: list[dict] = []
    for idx, record in frame.iterrows():
        component_id = f"lut:{_clean(record['head'])}+{_clean(record['tail'])}"
        for typed_head, readout, value_col, unit in endpoints:
            rows.append({
                "canonical_row_id": _row_id("lut_444_2026", str(idx), typed_head),
                "source_id": "lut_444_2026", "typed_head": typed_head, "readout_type": readout,
                "publication_id": "10.1038/s41551-026-01615-9", "experiment_id": "LuT_444",
                "formulation_id": "4A3-SC7_fixed_scaffold",
                "batch_id": f"round_{_clean(record.get('round'))}", "replicate_id": "",
                "raw_lipid_id": _clean(record.get("compound_id")) or component_id,
                "raw_smiles": "", "canonical_lipid_id": component_id, "has_full_structure": False,
                "head_smiles": "", "linker_smiles": "", "tail_smiles": "",
                "helper_lipids": "4A3-SC7_scaffold", "component_ratios": "", "np_or_charge_ratio": "",
                "cargo": context["cargo"], "cargo_type": "mRNA",
                "dose": str(context["dose_mg_per_kg"]), "dose_unit": "mg_per_kg",
                "route": context["route"], "species": context["species_strain"],
                "tissue": "lung", "cell_type": "whole_lung",
                "assay": "ivis_luciferase", "raw_value": _clean(record.get(value_col)),
                "raw_unit": unit, "timepoint": f"{context['time_hours']}h",
                "pooling_type": "individual", "barcode_id": "",
                "structure_family": "lut_component_defined", "reaction_family": "lut_4a3sc7_variable",
                "study_group": f"lut_444_2026:round_{_clean(record.get('round'))}",
                "chemical_group": component_id,
                "source_file": path.name, "source_row": str(idx), "license": license_id,
                "missing_fields": "raw_smiles|full_structure",
            })
    return rows


def overlap_report(rows: list[dict]) -> dict:
    by_source: dict[str, set[str]] = {}
    for row in rows:
        if row["has_full_structure"]:
            by_source.setdefault(row["source_id"], set()).add(row["canonical_lipid_id"])
    sources = sorted(by_source)
    overlaps = {}
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            shared = by_source[a] & by_source[b]
            overlaps[f"{a}__{b}"] = len(shared)
    return {
        "unique_full_structures_per_source": {s: len(v) for s, v in sorted(by_source.items())},
        "cross_source_exact_structure_overlap": overlaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "artifacts/oracles/pan_lung_canonical_v1")
    args = parser.parse_args()

    source_manifest = json.loads(
        (REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/source_manifest.json").read_text()
    )
    lnpdb_path = Path(next(s["local_artifact_path"] for s in source_manifest["sources"]
                           if s["source_id"] == "lnpdb_v1"))
    lumi_path = REPO_ROOT / "tmp/lipid_data/lumi_lab/4CR-1920.csv"
    lut_path = REPO_ROOT / "tmp/lipid_data/lut/lut_444_source_table.csv"
    lut_context = json.loads((REPO_ROOT / "configs/lut_444_systemic_lung_v1.json").read_text())["biological_context"]

    for path in (lumi_path, lut_path):
        if not path.exists():
            raise SystemExit(f"missing raw source {path}; re-fetch from primary origin first")

    rows: list[dict] = []
    rows += build_lnpdb_rows(lnpdb_path, "MIT")
    rows += build_lumi_rows(lumi_path, "CC-BY-4.0")
    rows += build_lut_rows(lut_path, lut_context, "springer_source_data")

    # deterministic order
    rows.sort(key=lambda r: (r["source_id"], r["typed_head"], r["source_row"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "canonical_rows.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    head_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    readout_counts: dict[str, int] = {}
    full_structure_rows = 0
    for row in rows:
        head_counts[row["typed_head"]] = head_counts.get(row["typed_head"], 0) + 1
        source_counts[row["source_id"]] = source_counts.get(row["source_id"], 0) + 1
        readout_counts[row["readout_type"]] = readout_counts.get(row["readout_type"], 0) + 1
        full_structure_rows += int(row["has_full_structure"])
    distinct_measurements = len({(r["source_id"], r["source_row"]) for r in rows})

    manifest = {
        "format": "compose_pan_lung_canonical_manifest_v1",
        "grain": "one row per (measurement x typed head)",
        "head_measurement_rows": len(rows),
        "distinct_source_measurements": distinct_measurements,
        "reconciliation_note": (
            "distinct_source_measurements matches the frozen corpus manifest count "
            "(1,975 LNPDB + 1,920 LUMI + 444 LuT = 4,339). LuT contributes 2 paired "
            "typed-head rows per compound (expression + selectivity), so head_measurement_rows > 4,339."
        ),
        "rows_by_typed_head": dict(sorted(head_counts.items())),
        "rows_by_source": dict(sorted(source_counts.items())),
        "rows_by_readout_type": dict(sorted(readout_counts.items())),
        "full_structure_rows": full_structure_rows,
        "component_only_rows": len(rows) - full_structure_rows,
        "representation_boundary": (
            "has_full_structure rows (LNPDB, LUMI) support molecular representations "
            "R1/R3/R4; LuT rows are component-defined and support only component "
            "representation R2. Do not impute molecular graphs for LuT."
        ),
        "readout_typing_boundary": (
            "in_vitro_transfection, functional_in_vivo_expression, "
            "biodistribution_barcode_uptake, and selectivity are distinct heads and "
            "are never pooled into one scalar target. Only the head matching the locked "
            "prospective profile defines reward."
        ),
        "split_groups": {
            "chemical_group": "held-lipid / exact-structure or component identity",
            "study_group": "leave-study-out (source_id:experiment_or_round)",
        },
        "overlap": overlap_report(rows),
        "sources": {
            "lnpdb_2026": {"path": str(lnpdb_path),
                           "sha256": hashlib.sha256(lnpdb_path.read_bytes()).hexdigest(), "license": "MIT"},
            "lumi_lab_4cr1920": {"path": str(lumi_path),
                                 "sha256": hashlib.sha256(lumi_path.read_bytes()).hexdigest(),
                                 "license": "CC-BY-4.0"},
            "lut_444_2026": {"path": str(lut_path),
                             "sha256": hashlib.sha256(lut_path.read_bytes()).hexdigest(),
                             "license": "springer_source_data"},
        },
        "canonical_rows_csv": {"path": str(csv_path.relative_to(REPO_ROOT)), "sha256": csv_sha,
                               "rows": len(rows)},
        "environment": {"rdkit": Chem.rdBase.rdkitVersion},
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"canonical rows: {len(rows)} head-measurement rows "
          f"({distinct_measurements} distinct source measurements)")
    print("rows by typed head:")
    for head, count in sorted(head_counts.items()):
        print(f"  {head:44s} {count}")
    print(f"full-structure rows: {full_structure_rows} | component-only: {len(rows)-full_structure_rows}")
    print(f"cross-source structure overlap: {manifest['overlap']['cross_source_exact_structure_overlap']}")
    print(f"csv: {csv_path.relative_to(REPO_ROOT)} (sha {csv_sha[:12]})")


if __name__ == "__main__":
    main()
