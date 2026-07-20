#!/usr/bin/env python3
"""Extract the 444 non-control LuT rows from the official source workbooks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def potency_records(path: Path, sheet: str, round_index: int) -> list[dict[str, object]]:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    records: list[dict[str, object]] = []
    if round_index == 1:
        heads = tuple(str(value) for value in raw.iloc[0, 1:] if pd.notna(value))
        for row in range(1, len(raw)):
            tail = str(raw.iloc[row, 0])
            for column, head in enumerate(heads, start=1):
                value = raw.iloc[row, column]
                if head != "DOTAP" and pd.notna(value):
                    records.append({"round": 1, "head": head, "tail": tail,
                                    "lung_expression_log10_photons_per_second": float(value)})
    else:
        tails = tuple(str(value) for value in raw.iloc[0, 1:] if pd.notna(value))
        for row in range(1, len(raw)):
            head = str(raw.iloc[row, 0])
            for column, tail in enumerate(tails, start=1):
                value = raw.iloc[row, column]
                if tail != "DOTAP" and pd.notna(value):
                    records.append({"round": 2, "head": head, "tail": tail,
                                    "lung_expression_log10_photons_per_second": float(value)})
    return records


def selectivity_records(path: Path, sheet: str, round_index: int) -> list[dict[str, object]]:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    records: list[dict[str, object]] = []
    if round_index == 1:
        heads = tuple(str(value) for value in raw.iloc[0, 2:] if pd.notna(value))
        for row in range(1, len(raw)):
            if str(raw.iloc[row, 1]) != "Lu":
                continue
            tail = str(raw.iloc[row - 2, 0])
            for column, head in enumerate(heads, start=2):
                value = raw.iloc[row, column]
                if pd.notna(value):
                    records.append({"round": 1, "head": head, "tail": tail,
                                    "lung_selectivity_fraction": float(value)})
    else:
        tails = tuple(str(value) for value in raw.iloc[0, 1:] if pd.notna(value))
        for row in range(1, len(raw)):
            head = str(raw.iloc[row, 0])
            for column, tail in enumerate(tails, start=1):
                value = raw.iloc[row, column]
                if pd.notna(value):
                    records.append({"round": 2, "head": head, "tail": tail,
                                    "lung_selectivity_fraction": float(value)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig2", type=Path, default=Path("tmp/lipid_data/lut/source_data_fig2.xlsx"))
    parser.add_argument("--fig3", type=Path, default=Path("tmp/lipid_data/lut/source_data_fig3.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("tmp/lipid_data/lut/lut_444_source_table.csv"))
    args = parser.parse_args()

    potency = pd.DataFrame.from_records(
        potency_records(args.fig2, "Fig.2b", 1) + potency_records(args.fig3, "Fig.3b", 2)
    )
    selectivity = pd.DataFrame.from_records(
        selectivity_records(args.fig2, "Fig.2c", 1)
        + selectivity_records(args.fig3, "Fig.3c", 2)
    )
    keys = ["round", "head", "tail"]
    frame = potency.merge(selectivity, on=keys, how="inner", validate="one_to_one")
    if len(frame) != 444 or frame.duplicated(keys).any() or frame.isna().any().any():
        raise ValueError("LuT source reconstruction failed the frozen 444-row contract")
    frame["head_family"] = frame["head"].str.extract(r"^(\d+A)", expand=False)
    frame["head_index"] = frame["head"].str.extract(r"A(\d+)$", expand=False).astype(int)
    frame["tail_index"] = frame["tail"].str.extract(r"B(\d+)$", expand=False).astype(int)
    frame["tripod_compatible_head"] = frame["head"].isin(
        {"1A1", "1A2", "1A3", "1A6", "1A7", "1A8"}
    ).astype(int)
    frame["tripod_compatible_tail"] = (frame["tail_index"] >= 4).astype(int)
    frame["branched_tail"] = frame["tail"].isin({f"B{i}" for i in range(13, 19)}).astype(int)
    frame["unsaturated_tail"] = frame["tail"].isin(
        {"B19", "B20", "B21", "B22", "B23", "B24", "B25"}
    ).astype(int)
    frame["near_head_ester"] = frame["tail"].isin({"B11", "B16", "B20"}).astype(int)
    frame["compound_id"] = frame["head"] + frame["tail"]
    frame = frame.sort_values(keys).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.groupby("round").size().to_json())


if __name__ == "__main__":
    main()
