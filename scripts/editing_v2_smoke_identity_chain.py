#!/usr/bin/env python
"""Smoke the rebuilt identity chain before spending GPU time.

Replicates, on the CPU and against the same files the run will read, every
binding check the training app performs at startup -- plus the ones it cannot
perform because they need the whole sequence rather than a membership test.

The point is not to re-derive the artifacts. It is to fail HERE, in seconds,
rather than three minutes into a paid container, on the class of mistake that
has already happened twice in this project: an artifact that describes a
different split than the one it is loaded next to.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--packed-header", type=Path, default=None,
                        help="PACKED_HEADER.json, if a copy is available locally.")
    args = parser.parse_args()

    inputs = args.inputs
    law = json.loads((inputs / "editing_v2_sampling_law_v2.json").read_text())
    manifest = json.loads(
        (inputs / "editing_v2_prepared_training_manifest_v2.json").read_text())
    index = json.loads(
        (inputs / "editing_v2_prepared_training_index_v2.json").read_text())
    gate = json.loads((inputs / "editing_v2_post_split_freeze_gate.json").read_text())
    with gzip.open(
            inputs / "editing_v2_matched_validation_reserve_ids.json.gz", "rt") as handle:
        reserve = json.load(handle)
    with gzip.open(inputs / "ENTRY_PROVENANCE.json.gz", "rt") as handle:
        provenance = json.load(handle)

    sequence = index["sequence"]
    training_ids = set(reserve["training_entry_ids"])
    reserve_ids = set(reserve["reserve_entry_ids"])
    bands = reserve["reserve_band_by_entry_id"]

    checks: dict[str, bool] = {}

    # ---- the app's own startup refusals, reproduced --------------------------
    checks["gate_is_frozen"] = gate["status"] == "FROZEN"
    checks["manifest_binds_this_law"] = (
        manifest["law_frozen_sha256"] == law["frozen_sha256"])
    checks["gate_ran_against_this_law"] = (
        gate["law_frozen_sha256"] == law["frozen_sha256"])
    checks["gate_ran_against_this_manifest"] = (
        gate["manifest_frozen_sha256"] == manifest["frozen_sha256"])
    checks["law_and_manifest_bind_one_reserve"] = (
        law.get("matched_validation_reserve_source_digest")
        == manifest.get("matched_validation_reserve_source_digest"))
    checks["sequence_stays_inside_the_split"] = not (set(sequence) - training_ids)
    checks["no_reserve_entry_in_the_sequence"] = not (reserve_ids & set(sequence))

    # ---- what the app cannot check cheaply at startup ------------------------
    checks["sequence_digest_matches_manifest"] = (
        hashlib.sha256("\n".join(sequence).encode()).hexdigest()
        == manifest["sequence_sha256"])
    checks["every_reserve_entry_has_a_band"] = all(i in bands for i in reserve_ids)
    checks["bands_are_declared_values"] = set(bands.values()) <= {
        "0", "1-4", "5-24", "25+"}
    checks["reserve_and_training_are_disjoint"] = not (reserve_ids & training_ids)

    lane_by_entry = provenance["lane_by_entry_id"]
    checks["every_reserve_entry_has_a_lane"] = all(
        i in lane_by_entry for i in reserve_ids)

    # Selection weights must actually reach the reserve, or the primary metric
    # silently renormalizes onto whatever happens to be there.
    synthetic_lane = provenance["synthetic_lane"]
    reserve_strata = collections.Counter()
    for entry_id in reserve_ids:
        lane = ("synthetic" if lane_by_entry.get(entry_id) == synthetic_lane
                else "real")
        reserve_strata[lane] += 1
    checks["reserve_carries_both_lanes"] = len(reserve_strata) == 2

    law_strata = {f'{s["family"]}|{s["provenance"]}': float(s["draw_share"])
                  for s in law["strata"]}
    measured = gate["standard_error_by_stratum"]
    covered = sum(law_strata[k] for k in law_strata if k in measured)
    checks["law_mass_reaching_the_reserve_above_99pc"] = covered >= 0.99 * sum(
        law_strata.values())

    # ---- the four views must be constructible from what is on disk ----------
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        capability_baseline,
        collapsed_capabilities,
        selection_criterion,
        summarize_panel,
    )
    import math

    band_of = dict(bands)
    fake = []
    for position, entry_id in enumerate(sorted(reserve_ids)[:400]):
        lane = ("synthetic" if lane_by_entry.get(entry_id) == synthetic_lane
                else "real")
        nll = 2.0 + (position % 5) * 0.1
        fake.append({
            "entry_id": entry_id,
            "model_family": "atom_delete",
            "capability_cell_id": f"cell{position % 3}",
            "lane": lane,
            "stratum": f"atom_delete|{lane}",
            "support_band": band_of.get(entry_id),
            "teacher_successor_nll": nll,
            "teacher_successor_probability": math.exp(-nll),
            "teacher_successor_log_probability": -nll,
        })
    summary = summarize_panel(
        fake,
        deployment_family_share=law["realized_coefficients"]["by_family"],
        reference_law_stratum_share=law_strata)
    baseline = capability_baseline(summary)
    criterion = selection_criterion(summary, step=1, baseline=baseline)
    checks["selection_number_is_computable"] = (
        summary["reference_law_weighted_mean_nll"] is not None)
    checks["support_bands_populate"] = len(summary["by_support_band"]) >= 2
    checks["nothing_collapses_against_its_own_baseline"] = not collapsed_capabilities(
        summary, baseline=baseline)
    checks["eligible_checkpoint_scores_eligible"] = criterion[0] == 1

    if args.packed_header and args.packed_header.exists():
        header = json.loads(args.packed_header.read_text())
        rows = set(header["entry_ids"])
        checks["packed_store_covers_the_whole_split"] = rows == (
            training_ids | reserve_ids)
        checks["packed_store_digest_is_the_expected_one"] = header[
            "records_sha256"].startswith("b232a6fa069f")

    width = max(len(name) for name in checks)
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<{width}}")
    failed = [name for name, passed in checks.items() if not passed]
    print(f"\nlaw          {law['frozen_sha256'][:12]}")
    print(f"manifest     {manifest['frozen_sha256'][:12]}")
    print(f"reserve      {law['matched_validation_reserve_source_digest']}")
    print(f"sequence     {manifest['sequence_sha256'][:12]}")
    print(f"training     {len(training_ids):,}   reserve {len(reserve_ids):,}   "
          f"sequence {len(sequence):,}")
    print(f"law mass reaching the reserve  {covered:.6f}")
    print(f"\n{'SMOKE PASSED' if not failed else 'SMOKE FAILED: ' + str(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
