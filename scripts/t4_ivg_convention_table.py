#!/usr/bin/env python3
"""Rebuild the T4 panel from round locks, in InVirtuoGen's reporting convention.

SUPERSEDED FOR REPORTING.  The authoritative T4 result is
``diagnostics/T4_FROZEN_RESULT_v1.json`` (human-readable: ``T4_FROZEN_RESULT_v1.md``).
Read that file; do not quote this script's output as the panel.

This builder resolves ONE run per volume -- the run whose launch pins the live
contract payload.  Work migrates to a new ``run_id`` after a resume or a re-seal, so
that rule reports an empty run as a cell's result: measured, braf-d06 holds one run
with 5 round locks and another with 27, and the empty one won.  It also cannot see a
molecule scored in a cell's FINAL round, because a round lock records a molecule only
once it becomes a PARENT -- which understated 5ht1b seed 3 at delta 0.4 by 0.2.

Kept for per-run inspection.  It is not the panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
IVG = ROOT / "docs/invirtuogen_t4_targets.json"

# volume -> contract, per (protein, delta). The contract is the delta authority.
CAMPAIGNS = {
    ("parp1", 0.6): ("compose-t4-held-target-distilled-parp1-d06-250",
                     "configs/t4_held_target_distilled_parp1_d06_250.json"),
    ("braf", 0.6): ("compose-t4-held-target-distilled-braf-d06-250",
                    "configs/t4_held_target_distilled_braf_d06_250.json"),
    ("5ht1b", 0.6): ("compose-t4-held-target-distilled-5ht1b-d06-250",
                     "configs/t4_held_target_distilled_5ht1b_d06_250.json"),
    ("fa7", 0.6): ("compose-t4-held-target-distilled-fa7-d06-250",
                   "configs/t4_held_target_distilled_fa7_d06_250.json"),
    # NOT the volume named `...jak2-d06-250`: that contract declares delta 0.4.
    ("jak2", 0.6): ("compose-t4-held-target-jak2-true-d06-250",
                    "configs/t4_held_target_distilled_jak2_true_d06_250.json"),
    ("parp1", 0.4): ("compose-t4-held-target-distilled-parp1-d04-250",
                     "configs/t4_held_target_distilled_parp1_d04_250.json"),
    ("braf", 0.4): ("compose-t4-held-target-distilled-braf-d04-250",
                    "configs/t4_held_target_distilled_braf_d04_250.json"),
    ("5ht1b", 0.4): ("compose-t4-held-target-distilled-5ht1b-d04-250",
                     "configs/t4_held_target_distilled_5ht1b_d04_250.json"),
    ("fa7", 0.4): ("compose-t4-held-target-distilled-fa7-d04-250",
                   "configs/t4_held_target_distilled_fa7_d04_250.json"),
    ("jak2", 0.4): ("compose-t4-held-target-distilled-jak2-d06-250",
                    "configs/t4_held_target_distilled_jak2_d06_250.json"),
}

# Rescue arms are a NAMED separate phase. Their own claim_boundary forbids splicing
# them into the unchanged panel, so they are reported beside it, never merged in.
RESCUES = {
    ("5ht1b", 0.4): "compose-t4-5ht1b2-protonation-rescue-d04",
    ("5ht1b", 0.6): "compose-t4-5ht1b2-protonation-rescue-d06",
    ("fa7", 0.4): "compose-t4-region-repair-rescue-fa7-d04",
    ("fa7", 0.6): "compose-t4-region-repair-rescue-fa7-d06",
    ("braf", 0.6): "compose-t4-region-repair-rescue-braf-d06",
}


def payload_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(volume, path):
    return json.loads(b"".join(volume.read_file(path)))


def contract_delta(relative: str) -> tuple[float, str]:
    envelope = json.loads((ROOT / relative).read_text())
    payload = envelope.get("payload", envelope)
    return float(payload["delta"]), payload_hash(payload)


def authoritative_run(volume, expected_payload: str):
    """The run whose launch pins the live contract payload. Never the longest."""
    candidates = []
    for entry in volume.listdir(""):
        run_id = entry.path.split("/")[-1]
        launch = None
        try:
            launch = read_json(volume, f"{run_id}/launch.json")["payload"]
        except Exception as error:  # noqa: BLE001 - a dir without a launch is not a run
            print(f"    skip {run_id[:12]}: {type(error).__name__}", flush=True)
        if launch is not None:
            candidates.append((run_id, launch.get("contract_payload_sha256"), launch))
    matching = [c for c in candidates if c[1] == expected_payload]
    if len(matching) == 1:
        return matching[0]
    if not matching:
        raise RuntimeError(
            f"no run pins the live contract payload {expected_payload[:16]}; "
            f"runs present: {[(r[:12], (p or '')[:12]) for r, p, _ in candidates]}"
        )
    raise RuntimeError(f"{len(matching)} runs pin one payload; reconcile by identity, not by score")


def cell_scores(volume, run_id, cell):
    """Best eligible docked score and reconciled charged calls, from locks alone."""
    best, charged, rounds = None, 0, 0
    for entry in volume.listdir(f"{run_id}/{cell}"):
        name = entry.path.split("/")[-1]
        if not (name.startswith("round_") and name.endswith("_lock.json")):
            continue
        lock = read_json(volume, entry.path)["payload"]
        rounds += 1
        # `charged_before` is a LOWER BOUND: a round that rolled back and was redone
        # rewrites its own lock with a smaller count, so take the maximum observed.
        charged = max(charged, int(lock.get("charged_before") or 0))
        for query in lock.get("queries") or ():
            score = query.get("parent_score")
            if score is not None and (best is None or float(score) < best):
                best = float(score)
    return best, charged, rounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="diagnostics/t4_ivg_convention_table_v1.json")
    args = parser.parse_args()
    import modal

    seeds = json.load(SEEDS.open())
    ivg = {i: row for i, row in enumerate(json.load(IVG.open())["rows"])}
    report = {"schema_version": "t4_ivg_convention_table_v1", "cells": [], "rescues": []}

    for (protein, delta), (volume_name, contract) in sorted(CAMPAIGNS.items()):
        declared, expected = contract_delta(contract)
        if abs(declared - delta) > 1e-9:
            raise RuntimeError(
                f"{contract} declares delta {declared}, mapped here as {delta}; "
                "delta comes from the contract, so fix the map"
            )
        volume = modal.Volume.from_name(volume_name)
        run_id, _, launch = authoritative_run(volume, expected)
        for row in [s for s in seeds if s["target"] == protein]:
            cell = f"{protein}_{row['idx'] % 3}"
            try:
                best, charged, rounds = cell_scores(volume, run_id, cell)
            except Exception as error:  # noqa: BLE001 - an absent cell is reported, not fatal
                best, charged, rounds = None, 0, 0
                print(f"  {protein} {cell} d{delta}: {type(error).__name__} {error}", flush=True)
            report["cells"].append({
                "target": protein, "seed_index": row["idx"], "cell": cell, "delta": delta,
                "best_eligible_docking": best, "reconciled_charged_calls": charged,
                "committed_rounds": rounds, "run_id": run_id, "volume": volume_name,
                "contract": contract, "contract_payload_sha256": expected,
                "code_revision": launch.get("code_revision"),
                "ivg": ivg[row["idx"]].get(f"invirtuo_d0{'4' if delta == 0.4 else '6'}"),
                "ivg_parenthesised": ivg[row["idx"]]["invirtuo_parenthesised"],
            })
            print(f"  {protein}_{row['idx']} d{delta}: best={best} charged={charged} "
                  f"rounds={rounds}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
