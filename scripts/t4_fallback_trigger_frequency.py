#!/usr/bin/env python3
"""How often would a zero-support fallback have fired, on every committed T4 round?

ZERO ORACLE CALLS. This reads committed artifacts on the campaign volumes and nothing
else: no proposal is synthesised, no molecule is docked, no Modal function is invoked.

The question it answers is the one that decides whether a generic zero-support fallback
is surgical or a fifteen-cell re-run. The shipped controller's round loop is

    proposals -> merge -> candidates (endpoints not already in the archive)
      -> selected = select_batch(candidates, ..., batch=room >= 1)
      -> if not selected:  TERMINAL candidate_exhaustion
      -> dock, update, and if budget <= 0:  TERMINAL complete_budget

so a round that yields zero eligible candidates does not merely fail, it ENDS THE CELL.
Every round that produced a lock therefore had at least one eligible candidate, and the
per-round margin -- how many candidates the smallest committed round actually carried --
is the quantity that says how close a searching cell came to the trigger.

`pool_census.unique_endpoints` is `len(candidates)`; `selected_census.unique_endpoints`
is `len(selected)`. Both are written into the lock BEFORE docking, so they are free.

Usage:
    MODAL_PROFILE=nitya python3 scripts/t4_fallback_trigger_frequency.py --out <path>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: volume -> the panel arm it produced. Delta is taken from the cell's own result
#: payload where present and from this table otherwise; both are recorded.
PANEL_VOLUMES = {
    "compose-t4-held-target-distilled-parp1-d06-250": ("parp1", 0.6),
    "compose-t4-held-target-distilled-braf-d06-250": ("braf", 0.6),
    "compose-t4-held-target-distilled-5ht1b-d06-250": ("5ht1b", 0.6),
    "compose-t4-held-target-distilled-fa7-d06-250": ("fa7", 0.6),
    "compose-t4-held-target-jak2-true-d06-250": ("jak2", 0.6),
    "compose-t4-held-target-distilled-parp1-d04-250": ("parp1", 0.4),
    "compose-t4-held-target-distilled-braf-d04-250": ("braf", 0.4),
    "compose-t4-held-target-distilled-5ht1b-d04-250": ("5ht1b", 0.4),
    "compose-t4-held-target-distilled-fa7-d04-250": ("fa7", 0.4),
    # named `...jak2-d06-250` and declaring delta 0.4; the contract is the authority.
    "compose-t4-held-target-distilled-jak2-d06-250": ("jak2", 0.4),
}

RESCUE_VOLUMES = {
    "compose-t4-5ht1b2-protonation-rescue-d04": ("5ht1b", 0.4, "protonation"),
    "compose-t4-5ht1b2-protonation-rescue-d06": ("5ht1b", 0.6, "protonation"),
    "compose-t4-region-repair-rescue-fa7-d04": ("fa7", 0.4, "region_repair"),
    "compose-t4-region-repair-rescue-fa7-d06": ("fa7", 0.6, "region_repair"),
    "compose-t4-region-repair-rescue-braf-d06": ("braf", 0.6, "region_repair"),
}


def _read(volume, path):
    return json.loads(b"".join(volume.read_file(path)))


def _payload(document):
    return document.get("payload", document)


def scan_volume(volume_name: str) -> list[dict]:
    import modal

    volume = modal.Volume.from_name(volume_name)
    rows = []
    for run_entry in volume.listdir(""):
        run_id = run_entry.path.split("/")[-1]
        if run_entry.type != 2:
            continue
        try:
            cells = [e for e in volume.listdir(run_id) if e.type == 2]
        except Exception as error:  # noqa: BLE001 - a malformed run is reported, not fatal
            rows.append({"volume": volume_name, "run_id": run_id, "error": repr(error)})
            continue
        for cell_entry in cells:
            cell = cell_entry.path.split("/")[-1]
            row = {
                "volume": volume_name,
                "run_id": run_id,
                "cell": cell,
                "result_status": None,
                "result_charged_calls": None,
                "result_final_best": None,
                "rounds": [],
            }
            try:
                names = [e.path for e in volume.listdir(f"{run_id}/{cell}")]
            except Exception as error:  # noqa: BLE001
                row["error"] = repr(error)
                rows.append(row)
                continue
            for path in names:
                leaf = path.split("/")[-1]
                if leaf == "result.json":
                    try:
                        result = _payload(_read(volume, path))
                    except Exception as error:  # noqa: BLE001
                        row["result_error"] = repr(error)
                        continue
                    row["result_status"] = result.get("status")
                    row["result_charged_calls"] = result.get("charged_calls")
                    row["result_final_best"] = result.get("final_best")
                    row["result_claim_boundary"] = result.get("claim_boundary")
                elif leaf.startswith("round_") and leaf.endswith("_lock.json"):
                    try:
                        lock = _payload(_read(volume, path))
                    except Exception as error:  # noqa: BLE001
                        row["rounds"].append({"lock": leaf, "error": repr(error)})
                        continue
                    pool = lock.get("pool_census") or {}
                    chosen = lock.get("selected_census") or {}
                    candidate_pool = lock.get("candidate_pool")
                    row["rounds"].append(
                        {
                            "round": lock.get("round"),
                            "charged_before": lock.get("charged_before"),
                            "pool_unique_endpoints": pool.get("unique_endpoints"),
                            "candidate_pool_len": (
                                None if candidate_pool is None else len(candidate_pool)
                            ),
                            "selected_unique_endpoints": chosen.get("unique_endpoints"),
                            "queries": len(lock.get("queries") or ()),
                            "parents": len(lock.get("parents") or ()),
                            "contract_payload_sha256": lock.get("contract_payload_sha256"),
                        }
                    )
            row["rounds"].sort(key=lambda entry: (entry.get("round") is None, entry.get("round")))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="diagnostics/t4_fallback_trigger_frequency_v1.json")
    parser.add_argument("--only", default=None, help="scan one volume name only")
    args = parser.parse_args()

    report = {
        "schema_version": "t4_fallback_trigger_frequency_v1",
        "oracle_calls": 0,
        "docking_calls": 0,
        "method": (
            "committed-artifact read only; a round lock is written after select_batch "
            "returns a non-empty batch and before any docking, so every number here is free"
        ),
        "panel": [],
        "rescue": [],
    }
    targets = []
    for name, (protein, delta) in PANEL_VOLUMES.items():
        targets.append(("panel", name, {"protein": protein, "delta": delta}))
    for name, (protein, delta, mechanism) in RESCUE_VOLUMES.items():
        targets.append(("rescue", name, {"protein": protein, "delta": delta, "mechanism": mechanism}))
    for bucket, name, meta in targets:
        if args.only and name != args.only:
            continue
        print(f"scanning {name}", flush=True)
        for row in scan_volume(name):
            report[bucket].append({**meta, **row})
            if row.get("cell"):
                print(
                    f"  {row['run_id'][:12]} {row['cell']}: status={row['result_status']} "
                    f"calls={row['result_charged_calls']} rounds={len(row['rounds'])}",
                    flush=True,
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
