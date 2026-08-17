"""WHICH field diverges between the cold and warm records? Do not speculate.

The cold/warm parity check failed on 4 of 8 slots while the terminal molecules,
QED and similarity all matched by eye. That is a specific and diagnosable state:
something inside the scientific payload differs without changing the outcome.

Walks the payload field by field, and for the sequence fields reports the FIRST
index that differs and both values, so the divergence can be located in the
trajectory rather than inferred from the summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-record-diff")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
RUNTIME_FIELDS = ("seconds", "law_cache")


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=20 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def diff(cold: str, warm: str, index: int, slot: int) -> dict[str, Any]:
    artifact_volume.reload()

    def load(d):
        f = Path(RUN_ROOT) / d / "replicates" / f"{index:03d}_smc_{slot:02d}.json"
        return json.loads(f.read_text())["record"]

    a, b = load(cold), load(warm)
    keys = sorted(set(a) | set(b))
    report = []
    for k in keys:
        if k in RUNTIME_FIELDS:
            continue
        va, vb = a.get(k), b.get(k)
        if va == vb:
            continue
        entry: dict[str, Any] = {"field": k}
        if isinstance(va, list) and isinstance(vb, list):
            entry["len"] = [len(va), len(vb)]
            first = next((i for i in range(min(len(va), len(vb)))
                          if va[i] != vb[i]), None)
            entry["first_differing_index"] = first
            if first is not None:
                entry["cold"] = va[first]
                entry["warm"] = vb[first]
        else:
            entry["cold"] = va
            entry["warm"] = vb
        report.append(entry)

    print(f"src{index} slot{slot}: {len(report)} differing scientific fields\n")
    for e in report:
        print(f"  FIELD {e['field']}")
        for kk, vv in e.items():
            if kk == "field":
                continue
            s = json.dumps(vv)
            print(f"      {kk}: {s[:400]}")
        print()
    return {"index": index, "slot": slot, "differing": report}


@app.local_entrypoint()
def main(cold: str = "cache_qual_cold", warm: str = "cache_qual_warm",
         index: int = 51, slot: int = 0) -> None:
    out = diff.remote(cold, warm, index, slot)
    Path("docs/RECORD_DIFF.json").write_text(json.dumps(out, indent=1))
    print(f"\n{len(out['differing'])} differing fields -> docs/RECORD_DIFF.json")
