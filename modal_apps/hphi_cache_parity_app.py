"""Does the persistent law cache change any trajectory? Cold run vs warm run.

The probe already showed a round-tripped law reproduces the float64 probability
bits, the sampled index under the same RNG state, and the successor SMILES. This
is the end-to-end version of that claim: run the same sentinel slots twice, once
with an empty cache and once with the cache the first run populated, and require
the SCIENTIFIC checksums to be identical.

The scientific checksum is the right comparison because it excludes the runtime
fields -- wall time, and the cache hit accounting itself -- which legitimately
differ between a cold and a warm run while the trajectory does not. Comparing
raw records would fail for exactly the reasons that do not matter, and comparing
only terminal molecules would pass even if the interior path had changed.

    modal run modal_apps/hphi_cache_parity_app.py \
        --cold cache_qual_cold --warm cache_qual_warm
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
app = modal.App("hphi-cache-parity")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=20 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def compare(cold: str, warm: str) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT))
    from modal_apps.hphi_smc_fast_app import scientific_sha256

    artifact_volume.reload()

    def load(d: str) -> dict[tuple, dict]:
        out = {}
        p = Path(RUN_ROOT) / d
        for f in sorted(p.glob("*.json")):
            try:
                body = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            rec = body.get("record", body)
            if "slot" not in rec:
                continue
            out[(body.get("index", rec.get("index")), int(rec["slot"]))] = rec
        return out

    a, b = load(cold), load(warm)
    shared = sorted(set(a) & set(b))
    print(f"cold {len(a)} records   warm {len(b)} records   shared {len(shared)}")

    rows, bad = [], []
    for k in shared:
        ha, hb = scientific_sha256(a[k]), scientific_sha256(b[k])
        lc = b[k].get("law_cache") or {}
        tot = lc.get("hit_mem", 0) + lc.get("hit_disk", 0) + lc.get("miss", 0)
        rows.append({
            "key": list(k), "match": ha == hb,
            "cold_seconds": a[k].get("seconds"), "warm_seconds": b[k].get("seconds"),
            "cross_attempt_hits": lc.get("hit_disk", 0), "law_calls": tot,
            "hit_rate": (lc.get("hit_disk", 0) / tot) if tot else None,
        })
        if ha != hb:
            bad.append((k, ha[:12], hb[:12]))
        flag = "OK  " if ha == hb else "DIFF"
        print(f"  {flag} src{k[0]:>3} slot{k[1]:>2}  "
              f"cold {a[k].get('seconds'):8.1f}s -> warm {b[k].get('seconds'):8.1f}s"
              f"   cross-attempt {lc.get('hit_disk',0):>4}/{tot:<4}"
              f" ({(lc.get('hit_disk',0)/tot*100) if tot else 0:5.1f}%)", flush=True)

    cs = sum(r["cold_seconds"] for r in rows if r["cold_seconds"])
    ws = sum(r["warm_seconds"] for r in rows if r["warm_seconds"])
    hits = sum(r["cross_attempt_hits"] for r in rows)
    calls = sum(r["law_calls"] for r in rows)
    print(f"\n  cold total {cs:9.1f} s")
    print(f"  warm total {ws:9.1f} s   speedup {cs/max(ws,1e-9):5.2f}x")
    print(f"  cross-attempt law hits {hits}/{calls} "
          f"({hits/max(calls,1)*100:.1f}%)")
    ok = not bad
    print(f"\n  TRAJECTORY PARITY: {'PASSED' if ok else 'FAILED'}  "
          f"({len(shared)} slots, scientific checksums "
          f"{'identical' if ok else 'DIFFER'})")
    for k, ha, hb in bad[:5]:
        print(f"    {k}: cold {ha} vs warm {hb}")
    return {"ok": ok, "rows": rows, "cold_seconds": cs, "warm_seconds": ws,
            "cross_attempt_hits": hits, "law_calls": calls}


@app.local_entrypoint()
def main(cold: str = "cache_qual_cold", warm: str = "cache_qual_warm") -> None:
    out = compare.remote(cold, warm)
    Path("docs/LAW_CACHE_PARITY.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/LAW_CACHE_PARITY.json")
    if not out["ok"]:
        raise SystemExit(1)
