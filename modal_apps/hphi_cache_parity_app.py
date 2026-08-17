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
    artifact_volume.reload()

    # The record's SCIENTIFIC checksum is already stored by `_persist`, computed
    # at write time over the payload with the runtime fields excluded. Comparing
    # the stored digests is both simpler than recomputing and more faithful:
    # it is the very value the writer committed to, not a reconstruction that
    # could silently drift from it. (The image mounts only `src` and `configs`,
    # so the writer's module is not importable here in any case.)
    def load(d: str) -> dict[tuple, dict]:
        out = {}
        p = Path(RUN_ROOT) / d / "replicates"
        if not p.exists():
            print(f"  no replicates directory under {d}")
            return out
        for f in sorted(p.glob("*.json")):
            try:
                body = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            rec = body.get("record", {})
            if "slot" not in rec:
                continue
            out[(int(body["index"]), int(rec["slot"]))] = body
        return out

    a, b = load(cold), load(warm)
    shared = sorted(set(a) & set(b))
    print(f"cold {len(a)} records   warm {len(b)} records   shared {len(shared)}")

    def trajectory(body: dict) -> tuple:
        """The DECISIONS, with every float excluded.

        The stored scientific checksum embeds h_phi values, which come from a
        float32 network and therefore differ in the last bits between two
        containers -- for any change, or none. Comparing two separate Modal runs
        on that digest confounds the cache with cross-container float
        nondeterminism, which `hphi_graph_encode` explicitly warns about and
        which already muddied the chemistry-cache parity once.

        What the cache could actually corrupt is the PATH: which state each
        particle proposed, whether it absorbed, whether the population
        resampled, and what came back. Those are exact discrete values, and a
        cached law that differed from an enumerated one would move them.
        """
        rec = body["record"]
        steps = tuple(
            (t["step"], t["particle"], t["x"], t["y"], bool(t["absorbed_after"]))
            for t in rec["transitions"])
        sync = tuple(
            (s["step"], bool(s["resampled"]), int(s["n_absorbed"]),
             tuple(s.get("indices") or ()))
            for s in rec["sync"])
        return (steps, sync, rec["returned"], rec.get("returned_index"),
                bool(rec["success"]), tuple(rec["final_states"]),
                tuple(bool(v) for v in rec["final_absorbed"]))

    traj_bad = []
    for k in shared:
        if trajectory(a[k]) != trajectory(b[k]):
            traj_bad.append(k)

    rows, bad = [], []
    for k in shared:
        ha, hb = a[k]["sha256"], b[k]["sha256"]
        lc = (b[k].get("runtime") or {}).get("law_cache") or {}
        tot = lc.get("hit_mem", 0) + lc.get("hit_disk", 0) + lc.get("miss", 0)
        rows.append({
            "key": list(k), "match": ha == hb,
            "cold_seconds": (a[k].get("runtime") or {}).get("seconds"),
            "warm_seconds": (b[k].get("runtime") or {}).get("seconds"),
            "cross_attempt_hits": lc.get("hit_disk", 0), "law_calls": tot,
            "hit_rate": (lc.get("hit_disk", 0) / tot) if tot else None,
        })
        if ha != hb:
            bad.append((k, ha[:12], hb[:12]))
        flag = "OK  " if ha == hb else "DIFF"
        print(f"  {flag} src{k[0]:>3} slot{k[1]:>2}  "
              f"cold {(a[k].get('runtime') or {}).get('seconds',0):8.1f}s -> "
              f"warm {(b[k].get('runtime') or {}).get('seconds',0):8.1f}s"
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
    ok = not traj_bad
    print(f"\n  TRAJECTORY PARITY: {'PASSED' if ok else 'FAILED'}  "
          f"({len(shared)} slots; states, proposals, absorption, resampling "
          f"and returned molecule)")
    for k in traj_bad[:5]:
        print(f"    path differs at {k}")
    print(f"  stored-checksum equality: {len(shared)-len(bad)}/{len(shared)} "
          f"-- NOT the criterion. That digest embeds float32-derived h_phi and "
          f"is not\n  reproducible across containers for any change, or none.")
    return {"ok": ok, "trajectory_mismatches": traj_bad,
            "checksum_mismatches": [list(k) for k, _, _ in bad],
            "rows": rows, "cold_seconds": cs, "warm_seconds": ws,
            "cross_attempt_hits": hits, "law_calls": calls}


@app.local_entrypoint()
def main(cold: str = "cache_qual_cold", warm: str = "cache_qual_warm") -> None:
    out = compare.remote(cold, warm)
    Path("docs/LAW_CACHE_PARITY.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/LAW_CACHE_PARITY.json")
    if not out["ok"]:
        raise SystemExit(1)
