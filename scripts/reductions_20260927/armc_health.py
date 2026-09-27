"""Gentle Arm C health check: ONE listdir of the root, then one progress.json read per
campaign, spaced out. Directory listings of oracle/ are what tripped the volume rate limit."""
import json, sys, time
import modal

v = modal.Volume.from_name("compose-pmo-fibercontrol")
ns = sorted({e.path.split("/")[0] for e in v.listdir("/")})
print(f"{len(ns)} arm-C campaigns on kosha-labs")
live = dead = stalled = 0
rows = []
for n in ns:
    time.sleep(0.4)
    prog = None
    try:
        prog = json.loads(b"".join(v.read_file(f"{n}/progress.json")))
    except Exception:
        pass
    rc = None
    try:
        rc = json.loads(b"".join(v.read_file(f"{n}/provenance.json"))).get("returncode")
    except Exception:
        pass
    ch = (prog or {}).get("charged_oracle_calls")
    rd = (prog or {}).get("round")
    best = (prog or {}).get("best_score")
    tag = n.replace("scored_rebind_", "").rsplit("_", 1)[0]
    st = "DEAD" if rc is not None else ("LIVE" if ch else "START")
    if rc is not None: dead += 1
    elif ch: live += 1
    else: stalled += 1
    rows.append((st, ch, rd, best, tag, rc))
for st, ch, rd, best, tag, rc in sorted(rows, key=lambda r: -(r[1] or 0)):
    b = f"{best:.4f}" if isinstance(best, float) else str(best)
    print(f"  {st:5s} charged={str(ch):>5s} round={str(rd):>4s} best={b:>8s}  {tag}"
          + (f"  rc={rc}" if rc is not None else ""))
print(f"\nLIVE {live} | DEAD {dead} | not yet charging {stalled}")
