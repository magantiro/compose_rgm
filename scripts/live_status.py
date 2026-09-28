"""Live status across the T4 replicate panel and the PMO campaigns.

Reads DURABLE ARTIFACTS only -- round locks for T4, progress.json for PMO -- never
`modal app list`, whose description column is truncated and which has reported 0 live
tasks for demonstrably running work three separate times this session.

Volume listing is rate-limited; this backs off rather than failing, and prints what it
managed to read. A partial table is honest, a crash is not.
"""
from __future__ import annotations

import json
import os
import sys
import time

import modal

T4_TARGETS = ("fa7", "braf", "5ht1b", "jak2", "parp1")


def _retry(fn, tries=4, wait=8.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as error:
            if "rate limit" not in str(error).lower() or i == tries - 1:
                raise
            time.sleep(wait * (i + 1))
    return None


def t4(profile="rahul-94866"):
    os.environ["MODAL_PROFILE"] = profile
    print("=== T4 replicates 2+3 (ceiling 15,000 docking calls) ===")
    cells = locks = 0
    best: dict[str, float] = {}
    for target in T4_TARGETS:
        try:
            volume = modal.Volume.from_name(f"compose-t4-unified-controller-{target}-v1")
            runs = _retry(lambda: sorted(x.path for x in volume.listdir("/")))
        except Exception as error:
            print(f"  {target:6s} unreadable: {str(error)[:60]}")
            continue
        for run in runs or []:
            for entry in _retry(lambda: list(volume.listdir(run))) or []:
                name = entry.path.split("/")[-1]
                if name.endswith(".json"):
                    continue
                cells += 1
                try:
                    files = _retry(lambda: list(volume.listdir(entry.path))) or []
                except Exception:
                    continue
                locks += sum(1 for f in files if "round_" in f.path and "lock" in f.path)
                # the checkpoint archive is the cheapest live read of a cell's best
                if any(f.path.endswith("checkpoint.json") for f in files):
                    try:
                        buf = b"".join(volume.read_file(f"{entry.path}/checkpoint.json"))
                        archive = json.loads(buf)["payload"].get("archive") or {}
                        if archive:
                            best[name] = min(archive.values())
                    except Exception:
                        pass
    print(f"  cells started {cells} of 60 | round locks {locks} | cells with a docked archive {len(best)}")
    for name in sorted(best, key=lambda k: best[k])[:8]:
        print(f"     {name:14s} best so far {best[name]:+.1f}")
    print("  NOTE: checkpoints are NOT authoritative -- reconcile from round locks before reporting.")


def pmo(profile="nitya"):
    os.environ["MODAL_PROFILE"] = profile
    print("\n=== PMO campaigns ===")
    live = dead = 0
    for name in ("compose-pmo-fibercontrol", "compose-pmo-fibercontrol-mpo"):
        try:
            volume = modal.Volume.from_name(name)
            entries = _retry(lambda: [x.path for x in volume.listdir("/")]) or []
        except Exception as error:
            print(f"  {name}: unreadable: {str(error)[:60]}")
            continue
        for path in entries:
            if not path.startswith("scored_"):
                continue
            try:
                inner = {x.path.split("/")[-1] for x in (_retry(lambda: list(volume.listdir(path))) or [])}
            except Exception:
                continue
            if "provenance.json" in inner:
                try:
                    code = json.loads(b"".join(volume.read_file(f"{path}/provenance.json"))).get("returncode")
                except Exception:
                    code = None
                if code == 1:
                    dead += 1
                    continue
            live += 1
    print(f"  campaigns alive-or-clean {live} | exited returncode=1 {dead}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "t4"):
        t4()
    if which in ("all", "pmo"):
        pmo()
