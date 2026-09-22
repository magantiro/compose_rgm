#!/usr/bin/env python3
"""Reconcile the T4 panel from round locks across EVERY run on each campaign volume.

The existing table builder pins the run whose launch matches the live contract
payload.  Work migrates to a new run_id on the same volume after a resume or a
re-seal, so that rule silently reports an empty run as the cell's result --
measured: braf-d06 holds one run with 5 round locks and another with 27.

A round lock is immutable evidence a molecule was scored, so the reconciliation
reads locks from ALL runs and takes the best eligible endpoint the cell ever
produced, recording which run supplied it.
"""
from __future__ import annotations

import json
import sys
import time

import modal

CAMPAIGNS = {
    ("parp1", 0.6): "compose-t4-held-target-distilled-parp1-d06-250",
    ("braf", 0.6): "compose-t4-held-target-distilled-braf-d06-250",
    ("5ht1b", 0.6): "compose-t4-held-target-distilled-5ht1b-d06-250",
    ("fa7", 0.6): "compose-t4-held-target-distilled-fa7-d06-250",
    ("jak2", 0.6): "compose-t4-held-target-jak2-true-d06-250",
    ("parp1", 0.4): "compose-t4-held-target-distilled-parp1-d04-250",
    ("braf", 0.4): "compose-t4-held-target-distilled-braf-d04-250",
    ("5ht1b", 0.4): "compose-t4-held-target-distilled-5ht1b-d04-250",
    ("fa7", 0.4): "compose-t4-held-target-distilled-fa7-d04-250",
    ("jak2", 0.4): "compose-t4-held-target-distilled-jak2-d06-250",
}
RESCUES = {
    ("5ht1b", 0.4): "compose-t4-5ht1b2-protonation-rescue-d04",
    ("5ht1b", 0.6): "compose-t4-5ht1b2-protonation-rescue-d06",
    ("fa7", 0.4): "compose-t4-region-repair-rescue-fa7-d04",
    ("fa7", 0.6): "compose-t4-region-repair-rescue-fa7-d06",
    ("braf", 0.6): "compose-t4-region-repair-rescue-braf-d06",
}


def retry(fn, *a, **k):
    """Volume listing is rate limited; back off rather than lose the sweep."""
    for attempt in range(8):
        try:
            return fn(*a, **k)
        except modal.exception.ResourceExhaustedError:
            time.sleep(2 * (attempt + 1))
        except Exception:
            return None
    return None


def entries(volume, path=""):
    got = retry(volume.listdir, path, recursive=False)
    return [] if got is None else got


def read(volume, path):
    got = retry(lambda: b"".join(volume.read_file(path)))
    if got is None:
        return None
    try:
        return json.loads(got)
    except Exception:
        return None


def cell_best(volume, run, cell):
    """Best eligible docking and max charged count over this cell's round locks."""
    best, calls, rounds = None, 0, 0
    for entry in entries(volume, f"{run}/{cell}"):
        name = entry.path.split("/")[-1]
        if not name.endswith("_lock.json") or "round_" not in name:
            continue
        rounds += 1
        payload = read(volume, entry.path)
        if not isinstance(payload, dict):
            continue
        body = payload.get("payload", payload)
        before = body.get("charged_before")
        queries = body.get("queries") or body.get("selected") or []
        n = before + len(queries) if isinstance(before, int) else None
        if isinstance(n, int):
            calls = max(calls, n)
        for q in queries if isinstance(queries, list) else []:
            if not isinstance(q, dict):
                continue
            for key in ("docking", "score", "best_eligible_docking", "value"):
                v = q.get(key)
                if isinstance(v, (int, float)) and (best is None or v < best):
                    best = v
    return best, calls, rounds


def main() -> None:
    out = {}
    for (target, delta), vol_name in sorted(CAMPAIGNS.items()):
        try:
            volume = modal.Volume.from_name(vol_name)
        except Exception as exc:
            print(f"{vol_name}: {type(exc).__name__}", flush=True)
            continue
        runs = [e.path.strip("/") for e in entries(volume) if "/" not in e.path.strip("/")]
        runs = [r for r in runs if not r.endswith(".json")]
        for run in runs:
            for e in entries(volume, run):
                cell = e.path.split("/")[-1]
                if cell.endswith(".json"):
                    continue
                best, calls, rounds = cell_best(volume, run, cell)
                key = f"{target}|{delta}|{cell}"
                prior = out.get(key)
                better = prior is None or (
                    best is not None and (prior["best"] is None or best < prior["best"])
                )
                if better:
                    out[key] = {"target": target, "delta": delta, "cell": cell,
                                "best": best, "charged": calls, "rounds": rounds,
                                "run": run, "volume": vol_name, "phase": "campaign"}
                print(f"  {vol_name.split('distilled-')[-1]:22s} {run[:10]} {cell:9s} "
                      f"best={best} calls={calls} rounds={rounds}", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
    print(f"\nwrote {sys.argv[1]}  ({len(out)} cells)", flush=True)


if __name__ == "__main__":
    main()
