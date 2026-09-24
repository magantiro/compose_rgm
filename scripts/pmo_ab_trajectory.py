"""Matched-arm trajectory reader for a PMO A/B, from charged ledgers only.

FOUR READOUTS, because no single one is fair on its own:

1. AUC-top10 at FIXED checkpoints (250/500/1000/2000) under the production mol_opt
   convention. This is the benchmark-aligned comparison.
2. Call of first improvement over the best initialization seed -- the arms are handed the
   same bank, so this isolates how fast each one gets off it.
3. Call of each breakthrough: first >= 0.9, first >= 0.999.
4. The best/top10 trajectory itself.

THE UNMATCHED TAIL IS PRESERVED, NOT DISCARDED. Arms progress at different rates (the
transplant lane is deliberately slower per proposal), so a comparison at differing n is
invalid -- but an arm that lands a 1.0 at call 280 while its sibling is only at 240 has
found something real. Events beyond the matched horizon are reported and labelled
PENDING (not yet causally comparable) rather than dropped; they become comparable the
moment the trailing arm passes that call.

Zero new oracle calls: initialization molecules are re-scored locally and were already
charged in every run.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

CHECKPOINTS = (250, 500, 1000, 2000, 5000)


def _tdc_oracle(name):
    stub = types.ModuleType("rdkit.six")
    stub.string_types = (str,)
    stub.iteritems = lambda d: iter(d.items())
    import rdkit

    sys.modules["rdkit.six"] = stub
    rdkit.six = stub
    from tdc import Oracle

    return Oracle(name=name)


def _arm_of(label: str) -> str:
    for tag in ("measured_score", "uniform", "nodonor"):
        if f"_{tag}_" in label:
            return tag
    return "donor" if "_donor_" in label else "unknown"


def _first(values, predicate):
    for i, v in enumerate(values, start=1):
        if predicate(v):
            return i
    return None


def read_volume(volume_name: str, init_dir: Path) -> dict:
    import modal

    from compose_v4.control.program_task import pmo_top_ten_auc

    volume = modal.Volume.from_name(volume_name)
    out: dict[str, dict] = {}
    for label in sorted(entry.path for entry in volume.listdir("/")):
        try:
            progress = json.loads(b"".join(volume.read_file(f"{label}/progress.json")).decode())
            raw = b"".join(volume.read_file(f"{label}/queries.jsonl")).decode()
        except (OSError, ValueError, KeyError) as error:
            # NEVER skip silently. A campaign that has written no ledger is either still
            # starting or already dead, and an arm that quietly vanishes from this report
            # is exactly how a dead donor arm read as running for three hours.
            print(f"  [no ledger yet] {label[:60]}: {type(error).__name__}")
            continue
        task = progress["task"]
        bank = json.loads((init_dir / f"{task}.json").read_text())
        oracle = _tdc_oracle(task)
        seeds = [float(oracle(row["endpoint"])) for row in bank["candidates"]]
        charged = [json.loads(line).get("score") for line in raw.strip().split("\n") if line]
        ordered = seeds + [s for s in charged if s is not None]
        best_seed = max(seeds)
        out.setdefault(task, {})[_arm_of(label)] = {
            "label": label,
            "n": len(ordered),
            "best_seed": best_seed,
            "best": max(ordered),
            "top10": sum(sorted(ordered)[-10:]) / 10,
            "first_beats_seed": _first(ordered, lambda v, s=best_seed: v > s + 1e-12),
            "first_at_0.9": _first(ordered, lambda v: v >= 0.9),
            "first_at_1.0": _first(ordered, lambda v: v >= 0.999),
            "auc": {
                c: (pmo_top_ten_auc(ordered[:c], budget=c, frequency=100)
                    if len(ordered) >= c else None)
                for c in CHECKPOINTS
            },
            "ordered": ordered,
        }
    return out


def report(data: dict) -> None:
    from compose_v4.control.program_task import pmo_top_ten_auc

    for task, arms in sorted(data.items()):
        matched = min(a["n"] for a in arms.values())
        print(f"\n=== {task}   matched horizon n={matched} ===")
        for arm, a in sorted(arms.items()):
            cut = a["ordered"][:matched]
            auc_m = pmo_top_ten_auc(cut, budget=matched, frequency=100) if matched >= 100 else None
            cells = " ".join(
                f"{c}:{a['auc'][c]:.4f}" for c in CHECKPOINTS if a["auc"][c] is not None
            ) or "(<250)"
            print(f"  {arm:15s} own_n={a['n']:5d}  at matched n: "
                  f"auc={auc_m if auc_m is None else round(auc_m, 4)} "
                  f"best={max(cut):.4f} top10={sum(sorted(cut)[-10:]) / 10:.4f}")
            print(f"  {'':15s} fixed checkpoints: {cells}")
            for name in ("first_beats_seed", "first_at_0.9", "first_at_1.0"):
                call = a[name]
                if call is None:
                    continue
                tag = "" if call <= matched else "   <-- PENDING, beyond matched horizon"
                print(f"  {'':15s} {name:18s} call {call}{tag}")


def main() -> int:
    volume = sys.argv[1]
    init_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "diagnostics/pmo_prescreen_v1/initialization")
    data = read_volume(volume, init_dir)
    if not data:
        print("no campaign has written both progress.json and queries.jsonl yet")
        return 0
    report(data)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, "src")
    raise SystemExit(main())
