import json, sys
from pathlib import Path
import modal
ROOT = Path("/Users/rmaganti/compose_v2_work")
r = json.loads((ROOT / "diagnostics/exact_route_parp1_s0.json").read_text())
fs = r["forward_states"]
picks = fs[::3] + ([fs[-1]] if fs[-1] not in fs[::3] else [])
f = modal.Function.from_name("genmol-t4-gate4", "dock_frontier")
res = f.remote({"smiles": picks, "target": "parp1", "idx": 0, "delta": 0.4,
                "beta": -1.0, "budget": len(picks), "tag": "witness"})
top = dict(res.get("top") or [])
print(f"  docked {res['n_docked']}/{res['n_requested']} route states\n")
print(f"  {'depth':>6s} {'ds':>7s}  smiles")
for i, s in enumerate(picks):
    d = top.get(s)
    print(f"  {i*3:>6d} {str(d):>7s}  {s[:58]}")
json.dump({"depths": [i*3 for i in range(len(picks))], "smiles": picks,
           "ds": [top.get(s) for s in picks]},
          open(ROOT / "diagnostics/parp1_witness_docking.json", "w"), indent=2)
