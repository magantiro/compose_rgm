"""Fit the committor from the collection units persisted on the volume.

Run separately from the collection so the fit does not depend on the
collection client surviving, and so straggler units that are past their
timeout and retrying cannot hold up a gate that already passes.
"""
import json
from pathlib import Path

from modal_apps.committor_bellman_app import app, collect_summary, fit_from_volume


@app.local_entrypoint()
def main(max_b: int = 6, epochs: int = 400):
    rows = collect_summary.remote()
    pos = [r for r in rows if r["role"] == "positive"]
    pos_mols = {r["region_smiles"] for r in pos if (r["n_terminal_states"] or 0) > 0}
    n_term = sum(r["n_terminal_states"] or 0 for r in rows)
    n_neg = sum(1 for r in rows if r["role"] != "positive")
    assert len(pos_mols) >= 3 and n_term >= 8 and n_neg >= len(pos_mols), \
        f"gate: mols={len(pos_mols)} term={n_term} neg={n_neg}"
    print(f"GATE PASSED units={len(rows)} pos_molecules={len(pos_mols)} "
          f"terminals={n_term} negatives={n_neg}")
    res = fit_from_volume.remote(max_b=max_b, epochs=epochs)
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/committor_fit.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:4000])
