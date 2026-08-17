"""Bitwise qualification for the cycle-close / atom-restate admission masks.

WHAT IS BEING CHANGED, AND WHY THIS IS NOT A SEMANTICS CHANGE
-------------------------------------------------------------
Profiling put 90% of a law call in two admission-mask computations, and inside
`enumerate_cycle_close_edges` the cost is not the legality decision -- it is
recomputing state-invariant facts once per candidate:

    is_valid_state(SOURCE)          984 calls/state    313.6 ms   19.9%
    canonical_state_key(SOURCE)     984 calls/state    200.1 ms   12.7%
    _exact_state_identity(SOURCE)   900 calls/state    146.0 ms    9.3%
    is_connected_or_null(SOURCE)    984 calls/state     71.7 ms    4.6%
                                                       ------
                                    pure redundancy    731.5 ms   46.4%

plus 960 Kekule alias instantiations per state drawn from only 2.4 DISTINCT
selections, so the alias validity / connectivity / canonical-key triple is
recomputed roughly 400x per distinct alias.

Removing that is memoization of pure functions. The legality predicate is
untouched: no candidate is filtered by a heuristic, no admission rule is
rewritten, no capability flag is changed. A hoist can only be wrong by returning
a DIFFERENT answer, which is exactly what this script is here to detect.

WHY THE BASELINE IS BANKED TO DISK RATHER THAN COMPARED IN-PROCESS
------------------------------------------------------------------
In-process comparison was required for the graph-only encoder because embeddings
are floats and two containers can differ in the last bits. These outputs are
boolean masks and integer graphs -- exactly reproducible -- so a banked baseline
is sound and lets the optimisation be developed against a fixed oracle.

The corpus is deliberately broader than the SMC trajectory distribution: the 64
dev-panel sources plus a ZINC250k sample. A pure chemistry predicate should be
qualified on chemical diversity, not on the states one controller happened to
visit.

    python3 scripts/verification/admission_mask_parity.py bank  --n 260
    python3 scripts/verification/admission_mask_parity.py check --n 260

`bank` writes the oracle; `check` recomputes and requires BITWISE equality of
both masks and of the ordered enumerator output. Any single differing bit fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

BASELINE = REPO / "docs" / "ADMISSION_MASK_BASELINE.json"
ZINC = REPO / "local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv"
PANEL = REPO / "data/jin/dev_panel_qed_64.txt"
SLOTS = 48


def corpus(n: int) -> list[str]:
    """Dev panel first, then a deterministic ZINC stride. No RNG, no seed.

    ZINC250k quotes each SMILES and embeds a newline inside the quotes, so a
    naive line split yields an empty string for every second record. Those
    empties parse to a degenerate zero-atom state rather than failing, so they
    are silently admitted to the corpus and collapse to ONE dict key -- which is
    what shrank a nominal 260-state corpus to 163. Parsed with `csv` instead.
    """
    import csv

    out = [s.strip() for s in PANEL.read_text().split("\n") if s.strip()]
    if ZINC.exists() and len(out) < n:
        with ZINC.open(newline="") as fh:
            smis = [row["smiles"].strip() for row in csv.DictReader(fh)
                    if row.get("smiles", "").strip()]
        want = n - len(out)
        stride = max(1, len(smis) // max(1, want))
        out += smis[::stride][:want]
    seen, unique = set(), []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:n]


def fingerprint(smi: str) -> dict | None:
    """Both admission masks plus the ordered enumerator output, hashed."""
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.model.factorized_tracelet_rate_model import (
        _semantic_atom_restate_admission_mask,
        _semantic_cycle_close_admission_mask,
    )
    from compose_v4.rewrite.operators import (
        enumerate_cycle_close_edges,
        enumerate_semantic_atom_restates,
    )

    try:
        mg = pad_molecular_graph(smiles_to_molecular_graph(smi), SLOTS)
    except Exception:
        return None

    t0 = time.perf_counter()
    close_mask = _semantic_cycle_close_admission_mask(mg)
    t_close = time.perf_counter() - t0
    t0 = time.perf_counter()
    restate_mask = _semantic_atom_restate_admission_mask(mg)
    t_restate = time.perf_counter() - t0

    # The ORDERED action tuples, not just the masks. A mask collapses ordering
    # and multiplicity; the law iterates the enumeration, so order is part of
    # the contract being preserved.
    close_actions = [(int(a.a), int(a.b), int(a.order))
                     for a in enumerate_cycle_close_edges(mg)]
    restate_actions = [(int(a.v), int(a.target_class_index))
                       for a in enumerate_semantic_atom_restates(mg)]

    def h(x) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(x).tobytes() if isinstance(x, np.ndarray)
            else json.dumps(x).encode()).hexdigest()

    return {
        "close_mask_sha256": h(close_mask),
        "restate_mask_sha256": h(restate_mask),
        "close_actions_sha256": h(close_actions),
        "restate_actions_sha256": h(restate_actions),
        "n_close": len(close_actions),
        "n_restate": len(restate_actions),
        "seconds_close": t_close,
        "seconds_restate": t_restate,
    }


def run(n: int) -> dict:
    smis = corpus(n)
    rows, skipped = {}, 0
    tc = tr = 0.0
    for i, smi in enumerate(smis, 1):
        fp = fingerprint(smi)
        if fp is None:
            skipped += 1
            continue
        tc += fp.pop("seconds_close")
        tr += fp.pop("seconds_restate")
        rows[smi] = fp
        if i % 25 == 0:
            print(f"  {i}/{len(smis)}  cycle-close {tc/len(rows)*1e3:7.1f} ms/state"
                  f"  atom-restate {tr/len(rows)*1e3:7.1f} ms/state", flush=True)
    return {"n_states": len(rows), "skipped": skipped,
            "mean_ms_close": tc / max(len(rows), 1) * 1e3,
            "mean_ms_restate": tr / max(len(rows), 1) * 1e3,
            "states": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bank", "check"])
    ap.add_argument("--n", type=int, default=260)
    a = ap.parse_args()

    print(f"{a.mode}: {a.n} states\n")
    got = run(a.n)
    print(f"\n{got['n_states']} states  ({got['skipped']} unparseable)")
    print(f"  cycle-close   {got['mean_ms_close']:8.1f} ms/state")
    print(f"  atom-restate  {got['mean_ms_restate']:8.1f} ms/state")

    if a.mode == "bank":
        BASELINE.write_text(json.dumps(got, indent=1))
        print(f"\nbanked -> {BASELINE.relative_to(REPO)}")
        return

    base = json.loads(BASELINE.read_text())
    keys = ("close_mask_sha256", "restate_mask_sha256",
            "close_actions_sha256", "restate_actions_sha256")
    shared = set(base["states"]) & set(got["states"])
    missing = set(base["states"]) - set(got["states"])
    bad = []
    for smi in sorted(shared):
        for k in keys:
            if base["states"][smi][k] != got["states"][smi][k]:
                bad.append((smi, k))
    print(f"\ncompared {len(shared)} states x {len(keys)} exact digests")
    if missing:
        print(f"  WARNING: {len(missing)} banked states absent from this run")
    if bad:
        print(f"\n  {len(bad)} MISMATCHES:")
        for smi, k in bad[:10]:
            print(f"    {k:<26} {smi[:56]}")
        print("\nBITWISE PARITY: FAILED")
        raise SystemExit(1)

    sc = base["mean_ms_close"] / max(got["mean_ms_close"], 1e-9)
    sr = base["mean_ms_restate"] / max(got["mean_ms_restate"], 1e-9)
    print(f"\n  cycle-close   {base['mean_ms_close']:8.1f} -> "
          f"{got['mean_ms_close']:8.1f} ms   {sc:5.2f}x")
    print(f"  atom-restate  {base['mean_ms_restate']:8.1f} -> "
          f"{got['mean_ms_restate']:8.1f} ms   {sr:5.2f}x")
    print("\nBITWISE PARITY: PASSED"
          f"  ({len(shared)} states, {len(shared)*len(keys)} digests identical)")


if __name__ == "__main__":
    main()
