"""One-shot frozen analysis of the DDSBM ZINC logP 2->4 endpoint run.

Metric definitions are taken VERBATIM from
`docs/DDSBM_ENDPOINT_COMPETENCE_PROTOCOL.md`, which was frozen before any
outcome existed:

  * logP W1  -- Wasserstein-1 between the GENERATED and TARGET marginals
  * QED MAD  -- preservation RELATIVE TO EACH SOURCE, i.e. mean |QED(y)-QED(x0)|
  * SA  MAD  -- likewise for synthetic accessibility
  * validity, uniqueness, novelty

Nothing is added after seeing a number, and no metric is dropped.

ON THE TARGET MARGINAL. The protocol BARS using the CSV's randomly paired
PRB-SMI as PER-SOURCE GOAL INFORMATION -- that would inject information the
transport problem does not contain. Using the AGGREGATE PRB-SMI distribution as
the evaluation target is a different thing and is required: W1 is defined
between marginals, so the target marginal has to come from somewhere. The
controller never saw it.

NSPDK and FCD are NOT computed here, and are NOT dropped. The protocol
anticipates exactly this: "Generate and persist the 5,984 endpoints once;
compute RDKit metrics immediately; add NSPDK and FCD from the same saved
endpoints once their packages are qualified. Never re-run COMPOSE because a
metric package was missing." They are reported as PENDING.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / ("local_runtime/ddsbm/DDSBM-main/data/raw/"
              "ZINC250k_logp_2_4_random_matched_no_nH.csv")
TRAIN_ROWS = 23936

#: Reported by DDSBM on this benchmark. Cited, never recomputed.
DDSBM_REPORTED = {"W1": 0.139, "QED_MAD": 0.120, "SA_MAD": 0.402,
                  "NSPDK": 7.30e-4, "FCD": 0.833}


def _sa_scorer():
    """RDKit's contrib SA scorer, if the installation exposes it."""
    try:
        import sys

        from rdkit.Chem import RDConfig
        sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
        import sascorer  # type: ignore
        return sascorer.calculateScore
    except Exception:  # noqa: BLE001
        return None


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """W1 between two empirical 1-D samples, by quantile coupling."""
    n = max(len(a), len(b))
    q = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def main() -> int:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, Crippen

    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("/tmp/ddsbm_full.json.gz"))
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/DDSBM_FROZEN_RESULT.json")
    args = ap.parse_args()

    payload = json.loads(gzip.decompress(args.run.read_bytes()).decode())
    results = payload["results"]
    assert payload["n"] == len(results) == 5984, "expected the full test split"

    rows = list(csv.DictReader(open(CSV)))
    test = rows[TRAIN_ROWS:]
    assert len(test) == 5984
    train_ref = {r["REF-SMI"] for r in rows[:TRAIN_ROWS]}

    sa_fn = _sa_scorer()
    ok = [r for r in results if r.get("status") == "OK"]

    endpoints, sources, traj_valid, edits = [], [], [], []
    for r in ok:
        endpoints.append(r["endpoint"])
        sources.append(r["source"])
        traj = r.get("trajectory") or []
        traj_valid.append(all(Chem.MolFromSmiles(s) is not None for s in traj))
        edits.append(r.get("edits", 0))

    mols_e = [Chem.MolFromSmiles(s) for s in endpoints]
    mols_s = [Chem.MolFromSmiles(s) for s in sources]
    valid_mask = [m is not None for m in mols_e]
    validity = float(np.mean(valid_mask))

    canon = [Chem.MolToSmiles(m) for m, v in zip(mols_e, valid_mask) if v]
    uniqueness = len(set(canon)) / max(1, len(canon))
    novelty = float(np.mean([c not in train_ref for c in set(canon)])) if canon else 0.0

    gen_logp = np.array([Crippen.MolLogP(m) for m, v in zip(mols_e, valid_mask) if v])
    tgt_mols = [Chem.MolFromSmiles(r["PRB-SMI"]) for r in test]
    tgt_logp = np.array([Crippen.MolLogP(m) for m in tgt_mols if m is not None])
    src_logp = np.array([Crippen.MolLogP(m) for m in mols_s if m is not None])

    w1 = wasserstein1(gen_logp, tgt_logp)

    qed_mad, sa_mad = [], []
    for me, ms, v in zip(mols_e, mols_s, valid_mask):
        if not v or ms is None:
            continue
        qed_mad.append(abs(QED.qed(me) - QED.qed(ms)))
        if sa_fn is not None:
            sa_mad.append(abs(sa_fn(me) - sa_fn(ms)))

    rng = np.random.default_rng(0)

    def boot(x, f=np.mean, draws=2000):
        x = np.asarray(x, float)
        idx = rng.integers(0, len(x), (draws, len(x)))
        s = f(x[idx], axis=1)
        return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

    out = {
        "schema": "compose.ddsbm.frozen_analysis",
        "protocol": "docs/DDSBM_ENDPOINT_COMPETENCE_PROTOCOL.md",
        "n_sources": len(results),
        "n_ok": len(ok),
        "compose": {
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            "logP_W1": w1,
            "QED_MAD": float(np.mean(qed_mad)),
            "QED_MAD_ci95": boot(qed_mad),
            "SA_MAD": float(np.mean(sa_mad)) if sa_mad else None,
            "SA_MAD_ci95": boot(sa_mad) if sa_mad else None,
            "trajectory_wide_validity": float(np.mean(traj_valid)),
            "mean_edits": float(np.mean(edits)),
        },
        "distributions": {
            "generated_logP_mean": float(gen_logp.mean()),
            "generated_logP_sd": float(gen_logp.std()),
            "target_logP_mean": float(tgt_logp.mean()),
            "target_logP_sd": float(tgt_logp.std()),
            "source_logP_mean": float(src_logp.mean()),
            "source_logP_sd": float(src_logp.std()),
        },
        "ddsbm_reported": DDSBM_REPORTED,
        "pending_not_dropped": {
            "NSPDK": "package not qualified; endpoints persisted, computable later",
            "FCD": "package not qualified; endpoints persisted, computable later",
            "authority": ("protocol says add these from the SAME saved endpoints "
                          "once packages are qualified, and never re-run COMPOSE"),
        },
        "sa_scorer_available": sa_fn is not None,
    }
    args.out.write_text(json.dumps(out, indent=2))

    c = out["compose"]
    print(f"n = {len(ok)}/{len(results)} OK\n")
    print(f"{'metric':<26}{'COMPOSE':>12}{'DDSBM (reported)':>20}")
    print(f"{'logP W1':<26}{w1:>12.4f}{DDSBM_REPORTED['W1']:>20.4f}")
    print(f"{'QED MAD':<26}{c['QED_MAD']:>12.4f}{DDSBM_REPORTED['QED_MAD']:>20.4f}")
    if c["SA_MAD"] is not None:
        print(f"{'SA MAD':<26}{c['SA_MAD']:>12.4f}{DDSBM_REPORTED['SA_MAD']:>20.4f}")
    print(f"{'validity':<26}{validity:>12.4f}{'—':>20}")
    print(f"{'uniqueness':<26}{uniqueness:>12.4f}{'—':>20}")
    print(f"{'novelty':<26}{novelty:>12.4f}{'—':>20}")
    print(f"{'trajectory-wide validity':<26}{c['trajectory_wide_validity']:>12.4f}"
          f"{'not reportable':>20}")
    d = out["distributions"]
    print(f"\nlogP  source {d['source_logP_mean']:.3f}±{d['source_logP_sd']:.3f}"
          f"  ->  generated {d['generated_logP_mean']:.3f}±{d['generated_logP_sd']:.3f}"
          f"   target {d['target_logP_mean']:.3f}±{d['target_logP_sd']:.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
