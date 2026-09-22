"""Celecoxib macro-horizon diagnostic: are macro BOUNDARIES monotone?

Answer-known OFFLINE diagnostic.  The declared celecoxib target is used to CONSTRUCT
routes for analysis only; nothing here selects a runtime proposal.

Every score is the registered PyTDC oracle, constructed exactly as TDC does.
"""
from __future__ import annotations
import json, sys, time, signal
sys.path.insert(0, "/Users/rmaganti/compose_pmo_macro_data")
from macro_dp import describe

from compose_v4.experiments.winner_paths import find_path, PathConfig
from compose_v4.rewrite.trace_shard import decode_state
from compose_v4.rewrite.kernel import canonical_state_key
from tdc.chem_utils.oracle.oracle import rediscovery_meta

TARGET = 'CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'
oracle = rediscovery_meta(TARGET, fp='ECFP4')
OUT = "/Users/rmaganti/compose_pmo_macro_data/macro_horizon_v1.json"

class Timeout(Exception): pass
def _alarm(sig, frm): raise Timeout()
signal.signal(signal.SIGALRM, _alarm)

def sources():
    rows = []
    bank = json.load(open("diagnostics/parent_edit_cycles/prepared/init_20260921.json"))["candidates"]
    for i, c in enumerate(bank):
        rows.append(("init_%02d" % i, c["endpoint"]))
    art = json.load(open("/Users/rmaganti/compose_pmo_macro_data/B_memory_celecoxib_result_at_250_calls.json"))
    snap = art["campaign"]["snapshot"]
    scored = sorted(
        {o["endpoint"]: o["score"] for o in snap["observations"].values()}.items(),
        key=lambda kv: -kv[1])
    for rank, (smi, sc) in enumerate(scored[:8]):
        rows.append(("blindtop_%d_%.4f" % (rank, sc), smi))
    for rank, (smi, sc) in enumerate(scored[len(scored)//2:len(scored)//2+4]):
        rows.append(("blindmid_%d_%.4f" % (rank, sc), smi))
    return rows

def main():
    cfg = PathConfig(max_expansions=512, children_per_expansion=6,
                     mapping_timeout_seconds=2, matches_per_molecule=6, max_steps=64)
    out = {"schema_version": "pmo_macro_horizon_v1",
           "evidence_role": "answer_known_offline_diagnostic",
           "target": TARGET, "oracle": "tdc rediscovery_meta ECFP4 (count, unhashed)",
           "path_config": cfg.__dict__, "routes": []}
    for name, smi in sources():
        rec = {"source_name": name, "source_smiles": smi}
        try:
            rec["source_score"] = oracle(smi)
        except Exception as e:
            rec["error"] = f"source unscorable: {e}"; out["routes"].append(rec); continue
        t = time.time()
        signal.alarm(240)
        try:
            r = find_path(smi, TARGET, cfg)
            rec["status"] = r.get("status")
        except Timeout:
            rec["status"] = "wall_timeout_240s"; r = {}
        except Exception as e:
            rec["status"] = f"error:{type(e).__name__}"; rec["error"] = str(e); r = {}
        finally:
            signal.alarm(0)
        rec["seconds"] = round(time.time() - t, 1)
        rec["primitive_lower_bound"] = r.get("primitive_lower_bound")
        if rec["status"] == "witness_found":
            smis = [canonical_state_key(decode_state(s)) for s in r["states"]]
            scores = [oracle(x) for x in smis]
            rec["n_steps"] = len(r["actions"])
            rec["trajectory_smiles"] = smis
            rec["trajectory_scores"] = scores
            rec["analysis"] = describe(scores, max_len=23)
        out["routes"].append(rec)
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"{name:26s} {rec['status']:20s} {rec['seconds']:6.1f}s "
              f"steps={rec.get('n_steps','-')} ", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT, flush=True)

if __name__ == "__main__":
    main()
