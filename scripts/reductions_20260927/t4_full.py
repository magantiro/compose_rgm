import json, re, sys, time
import modal
SUFFIX, LABEL = sys.argv[1], sys.argv[2]
rows=[]
for p in ("braf","fa7","5ht1b","jak2","parp1"):
    name=f"compose-t4-unified-controller-{p}-{SUFFIX}"
    try: v=modal.Volume.from_name(name)
    except Exception: continue
    try: paths=[e.path for e in v.listdir("/", recursive=True)]
    except Exception: continue
    cells={}
    for path in paths:
        q=path.split("/")
        if len(q)<3 or q[1].endswith(".json"): continue
        cells.setdefault((q[0],q[1]), set()).add(q[-1])
    for (run,cell),files in sorted(cells.items()):
        time.sleep(0.2)
        rec={"protein":p,"run":run[:8],"cell":cell,"arm":LABEL,
             "rounds":sum(1 for f in files if f.startswith("round_") and f.endswith("lock.json"))}
        src="result.json" if "result.json" in files else ("checkpoint.json" if "checkpoint.json" in files else None)
        if src:
            try:
                d=json.loads(b"".join(v.read_file(f"{run}/{cell}/{src}")))["payload"]
                rec["charged"]=d.get("charged_calls"); rec["status"]=d.get("status","running")
                rec["done"]= src=="result.json"
                best=d.get("final_best")
                if best is None:
                    rr=d.get("rounds") or []
                    cands=[x.get("best_so_far") for x in rr if x.get("best_so_far") is not None]
                    best=min(cands) if cands else None
                rec["best"]=best
                rec["smiles"]=d.get("best_smiles")
                cb=d.get("claim_boundary") or ""
                m=re.search(r"delta (\d\.\d)", cb); rec["delta"]=float(m.group(1)) if m else None
            except Exception as e: rec["status"]=f"unreadable:{type(e).__name__}"; rec["done"]=False
        else: rec["status"]="no checkpoint"; rec["done"]=False
        rows.append(rec)
print(json.dumps(rows))
