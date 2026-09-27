import json, re, sys, time
import modal
pref=sys.argv[1]
v=modal.Volume.from_name("compose-pmo-fibercontrol")
ns=sorted(n for n in {e.path.split("/")[0] for e in v.listdir("/")} if n.startswith(pref))
out=[]
for n in ns:
    time.sleep(0.25)
    m=re.search(r"_(.+?)_seed(\d+)_", n.replace(pref,"_"))
    task, seed = (m.group(1), int(m.group(2))) if m else (n, 0)
    rec={"task":task,"seed":seed,"namespace":n}
    try:
        c=json.loads(b"".join(v.read_file(f"{n}/canary_v1.json")))
        rec.update({"best":c.get("best_score"),"top10":c.get("final_top10"),
                    "auc":c.get("auc_official"),"state":"COMPLETE","seconds":c.get("seconds")})
    except Exception:
        try:
            tr=[json.loads(l) for l in b"".join(v.read_file(f"{n}/trajectory.jsonl")).decode().strip().splitlines()]
            last=tr[-1]
            rec.update({"best":last.get("best"),"top10":last.get("top10"),"auc":None,
                        "state":f"PARTIAL@{last.get('charged')}"})
        except Exception:
            rec.update({"best":None,"top10":None,"auc":None,"state":"NO DATA"})
    try:
        if json.loads(b"".join(v.read_file(f"{n}/provenance.json"))).get("returncode") not in (0,None):
            rec["state"]="FAILED"
    except Exception: pass
    out.append(rec)
print(json.dumps(out))
