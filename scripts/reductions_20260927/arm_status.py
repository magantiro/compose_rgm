"""Correct arm status. rc=0 COMPLETE, rc=1 FAILED, rc=None RUNNING."""
import json, sys, time
import modal
prof_ns = sys.argv[1]
v = modal.Volume.from_name("compose-pmo-fibercontrol")
ns = sorted(n for n in {e.path.split("/")[0] for e in v.listdir("/")} if n.startswith(prof_ns))
done=fail=run=0; rows=[]
for n in ns:
    time.sleep(0.3)
    ch=best=rc=None; err=""
    try:
        p=json.loads(b"".join(v.read_file(f"{n}/progress.json")))
        ch=p.get("charged_oracle_calls"); best=p.get("best_score")
    except Exception: pass
    try: rc=json.loads(b"".join(v.read_file(f"{n}/provenance.json"))).get("returncode")
    except Exception: pass
    state = "RUNNING" if rc is None else ("COMPLETE" if rc==0 else "FAILED")
    if rc is None: run+=1
    elif rc==0: done+=1
    else:
        fail+=1
        try:
            t=b"".join(v.read_file(f"{n}/stderr.log")).decode(errors="replace").strip().splitlines()
            err=[l for l in t if l and not l.startswith(" ")][-1][:90]
        except Exception: err="(no stderr)"
    rows.append((state, ch or 0, best, n.replace(prof_ns,"").rsplit("_",1)[0].lstrip("_"), err))
print(f"{prof_ns}*  COMPLETE {done} | RUNNING {run} | FAILED {fail}   (of {len(ns)})\n")
for st,ch,b,tag,err in sorted(rows, key=lambda r:(r[0],-r[1])):
    bs = f"{b:.4f}" if isinstance(b,float) else str(b)
    print(f"  {st:8s} {ch:5d} calls  best={bs:>8s}  {tag}" + (f"\n      {err}" if err else ""))
