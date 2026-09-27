import modal, json, statistics as st, sys, time
SUFFIX=sys.argv[1]
rows=[]
for p in ("braf","fa7","5ht1b","jak2","parp1"):
    try: v=modal.Volume.from_name(f"compose-t4-unified-controller-{p}-{SUFFIX}")
    except Exception: continue
    try: paths=[e.path for e in v.listdir("/", recursive=True)]
    except Exception: continue
    seen={}
    for path in paths:
        q=path.split("/")
        if len(q)>=3 and q[-1] in ("checkpoint.json","result.json"):
            seen.setdefault((q[0],q[1]),set()).add(q[-1])
    for (run,cell),files in sorted(seen.items()):
        if "result.json" in files:
            rows.append((0.0, f"{p}/{cell}", 250, None, None, True)); continue
        time.sleep(0.2)
        try: d=json.loads(b"".join(v.read_file(f"{run}/{cell}/checkpoint.json")))["payload"]
        except Exception: continue
        rr=d.get("rounds") or []
        ts=[x["timing_seconds"]["round_total"] for x in rr
            if isinstance(x.get("timing_seconds"),dict) and "round_total" in x["timing_seconds"]]
        ch=d.get("charged_calls") or 0
        if not ts: rows.append((None,f"{p}/{cell}",ch,None,len(rr),False)); continue
        med=st.median(ts[-6:] if len(ts)>=6 else ts)
        eta=(max(0,250-ch)/8.0)*med/3600
        rows.append((eta,f"{p}/{cell}",ch,med,len(rr),False))
json.dump([[r[0],r[1],r[2],r[3],r[4],r[5]] for r in rows], open(f"/tmp/t4_eta_{SUFFIX}.json","w"))
live=[r for r in rows if r[0] is not None and not r[5]]
meds=[r[3] for r in live if r[3]]
print(f"{SUFFIX}: {len(rows)} cells, {sum(1 for r in rows if r[5])} done, {len(live)} with timing")
if meds:
    print(f"  seconds/round: median {st.median(meds):.0f}  p10 {sorted(meds)[len(meds)//10]:.0f}  p90 {sorted(meds)[9*len(meds)//10]:.0f}")
    print(f"  => per ACTIVE cell ~{8*3600/st.median(meds):.1f} calls/h")
    e=[r[0] for r in live]
    print(f"  remaining per cell (h): median {st.median(e):.1f}  p90 {sorted(e)[9*len(e)//10]:.1f}  max {max(e):.1f}")
    for r in sorted(live, reverse=True)[:8]:
        print(f"     {r[0]:6.1f} h  {r[1]:18s} {r[2]:3d}/250  {r[3]:6.0f}s/round  {r[4]}r")
