# broad-C runs r20/r21/r22 — EXPLORATORY, INVALID FOR SEMANTIC CREDIT

Cell 5ht1b_s7_d0.6, launched 2026-08-27 01:37, stopped mid-run deliberately.

## Why invalid

The semantic credit path in `drive_episodes` credited every ring request that a
particle SAMPLED with that particle's endpoint docking score, without checking
whether the request actually executed:

    for _q in _rr:                      # _rr = SAMPLED requests
        _rq_all.append(_q); _rq_sc.append(docked_smi[_sm2])

So a request that returned UNSAT still received full credit whenever the
particle's endpoint happened to dock. Observed directly in r21: a particle whose
trace was

    ring:fused/4/C2S2/saturated:UNSAT@closure
    ring:linked/5/C4P1/aromatic:UNSAT@aromatise
    ring:fused/6/C6/aromatic:UNSAT@closure

had every ring macro fail, yet all three tuples were eligible for credit.

## What these runs may and may not be used for

MAY be used for: ring-macro EXECUTION statistics (sampled -> ok/UNSAT), which
are read from traces and unaffected by the credit bug. Those showed fused
executing at 67-93%, i.e. fused realizes fine.

MAY NOT be used for: any claim about learned topology, semantic mass, or
"the controller discovered linked/fused". P(fused) reached 0.959 in r21 while
its docked set contained zero fused pairs; that mass movement is contaminated.

## Observed anyway (kept for the record, not for claims)

    r20  P(linked) 0.769 -> 0.938   best -9.50 -> -10.60   docked mean heavy 19.4
    r21  P(fused)  0.769 -> 0.959   best -8.40 FROZEN      docked mean heavy 16.8
    r22  P(linked) 0.769 -> 0.894   best -8.30 -> -10.30

A second, separate observation NOT caused by the bug: fused rings were built
successfully but did not survive to the docked endpoint (zero fused pairs in any
docked set). That remains unexplained and is worth revisiting.

Artifacts: volume compose-v4-artifacts, macro_basin/episodes_5ht1b_s7_d0.6_pooled_broad_r{20,21,22}.json
