# INVALID — arms bound by different resources

R_M      trials=4597 calls/trial=  10.9 deadline_hit=44/45 sec/region=1515
R_M_hphi trials= 177 calls/trial=1280.3 deadline_hit= 0/45 sec/region=  95

The call budget bound the tilt arm (one guided trial costs ~1280 calls, so it
stopped after ~4 trials per region in 95s) while the wall-clock valve bound the
base arm (44/45 regions, using only ~1100 of its 4000 calls). The tilt got 4.5x
the executor calls, 1/16 the wall-clock, and 1/26 the restarts. Not a matched
comparison in either direction. Do not cite these numbers.

Two separate defects, both to fix before any rerun:

1. MATCHING. One binding resource for both arms, not two. Either equal executor
   calls with no wall-clock valve, or equal wall-clock with no call budget --
   never both, or whichever binds first differs per arm.

2. AMORTIZATION. h_model featurizes each successor by APPLYING it, so the
   committor still pays one executor call per admissible action per step. The
   network eval is cheap; the featurization is not. As integrated it is not
   cheaper than the tree lookahead it was meant to replace.

POOL. Independent of both: 1/45 regions here are reachable, so a correctly
matched run on this pool still yields 0 vs 0. The pool has to change before the
end-to-end question is answerable.
