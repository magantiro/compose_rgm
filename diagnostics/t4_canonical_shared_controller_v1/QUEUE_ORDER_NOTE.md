# Canonical T4 run: queue ordering, and one deliberate deferral

Run id `3960cba7d11e2d9dddcbe0a56d8fd886373d59a6a654a48be18ecf4791db3721`.

All 60 cells were spawned at once as independent calls, in the order C, B, A. The
deployed `run_cell` admits 20 concurrent containers (the three `max_containers`
limits sum to the measured ~100-CPU workspace ceiling), so the first 20 spawned
cells started and the remaining 40 queued behind them.

That ordering put all 40 queued cells — every cell of arms A and B — behind the ten
remaining arm-C delta=0.4 cells. Since the total work is fixed, ordering does not
change when the run finishes; it changes only WHICH results exist first. Complete
arm C with empty A and B yields the canonical table and NO ablation, whereas partial
progress on all three arms yields both, because arms A, B and C share the same
delta=0.6 cells with the same seeds and can be compared at any matched charged-call
checkpoint.

So the ten arm-C delta=0.4 cells that had NOT yet started were cancelled and deferred
to the end of the run:

    5ht1b_0_d04  5ht1b_1_d04  5ht1b_2_d04
    braf_0_d04   braf_1_d04   braf_2_d04
    fa7_2_d04
    jak2_0_d04   jak2_1_d04   jak2_2_d04

They had produced nothing — no directory on the volume, no round lock, no charged
call — so the cancellation forfeits no budget and voids no measurement. They are
re-spawned by `tools/launch_t4_canonical.py --mode resume --run-id <id>`, which
re-spawns exactly the cells with no `result.json`.

The delta=0.4 half is also the EASY half: the preflight measured a median of 312
eligible endpoints across lanes at delta=0.4 against 47 at delta=0.6, with zero
empty cells. Deferring it costs the least information per hour deferred.

Nothing about the controller, the contracts or the per-cell budgets changed. This is
a scheduling decision, recorded here rather than left implicit in a queue.
