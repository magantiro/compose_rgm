# Persistent two-option planning: no champion improvement

The complementary donor-channel test retained actual witnessed continuations,
including lower-scoring first moves. Both immediate and lookahead arms remain
at **0.6747477698**, with zero temporary-loss recoveries above the original root.
Stop this unchanged two-option recipe. This does not reject all future-aware
control; it is a negative result for this fixed proposal/scheduling combination.

Two exact starts, 0.6747477698 and 0.6720215050, shared four first-option slots
per root. Each arm then had eight continuation slots per root: immediate from
its best first state (or incumbent), lookahead two from each first slot without
score pruning. Failed first slots remain failures. Donors were the unchanged
116-row bank, not public winners. This is a channel ablation, not narrowing
the full controller to donor-only or ring-only chemistry.

Thirty-two unique tasks: 17 compiled, 12 unsupported size, two search-unresolved,
one self-proposal. Immediate completed 5/16 continuations, lookahead 8/16.
Their continuation means were 0.117580 and 0.260952. These secondary differences
do not meet the champion-improvement criterion. Compiled programs span 4-64
primitives and intended release 0.125-0.974359. All 574 primitive transitions
replayed exactly. Full structures, ancestry and realized-change metadata remain
in the raw candidate records; no endpoint teleportation was introduced.

Fourteen new calls, 15.501 seconds wall, 11.903 proposal seconds. Development
accounting became **1992** calls plus the separately recorded 249455 prescreen
evaluations. No new neural-law calls, GPU, Modal job, or docking.

`report.json` verifies source/input identities, locks, paid receipts, immediate
and lookahead parent allocation, all compiled paths and retained winners. Five
focused scheduling/compiler tests passed in 2.93 seconds; touched-code lint
passed. No full-suite or milestone claim. Raw files:
`/private/tmp/compose-pmo-persistent-lookahead-20260911a`.
