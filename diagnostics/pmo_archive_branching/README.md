# Archive parent allocation: primary null

Both arms finished at their initial best **0.6030226892**. Archive allocation
improved the top-ten mean to **0.5772988909**, versus **0.5703159040** for forward
SMC, but did not satisfy the preregistered new-best criterion. Do not extend this
unchanged recipe. This result supports only improved refinement in this one
warm, prescreened development comparison, not a generalization or SOTA claim.

The run used the same top-16 exact starting states, donor/reference proposal law,
four option boundaries, and 16 slots per arm. Archive allocation reserved one
incumbent slot and drew the other 15 from its own canonical scored archive.
SMC used the previous forward-particle rule. Candidates and score batches were
locked before evaluation; cross-arm results entered an arm only if requested.

There were 89 new physical oracle queries in 349.619 wall seconds (289.514
proposal seconds), plus 80.114 deployment seconds. The 108 unique workers used
1101.707 summed worker-seconds; oracle execution used 0.098 seconds. Shared
workers and labels are counted once physically. Historical prescreen and previous
development queries remain charged in the configuration, not erased by reuse.

SMC produced 50 complete options from 60 attempts, archive 55 from 64. Their
complete offspring improved over their own parents in 7/50 and 8/55 cases,
respectively. Thus better parent retention did not by itself make most edits
productive. This does not rule out useful lower-scoring intermediates.

`report.json` binds the raw result, launch receipt, complete executed source
snapshot, prepared history, and analyzer. Both SMC and archive allocation replay
exactly; maximum numeric replay error was zero. Two focused analyzer tests passed
in 3.39 seconds, and the modified analyzer also reproduced the preceding donor
replication. No molecular enumeration or new oracle calls were needed to audit.
No full repository suite was run; the controller milestone remains incomplete.

Run: `4e96e69b5d28fa371f02ea0810fc52055678848db2aa67ebf1d6d32d60b47b8d`.
Source: `96b0a316b9cf`. Modal volume: `compose-v4-artifacts`, prefix
`pmo_archive_branching/<run>/`. Raw result SHA-256:
`46cec002fbd450f1a1fb65f9bf1a0b2278b95b613398c8345e78d185d2b3bc09`.
