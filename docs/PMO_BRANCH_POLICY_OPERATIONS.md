# Branch-policy deployment correction, 2026-09-10

The first deployment at `82e68cd29b3969df6a389eb63c92ca002630362e`
failed in the frozen model loader before any proposal completed or new oracle
call was made. Its policy fit used only the 192 historical edit observations.
The remote failure receipt reports zero calls and zero of four workers complete.

- Volume: `compose-v4-artifacts`.
- Failed run: `pmo_branch_policy/73b432c41708cf70a4ccda3a90a8ba2d8d528f3752a4202079d6e4f630a13744`.
- Call: `fc-01M26QMM280MQ9EWFMQ5MS2Y02`.
- Failure payload SHA-256: `efe1d3e5de794e9bee318d3acab1b7fa750bd70d95bddb12524b21e64f3e483f`.
- Frozen process: `0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd`.
- Rejected live process: `f2739338c561cdc38d550aba08e5ec756005beb9b6a97aa9b4177bd57b270682`.

Cause: the serialization optimization in `ff31dbd` changed source bytes bound by
the historical process identity. Its bounded local parity experiment was not
qualification for rebinding the model loader. No identity guard is waived.

Correction: create the isolated `pmo-branch-policy-frozen` deployment branch from
the qualified `8ef5eb2` source and transfer the subsequent policy/evidence commits,
excluding `ff31dbd`. The optimized branch remains intact. The live process hash
now equals the frozen hash. The complete-edit policy, prepared inputs, parent
draws, oracle, seeds, recipe and 32-new-call contract remain unchanged. No new
scores existed to affect this operational choice. Use the qualified slower
serialization for this experiment; the earlier speedup is not claimed here.

The launcher now checks the same process binding locally before spawning. A
focused regression checks both acceptance of the frozen identity and rejection
of drift. The failed namespace remains immutable; the clean replacement gets
its own source-bound run identity. Total new-call authorization across the failed
attempt and replacement remains 32, not 64. There are no completed molecular
proposal units to reuse. Repeating the small historical policy fit costs no oracle
calls. This correction does not authorize a larger experiment or reference training.
