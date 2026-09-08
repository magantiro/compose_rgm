# Frozen-model fused construction: one completed reference path

The repaired audit completed on PARP1 seed0, with no docking or learned
lookahead. Four carbon insertions and one cycle closure added a fused aromatic
six-membered cycle while retaining the original atoms and induced connectivity.
This is one verified constructive path, not a completion-rate estimate or an
objective-driven controller comparison.

Run revision: `0fec8ad05ab6589043dbeb721a62c1217d7a8105`.
Call: `fc-01M1ZHKME3NE1XAJDJGSEA7EQ5`, completed.
Contract semantic SHA-256:
`fefbcbdcbb1bbded2a44506c14aab57244ab0b7b5c12b3e21da48e6273c221f9`.
The exact source, checkpoint, and run-path identities remain those of attempt 1.
Its failed evidence is unchanged. No further scientific job was launched.

## Measured result

- Five sampled edits; four new heavy atoms; graph cycle-rank gain one;
  aromatic-ring gain one; ring-system-count change zero, as expected for fusion
  into an existing system.
- Fixed region: atoms 14 and 15, intended release 2/19 = 0.105263. Selected
  orientation: 15 to 14. Realized coherent component: six atoms (two originals
  plus four births), or 6/19 = 0.315789 relative to original size. This is not
  a claim that 31.6% of the original scaffold was replaced.
- Five completed law/option rows, with support sizes 2, 1, 1, 1, 1. All required
  descriptor matches survived the inherited proposal cap and product checks.
- Six canonically distinct counterfactual intermediate products, but only one
  completed sampled endpoint. They are not six independent proposals or extra
  draws from the outer region controller.
- 246 total public executor calls: 240 within raw law enumeration and six option
  product checks. Across all calls, 186 execution returns and 60 invalid-rewrite
  outcomes were recorded. None of the sampled edits or option products failed.

Endpoint:

```text
CN(C)Cc1ccc2c(c1)CNC(=O)c1c3ccccc3cn1-2
```

Proposal wall time was 82.970 seconds: 19.702 seconds of law enumeration and
63.094 seconds of audit persistence. Initialization took 131.748 seconds;
elapsed time before result publication was 220.829 seconds. The run used one
CPU/thread and a 6 GiB allocation, with no GPU or docking. The pinned runtime
reported Python 3.11.12, NumPy 1.26.4, PyTorch 2.4.0+cu121, and RDKit 2024.03.5.
Actual model parameter dtype was float32. GPU-enabled package naming does not
mean GPU computation was used.

The measured raw-law normalization errors ranged from about 3.01e-8 to
1.33e-7, within the unchanged production tolerance of 2e-5. The audit saved the
unmodified raw weights. This directly demonstrates that ordinary numerical
roundoff can exceed the earlier audit-only tolerance; the exact rejected values
from attempt 1 remain unknown.

## Verification and limits

`review.json` records all remote/local byte comparisons, content-address and
hash checks, RNG/path agreement, complete executor-call census, canonical
diversity checks, and exact replay of the five selected primitives. The
independent fused-cycle witness passed again locally, using exact slot states,
not SMILES reconstruction. The verification command is included in that file.
Local verification made ten executor calls over two five-step replays, with
zero new model or oracle calls; verifier serialization corrections are recorded.
The repair's 60 focused tests, lint, and formatting passed. The historical full
suite remains non-green; it was not rerun for this isolated interface repair.

This is a fixed-option, applicability-conditioned reference trajectory. It
does not sample Q(M) or Q(o), test a difficult multi-branch continuation problem,
show guidance superiority, or establish docking/IVG competitiveness. In
particular, deterministic later phases leave no meaningful action choice for
guidance to improve on this particular path.

## Next scientific question

Move to a small matched-budget objective-driven end-to-end comparison, with
both ring and non-ring options active and generic retained. Hold the generator,
region policy, applicable options, constraints, and budgets fixed when disabling
guidance for its causal baseline. Evaluate feasible objective versus oracle
calls and wall time, plus diversity and intended versus realized structural
change. Ring growth, modification, opening, shrinking, and non-ring changes
are available capabilities, not mandatory objectives on every task.

The observed persistence cost is an operational optimization target: batch audit
publication at a documented checkpoint unit before a throughput comparison,
without changing probability laws or losing required restart evidence. Do not
spend further runs polishing this one deterministic ring demonstration. No
winner molecules enter templates, rewards, or selection, and no full T4 campaign
is authorized by this result alone.
