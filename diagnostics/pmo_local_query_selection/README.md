# Paid local labels improve query allocation, not the champion

On a locked prospective batch of previously unscored neighbors, 16 endpoint-model
choices averaged **0.6461213347**, versus **0.5527572094** for 16 uniform choices.
Their best scores were 0.6693280212 and 0.6529598679, with two versus one
improvements above the originating 0.649519 parent. Neither beats the existing
0.6747477698 champion. Thirty-two new queries, 7.008 seconds preparation plus
2.402 seconds scoring/replay. Historical development calls became 1978.

The fixed existing kernel-ridge recipe, ridge 1.0, was fitted to local endpoint
scores. It is not a future-value estimator. Pre-fit deterministic canonical split:
817 training and 227 calibration neighbors. Calibration top eight contained
three of six available parent improvers, precision 0.375 and recall 0.5;
mean 0.649731 versus uniform expected 0.572611. MAE was 0.015250.
This is within-parent interpolation on exposed development, not source/scaffold
generalization. No public winner was a feature, label or reward input.

Decision: the component criterion passed; next test the unchanged selector on
new parents before broad-controller integration. Do not claim statistical
superiority from one small batch or matched PMO performance.

The fingerprint kernel previously rebuilt Python sets per pair. Integer bit
operations preserve exact intersections/unions. On the same recorded 256-row
fixture: 3.574646 seconds to 0.116977 seconds, **30.56x**, all 65536 values
identical. Full 1580-molecule state kernel: 5.586 seconds. Peak process memory:
376586240 bytes (macOS). `kernel_profile.json` binds old/new source and input.

`report.json` checks the source snapshot, split, fitted normal equations,
prediction replay (zero error), locked allocations, 32 physical receipts and
every scored exact primitive endpoint. Nine focused selector/kernel/chronology
tests passed in 3.05 seconds. Touched-code lint passed. The unused-import-only
cleanup after execution does not alter the archived executed source. No full
suite or milestone completion. Raw files:
`/private/tmp/compose-pmo-local-query-selection-20260911a`.
