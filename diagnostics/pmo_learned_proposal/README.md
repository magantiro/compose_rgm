# Learned proposal comparison: best-score null

Both arms retained their starting best, 0.5222329678670935. Baseline top-ten mean
was 0.5031932691 versus 0.5073800970 for the learned proposal. New-child best was
0.5137011669 versus 0.5173061316; 7/36 versus 9/36 options improved their parents.
These are limited local-refinement differences in one exposed warm comparison,
not evidence of better long-range search or statistical superiority.

67 new physical PMO calls, 801 historical unique labels available, 72 workers,
212.817 seconds driver wall time, 167.626 seconds proposal wall time. Oracle
execution totaled 0.0834 seconds. Learned rejection sampling took 4.206 attempts
per primitive, versus one in the reference arm. No failures or best improvement.
Do not extend this recipe unchanged. It is not a learned future-value comparison.

`report.json` verifies source archives, score accounting and particle decisions.
Run `1d4ae62017d41ab1f0736a74dd49a984e92b1c332daa7df2e2472dd2d1a187cf`,
source `9c5964b6678f`, on `compose-v4-artifacts/pmo_learned_proposal/<run_id>`.
Reported wall times are measured; actual billed cost was not collected.
