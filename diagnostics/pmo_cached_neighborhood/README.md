# Cached primitive neighborhood: better endpoint, poor query efficiency

Best endpoint improved from 0.6720215050 to **0.6747477698**, using a single
atom deletion from the earlier 0.649519 parent. This is an exposed development
coverage diagnostic, not a full-controller or PMO AUC result.

Reused positive-mass frozen-law marks: 1591 marks, 1580 canonical successors.
Locked uniform selection scored 1024 novel neighbors; 20 others were already
known. Of 1044 scored, 30 improve the parent and one beats the prior champion.
The remaining 536 were unscored; no negative claim extends to them.

Generation census: 16.936 seconds, no fresh neural calls. Cached scoring pass:
5.273 seconds total, 0.456 oracle seconds. Historical development calls increased
from 922 to 1946; 249455 prescreen labels remain separately accounted.
Do not promote exhaustive neighborhood scoring. Use the paid labels to test
query selection while preserving broad multi-step search.

`report.json` verifies executed-source snapshot, physical hashes, fixed-seed
selection, query locks and receipts, and all 1044 scored primitive replays.
Two focused tests passed in 2.26 seconds; the exact-support fixture passed in
2.12 seconds. No full-suite or milestone-completion claim. Raw files:
`/private/tmp/compose-pmo-cached-neighborhood-20260911b`.
