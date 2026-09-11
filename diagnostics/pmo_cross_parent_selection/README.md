# Frozen local selector transfers to two neighborhoods, no new champion

The unchanged 1044-label endpoint model found six improving candidates among
32 choices across two new parent neighborhoods, versus zero among 32 uniform
choices. Mean scores were 0.667499 versus 0.611814 on the 0.669328 parent, and
0.659941 versus 0.596829 on the 0.660687 parent. Precision was 4/16 and 2/16
versus 0/16 in both controls. Best remains **0.6747477697966318**. This component
pass is not a full-controller improvement or matched PMO AUC result.

Neither parent was a training endpoint, but both are related exposed development
chemistry, not source/scaffold-held-out validation. The experiment retained all
1474 and 1721 cached positive-mass canonical successors. Previously paid labels
were excluded from query allocation; unqueried pools were 1469 and 1716.
Recall over unqueried improvements remains unknown.

Sixty-four physical calls, 52.328 seconds preparation and 3.549 seconds exact
replay/scoring. Historical costs are now 249455 prescreen plus **2056 development
calls**. No GPU, new neural scoring or remote scientific job. Peak memory was
391086080 bytes on macOS. `report.json` verifies input/source identities, unchanged
coefficients, zero-error prediction replay, allocation over the full available
pool, locks, physical receipts and every selected primitive endpoint.

Three focused tests passed in 2.52 seconds; touched-code lint passed. No full
suite or milestone completion. Raw output is
`/private/tmp/compose-pmo-cross-parent-selection-20260911a`.
The complete raw bundle is saved on volume `compose-v4-artifacts` at
`controller_local/0c81c535c0b1c78578f88f83e16bb4809feffb08ad62024d4d4c5019489f3449/artifacts.tar.gz`;
the directory name is its SHA-256. Upload completed.

Decision: integrate this fixed selector as an optional channel of the broad
controller, then run the short matched comparison in `docs/PMO_LOCAL_GUIDANCE.md`.
Preserve all broad reference options and donor programs. The endpoint model is
not future value and is not assumed calibrated across all PMO chemical space.
