# T4 structural-subgoal proposal gate, attempt 2

## Outcome

This zero-oracle held-source gate failed. Neither the source-balanced marginal
nor the graph-conditioned policy recovered an exact teacher subgoal, exact
teacher endpoint, or radius-2 transformation-equivalent teacher endpoint at
Recall@8, Recall@32, or Recall@128 on any of the 15 held sources.

The negative result is a proposal-support result, not a docking result. The
locked candidates were not scored by a T4 oracle.

| Fold | Policy | Complete / attempted | Execution precision | Unique endpoint yield | Summed proposal CPU seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | source-balanced marginal | 502 / 640 | 0.784375 | 0.784375 | 5,219.139 |
| 1 | source-balanced marginal | 584 / 640 | 0.912500 | 0.912500 | 2,453.658 |
| 2 | source-balanced marginal | 614 / 640 | 0.959375 | 0.959375 | 229.035 |
| **All** | **source-balanced marginal** | **1,700 / 1,920** | **0.885417** | **0.885417** | **7,901.832** |
| 0 | graph-conditioned | 402 / 640 | 0.628125 | 0.628125 | 18,991.054 |
| 1 | graph-conditioned | 512 / 640 | 0.800000 | 0.800000 | 22,966.578 |
| 2 | graph-conditioned | 527 / 640 | 0.823438 | 0.823438 | 8,547.182 |
| **All** | **graph-conditioned** | **1,441 / 1,920** | **0.750521** | **0.750521** | **50,504.814** |

Every exact-subgoal, exact-endpoint and radius-2 transformation recovery and
precision entry is zero at all three cutoffs for both policies. The
graph-conditioned policy also reduced execution precision by 0.134896 and used
6.39 times as much summed proposal CPU time as the marginal policy.

## Post-gate support diagnosis

The runtime policies can only select and rebind complete structural-delta
templates extracted from their training sources. Across the three folds, the
held sets contain 147 subgoal instances (137 unique serialized templates), and
zero held template identifiers occur in the corresponding training-fold
vocabulary:

| Fold | Training templates | Held instances | Held unique templates | Exact identifier overlap |
| --- | ---: | ---: | ---: | ---: |
| 0 | 86 | 52 | 51 | 0 |
| 1 | 99 | 43 | 38 | 0 |
| 2 | 89 | 52 | 48 | 0 |

This establishes a finite-template support failure for exact subgoal recovery.
The policy still generated many valid, unique transferred endpoints, so the
result does not show that every candidate lacks docking utility. It shows that
ranking and rebinding an enumerated training-template vocabulary is not a
sufficient generative model for held-source structural transformations.

The next controller revision should generate the factorized contents of a
structural delta, including region, output topology, atom and bond attributes,
attachments, dependency structure and stopping, rather than retrieve an entire
training delta as one indivisible class. The exact realizer remains valid and
should be reused unchanged.

## Authoritative artifacts

The fold reports are `fold_0/report.json`, `fold_1/report.json`, and
`fold_2/report.json`. Their payload SHA-256 values are:

- fold 0: `211b12ac3d1a7972e546abb8de5ed0e328b92118a6c078c3391fe3480d66c5e7`
- fold 1: `d1dba8f152fac122e4b0443499b032fab42879abdd6f800c53fbb03d2bcbb843`
- fold 2: `3c036727eb7fa0c54c6a8d3177826435475a8dbf9067d40427619340ad319877`

All three report zero new oracle calls. Candidate locks were generated from
clean revision `6626a5f5f6a8a3059647535e50bf35fd13d7177c`.
