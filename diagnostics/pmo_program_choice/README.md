# Fresh program choice: damage avoidance, no optimization improvement

Neither arm improved the PMO best **0.6030226892**. Learned choice raised mean
selected score from **0.2818898006** to **0.4000285361**, but produced **0/16**
parent improvements versus **1/16** for random choice. Mean parent-to-child
changes remained negative, -0.170697 and -0.288836. Stop this unchanged model.

Both arms used identical completed donor pools from the same 16 exact parents.
Eight attempts per parent produced 90 completed programs from 128 attempts.
Failures: 36 unsupported sizes, one charged-center mapping violation, one
unresolved search. The novel pools held 81 distinct molecules. Predictions,
canonical probabilities and coupled selections were locked before new scores.
Both arms requested 16 endpoints, with two shared choices: **30 physical queries**.
There was no positive-gain hard filter or within-batch learning.

Execution took **46.672 seconds**, including 38.085 summed proposal seconds and
6.687 model seconds on local CPU. No R_theta enumeration, GPU or Modal compute.
This is prescreened component development, not a whole-controller benchmark or
PMO AUC. The 249455 reported prescreen calls and 588 historical development calls
remain accounted for.

## Separately charged full-pool diagnostic

After the comparison, a separate locked diagnostic queried the **51 unselected
candidates** in 2.124 seconds. These labels were unavailable to both original
selectors and do not become part of their 30-query comparison.

The full pool's best was **0.5932958790**, below the incumbent. Only two of 16
parents had an improving candidate, one each. Learned choice reduced the mass
on both improvements: 0.0946 versus 0.1667, and 0.0515 versus 0.3333.
Thus this pool limits even an oracle-perfect selector, and the ranker also
underweights its few constructive edits. This is not a ceiling for the executor,
all donor combinations, or multi-step search.

Decision: change how useful programs are generated before another ranking refit
or longer unchanged SMC run. Preserve the earlier replicated donor gains, but
neither uniform donor exchange nor this damage-avoiding ranker is sufficient.
Broad reference chemistry remains available; this isolated experiment did not
replace the controller with a donor-only generator.

## Artifacts and checks

`report.json` binds configurations, source/input hashes, selected chemistry,
per-parent pool ceilings, costs and query accounting. Raw locks, oracle receipts,
exact primitive witnesses and source/model snapshot were uploaded to
`compose-v4-artifacts` under:

`pmo_program_choice/28b2a2742c024dd053d5a54730776fe4743cfaf3157913ae80521375876775de/`

- `receipts.tgz`: SHA-256 `28b2a2742c024dd053d5a54730776fe4743cfaf3157913ae80521375876775de`.
- `source.tgz`: SHA-256 `6bf34dcf3235286cec764f3ab30f9996afef4a4c14590a967a6506816813914a`.

Four focused split/ranking, probability-parity and canonical-pool tests passed
in 2.79 seconds. Ruff check/format and diff whitespace checks passed. Preflight
reported the uncommitted selector explicitly; this was a local hash-bound run,
not a dirty-code Modal launch. No full suite, milestone completion, commit or
push is claimed.
