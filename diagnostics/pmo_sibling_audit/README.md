# Same-parent edit-choice result

## Outcome

The saved predictions contain some candidate-ranking signal, but do not establish
reliable edit selection. The edit-aware model is correct on 9/16 balanced and
18/27 adaptive matched-parent pairs. On the stricter subset whose parents were
already observed at the fitting snapshot, this becomes 6/13 and 17/26. The
previous `no_predictor_qualified` decision is unchanged. No model was refitted,
no guidance was deployed, and no oracle or optimization run was launched.

The diagnostic used 0.088 seconds of analysis time (0.661 seconds for the CLI
process), **zero new oracle calls**, and **zero new fits**. These are logged,
retrospective comparisons, not prospective policy returns or official PMO scores.

## Coverage

Both inputs contain 80 previously saved chronological predictions.

| Coverage unit | Balanced | Adaptive |
| --- | ---: | ---: |
| Distinct exact archive parents across all predictions | 43 | 42 |
| Parents with multiple children, ignoring snapshot boundaries | 17 | 15 |
| Exact-parent/snapshot groups with at least two children | 11 | 15 |
| Distinct parents represented by those groups | 9 | 11 |
| Candidates represented | 25/80 | 36/80 |
| Candidates excluded as singleton parent/snapshot groups | 55 | 44 |
| Unequal-score pairs | 16 | 27 |
| Equal-score groups retained for selection accounting | 0 | 1 |
| Groups mixing cycle-increasing and other candidates | 4 | 6 |

Groups and pairs share parents and are not independent replicate experiments.
Exact parent variants are not merged by SMILES. Candidates from different model
snapshots are not pooled to inflate ranking coverage. Every excluded prediction
is recorded with its reason.

## All frozen methods

Concordance counts prediction ties as one half. Gain is the mean selected true
score minus uniform choice among that same group's logged candidates, weighting
groups equally. Prediction ties are averaged without using outcomes to break
them. These gains are descriptive menu comparisons, not optimizer improvement.

| Method | Balanced concordance | Adaptive concordance | Balanced gain | Adaptive gain |
| --- | ---: | ---: | ---: | ---: |
| Copy parent score | 0.5000 | 0.5000 | 0.00000 | 0.00000 |
| Parent + option mean change | 0.4063 | 0.5741 | -0.01251 | +0.00656 |
| Parent/option/scale context | 0.4063 | 0.6481 | -0.03628 | -0.01138 |
| Completed-endpoint score | 0.6875 | 0.5926 | +0.04053 | +0.02079 |
| Parent/endpoint/edit change | 0.5625 | 0.6667 | +0.03650 | +0.00791 |

Parent-copy cannot distinguish siblings: its decisive prediction coverage is
zero. Endpoint and edit-aware models distinguish all eligible pairs. Option-only
coverage is 11/16 and 22/27; context-model coverage is 15/16 and 26/27. Conditional
accuracy, equal-group concordance, best-choice probability, and regret are all
retained in the machine-readable artifact.

Selection and accuracy need not agree. For example, the adaptive context model
ranks many pairs correctly but its chosen candidates score worse on average
than uniform selection, indicating costly mistakes in this logged sample.
The endpoint model's positive selection result does not reverse its failed
overall error criterion in the preceding check.

## Snapshot and topology limits

For parents observed by the model fitting cutoff, there are only 8 balanced
groups (19 candidates, 13 pairs) and 13 adaptive groups (32 candidates, 26 pairs).
Edit-aware concordance is 0.4615 and 0.6538, with selection gain +0.01258 and
+0.00819. Thus balanced-arm accuracy is not consistently above the tie baseline.
The remaining 3 and 2 groups use parents whose scores became available after
the fitting snapshot but before each candidate query, as permitted in the
original prediction contract; their results are separately retained.

Mixed cycle-increase groups contain only 6 balanced and 10 adaptive pairs,
including comparisons between two non-increasing siblings within a mixed group.
Edit-aware concordance is 0.6667 and 0.5000, and selection gain +0.05110 and
-0.00194. This does not show reliable ring-versus-other selection. Cycle rank is
not SSSR ring count, ring-system count, or a medicinal-chemistry assessment.

## Concrete recorded misranking

Adaptive parent `attempts/0019`, snapshot cutoff 30, scores 0.43793. Three
completed alternatives from that same parent were later queried:

| Candidate | Option | Measured score | Context prediction | Edit-aware prediction |
| --- | --- | ---: | ---: | ---: |
| `attempts/0030` | open | 0.16567 | 0.43833 | 0.43789 |
| `attempts/0033` | add_carbonyl | 0.43350 | 0.43056 | 0.42893 |
| `attempts/0034` | annulate | 0.44721 | 0.42959 | 0.43195 |

Both predictors put the damaging opening ahead of the improving annulation.
The beneficial option was executable and generated. This is evidence of a
prediction error, not missing chemistry support in this particular comparison.
The completed-endpoint model instead favors carbonyl addition, avoiding the
large loss but still missing the best sibling. All parent/product SMILES are
stored in the result for graph inspection; string distance was not used as a
surrogate label. This illustrative group does not estimate a causal effect.

## Reproduction and verification

Authoritative artifact: [result.json](result.json), SHA-256
`f8fba944c60027c55f68b1ccf443c740d24d1bbb4d08fb356a4ec3fca002973d`.
Analysis revision: `34b99d53ac3c8e68c79b54cd6c0c704c8eef3f2e`, clean at execution.
Rules were recorded in `docs/PMO_SIBLING_AUDIT.md` before computing the result.
The report binds the previous prediction report, both original run receipts,
the rule document, and analysis implementation by physical SHA-256.

The original input joins, exact parent IDs, prediction chronology, score ranges,
and parent-label consistency validated. Two focused tests passed in 0.05 seconds,
covering grouping, cross-snapshot exclusion, label chronology, honest tie
accounting, and empty comparison coverage. Ruff formatting/checks and
`git diff --check` passed. No repository-wide suite was run; this is not a
completed controller or deployment milestone.

```sh
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 \
/Users/rmaganti/compose_rgm_git/.venv/bin/python tools/pmo_sibling_audit.py \
  --output /private/tmp/new_pmo_sibling_result.json
```

The command refuses to overwrite an existing result. It consumes saved labels
and predictions only; no molecular enumeration, fitting, or scoring is repeated.

## Recommended next experiment, not launched

Do not spend another long optimization run testing an unqualified guide. A
separate small **matched-branch diagnostic** should directly measure edit choice:
for example, eight archived parents, four complete alternatives per parent,
at most 32 new PMO calls, all charged. Choose parents and applicability-balanced
alternatives without looking at their new scores; retain generic, local chemical
edits, and constructive topology options under the existing region controller.
Lock candidates and predictions before scoring any branch. Score every branch,
not only the model's favorite, to expose both missed gains and harmful choices.

This is new data collection, not deployment of the failed predictor. It requires
separate authorization and a frozen task-specific recipe. The old failed rule
must remain intact. Sequentially logged alternatives cannot substitute for that
matched design, and eventual performance claims still need an untouched,
same-generator prospective optimizer comparison at the declared oracle budget.
