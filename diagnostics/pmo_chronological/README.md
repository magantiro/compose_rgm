# Chronological PMO prediction check

## Finding and decision

**No predictor qualified under the frozen admission rule.** On two existing
100-query perindopril development streams, the edit-aware model improved ranking
slightly, but every learned model had worse overall mean absolute error (MAE)
than copying the observed parent score. No guidance was deployed and no new
optimization or oracle evaluation was launched.

The fixed analysis completed in 2.432 seconds with **zero new oracle calls**.
This is retrospective predictive evidence on inspected development trajectories,
not a prospective optimization result or an InVirtuoGen comparison.

## Overall results

Each arm contributes 80 predictions: fit on the first 20 charged unique queries,
predict the next ten, then refit every ten queries using earlier labels only.
The four original root queries are included in the warmup. Parent labels known
before the candidate query are available equally to every method. There is no
cross-arm fitting, hyperparameter selection, or winner input.

| Fixed method | Balanced MAE | Balanced concordance | Adaptive MAE | Adaptive concordance |
| --- | ---: | ---: | ---: | ---: |
| Copy parent score | 0.04386 | 0.7598 | 0.03827 | 0.7683 |
| Parent + pooled option change | 0.04748 | 0.7682 | 0.04009 | 0.7837 |
| Parent/option/scale context model | 0.05667 | 0.7249 | 0.04046 | 0.7725 |
| Completed-endpoint score model | 0.07273 | 0.7654 | 0.08276 | 0.7697 |
| Parent/endpoint/edit-change model | 0.05051 | 0.7807 | 0.04024 | 0.7992 |

Lower MAE and higher concordance are preferred. Concordance compares pairs within
the same ten-query prediction window, with prediction ties worth one half and
true-score ties omitted. There are 358 and 356 eligible pairs respectively;
these are dependent pairs, not independent observations. Whole-window ranking
is not a same-parent or counterfactual action-choice assessment.

The frozen rule requires beating **both** baselines in MAE and concordance in
**each** arm, with concordance above 0.5. The edit-aware model's ranking result
does not waive its failed MAE criterion. The endpoint model's relatively large
error is a negative result, not a reason to choose a different recipe after
seeing these outcomes.

## Predeclared topology subgroup

Here topology change means graph cycle rank increased from the actual parent,
not a particular SSSR ring count, ring family, or medicinal-chemistry judgment.

| Method | Balanced cycle-increase MAE (n=13) | Adaptive cycle-increase MAE (n=19) |
| --- | ---: | ---: |
| Copy parent score | 0.13714 | 0.05181 |
| Parent + pooled option change | 0.10646 | 0.04374 |
| Context model | 0.11115 | 0.04897 |
| Endpoint model | 0.09659 | 0.11264 |
| Edit-aware model | 0.09571 | 0.04535 |

The edit-aware model improves over parent-copy MAE for cycle-increasing edits in
both arms, but not over the option baseline in the adaptive arm. Within this
subgroup only seven balanced and seventeen adaptive within-window pairs are
rankable. Edit-aware concordance is 0.7143 and 0.9118, versus parent-copy 0.7143
and 0.8824. These small descriptive subsets do not support a new admission rule.

For the other 67 balanced and 61 adaptive candidates, edit-aware MAE is 0.04175
and 0.03865, versus parent-copy 0.02576 and 0.03405. Predicting a nonzero change
therefore also introduces error where persistence is a strong baseline. The
full artifact retains all subgroup methods and sign-of-change metrics.

## Provenance and verification

- Authoritative artifact: [result.json](result.json).
- Artifact SHA-256:
  `6e164790e344eaeb41dff735fcb14ef17f1928211bf5a8c07f2b7118dc665873`.
- Clean execution revision: `b21e4c53bb195c93abbee72495515544e1e20e52`.
  The recipe, implementation, focused tests, and original root receipts were
  committed before fitting. The result reports `source_dirty: false`.
- Recipe SHA-256:
  `5966010daa2a828650699524368998130c1c8b5964870c040038b2d91fd92a16`.
- Inputs: the two committed `diagnostics/pmo_archive_pilot_100/*.json` results
  and eight original `roots/{balanced,adaptive}/0000..0003.json` receipts.
  Every physical input hash is recorded. Root hashes match the original oracle
  ledgers; reconstructed query labels reproduce the full locked best/top-ten
  curves. Exact archive graphs agree with canonical molecular identities.
- Exclusions: balanced has one already-charged canonical outcome and one support
  dead end; adaptive has six already-charged canonical outcomes. Each arm also
  excludes its sixteen warmup offspring from prediction metrics. No charged
  query was silently dropped, and no missing root label was synthesized.
- Environment: CPU arm64, float64, Python 3.12.9, NumPy 1.26.4, RDKit 2024.03.5,
  Torch 2.14.0. Four numeric thread limits were set to one. Recorded peak RSS is
  257,736,704 bytes on macOS. The recipe uses no random sampling.
- Focused verification: `tests/test_pmo_chronological.py`, **2 passed in 2.64s**,
  covering future-label invariance, disjoint fitting, and kernel correctness.
  Ruff format/check on the three added Python files and `git diff --check`
  passed. The repository-wide suite was not run; this is a bounded offline
  diagnostic, not a deployment qualification or completed controller milestone.

Execution command, from the isolated worktree:

```sh
PATH=/Users/rmaganti/compose_rgm_git/.venv/bin:$PATH \
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
/Users/rmaganti/compose_rgm_git/.venv/bin/python tools/pmo_chronological_check.py \
  --output diagnostics/pmo_chronological/result.json
```

## Interpretation and next safe action

The tested simple predictors do not establish effective guidance. The modest
edit-aware ranking signal supports further diagnosis, not deployment or a claim
that more oracle spending will improve search. This result also does not show
that molecular context is uninformative or that COMPOSE cannot learn good edits.

Before another model or optimization run, inspect whether the existing logs
contain enough chronological, same-parent alternative edits to evaluate the
actual decision: which edit improves this molecule? Report coverage and abstain
if those comparisons are too sparse. This is a proposed separate diagnostic,
not an excuse to replace the frozen failed criterion. Endpoint-aware predictors
also require completed candidates; success there would not by itself establish
primitive-step future-value guidance. Any later prospective experiment needs a
separately declared comparison and authorization.
