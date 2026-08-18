# Amendment — a budget-extended (H40-aware) h_phi

**Amends the horizon diagnostic** (`AMENDMENT_HORIZON_DIAGNOSTIC.md`), which
recorded that a null at H > 24 is CONFOUNDED because `h_phi` has no input for
step 25 and every caller clamps with `min(budget, 24)`. That amendment named
the fix and put it out of scope: "answering the question properly would need a
budget-extended h_phi, which is a retraining decision". This is that decision.

Recorded before the data.

## What is being changed, and what is NOT

Changed: `BUDGET_MAX` becomes a threaded parameter rather than a module
constant, so the budget one-hot can be 41-wide instead of 25-wide, and a head
can be trained on trajectories longer than 24 steps.

NOT changed: the target, the objective, the estimators, the boundary
condition, the region grid, the held-out-by-source split, R_theta. The
quantity is still `h_b(x, z) = P(exists t <= b : X_t in B_z)`, still fitted by
MC future-event cross-entropy plus a stopped-gradient Bellman consistency term,
still with `h = 1` in-region ENFORCED and in-region states excluded from the
target rather than labelled.

## The frozen H24 head must survive untouched

Every edited line is on the path that produced it. Four guards:

1. `scripts/hphi_budget_max_regression.py` replays the assemble inner loop
   through the old path and the new one at `budget_max = 24` and requires
   `Xtr`, `Ytr`, `Btr` and `Mva` to be **bitwise** identical. It passes:
   1,159 x 1055 and 1,032 x 1055, zero differing entries.
2. `train()` takes its head width from the feature matrix it loaded, not from a
   module constant, so a `budget_max` mismatch raises instead of silently
   truncating a one-hot.
3. The entrypoint REFUSES to write a `budget_max != 24` head into `hphi_v2/`,
   which holds the frozen `head.pt` and `norm.json`. The H40 head goes to
   `hphi_v2_h40/`.
4. The hardcoded `647f82..` corpus sha is recorded only when the assembled
   corpus actually is the frozen H24 one; otherwise the file's own hash is
   recorded. A head trained on a different corpus never claims that provenance.

## The corpus, and its honest limitation

`train_0256x02_H40` — 256 sources x 2 trajectories x H40 = 20,480 transitions,
which is **42% of the frozen 49,152-transition budget**, spent as depth rather
than width. Against the H24 corpus's 2,048 trajectories this is 512, so at any
budget b <= 24 the H40 head trains on roughly a quarter as many examples.

That asymmetry sets what the result can mean:

- A CLEAR WIN is actionable, because it is won from a quarter of the data.
- A MODEST OR NULL result is NOT a kill. It is confounded with corpus size,
  and the next step would be the full-size H40 corpus, not abandonment.

## THE CAVEAT THAT MUST BE FIXED BEFORE THE DATA

The preregistered readout is prospective AUC at 4, 8, 12 and 16 steps, against
the banked h_phi collapse of 0.796 -> 0.667 -> 0.556. Every one of those
lookaheads lies INSIDE both heads' training range. In that range the H40 head
has no structural advantage and strictly less data, so the comparison is if
anything stacked against it.

The H40 head's structural advantage is at budgets 25..40, where the H24 head
has no training data at all and its one-hot cannot even represent the budget.
Banked SMC runs stop at H = 24 and therefore cannot probe it. So a second
readout is added — not a redesign of the objective, but a measurement the
preregistered one is structurally unable to make.

## Two readouts

**Primary — banked re-scoring.** `hphi_prospective_two_head_app.py`. Identical
runs, identical positions, identical AUC code as the banked baseline; only the
head changes. Lookaheads 4, 8, 12, 16 on marginal+hard runs.

  The parity column is the control. Re-scoring the OLD head must reproduce the
  `h_y_bm1` each run recorded. If it does not, the feature build, the
  normalisation or the budget clamp is wrong, and the new head's column is
  wrong the same invisible way. Reported as worst absolute difference.

**Secondary — unguided long range.** `hphi_longrange_auc_app.py` on the
permanently excluded pilot-64 sources rolled out to H40, over the 20
preregistered regions, at lookaheads 4..40. Above 24 the old head is clamped
and is a HANDICAPPED reference, not a fair competitor; that is the point, since
the handicap is precisely the deficiency being fixed.

Both panels are clean for both heads: `hphi_train_1024`, `hphi_valid_128`,
`dev_panel_qed_64` and `hphi_pilot_64` are pairwise disjoint (verified, zero
overlap), and the H40 head trains only on the first 256 of the train list.

## The baseline's own error bar

`PROSPECTIVE_SIGNAL.json` rests on **8 hit runs against 120 miss runs**. Eight
positives is a very thin basis for an AUC, so 0.556 should be read as "no
demonstrated long-range signal", not as a precisely located value. A comparison
against it inherits that width, and no small movement should be called a
result.

## Decision rule, fixed now

- **Clear long-range gain** (primary AUC at 12 and 16 materially above the
  banked baseline, and the secondary showing the new head holding up past 24)
  -> generate the FULL 49,152-transition H40 corpus, retrain, and only then run
  an H40 SMC arm on the 12-source marginal/hard panel.
- **Modest or null** -> NOT a kill, per the corpus-size confound above. Record
  it and stop before spending SMC compute; the open question stays "does more
  corpus fix it", not "is the horizon the lever".
- **Parity check fails** -> no AUC is reported at all, from either head.

## What this does not license

Using the H40 head in any reported controller, on the validation panel, or on
the official 800; or treating any number here as a coverage claim. `hphi_v2/`
remains the frozen head until a full-size corpus and an SMC result say
otherwise.
