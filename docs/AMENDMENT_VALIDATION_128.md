# Amendment — one prospective validation on the 128-source panel

**Amends `AMENDMENT_HORIZON_DIAGNOSTIC.md`**, which introduced H > 24 as a
diagnostic and stated explicitly that it does NOT license "reporting any H > 24
configuration as the controller, using it on the validation panel or the
official 800". This amendment lifts that restriction for ONE prospective run,
and for nothing else.

Recorded before the data.

## What is selected, and on what evidence

**Controller:** H = 40 receding horizon, `b_eff = min(24, b)`, N = 32 particles,
the FROZEN H24 `h_phi` head (`hphi_v2/head.pt`, sha256 `9ea51ec4...`), arm
`restart`, region (QED >= 0.90, sim >= 0.40).

**Returned candidates:** k = 8 per source.

Selected exclusively on the consumed 64-source development panel:

    candidates      1    2    3    4    5    6    7    8    9   10
    ALL /64        28   32   35   36   37   40   40   41   42   43
    reliable /19   18   18   18   18   18   18   18   18   18   18
    marginal /8     2    3    5    5    6    6    6    6    6    6
    hard /37        8   11   12   13   13   16   16   17   18   19

H = 40 beats H = 24 at k = 4 on the same seeds: 36/64 against 32/64, paired
5 sources won to 1 lost, with contact rising on both decision strata (marginal
6/32 -> 11/32, hard 16/148 -> 22/148). It costs 1.66x the wall clock for 1.58x
the transitions -- per-transition cost is flat, so the horizon is bought at the
expected price and nothing is anomalous about it.

k = 8 rather than 6: the efficiency knee is candidate 6 (2.00 sources per
candidate index over 1-6, falling to 0.75 over 7-10), so 6 is the defensible
minimum and 8 buys exactly one further development source. Eight is chosen
anyway because the marginal cost on the 128 panel is ~\$0.80 and because a
round pre-registered budget is worth more than a number tuned to the knee of
the very curve being used to justify it. This is a judgement call and is
recorded as one.

## Why the retrained twist is NOT in this configuration

A budget-extended (H40-aware) `h_phi` was built and tested this session and is
NOT used. Four measurements agreed it does not help: unguided pooled AUC worse
at every lookahead; run-level prospective AUC with no statistical support
(P(new > old) ~ 0.61, 95% CI [-0.44, +0.80]); paired controller A/B 4/8 -> 3/8
on the decision strata; and monotonicity violations on 28.9% of states against
4.7% for the frozen head. Monotonicity was then localised and found neither
concentrated where the controller fails nor load-bearing on particle ordering
(Spearman 0.996 under isotonic projection).

So the H40 gain comes from giving the FROZEN dynamics more edit opportunities,
not from a better budget-conditioned value model. That is the claim this
validation tests.

## The design, and the thing that would invalidate it

- **Panel:** all 128 sources of `data/jin/hphi_valid_128.txt`, verified
  pairwise disjoint from `hphi_train_1024`, `dev_panel_qed_64` and
  `hphi_pilot_64` (zero overlap).
- **ONE run, all 128 at once.** Running 64, reading the answer, and then
  deciding about the rest would convert the validation panel into a second
  development ladder. The panel is spent the moment it is looked at.
- **No hyperparameter selection afterwards.** Whatever comes back is banked as
  the prospective number, including a disappointing one.
- **Reported by stratum is NOT possible here** -- the 128 have no banked
  solvability strata, since nothing has ever been run on them. That is the
  point of a prospective panel, and it means the headline is a single coverage
  number plus per-source detail.

## What this does NOT license

Calling the result the GrIDDD benchmark comparison. The published contract is
**800 official sources x 20 candidates**, and that panel stays untouched until
the paper-bearing run. Any external claim of the form "COMPOSE outperforms
GrIDDD on the Jin QED benchmark" waits for it.

What this run can support is narrower and still worth having: a controller
configuration chosen entirely on development data reproduced its performance on
a prospectively untouched 128-molecule panel, at a stated candidate budget.

## After this run

QED controller development stops. The next scientific expansion is MOLLEO --
five-objective optimisation under an oracle budget -- which tests a
qualitatively different regime rather than refining one benchmark number.
