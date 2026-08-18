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

    candidates      1    2    3    4    5    6    7    8   10   20
    ALL /64        28   32   35   36   37   40   40   41   43   47
    reliable /19   18   18   18   18   18   18   18   18   18   19
    marginal /8     2    3    5    5    6    6    6    6    6    7
    hard /37        8   11   12   13   13   16   16   17   19   21

Full development curve @1..@20 (ALL): 28 32 35 36 37 40 40 41 42 43 44 44 44
44 44 44 44 46 46 47. Note candidates 12-17 added exactly zero and 18 added
two, so the tail is lumpy rather than smoothly decaying: a "stop after two
quiet candidates" rule would have stopped at 12 and lost 3 sources.

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
- **k = 8 is the PREREGISTERED point.** Extending the panel to k = 20 later is
  permitted -- candidates are independent for the `restart` arm and seeds key on
  (arm, source, candidate), so a later slice is exactly what a single longer run
  would have produced (slice parity: 64/64 on the development panel). But it is
  permitted ONLY under this rule:

      the full candidate-efficiency curve is reported for whatever k is run,
      and k = 8 remains the preregistered operating point.

  The danger of extending is not the extra compute, it is choosing the headline
  k after seeing which k reads better. Publishing every k removes that freedom:
  there is nothing to select. A later extension therefore adds information
  without moving the goalpost, and the development curve is reported the same
  way (@1..@20) rather than at a flattering point.
- **EXTENSION TO k = 20 IS INTENDED, DECIDED NOW.** It is not contingent on the
  k = 8 result. Recording the intent before the data is what makes it
  preregistration rather than a forking path: had it been left conditional, a
  disappointing @8 followed by an extension would be indistinguishable from
  shopping for a better k. k = 8 is run first only because of the budget
  ceiling, not because it is a decision point.
- **Reported by stratum is NOT possible here** -- the 128 have no banked
  solvability strata, since nothing has ever been run on them. That is the
  point of a prospective panel, and it means the headline is a single coverage
  number plus per-source detail.

## THE COMPARATOR, AND THE RULE ALREADY ON THE BOOKS

GrIDDD reports **45.1%** at QED >= 0.90 / Tanimoto >= 0.40 -- our exact success
event. `AUDIT_EXISTING_CONTROLLER_LANE.md` records a standing ruling about it:
an earlier 0.5000 obtained at `budget_per_lead = 1000` against the protocol's 20
candidates was a 50x budget advantage and "must never be placed beside 45.1%".

This run does not repeat that error: it returns exactly k candidates per source,
which is the protocol's interface. Two mismatches remain and must travel with
any number quoted from it:

1. **Different panel.** 128 held-out validation sources, not the official 800.
2. **Different per-candidate compute.** One returned candidate here is a full
   SMC run -- 32 particles x 40 steps = 1,280 internal molecule evaluations --
   so the interface matches while the compute behind each returned molecule does
   not. Disclose it; do not let "same number of candidates" imply same cost.

One mismatch runs in the CONSERVATIVE direction and should be stated as such:
k = 8 is fewer returned candidates than GrIDDD's 20, so a favourable number at
k = 8 is achieved with a smaller candidate budget than the comparator.

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
