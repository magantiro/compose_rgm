# Amendment — matched 2-objective head-to-head vs InversionGNN (JNK3 + GSK3b)

Recorded before any run. Opens the first EXTERNAL multiobjective comparator.
`COMPARATOR_ROLES_CANONICAL.md` already assigns InversionGNN the role of
"global multiobjective competence", so this fills a planned slot rather than
opening a new lane.

## Why this benchmark fits, when MOLLEO did not

MOLLEO Task 3 asks an optimizer to find a remote kinase basin from 120 random
ZINC molecules with **no task-specific training**. Four probes measured why that
failed for us, and the decisive one is that random-ZINC one-step fibers are
genuinely flat: **0 of 24 exactly enumerated fibers contained a JNK3 successor
at 0.3, maximum 0.16 over ~16,000 successors**.

InversionGNN's benchmark differs in exactly the way that matters: **task-specific
training is part of the protocol**. It spends oracle calls labelling ZINC
molecules and trains a property predictor before optimizing. That licenses
COMPOSE to train the multiobjective analogue of the QED `h_phi` on the same
budget, which is the comparison we actually want:

    InversionGNN   10K labels -> F_hat(x)      -> gradient of predicted properties
    COMPOSE        10K labels -> h_phi(x,z,b)  -> future-reachability control over
                                                  the exact legal successor graph

Same information budget. Different inference principle.

## WHAT IS AND IS NOT PINNED IN THE PUBLISHED PROTOCOL

Verified from the arXiv HTML and the released code, both cited so a reader can
check rather than trust:

**Pinned by the paper**
- oracle budgets: **10K train + 5K optimize** (2 objectives); 20K + 5K (4)
- the 5K is "1K per weight vector x 5 weights"
- 5 weight vectors, constructed by the spherical-coordinate algorithm in App. D.3
- **C = 10** molecules kept per generation
- APS = "average score of the top-100 molecules"; novelty; top-K diversity
- vocabulary: 82 substructures appearing >1000x in ZINC-250K
- reported: APS 0.841, novelty 100%, diversity 0.768, HV 0.763 +/- 0.031

**NOT pinned, and this is why we cannot claim an exact head-to-head against
0.841 from their numbers alone**
- the starting molecule(s) for the molecular experiments are never specified.
  Section 5.2 (Q1) says "InversionGNN and I-LS are optimized from random
  initialization for each weight vector", but no bank, distribution or seed is
  given.
- the **hypervolume reference point for the molecular tasks is never stated**;
  only the synthetic task's (1,1). InversionGNN says it follows HN-GFN, which
  uses the origin for normalized higher-is-better objectives -- strongly
  suggestive, not stated.
- the released `denovo.py` is NOT a faithful Table-3 reproduction: it hardcodes
  one starting molecule `C1=CC=CC=C1NC2=NC=CC=N2`, `population_size = 1`, and a
  single preference `[1,3]`, against the paper's C=10 and 5 weight vectors.

That starting molecule is not a neutral choice: scored on our frozen oracle it
sits at **JNK3 = 0.100, the maximum of our entire random-ZINC dev cohort**
(median 0.010), and it is an anilinopyrimidine -- a kinase hinge-binding
chemotype. Starting there is a materially easier problem than random ZINC, so
which initialization is used changes the difficulty of the benchmark itself.

## THE DESIGN: rerun the comparator rather than quote it

Because initialization is unspecified, quoting 0.841 beside a COMPOSE number
would be a comparison of two different experiments. So:

1. **Freeze a common initialization bank ourselves** -- a deterministic,
   objective-blind sample of ZINC-250K molecules, selected by hash and recorded
   with its sha256 before either method runs.
2. **Run BOTH methods** from those same starts, with the same 10K task labels,
   the same 5 preference vectors, and the same 5K optimization budget.
3. Report InversionGNN's published 0.841 as **reported context**, clearly
   labelled as a different (under-specified) initialization -- never as the
   quantity our number is beating.

The engineering tax of rerunning an external method is accepted deliberately
here; the alternative is an unfalsifiable comparison.

## Metrics

- **PRIMARY: APS** (mean of top-100), novelty, top-K diversity. These are fully
  specified in the paper and need no reconstruction.
- **SECONDARY: hypervolume under a PREREGISTERED origin reference point**,
  computed identically for both arms. Reported as our convention, not as
  comparable to their 0.763 -- their reference point is unknown.

## Scope

Two objectives only (JNK3 + GSK3b, both maximised). The 4-objective task waits.
If the multiobjective `h_phi` cannot compete at 10K+5K on the easier task, the
fix belongs there and not behind two more objectives.

Note this is NOT the MOLLEO objective set: there GSK3b was minimised, here both
are maximised. The frozen oracle bundle's `gsk3b` field is stored as `1 - gsk3b`
(a minimisation transform); **it must be inverted for this benchmark**, and that
conversion is a parity check, not an assumption.

## Decision rule

- COMPOSE competitive or better on APS at matched budget -> escalate to 4
  objectives at 20K+5K.
- Clearly worse -> troubleshoot the multiobjective `h_phi` before adding
  objectives. Do not escalate to hide a 2-objective deficit.

---

# ADDENDUM — the preference regions `B_lambda`, fixed before any label is spent

Recorded 2026-08-19, before the 10K corpus exists. This is the piece most
vulnerable to unconscious tuning, so it is written down first and deliberately
made boring.

## The success region

Both objectives are raw higher-is-better here: `F(x) = (J(x), G(x))` with
`J = jnk3` and `G = gsk3b` **after undoing the bundle's `1 - gsk3b` minimisation
transform** (orientation verified: actives 0.880 above inactives).

For each published preference `lambda`, define the **Chebyshev shortfall from
the ideal point**:

    d_lambda(x) = max_i  lambda_i * (1 - F_i(x))            smaller is better

Chebyshev rather than a linear weighted sum, deliberately: a weighted sum
collapses every preference toward one convex compromise, whereas Chebyshev
targeting can represent genuinely different points on the front -- which is the
entire reason the benchmark specifies five preferences.

    B_lambda = { x : d_lambda(x) <= q_lambda }

where `q_lambda` is the **10th percentile of d_lambda over the 10,000 training
labels ONLY**.

In words: for each desired JNK3/GSK3b tradeoff, "success" is reaching the best
10% of the training distribution *in that direction*.

Properties this buys, and why each matters:
- no optimization or test result enters the definition;
- every preference receives the same nominal positive count (~1,000), so no
  preference is starved by construction;
- monotone -- improving either objective can never move a molecule out of
  `B_lambda`;
- **one fixed number, 10%**, chosen now rather than discovered later.

## The training target

For a trajectory `x_0 -> x_1 -> ... -> x_T`, each state, budget and preference:

    Y(x_t, lambda, b) = 1[ exists k <= b : x_{t+k} in B_lambda ]

and `h_phi(x_t, lambda, b) ~ P(Y = 1)`. That is the QED region-`h_phi` logic
generalised to a preference-conditioned multiobjective region: *from this
molecule, with b edits left and this desired tradeoff, what is the probability
of reaching the elite region?*

## THE RULE THAT MAKES THIS PREREGISTRATION RATHER THAN A KNOB

**The 10% threshold does not move.** Not after seeing the diagnostic below, not
after seeing `h_phi`'s validation, not after seeing APS. If the corpus turns out
degenerate, the response is to revisit DATA GENERATION and say so -- never to
slide to 20% because it reads better. A threshold that moves after seeing
outcomes is a fitted hyperparameter wearing a preregistration's clothes.

## Diagnostic to report after `B_lambda` is computed, before training

Purely descriptive, changes nothing:
1. positives per preference (~1,000 by construction -- confirm, do not assume);
2. **number of distinct source trajectories containing at least one positive**;
3. pairwise overlap of the five `B_lambda`.

Item 2 is the one that matters. If 900 of 1,000 positives come from six
trajectories, the value-learning problem is badly undersampled even though the
molecule count looks healthy -- a failure mode invisible in (1) alone. Report
it; do not repair it by moving the threshold.

## Quarantine, restated

`data/diagnostic_kinase_rf/kinase.tsv` is diagnostic-only and MUST NOT enter
this lane as training data, initialization, seed or filter. It was used once
here, legitimately, as an oracle-orientation panel. The task-training regime is
ZINC plus oracle labels on a fixed budget.
